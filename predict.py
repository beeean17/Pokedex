from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required. Install dependencies with: python -m pip install -r requirements.txt") from exc


def load_labels(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def preprocess_image(image: Image.Image, labels: dict[str, Any]) -> np.ndarray:
    return preprocess_canvas(letterbox_image(image, int(labels["inputSize"])), labels)


def preprocess_image_views(image: Image.Image, labels: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    input_size = int(labels["inputSize"])
    canvases = [
        ("full", letterbox_image(image, input_size)),
        ("object", crop_and_resize(image, input_size, scale=0.86, center_x=0.5, center_y=0.5)),
        ("feature", crop_and_resize(image, input_size, scale=0.58, center_x=0.5, center_y=0.42)),
    ]
    tensors = [preprocess_canvas(canvas, labels)[0] for _, canvas in canvases]
    return np.stack(tensors, axis=0).astype(np.float32), [name for name, _ in canvases]


def letterbox_image(image: Image.Image, input_size: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((input_size, input_size))
    canvas = Image.new("RGB", (input_size, input_size), (255, 255, 255))
    left = (input_size - image.width) // 2
    top = (input_size - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def crop_and_resize(
    image: Image.Image,
    input_size: int,
    scale: float,
    center_x: float,
    center_y: float,
) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    side = max(1, round(min(width, height) * scale))
    center_left = round(width * center_x - side / 2)
    center_top = round(height * center_y - side / 2)
    left = min(max(center_left, 0), width - side)
    top = min(max(center_top, 0), height - side)
    crop = image.crop((left, top, left + side, top + side))
    return crop.resize((input_size, input_size), Image.Resampling.BILINEAR)


def preprocess_canvas(image: Image.Image, labels: dict[str, Any]) -> np.ndarray:
    input_size = int(labels["inputSize"])
    mean = np.array(labels["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(labels["std"], dtype=np.float32).reshape(3, 1, 1)

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    normalized = (chw - mean) / std
    return normalized[np.newaxis, ...].astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64)
    values = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(values)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)


def top_k(probabilities: np.ndarray, classes: list[str], k: int) -> list[dict[str, float | int | str]]:
    vector = probabilities.reshape(-1)
    indexes = np.argsort(vector)[::-1][:k]
    return [
        {
            "rank": rank + 1,
            "classIndex": int(index),
            "label": classes[int(index)],
            "confidence": float(vector[int(index)]),
        }
        for rank, index in enumerate(indexes)
    ]


def load_torch_model(checkpoint_path: Path, num_classes: int):
    try:
        import torch
        from torchvision import models
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch/torchvision are required for checkpoint inference.") from exc

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_name = checkpoint.get("modelName", "pokemon-efficientnet-b0")
    if model_name == "pokemon-efficientnet-b0":
        model = models.efficientnet_b0(weights=None)
    elif model_name == "pokemon-mobilenetv3-small":
        model = models.mobilenet_v3_small(weights=None)
    else:
        raise ValueError(f"Unsupported checkpoint modelName: {model_name}")

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(checkpoint["stateDict"])
    model.eval()
    return model


def predict_torch_model(
    model: Any,
    image: Image.Image,
    labels: dict[str, Any],
    k: int = 5,
    multi_crop: bool = True,
) -> dict[str, Any]:
    import torch

    started_at = time.perf_counter()
    if multi_crop:
        inputs, views = preprocess_image_views(image, labels)
    else:
        inputs = preprocess_image(image, labels)
        views = ["full"]
    tensor = torch.from_numpy(inputs)
    with torch.no_grad():
        logits = model(tensor).detach().cpu().numpy()
    logits = logits.mean(axis=0, keepdims=True)
    probabilities = softmax(logits)
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    return {
        "engine": "pytorch",
        "elapsedMs": elapsed_ms,
        "views": views,
        "predictions": top_k(probabilities, list(labels["classes"]), k),
    }


def predict_torch(
    checkpoint_path: Path,
    image: Image.Image,
    labels: dict[str, Any],
    k: int = 5,
    multi_crop: bool = True,
) -> dict[str, Any]:
    model = load_torch_model(checkpoint_path, len(labels["classes"]))
    return predict_torch_model(model, image, labels, k, multi_crop)


def load_onnx_session(model_path: Path):
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("onnxruntime is required for ONNX inference.") from exc

    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def predict_onnx_session(
    session: Any,
    image: Image.Image,
    labels: dict[str, Any],
    k: int = 5,
    multi_crop: bool = True,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    input_name = session.get_inputs()[0].name
    if multi_crop:
        inputs, views = preprocess_image_views(image, labels)
    else:
        inputs = preprocess_image(image, labels)
        views = ["full"]
    logits = session.run(None, {input_name: inputs})[0]
    logits = logits.mean(axis=0, keepdims=True)
    probabilities = softmax(logits)
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    return {
        "engine": "onnx",
        "elapsedMs": elapsed_ms,
        "views": views,
        "predictions": top_k(probabilities, list(labels["classes"]), k),
    }


def predict_onnx(
    model_path: Path,
    image: Image.Image,
    labels: dict[str, Any],
    k: int = 5,
    multi_crop: bool = True,
) -> dict[str, Any]:
    session = load_onnx_session(model_path)
    return predict_onnx_session(session, image, labels, k, multi_crop)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pokemon image prediction with PyTorch or ONNX.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--labels", default="artifacts/labels.v1.json", type=Path)
    parser.add_argument("--checkpoint", default="artifacts/pokemon-efficientnet-b0.pt", type=Path)
    parser.add_argument("--onnx", default="artifacts/pokemon-efficientnet-b0-fp32.onnx", type=Path)
    parser.add_argument("--engine", choices=["torch", "onnx"], default="torch")
    parser.add_argument("--top-k", default=5, type=int)
    parser.add_argument("--single-view", action="store_true", help="Use only the full letterboxed image for inference.")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    with Image.open(args.image) as image:
        if args.engine == "torch":
            result = predict_torch(args.checkpoint, image, labels, args.top_k, not args.single_view)
        else:
            result = predict_onnx(args.onnx, image, labels, args.top_k, not args.single_view)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
