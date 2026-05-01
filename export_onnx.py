from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import models
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Export dependencies are missing. Install them with: "
        "python -m pip install -r requirements.txt"
    ) from exc

from predict import load_labels, predict_onnx, predict_torch


def build_model(checkpoint_path: Path, num_classes: int) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(checkpoint["stateDict"])
    model.eval()
    return model


def quantize_dynamic(fp32_path: Path, int8_path: Path) -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic as ort_quantize_dynamic
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("onnxruntime quantization tools are required for int8 export.") from exc

    ort_quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )


def compare_predictions(
    image_path: Path,
    checkpoint_path: Path,
    onnx_path: Path,
    labels_path: Path,
) -> dict[str, object]:
    labels = load_labels(labels_path)
    with Image.open(image_path) as image:
        torch_result = predict_torch(checkpoint_path, image, labels)
    with Image.open(image_path) as image:
        onnx_result = predict_onnx(onnx_path, image, labels)

    return {
        "image": str(image_path),
        "torchTop1": torch_result["predictions"][0],
        "onnxTop1": onnx_result["predictions"][0],
        "torchPredictions": torch_result["predictions"],
        "onnxPredictions": onnx_result["predictions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Pokemon checkpoint to ONNX and int8 ONNX.")
    parser.add_argument("--checkpoint", default="artifacts/pokemon-mobilenetv3-small.pt", type=Path)
    parser.add_argument("--labels", default="artifacts/labels.v1.json", type=Path)
    parser.add_argument("--fp32-output", default="artifacts/pokemon-mobilenetv3-small-fp32.onnx", type=Path)
    parser.add_argument("--int8-output", default="artifacts/pokemon-mobilenetv3-small-int8-v1.onnx", type=Path)
    parser.add_argument("--smoke-image", type=Path)
    parser.add_argument("--opset", default=17, type=int)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    model = build_model(args.checkpoint, len(labels["classes"]))
    dummy = torch.randn(1, 3, int(labels["inputSize"]), int(labels["inputSize"]))

    args.fp32_output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.fp32_output,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"Wrote fp32 ONNX: {args.fp32_output}")

    onnx_output = args.fp32_output
    try:
        args.int8_output.parent.mkdir(parents=True, exist_ok=True)
        quantize_dynamic(args.fp32_output, args.int8_output)
        onnx_output = args.int8_output
        print(f"Wrote int8 ONNX: {args.int8_output}")
    except Exception as exc:
        print(f"Skipped int8 quantization: {exc}")

    if args.smoke_image:
        comparison = compare_predictions(args.smoke_image, args.checkpoint, onnx_output, args.labels)
        smoke_output = args.fp32_output.parent / "onnx_smoke_test.json"
        smoke_output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote smoke test comparison: {smoke_output}")


if __name__ == "__main__":
    main()
