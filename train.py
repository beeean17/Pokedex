from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

try:
    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
    from torchvision.transforms import functional as TF
except ImportError as exc:  # pragma: no cover - exercised before dependencies exist
    raise SystemExit(
        "Training dependencies are missing. Install them with: "
        "python -m pip install -r requirements.txt"
    ) from exc


INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def configure_ssl_cert_file() -> None:
    if os.environ.get("SSL_CERT_FILE"):
        return

    try:
        import certifi
    except ImportError:
        return

    cert_path = certifi.where()
    os.environ["SSL_CERT_FILE"] = cert_path
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cert_path)


class PokemonDataset(Dataset):
    def __init__(self, records: list[dict[str, object]], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(str(record["path"])) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(record["class_idx"])


class LetterboxResize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        image.thumbnail((self.size, self.size))
        canvas = Image.new("RGB", (self.size, self.size), (255, 255, 255))
        left = (self.size - image.width) // 2
        top = (self.size - image.height) // 2
        canvas.paste(image, (left, top))
        return canvas


class BiasedRandomResizedCrop:
    def __init__(
        self,
        size: int,
        scale: tuple[float, float],
        ratio: tuple[float, float],
        center_x: tuple[float, float],
        center_y: tuple[float, float],
    ) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.center_x = center_x
        self.center_y = center_y

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        area = width * height

        for _ in range(10):
            target_area = area * float(torch.empty(1).uniform_(self.scale[0], self.scale[1]).item())
            log_ratio = torch.empty(1).uniform_(math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = float(torch.exp(log_ratio).item())

            crop_width = round((target_area * aspect_ratio) ** 0.5)
            crop_height = round((target_area / aspect_ratio) ** 0.5)
            if crop_width <= 0 or crop_height <= 0 or crop_width > width or crop_height > height:
                continue

            center_x = width * float(torch.empty(1).uniform_(self.center_x[0], self.center_x[1]).item())
            center_y = height * float(torch.empty(1).uniform_(self.center_y[0], self.center_y[1]).item())
            left = int(round(center_x - crop_width / 2))
            top = int(round(center_y - crop_height / 2))
            left = min(max(left, 0), width - crop_width)
            top = min(max(top, 0), height - crop_height)

            return TF.resized_crop(
                image,
                top=top,
                left=left,
                height=crop_height,
                width=crop_width,
                size=[self.size, self.size],
                interpolation=transforms.InterpolationMode.BILINEAR,
            )

        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        return TF.resized_crop(
            image,
            top=top,
            left=left,
            height=side,
            width=side,
            size=[self.size, self.size],
            interpolation=transforms.InterpolationMode.BILINEAR,
        )


class PokemonViewSampler:
    def __init__(
        self,
        size: int,
        full_probability: float,
        object_probability: float,
        feature_probability: float,
        object_scale_min: float,
        feature_scale_min: float,
    ) -> None:
        self.letterbox = LetterboxResize(size)
        self.full_probability = full_probability
        self.object_probability = object_probability
        self.feature_probability = feature_probability
        self.object_crop = BiasedRandomResizedCrop(
            size,
            scale=(object_scale_min, 1.0),
            ratio=(0.75, 1.33),
            center_x=(0.35, 0.65),
            center_y=(0.35, 0.65),
        )
        self.feature_crop = BiasedRandomResizedCrop(
            size,
            scale=(feature_scale_min, 0.55),
            ratio=(0.75, 1.33),
            center_x=(0.35, 0.65),
            center_y=(0.22, 0.58),
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        draw = torch.rand(1).item()
        if draw < self.full_probability:
            return self.letterbox(image)
        if draw < self.full_probability + self.object_probability:
            return self.object_crop(image)
        if self.feature_probability > 0:
            return self.feature_crop(image)
        return self.letterbox(image)


def load_records(index_path: Path) -> list[dict[str, object]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return list(payload["records"])


def build_model(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested, but torch.backends.mps.is_available() is false.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_view_config(args: argparse.Namespace) -> dict[str, float]:
    uses_legacy_crop = args.crop_probability is not None
    uses_view_probabilities = any(
        value is not None
        for value in (args.full_image_probability, args.object_crop_probability, args.feature_crop_probability)
    )
    if uses_legacy_crop and uses_view_probabilities:
        raise SystemExit("Use either --crop-probability or the explicit view probabilities, not both.")

    if uses_legacy_crop:
        crop_probability = float(args.crop_probability)
        if not 0.0 <= crop_probability <= 1.0:
            raise SystemExit("--crop-probability must be between 0.0 and 1.0.")
        return {
            "fullProbability": 1.0 - crop_probability,
            "objectProbability": crop_probability * 0.6,
            "featureProbability": crop_probability * 0.4,
            "objectScaleMin": max(float(args.crop_scale_min), 0.45),
            "featureScaleMin": min(float(args.crop_scale_min), 0.55),
        }

    config = {
        "fullProbability": 0.5 if args.full_image_probability is None else float(args.full_image_probability),
        "objectProbability": 0.3 if args.object_crop_probability is None else float(args.object_crop_probability),
        "featureProbability": 0.2 if args.feature_crop_probability is None else float(args.feature_crop_probability),
        "objectScaleMin": float(args.object_crop_scale_min),
        "featureScaleMin": float(args.feature_crop_scale_min),
    }

    probabilities = [config["fullProbability"], config["objectProbability"], config["featureProbability"]]
    if any(probability < 0.0 or probability > 1.0 for probability in probabilities):
        raise SystemExit("View probabilities must be between 0.0 and 1.0.")
    if abs(sum(probabilities) - 1.0) > 1e-6:
        raise SystemExit("View probabilities must sum to 1.0.")
    return config


def make_transforms(view_config: dict[str, float]) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            PokemonViewSampler(
                INPUT_SIZE,
                full_probability=view_config["fullProbability"],
                object_probability=view_config["objectProbability"],
                feature_probability=view_config["featureProbability"],
                object_scale_min=view_config["objectScaleMin"],
                feature_scale_min=view_config["featureScaleMin"],
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.85, 1.15),
                fill=(255, 255, 255),
            ),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.2, fill=(255, 255, 255)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
        ]
    )
    val_transform = transforms.Compose(
        [
            LetterboxResize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def accuracy(logits: torch.Tensor, targets: torch.Tensor, top_k: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    max_k = min(max(top_k), logits.shape[1])
    _, predictions = logits.topk(max_k, dim=1)
    predictions = predictions.t()
    correct = predictions.eq(targets.reshape(1, -1).expand_as(predictions))

    results: dict[str, float] = {}
    for k in top_k:
        capped_k = min(k, logits.shape[1])
        correct_k = correct[:capped_k].reshape(-1).float().sum(0)
        results[f"top{ k }"] = float(correct_k.item() / targets.size(0))
    return results


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total = 0
    top1 = 0.0
    top5 = 0.0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            batch_size = targets.size(0)
            metrics = accuracy(logits, targets)

            total_loss += float(loss.item()) * batch_size
            top1 += metrics["top1"] * batch_size
            top5 += metrics["top5"] * batch_size
            total += batch_size

            predicted = logits.argmax(dim=1).detach().cpu()
            for true_idx, pred_idx in zip(targets.detach().cpu(), predicted, strict=True):
                confusion[int(true_idx), int(pred_idx)] += 1

    return {
        "loss": total_loss / max(total, 1),
        "top1": top1 / max(total, 1),
        "top5": top5 / max(total, 1),
        "confusionMatrix": confusion.tolist(),
    }


def main() -> None:
    configure_ssl_cert_file()

    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 for Pokemon classification.")
    parser.add_argument("--index", default="artifacts/dataset_index.json", type=Path)
    parser.add_argument("--labels", default="artifacts/labels.v1.json", type=Path)
    parser.add_argument("--output", default="artifacts/pokemon-efficientnet-b0.pt", type=Path)
    parser.add_argument("--metrics-output", default="artifacts/training_metrics.json", type=Path)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--crop-probability",
        default=None,
        type=float,
        help="Legacy shortcut: total probability of cropped training views instead of full letterbox.",
    )
    parser.add_argument(
        "--crop-scale-min",
        default=0.35,
        type=float,
        help="Legacy smallest area ratio for --crop-probability feature-like crops.",
    )
    parser.add_argument(
        "--full-image-probability",
        default=None,
        type=float,
        help="Probability of training on the full letterboxed image. Default: 0.50.",
    )
    parser.add_argument(
        "--object-crop-probability",
        default=None,
        type=float,
        help="Probability of training on a large center-biased object crop. Default: 0.30.",
    )
    parser.add_argument(
        "--feature-crop-probability",
        default=None,
        type=float,
        help="Probability of training on a smaller upper/center-biased feature crop. Default: 0.20.",
    )
    parser.add_argument(
        "--object-crop-scale-min",
        default=0.55,
        type=float,
        help="Smallest area ratio for object-centered crops.",
    )
    parser.add_argument(
        "--feature-crop-scale-min",
        default=0.25,
        type=float,
        help="Smallest area ratio for feature-centered crops.",
    )
    args = parser.parse_args()

    labels_manifest = json.loads(args.labels.read_text(encoding="utf-8"))
    classes = list(labels_manifest["classes"])
    records = load_records(args.index)
    train_records = [row for row in records if row["split"] == "train"]
    val_records = [row for row in records if row["split"] == "val"]

    if not 0.05 <= args.crop_scale_min <= 1.0:
        raise SystemExit("--crop-scale-min must be between 0.05 and 1.0.")
    if not 0.05 <= args.object_crop_scale_min <= 1.0:
        raise SystemExit("--object-crop-scale-min must be between 0.05 and 1.0.")
    if not 0.05 <= args.feature_crop_scale_min <= 0.55:
        raise SystemExit("--feature-crop-scale-min must be between 0.05 and 0.55.")

    view_config = resolve_view_config(args)

    train_transform, val_transform = make_transforms(view_config)
    train_loader = DataLoader(
        PokemonDataset(train_records, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        PokemonDataset(val_records, val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(
        "Training augmentation: "
        f"{view_config['fullProbability']:.0%} full letterbox, "
        f"{view_config['objectProbability']:.0%} object-centered crop "
        f"(scale={view_config['objectScaleMin']:.2f}-1.00), "
        f"{view_config['featureProbability']:.0%} feature-centered crop "
        f"(scale={view_config['featureScaleMin']:.2f}-0.55)"
    )
    model = build_model(
        num_classes=len(classes),
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )

    best_top1 = -1.0
    history: list[dict[str, object]] = []
    started_at = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        val_metrics = evaluate(model, val_loader, criterion, device, len(classes))
        epoch_metrics = {
            "epoch": epoch,
            "trainLoss": running_loss / max(seen, 1),
            "valLoss": val_metrics["loss"],
            "valTop1": val_metrics["top1"],
            "valTop5": val_metrics["top5"],
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={epoch_metrics['trainLoss']:.4f} "
            f"val_top1={epoch_metrics['valTop1']:.4f} "
            f"val_top5={epoch_metrics['valTop5']:.4f}"
        )

        if float(val_metrics["top1"]) > best_top1:
            best_top1 = float(val_metrics["top1"])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "modelName": "pokemon-efficientnet-b0",
                    "stateDict": model.state_dict(),
                    "classes": classes,
                    "inputSize": INPUT_SIZE,
                    "mean": IMAGENET_MEAN,
                    "std": IMAGENET_STD,
                    "metrics": val_metrics,
                    "augmentation": {
                        **view_config,
                    },
                },
                args.output,
            )

    final_metrics = {
        "durationSeconds": round(time.time() - started_at, 3),
        "bestValTop1": best_top1,
        "augmentation": {
            **view_config,
        },
        "history": history,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(final_metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved best checkpoint: {args.output}")
    print(f"Saved metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
