from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
except ImportError as exc:  # pragma: no cover - exercised before dependencies exist
    raise SystemExit(
        "Training dependencies are missing. Install them with: "
        "python -m pip install -r requirements.txt"
    ) from exc


INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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


def load_records(index_path: Path) -> list[dict[str, object]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return list(payload["records"])


def build_model(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
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


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            LetterboxResize(INPUT_SIZE),
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
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Small for Pokemon classification.")
    parser.add_argument("--index", default="artifacts/dataset_index.json", type=Path)
    parser.add_argument("--labels", default="artifacts/labels.v1.json", type=Path)
    parser.add_argument("--output", default="artifacts/pokemon-mobilenetv3-small.pt", type=Path)
    parser.add_argument("--metrics-output", default="artifacts/training_metrics.json", type=Path)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    labels_manifest = json.loads(args.labels.read_text(encoding="utf-8"))
    classes = list(labels_manifest["classes"])
    records = load_records(args.index)
    train_records = [row for row in records if row["split"] == "train"]
    val_records = [row for row in records if row["split"] == "val"]

    train_transform, val_transform = make_transforms()
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
                    "modelName": "pokemon-mobilenetv3-small",
                    "stateDict": model.state_dict(),
                    "classes": classes,
                    "inputSize": INPUT_SIZE,
                    "mean": IMAGENET_MEAN,
                    "std": IMAGENET_STD,
                    "metrics": val_metrics,
                },
                args.output,
            )

    final_metrics = {
        "durationSeconds": round(time.time() - started_at, 3),
        "bestValTop1": best_top1,
        "history": history,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(final_metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved best checkpoint: {args.output}")
    print(f"Saved metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
