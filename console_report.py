from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_invalid_count(path: Path) -> tuple[int, list[dict[str, str]]]:
    if not path.exists():
        return 0, []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows), rows


def print_dataset_report(class_report_path: Path, invalid_path: Path) -> None:
    class_report = load_json(class_report_path, {})
    invalid_count, invalid_rows = load_invalid_count(invalid_path)

    if not class_report:
        print("Dataset report: missing. Run `python prepare_data.py` first.")
        return

    class_rows = [
        {
            "label": label,
            "train": int(stats["train"]),
            "val": int(stats["val"]),
            "total": int(stats["total"]),
        }
        for label, stats in class_report.items()
    ]
    total_train = sum(row["train"] for row in class_rows)
    total_val = sum(row["val"] for row in class_rows)
    total_images = sum(row["total"] for row in class_rows)
    smallest = sorted(class_rows, key=lambda row: row["total"])[:5]
    largest = sorted(class_rows, key=lambda row: row["total"], reverse=True)[:5]

    print("== Dataset ==")
    print(f"Classes: {len(class_rows)}")
    print(f"Valid images: {total_images}")
    print(f"Train / Val: {total_train} / {total_val}")
    print(f"Excluded files: {invalid_count}")
    print()

    print("Smallest classes:")
    for row in smallest:
        print(f"  {row['label']}: {row['total']} (train={row['train']}, val={row['val']})")
    print()

    print("Largest classes:")
    for row in largest:
        print(f"  {row['label']}: {row['total']} (train={row['train']}, val={row['val']})")
    print()

    if invalid_rows:
        print("Excluded file examples:")
        for row in invalid_rows[:8]:
            print(f"  {row['path']} - {row['reason']}")
        print()


def print_training_report(metrics_path: Path) -> None:
    metrics = load_json(metrics_path)
    if not metrics:
        print("Training report: missing. Run `python train.py` after preparing data.")
        return

    print("== Training ==")
    print(f"Duration: {metrics.get('durationSeconds', 'n/a')}s")
    print(f"Best val top-1: {float(metrics.get('bestValTop1', 0.0)):.2%}")
    augmentation = metrics.get("augmentation")
    if augmentation:
        print(
            "Augmentation: "
            f"{float(augmentation['cropProbability']):.0%} random crop, "
            f"crop scale {float(augmentation['cropScaleMin']):.2f}-1.00"
        )
    print()

    history = metrics.get("history", [])
    if not history:
        return

    print("Epoch history:")
    for row in history:
        print(
            f"  epoch {row['epoch']:>2}: "
            f"train_loss={row['trainLoss']:.4f}, "
            f"val_loss={row['valLoss']:.4f}, "
            f"top1={row['valTop1']:.2%}, "
            f"top5={row['valTop5']:.2%}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Print console summaries for the Pokemon ML pipeline.")
    parser.add_argument("--class-report", default="artifacts/class_report.json", type=Path)
    parser.add_argument("--invalid-files", default="artifacts/invalid_files.csv", type=Path)
    parser.add_argument("--training-metrics", default="artifacts/training_metrics.json", type=Path)
    args = parser.parse_args()

    print_dataset_report(args.class_report, args.invalid_files)
    print_training_report(args.training_metrics)


if __name__ == "__main__":
    main()
