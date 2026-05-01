from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - exercised before dependencies exist
    raise SystemExit(
        "Pillow is required. Install dependencies with: "
        "python -m pip install -r requirements.txt"
    ) from exc


ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class ValidImage:
    path: Path
    label: str
    width: int
    height: int


@dataclass(frozen=True)
class InvalidImage:
    path: Path
    label: str
    reason: str


def relative_to_repo(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def iter_class_dirs(data_dir: Path) -> list[Path]:
    return sorted(
        [path for path in data_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name.casefold(),
    )


def validate_image(path: Path) -> tuple[bool, str, int | None, int | None]:
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        return False, f"unsupported_extension:{path.suffix or '<none>'}", None, None

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        return True, "ok", width, height
    except Exception as exc:  # PIL raises several image-specific exceptions.
        return False, f"unreadable:{exc.__class__.__name__}", None, None


def collect_images(data_dir: Path) -> tuple[list[str], list[ValidImage], list[InvalidImage]]:
    if not data_dir.exists():
        raise SystemExit(f"Data directory does not exist: {data_dir}")

    classes = [path.name for path in iter_class_dirs(data_dir)]
    valid: list[ValidImage] = []
    invalid: list[InvalidImage] = []

    for class_dir in iter_class_dirs(data_dir):
        for path in sorted(class_dir.iterdir(), key=lambda item: item.name.casefold()):
            if not path.is_file():
                continue
            ok, reason, width, height = validate_image(path)
            if ok:
                valid.append(ValidImage(path=path, label=class_dir.name, width=width or 0, height=height or 0))
            else:
                invalid.append(InvalidImage(path=path, label=class_dir.name, reason=reason))

    return classes, valid, invalid


def split_by_class(
    images: Iterable[ValidImage],
    class_to_idx: dict[str, int],
    val_ratio: float,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    grouped: dict[str, list[ValidImage]] = {label: [] for label in class_to_idx}
    for image in images:
        grouped[image.label].append(image)

    records: list[dict[str, object]] = []
    repo_root = Path.cwd()

    for label, label_images in grouped.items():
        shuffled = list(label_images)
        rng.shuffle(shuffled)

        if len(shuffled) <= 1:
            val_count = 0
        else:
            val_count = max(1, round(len(shuffled) * val_ratio))
            val_count = min(val_count, len(shuffled) - 1)

        val_paths = {image.path for image in shuffled[:val_count]}
        for image in shuffled:
            records.append(
                {
                    "path": relative_to_repo(image.path, repo_root),
                    "label": image.label,
                    "class_idx": class_to_idx[image.label],
                    "split": "val" if image.path in val_paths else "train",
                    "width": image.width,
                    "height": image.height,
                }
            )

    return sorted(records, key=lambda row: (str(row["split"]), str(row["label"]), str(row["path"])))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_invalid_csv(path: Path, invalid: list[InvalidImage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "reason"])
        writer.writeheader()
        for item in invalid:
            writer.writerow(
                {
                    "path": relative_to_repo(item.path, Path.cwd()),
                    "label": item.label,
                    "reason": item.reason,
                }
            )


def write_class_report(path: Path, records: list[dict[str, object]], classes: list[str]) -> None:
    counts = {
        label: {
            "class_idx": index,
            "train": 0,
            "val": 0,
            "total": 0,
        }
        for index, label in enumerate(classes)
    }
    for record in records:
        label = str(record["label"])
        split = str(record["split"])
        counts[label][split] += 1
        counts[label]["total"] += 1

    write_json(path, counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PokemonData and create train/val indexes.")
    parser.add_argument("--data-dir", default="PokemonData", type=Path)
    parser.add_argument("--artifacts-dir", default="artifacts", type=Path)
    parser.add_argument("--val-ratio", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    classes, valid, invalid = collect_images(args.data_dir)
    class_to_idx = {label: index for index, label in enumerate(classes)}
    records = split_by_class(valid, class_to_idx, args.val_ratio, args.seed)

    labels_manifest = {
        "version": "v1",
        "modelName": "pokemon-mobilenetv3-small-int8-v1",
        "inputSize": INPUT_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "classes": classes,
    }

    write_json(args.artifacts_dir / "dataset_index.json", {"records": records})
    write_json(args.artifacts_dir / "class_to_idx.json", class_to_idx)
    write_json(args.artifacts_dir / "labels.v1.json", labels_manifest)
    write_invalid_csv(args.artifacts_dir / "invalid_files.csv", invalid)
    write_class_report(args.artifacts_dir / "class_report.json", records, classes)

    train_count = sum(1 for row in records if row["split"] == "train")
    val_count = sum(1 for row in records if row["split"] == "val")
    print(f"Classes: {len(classes)}")
    print(f"Valid images: {len(valid)}")
    print(f"Invalid/excluded files: {len(invalid)}")
    print(f"Train images: {train_count}")
    print(f"Val images: {val_count}")
    print(f"Wrote: {args.artifacts_dir / 'dataset_index.json'}")
    print(f"Wrote: {args.artifacts_dir / 'labels.v1.json'}")


if __name__ == "__main__":
    main()
