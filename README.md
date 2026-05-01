# Pokedex

Pokemon classifier pipeline with console-first model training and validation.

## First Milestone

1. Prepare the local dataset index from `PokemonData/`.
2. Train a MobileNetV3-Small transfer-learning checkpoint.
3. Check dataset, training metrics, and sample predictions in the console.

The browser app milestone is intentionally deferred. This repository is currently focused on Python-based model verification.

## Setup

```bash
python -m pip install -r requirements.txt
```

PyTorch may require a Python version supported by the official wheels. If install fails on a very new Python runtime, create a Python 3.11 or 3.12 virtual environment and reinstall the requirements.

## Python ML Pipeline

```bash
python prepare_data.py
python console_report.py
python train.py --epochs 8 --batch-size 32 --freeze-backbone
python console_report.py
python predict.py PokemonData/Pikachu/00000004.jpg --engine torch
```

Generated outputs:

- `artifacts/dataset_index.json`
- `artifacts/class_to_idx.json`
- `artifacts/labels.v1.json`
- `artifacts/pokemon-mobilenetv3-small.pt`
- `artifacts/training_metrics.json`

Current baseline from `--epochs 8 --batch-size 32 --freeze-backbone`:

- validation top-1: 77.81%
- validation top-5: 92.14%
- checkpoint size: about 6.6 MB

## Optional ONNX Export

```bash
python export_onnx.py --smoke-image PokemonData/Pikachu/00000004.jpg
```

Generated output:

- `artifacts/pokemon-mobilenetv3-small-int8-v1.onnx`
