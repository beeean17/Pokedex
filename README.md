# Pokedex

Pokemon classifier pipeline with Python training and a static React browser classifier.

## First Milestone

1. Prepare the local dataset index from `PokemonData/`.
2. Train a MobileNetV3-Small transfer-learning checkpoint.
3. Check dataset, training metrics, and sample predictions in the console.

## Setup

```bash
python -m pip install -r requirements.txt
npm install
```

PyTorch may require a Python version supported by the official wheels. If install fails on a very new Python runtime, create a Python 3.11 or 3.12 virtual environment and reinstall the requirements.

## Python ML Pipeline

```bash
python prepare_data.py
python console_report.py
python train.py --epochs 20 --batch-size 32 --lr 5e-5 --device cuda
python console_report.py
python predict.py PokemonData/Pikachu/00000004.jpg --engine torch
```

Generated outputs:

- `artifacts/dataset_index.json`
- `artifacts/class_to_idx.json`
- `artifacts/labels.v1.json`
- `artifacts/pokemon-mobilenetv3-small.pt`
- `artifacts/training_metrics.json`

Previous baseline from `--epochs 8 --batch-size 32 --freeze-backbone`:

- validation top-1: 77.81%
- validation top-5: 92.14%
- checkpoint size: about 6.6 MB

The current recommended training run fine-tunes the full MobileNetV3-Small backbone:

```bash
python train.py --epochs 20 --batch-size 32 --lr 5e-5 --device cuda
```

If CUDA is not available on the machine, omit `--device cuda` to let the script fall back to the best available device.

## Optional ONNX Export

```bash
python export_onnx.py --smoke-image PokemonData/Pikachu/00000004.jpg
```

Generated output:

- `artifacts/pokemon-mobilenetv3-small-int8-v1.onnx`

For the React app, copy the FP32 browser model and labels into `public/models/`:

```powershell
Copy-Item artifacts/pokemon-mobilenetv3-small-fp32.onnx public/models/pokemon-mobilenetv3-small-fp32.onnx -Force
Copy-Item artifacts/labels.v1.json public/models/labels.v1.json -Force
```

## Static React Classifier

The React app runs inference entirely in the browser with `onnxruntime-web`. Images are preprocessed locally in a canvas and are not uploaded to a server.

```bash
npm run dev
npm run build
```

Local app URL:

```text
http://127.0.0.1:5173
```
