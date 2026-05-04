# PokeFinder

Pokemon classifier pipeline with Python training and a static React browser classifier.

## First Milestone

1. Prepare the local dataset index from `PokemonData/`.
2. Train an EfficientNet-B0 transfer-learning checkpoint.
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
python train.py --epochs 20 --batch-size 32 --lr 5e-5 --device cuda --crop-probability 0.55 --crop-scale-min 0.35
python console_report.py
python predict.py PokemonData/Pikachu/00000004.jpg --engine torch
```

Generated outputs:

- `artifacts/dataset_index.json`
- `artifacts/class_to_idx.json`
- `artifacts/labels.v1.json`
- `artifacts/pokemon-efficientnet-b0.pt`
- `artifacts/training_metrics.json`

Previous baseline from `--epochs 8 --batch-size 32 --freeze-backbone`:

- validation top-1: 77.81%
- validation top-5: 92.14%
- checkpoint size: about 6.6 MB

The current recommended training run fine-tunes the full EfficientNet-B0 backbone:

```bash
python train.py --epochs 20 --batch-size 32 --lr 5e-5 --device cuda --crop-probability 0.55 --crop-scale-min 0.35
```

If CUDA is not available on the machine, omit `--device cuda` to let the script fall back to the best available device.

For face-only or partial Pokemon images, the training pipeline now mixes full letterboxed images with random partial crops. Increase `--crop-probability` if partial images matter more, or lower it if full-body accuracy drops.

## Optional ONNX Export

```bash
python export_onnx.py --smoke-image PokemonData/Pikachu/00000004.jpg
```

Generated output:

- `artifacts/pokemon-efficientnet-b0-fp32.onnx`
- `artifacts/pokemon-efficientnet-b0-int8-v1.onnx` if int8 quantization succeeds

For the React app, copy the FP32 browser model and labels into `public/models/`:

```powershell
Copy-Item artifacts/pokemon-efficientnet-b0-fp32.onnx public/models/pokemon-efficientnet-b0-fp32.onnx -Force
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
