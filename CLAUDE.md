# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Docker-based pipeline for training and deploying deep learning segmentation models on Whole Slide Images (WSI). Three sequential inference stages:
1. **Tissue/background (TB)** segmentation — U-Net + MobileNetV2 at 4.0 μm spacing
2. **Epithelium/tumor** segmentation — ensemble of 5 models at 1.0 μm spacing
3. **Tumor-stroma ratio (TSR)** computation — hotspot detection with concave hull

## Key Commands

### Build Docker Image
```bash
docker build -t pathology-pipeline .
```

### Download Model Weights (from HuggingFace)
```bash
docker run --gpus all \
  -v /host/models:/home/user/source/models \
  pathology-pipeline python3 /home/user/source/download_models.py [tb|epithelium|multi-tissue|all]
```

### Run Full Inference Pipeline
```bash
docker run --gpus all \
  -v /path/to/data:/home/user/data \
  -v /path/to/output:/home/user/process \
  -v /path/to/models:/home/user/source/models \
  pathology-pipeline \
  bash /home/user/source/code/start_characterization.sh /home/user/data/slide.tif
```

### Train a Model
```bash
docker run --gpus all -it pathology-pipeline \
  python3 /home/user/source/code/pytorch_exp_run.py \
  --project_name <name> \
  --data_path /path/to/data.yaml \
  --config_path /home/user/source/code/network_configuration.yaml \
  --output_path /path/to/output
```

### Evaluate (Dice/Jaccard)
```bash
docker run --gpus all pathology-pipeline \
  python3 /home/user/source/code/awesomedice.py \
  --input_mask_path "/results/*.tif" \
  --ground_truth_path "/gt/{image}.tif" \
  --classes "{'background': 1, 'epithelium': 2, 'stroma': 3}" \
  --spacing 1.0 \
  --output_path /results/scores.yaml
```

### Interactive Container (JupyterLab on port 8888)
```bash
docker run --gpus all -p 8888:8888 \
  -v /path/to/data:/home/user/data \
  pathology-pipeline
```

## Architecture

### Module Layout

```
pathology-segmentation-pipeline/
├── code/                          # Inference orchestration & training scripts
├── pathology-common/              # Core WSI utilities (digitalpathology library)
└── pathology-fast-inference/      # Async GPU tile-processing engine
```

### `pathology-common/digitalpathology/`
Core library (~115 files). Key subdirectories:
- `image/io/` — WSI I/O via ASAP
- `adapters/` — Augmentation (Albumentations), normalization, label adapters
- `generator/batch/`, `generator/patch/`, `generator/mask/` — Patch extraction & sampling

### `pathology-fast-inference/fastinference/`
Async engine for GPU inference on large WSIs. Key files:
- `async_wsi_consumer.py` — Orchestrates readers → processor → writers
- `async_tile_processor.py` / `torch_processor.py` — GPU inference (single model or ensemble)
- `async_wsi_reader.py` / `async_wsi_writer.py` — Parallel tile I/O
- `scripts/applynetwork_multiproc.py` — CLI entry point

### `code/`
- `start_characterization.sh` — Orchestrates the 3-stage inference pipeline
- `pytorch_exp_run.py` — Training loop (interactive arch selection, AMP, WandB, Rich progress)
- `torch_data_generator.py` — DataLoader wrapper
- `compute_tsr.py` + `concave_hull.py` — TSR heatmap & hotspot computation
- `convert.py` — WSI format conversion (non-TIFF → TIFF via ASAP)
- `awesomedice.py` — Per-class Dice/Jaccard evaluation
- `network_configuration.yaml` — Model architecture & training hyperparameters

## Configuration

Model architecture, loss function, and training schedule are controlled by `code/network_configuration.yaml`. Supported architectures (via segmentation-models-pytorch): `unet`, `unet-plus`, `manet`, `linknet`, `fpn`, `pspnet`, `deeplabv3`, `deeplabv3+`, `pan`. Backbone selection is also configurable (e.g., EfficientNet-B0, MobileNetV3, SE-ResNeXt50).

## Container Details

- **Base:** `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`, Python 3.10
- **WSI I/O:** ASAP 2.2 (requires system Qt5/Boost/OpenGL libs installed in Dockerfile)
- **PYTHONPATH** inside container includes both `pathology-common` and `pathology-fast-inference`
- **Non-root user:** `user` (UID 1000)
- **Mount points:** `/home/user/data/` (input), `/home/user/process/` (output), `/home/user/source/models/` (weights cache)
- **Default entrypoint** (`execute.sh`): no args → SSH + JupyterLab; with args → runs the command as `user`

## Model Hosting

Models are on HuggingFace Hub at `PierpaoloV93/pathology-segmentation-models`. `download_models.py` handles selective downloads by family (`tb`, `epithelium`, `multi-tissue`). Model weights (`.pt`) are excluded from git.

## No Test Suite

There are no automated tests or CI/CD pipelines. `awesomedice.py` serves as the primary validation tool (run manually against held-out data).
