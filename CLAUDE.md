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

### Upload Trained Models to HuggingFace
```bash
# Edit the CONFIGURATION section at the top of the file first
python3 upload_models_to_hf.py
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

Inference outputs are written to:

| Path | Content |
|------|---------|
| `/home/user/process/tb/` | Tissue/background masks |
| `/home/user/process/epithelium/` | Epithelium segmentation masks |
| `/home/user/process/tumor/` | Tumour masks |
| `/home/user/process/concave_hull_masks/` | Concave hull annotations |

### Train a Model
```bash
docker run --gpus all -it pathology-pipeline \
  python3 /home/user/source/code/pytorch_exp_run.py \
  --project_name <name> \
  --data_path /path/to/data.yaml \
  --config_path /home/user/source/code/network_configuration.yaml \
  --output_path /path/to/output
```

At startup, training interactively prompts to confirm or change the segmentation architecture before proceeding. Training progress is shown via Rich live progress bars with per-epoch loss, IoU, and LR.

### Evaluate (Dice/Jaccard)
```bash
docker run --gpus all pathology-pipeline \
  python3 /home/user/source/code/awesomedice.py \
  --input_mask_path "/results/*.tif" \
  --ground_truth_path "/gt/{image}.tif" \
  --classes "{'background': 1, 'epithelium': 2, 'stroma': 3}" \
  --spacing 1.0 \
  --output_path /results/scores.yaml \
  --mapping "{'background': 1, 'epithelium': 2, 'stroma': 3}" \
  --all_cm
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

Also contains `pathology-common/scripts/` with 19 CLI data-prep tools including `convertannotations.py`, `extractpatches.py`, `createdataconfig.py`, `createfolds.py`, `preprocessmasks.py`, `combinemasks.py`, and others for dataset preparation workflows.

### `pathology-fast-inference/fastinference/`
Async engine for GPU inference on large WSIs. Key files:
- `async_wsi_consumer.py` — Orchestrates readers → processor → writers
- `async_tile_processor.py` / `torch_processor.py` — GPU inference (single model or ensemble)
- `async_wsi_reader.py` / `async_wsi_writer.py` — Parallel tile I/O
- `scripts/applynetwork_multiproc.py` — CLI entry point for segmentation inference
- `gan_inference/` — CycleGAN-based stain transfer inference (`scripts/applygan_multiproc.py`)

### `code/`
- `start_characterization.sh` — Orchestrates the 3-stage inference pipeline
- `pytorch_exp_run.py` — Training loop (interactive arch selection, AMP, WandB, Rich progress)
- `torch_data_generator.py` — DataLoader wrapper
- `compute_tsr.py` + `concave_hull.py` — TSR heatmap & hotspot computation
- `convert.py` — WSI format conversion (non-TIFF → TIFF via ASAP)
- `awesomedice.py` — Per-class Dice/Jaccard evaluation
- `network_configuration.yaml` — Model architecture & training hyperparameters

## Configuration

Model architecture, loss function, training schedule, and patch sampling are controlled by `code/network_configuration.yaml`. Key sections:

```yaml
model:
    modelname: 'unet'          # unet, unet-plus, manet, linknet, fpn, pspnet, deeplabv3, deeplabv3+, pan
    backbone: 'efficientnet-b0' # or mobilenet_v2, timm-mobilenetv3_large_100, se_resnext50_32x4d
    loss: 'lovasz'             # lovasz, dice, cc
    learning_rate: 0.0001
    learning_rate_schedule: 'plateau'
sampler:
    training:
        iterations: 1250
        label_dist: {1: 1.0, 2: 2.0, 3: 5.0}  # sampling weight per class label
        label_map: {1: 0, 2: 1, 3: 2}          # remap annotation labels to model outputs
        patch_shapes: {1.0: [512, 512]}          # spacing (µm): [H, W]
        mask_spacing: 2.0
training:
    epochs: 100
    stop_plateau: 50           # early stopping patience
    training_batch_size: 10
    mixed_precision: true
```

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

## Known Dependency Conflicts

### `timm` version conflict in the unified image (`Dockerfile`)

`segmentation-models-pytorch==0.3.4` (used by our pipeline) is only compatible with `timm<1.0` — it relies on internal timm APIs (`timm.models.layers`, etc.) that were removed/restructured in timm 1.0.

`slide2vec` requires `timm>=1.0` (currently pinned to `timm==1.0.8` in the unified Dockerfile).

These requirements are **mutually exclusive**. As of now the unified image installs `timm==1.0.8` last, which means `segmentation-models-pytorch` will fail at import time inside the unified container. The non-unified `Dockerfile.backup` is unaffected because it does not install slide2vec or pin timm.

**Possible resolutions (not yet implemented):**
- Upgrade to `segmentation-models-pytorch>=0.4.x` which added timm 1.x compatibility — but requires auditing all training/inference code for API changes.
- Patch the `segmentation-models-pytorch` 0.3.4 source at build time to use the updated timm API.
- Keep segmentation inference in a separate container (i.e., do not unify).
