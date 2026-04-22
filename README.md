# Pathology Segmentation Pipeline

Docker-based whole-slide pathology pipeline for tissue/background segmentation, epithelium or multi-tissue segmentation, and tumour-stroma ratio (TSR) computation.

This `atlas` branch adds an alternative tissue/background preprocessing path based on the [AtlasPatch](https://github.com/AtlasAnalyticsLab/AtlasPatch) SAM2 tissue model from Atlas Analytics Lab. The original U-Net tissue model remains available and is still the default unless `--sam` is used.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-cu128-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.8.1-green)
![Docker](https://img.shields.io/badge/Docker-GPU-informational)
![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)

## Overview

The repository contains two closely related workflows:

1. The original three-stage inference pipeline:
   - tissue/background segmentation
   - epithelium or multi-tissue segmentation
   - TSR computation
2. A dataset-level preprocessing workflow for generating tissue masks only, with either:
   - the original fast-inference TB model at `4.0` um/px
   - the AtlasPatch SAM2 tissue model at `8.0` um/px by default

The core inference stack is built around ASAP-compatible WSI IO plus the asynchronous tile engine in [pathology-fast-inference](/Users/pierpaolovendittelli/projects/pathology-segmentation-pipeline/pathology-fast-inference).

## What's New On `atlas`

This branch introduces:

- [code/process_dataset.sh](/Users/pierpaolovendittelli/projects/pathology-segmentation-pipeline/code/process_dataset.sh): dataset-level tissue-mask preprocessing entrypoint
- [code/atlas_tb_mask.py](/Users/pierpaolovendittelli/projects/pathology-segmentation-pipeline/code/atlas_tb_mask.py): single-slide Atlas SAM2 tissue mask writer
- [code/atlas_tb_batch.py](/Users/pierpaolovendittelli/projects/pathology-segmentation-pipeline/code/atlas_tb_batch.py): batch-oriented Atlas helper
- [download_models.py](/Users/pierpaolovendittelli/projects/pathology-segmentation-pipeline/download_models.py): now supports `sam` downloads in addition to `tb`, `epithelium`, and `multi-tissue`

The intended usage is:

- use the original TB model when you want the legacy pipeline behavior
- use `--sam` when you want Atlas-backed tissue masks for research workflows

## AtlasPatch Credit

This branch uses the AtlasPatch tissue model weights and config distributed by Atlas Analytics Lab:

- AtlasPatch GitHub: [AtlasAnalyticsLab/AtlasPatch](https://github.com/AtlasAnalyticsLab/AtlasPatch)
- AtlasPatch model card: [AtlasAnalyticsLab/AtlasPatch on Hugging Face](https://huggingface.co/AtlasAnalyticsLab/AtlasPatch)
- Paper: *AtlasPatch: An Efficient and Scalable Tool for Whole Slide Image Preprocessing in Computational Pathology* ([arXiv:2602.03998](https://arxiv.org/abs/2602.03998))

Important implementation note:

- this repository does **not** vendor the AtlasPatch codebase directly
- instead, it downloads the AtlasPatch SAM2 checkpoint and config at runtime and applies them through a local ASAP/WholeSlideData-based adapter
- the adapter follows an `hs2p`-style coarse-spacing whole-slide read path rather than AtlasPatch’s canonical `1.25x -> 1024x1024` thumbnail preprocessing

If you use this branch in research that depends on the Atlas tissue model, please credit the AtlasPatch authors and cite their paper.

## License Note For AtlasPatch

The main repository remains Apache 2.0. However, the AtlasPatch tissue model itself is distributed under `CC-BY-NC-SA-4.0` according to the AtlasPatch model card.

That means:

- the original pipeline code in this repo stays under its own license
- the optional Atlas model path is for non-commercial use unless you obtain separate rights from the AtlasPatch authors

See the AtlasPatch model card for the exact terms:

- [AtlasPatch model card on Hugging Face](https://huggingface.co/AtlasAnalyticsLab/AtlasPatch)

## Repository Structure

```text
pathology-segmentation-pipeline/
├── code/
│   ├── process_dataset.sh
│   ├── atlas_tb_mask.py
│   ├── atlas_tb_batch.py
│   ├── start_characterization.sh
│   ├── compute_tsr.py
│   ├── awesomedice.py
│   └── pytorch_exp_run.py
├── pathology-common/
├── pathology-fast-inference/
├── download_models.py
├── Dockerfile
└── execute.sh
```

## Build The Docker Image

```bash
docker build -t pathology-pipeline .
```

## Download Model Weights

The downloader now supports four model families:

```bash
python3 /home/user/source/download_models.py tb
python3 /home/user/source/download_models.py epithelium
python3 /home/user/source/download_models.py multi-tissue
python3 /home/user/source/download_models.py sam
python3 /home/user/source/download_models.py all
```

Expected locations inside the container:

- TB model: `/home/user/source/models/tb/playground_soft-cloud-137_best_model.pt`
- Atlas SAM checkpoint: `/home/user/source/models/sam/model.pth`
- Atlas SAM config: `/home/user/source/models/sam/sam2.1_hiera_t.yaml`

## Full Inference Pipeline

The original single-slide pipeline is still available through [start_characterization.sh](/Users/pierpaolovendittelli/projects/pathology-segmentation-pipeline/code/start_characterization.sh).

```bash
docker run --gpus all \
  -v /path/to/data:/home/user/data \
  -v /path/to/output:/home/user/process \
  pathology-pipeline \
  bash /home/user/source/code/start_characterization.sh /home/user/data/slide.tif
```

Outputs are written to:

- `/home/user/process/tb/`
- `/home/user/process/epithelium/`
- `/home/user/process/tumor/`
- `/home/user/process/concave_hull_masks/`

## Dataset Preprocessing

The new dataset-level entrypoint is [process_dataset.sh](/Users/pierpaolovendittelli/projects/pathology-segmentation-pipeline/code/process_dataset.sh).

It generates one tissue-mask TIFF per slide:

- input `/path/to/images/WT_S02_P000001_C0001_E0001.mrxs`
- output `/path/to/output/WT_S02_P000001_C0001_E0001.tif`

### Default TB Mode

Without `--sam`, the script:

- downloads the legacy TB model if needed
- uses [applynetwork_multiproc.py](/Users/pierpaolovendittelli/projects/pathology-segmentation-pipeline/pathology-fast-inference/scripts/applynetwork_multiproc.py)
- defaults to `--read_spacing 4.0`
- defaults to `--write_spacing 4.0`

Example:

```bash
bash /home/user/source/code/process_dataset.sh \
  --input_wsi_path /home/user/image \
  --output_wsi_path /home/user/process \
  --overwrite
```

### Atlas SAM Mode

With `--sam`, the script:

- downloads the Atlas SAM assets if needed
- runs [atlas_tb_mask.py](/Users/pierpaolovendittelli/projects/pathology-segmentation-pipeline/code/atlas_tb_mask.py)
- defaults to `--sam-input-spacing 8.0`
- keeps native output spacing by default to avoid the final resampling step that can distort mask geometry

Example:

```bash
bash /home/user/source/code/process_dataset.sh \
  --input_wsi_path /home/user/image \
  --output_wsi_path /home/user/process \
  --sam \
  --sam-input-spacing 8.0 \
  --overwrite
```

### Direct Read Vs Temporary Staging

By default, [process_dataset.sh](/Users/pierpaolovendittelli/projects/pathology-segmentation-pipeline/code/process_dataset.sh) reads slides directly from `--input_wsi_path`.

If you pass `--tmp`, it instead:

1. creates `/home/user/tmp/process_dataset`
2. copies one slide into local temporary storage
3. processes the staged local copy
4. deletes the staged copy
5. moves to the next slide

This is especially useful when the input dataset is hosted on Samba or another network share.

Example:

```bash
bash /home/user/source/code/process_dataset.sh \
  --input_wsi_path /data/shared/slides \
  --output_wsi_path /home/user/process \
  --sam \
  --sam-input-spacing 8.0 \
  --tmp \
  --overwrite
```

### Important Flags

Common:

- `--input_wsi_path`
- `--output_wsi_path`
- `--input_filter`
- `--tile_size`
- `--overwrite`
- `--tmp`

TB-specific:

- `--read_spacing`
- `--write_spacing`
- `--readers`
- `--writers`
- `--batch_size`
- `--gpu_count`

SAM-specific:

- `--sam`
- `--sam-input-spacing`
- `--spacing-at-level-0`

## Atlas SAM Adapter Behavior

[atlas_tb_mask.py](/Users/pierpaolovendittelli/projects/pathology-segmentation-pipeline/code/atlas_tb_mask.py) currently:

- opens the slide with `wholeslidedata` using ASAP backend
- reads the full slide near a requested physical spacing
- runs the Atlas SAM2 predictor on that coarse RGB view
- writes a multiresolution TIFF mask with ASAP’s writer

Important defaults:

- SAM input spacing: `8.0` um/px
- output label: `1`
- output format: multiresolution TIFF

The script also supports:

- `--keep-native-output-spacing`
- `--spacing-at-level-0`
- local checkpoint/config overrides

## Training

Training support remains available through [code/pytorch_exp_run.py](/Users/pierpaolovendittelli/projects/pathology-segmentation-pipeline/code/pytorch_exp_run.py) and [code/network_configuration.yaml](/Users/pierpaolovendittelli/projects/pathology-segmentation-pipeline/code/network_configuration.yaml).

Example:

```bash
python3 /home/user/source/code/pytorch_exp_run.py \
  --project_name my_experiment \
  --data_path /path/to/data.yaml \
  --config_path /home/user/source/code/network_configuration.yaml \
  --output_path /path/to/output
```

## Evaluation

[awesomedice.py](/Users/pierpaolovendittelli/projects/pathology-segmentation-pipeline/code/awesomedice.py) computes Dice and Jaccard scores for generated masks.

```bash
python3 /home/user/source/code/awesomedice.py \
  --input_mask_path "/results/*.tif" \
  --ground_truth_path "/gt/{image}.tif" \
  --classes "{'background': 1, 'epithelium': 2, 'stroma': 3}" \
  --spacing 1.0 \
  --output_path /results/scores.yaml
```

## Key Dependencies

- PyTorch
- CUDA 12.8.1
- ASAP
- wholeslidedata
- segmentation-models-pytorch
- huggingface_hub
- sam2
- jupyterlab

## Acknowledgements

This repository builds on its own internal async inference stack plus external open-source tooling. For the Atlas tissue path in particular, thanks to:

- Atlas Analytics Lab for AtlasPatch and the released SAM2 tissue model
- Facebook Research for SAM2
- the `hs2p` project for helping shape the coarse-spacing ASAP input strategy used in this branch

## License

This repository is licensed under Apache License 2.0. See [LICENSE](/Users/pierpaolovendittelli/projects/pathology-segmentation-pipeline/LICENSE).

The optional Atlas model assets are governed separately by the AtlasPatch license and terms.
