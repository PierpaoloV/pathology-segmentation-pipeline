#!/usr/bin/env python3
"""
Download model weights from HuggingFace Hub on demand.

Usage:
    python3 download_models.py tb
    python3 download_models.py epithelium
    python3 download_models.py multi-tissue
    python3 download_models.py sam
    python3 download_models.py all

Configuration via environment variables:
  HF_REPO_ID      — HuggingFace repo for the pathology models
                    (default: hardcoded below)
  HF_SAM_REPO_ID  — HuggingFace repo for the Atlas SAM model/config
                    (default: hardcoded below)
  HF_TOKEN        — token for private repos (optional)
  MODELS_DIR      — local destination   (default: /home/user/source/models)
"""

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
HF_REPO_ID = os.environ.get("HF_REPO_ID", "PierpaoloV93/pathology-segmentation-models")
HF_SAM_REPO_ID = os.environ.get("HF_SAM_REPO_ID", "AtlasAnalyticsLab/AtlasPatch")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/home/user/source/models"))

# Each family maps to:
#   "allow_patterns" — files to download from the HF repo
#   "sentinel"       — a single file that confirms the family is present
FAMILIES = {
    "tb": {
        "repo_id": HF_REPO_ID,
        "local_dir": MODELS_DIR,
        "allow_patterns": ["tb/**"],
        "sentinel": "tb/playground_soft-cloud-137_best_model.pt",
    },
    "epithelium": {
        "repo_id": HF_REPO_ID,
        "local_dir": MODELS_DIR,
        "allow_patterns": ["epithelium/**"],
        "sentinel": "epithelium/best_models/Tumour_vs_Healthy_Epitheilum_vivid-dew-9_best_model.pt",
    },
    "multi-tissue": {
        "repo_id": HF_REPO_ID,
        "local_dir": MODELS_DIR,
        "allow_patterns": ["multi-tissue/**"],
        "sentinel": "multi-tissue/best_models/Multi_Tissue_augmentations_treasured-planet-1_best_model.pt",
    },
    "sam": {
        "repo_id": HF_SAM_REPO_ID,
        "local_dir": MODELS_DIR / "sam",
        "allow_patterns": ["model.pth", "sam2.1_hiera_t.yaml"],
        "sentinel": "model.pth",
    },
}

VALID_FLAGS = list(FAMILIES.keys()) + ["all"]


def download_family(name: str) -> None:
    cfg = FAMILIES[name]
    local_dir = Path(cfg["local_dir"])
    sentinel = local_dir / cfg["sentinel"]
    if sentinel.exists():
        print(f"[{name}] Already present — skipping.")
        return
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] Downloading from {cfg['repo_id']} ...")
    snapshot_download(
        repo_id=cfg["repo_id"],
        repo_type="model",
        local_dir=str(local_dir),
        token=HF_TOKEN,
        allow_patterns=cfg["allow_patterns"],
        ignore_patterns=["*.git*", "*.gitattributes"],
    )
    print(f"[{name}] Done → {local_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in VALID_FLAGS:
        print(f"Usage: python3 download_models.py [{' | '.join(VALID_FLAGS)}]")
        sys.exit(1)

    flag = sys.argv[1]
    targets = list(FAMILIES.keys()) if flag == "all" else [flag]

    for name in targets:
        download_family(name)
