#!/usr/bin/env python3
"""
Upload model weights to HuggingFace Hub.

Usage:
    python3 upload_models_to_hf.py

Edit the CONFIGURATION section below before running.
"""

from huggingface_hub import HfApi, login

# -----------------------------------------------------------------------
# CONFIGURATION — edit these before running
# -----------------------------------------------------------------------

# Your HuggingFace username or organisation name
HF_USERNAME = "PierpaoloV93"

# Repository name to create (or update if it already exists)
HF_REPO_NAME = "pathology-segmentation-models"

# Set to True to make the repo private
PRIVATE = True

# Local path to the models/ directory
MODELS_DIR = "./models"

# -----------------------------------------------------------------------

if __name__ == "__main__":
    repo_id = f"{HF_USERNAME}/{HF_REPO_NAME}"

    # Log in — opens a prompt asking for your HF token.
    # Get your token at: https://huggingface.co/settings/tokens
    # You only need to do this once; the token is cached locally.
    login()

    api = HfApi()

    print(f"Creating repo '{repo_id}' (private={PRIVATE}) ...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=PRIVATE,
        exist_ok=True,
    )

    print(f"Uploading {MODELS_DIR} ...")
    api.upload_folder(
        folder_path=MODELS_DIR,
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Done. Models available at: https://huggingface.co/{repo_id}")
