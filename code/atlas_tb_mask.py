#!/usr/bin/env python3
"""
Generate a tissue/background mask with AtlasPatch SAM2 using an hs2p-style ASAP path.

This script:
1. Opens a WSI through WholeSlideData with the ASAP backend.
2. Reads the whole slide at a coarse physical spacing (default: 8.0 um/px).
3. Runs the AtlasPatch SAM2 tissue model on that RGB array.
4. Resizes the binary mask to a chosen output spacing (default: 8.0 um/px).
5. Writes the result as a multi-resolution TIFF mask using ASAP's writer.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PATHOLOGY_COMMON_ROOT = REPO_ROOT / "pathology-common"
if str(PATHOLOGY_COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(PATHOLOGY_COMMON_ROOT))

from digitalpathology.image.io.imagewriter import ImageWriter


DEFAULT_SAM2_MODEL_REPO = "AtlasAnalyticsLab/AtlasPatch"
DEFAULT_SAM2_MODEL_FILENAME = "model.pth"
DEFAULT_SAM2_CONFIG_FILENAME = "sam2.1_hiera_t.yaml"
DEFAULT_SAM2_INPUT_SPACING_UM = 8.0
DEFAULT_SAM2_TOLERANCE = 0.05
DEFAULT_OUTPUT_SPACING_UM = 8.0
DEFAULT_OUTPUT_CLASS = 1
DEFAULT_TILE_SIZE = 512
MAX_DOWNSAMPLE_AXIS_MISMATCH_RATIO = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run AtlasPatch SAM2 tissue segmentation with an hs2p-style ASAP read path "
            "and write the result as a multi-resolution TIFF mask."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-wsi", required=True, type=Path, help="Input WSI path.")
    parser.add_argument("--output-mask", required=True, type=Path, help="Output mask TIFF path.")
    parser.add_argument(
        "--backend",
        default="asap",
        choices=["asap"],
        help="WholeSlideData backend used to read the WSI.",
    )
    parser.add_argument(
        "--sam-input-spacing",
        "--sam2-input-spacing",
        dest="sam2_input_spacing",
        type=float,
        default=DEFAULT_SAM2_INPUT_SPACING_UM,
        help="Effective spacing used to build the RGB image passed to SAM2.",
    )
    parser.add_argument(
        "--sam2-tolerance",
        type=float,
        default=DEFAULT_SAM2_TOLERANCE,
        help=(
            "Tolerance used when selecting a native level near --sam2-input-spacing. "
            "If no level is close enough, the selected level is resized to the exact target spacing."
        ),
    )
    parser.add_argument(
        "--output-spacing",
        "--write_spacing",
        dest="output_spacing",
        type=float,
        default=DEFAULT_OUTPUT_SPACING_UM,
        help="Spacing of the written mask level 0 in um/px.",
    )
    parser.add_argument(
        "--output-class",
        type=int,
        default=DEFAULT_OUTPUT_CLASS,
        help="Foreground label written into the output mask.",
    )
    parser.add_argument(
        "--keep-native-output-spacing",
        action="store_true",
        help=(
            "Keep the SAM2 mask on its native grid and write that spacing to the TIFF metadata, "
            "instead of resizing the mask to --output-spacing."
        ),
    )
    parser.add_argument(
        "--spacing-at-level-0",
        type=float,
        default=None,
        help="Optional override for the input WSI level-0 spacing in um/px.",
    )
    parser.add_argument(
        "--sam2-device",
        type=str,
        default="cuda",
        help="SAM2 device, for example cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--sam2-checkpoint-path",
        type=Path,
        default=None,
        help="Optional local SAM2 checkpoint path. If omitted, download from Hugging Face.",
    )
    parser.add_argument(
        "--sam2-config-path",
        type=Path,
        default=None,
        help="Optional local SAM2 config path. If omitted, download from Hugging Face.",
    )
    parser.add_argument(
        "--tile-size",
        "--tile_size",
        dest="tile_size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help="Tile size used when writing the multiresolution TIFF.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output mask.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_wsi.is_file():
        raise FileNotFoundError(f"Input WSI not found: {args.input_wsi}")
    if args.sam2_input_spacing <= 0:
        raise ValueError("--sam2-input-spacing must be > 0")
    if args.output_spacing <= 0:
        raise ValueError("--output-spacing must be > 0")
    if args.output_class <= 0 or args.output_class > 255:
        raise ValueError("--output-class must be in [1, 255]")
    if args.sam2_tolerance < 0:
        raise ValueError("--sam2-tolerance must be >= 0")
    if args.spacing_at_level_0 is not None and args.spacing_at_level_0 <= 0:
        raise ValueError("--spacing-at-level-0 must be > 0 when provided")
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be > 0")
    if args.output_mask.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output_mask}. Use --overwrite to replace it."
        )


def open_wsi(path: Path, backend: str):
    try:
        import wholeslidedata as wsd
    except ImportError as exc:
        raise ImportError(
            "wholeslidedata is required for the ASAP-style Atlas reader path."
        ) from exc
    return wsd.WholeSlideImage(path, backend=backend)


def get_downsamples(wsi) -> list[float]:
    level0_width, level0_height = wsi.shapes[0]
    downsamples: list[float] = []
    for width, height in wsi.shapes:
        downsample_x = float(level0_width) / float(width)
        downsample_y = float(level0_height) / float(height)
        mismatch_ratio = abs(downsample_x - downsample_y) / max(downsample_x, downsample_y)
        if mismatch_ratio > MAX_DOWNSAMPLE_AXIS_MISMATCH_RATIO:
            raise ValueError(
                f"Non-isotropic downsample at level with shape {(width, height)}: "
                f"{downsample_x} vs {downsample_y}"
            )
        downsamples.append((downsample_x + downsample_y) / 2.0)
    return downsamples


def get_effective_spacings(
    wsi,
    *,
    spacing_at_level_0: float | None,
) -> tuple[list[float], list[float]]:
    native_spacings = list(wsi.spacings)
    if not native_spacings or native_spacings[0] is None:
        if spacing_at_level_0 is None:
            raise ValueError(
                "WSI spacing metadata is missing. Provide --spacing-at-level-0 explicitly."
            )
        downsamples = get_downsamples(wsi)
        effective_spacings = [float(spacing_at_level_0) * downsample for downsample in downsamples]
        native_for_reads = effective_spacings
        return native_for_reads, effective_spacings

    native_for_reads = [float(spacing) for spacing in native_spacings]
    if spacing_at_level_0 is None:
        return native_for_reads, native_for_reads

    scale = float(spacing_at_level_0) / native_for_reads[0]
    effective_spacings = [spacing * scale for spacing in native_for_reads]
    return native_for_reads, effective_spacings


def pick_level_for_spacing(
    *,
    effective_spacings: list[float],
    requested_spacing_um: float,
    tolerance: float,
) -> tuple[int, float, bool]:
    level = int(np.argmin([abs(spacing - requested_spacing_um) for spacing in effective_spacings]))
    level_spacing = float(effective_spacings[level])

    if abs(level_spacing - requested_spacing_um) / requested_spacing_um <= tolerance:
        return level, level_spacing, False

    while level > 0 and level_spacing > requested_spacing_um:
        level -= 1
        level_spacing = float(effective_spacings[level])
        if abs(level_spacing - requested_spacing_um) / requested_spacing_um <= tolerance:
            return level, level_spacing, False

    if level_spacing > requested_spacing_um:
        pretty = ", ".join(f"{spacing:.4g}" for spacing in effective_spacings)
        raise ValueError(
            "Unable to resolve a read spacing at or below the requested spacing "
            f"({requested_spacing_um:.4g} um/px). Available spacings: [{pretty}]"
        )

    return level, level_spacing, True


def normalize_rgb_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected an RGB array, got shape {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    elif arr.shape[2] != 3:
        raise ValueError(f"Expected 3 or 4 channels, got {arr.shape[2]}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def prepare_sam2_input(
    *,
    wsi,
    requested_spacing_um: float,
    tolerance: float,
    spacing_at_level_0: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    native_spacings, effective_spacings = get_effective_spacings(
        wsi,
        spacing_at_level_0=spacing_at_level_0,
    )
    downsamples = get_downsamples(wsi)
    level, read_spacing_um, needs_resizing = pick_level_for_spacing(
        effective_spacings=effective_spacings,
        requested_spacing_um=requested_spacing_um,
        tolerance=tolerance,
    )

    width, height = wsi.shapes[level]
    native_level_spacing = float(native_spacings[level])
    image = np.asarray(
        wsi.get_patch(0, 0, int(width), int(height), spacing=native_level_spacing, center=False)
    )
    image = normalize_rgb_image(image)

    output_width = int(width)
    output_height = int(height)
    if needs_resizing:
        level0_width, level0_height = wsi.shapes[0]
        level0_spacing = float(effective_spacings[0])
        output_width = max(
            1,
            int(round(float(level0_width) * level0_spacing / float(requested_spacing_um))),
        )
        output_height = max(
            1,
            int(round(float(level0_height) * level0_spacing / float(requested_spacing_um))),
        )
        interpolation = (
            cv2.INTER_AREA
            if output_width < image.shape[1] or output_height < image.shape[0]
            else cv2.INTER_CUBIC
        )
        image = cv2.resize(image, (output_width, output_height), interpolation=interpolation)

    info = {
        "level": int(level),
        "native_level_spacing_um": float(native_level_spacing),
        "effective_read_spacing_um": float(read_spacing_um),
        "requested_sam2_spacing_um": float(requested_spacing_um),
        "resized_to_requested_spacing": bool(needs_resizing),
        "level0_spacing_um": float(effective_spacings[0]),
        "level0_width": int(wsi.shapes[0][0]),
        "level0_height": int(wsi.shapes[0][1]),
        "sam2_width": int(output_width),
        "sam2_height": int(output_height),
        "downsample": float(downsamples[level]),
    }
    return image, info


def pad_rgb_image_to_square(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    if height == width:
        return image, (height, width)
    side = max(height, width)
    padded = np.full((side, side, image.shape[2]), 255, dtype=image.dtype)
    padded[:height, :width, :] = image
    return padded, (height, width)


def validate_sam2_device(device: str) -> str:
    dev = str(device).strip().lower()
    if dev == "cpu" or dev == "cuda":
        return dev
    if dev.startswith("cuda:") and dev.split("cuda:", 1)[1].isdigit():
        return dev
    raise ValueError(f"Invalid SAM2 device '{device}'. Expected cpu, cuda, or cuda:<index>.")


class Sam2Predictor:
    def __init__(
        self,
        *,
        checkpoint_path: Path | None,
        config_path: Path | None,
        device: str,
    ) -> None:
        self.device = validate_sam2_device(device)
        self.checkpoint_path = self.resolve_checkpoint_path(checkpoint_path)
        self.config_path = self.resolve_config_path(config_path)
        self.predictor = self.load_predictor(
            checkpoint_path=self.checkpoint_path,
            config_path=self.config_path,
            device=self.device,
        )

    @staticmethod
    def resolve_checkpoint_path(checkpoint_path: Path | None) -> Path:
        if checkpoint_path is not None:
            resolved = Path(checkpoint_path)
            if not resolved.is_file():
                raise FileNotFoundError(f"SAM2 checkpoint not found: {resolved}")
            return resolved

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "Automatic checkpoint download requires huggingface-hub."
            ) from exc

        downloaded = hf_hub_download(
            repo_id=DEFAULT_SAM2_MODEL_REPO,
            filename=DEFAULT_SAM2_MODEL_FILENAME,
        )
        return Path(downloaded)

    @staticmethod
    def resolve_config_path(config_path: Path | None) -> Path:
        if config_path is not None:
            resolved = Path(config_path)
            if not resolved.is_file():
                raise FileNotFoundError(f"SAM2 config not found: {resolved}")
            return resolved

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "Automatic config download requires huggingface-hub."
            ) from exc

        downloaded = hf_hub_download(
            repo_id=DEFAULT_SAM2_MODEL_REPO,
            filename=DEFAULT_SAM2_CONFIG_FILENAME,
        )
        return Path(downloaded)

    @staticmethod
    def load_predictor(
        *,
        checkpoint_path: Path,
        config_path: Path,
        device: str,
    ):
        try:
            import torch
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError(
                "SAM2 inference requires torch, hydra-core, omegaconf, and the sam2 package."
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA requested but not available; falling back to cpu.", file=sys.stderr)
            device = "cpu"

        conf = OmegaConf.load(str(config_path))
        model_cfg: Any = conf.get("model", conf)
        model = instantiate(model_cfg)
        predictor = SAM2ImagePredictor(model, mask_threshold=0.0)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        predictor.model.load_state_dict(checkpoint["model"], strict=True)
        predictor.model.to(device).eval()
        return predictor

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        normalized = normalize_rgb_image(image)
        padded, (height, width) = pad_rgb_image_to_square(normalized)
        self.predictor.set_image(padded)
        bbox = np.array([0, 0, width, height], dtype=np.float32)
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox,
            multimask_output=False,
            return_logits=False,
        )
        mask = np.asarray(masks[0], dtype=np.float32)
        mask = mask[:height, :width]
        return (mask > 0).astype(np.uint8)


def resize_mask_to_output_spacing(
    *,
    mask: np.ndarray,
    level0_width: int,
    level0_height: int,
    level0_spacing_um: float,
    output_spacing_um: float,
    output_class: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    output_width = max(
        1,
        int(round(float(level0_width) * float(level0_spacing_um) / float(output_spacing_um))),
    )
    output_height = max(
        1,
        int(round(float(level0_height) * float(level0_spacing_um) / float(output_spacing_um))),
    )
    resized = cv2.resize(mask.astype(np.uint8), (output_width, output_height), interpolation=cv2.INTER_NEAREST)
    resized = np.where(resized > 0, int(output_class), 0).astype(np.uint8)
    info = {
        "output_width": int(output_width),
        "output_height": int(output_height),
        "output_spacing_um": float(output_spacing_um),
        "output_class": int(output_class),
    }
    return resized, info


def get_mask_native_spacing_um(sam2_info: dict[str, Any]) -> float:
    if bool(sam2_info["resized_to_requested_spacing"]):
        return float(sam2_info["requested_sam2_spacing_um"])
    return float(sam2_info["effective_read_spacing_um"])


def prepare_native_output_mask(
    *,
    mask: np.ndarray,
    mask_spacing_um: float,
    output_class: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    labeled = np.where(mask > 0, int(output_class), 0).astype(np.uint8)
    info = {
        "output_width": int(labeled.shape[1]),
        "output_height": int(labeled.shape[0]),
        "output_spacing_um": float(mask_spacing_um),
        "output_class": int(output_class),
    }
    return labeled, info


def write_mask_pyramid(
    *,
    mask: np.ndarray,
    output_path: Path,
    spacing_um: float,
    tile_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = ImageWriter(
        image_path=str(output_path),
        shape=mask.shape,
        spacing=float(spacing_um),
        dtype=np.uint8,
        coding="monochrome",
        compression="lzw",
        interpolation="nearest",
        tile_size=int(tile_size),
        jpeg_quality=None,
        empty_value=0,
        skip_empty=None,
        cache_path=None,
    )
    try:
        for row in range(0, mask.shape[0], tile_size):
            for col in range(0, mask.shape[1], tile_size):
                tile = mask[row : row + tile_size, col : col + tile_size]
                writer.write(tile=tile, row=row, col=col)
    finally:
        writer.close()


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.output_mask.exists() and args.overwrite:
        args.output_mask.unlink()

    print(f"Input WSI: {args.input_wsi}")
    print(f"Output mask: {args.output_mask}")
    print(f"Backend: {args.backend}")
    print(f"SAM2 input spacing: {args.sam2_input_spacing:.4f} um/px")
    print(f"Output spacing: {args.output_spacing:.4f} um/px")
    print(f"Output class: {args.output_class}")
    print(f"SAM2 device: {args.sam2_device}")

    wsi = open_wsi(args.input_wsi, args.backend)
    try:
        sam2_image, sam2_info = prepare_sam2_input(
            wsi=wsi,
            requested_spacing_um=args.sam2_input_spacing,
            tolerance=args.sam2_tolerance,
            spacing_at_level_0=args.spacing_at_level_0,
        )
    finally:
        # WholeSlideData readers do not require explicit close here.
        pass

    print(
        "Prepared SAM2 input: "
        f"level={sam2_info['level']}, "
        f"effective_spacing={sam2_info['effective_read_spacing_um']:.4f} um/px, "
        f"size={sam2_info['sam2_width']}x{sam2_info['sam2_height']}, "
        f"resized={sam2_info['resized_to_requested_spacing']}"
    )

    predictor = Sam2Predictor(
        checkpoint_path=args.sam2_checkpoint_path,
        config_path=args.sam2_config_path,
        device=args.sam2_device,
    )
    thumbnail_mask = predictor.predict_mask(sam2_image)
    print(
        "SAM2 output mask: "
        f"shape={thumbnail_mask.shape[1]}x{thumbnail_mask.shape[0]}, "
        f"foreground_pixels={int(thumbnail_mask.sum())}"
    )

    if args.keep_native_output_spacing:
        native_mask_spacing_um = get_mask_native_spacing_um(sam2_info)
        output_mask, output_info = prepare_native_output_mask(
            mask=thumbnail_mask,
            mask_spacing_um=native_mask_spacing_um,
            output_class=args.output_class,
        )
        print(
            "Keeping native SAM2 mask grid: "
            f"size={output_info['output_width']}x{output_info['output_height']}, "
            f"spacing={output_info['output_spacing_um']:.4f} um/px"
        )
    else:
        output_mask, output_info = resize_mask_to_output_spacing(
            mask=thumbnail_mask,
            level0_width=int(sam2_info["level0_width"]),
            level0_height=int(sam2_info["level0_height"]),
            level0_spacing_um=float(sam2_info["level0_spacing_um"]),
            output_spacing_um=args.output_spacing,
            output_class=args.output_class,
        )
        print(
            "Resized output mask: "
            f"size={output_info['output_width']}x{output_info['output_height']}, "
            f"spacing={output_info['output_spacing_um']:.4f} um/px"
        )

    write_mask_pyramid(
        mask=output_mask,
        output_path=args.output_mask,
        spacing_um=float(output_info["output_spacing_um"]),
        tile_size=args.tile_size,
    )
    print(f"Wrote multiresolution mask TIFF: {args.output_mask}")


if __name__ == "__main__":
    main()
