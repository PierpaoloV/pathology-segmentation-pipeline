#!/usr/bin/env python3
"""
Batch tissue/background segmentation with Atlas SAM2.

This script scans an input folder for WSIs, optionally stages each slide into a
local scratch directory one at a time, runs Atlas SAM2, and writes one mask TIFF
per slide to the output directory.
"""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from pathlib import Path

from atlas_tb_mask import (
    Sam2Predictor,
    get_mask_native_spacing_um,
    open_wsi,
    prepare_native_output_mask,
    prepare_sam2_input,
    resize_mask_to_output_spacing,
    validate_args as validate_single_args,
    write_mask_pyramid,
)


SUPPORTED_SUFFIXES = {".mrxs", ".ndpi", ".svs", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process all supported WSI files in a folder with Atlas SAM2 and "
            "write {slide_id}.tif masks to the output directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        "--input_wsi_path",
        dest="input_dir",
        required=True,
        type=Path,
        help="Folder containing input WSIs, or a single WSI file.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_wsi_path",
        dest="output_wsi_path",
        required=True,
        type=str,
        help=(
            "Output path template. Use {image} to match the slide id, or pass a directory path "
            "to write {image}.tif inside it."
        ),
    )
    parser.add_argument(
        "--input_filter",
        type=str,
        default=None,
        help="Optional regular expression used to select input files when the input path is a directory.",
    )
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
        default=8.0,
        help="Effective spacing used to build the RGB image passed to SAM2.",
    )
    parser.add_argument(
        "--sam2-tolerance",
        type=float,
        default=0.05,
        help="Tolerance used when selecting a native level near --sam2-input-spacing.",
    )
    parser.add_argument(
        "--output-spacing",
        "--write_spacing",
        dest="output_spacing",
        type=float,
        default=8.0,
        help="Spacing of the written mask level 0 in um/px when resampling the SAM2 mask.",
    )
    parser.add_argument(
        "--output-class",
        type=int,
        default=1,
        help="Foreground label written into the output mask.",
    )
    parser.add_argument(
        "--keep-native-output-spacing",
        action="store_true",
        help="Keep the SAM2 mask on its native grid instead of resizing it to --output-spacing.",
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
        help="Optional local SAM2 checkpoint path.",
    )
    parser.add_argument(
        "--sam2-config-path",
        type=Path,
        default=None,
        help="Optional local SAM2 config path.",
    )
    parser.add_argument(
        "--tile-size",
        "--tile_size",
        dest="tile_size",
        type=int,
        default=512,
        help="Tile size used when writing the multiresolution TIFF.",
    )
    parser.add_argument(
        "--scratch-dir",
        "--cache_path",
        dest="scratch_dir",
        type=Path,
        default=Path("/tmp/atlas_tb_batch"),
        help="Local directory used to stage one slide at a time before processing.",
    )
    parser.add_argument(
        "--no-local-copy",
        action="store_true",
        help="Read slides directly from the input path instead of staging them into --cache_path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output masks.",
    )
    return parser.parse_args()


def validate_batch_args(args: argparse.Namespace) -> None:
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input path not found: {args.input_dir}")
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


def iter_input_slides(input_path: Path, input_filter: str | None) -> list[Path]:
    if input_path.is_file():
        slides = [input_path]
    else:
        pattern = re.compile(input_filter) if input_filter else None
        slides = [
            path
            for path in sorted(input_path.iterdir())
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
            and (pattern is None or pattern.search(path.name))
        ]
    if not slides:
        pretty = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise FileNotFoundError(
            f"No supported slides found in {input_path}. Expected one of: {pretty}"
        )
    return slides


def remove_if_exists(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def copy_slide_to_scratch(slide_path: Path, scratch_root: Path) -> Path:
    slide_id = slide_path.stem
    slide_scratch_dir = scratch_root / slide_id
    remove_if_exists(slide_scratch_dir)
    slide_scratch_dir.mkdir(parents=True, exist_ok=True)

    staged_slide = slide_scratch_dir / slide_path.name
    shutil.copy2(slide_path, staged_slide)

    if slide_path.suffix.lower() == ".mrxs":
        sibling_dir = slide_path.with_suffix("")
        if not sibling_dir.is_dir():
            raise FileNotFoundError(f"Sibling MRXS directory not found: {sibling_dir}")
        shutil.copytree(sibling_dir, slide_scratch_dir / sibling_dir.name)

    return staged_slide


def build_output_path(output_wsi_path: str, slide_path: Path) -> Path:
    image_key = slide_path.stem
    if "{image}" in output_wsi_path:
        return Path(output_wsi_path.format(image=image_key))
    return Path(output_wsi_path) / f"{image_key}.tif"


def process_one_slide(
    *,
    slide_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    predictor: Sam2Predictor,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    single_args = argparse.Namespace(
        input_wsi=slide_path,
        output_mask=output_path,
        sam2_input_spacing=args.sam2_input_spacing,
        output_spacing=args.output_spacing,
        output_class=args.output_class,
        sam2_tolerance=args.sam2_tolerance,
        spacing_at_level_0=args.spacing_at_level_0,
        tile_size=args.tile_size,
        overwrite=args.overwrite,
    )
    validate_single_args(single_args)
    if output_path.exists() and args.overwrite:
        output_path.unlink()

    print(f"Input WSI: {slide_path}")
    print(f"Output mask: {output_path}")

    wsi = open_wsi(slide_path, args.backend)
    sam2_image, sam2_info = prepare_sam2_input(
        wsi=wsi,
        requested_spacing_um=args.sam2_input_spacing,
        tolerance=args.sam2_tolerance,
        spacing_at_level_0=args.spacing_at_level_0,
    )
    print(
        "Prepared SAM2 input: "
        f"level={sam2_info['level']}, "
        f"effective_spacing={sam2_info['effective_read_spacing_um']:.4f} um/px, "
        f"size={sam2_info['sam2_width']}x{sam2_info['sam2_height']}, "
        f"resized={sam2_info['resized_to_requested_spacing']}"
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
        output_path=output_path,
        spacing_um=float(output_info["output_spacing_um"]),
        tile_size=args.tile_size,
    )
    print(f"Wrote multiresolution mask TIFF: {output_path}")


def main() -> None:
    args = parse_args()
    validate_batch_args(args)

    slides = iter_input_slides(args.input_dir, args.input_filter)
    output_root = (
        Path(args.output_wsi_path)
        if "{image}" not in args.output_wsi_path
        else Path(args.output_wsi_path).parent
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.no_local_copy:
        args.scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(slides)} slide(s) in {args.input_dir}")
    print(f"Output path: {args.output_wsi_path}")
    print(f"Local staging: {'disabled' if args.no_local_copy else args.scratch_dir}")
    print(f"SAM2 input spacing: {args.sam2_input_spacing:.4f} um/px")
    print(f"Output spacing: {args.output_spacing:.4f} um/px")
    print(f"Output class: {args.output_class}")
    print(f"SAM2 device: {args.sam2_device}")

    predictor = Sam2Predictor(
        checkpoint_path=args.sam2_checkpoint_path,
        config_path=args.sam2_config_path,
        device=args.sam2_device,
    )

    failures: list[tuple[Path, str]] = []

    for index, source_slide in enumerate(slides, start=1):
        output_path = build_output_path(args.output_wsi_path, source_slide)
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(slides)}] Skipping {source_slide.name}: output exists at {output_path}")
            continue

        print(f"[{index}/{len(slides)}] Processing {source_slide.name}")

        staged_slide: Path | None = None
        scratch_context = None
        try:
            if args.no_local_copy:
                slide_to_process = source_slide
            else:
                scratch_context = tempfile.TemporaryDirectory(
                    dir=str(args.scratch_dir),
                    prefix=f"{source_slide.stem}_",
                )
                staged_slide = copy_slide_to_scratch(source_slide, Path(scratch_context.name))
                slide_to_process = staged_slide
                print(f"Staged locally at {slide_to_process}")

            process_one_slide(
                slide_path=slide_to_process,
                output_path=output_path,
                args=args,
                predictor=predictor,
            )
        except Exception as exc:
            failures.append((source_slide, str(exc)))
            print(f"Failed on {source_slide}: {exc}")
        finally:
            if scratch_context is not None:
                scratch_context.cleanup()
                if staged_slide is not None:
                    print(f"Removed local staging for {source_slide.name}")

    print(
        f"Done. Successful: {len(slides) - len(failures)}, "
        f"Failed: {len(failures)}, Total: {len(slides)}"
    )
    if failures:
        for slide_path, message in failures:
            print(f"FAIL {slide_path.name}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
