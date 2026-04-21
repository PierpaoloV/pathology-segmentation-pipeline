# Implementing AtlasPatch SAM2 Tissue Segmentation

Date: 2026-04-21

## Goal

Evaluate how difficult it would be to replace the current tissue/background (TB) segmentation stage with AtlasPatch's SAM2-based tissue segmentation, and define a concrete implementation plan for this repository.

## Decisions Made

- Use case is research only, not commercial.
- AtlasPatch licensing is therefore acceptable for the intended use.
- `mask_class` should be treated as configurable. We do not need to preserve `mask_class=2`.
- Mask spacing should also be treated as configurable. Coarser mask spacings such as `8.0` um or `16.0` um are acceptable if they speed up the pipeline without harming downstream quality too much.
- The goal is no longer just to evaluate AtlasPatch. The working assumption is that we want to adopt it because it is expected to outperform the current TB model.

## Current Repo State

- The current pipeline runs TB segmentation first in `code/start_characterization.sh`.
- That stage writes a mask to `/home/user/process/tb/${filename%.*}_tissue.tif`.
- The current TB model path is `/home/user/source/models/tb/playground_soft-cloud-137_best_model.pt`.
- The second stage reuses that TB mask as `--mask_wsi_path` with `--mask_spacing=4.0` and `--mask_class=2`.
- The local fast inference pipeline expects a WSI-like mask file, not just an in-memory array or PNG.
- Local mask reading happens through `pathology-fast-inference/fastinference/async_wsi_reader.py`, which loads the mask as a multi-resolution image and selects a level by spacing.
- Local mask writing happens through ASAP / `multiresolutionimageinterface`, wrapped by `pathology-common/digitalpathology/image/io/imagewriter.py`.
- Hard classification in the current stack uses `argmax + 1`, so a binary model naturally produces labels `1` and `2`. This explains why downstream tissue selection uses `mask_class=2`.

## AtlasPatch State

- AtlasPatch uses SAM2 for tissue segmentation.
- AtlasPatch's segmentation service prepares a whole-slide thumbnail at `1.25x` objective power.
- That thumbnail is resized to at most `1024 x 1024`.
- SAM2 is then run with a full-frame box prompt.
- The returned mask is a float32 binary mask aligned to the thumbnail dimensions.
- If no checkpoint path is supplied, AtlasPatch downloads the checkpoint from Hugging Face repo `AtlasAnalyticsLab/AtlasPatch`.
- AtlasPatch's `detect-tissue` CLI writes visualization outputs under `output/visualization/`.
- AtlasPatch does not appear to emit a multiresolution tissue-mask TIFF as a first-class output artifact.
- AtlasPatch's patch extraction pipeline uses the thumbnail mask internally by turning it into contours, scaling those contours to level 0, and then generating patch coordinates.

## AtlasPatch Input And Output

### Input

- Input is a WSI path.
- AtlasPatch loads the slide through its WSI abstraction.
- It requires usable slide magnification metadata to generate a `1.25x` thumbnail in a power-aware way.

### Output

- The core segmentation service returns `Mask(data=<float32 array>, source_shape=(H, W))`.
- `Mask.data` is aligned to the thumbnail, not to a native WSI pyramid level.
- The mask is effectively binary tissue vs background.
- The public `detect-tissue` command writes overlays for inspection, not a reusable TB mask WSI for downstream segmentation.

## Focus Areas

This section narrows the work to the three interfaces that matter most for implementation:

1. Producing the right input to AtlasPatch SAM2.
2. Capturing and interpreting the output mask correctly.
3. Writing that mask as a multiresolution TIFF at a chosen spacing, starting with `8.0` um.

## 1. Producing The Right Input To Atlas

The real input to the SAM2 model is not the raw WSI. It is a preprocessed thumbnail.

AtlasPatch's effective input contract is:

- Open a WSI with valid size metadata.
- Determine the base magnification.
- Generate a whole-slide thumbnail at `1.25x` objective power.
- Resize that thumbnail to fit within `1024 x 1024`.
- Convert to RGB uint8.
- Run SAM2 on that thumbnail with a full-image bounding box.

For this repository, that means the adapter should do the following:

1. Open the WSI.
2. Read or infer the base spacing / magnification.
3. Create a whole-slide thumbnail corresponding to `1.25x`.
4. Resize that thumbnail to a maximum side length of `1024`.
5. Feed that RGB thumbnail into SAM2.

Important implementation note:

- We should treat thumbnail generation as part of our adapter, not as an implementation detail hidden elsewhere.
- If a slide is missing reliable magnification metadata, AtlasPatch's preferred power-based thumbnailing can become fragile.
- In that case, we may need an explicit fallback path based on MPP or on a chosen pyramid level.

### Minimal Input Specification For Our Adapter

- Input file: one WSI path.
- Expected slide metadata:
- Level-0 dimensions.
- Level downsampling factors.
- Level-0 pixel spacing or enough metadata to infer it.
- Base magnification if available.
- Generated model input:
- RGB thumbnail.
- Shape bounded by `1024 x 1024`.
- Represents approximately `1.25x` objective power over the whole slide.

### Double-Check From `hs2p` With ASAP Backend

The current `hs2p` code path does not pass a raw WSI handle directly into SAM2. It passes an RGB NumPy array prepared from the ASAP-backed reader.

What happens in `hs2p` when the backend is `asap`:

1. `open_slide(..., backend="asap")` creates an `ASAPReader`.
2. `ASAPReader` wraps `wholeslidedata.WholeSlideImage(path, backend="asap")`.
3. For SAM2 tissue detection, `prepare_sam2_thumbnail()` chooses a pyramid level near a target spacing of `8.0` um by default.
4. It then reads the entire image at that level with:
   - `slide.read_region((0, 0), level, seg_size)`
5. On the ASAP backend, that becomes a WholeSlideData patch read:
   - `self._wsi.get_patch(x, y, width, height, spacing=self._spacings[level], center=False)`
6. The returned region is converted to a NumPy RGB array.
7. If the chosen level spacing is not close enough to the target spacing, `hs2p` resizes that full-level RGB array to the exact target dimensions.
8. That RGB array is then passed into `segment_tissue_image(...)`.
9. For method `sam2`, `segment_tissue_image(...)` calls `_segment_sam2(image, ...)`.
10. The SAM2 helper pads non-square inputs to a square canvas with white background, then runs SAM2 with a full-frame bounding box matching the original unpadded width and height.

So the effective SAM2 input in `hs2p` with ASAP backend is:

- a full-slide RGB NumPy image
- read from one ASAP pyramid level through WholeSlideData
- optionally resized to match the requested SAM2 thumbnail spacing
- then letterboxed to a square only inside the SAM2 predictor wrapper

This is useful because it means we do not need to mimic AtlasPatch's exact `1.25x -> 1024x1024` thumbnail logic if we want to follow the `hs2p` integration style instead. `hs2p` already proves a simpler contract can work:

- choose a coarse physical spacing
- read the full slide at that spacing through the backend
- pass the resulting RGB array to SAM2

## 2. Having The Output

The useful output from AtlasPatch is not the visualization image. It is the in-memory segmentation mask returned by the segmentation service.

That output has these properties:

- It is a NumPy array.
- It is aligned to the thumbnail, not level 0.
- It is binary tissue vs background.
- It is stored as float32 in AtlasPatch's service layer.

For our purposes, the adapter should standardize the output immediately after inference:

1. Convert the SAM2 output into a binary mask.
2. Make the representation explicit, for example `uint8` with values `{0, 1}`.
3. Record the thumbnail width and height used for inference.
4. Record the original slide width and height at level 0.
5. Preserve enough geometry to rescale the mask into a downstream mask grid later.

### Recommended Internal Output Object

The adapter should probably return or carry:

- `mask_thumbnail`: binary thumbnail mask, shape `(H_thumb, W_thumb)`, uint8 values `{0,1}`.
- `thumb_shape`: `(H_thumb, W_thumb)`.
- `slide_level0_shape`: `(H0, W0)`.
- `slide_level0_spacing`: spacing at level 0 in micrometers.
- `target_mask_spacing`: chosen output spacing, initially `8.0`.

This is enough information to write a downstream-compatible mask file.

## 3. Producing A Multiresolution TIFF Mask At `8.0` um

This is the key conversion step.

The current local stack already has the required pieces:

- `ImageWriter` can create multiresolution images and store spacing metadata.
- `ImageReader` resolves mask levels by spacing.
- Downstream inference only needs a readable mask image plus the correct `mask_spacing` and `mask_class` parameters.

### Target Artifact

The new Atlas-based TB artifact should be:

- A TIFF written through the local ASAP-backed multiresolution writer.
- Monochrome or indexed single-channel mask.
- Binary values `{0, 1}`.
- Base spacing set to the chosen mask spacing, initially `8.0` um.
- Readable by `pathology-fast-inference/fastinference/async_wsi_reader.py`.

### Geometry For The `8.0` um Mask

Let:

- `W0, H0` be the WSI dimensions at level 0.
- `S0` be the WSI spacing at level 0 in um/pixel.
- `S_mask = 8.0` um/pixel.

Then the mask dimensions should be approximately:

- `W_mask = round(W0 * S0 / S_mask)`
- `H_mask = round(H0 * S0 / S_mask)`

This gives the shape of the base mask level to write.

Then:

1. Resize the thumbnail mask from `(H_thumb, W_thumb)` to `(H_mask, W_mask)`.
2. Use nearest-neighbor interpolation only.
3. Write the resized mask as a single-channel uint8 image with spacing `8.0`.
4. Close the writer and let ASAP build the pyramid.

### Why Nearest-Neighbor Matters

- This is a label mask, not an intensity image.
- Linear interpolation would create ambiguous intermediate values.
- Downstream mask reading uses equality against `mask_class`, so clean binary values are important.

### Suggested Writing Contract

- Output dtype: `uint8`
- Output values: `0=background`, `1=tissue`
- Output coding: `monochrome`
- Compression: `lzw`
- Interpolation for pyramid generation: `nearest`
- Base spacing: `8.0`
- Tile size: keep the existing default such as `512`

### Downstream Contract After Writing

Once written, the epithelium stage should be able to use:

- `--mask_wsi_path=<atlas mask path>`
- `--mask_spacing=8.0`
- `--mask_class=1`

This is much simpler than preserving the previous label-2 convention.

## Practical First Prototype

The first useful implementation should do exactly this and nothing more:

1. Open one WSI.
2. Build the Atlas-compatible thumbnail.
3. Run SAM2.
4. Convert the result to binary `{0,1}`.
5. Resize it onto an `8.0` um grid.
6. Write a multiresolution TIFF mask.
7. Verify that local `ImageReader` can reopen it and resolve a level at `8.0` um.
8. Feed it into the existing epithelium stage with `mask_class=1`.

If this works, then we have validated the critical path.

## Feasibility Assessment

Overall difficulty: medium.

Why it is feasible:

- The unified Docker image already includes `sam2`, `openslide-python`, `opencv`, `torch`, and `huggingface_hub`.
- This repo already has a multi-resolution image writer we can reuse to generate a proper mask artifact.
- AtlasPatch's model is thumbnail-based, so runtime cost for TB segmentation should be manageable.
- Allowing configurable `mask_class` removes one of the main semantic compatibility constraints.
- Allowing configurable mask spacing removes the need to target `4.0` um exactly.

Why it is not a drop-in replacement:

- AtlasPatch outputs a thumbnail-aligned binary mask, while this pipeline expects a mask file on disk that can be sampled by spacing during downstream WSI inference.
- AtlasPatch's public CLI is oriented around overlays and patch coordinates, not a multiresolution TIFF mask artifact.

## Main Challenges

- Output mismatch: thumbnail mask vs WSI-like mask file.
- Resolution mismatch: AtlasPatch mask is not naturally stored at the mask spacing we choose for downstream inference.
- Pyramid mismatch: AtlasPatch does not directly provide a multiresolution mask pyramid.
- Metadata risk: power-based thumbnail generation depends on valid WSI magnification / MPP metadata.
- Validation risk: TB replacement can change which tiles are passed into the epithelium stage, so accuracy must be checked end-to-end.

Challenges that are now less important than initially assumed:

- We do not need to preserve label `2` specifically if we update downstream `mask_class`.
- We do not need to preserve `4.0` um spacing specifically if a coarser grid works well enough.

## Recommended Integration Strategy

Prefer a thin local adapter instead of pulling in the whole AtlasPatch processing pipeline.

Recommended shape of the adapter:

- Read one WSI path.
- Follow the `hs2p` ASAP pattern rather than AtlasPatch's native `1.25x` thumbnail path.
- Read the whole slide at a coarse physical spacing, starting with `8.0` um.
- If no native level matches closely enough, resize the full-slide RGB array to the exact requested spacing.
- Load the AtlasPatch SAM2 checkpoint from Hugging Face.
- Pad to square only inside the SAM2 wrapper, as `hs2p` does.
- Run SAM2 with a full-image box prompt.
- Resize the binary result from SAM2-input space into a target mask grid at a configurable spacing, with `8.0` um or `16.0` um as strong candidates for a first prototype.
- Keep mask values binary and set downstream `mask_class` accordingly, preferably `1`.
- Write the result as a multi-resolution TIFF using the existing ASAP-backed `ImageWriter`.

This would preserve the current downstream contract while minimizing invasive changes.

## Multiresolution Mask Plan

If AtlasPatch does not produce a multiresolution mask directly, the plan should be:

1. Choose the canonical output grid.
2. Start with a configurable mask spacing.
3. Prefer trying `8.0` um first, and optionally `16.0` um for a speed-oriented benchmark.
4. Compute output width and height from slide level-0 dimensions and level-0 spacing.
5. Resize the thumbnail mask onto that target grid with nearest-neighbor interpolation.
6. Keep the mask binary, ideally `0=background`, `1=tissue`.
7. Write the mask through `pathology-common/digitalpathology/image/io/imagewriter.py` with nearest interpolation and LZW compression.
8. Let the writer build the pyramid so downstream readers can resolve levels by spacing.
9. Verify that `async_wsi_reader.py` can load the produced mask and that epithelium inference works with `mask_class=1`.

## Proposed Implementation Steps

1. Create a small local script for Atlas-based TB inference, for example `code/atlas_tb_mask.py`.
2. Reuse only the minimal AtlasPatch logic needed for SAM2 inference, while following `hs2p`'s ASAP-based input preparation.
3. Add checkpoint download support, ideally parallel to the current `download_models.py` flow.
4. Produce a configurable mask grid from the SAM2 thumbnail mask, starting with `8.0` um and optionally `16.0` um.
5. Write a multiresolution TIFF mask with binary values and update downstream calls to use the chosen `mask_class`.
6. Update `code/start_characterization.sh` so Atlas can be selected and so `mask_spacing` and `mask_class` are configurable.
7. Keep the old backend available during validation, but treat Atlas as the intended replacement path.
8. Run a small slide comparison study.
9. Compare TB masks visually.
10. Compare epithelium outputs when driven by the old vs Atlas-derived TB mask.
11. Measure runtime and GPU memory at `8.0` um and, if useful, `16.0` um.
12. Choose the best mask spacing and switch the pipeline defaults once validated.

## Current Implementation Status

- Added `code/atlas_tb_mask.py`.
- The script follows the `hs2p` SAM2 input path with the ASAP backend:
- open the WSI through WholeSlideData
- resolve a level near the requested spacing, default `8.0` um
- read the whole slide RGB image at that coarse level
- resize to the exact requested spacing when needed
- pad to square only inside the SAM2 wrapper
- run SAM2 with a full-frame box prompt
- resize the binary output mask onto a configurable output grid, default `8.0` um
- write the result as a multiresolution TIFF with the local ASAP-backed `ImageWriter`
- The script also supports:
- explicit override of level-0 spacing when slide metadata is incomplete
- configurable output class, default `1`
- configurable output spacing
- optional local SAM2 checkpoint/config overrides
- overwrite mode for replacing an existing output TIFF
- Current validation state:
- syntax checked with `python3 -m py_compile code/atlas_tb_mask.py`
- not yet executed end-to-end on a real slide in this repository
- not yet wired into `code/start_characterization.sh`

## To-Do List

- [x] Confirm how the current pipeline consumes the TB mask.
- [x] Confirm AtlasPatch model input and output shape at a code level.
- [x] Confirm AtlasPatch does not directly provide the same mask artifact expected here.
- [x] Confirm that a multiresolution mask can likely be written with the existing local writer.
- [x] Decide whether Atlas integration must support commercial use.
- [x] Decide whether to vendor minimal Atlas logic or install `atlas-patch` directly.
- [x] Implement a standalone Atlas TB prototype script following the `hs2p` ASAP path.
- [ ] Run standalone Atlas TB inference on one WSI.
- [ ] Validate conversion from SAM2 input mask to `8.0` um mask grid on a real slide.
- [ ] Benchmark whether `16.0` um is still acceptable downstream.
- [ ] Validate the written multiresolution TIFF with local readers on a real slide.
- [ ] Verify downstream epithelium stage works with `mask_class=1`.
- [ ] Add backend selection to the pipeline.
- [ ] Benchmark runtime against the current TB model.
- [ ] Validate output quality on a representative slide set.

## Notes For Implementation

- The simplest robust compatibility target is not "make AtlasPatch look native", but "make AtlasPatch produce the same TB artifact contract this pipeline already expects".
- A single-resolution TIFF at the correct spacing might already be enough for some cases, but writing a true multiresolution pyramid is safer and more consistent with the current stack.
- Your latest constraints simplify the integration substantially: binary mask output is acceptable, and both `mask_class` and mask spacing can be configured downstream.
- The most likely high-value prototype is: Atlas SAM2 thumbnail inference, nearest-neighbor resize to `8.0` um grid, binary mask output with `mask_class=1`, and ASAP pyramid write.

## Sources Consulted

- AtlasPatch GitHub repo: https://github.com/AtlasAnalyticsLab/AtlasPatch
- AtlasPatch model card: https://huggingface.co/AtlasAnalyticsLab/AtlasPatch
- Local pipeline files: `code/start_characterization.sh`, `pathology-fast-inference/fastinference/async_wsi_reader.py`, `pathology-common/digitalpathology/image/io/imagewriter.py`, `pathology-fast-inference/fastinference/processors/torch_processor.py`, `pathology-common/digitalpathology/image/classification/imageclassifier.py`
