# pathology-common / digitalpathology

Core Python library for processing Whole Slide Images (WSI) in computational pathology pipelines. Covers the full lifecycle from raw slide ingestion through annotation conversion, patch extraction, data augmentation, and network training.

---

## Table of Contents

- [Overview](#overview)
- [Package Structure](#package-structure)
- [Data Flow](#data-flow)
- [Modules](#modules)
  - [image/io — WSI I/O](#imageio--wsi-io)
  - [image/processing — Image Manipulation](#imageprocessing--image-manipulation)
  - [generator — Patch & Batch Extraction](#generator--patch--batch-extraction)
  - [adapters — Data Transformation](#adapters--data-transformation)
  - [utils — Utilities](#utils--utilities)
  - [errors — Error Hierarchy](#errors--error-hierarchy)
- [Scripts](#scripts)
- [Key Data Types](#key-data-types)

---

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        digitalpathology                         │
│                                                                 │
│   WSI + Annotations                                             │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────┐    ┌────────────┐    ┌───────────┐               │
│   │ image/  │───▶│ generator/ │───▶│ adapters/ │               │
│   │  io     │    │            │    │           │               │
│   │ reading │    │ patch &    │    │ augment,  │               │
│   │ writing │    │ batch      │    │ normalize,│               │
│   │ annots  │    │ sampling   │    │ label map │               │
│   └─────────┘    └────────────┘    └───────────┘               │
│        │                                  │                     │
│        ▼                                  ▼                     │
│   ┌──────────────┐              ┌──────────────────┐            │
│   │ image/       │              │  Training Batches │            │
│   │ processing   │              │  (patches, labels,│            │
│   │ convert,     │              │   weights)        │            │
│   │ threshold,   │              └──────────────────┘            │
│   │ arithmetic   │                                              │
│   └──────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Package Structure

```
pathology-common/
│
├── digitalpathology/               # Main package (v1.0.0)
│   │
│   ├── image/
│   │   ├── io/                     # Multi-resolution WSI I/O
│   │   │   ├── imagereader.py      # Read WSI tiles at any spacing
│   │   │   ├── imagewriter.py      # Write multi-res image pyramids
│   │   │   ├── plainimage.py       # Single-level PNG/JPG I/O
│   │   │   └── annotation.py       # ASAP XML annotation parser
│   │   │
│   │   └── processing/             # Image manipulation
│   │       ├── conversion.py       # Annotation→mask, mask→annotation
│   │       ├── arithmetic.py       # Mask arithmetic (+, -, *)
│   │       ├── threshold.py        # Binary thresholding
│   │       ├── comparison.py       # Image similarity metrics
│   │       ├── dilate.py           # Morphological dilation
│   │       ├── zoom.py             # Resize / resample
│   │       ├── regions.py          # Connected component filtering
│   │       ├── stainnormalization.py # H&E stain normalization
│   │       └── inference.py        # Apply networks to WSI
│   │
│   ├── generator/
│   │   ├── patch/
│   │   │   ├── patchsource.py      # Image-mask-stat descriptor
│   │   │   └── patchsampler.py     # Extract patches from a single WSI
│   │   │
│   │   ├── batch/
│   │   │   ├── batchsource.py      # Organise sources by category/purpose
│   │   │   ├── batchsampler.py     # Multi-process batch extraction
│   │   │   └── batchsamplerdaemon.py # IPC daemon for batchsampler
│   │   │
│   │   ├── mask/
│   │   │   └── maskstats.py        # Mask label statistics & caching
│   │   │
│   │   └── tf_data_generator.py    # Keras/TF Dataset integration
│   │
│   ├── adapters/
│   │   ├── batchadapter.py         # Full transformation pipeline
│   │   ├── augmenters/             # Spatial, colour, noise augmenters
│   │   ├── range/                  # Pixel range normalizers
│   │   ├── label/                  # Label value remapping
│   │   └── weight/                 # Per-pixel loss weighting
│   │
│   ├── utils/
│   │   ├── imagefile.py            # WSI-aware file copy/move/delete
│   │   ├── dataconfig.py           # BatchSource builder from directories
│   │   ├── foldercontent.py        # Glob/recursive directory listing
│   │   ├── loggers.py              # Console/file logger setup
│   │   ├── population.py           # Train/val/test split distribution
│   │   └── serialize.py            # Object serialization helpers
│   │
│   └── errors/                     # Exception hierarchy
│       ├── imageerrors.py
│       ├── processingerrors.py
│       ├── labelerrors.py
│       ├── configerrors.py
│       └── ...
│
└── scripts/                        # 19 command-line tools
```

---

## Data Flow

### Annotation → Mask → Training Batch

```
  ┌──────────────┐     ┌──────────────┐
  │  WSI (.mrxs  │     │  ASAP XML    │
  │   .svs .tif) │     │ annotations  │
  └──────┬───────┘     └──────┬───────┘
         │                    │
         │         annotation.py (parse polygons)
         │                    │
         └────────┬───────────┘
                  │
         conversion.py (rasterize at target spacing)
                  │
                  ▼
         ┌────────────────┐
         │  Mask (.tif)   │  ← integer label per pixel
         │  label 0 = bg  │
         │  label 1 = epi │
         │  label 2 = str │
         └───────┬────────┘
                 │
         maskstats.py (pre-compute label distributions per tile)
                 │
                 ▼
         ┌───────────────┐
         │  .stat file   │  ← serialized tile statistics
         └───────┬───────┘
                 │
         batchsampler.py (stratified sampling, N workers)
                 │
                 ▼
         ┌───────────────────────────────────┐
         │  Raw batch                        │
         │  patches : (N, H, W, C) uint8     │
         │  labels  : (N, H, W)   uint8      │
         └───────────────┬───────────────────┘
                         │
                batchadapter.py
                         │
          ┌──────────────┼──────────────────┐
          │              │                  │
    augmenters/     range/            label/ & weight/
    flip, rotate    normalize         remap & weight
    H&E jitter      uint8→float32     one-hot encode
          │              │                  │
          └──────────────┴──────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │  Training batch                   │
         │  patches : (N, H, W, C) float32   │
         │  labels  : (N, H, W, K) one-hot   │
         │  weights : (N, H, W)   float32    │
         └───────────────────────────────────┘
```

### Inference Pipeline

```
  ┌──────────────┐
  │  WSI slide   │
  └──────┬───────┘
         │  ImageReader (read tiles at inference spacing)
         ▼
  ┌──────────────┐
  │  Tile stream │  ← configurable overlap/stride
  └──────┬───────┘
         │  inference.py / apply_network()
         │  (normalization → forward pass → argmax)
         ▼
  ┌──────────────┐
  │  Output mask │  ← same resolution as input
  │  (multi-res) │     written incrementally
  └──────────────┘
```

---

## Modules

### `image/io` — WSI I/O

#### `ImageReader` — `image/io/imagereader.py`

Multi-resolution WSI reader built on ASAP's `multiresolutionimageinterface`.

```python
reader = ImageReader(
    image_path        = '/data/slide.mrxs',
    spacing_tolerance = 0.25,   # ±25% spacing tolerance
    input_channels    = None,   # None = all channels
    cache_path        = None    # optional copy-to-local cache
)

# Query metadata
reader.spacing        # spacing at level 0 (µm/px)
reader.spacings       # list of spacings for all levels
reader.shape          # (H, W) at level 0
reader.shapes         # list of shapes for all levels

# Find the level closest to a target spacing
level = reader.level(spacing=1.0)

# Extract a patch at (row, col) at a given spacing
patch = reader.patch(
    coordinates = (1024, 2048),
    spacing     = 1.0,
    shape       = (512, 512)
)  # → numpy array (512, 512, C) uint8

reader.close()
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_path` | `str` | Path to WSI (`.mrxs`, `.svs`, `.tif`, …) |
| `spacing_tolerance` | `float` | Fractional tolerance when matching spacings |
| `input_channels` | `list` or `None` | Channel indices to load; `None` = all |
| `cache_path` | `str` or `None` | Directory to copy slide before reading |

---

#### `ImageWriter` — `image/io/imagewriter.py`

Writes multi-resolution image pyramids tile-by-tile.

```python
writer = ImageWriter(
    image_path  = '/output/mask.tif',
    shape       = (40000, 30000),
    spacing     = 0.5,           # µm/px at level 0
    dtype       = 'uint8',
    coding      = 'monochrome',  # or 'rgb', 'rgba', 'indexed'
    compression = 'lzw'
)

writer.write_patch(coordinates=(0, 0), patch=tile_array)
writer.close()   # finalise pyramid & flush to disk
```

---

#### `Annotation` — `image/io/annotation.py`

Parses ASAP XML annotation files and rasterizes polygons to pixel masks.

```python
ann = Annotation(annotation_path='/data/slide.xml')
ann.open()

# Rasterize polygons at a specific spacing
mask = ann.convert(
    image_path       = '/data/slide.mrxs',
    label_map        = {'tumour': 1, 'stroma': 2},
    conversion_order = ['tumour', 'stroma'],
    spacing          = 1.0
)  # → numpy array (H, W) uint8
```

---

### `image/processing` — Image Manipulation

#### `conversion.py`

| Function | Description |
|----------|-------------|
| `create_annotation_mask(image, annotation, label_map, conversion_order, conversion_spacing, spacing_tolerance, output_path, strict, accept_all_empty, work_path, clear_cache, overwrite)` | Rasterize ASAP XML to mask TIFF |
| `create_mask_annotation(mask, annotation, label_map, conversion_spacing, target_spacing, spacing_tolerance, keep_singles, rdp_epsilon, overwrite)` | Vectorize mask back to ASAP XML polygons |
| `calculate_preview(image, mask, preview_path, level, pixel_spacing, spacing_tolerance, alpha, palette, copy_path, clear_cache, overwrite)` | Overlay colourised mask on WSI thumbnail |
| `map_mask_image(mask, output_path, label_map, copy_path, work_path, clear_cache, overwrite)` | Remap integer label values in a mask |
| `save_mrimage_as_image(image, output_path, level, pixel_spacing, spacing_tolerance, multiplier, overwrite)` | Export a single pyramid level as PNG/JPG |

**Label map example:**
```python
label_map = {
    'background' : 0,
    'epithelium' : 1,
    'stroma'     : 2,
    'necrosis'   : 3,
}
```

---

#### `arithmetic.py`

Pixel-wise operations between two mask images.

```python
image_arithmetic(
    left        = '/masks/pred.tif',
    right       = '/masks/gt.tif',
    result_path = '/masks/diff.tif',
    operand     = '-',           # '+', '-', or '*'
    overwrite   = True
)
```

---

#### `threshold.py`

```python
low_threshold_image(
    image         = '/masks/prob.tif',
    output_path   = '/masks/binary.tif',
    low_threshold = 0.5,
    overwrite     = True
)
```

---

#### `zoom.py`

```python
zoom_image(
    image         = '/data/slide.tif',
    output_path   = '/data/slide_4x.tif',
    zoom          = 4,
    pixel_spacing = 1.0,     # source spacing
    overwrite     = True
)
```

---

#### `dilate.py`

```python
dilate_image(
    image             = '/masks/tissue.tif',
    output_path       = '/masks/tissue_dilated.tif',
    dilation_distance = 50,   # µm
    overwrite         = True
)
```

---

### `generator` — Patch & Batch Extraction

#### Architecture

```
  BatchSource                       (organises all data)
      │
      ├── category 0 ──┬── PatchSource(image_0, mask_0, stat_0)
      │                ├── PatchSource(image_1, mask_1, stat_1)
      │                └── PatchSource(image_2, mask_2, stat_2)
      │
      └── category 1 ──┬── PatchSource(image_3, mask_3, stat_3)
                       └── PatchSource(image_4, mask_4, stat_4)

  BatchSampler
      ├── Worker 0 → PatchSampler → ImageReader + MaskStats → patches
      ├── Worker 1 → PatchSampler → ImageReader + MaskStats → patches
      └── Worker 2 → PatchSampler → ImageReader + MaskStats → patches
                                            │
                                            ▼
                                    BatchAdapter → network-ready batch
```

---

#### `PatchSource` — `generator/patch/patchsource.py`

Lightweight descriptor for one image-mask pair.

```python
source = PatchSource(
    image_path       = '/data/slide.mrxs',
    mask_path        = '/masks/slide.tif',
    stat_path        = '/stats/slide.stat',   # optional pre-computed stats
    available_labels = (1, 2, 3)              # labels present in this mask
)

source.image   # → '/data/slide.mrxs'
source.mask    # → '/masks/slide.tif'
source.labels  # → (1, 2, 3)
```

---

#### `MaskStats` — `generator/mask/maskstats.py`

Pre-computes and caches label distribution per tile for stratified sampling.

```python
stats = MaskStats(
    file              = '/masks/slide.tif',   # or pre-computed .stat file
    mask_spacing      = 4.0,                  # µm for stats extraction
    spacing_tolerance = 0.25,
    mask_labels       = (1, 2, 3)
)
# Stores: {tile_index: {label: pixel_count}} for the whole slide
# Enables O(1) lookup during batch assembly
```

---

#### `BatchSampler` — `generator/batch/batchsampler.py`

Multi-process patch extractor. The core of the training data pipeline.

```python
sampler = BatchSampler(
    label_dist    = {1: 0.33, 2: 0.33, 3: 0.34},      # class balance
    patch_shapes  = {0.5: (512, 512), 2.0: (256, 256)}, # multi-scale
    mask_spacing  = 4.0,
    input_channels = None,
    label_mode    = 'central',   # 'central' | 'synthesize' | 'load'
    patch_sources = batch_source,
    data_adapter  = batch_adapter,
    category_dist = {0: 0.5, 1: 0.5},
    process_count = 4            # parallel workers
)

patches, labels, weights = sampler.batch(batch_size=32)
```

**Label modes:**

| Mode | Description |
|------|-------------|
| `'central'` | Label of the centre pixel of each patch |
| `'synthesize'` | Full label map synthesized from mask statistics |
| `'load'` | Pre-computed label map loaded from `.stat` file |

---

#### `BatchSource` — `generator/batch/batchsource.py`

Container that organizes `PatchSource` objects by category and purpose.

```python
source = BatchSource()
source.load('/configs/dataset.yaml')
source.update(path_replacements={'/old/path': '/new/path'})

for item in source.items(purpose_id='train', replace=True):
    print(item.image, item.mask, item.labels)
```

**Dataset YAML structure:**
```yaml
categories:
  0:
    train:
      - image: /data/slide_01.mrxs
        mask:  /masks/slide_01.tif
        labels: [1, 2, 3]
    val:
      - image: /data/slide_02.mrxs
        mask:  /masks/slide_02.tif
        labels: [1, 2]
```

---

### `adapters` — Data Transformation

#### `BatchAdapter` — `adapters/batchadapter.py`

Full transformation pipeline applied to raw patches before they reach the network.

```
Raw patch (uint8)
     │
     ▼  squash_range
float32 [0.0, 1.0]
     │
     ▼  AugmenterPool
augmented patch
     │
     ▼  RangeNormalizer (early)
normalised patch
     │
     ▼  LabelMapper
network labels [0, N-1]
     │
     ▼  one_hot_encode
(H, W, N) float32
     │
     ▼  WeightMapper
(H, W) loss weights
     │
     ▼  RangeNormalizer (late)
final patch
```

```python
adapter = BatchAdapter(
    squash_range     = True,
    augmenter_pool   = pool,
    range_normalizer = normalizer,
    label_mapper     = mapper,
    labels_one_hot   = True,
    weight_mapper    = weight_mapper
)
```

---

#### Augmenters — `adapters/augmenters/`

```
AugmenterPool
    │
    ├── Group 0 (always applied)
    │       ├── FlipAugmenter          horizontal / vertical flip
    │       └── Rotate90Augmenter      0° / 90° / 180° / 270°
    │
    ├── Group 1 (randomly applied)
    │       ├── HedColorAugmenter      H&E channel perturbation
    │       ├── HsbColorAugmenter      HSB value shifts
    │       └── ContrastAugmenter      brightness / contrast
    │
    └── Group 2 (randomly applied)
            ├── ElasticAugmenter            elastic deformation
            ├── GaussianBlurAugmenter       sigma-controlled blur
            └── AdditiveGaussianNoiseAugmenter
```

```python
pool = AugmenterPool()

pool.appendgroup(
    group      = [FlipAugmenter(flip_probability=0.5),
                  Rotate90Augmenter()],
    randomized = False
)
pool.appendgroup(
    group      = [HedColorAugmenter(
                      haematoxylin_sigma_range=(-0.05, 0.05),
                      eosin_sigma_range=(-0.05, 0.05))],
    randomized = True
)

aug_patches, aug_labels = pool.augment(patches, labels)
```

---

#### `LabelMapper` — `adapters/label/labelmapper.py`

Maps arbitrary image label values to contiguous network labels `[0, N-1]`.

```python
mapper = LabelMapper(
    label_map = {
        0: 0,   # background → class 0
        1: 1,   # epithelium → class 1
        2: 2,   # stroma     → class 2
    }
)
mapped = mapper.process(label_patch)  # uint8 → uint8
```

---

#### Range Normalizers — `adapters/range/`

| Class | Input | Output |
|-------|-------|--------|
| `RgbToZeroOneRangeNormalizer` | uint8 `[0, 255]` | float32 `[0.0, 1.0]` |
| `RgbRangeNormalizer` | configurable | configurable |
| `GeneralRangeNormalizer` | any | any |

---

#### Weight Mappers — `adapters/weight/`

Produce a per-pixel weight map used to balance class frequencies in the loss.

```python
weight_map = mapper.process(label_patch)  # → (H, W) float32
```

| Class | Strategy |
|-------|----------|
| `CleanWeightMapper` | Fixed per-class weights |
| `BatchWeightMapper` | Normalise across the whole batch |
| `PatchWeightMapper` | Normalise within each patch |

---

### `utils` — Utilities

#### `imagefile.py`

WSI-aware file operations that correctly handle `.mrxs` format (which stores data in a companion folder alongside the `.mrxs` file).

```python
from digitalpathology.utils import imagefile as dptimagefile

# Copy slide + companion directory
dptimagefile.copy_image(source='/data/slide.mrxs', target='/backup/slide.mrxs')

# Move slide preserving companion directory
dptimagefile.move_image(source='/staging/slide.mrxs', target='/data/slide.mrxs')

# Relocate (copy or move depending on flag)
dptimagefile.relocate_image(source_path=..., target_path=..., move=True, overwrite=False)

# Delete slide + companion directory safely
dptimagefile.remove_image(image_path='/tmp/slide.mrxs')
```

---

#### `dataconfig.py`

Builds a `BatchSource` configuration from image/mask/stat directory expressions.

```python
from digitalpathology.utils import dataconfig as dptdataconfig

batch_source = dptdataconfig.build_batch_source(
    image_path           = '/data/images/*.mrxs',
    mask_path            = '/data/masks/{image}.tif',
    stat_path            = '/data/stats/{image}.stat',
    labels               = [1, 2, 3],
    read_spacing         = 4.0,
    purpose_distribution = {'train': 0.8, 'val': 0.2},
    random_item_order    = True
)
batch_source.save('/configs/dataset.yaml')
```

---

#### `loggers.py`

```python
from digitalpathology.utils import loggers as dptloggers

dptloggers.init_console_logger(debug=True)           # verbose console output
dptloggers.init_file_logger('/logs/run.log', debug=False)
dptloggers.init_silent_logger()                      # suppress all output
```

---

### `errors` — Error Hierarchy

```
DigitalPathologyError  (base)
    │
    ├── DigitalPathologyImageError
    │       └── raised by imagereader, imagewriter, annotation
    │
    ├── DigitalPathologyProcessingError
    │       └── raised by conversion, arithmetic, threshold, …
    │
    ├── DigitalPathologyLabelError
    │       └── raised by labelmapper (invalid/non-contiguous labels)
    │
    ├── DigitalPathologyConfigError
    │       └── raised by batchsource, dataconfig (bad YAML)
    │
    ├── DigitalPathologyDataError
    │       └── raised by batchsampler (missing files, empty batches)
    │
    ├── DigitalPathologyAugmentationError
    ├── DigitalPathologyWeightError
    ├── DigitalPathologyRangeError
    └── DigitalPathologyBufferError
```

---

## Scripts

19 command-line tools in `scripts/`. All batch scripts support `-j N` for parallel processing and display a Rich progress bar.

| Script | Description | Key Arguments |
|--------|-------------|---------------|
| `convertannotations.py` | ASAP XML → mask TIFF | `-i` images, `-a` annotations, `-m` masks, `-l` label map, `-s` spacing, `-j` workers |
| `convertmasks.py` | Mask TIFF → ASAP XML | `-m` masks, `-a` output XML, `-c` spacing, `-j` workers |
| `createpreviews.py` | WSI + mask → overlay preview | `-i` image, `-m` mask, `-p` preview, `-s` spacing, `-j` workers |
| `combinemasks.py` | Arithmetic on mask pairs | `-l` left, `-r` right, `-e` result, `-o` operand, `-j` workers |
| `comparemasks.py` | Similarity between two masks | `-i` reference, `-m` template, `-s` spacing |
| `thresholdimage.py` | Binary threshold | `-i` input, `-o` output, `-t` threshold, `-j` workers |
| `maplabels.py` | Remap label values | `-i` input, `-o` output, `-d` label map, `-j` workers |
| `checkdataset.py` | Validate dataset config | `-d` data YAML, `-m` spacing, `-j` workers |
| `createdataconfig.py` | Build dataset YAML | `-i` images, `-m` masks, `-l` labels, `-s` spacing |
| `extractpatches.py` | Extract patches to HDF5 | `-d` data config, `-o` output |
| `imageinfo.py` | Print WSI metadata | `-i` image |
| `setspacing.py` | Override pixel spacing | `-i` image, `-s` spacing |
| `zoomimage.py` | Resize/resample image | `-i` image, `-o` output, `-z` zoom |
| `saveimagesatlevel.py` | Export one pyramid level | `-i` image, `-o` output, `-s` spacing |
| `preprocessmasks.py` | Clean/filter masks | `-i` masks, `-o` output |
| `collectdataset.py` | Copy dataset files | `-d` data config, `-o` output |
| `copyfilesinconfig.py` | Copy files from config | `-d` config, `-o` output |
| `createfolds.py` | Create cross-validation folds | `-d` data config, `-n` folds |
| `applynetwork.py` | Run network on WSI | `-i` image, `-n` network, `-o` output |

### Usage Examples

```bash
# Convert all annotations in a folder (8 parallel workers)
python3 scripts/convertannotations.py \
    -i "/data/images/*.mrxs" \
    -a "/data/annotations/{image}.xml" \
    -m "/data/masks/{image}.tif" \
    -l "{'background': 0, 'epithelium': 1, 'stroma': 2}" \
    -s 1.0 \
    -j 8

# Build dataset config from directory structure
python3 scripts/createdataconfig.py \
    -i "/data/images/*.mrxs" \
    -m "/data/masks/{image}.tif" \
    -l "[1, 2, 3]" \
    -s 4.0 \
    -o dataset.yaml

# Validate every image-mask pair in the dataset
python3 scripts/checkdataset.py \
    -d dataset.yaml \
    -m 1.0 \
    -j 8

# Create coloured overlay previews
python3 scripts/createpreviews.py \
    -i "/data/images/*.mrxs" \
    -m "/data/masks/{image}.tif" \
    -p "/data/previews/{image}.png" \
    -s 4.0 \
    -j 8
```

---

## Key Data Types

| Type | Format | Description |
|------|--------|-------------|
| WSI | `.mrxs`, `.svs`, `.tif`, `.ndpi` | Multi-resolution whole slide image |
| Mask | `.tif` (uint8/uint16) | Integer label per pixel, same resolution as WSI |
| Annotation | `.xml` (ASAP format) | Polygon regions with group names |
| Stat file | `.stat` (binary) | Pre-computed label distribution per tile |
| Data config | `.yaml` / `.json` | Dataset descriptor (image-mask-label-purpose mapping) |
| Patch batch | `numpy` `(N, H, W, C)` | Extracted patches ready for training |
| Label batch | `numpy` `(N, H, W)` or `(N, H, W, K)` | Integer or one-hot encoded labels |
| Weight batch | `numpy` `(N, H, W)` float32 | Per-pixel loss weights |

---

## Dependencies

| Package | Role |
|---------|------|
| `multiresolutionimageinterface` (ASAP 2.2) | WSI I/O |
| `numpy` | Array operations |
| `scipy` | Morphological ops, interpolation |
| `scikit-image` | Connected components, resize |
| `imageio` | Plain image I/O |
| `albumentations` | Augmentation backend |
| `rdp` | Polygon simplification (mask → annotation) |
| `pyyaml` | Config file parsing |
| `rich` | Progress bars and console output |
