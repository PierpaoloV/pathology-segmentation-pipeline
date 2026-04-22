#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash /home/user/source/code/process_dataset.sh \
    --input_wsi_path /home/user/image \
    --output_wsi_path /home/user/output

Description:
  Preprocess a dataset folder into one tissue mask TIFF per slide.

  By default the script reads slides directly from --input_wsi_path.
  If --tmp is set, it stages one slide at a time into /home/user/tmp,
  processes that local copy, and then deletes it before moving to the
  next slide. This is useful for Samba-mounted datasets.

  Default TB path:
    - downloads the Hugging Face "tb" family on demand
    - runs fast inference with applynetwork_multiproc.py
    - defaults to 4.0 um/px for both --read_spacing and --write_spacing

  SAM path (--sam):
    - downloads the Hugging Face "sam" family on demand
    - runs Atlas SAM through atlas_tb_mask.py
    - defaults to 8.0 um/px for --sam-input-spacing
    - keeps the native SAM output spacing to avoid the final mask resampling step

Required arguments:
  --input_wsi_path PATH      Input folder containing WSIs, or a single WSI file
  --output_wsi_path PATH     Output directory or output template containing {image}

Common optional arguments:
  --input_filter REGEX       File-selection regex for folder mode
  --tmp                      Stage one slide at a time into /home/user/tmp before processing
  --tile_size INT            Tile size
  --overwrite                Overwrite existing output masks

TB optional arguments:
  --read_spacing FLOAT       Input spacing for applynetwork TB inference (default: 4.0)
  --write_spacing FLOAT      Output spacing for applynetwork TB masks (default: 4.0)
  --readers INT              Number of readers (default: 20)
  --writers INT              Number of writers (default: 20)
  --batch_size INT           GPU batch size (default: 90)
  --gpu_count INT            GPU count (default: 1)

SAM optional arguments:
  --sam                      Use Atlas SAM instead of the standard TB model
  --sam-input-spacing FLOAT  SAM input spacing in um/px (default: 8.0)
  --spacing-at-level-0 FLOAT Override missing level-0 spacing metadata

Examples:
  bash /home/user/source/code/process_dataset.sh \
    --input_wsi_path /home/user/image \
    --output_wsi_path /home/user/output \
    --overwrite

  bash /home/user/source/code/process_dataset.sh \
    --input_wsi_path /home/user/image \
    --output_wsi_path /home/user/output \
    --sam \
    --sam-input-spacing 8.0 \
    --overwrite
EOF
}

export PYTHONPATH="${PYTHONPATH:-}:/opt/ASAP/bin:/home/user/source/pathology-common:/home/user/source/pathology-fast-inference"

INPUT_WSI_PATH=""
OUTPUT_WSI_PATH=""
INPUT_FILTER='.*\.(mrxs|ndpi|svs|tif|tiff)$'
CACHE_PATH="/home/user/tmp/process_dataset"
TILE_SIZE="512"
OVERWRITE=0
USE_TMP=0

USE_SAM=0
SAM_INPUT_SPACING="8.0"
SPACING_AT_LEVEL_0=""

READ_SPACING="4.0"
WRITE_SPACING="4.0"
READERS="20"
WRITERS="20"
BATCH_SIZE="90"
GPU_COUNT="1"
AXES_ORDER="cwh"
CUSTOM_PROCESSOR="torch_processor"
RECONSTRUCTION_INFORMATION="[[0,0,0,0],[1,1],[96,96,96,96]]"

TB_MODEL_PATH="/home/user/source/models/tb/playground_soft-cloud-137_best_model.pt"
SAM_CHECKPOINT_PATH="/home/user/source/models/sam/model.pth"
SAM_CONFIG_PATH="/home/user/source/models/sam/sam2.1_hiera_t.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_wsi_path)
            INPUT_WSI_PATH="$2"
            shift 2
            ;;
        --output_wsi_path)
            OUTPUT_WSI_PATH="$2"
            shift 2
            ;;
        --input_filter)
            INPUT_FILTER="$2"
            shift 2
            ;;
        --tile_size)
            TILE_SIZE="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE=1
            shift
            ;;
        --tmp)
            USE_TMP=1
            shift
            ;;
        --sam)
            USE_SAM=1
            shift
            ;;
        --sam-input-spacing)
            SAM_INPUT_SPACING="$2"
            shift 2
            ;;
        --spacing-at-level-0)
            SPACING_AT_LEVEL_0="$2"
            shift 2
            ;;
        --read_spacing)
            READ_SPACING="$2"
            shift 2
            ;;
        --write_spacing)
            WRITE_SPACING="$2"
            shift 2
            ;;
        --readers)
            READERS="$2"
            shift 2
            ;;
        --writers)
            WRITERS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gpu_count)
            GPU_COUNT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "${INPUT_WSI_PATH}" || -z "${OUTPUT_WSI_PATH}" ]]; then
    echo "--input_wsi_path and --output_wsi_path are required." >&2
    usage >&2
    exit 1
fi

if [[ "${USE_TMP}" -eq 1 ]]; then
    mkdir -p "/home/user/tmp"
    mkdir -p "${CACHE_PATH}"
fi

if [[ "${OUTPUT_WSI_PATH}" == *"{image}"* ]]; then
    OUTPUT_TEMPLATE="${OUTPUT_WSI_PATH}"
    mkdir -p "$(dirname "${OUTPUT_TEMPLATE}")"
else
    mkdir -p "${OUTPUT_WSI_PATH}"
    OUTPUT_TEMPLATE="${OUTPUT_WSI_PATH%/}/{image}.tif"
fi

OVERWRITE_FLAG=()
if [[ "${OVERWRITE}" -eq 1 ]]; then
    OVERWRITE_FLAG=(--overwrite)
fi

echo "Input path: ${INPUT_WSI_PATH}"
echo "Output path: ${OUTPUT_TEMPLATE}"
echo "Input filter: ${INPUT_FILTER}"
echo "Tmp staging: $( [[ "${USE_TMP}" -eq 1 ]] && printf '%s' "${CACHE_PATH}" || printf '%s' 'disabled' )"
echo "Tile size: ${TILE_SIZE}"

resolve_output_path() {
    local slide_path="$1"
    local slide_id
    slide_id="$(basename "${slide_path}")"
    slide_id="${slide_id%.*}"
    echo "${OUTPUT_TEMPLATE//\{image\}/${slide_id}}"
}

copy_slide_to_cache() {
    local source_slide="$1"
    local stage_dir="$2"
    local staged_slide="${stage_dir}/$(basename "${source_slide}")"

    cp -p "${source_slide}" "${staged_slide}"
    echo "Copied slide to local staging: ${staged_slide}" >&2

    if [[ "${source_slide##*.}" == "mrxs" || "${source_slide##*.}" == "MRXS" ]]; then
        local source_bundle_dir="${source_slide%.*}"
        local source_bundle_name
        source_bundle_name="$(basename "${source_bundle_dir}")"
        if [[ ! -d "${source_bundle_dir}" ]]; then
            echo "Missing MRXS sibling directory: ${source_bundle_dir}" >&2
            return 1
        fi
        cp -a "${source_bundle_dir}" "${stage_dir}/${source_bundle_name}"
        echo "Copied MRXS bundle directory: ${stage_dir}/${source_bundle_name}" >&2
    fi

    echo "${staged_slide}"
}

run_sam_on_slide() {
    local slide_to_process="$1"
    local output_path="$2"

    local sam_args=(
        python3 /home/user/source/code/atlas_tb_mask.py
        --input-wsi "${slide_to_process}"
        --output-mask "${output_path}"
        --sam2-device cuda
        --sam-input-spacing "${SAM_INPUT_SPACING}"
        --sam2-checkpoint-path "${SAM_CHECKPOINT_PATH}"
        --sam2-config-path "${SAM_CONFIG_PATH}"
        --tile-size "${TILE_SIZE}"
        --keep-native-output-spacing
    )

    if [[ -n "${SPACING_AT_LEVEL_0}" ]]; then
        sam_args+=(--spacing-at-level-0 "${SPACING_AT_LEVEL_0}")
    fi

    sam_args+=("${OVERWRITE_FLAG[@]}")
    "${sam_args[@]}"
}

run_tb_on_slide() {
    local slide_to_process="$1"
    local output_path="$2"
    local tb_args=(
        python3 /home/user/source/pathology-fast-inference/scripts/applynetwork_multiproc.py
        --model_path "${TB_MODEL_PATH}"
        --input_wsi_path "${slide_to_process}"
        --output_wsi_path "${output_path}"
        --read_spacing "${READ_SPACING}"
        --write_spacing "${WRITE_SPACING}"
        --tile_size "${TILE_SIZE}"
        --readers "${READERS}"
        --writers "${WRITERS}"
        --batch_size "${BATCH_SIZE}"
        --gpu_count "${GPU_COUNT}"
        --axes_order "${AXES_ORDER}"
        --custom_processor "${CUSTOM_PROCESSOR}"
        --reconstruction_information "${RECONSTRUCTION_INFORMATION}"
        --quantize
    )

    if [[ "${USE_TMP}" -eq 1 ]]; then
        tb_args+=(--cache_path "${CACHE_PATH}")
    fi

    tb_args+=("${OVERWRITE_FLAG[@]}")
    "${tb_args[@]}"
}

process_one_slide() {
    local source_slide="$1"
    local output_path
    output_path="$(resolve_output_path "${source_slide}")"
    local stage_dir=""
    cleanup_stage_dir() {
        if [[ -n "${stage_dir}" && -d "${stage_dir}" ]]; then
            rm -rf "${stage_dir}"
            echo "Removed staged copy: ${stage_dir}"
        fi
    }
    trap cleanup_stage_dir RETURN

    if [[ -f "${output_path}" && "${OVERWRITE}" -ne 1 ]]; then
        echo "Skipping $(basename "${source_slide}"): output exists at ${output_path}"
        return 0
    fi

    mkdir -p "$(dirname "${output_path}")"

    local slide_to_process="${source_slide}"
    if [[ "${USE_TMP}" -eq 1 ]]; then
        local slide_id
        slide_id="$(basename "${source_slide}")"
        slide_id="${slide_id%.*}"
        stage_dir="$(mktemp -d "${CACHE_PATH%/}/${slide_id}.XXXXXX")"
        echo "Staging ${source_slide} into ${stage_dir}"
        slide_to_process="$(copy_slide_to_cache "${source_slide}" "${stage_dir}")"
    fi

    echo "Processing slide: ${source_slide}"
    echo "Using input: ${slide_to_process}"
    echo "Writing mask: ${output_path}"

    if [[ "${USE_SAM}" -eq 1 ]]; then
        run_sam_on_slide "${slide_to_process}" "${output_path}"
    else
        run_tb_on_slide "${slide_to_process}" "${output_path}"
    fi
}

INPUT_SLIDES=()
if [[ -f "${INPUT_WSI_PATH}" ]]; then
    INPUT_SLIDES+=("${INPUT_WSI_PATH}")
else
    shopt -s nullglob
    shopt -s nocasematch
    for candidate in "${INPUT_WSI_PATH}"/*; do
        [[ -f "${candidate}" ]] || continue
        if [[ "$(basename "${candidate}")" =~ ${INPUT_FILTER} ]]; then
            INPUT_SLIDES+=("${candidate}")
        fi
    done
    shopt -u nocasematch
    shopt -u nullglob
fi

if [[ "${#INPUT_SLIDES[@]}" -eq 0 ]]; then
    echo "No matching input slides found in ${INPUT_WSI_PATH}" >&2
    exit 1
fi

if [[ "${USE_SAM}" -eq 1 ]]; then
    echo "Mode: sam"
    echo "SAM input spacing: ${SAM_INPUT_SPACING}"
    python3 /home/user/source/download_models.py sam
else
    echo "Mode: tb"
    echo "Read spacing: ${READ_SPACING}"
    echo "Write spacing: ${WRITE_SPACING}"
    python3 /home/user/source/download_models.py tb
fi

echo "Slides to process: ${#INPUT_SLIDES[@]}"
for slide_path in "${INPUT_SLIDES[@]}"; do
    process_one_slide "${slide_path}"
done
