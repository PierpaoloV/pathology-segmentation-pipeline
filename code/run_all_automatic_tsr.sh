#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:/opt/ASAP/bin:/home/user/source/pathology-common:/home/user/source/pathology-fast-inference"


inputs="/home/user/process/image/*.tif"

output_folder="/home/user/process/tsr"
mkdir -p $output_folder


classification_image_folder="/home/user/process/epithelium"

tumor_bulk_folder="/home/user/process/tb"

tissue_mask_folder="/home/user/process/tb"


echo "Hello"

#Print how many elements are in inputs
echo "Number of elements in inputs: $(echo ${inputs} | wc -w)"
# COUNTER=0

for entry in ${inputs} 
do 
    echo "ENTRY: ${entry}, COUNTER: [${COUNTER} / $(echo ${inputs} | wc -w)]"

    filename=$(basename -- "$entry")
    stripped_filename="$(cut -d'.' -f1 <<<"$filename")"


    tissue_mask_file=$tissue_mask_folder/${stripped_filename}_tissue.tif
    classification_file=$classification_image_folder/${stripped_filename}_epithelium.tif
    tumor_bulk_file=$tumor_bulk_folder/${stripped_filename}_tissue.tif
# Check if tumour bulk file exists else use the tumour mask
    if [ -f "${tumor_bulk_file}" ]; then
        echo "Tumour bulk file: ${tumor_bulk_file}"
    else
        tumor_bulk_file=$tumor_bulk_folder/${stripped_filename}.tif
    fi
    output_file_name=$output_folder/${stripped_filename}_likelihood_map.tif
    echo "Tissue mask: ${tissue_mask_file}"
    if [ -f "${output_file_name}" ]; then
        echo "${output_file_name} already exists"
    else
    python3 /home/user/source/code/compute_tsr.py \
        --img_filename="${entry}" \
        --output_dir="${output_folder}" \
        --classification_map_filename="${classification_file}" \
        --tissue_mask_filename="${tissue_mask_file}" \
        --level=5 \
        --bulk_mask_filename="${tumor_bulk_file}" \
        --save_fig
    fi
    COUNTER=$((COUNTER+1))
done

echo "Combining all csv files..."
python3 /home/user/source/code/combine_automatic_tsr_output.py --input_folder="${output_folder}" --name="TSR_combined"

echo "Done!"
