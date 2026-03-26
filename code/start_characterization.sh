#!/bin/bash

image_to_process=$1

mkdir -p /home/user/process/tb \
         /home/user/process/epithelium \
         /home/user/process/tumor \
         /home/user/process/concave_hull_masks

filename=$(basename -- "$image_to_process")

output_tb=/home/user/process/tb/${filename%.*}_tissue.tif
output_epi=/home/user/process/epithelium/${filename%.*}_epithelium.tif
output_tum=/home/user/process/tumor/${filename%.*}_tumor.tif
convex_hulls_path=/home/user/process/concave_hull_masks

#MODEL TB PATH
model_tb_path="/home/user/source/models/tb/playground_soft-cloud-137_best_model.pt"
#model_tb_path="/data/pathology/users/pierpaolo/Pierpaolo/dockers/epithelium_segmentation_gc/models/tb/playground_soft-cloud-137_best_model.pt"
#MODEL EPITHELIUM PATH
model_epi_path="/home/user/source/models/multi-tissue/best_models"
#model_epi_path="/data/pathology/users/pierpaolo/Pierpaolo/dockers/epithelium_segmentation_gc/models/epithelium/best_models"



extension="${image_to_process##*.}"
echo "Extension: $extension"

echo $image_to_process

if [[ "$extension" != "tif" ]] 
    then
    echo "Converting: $filename"
    output_tif_path=/home/user/data/tmp_masks/${filename%.*}_converted.tif
    python3 /home/user/source/code/convert.py --input_path "$image_to_process" \
                                                --output_dir "/home/user/tmp_input" \
                                                --ext "$extension" 
fi

echo "starting tissue segmentation"
echo "Filename: ${filename}"
python3 /home/user/source/pathology-fast-inference/scripts/applynetwork_multiproc.py \
                                            --input_wsi_path=${image_to_process} \
                                            --output_wsi_path=${output_tb} \
                                            --model_path=${model_tb_path} \
                                            --read_spacing=4.0 \
                                            --write_spacing=4.0 \
                                            --tile_size=512 \
                                            --readers=20 \
                                            --writers=20 \
                                            --batch_size=90 \
                                            --gpu_count=1 \
                                            --axes_order='cwh' \
                                            --custom_processor="torch_processor" \
                                            --reconstruction_information="[[0,0,0,0],[1,1],[96,96,96,96]]" \
                                            --quantize



echo "starting epithelium segmentation"
python3 /home/user/source/pathology-fast-inference/scripts/applynetwork_multiproc.py \
                                                --input_wsi_path=${image_to_process} \
                                                --output_wsi_path=${output_epi} \
                                                --model_path="${model_epi_path}" \
                                                --read_spacing=1.0 \
                                                --write_spacing=1.0 \
                                                --mask_wsi_path=${output_tb} \
                                                --mask_spacing=4.0 \
                                                --mask_class=2 \
                                                --tile_size=512 \
                                                --readers=20 \
                                                --writers=20 \
                                                --batch_size=30 \
                                                --gpu_count=1 \
                                                --axes_order='cwh' \
                                                --custom_processor="torch_processor" \
                                                --reconstruction_information="[[0,0,0,0],[1,1],[100,100,100,100]]" \
                                                --quantize  



echo "Computing TSR"
bash /home/user/source/code/run_all_automatic_tsr.sh