export PYTHONPATH="${PYTHONPATH}:/opt/ASAP/bin:/home/user/source/pathology-common:/home/user/source/pathology-fast-inference"

echo "Starting copying relevant files"
rm -r "/home/user/process"
mkdir -p "/home/user/process" 
cp -r "/data/pathology/users/pierpaolo/process" "/home/user/"



#BASE PATH
base_path="/home/user/process"
#OUTPUT TB PATH
output_tb_path="$base_path/tb"


mkdir -p "$input_wsi_path" "$output_tif_path" "$(dirname "$output_tb_path")"
#PRINT PARAMETERS   
echo "tb_mask: $tb_mask"


all_files=("$input_wsi_path"/*)

for file in "${all_files[@]}" 
do
    echo "File: $file"
    filename=$(basename "$file")
    extension="${file##*.}"
    echo "Extension: $extension"
    
    tb_mask="${output_tb_path}/${filename%.*}.tif"
    echo "Processing file"
    model_tb_path="/home/user/source/models/tb/playground_soft-cloud-137_best_model.pt"
    echo "starting tissue segmentation"
    echo "Filename: ${filename}"
    python3 /home/user/source/pathology-fast-inference/scripts/applynetwork_multiproc.py \
                                                --input_wsi_path=$file \
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


echo "Updating Archive"
cp -ru "/home/user/process" "/data/pathology/users/pierpaolo" 

echo "DONE"
