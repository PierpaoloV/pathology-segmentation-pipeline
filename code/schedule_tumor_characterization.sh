export PYTHONPATH="${PYTHONPATH}:/opt/ASAP/bin:/home/user/source/pathology-common:/home/user/source/pathology-fast-inference"

echo "Starting copying relevant files"
rm -r "/home/user/process"
mkdir -p "/home/user/process" 
cp -r "/data/pathology/users/pierpaolo/process" "/home/user/"



#BASE PATH
base_path="/home/user/process"
#TB SEGMENTATION
#INPUT WSI PATH
input_wsi_path="$base_path/image"
#OUTPUT TIF PATH
output_tif_path="$base_path/tif/"
#OUTPUT TB PATH
output_tb_path="$base_path/tb"


#OUTPUT EPITHELIUM PATH
epithelium_path="$base_path/epithelium"
#OUTPUT MULTI TISSUE PATH
multi_tissue_path="$base_path/multi_tissue/{image}.tif"
#TUMOR PATH
tumor_path="$base_path/tumor/{image}.tif"


mkdir -p "$input_wsi_path" "$output_tif_path" "$(dirname "$output_tb_path")" "$(dirname "$epithelium_path")" "$(dirname "$multi_tissue_path")"

#PRINT PARAMETERS   
echo "input_wsi_path: $input_wsi_path"
echo "output_tb_path: $output_tb_path"
echo "tb_mask: $tb_mask"
echo "epithelium_path: $epithelium_path"

all_files=("$input_wsi_path"/*)

for file in "${all_files[@]}" 
do
    echo "File: $file"
    filename=$(basename "$file")
    extension="${file##*.}"
    echo "Extension: $extension"
    
    tb_mask="${output_tb_path}/${filename%.*}.tif"
    
    if [[ "$extension" != "tif" && "$extension" != "json" ]] 
    then
        #convert
        echo "Converting: $filename"
        python3 /home/user/source/code/convert.py --input_path "$file" \
                                                  --output_dir "$output_tif_path" \
                                                  --ext "$extension" 
        #tb
        
       echo "Processing file"
       epithelium="${epithelium_path}/${filename%.*}.tif"
       bash /home/user/source/code/start_characterization.sh "$file" \
                                     "$tb_mask" \
                                     "$tb_mask" \
                                     "$epithelium" \
                                     "$multi_tissue_path" \
                                     "$tumor_path"
   elif [[ "$extension" == "tif" ]]
   then
       echo "Processing file"
       epithelium="${epithelium_path}/${filename%.*}.tif"
       bash /home/user/source/code/start_characterization.sh "$file" \
                                     "$tb_mask" \
                                     "$tb_mask" \
                                     "$epithelium" \
                                     "$multi_tissue_path" \
                                     "$tumor_path"
    else
        echo "$file Is not recognised."

    fi
done

# Call the second script with the parameters
# bash /home/user/source/code/start_characterization.sh "$output_tif_path" \
#                                 "$output_tb_path" \
#                                 "$tb_mask" \
#                                 "$epithelium_path" \
#                                 "$multi_tissue_path" \
#                                 "$tumor_path"


echo "Updating Archive"
cp -ru "/home/user/process" "/data/pathology/users/pierpaolo" 

echo "DONE"
