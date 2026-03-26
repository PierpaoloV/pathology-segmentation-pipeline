#!/bin/bash
#
set -e

mkdir -p "/home/user/data/tmp_input" 
mkdir -p "/home/user/data/tmp_masks"
mkdir -p "/home/user/data/tissue_masks"
mkdir -p "/home/user/data/concave_hull_masks"
mkdir /output/images
mkdir /output/images/fixed-mask
mkdir /output/images/gross-tumor-volume-segmentation

echo "Processing all input files:"
ls /input

cp /input/*.tif /home/user/data/tmp_input

for file in /home/user/data/tmp_input/*.tif 
do 
    filename=$(basename -- "$file")
    echo "Start processing: ${filename}"
    sh /home/user/source/code/start_characterization.sh "${file}" 

done


mv /home/user/data/tmp_masks/*_epithelium.tif /output/images/fixed-mask
mv /home/user/data/tmp_masks/*_tumor.tif /output/images/gross-tumor-volume-segmentation
echo "Processing complete..."
exit 0
