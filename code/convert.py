import argparse
import logging
import os
import sys
import time

import digitalpathology.errors.imageerrors as dptimageerrors
import digitalpathology.image.processing.zoom as dptzoom
# import digitalpathology.utils.anonymize as dptanonymize
import digitalpathology.utils.imagefile as dptimagefile
import digitalpathology.utils.loggers as dptloggers
import numpy as np
import pandas as ps


def collect_single_image(image_source_path,
                         image_target_folder_path,
                         annotation_source_path,
                         annotation_target_folder_path,
                         target_basename,
                         index,
                         level,
                         pixel_spacing,
                         spacing_tolerance,
                         jpeg_quality,
                         anonymize,
                         convert,
                         overwrite):
    """
    Collect a single image: copy, anonymize, convert and rename it.

    Args:
        image_source_path (str): Source image path.
        image_target_folder_path (str): Target image folder path.
        annotation_source_path (str, None): Annotation source path.
        annotation_target_folder_path (str, None): Target annotation folder path.
        target_basename (str, None): Target basename for renaming the image and the annotation.
        index (tuple): Tuple of (current, total) processed images.
        level (int, None): Collection level of the image.
        pixel_spacing (float, None): Collection pixel spacing of the image to process. (micrometer).
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        jpeg_quality (int, None): JPEG quality (1-100) when using JPEG as compression method. If not set, the default of ImageWriter is used.
        anonymize (bool): Anonymization flag.
        convert (bool): Image conversion to TIFF format flag.
        overwrite (bool): Overwrite flag.

    Raises:
        ValueError: The target file name for conversion is not a TIFF file name.
        ValueError: The source file is renamed to a different file format extension.

        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
        DigitalPathololgyProcessingError: Processing errors.
    """

    # Log progress.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Processing [{index}/{count}]: {name}'.format(index=index[0], count=index[1], name=os.path.basename(image_source_path)))
    # Normalize paths.
    #
    norm_image_source_path = os.path.normpath(image_source_path)
    image_basename = os.path.basename(norm_image_source_path)
    image_base, image_extentsion = os.path.splitext(image_basename)

    # Check if conversion needed. It can be the conversion flag or the fact that the images are saved at a given level or pixel spacing.
    #
    conversion_enabled = convert or level is not None or pixel_spacing is not None

    # Construct the target paths.
    #
    if target_basename:
        if os.path.splitext(target_basename)[1]:
            final_target_basename = target_basename
        else:
            final_target_basename_ext = '.tif' if conversion_enabled else image_extentsion
            final_target_basename = '{base}{ext}'.format(base=target_basename, ext=final_target_basename_ext)
    else:
        final_target_basename_ext = '.tif' if conversion_enabled else image_extentsion
        final_target_basename = '{base}{ext}'.format(base=image_base, ext=final_target_basename_ext)

    norm_image_target_path = os.path.join(os.path.normpath(image_target_folder_path), final_target_basename)

    # Check if the final target image path (after anonymizing, renaming or converting) exists.
    #
    if not os.path.exists(norm_image_target_path) or overwrite:
        # Check if conversion is configured.
        #
        tiff_extensions = ('.tif', '.tiff')
        if conversion_enabled:
            # Conversion is necessary. Check it the target extension for TIFF conversion is correct.
            #
            if target_basename and os.path.splitext(norm_image_target_path)[1].lower() not in tiff_extensions:
                raise ValueError('Invalid extension for conversion: {name}'.format(name=target_basename))

            # Check if collecting an image at a given level or pixel spacing is necessary. That means separate anonymization and conversion steps
            # are not necessary since the internal TIFF writer library ignore everything except pixel data and can only write TIFF file format.
            #
            if level is None and pixel_spacing is None:
                # Check if conversion is necessary. Copy or convert the image and anonymize it.
                #
                if image_extentsion.lower() not in tiff_extensions:
                    if anonymize:
                        intermediate_target_path = os.path.join(os.path.normpath(image_target_folder_path), image_basename)

                        # Copy the image to an intermediate path, anonymize it, convert it to the target path and remove the intermediate image.
                        #
                        dptimagefile.copy_image(source_path=norm_image_source_path, target_path=intermediate_target_path, overwrite=overwrite)
                        dptanonymize.anonymize_image(image_path=intermediate_target_path)

                        dptzoom.save_image_at_level(image=intermediate_target_path,
                                                    output_path=norm_image_target_path,
                                                    level=0,
                                                    pixel_spacing=None,
                                                    spacing_tolerance=spacing_tolerance,
                                                    jpeg_quality=jpeg_quality,
                                                    work_path=None,
                                                    clear_cache=True,
                                                    overwrite=overwrite)

                        dptimagefile.remove_image(image_path=intermediate_target_path)
                    else:
                        # Move the image by directly writing he result of the conversion to the target path.
                        #
                        dptzoom.save_image_at_level(image=norm_image_source_path,
                                                    output_path=norm_image_target_path,
                                                    level=0,
                                                    pixel_spacing=None,
                                                    spacing_tolerance=spacing_tolerance,
                                                    jpeg_quality=jpeg_quality,
                                                    work_path=None,
                                                    clear_cache=True,
                                                    overwrite=overwrite)
                else:
                    # The source image is already TIFF. Just copy it and anonymize it if necessary.
                    #
                    dptimagefile.copy_image(source_path=norm_image_target_path, target_path=norm_image_target_path, overwrite=overwrite)
                    if anonymize:
                        dptanonymize.anonymize_image(image_path=norm_image_target_path)
            else:
                try:
                    # Collect the images on a given level or pixel spacing.
                    #
                    dptzoom.save_image_at_level(image=norm_image_source_path,
                                                output_path=norm_image_target_path,
                                                level=level,
                                                pixel_spacing=pixel_spacing,
                                                spacing_tolerance=spacing_tolerance,
                                                jpeg_quality=jpeg_quality,
                                                work_path=None,
                                                clear_cache=True,
                                                overwrite=overwrite)

                except dptimageerrors.PixelSpacingLevelError:
                    # There is no appropriate level found for the given pixel spacing. Try to find a level with smaller pixel spacing and zoom out to get to the required pixel spacing.
                    #
                    alternative_pixel_spacing = pixel_spacing / 2.0

                    logger.info('The {spacing} um pixel spacing is not available in the source image, trying to zoom out from {alternative} um'.format(spacing=pixel_spacing,
                                                                                                                                                       alternative=alternative_pixel_spacing))

                    dptzoom.zoom_image(image=norm_image_source_path,
                                       output_path=norm_image_target_path,
                                       zoom=0.5,
                                       level=level,
                                       pixel_spacing=alternative_pixel_spacing,
                                       spacing_tolerance=spacing_tolerance,
                                       round_shape=False,
                                       interpolation_order=2,
                                       jpeg_quality=jpeg_quality,
                                       work_path=None,
                                       clear_cache=True,
                                       overwrite=overwrite)
                except Exception:
                    # Some other, not expected error happened.
                    #
                    raise
        else:
            # Check extension changes.
            #
            target_extension = os.path.splitext(norm_image_target_path)[1]
            if image_extentsion.lower() != target_extension.lower() and (image_extentsion.lower() not in tiff_extensions or target_extension.lower() not in tiff_extensions):
                raise ValueError('Extension mismatch for renaming from {source} to {target}.'.format(source=os.path.basename(norm_image_target_path), target=target_basename))

            # Copy the image and the annotation to the target folder and anonymize the image.
            #
            dptimagefile.copy_image(source_path=norm_image_source_path, target_path=norm_image_target_path, overwrite=overwrite)
            if anonymize:
                dptanonymize.anonymize_image(image_path=norm_image_target_path)
    else:
        logger.info('Skipping, target already exists: {path}'.format(path=norm_image_target_path))

    # Handle the annotation file.
    #
    if annotation_source_path:
        # Construct target annotation path.
        #
        norm_annotation_source_path = os.path.normpath(annotation_source_path)
        final_annotation_target_path = os.path.join(os.path.normpath(annotation_target_folder_path), '{base}{ext}'.format(base=os.path.splitext(final_target_basename)[0],
                                                                                                                          ext=os.path.splitext(norm_annotation_source_path)[1]))

        # Copy annotation file.
        #
        dptimagefile.copy_image(source_path=norm_annotation_source_path, target_path=final_annotation_target_path, overwrite=overwrite)



def collect_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_path', dest='input_path', required=True, help='input path expression')
    parser.add_argument('--output_dir', dest='output_dir', required=True, help='mask output directory')
    parser.add_argument('--ext', dest='ext', required=True, help='extension')
    return parser.parse_args()
def main():
     
    args = collect_arguments()
    extension = args.ext
    extension = '.' + extension 
    print('Extension is {}'.format(extension))
    image_path = args.input_path
#     img_list = [i.replace(extension, '') for i in os.listdir(image_path) if extension in i]
    # element_to_remove = 'pathology-tissue-background-segmentation'
    # if element_to_remove in img_list:
    #     img_list = img_list.remove(element_to_remove)
    image_target_folder_path = args.output_dir
    spacing_tolerance = 0.25
    anonymize = False
    convert = True
    overwrite = False
    c = 0 
    i = image_path.replace(extension,'')
    i = i.split('/')[-1]
#     for i in img_list:
        
    a = time.time()
#     print('{} / {} --> Converting {}'.format(c,len(img_list), i))
    print(f'Converting: {i}')
    target_basename = i
#     image_source_path = image_path + i + extension
    image_source_path = image_path
    print(f'Image source path is {image_source_path}')
    print(f'Image target folder path is {image_target_folder_path}')
    output = image_target_folder_path + target_basename + '.tif'
    if os.path.exists(output):
        print(f'File {output} already exists, skipping')
    else:
        print(f'File {output} does not exist, converting')
        collect_single_image(image_source_path=image_source_path,
                                image_target_folder_path=image_target_folder_path,
                                target_basename=target_basename,
                                index=(0,0),
                                annotation_source_path =None,
                                annotation_target_folder_path =None,
                                level=None,
                                pixel_spacing = None,
                                spacing_tolerance=spacing_tolerance,
                                jpeg_quality= 70,
                                anonymize=anonymize,
                                convert=convert,
                                overwrite=overwrite)
    b = time.time()
    print('Conversion done in {} minutes'.format(np.round((b-a)/60)))

if __name__ == '__main__':
	print('Conversion script. ??? ---> TIF')
	k = time.time()
	main()
	j = time.time()
	print('Everything done in about {} hours '.format(np.round((j-k)/3600)))