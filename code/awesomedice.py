from skimage.measure import label, regionprops
import ast
import numpy as np
import os
import yaml
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import digitalpathology.image.io.imagereader as dptimagereader
from sklearn.metrics import confusion_matrix
import pickle

from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress,
    SpinnerColumn, TextColumn, TimeElapsedColumn,
)
from rich.table import Table

console = Console()

def get_asap_colors():
    colormap = [[0, 0, 0, 0], [0, 224, 249, 255], [0, 249, 50, 255], [174, 249, 0, 255], [249, 100, 0, 255],
                [249, 0, 125, 255], [149, 0, 249, 255], [0, 0, 206, 255], [0, 185, 206, 255],
                [0, 206, 41, 255], [143, 206, 0, 255], [206, 82, 0, 255], [206, 0, 103, 255], [124, 0, 206, 255],
                [0, 0, 162, 255], [0, 145, 162, 255], [0, 162, 32, 255], [114, 162, 0, 255],
                [162, 65, 0, 255], [162, 0, 81, 255], [97, 0, 162, 255], [0, 0, 119, 255], [0, 107, 119, 255],
                [0, 119, 23, 255], [83, 119, 0, 255], [119, 47, 0, 255], [119, 0, 59, 255],
                [71, 0, 119, 255], [100, 100, 249, 255], [100, 234, 249, 255], [100, 249, 129, 255],
                [204, 249, 100, 255], [249, 159, 100, 255], [249, 100, 174, 255], [189, 100, 249, 255],
                [82, 82, 206, 255], [82, 193, 206, 255], [82, 206, 107, 255], [168, 206, 82, 255], [206, 131, 82, 255],
                [206, 82, 143, 255], [156, 82, 206, 255], [65, 65, 162, 255],
                [65, 152, 162, 255], [65, 162, 84, 255], [132, 162, 65, 255], [162, 104, 65, 255], [162, 65, 114, 255],
                [123, 65, 162, 255], [47, 47, 119, 255], [47, 112, 119, 255],
                [47, 119, 61, 255], [97, 119, 47, 255], [119, 76, 47, 255], [119, 47, 83, 255], [90, 47, 119, 255],
                [174, 174, 249, 255], [174, 242, 249, 255], [174, 249, 189, 255],
                [227, 249, 174, 255], [249, 204, 174, 255], [249, 174, 212, 255], [219, 174, 249, 255],
                [143, 143, 206, 255], [143, 199, 206, 255], [143, 206, 156, 255], [187, 206, 143, 255],
                [206, 168, 143, 255], [206, 143, 175, 255], [181, 143, 206, 255], [114, 114, 162, 255],
                [114, 157, 162, 255], [114, 162, 123, 255], [147, 162, 114, 255], [162, 132, 114, 255],
                [162, 114, 137, 255], [142, 114, 162, 255], [83, 83, 119, 255], [83, 115, 119, 255], [83, 119, 90, 255],
                [108, 119, 83, 255], [119, 97, 83, 255], [119, 83, 101, 255],
                [104, 83, 119, 255], [224, 224, 249, 255], [224, 247, 249, 255], [224, 249, 229, 255],
                [242, 249, 224, 255], [249, 234, 224, 255], [249, 224, 237, 255], [239, 224, 249, 255],
                [185, 185, 206, 255], [185, 204, 206, 255], [185, 206, 189, 255], [199, 206, 185, 255],
                [206, 193, 185, 255], [206, 185, 195, 255], [197, 185, 206, 255], [145, 145, 162, 255],
                [145, 160, 162, 255], [145, 162, 149, 255], [157, 162, 145, 255], [162, 152, 145, 255],
                [162, 145, 153, 255], [155, 145, 162, 255], [107, 107, 119, 255], [107, 118, 119, 255],
                [107, 119, 109, 255], [115, 119, 107, 255], [119, 112, 107, 255], [119, 107, 113, 255],
                [114, 107, 119, 255], [0, 0, 249, 255], [0, 224, 249, 255], [0, 249, 50, 255],
                [174, 249, 0, 255], [249, 100, 0, 255], [249, 0, 125, 255], [149, 0, 249, 255], [0, 0, 206, 255],
                [0, 185, 206, 255], [0, 206, 41, 255], [143, 206, 0, 255], [206, 82, 0, 255],
                [206, 0, 103, 255], [124, 0, 206, 255], [0, 0, 162, 255], [0, 145, 162, 255], [0, 162, 32, 255],
                [114, 162, 0, 255], [162, 65, 0, 255], [162, 0, 81, 255], [97, 0, 162, 255],
                [0, 0, 119, 255], [0, 107, 119, 255], [0, 119, 23, 255], [83, 119, 0, 255], [119, 47, 0, 255],
                [119, 0, 59, 255], [71, 0, 119, 255], [100, 100, 249, 255], [100, 234, 249, 255],
                [100, 249, 129, 255], [204, 249, 100, 255], [249, 159, 100, 255], [249, 100, 174, 255],
                [189, 100, 249, 255], [82, 82, 206, 255], [82, 193, 206, 255], [82, 206, 107, 255],
                [168, 206, 82, 255], [206, 131, 82, 255], [206, 82, 143, 255], [156, 82, 206, 255], [65, 65, 162, 255],
                [65, 152, 162, 255], [65, 162, 84, 255], [132, 162, 65, 255],
                [162, 104, 65, 255], [162, 65, 114, 255], [123, 65, 162, 255], [47, 47, 119, 255], [47, 112, 119, 255],
                [47, 119, 61, 255], [97, 119, 47, 255], [119, 76, 47, 255], [119, 47, 83, 255],
                [90, 47, 119, 255], [174, 174, 249, 255], [174, 242, 249, 255], [174, 249, 189, 255],
                [227, 249, 174, 255], [249, 204, 174, 255], [249, 174, 212, 255], [219, 174, 249, 255],
                [143, 143, 206, 255], [143, 199, 206, 255], [143, 206, 156, 255], [187, 206, 143, 255],
                [206, 168, 143, 255], [206, 143, 175, 255], [181, 143, 206, 255], [114, 114, 162, 255],
                [114, 157, 162, 255], [114, 162, 123, 255], [147, 162, 114, 255], [162, 132, 114, 255],
                [162, 114, 137, 255], [142, 114, 162, 255], [83, 83, 119, 255], [83, 115, 119, 255],
                [83, 119, 90, 255], [108, 119, 83, 255], [119, 97, 83, 255], [119, 83, 101, 255], [104, 83, 119, 255],
                [224, 224, 249, 255], [224, 247, 249, 255], [224, 249, 229, 255],
                [242, 249, 224, 255], [249, 234, 224, 255], [249, 224, 237, 255], [239, 224, 249, 255],
                [185, 185, 206, 255], [185, 204, 206, 255], [185, 206, 189, 255], [199, 206, 185, 255],
                [206, 193, 185, 255], [206, 185, 195, 255], [197, 185, 206, 255], [145, 145, 162, 255],
                [145, 160, 162, 255], [145, 162, 149, 255], [157, 162, 145, 255], [162, 152, 145, 255],
                [162, 145, 153, 255], [155, 145, 162, 255], [107, 107, 119, 255], [107, 118, 119, 255],
                [107, 119, 109, 255], [115, 119, 107, 255], [119, 112, 107, 255], [119, 107, 113, 255],
                [114, 107, 119, 255], [0, 0, 249, 255], [0, 224, 249, 255], [0, 249, 50, 255], [174, 249, 0, 255],
                [249, 100, 0, 255], [249, 0, 125, 255], [149, 0, 249, 255], [0, 0, 206, 255],
                [0, 185, 206, 255], [0, 206, 41, 255], [143, 206, 0, 255], [206, 82, 0, 255], [206, 0, 103, 255],
                [124, 0, 206, 255], [0, 0, 162, 255], [0, 145, 162, 255], [0, 162, 32, 255],
                [114, 162, 0, 255], [162, 65, 0, 255], [162, 0, 81, 255], [97, 0, 162, 255], [0, 0, 119, 255],
                [0, 107, 119, 255], [0, 119, 23, 255], [83, 119, 0, 255], [119, 47, 0, 255],
                [119, 0, 59, 255], [71, 0, 119, 255], [100, 100, 249, 255], [100, 234, 249, 255], [100, 249, 129, 255],
                [0, 249, 50, 255]]

    return np.array(colormap)

def assemble_jobs(mask_path, ground_truth_path, dc_path):
    """
    Assemble (source image path, source mask path, copy image path, copy mask path, work output path, work interval path, target output path, target interval path) job octets for
        network application.

    Args:
        mask_path (str): Path of the mask image to use.
        ground_truth_path (str): Path of the ground_truth_path to load.

    Returns:
        list: List of job tuples.

    Raises:
        InvalidDataFileExtensionError: The format cannot be derived from the file extension.
        InvalidDataSourceTypeError: Invalid JSON or YAML file.
        PurposeListAndRatioMismatchError: The configured purpose distribution does not match the available purposes.
    """

    # If data config path is given, only testing set is evaluated
    #
    if dc_path:
        print("Data config path is given, only testing data is evaluated.")
        with open(file=dc_path, mode='r') as param_file:
            dc_params = yaml.load(stream=param_file, Loader=yaml.SafeLoader)

        testing_images = []
        for category in list(dc_params['data']['testing'].keys()):
            for item in dc_params['data']['testing'][category]:
                image_name = os.path.splitext(os.path.basename(item['image']))[0]
                testing_images.append(image_name)

    # Find out operation mode. If all paths are file paths the mode is 'file'.
    #
    file_mode = os.path.isfile(mask_path)

    result_job_list = []
    if file_mode:
        # Return a single triplet if the paths were existing files.
        #
        if os.path.isfile(ground_truth_path):
            result_job_list.append((mask_path, ground_truth_path))
        else:
            raise ValueError("Files not found")

    else:
        mask_folder_content = glob.glob(mask_path)

        for mask_filepath in mask_folder_content:

            mask_filename = os.path.splitext(os.path.basename(mask_filepath))[0]
            if dc_path and mask_filename not in testing_images:
                continue
            ground_truth_filepath = ground_truth_path.format(image=mask_filename)

            if os.path.isfile(ground_truth_filepath):
                result_job_list.append((mask_filepath, ground_truth_filepath))
            else:
                print("ground truth path {} does not exist for segmentation mask {}, skipping".format(
                    ground_truth_filepath, mask_filepath))

    # Return the result list.
    #
    return result_job_list

def map_gt_to_segmentation(gt, mapping):
    max_key = max(mapping.keys())
    lut = np.zeros(max_key + 1, dtype=gt.dtype)
    for k, v in mapping.items():
        lut[k] = v
    return lut[gt]

def merge_if_overlapping(a, b):
    do_intersect = check_overlap_bounding_boxes(a, b)
    if do_intersect:
        x_min = np.min([a[1], b[1]])
        y_min = np.min([a[0], b[0]])
        x_max = np.max([a[1] + a[3], b[1] + b[3]])
        y_max = np.max([a[0] + a[2], b[0] + b[2]])
        new_bbox = (y_min, x_min, y_max - y_min, x_max - x_min)
        return True, new_bbox

    return False, []

def check_overlap_bounding_boxes(a, b):
    bottom = np.max([a[0], b[0]])
    top = np.min([a[0] + a[2], b[0] + b[2]])
    left = np.max([a[1], b[1]])
    right = np.min([a[1] + a[3], b[1] + b[3]])
    do_intersect = bottom < top and left < right
    return do_intersect

def get_gt_bounding_boxes(ground_truth_wsi, spacing, overview_spacing=8.0):
    """
    Bounding boxes will have form of (y,x,height,width)
    :param ground_truth_wsi:
    :param spacing:
    :param overview_spacing:
    :return:
    """
    overview_level = ground_truth_wsi.level(spacing=overview_spacing)
    overview_shape = ground_truth_wsi.shapes[overview_level]
    gt_patch = ground_truth_wsi.read(overview_spacing, 0, 0, overview_shape[0], overview_shape[1]).squeeze()

    spacing_ratio = overview_spacing / spacing
    padding = 1
    labels, n = label(gt_patch > 0, return_num=True, connectivity=1)
    regions = regionprops(labels)
    bboxes = [(int((region.bbox[0] - padding) * spacing_ratio),
               int((region.bbox[1] - padding) * spacing_ratio),
               int((region.bbox[2] - region.bbox[0] + padding * 2) * spacing_ratio),
               int((region.bbox[3] - region.bbox[1] + padding * 2) * spacing_ratio)) for region in regions]

    # remove overlapping bounding boxes
    merge_overlapping_bboxes(bboxes)

    return bboxes, len(bboxes)

def merge_overlapping_bboxes(bboxes):
    candidate_count = 0
    while candidate_count < len(bboxes):
        candidate_count += 1
        overlap = False
        candidate_box = bboxes.pop(0)
        for index, compare_box in enumerate(bboxes):
            overlapping, new_bbox = merge_if_overlapping(candidate_box, compare_box)
            if overlapping:
                overlap = True
                candidate_count = 0
                bboxes.pop(index)
                bboxes.append(new_bbox)
                break
        if not overlap:
            bboxes.append(candidate_box)

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, str, str):
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Calculate DICE and JACCARD scores for given image paths')

    argument_parser.add_argument('-i', '--input_mask_path', required=True, type=str, help='segmentation mask path')
    argument_parser.add_argument('-g', '--ground_truth_path', required=True, type=str, help='ground truth mask path')
    argument_parser.add_argument('-c', '--classes', required=True, type=str,
                                 help='classes to check, will be converted to dict')
    argument_parser.add_argument('-o', '--output_path', required=True, type=str, help='output_path')
    argument_parser.add_argument('-s', '--spacing', required=True, type=float,
                                 help='spacing at which to sample mask and gt')
    argument_parser.add_argument('-m', '--mapping', required=False, type=str, default=None,
                                 help='optional class mapping, will be converted to dict')
    argument_parser.add_argument('-p', '--plot_images', action='store_true', default=False, help='plot result images')

    # extras
    argument_parser.add_argument('-w', '--wsi_path', required=False, type=str, help='wsi dir path')
    argument_parser.add_argument('-a', '--all_cm', action='store_true', default=False, help='save all individual CMs')
    argument_parser.add_argument('-d', '--data_config_path', required=False, default=None, type=str,
                                 help='data config path')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #

    arguments['classes'] = ast.literal_eval(arguments['classes'])
    if arguments['mapping'] is not None:
        arguments['mapping'] = ast.literal_eval(arguments['mapping'])

    print(argument_parser.description)
    print('Input mask path: {path}'.format(path=arguments['input_mask_path']))
    print('Ground truth path: {path}'.format(path=arguments['ground_truth_path']))
    print('Classes included: {cls}'.format(cls=arguments['classes']))
    print('GT to segmentation mask mapping: {mapping}'.format(mapping=arguments['mapping']))
    print('Output path: {path}'.format(path=arguments['output_path']))
    print('Plot images: {flag}'.format(flag=arguments['plot_images']))
    print('Spacing: {spacing}'.format(spacing=arguments['spacing']))

    # extras
    print('WSI path: {path}'.format(path=arguments['wsi_path']))
    print('Save all CMs: {cm}'.format(cm=arguments['all_cm']))
    print('Data config path: {path}'.format(path=arguments['data_config_path']))

    # Return parsed values.
    #
    return arguments

def map_classes(classes, mapping):
    new_classes = {}
    for k, v in classes.items():
        if v in mapping.keys():
            mapped_class = mapping[v]
            if mapped_class in new_classes.values():
                continue
            new_classes[k] = mapped_class
        else:
            new_classes[k] = v
    return new_classes

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig_width = max(len(classes) * 0.5 + 1.0, 10.0)

    fig = plt.figure(figsize=(fig_width, fig_width))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.5 / fig_width, right=1 - 0.5 / fig_width, top=0.94, bottom=0.1)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    return ax

def get_scores_cm(cm):
    tp = np.diag(cm)
    fn = np.sum(cm, axis=1) - tp
    fp = np.sum(cm, axis=0) - tp

    f1 = (2 * tp) / (2 * tp + fn + fp)
    f1[np.isnan(f1)] = 0

    return f1

def make_legend(classes, color_map, output_path):
    figure = plt.figure()

    class_names = list(classes.keys())
    class_numbers = list(classes.values())

    legend_handles = []
    for i, class_number in enumerate(class_numbers):
        legend_handles.append(mpatches.Patch(color=color_map[class_number]/255, label=class_names[i]))

    figure.legend(handles=legend_handles, loc='center')
    plt.tight_layout()
    plt.savefig(output_path)

if __name__ == '__main__':
    # arguments = {'input_mask_path': '/home/ludova/master_internship_diag/chansey_pathology/projects/projects/pathology-wilms-tumor-kidney/result_masks/densenet/N7_L15_412412/test_set/WT_S01_P000063_C0001_L06_A15_V10.tif',
    #              'ground_truth_path': '/home/ludova/master_internship_diag/chansey_pathology/archives/kidney/Wilmstumor_retrospective_prinsesmaximacentrum/annotation_masks/WT_S01_P000063_C0001_L06_A15_V10.tif',
    #              'classes': "{'Achtergrond': 1, 'WT-blasteem': 2, 'WT-stroma': 3, 'WT-epitheel': 4, 'Necrose': 5, 'Bloeding': 6, 'Regressie': 8, 'Fibrose': 7, 'Glomeruli': 9, 'Tubuli': 10, 'Vet': 11, 'Bindweefsel': 12, 'Bloedvaten': 13, 'Zenuwtakken': 14, 'Lymfklier': 15, 'Urotheel': 18, 'Nefrogene rest': 20}",
    #              'output_path': '/home/ludova/master_internship_diag/chansey_pathology/projects/pathology-wilms-tumor-kidney/dice_scores/test',
    #              'spacing': 0.5,
    #              'mapping': "{1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 6, 9: 7, 10: 8, 11: 9, 12: 10, 13: 11, 14: 12, 15: 13, 18: 14, 20: 15}",
    #              'plot_images': True
    #             }
    arguments = collect_arguments()

    mask_path = arguments['input_mask_path']
    ground_truth_path = arguments['ground_truth_path']
    classes = arguments['classes']
    mapping = arguments['mapping']
    output_path = arguments['output_path']
    plot_flag = arguments['plot_images']
    spacing = arguments['spacing']

    # extras
    wsi_path = arguments['wsi_path']
    save_indiv_cms = arguments['all_cm']
    dc_path = arguments['data_config_path']
    output_dir = os.path.dirname(output_path)
    slide_cms_dir = os.path.join(output_dir, 'slide_cms')
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(slide_cms_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    counter = 0

    job_list = assemble_jobs(mask_path, ground_truth_path, dc_path)
    console.print(f"[bold]Jobs found:[/bold] {len(job_list)}")

    results = {}
    results['final_overall_scores'] = {}
    for annotation_class in classes.keys():
        results['final_overall_scores'][annotation_class] = {}
        results['final_overall_scores'][annotation_class]['f1'] = []
        results['final_overall_scores'][annotation_class]['jaccard'] = []

    if mapping is not None:
        print("\nmapping classses: {}".format(classes))
        classes = map_classes(classes, mapping)
        print("mapped classes: {}\n".format(classes))
    classes_by_number = {v: k for k, v in classes.items()}
    mapping_set = list(set(mapping.values()))

    # Go though all files
    #
    color_map = get_asap_colors()
    full_cm = np.zeros((len(mapping_set), len(mapping_set)))

    # Plot and save legend
    #
    legend_path = os.path.join(output_dir, "legend.png")
    make_legend(classes, color_map, output_path=legend_path)

    _progress_cols = [
        SpinnerColumn(),
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("slide=[cyan]{task.fields[slide]}[/cyan]"),
        TimeElapsedColumn(),
    ]

    with Progress(*_progress_cols, console=console) as progress:
        job_task = progress.add_task("Evaluating", total=len(job_list), slide="—")

        for job_item in job_list:

            mask_filepath = job_item[0]
            ground_truth_filepath = job_item[1]
            file_name = os.path.splitext(os.path.basename(mask_filepath))[0]
            progress.update(job_task, slide=file_name)
            results[file_name] = {}

            mask_wsi = dptimagereader.ImageReader(image_path=mask_filepath)
            ground_truth_wsi = dptimagereader.ImageReader(image_path=ground_truth_filepath)
            bboxes, n = get_gt_bounding_boxes(ground_truth_wsi, spacing)

            if wsi_path:
                wsi = dptimagereader.ImageReader(image_path=wsi_path)

            for annotation_class in classes.keys():
                results[file_name][annotation_class] = {}
                results[file_name][annotation_class]['f1'] = []
                results[file_name][annotation_class]['jaccard'] = []
                results[file_name][annotation_class]['weights'] = []

            slide_cm = np.zeros((len(mapping_set), len(mapping_set)))

            for bbox in bboxes:
                mask_patch = mask_wsi.read(spacing, bbox[0], bbox[1], bbox[2], bbox[3]).squeeze()
                ground_truth_patch = ground_truth_wsi.read(spacing, bbox[0], bbox[1], bbox[2], bbox[3]).squeeze()

                if wsi_path:
                    wsi_patch = wsi.read(spacing, bbox[0], bbox[1], bbox[2], bbox[3]).squeeze()[:-1]

                if mapping is not None:
                    ground_truth_patch = map_gt_to_segmentation(ground_truth_patch, mapping)
                ground_truth_patch *= (mask_patch > 0)
                mask_patch *= (ground_truth_patch > 0)

                if plot_flag:
                    counter += 1
                    mask_patch_labels, gt_patch_labels = [], []
                    for class_idx in range(len(mapping_set)):
                        if (mask_patch == class_idx).any():
                            mask_patch_labels.append(class_idx)
                        if (ground_truth_patch == class_idx).any():
                            gt_patch_labels.append(class_idx)
                    plt.subplot(131)
                    plt.title('mask\nlabels: {}'.format(mask_patch_labels))
                    plt.imshow(color_map[mask_patch])
                    plt.subplot(132)
                    plt.title('gt\nlabels: {}'.format(gt_patch_labels))
                    plt.imshow(color_map[ground_truth_patch])
                    if wsi_path:
                        plt.subplot(133)
                        plt.imshow(wsi_patch)
                    plt.savefig(os.path.join(plots_dir, f'image{counter}.png'))
                    plt.clf()
                    if counter == 50:
                        break

                if len(ground_truth_patch[ground_truth_patch != 0]) == 0:
                    continue

                cm = confusion_matrix(y_true=ground_truth_patch[ground_truth_patch != 0],
                                      y_pred=mask_patch[ground_truth_patch != 0],
                                      labels=mapping_set)
                slide_cm += cm
                full_cm += cm

                f1 = get_scores_cm(cm)
                for c in classes.values():
                    results[file_name][classes_by_number[c]]['f1'].append(float(f1[c - 1]))

            f1_slide = get_scores_cm(slide_cm)
            for c in classes.values():
                results[file_name][classes_by_number[c]]['f1_overall'] = float(f1_slide[c - 1])

            if save_indiv_cms:
                with open(os.path.join(slide_cms_dir, f'cm_{file_name}.p'), 'wb') as f:
                    pickle.dump(slide_cm, f)

                idxs = np.asarray(sorted(classes.values())) - 1
                plot_slide_cm = slide_cm[idxs, :][:, idxs]
                plot_confusion_matrix(plot_slide_cm, [classes_by_number[x] for x in sorted(classes.values())], normalize=True)
                plt.savefig(os.path.join(slide_cms_dir, f'cm_{file_name}.png'))
                plt.clf()

            progress.update(job_task, advance=1)

    f1_full = get_scores_cm(full_cm)
    for c in classes.values():
        results['final_overall_scores'][classes_by_number[c]]['f1'].append(float(f1_full[c - 1]))

    idxs = np.asarray(sorted(classes.values())) - 1
    plot_cm = full_cm[idxs, :][:, idxs]
    tp = np.sum(np.diag(plot_cm))
    f = np.sum(plot_cm) - tp
    results['final_overall_scores']['all_classes'] = {}
    results['final_overall_scores']['all_classes']['f1'] = float(2 * tp / (2 * (tp + f)))

    with open(output_path + '.p', 'wb') as pkl_f:
        pickle.dump(plot_cm, pkl_f)

    plot_confusion_matrix(plot_cm, [classes_by_number[x] for x in sorted(classes.values())], normalize=True)
    cm_path = os.path.join(output_dir, "{}.png".format(os.path.basename(output_path).split(".")[0]))
    plt.savefig(cm_path)
    plt.clf()

    with open(output_path, 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False, indent=4)

    # ── Summary table ─────────────────────────────────────────────────────
    table = Table(title="Evaluation results", show_lines=True)
    table.add_column("Class", style="bold")
    table.add_column("F1 (overall)", justify="right")

    sorted_class_names = [classes_by_number[x] for x in sorted(classes.values())]
    for name, f1_val in zip(sorted_class_names, f1_full[idxs]):
        table.add_row(name, f"[cyan]{f1_val:.4f}[/cyan]")
    table.add_row(
        "[bold]All classes[/bold]",
        f"[bold green]{results['final_overall_scores']['all_classes']['f1']:.4f}[/bold green]",
    )
    console.print(table)
    console.print(f"Results saved to [bold]{output_path}[/bold]")