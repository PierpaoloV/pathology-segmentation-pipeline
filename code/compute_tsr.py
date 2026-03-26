import argparse
import json
import ntpath
import os

import cv2
import matplotlib.pyplot as plt
import multiresolutionimageinterface as mir
import numpy as np
import scipy.ndimage.interpolation as sni
import scipy.ndimage.morphology
import scipy.signal
import skimage.morphology
from skimage.measure import label, regionprops
from tqdm import tqdm


class TumorStromaRatioHeatMapper:

    def __init__(self, parameters):

        # assign class variables
        self.parameters = parameters
        self.img_filename = parameters['img_filename']
        self.tissue_mask_filename = parameters['tissue_mask_filename']
        self.bulk_mask_filename = parameters['bulk_mask_filename']
        self.classification_map_filename = parameters['classification_map_filename']
        self.level = parameters['level']
        self.circle_diameter_mm = parameters['circle_diameter_mm']
        self.percentage_tumor_stroma_in_circle = parameters['percentage_tumor_stroma_in_circle']
        self.output_dir = parameters['output_dir']
        self.save_fig = parameters['save_fig']
        self.labels_file = None
        self.circle_in_bulk = parameters['circle_in_bulk']
        self.number_of_hotspots = 10
        self.img = None
        self.tissue_mask = None
        self.bulk_mask = None
        self.bulk_mask_np = None
        self.classification_map = None
        self.ratio_msk_img = None
        self.ratio_map_img = None
        self.ratio_bulk_img = None
        self.pixel_size_level_0 = None
        self.pixel_size_selected_level = None
        self.circle_diameter_px = None
        self.circle_mask_px = None
        self.circle_mask_px_flat = None
        self.circle_area = None
        self.img_np = None
        self.tsr_heatmap = None
        self.tissue_mask_np = None
        self.label_dict = None

        # run initialization functions
        self.init_labels()
        self.load_wsis()
        self.compute_ratios()
        self.tissue_mask_level = self.tissue_mask.getBestLevelForDownSample(self.ratio_msk_img)
        if self.bulk_mask:
            self.bulk_mask_level = self.bulk_mask.getBestLevelForDownSample(self.ratio_bulk_img)
        self.classification_map_level = self.classification_map.getBestLevelForDownSample(self.ratio_map_img)
        self.compute_circle_diameter_px()
        self.build_circle_mask()
        self.extract_numpy_arrays()
        self.print_log()

    def init_labels(self):
        """Initialize label dictionary."""
        with open(self.labels_file) as f:
            self.label_dict = json.load(f)

    def build_circle_mask(self):
        """Build the circle mask to simulate hot-spot and microscope area."""
        radius = self.circle_diameter_px // 2
        xv, yv = np.meshgrid(np.linspace(-radius, radius, 2 * radius + 1),
                              np.linspace(-radius, radius, 2 * radius + 1))
        distance = np.sqrt(xv ** 2 + yv ** 2)
        self.circle_mask_px = (distance < radius).astype('int')
        self.circle_mask_px_flat = self.circle_mask_px.flatten()
        self.circle_area = np.sum(self.circle_mask_px_flat)

    def print_log(self):
        """Print log of class initialization."""
        print('')

    def load_wsis(self):
        """Load all input whole-slide images."""
        wsireader = mir.MultiResolutionImageReader()
        self.img = wsireader.open(self.img_filename)
        self.tissue_mask = wsireader.open(self.tissue_mask_filename)
        if self.bulk_mask_filename is not None:
            print('Opening: {}'.format(self.bulk_mask_filename))
            self.bulk_mask = wsireader.open(self.bulk_mask_filename)
            assert self.bulk_mask, "Bulk mask is not valid"
        if os.path.isfile(self.classification_map_filename):
            self.classification_map = wsireader.open(self.classification_map_filename)
        else:
            print('This file doesnt exist')
        assert self.img, 'Input image is not valid'
        assert self.tissue_mask, 'Tissue mask is not valid'
        assert self.classification_map, 'Classification map is not valid'

    def compute_ratios(self):
        """Compute ratio between image and tissue mask / likelihood map."""
        img_dims = self.img.getLevelDimensions(self.level)
        tissue_mask_dims = self.tissue_mask.getDimensions()
        classification_map_dims = self.classification_map.getDimensions()
        if self.bulk_mask:
            bulk_mask_dims = self.bulk_mask.getDimensions()
            self.ratio_bulk_img = bulk_mask_dims[0] // img_dims[0]
        self.ratio_msk_img = tissue_mask_dims[0] // img_dims[0]
        self.ratio_map_img = classification_map_dims[0] // img_dims[0]

    def extract_numpy_arrays(self):
        """Extract WSI, tissue mask and classification mask as Numpy arrays."""
        dims_map_selected_level = self.classification_map.getLevelDimensions(self.classification_map_level)
        self.classification_map_np = self.classification_map.getUCharPatch(
            0, 0, dims_map_selected_level[0], dims_map_selected_level[1], self.classification_map_level).squeeze()
        tissue_mask_dims_at_selected_level = self.tissue_mask.getLevelDimensions(self.tissue_mask_level)
        self.tissue_mask_np = self.tissue_mask.getUCharPatch(
            0, 0, tissue_mask_dims_at_selected_level[0], tissue_mask_dims_at_selected_level[1], self.tissue_mask_level).squeeze()
        if self.bulk_mask:
            bulk_mask_dims_at_selected_level = self.bulk_mask.getLevelDimensions(self.bulk_mask_level)
            self.bulk_mask_np = self.bulk_mask.getUCharPatch(
                0, 0, bulk_mask_dims_at_selected_level[0], bulk_mask_dims_at_selected_level[1], self.bulk_mask_level).squeeze()

    def get_top_hottest_regions(self, output_path):
        max_values = list()
        nonmaxsur = np.copy(self.hotspot_array)
        hottest_spots_array = np.zeros(self.hotspot_array.shape, dtype=np.uint8)

        count = 1
        while len(max_values) < self.number_of_hotspots:
            if np.max(nonmaxsur) == 0:
                break

            tmp_y, tmp_x = np.unravel_index(np.argmax(nonmaxsur), nonmaxsur.shape)
            max_value = nonmaxsur[tmp_y, tmp_x]

            nonmaxsur = cv2.circle(nonmaxsur, (tmp_x, tmp_y), int(self.radius_in_pixels), 0, -1)
            hottest_spots_array = cv2.circle(hottest_spots_array, (tmp_x, tmp_y), int(self.radius_in_pixels), count, -1)

            max_values.append([max_value, tmp_x, tmp_y])
            count += 1

        self.max_values = max_values
        self.write_output_tif(output_path, hottest_spots_array)

    def compute_circle_diameter_px(self):
        """Compute the circle size in px for the given settings."""
        self.pixel_size_level_0 = self.img.getSpacing()[0]  # um/px
        self.pixel_size_selected_level = (self.pixel_size_level_0
                                          * self.img.getDimensions()[0]
                                          / self.img.getLevelDimensions(self.level)[0])
        self.circle_diameter_px = int(1000. * self.circle_diameter_mm / self.pixel_size_selected_level)

    def compute_tsr_wsi(self):
        """Compute TSR heatmap for the entire image."""
        tumor_map = np.zeros(self.classification_map_np.shape, dtype=np.uint8)
        tumor_map[self.classification_map_np == 8] = 1

        stroma_map = np.zeros(self.classification_map_np.shape, dtype=np.uint8)
        stroma_map[self.classification_map_np == 1] = 1

        stroma_in_tumour = stroma_map * self.bulk_mask_np
        tumor_in_bulk = tumor_map * self.bulk_mask_np

        tsr = np.sum(stroma_in_tumour) / (np.sum(stroma_in_tumour) + np.sum(tumor_in_bulk))

        series = os.path.splitext(ntpath.basename(self.img_filename))[0]
        print('TSR  = {}'.format(tsr))

        output_csv_file = os.path.join(self.output_dir, '{}_TSR.csv'.format(series))
        with open(output_csv_file, 'w') as f:
            f.write('case_id,tsr_ratio\n')
            f.write('{case_id},{tsr_ratio}'.format(case_id=series, tsr_ratio=tsr))


def collect_input_arguments():
    """Collect input parameters."""
    argument_parser = argparse.ArgumentParser(description='Compute TSR map as hot-spot level.')

    argument_parser.add_argument('-i', '--img_filename', required=True, type=str, help='input image filename')
    argument_parser.add_argument('-o', '--output_dir', required=True, type=str, help='output directory for TSR map (and optionally png) file')
    argument_parser.add_argument('-m', '--classification_map_filename', required=True, type=str, help='likelihood map file')
    argument_parser.add_argument('-t', '--tissue_mask_filename', required=True, type=str, help='tissue mask file')
    argument_parser.add_argument('-l', '--labels_file', required=False, type=str, help='json file containing the dictionary with the definition of labels')
    argument_parser.add_argument('-il', '--level', required=False, default=4, type=int, help='level at which the image is processed (default: 6)')
    argument_parser.add_argument('-p', '--percentage_tumor_stroma_in_circle', required=False, default=0.6, type=float, help='minimum percentage of tumor + stroma in circle (default: 0.8)')
    argument_parser.add_argument('-d', '--circle_diameter_mm', required=False, default=2.2, type=float, help='diameter (in mm) used to simulate the microscopy')
    argument_parser.add_argument('--save_fig', dest='save_fig', action='store_true', help='save snapshot (boolean)')
    argument_parser.add_argument('-bm', '--bulk_mask_filename', required=True, default=None, type=str, help='tumor bulk mask file')
    argument_parser.add_argument('--circle_in_bulk', dest='circle_in_bulk', action='store_true', help='only consider circles completely enclosed in the bulk')

    return vars(argument_parser.parse_args())


if __name__ == "__main__":
    arguments = collect_input_arguments()

    output_dir = arguments['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tsrhm = TumorStromaRatioHeatMapper(arguments)
    tsrhm.compute_tsr_wsi()
