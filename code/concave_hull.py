import argparse
import glob
import math
import os
import xml.etree.ElementTree as ET
from distutils import extension

import multiresolutionimageinterface as mir
import numpy as np
import shapely
import shapely.geometry as geometry
import skimage.morphology
from scipy.spatial import Delaunay
from shapely.ops import polygonize, unary_union


def alpha_shape(points, alpha):
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_points.append([coords[i], coords[j]])

    coords = [(i[0], i[1]) if type(i) or tuple else i for i in points]
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    # for ia, ib, ic in tri.vertices:
    for ia, ib, ic in tri.simplices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        # Semiperimeter of triangle
        s = (a + b + c) / 2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        # Here's the radius filter.
        # print circum_r
        if circum_r < 1 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return unary_union(triangles), edge_points
    # return cascaded_union(triangles), edge_points


# concave_hull, edge_points = alpha_shape(points,
#                                         alpha=0.3)

def set_coordinate_asap(coords_xml, order, x, y):
    coord_xml = ET.SubElement(coords_xml, 'Coordinate')
    coord_xml.set('Order', str(order))
    coord_xml.set('X', str(x))
    coord_xml.set('Y', str(y))


def create_asap_xml_from_coords(coords):
    root = ET.Element('ASAP_Annotations')
    annot_xml = ET.SubElement(root, 'Annotations')
    for j, coord_set in enumerate(coords):
        annot = ET.SubElement(annot_xml, 'Annotation')
        annot.set('Name', 'Annotation {}'.format(j))
        annot.set('Type', 'Polygon')
        annot.set('PartOfGroup', 'Region')
        annot.set('Color', '#F4FA58')
        coords_xml = ET.SubElement(annot, 'Coordinates')
        for i, point in enumerate(coord_set):
            set_coordinate_asap(coords_xml, i, point[1], point[0])
    groups_xml = ET.SubElement(root, 'AnnotationGroups')
    group_xml = ET.SubElement(groups_xml, 'Group')
    group_xml.set('Name', 'Region')
    group_xml.set('PartOfGroup', 'None')
    group_xml.set('Color', '#00ff00')
    ET.SubElement(group_xml, 'Attributes')
    return ET.ElementTree(root)


def concave_hull(input_file, output_dir, input_level, output_level, level_offset, alpha, min_size):
    wsi = mir.MultiResolutionImageReader().open(input_file)
    wsi_dim = wsi.getLevelDimensions(input_level)
    wsi_patch = wsi.getUCharPatch(0, 0, wsi_dim[0], wsi_dim[1], input_level).squeeze()
    # print(f'Unique values in mask: {np.unique(wsi_patch)}')
    wsi_patch = skimage.morphology.remove_small_objects((wsi_patch), min_size=min_size, connectivity=2)
    print('wsi_patch.shape', wsi_patch.shape)
    points = np.argwhere(wsi_patch == 3)
    # print('points', points)

    print("calculating concave hull, this might take a while..")
    concave_hull, edge_points = alpha_shape(points, alpha=alpha)
#Commented
    # if isinstance(concave_hull, shapely.geometry.polygon.Polygon) or isinstance(concave_hull, shapely.geometry.GeometryCollection):
    #     polygons = [concave_hull]
    # else:
    #     polygons = list(concave_hull)

    # coordinates = []
    # for polygon in polygons:
    #     coordinates.append([[x[0] * 2 ** (input_level + level_offset - output_level),
    #                          x[1] * 2 ** (input_level + level_offset - output_level)] for x in polygon.boundary.coords[:-1]])
#commented
    if isinstance(concave_hull, shapely.geometry.polygon.Polygon):
        polygons = [concave_hull]
    elif isinstance(concave_hull, shapely.geometry.MultiPolygon):
        polygons = concave_hull.geoms
    else:
        raise ValueError("Unexpected geometry type: {}".format(type(concave_hull)))

    coordinates = []
    for polygon in polygons:
        if isinstance(polygon, shapely.geometry.polygon.Polygon):
            polygon = [polygon]
        coordinates.extend([[[x[0] * 2 ** (input_level + level_offset - output_level),
                              x[1] * 2 ** (input_level + level_offset - output_level)] for x in poly.boundary.coords[:-1]] for poly in polygon])
    asap_annot = create_asap_xml_from_coords(coordinates)

    output_filename = os.path.basename(input_file).split('.')[0]
    asap_annot.write(os.path.join(output_dir, output_filename + ".xml"))


def collect_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_path', dest='input_path', required=True, help='input path expression')
    parser.add_argument('--output_dir', dest='output_dir', required=True, help='mask output directory')
    parser.add_argument('--cls', dest='cls', required=True, help='class to create concave hull around')
    parser.add_argument('--alpha', dest='alpha', default=0.1, type=float, help='alpha value determines max radius')
    parser.add_argument('--min_size', dest='min_size', default=0, type=int, help='minimum size of class blobs in mask')
    parser.add_argument('--input_level', dest='input_level', default=6, type=int, help='input level of mask')
    parser.add_argument('--output_level', dest='output_level', default=0, type=int, help='output level of asap annotations')
    parser.add_argument('--level_offset', dest='level_offset', default=0, type=int, help='difference between mask and original wsi')

    return parser.parse_args()


if __name__ == "__main__":
    args = collect_arguments()
    # files = glob.glob(args.input_path)
    # print(f'{files}')
    files = [args.input_path + '/' + a for a in os.listdir(args.input_path) if '.tif' in a]
    print(f'There are {len(files)} files to process')
    for f in files:
        ext = f.split('.')[-1]
        name = f.replace(ext,'.xml')
        print("working on {}..".format(f))
        opt_name = os.path.join(args.output_dir,name)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        # concave_hull(f, args.output_dir, args.input_level, args.output_level, args.level_offset, args.alpha, args.min_size)
        if not os.path.exists(opt_name):
            try:
                concave_hull(f, args.output_dir, args.input_level, args.output_level, args.level_offset, args.alpha, args.min_size)
            except Exception as e:
                print("Error processing file {}..".format(f))
                print(e)


""" EXAMPLE CONFIG
--input_path
"/mnt/synology/pathology/../*.tif"
--output_dir
"/tmp"
--cls
1
--alpha
0.07
--min_size
10
--input_level
6
--output_level
0
--level_offset
1
"""






