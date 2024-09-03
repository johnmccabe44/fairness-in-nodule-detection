import argparse
from ast import List
import datetime
from functools import partial
import logging
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import scipy.ndimage
import SimpleITK as sitk
from skimage import measure

def parse_arguments():

    parser = argparse.ArgumentParser(description='Generate lung mask for SUMMIT dataset')

    parser.add_argument(
        '--scan-ids',
        nargs='+',
        type=Path,
        help='List of scan ids'
    )

    parser.add_argument(
        '--scans-path', 
        type=Path,
        help='Path to the case'
    )

    parser.add_argument(
        '--segmentation-path',
        type=Path,
        help='Path to the output mask'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=-1,
        help='Batch size'
    )

    parser.add_argument(
        '--batch-number',
        type=int,
        default=1,
        help='Batch number'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of workers to use'
    )

    return parser.parse_args()

def process_logs(log_list):

    for log in log_list:
        logging.log(log[0], f"{log[1]}, {log[2]}")

def load_metaio_scan(path):
    """
    Loads the scan from raw. Keeps all properties as part of the slices. 
    """

    # unique identifier can be found from file name
    scan_uid = os.path.basename(path).split('.')[0]

    # read in the scan
    metadata = sitk.ReadImage(path)
    image = np.array(sitk.GetArrayFromImage(metadata), dtype=np.int16)
    # Pull out the salient bits of info needed
    origin = np.array(metadata.GetOrigin(), dtype=np.float32)[::-1]
    voxel_size = np.array(metadata.GetSpacing(),dtype=np.float32)[::-1]
    orientation = np.array(metadata.GetDirection(),dtype=np.float32)

    return image, metadata

def binarize_per_slice(image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10):
    bw = np.zeros(image.shape, dtype=bool)
    
    # prepare a mask, with all corner values set to nan
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size/2+0.5, image_size/2-0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x**2+y**2)**0.5
    nan_mask = (d<image_size/2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma, truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.gaussian_filter(image[i].astype('float32'), sigma, truncate=2.0) < intensity_th
        
        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.isin(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw
        
    return bw

def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1-cut_num, 0, 0], label[-1-cut_num, 0, -1], label[-1-cut_num, -1, 0], label[-1-cut_num, -1, -1], \
                    label[0, 0, mid], label[0, -1, mid], label[-1-cut_num, 0, mid], label[-1-cut_num, -1, mid]])
    for l in bg_label:
        label[label == l] = 0
        
    # select components based on volume
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0
            
    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x**2+y**2)**0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))
        
        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)
            
    bw = np.isin(label, list(valid_label)).reshape(label.shape)
    
    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label==l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.isin(label3, list(valid_l3)).reshape(label3.shape)
    
    return bw, len(valid_label)

def fill_hole(bw):
    # fill 3d holes
    label = measure.label(~bw)
    # idendify corner components
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0], label[-1, -1, -1]])
    bw = ~np.isin(label, list(bg_label)).reshape(label.shape)
    
    return bw

def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):    
    def extract_main(bw, cover=0.95):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area)*cover:
                sum = sum+area[count]
                count = count+1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
            bw[i] = bw[i] & filter
           
        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label==properties[0].label

        return bw
    
    def fill_2d_hole(bw):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
            bw[i] = current_slice

        return bw
    
    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area/properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1
    
    if found_flag:
        d1 = scipy.ndimage.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)
                
        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)
        
    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')
        
    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw

def generate_lung_mask(idx, scan_paths, segmentation_path):
    try:
        case_path = scan_paths[idx]

        print(f'Processing {case_path}')

        image, metadata = load_metaio_scan(case_path)
        spacing_array = np.array(metadata.GetSpacing())
        bw = binarize_per_slice(image, spacing_array)
        flag = 0
        cut_num = 0
        cut_step = 2
        bw0 = np.copy(bw)
        while flag == 0 and cut_num < bw.shape[0]:
            bw = np.copy(bw0)
            bw, flag = all_slice_analysis(bw, spacing_array, cut_num=cut_num, vol_limit=[0.68,7.5])
            cut_num = cut_num + cut_step

        bw = fill_hole(bw)
        bw1, bw2, bw = two_lung_only(bw, spacing_array)

        bw1 = bw1 * 3
        bw2 = bw2 * 4
        bw = bw1 + bw2
        
        # Save bw as mhd format with original image config
        output_image = sitk.GetImageFromArray(bw.astype(np.uint8))
        output_image.SetSpacing(metadata.GetSpacing())
        output_image.SetOrigin(metadata.GetOrigin())
        output_image.SetDirection(metadata.GetDirection())
        sitk.WriteImage(output_image, str(segmentation_path / f'{case_path.stem}.mhd'))

        return (logging.INFO, datetime.datetime.now(), f'Processed {case_path}')

    except Exception as e:
        return (logging.ERROR, datetime.datetime.now(), f'Error processing {case_path}: {e}')

def main():


    args = parse_arguments()

    scan_ids = pd.concat([
        pd.read_csv(scan_id, usecols=['scan_id'])
        for scan_id in args.scan_ids
    ])['scan_id'].tolist()

    scans_path = args.scans_path
    scan_paths = [
        scan_path 
        for scan_path in scans_path.rglob('**/*.mhd') 
        if scan_path.stem in scan_ids
    ]
                  
    if args.batch_size == -1:
        args.batch_size = len(scan_paths)

    scan_paths = scan_paths[args.batch_size * (args.batch_number - 1):args.batch_size * args.batch_number]

    logging.basicConfig(filename=f'logs/generate_lung_mask_{datetime.datetime.now()}_{args.batch_number}.log', level=logging.INFO)

    print(f'Processing {len(scan_paths)} scans')
    print(f'Scan paths {scan_paths}')

    segmentation_path = args.segmentation_path

    # Create output directory if not exist
    segmentation_path.mkdir(parents=True, exist_ok=True)

    N = len(scan_paths)

    if args.workers > 1:

        partial_generate_lung_mask = partial(generate_lung_mask, scan_paths=scan_paths, segmentation_path=segmentation_path)
        with Pool(args.workers) as p:
            logs = p.map(partial_generate_lung_mask, range(N))

    else:            
            logs = []
            for idx in range(N):
                print(f'idx:{idx}')                

                logs.append(generate_lung_mask(idx, scan_paths, segmentation_path))

    process_logs(logs)


if __name__ == '__main__':
    main()
