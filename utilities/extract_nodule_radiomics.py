import argparse
from datetime import datetime
from functools import partial
import random
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from radiomics import featureextractor
import sys

def parse_args():

    parser = argparse.ArgumentParser(description='Extract radiomic features from nodule images')

    parser.add_argument(
        '--image-path',
        type=str,
        help='Path to the directory containing the nodule images',
        default='/Users/john/Projects/SOTAEvaluationNoduleDetection/data/summit/scans'
    )

    parser.add_argument(
        '--nodule-metadata',
        type=str,
        help='Path to the CSV file containing nodule metadata',
        default='/Users/john/Projects/SOTAEvaluationNoduleDetection/data/summit/data/nodule_data.csv'
    )

    parser.add_argument(
        '--workers',
        type=int,
        help='Number of workers to use for parallel processing',
        default=2
    )

    parser.add_argument(
        '--throttle',
        type=int,
        help='Number of images to process',
        default=5
    )

    parser.add_argument(
        '--segment-or-crop',
        type=str,
        help='Whether to segment or crop the nodule',
        choices=['segment', 'crop'],
        default='segment'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        help='how many to include in the batch',
        default=3
    )

    parser.add_argument(
        '--batch-number',
        type=int,
        help='which batch to process',
        default=2
    )

    return parser.parse_args()

def is_debug_mode():
    """
    Check if the Python code is running in debug mode in Visual Studio Code.
    """
    try:
        # Check if the debugger is attached
        return sys.gettrace() is not None
    except AttributeError:
        # sys.gettrace() is not available outside of the CPython interpreter
        return False

def save_image_with_mask_as_png(image, coordinates, diameter, output_path):
    # Apply the mask to the image
    image_array = sitk.GetArrayFromImage(image)
    
    # Calculate the coordinates of the bounding box
    x, y, z = coordinates
    r = (diameter / 2) * 1.5 # Increase the radius to ensure the entire nodule is included
    x_start = max(0, int(x - r))
    x_end = min(image_array.shape[2], int(x + r))
    y_start = max(0, int(y - r))
    y_end = min(image_array.shape[1], int(y + r))
    z_start = max(0, int(z - r))
    z_end = min(image_array.shape[0], int(z + r))
    z_mid = int((z_start + z_end) / 2)
    
    # Draw the bounding box on the image
    plt.imshow(image_array[z_mid,:,:], cmap='gray')
    plt.gca().add_patch(plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, 
                                      edgecolor='r', facecolor='none'))
    
    # Save the image with the bounding box as a PNG
    plt.savefig(output_path)
    plt.close()

def save_cropped_masked_image_as_png(image, output_path):
    # Apply the mask to the image
    image_array = sitk.GetArrayFromImage(image)

    # Save the masked image as a PNG
    plt.imsave(output_path, image_array[int(image_array.shape[0]//2),:,:], cmap='gray')

def flatten_with_position(lst):
    def _flatten(lst, position):
        for i, item in enumerate(lst):
            if isinstance(item, list):
                yield from _flatten(item, position + (i,))
            else:
                yield item, position + (i,)

    return _flatten(lst, ())

# Load the image using SimpleITK
def load_image(image_path):
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    return image, image_array

# Convert nodule coordinates from real world to row column index
def xyz2irc(xyz, image):
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    return (xyz - origin) / spacing

# Extract radiomic features
def extract_features(scan_id, nodule_id, image, mask):

    # Initialize feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()

    # Enable specific feature classes
    extractor.enableFeatureClassByName('shape')
    extractor.enableFeatureClassByName('firstorder')

    # Run feature extraction
    features = extractor.execute(image, mask)

    feature_dict = {t[0]:t[1] for t in features.items()}
    feature_dict['scan_id'] = scan_id
    feature_dict['nodule_id'] = nodule_id

    return pd.DataFrame.from_dict(feature_dict,orient='index').T

def segment_nodule(image, image_array, irc_coordinates, diameter, hu_threshold):

    # Create a binary mask with the same shape as the image
    mask = np.zeros_like(image_array, dtype=np.uint8)

    # Calculate the coordinates of the nodule boundary
    row, col, idx = irc_coordinates
    radius = (diameter / 2) * 1.5 # Increase the radius to ensure the entire nodule is included

    row_start = max(0, int(row - radius))
    row_end = min(image_array.shape[2], int(row + radius))

    col_start = max(0, int(col - radius))
    col_end = min(image_array.shape[1], int(col + radius))

    idx_start = max(0, int(idx - radius))
    idx_end = min(image_array.shape[0], int(idx + radius))

    # Set the nodule region to 1 in the mask
    mask[idx_start:idx_end, col_start:col_end, row_start:row_end] = 1
    mask[image_array < hu_threshold] = 0 # Set the background to 0

    mask = sitk.GetImageFromArray(mask)
    mask.CopyInformation(image)

    return image, mask

# Segment the nodule in the image
def crop_nodule(image, image_array, irc_coordinates, diameter, hu_threshold):
    
    # Calculate the coordinates of the nodule boundary
    row, col, idx = irc_coordinates
    radius = (diameter / 2) * 1.5 # Increase the radius to ensure the entire nodule is included

    row_start = max(0, int(row - radius))
    row_end = min(image_array.shape[2], int(row + radius))

    col_start = max(0, int(col - radius))
    col_end = min(image_array.shape[1], int(col + radius))

    idx_start = max(0, int(idx - radius))
    idx_end = min(image_array.shape[0], int(idx + radius))
    
    # Set the nodule region to 1 in the mask
    cropped_image = image_array[idx_start:idx_end, col_start:col_end, row_start:row_end]
    cropped_mask = np.ones_like(cropped_image, dtype=np.uint8)   
    cropped_mask[cropped_image < hu_threshold] = 0 # Set the background to 0
    cropped_mask[0,0,0] = 0

    cropped_image = sitk.GetImageFromArray(cropped_image)
    cropped_mask = sitk.GetImageFromArray(cropped_mask)
    cropped_mask.CopyInformation(cropped_image)

    return cropped_image, cropped_mask

def get_radiomic_features(idx, image_list, nodule_metadata, segment_or_crop):
    # Load the image
    raw_image, raw_image_array = load_image(image_list[idx])
    
    scan_id = image_list[idx].stem
    study_id = scan_id.split('_')[0]
    scan_nodule_data = nodule_metadata.loc[nodule_metadata['participant_id'] == study_id]



    log_msgs = []

    log_msgs.append(f'Processing {scan_id} - Nodules Count: {scan_nodule_data.shape[0]} , start time:{datetime.now()}')

    features = []
    
    for idx, nodule_data in scan_nodule_data.iterrows():

        try:

            nodule_id = nodule_data['radiology_report_nodule_lesion_id']

            xyz_coordinates = nodule_data[[
                'radiology_report_nodule_x_coordinate', 
                'radiology_report_nodule_y_coordinate',
                'radiology_report_nodule_z_coordinate'
            ]].values

            irc_coordinates = xyz2irc(xyz_coordinates, raw_image)

            diameter = nodule_data['radiology_report_nodule_diameter_mm']

            hu_threshold = -50 if nodule_data['radiology_report_nodule_type'] in ['SOLID','PERIFISSURAL'] else -400

            if segment_or_crop == 'segment':
                # Segment the nodule
                image, mask = segment_nodule(
                                raw_image,
                                raw_image_array,
                                irc_coordinates,
                                diameter,
                                hu_threshold
                            )

            else:
                # Crop the nodule
                image, mask = crop_nodule(
                                raw_image,
                                raw_image_array,
                                irc_coordinates,
                                diameter,
                                hu_threshold
                            )

            # Extract radiomic features
            mask_array = sitk.GetArrayFromImage(mask)
            if np.sum(mask_array) == 0:
                log_msgs.append(f"Scan Id: {scan_id}, Nodule Id: {nodule_id}, Nodule Type: {nodule_data['radiology_report_nodule_type']}, Warning: no hu > {hu_threshold}")
            else:
                features.append(extract_features(scan_id, nodule_id, image, mask))
            
            if random.random() < 0.1:
                Path('images').mkdir(exist_ok=True, parents=True)

                if segment_or_crop == 'segment':
                    save_image_with_mask_as_png(
                        image=image,
                        coordinates=irc_coordinates,
                        diameter=diameter,
                        output_path=f'images/segmented_{scan_id}_nodule_{idx}.png'
                    )

                else:
                    save_cropped_masked_image_as_png(
                        image=image,
                        output_path=f'images/cropped_{scan_id}_nodule_{idx}.png'
                    )

        except Exception as err:
            log_msgs.append(f'Error processing {scan_id}')

    return log_msgs, pd.concat(features) if features else pd.DataFrame()

def collate_nodule_radiomics(image_path, nodule_metadata, workers, segment_or_crop, throttle=None, batch_size=None, batch_number=None):

    # Get list of images
    image_list = list(Path(image_path).rglob('*.mhd'))

    if batch_size:
        image_list = image_list[batch_size * (batch_number - 1):batch_size * batch_number]


    N = len(image_list) if throttle and len(image_list) < throttle else throttle

    if workers > 1:

        partial_get_radiomic_fatures = partial(
                                            get_radiomic_features,
                                            image_list=image_list,
                                            nodule_metadata=nodule_metadata,
                                            segment_or_crop=segment_or_crop
                                        )

        with Pool(workers) as p:
            results = p.map(partial_get_radiomic_fatures, range(N))

        features = [result[1] for result in results]
        log_msgs = [msg for result in results for msg in result[0]]

    else:
        features = []
        log_msgs = []
        for idx in range(N):
            log_msg, feature = get_radiomic_features(
                                    idx,
                                    image_list,
                                    nodule_metadata,
                                    segment_or_crop
                                )
            
            log_msgs.extend(log_msg)
            features.append(feature)

    # dump out the log msgs
    for log_msg in log_msgs:
        print(log_msg)

    features = pd.concat(features)

    Path('nodule_stats').mkdir(exist_ok=True, parents=True)
    features.to_csv(f'nodule_stats/{segment_or_crop}_features.csv')

if __name__ == '__main__':

    args = parse_args()

    image_path          = args.image_path
    nodule_metadata     = pd.read_csv(args.nodule_metadata)
    workers             = int(args.workers)
    throttle            = int(args.throttle)
    segment_or_crop     = args.segment_or_crop
    batch_size          = args.batch_size
    batch_number        = args.batch_number

    collate_nodule_radiomics(
        image_path=image_path,
        nodule_metadata=nodule_metadata,
        workers=workers,
        segment_or_crop=segment_or_crop,
        throttle=throttle,
        batch_size=batch_size,
        batch_number=batch_number
    )
