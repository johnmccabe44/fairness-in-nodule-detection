import json
import multiprocessing as mp
import os
import tempfile
import zipfile
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt


def xyz2irc(coord_xyz: Tuple[float, float, float], origin: Tuple[float, float, float], voxel_size: Tuple[float, float, float], orientation: np.ndarray = np.array([[1,0,0],[0,1,0],[0,0,1]])) -> Tuple[int, int, int]:
    """
    Converts coordinates from the XYZ coordinate system to the IRC (Index, Row, Column) coordinate system.

    Args:
        coord_xyz (Tuple[float, float, float]): The (x, y, z) coordinates in the XYZ coordinate system.
        origin (Tuple[float, float, float]): The origin (x, y, z) of the XYZ coordinate system.
        voxel_size (Tuple[float, float, float]): The size of each voxel in the XYZ coordinate system.
        orientation (np.ndarray, optional): The orientation matrix of the XYZ coordinate system. Defaults to the identity matrix.

    Returns:
        Tuple[int, int, int]: The (i, r, c) coordinates in the IRC coordinate system.
    """
    origin_a = np.array(origin)
    voxel_size_a = np.array(voxel_size)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(orientation)) / voxel_size_a
    cri_a = np.round(cri_a)
    return (int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))

def build_lung_masks(segmentation_path: str) -> np.ndarray:
    """
    Extracts lung lobe segmentation files from a zip archive, combines them into a single mask, 
    and returns the combined mask as a numpy array.
    Args:
        segmentation_path (str): Path to the zip file containing lung lobe segmentation files.
    Returns:
        np.ndarray: A 3D numpy array representing the combined lung lobe masks.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(segmentation_path, 'r') as zip_ref:
            zip_ref.extract('lung_lower_lobe_left.nii.gz', temp_dir)
            zip_ref.extract('lung_lower_lobe_right.nii.gz', temp_dir)
            zip_ref.extract('lung_upper_lobe_left.nii.gz', temp_dir)
            zip_ref.extract('lung_upper_lobe_right.nii.gz', temp_dir)
            zip_ref.extract('lung_middle_lobe_right.nii.gz', temp_dir)

        for idx, nii_path in enumerate(os.listdir(temp_dir)):
            mask = nib.load(os.path.join(temp_dir, nii_path))
            if idx == 0:
                combined_mask = mask.get_fdata().astype(np.uint8)
            else:
                combined_mask += mask.get_fdata().astype(np.uint8)
        
        return np.transpose(combined_mask.astype(np.uint8), (2, 1, 0))

def load_scan(image_path: str) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float], np.ndarray]:
    """
    Loads a medical scan from the given image path and extracts relevant metadata.

    Args:
        image_path (str): The file path to the medical scan image.

    Returns:
        Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float], np.ndarray]:
            - image (np.ndarray): The image data as a NumPy array.
            - origin (Tuple[float, float, float]): The origin of the image in physical space.
            - voxel_size (Tuple[float, float, float]): The size of each voxel in physical space.
            - orientation (np.ndarray): The orientation matrix of the image.
    """
    metadata = sitk.ReadImage(image_path)
    image = np.array(sitk.GetArrayFromImage(metadata), dtype=np.float32)
    origin = metadata.GetOrigin()
    voxel_size = metadata.GetSpacing()
    orientation = np.array(metadata.GetDirection()).reshape(3,3)        
    return image, origin, voxel_size, orientation

def xyz2irc_wrapper(nodule: Dict[str, Any], origin: Tuple[float, float, float], voxel_size: Tuple[float, float, float], orientation: np.ndarray) -> Tuple[int, int, int]:
    """
    Converts nodule coordinates from the XYZ coordinate system to the IRC coordinate system.

    Args:
        nodule (Dict[str, Any]): A dictionary containing the nodule coordinates with keys 'nodule_x_coordinate', 'nodule_y_coordinate', and 'nodule_z_coordinate'.
        origin (Tuple[float, float, float]): The origin of the coordinate system.
        voxel_size (Tuple[float, float, float]): The size of each voxel in the coordinate system.
        orientation (np.ndarray): The orientation matrix of the coordinate system.

    Returns:
        Tuple[int, int, int]: The nodule coordinates in the IRC coordinate system.
    """
    x, y, z = nodule['nodule_x_coordinate'], nodule['nodule_y_coordinate'], nodule['nodule_z_coordinate']
    return xyz2irc((x, y, z), origin, voxel_size, orientation)

def calculate_distance(args: Tuple[str, str, Tuple[int, int, int], np.ndarray, np.ndarray]) -> Dict[str, Any]:
    """
    Calculate the distance from a point in a mask to the nearest non-zero element using the Euclidean distance transform.

    Args:
        args (Tuple[str, str, Tuple[int, int, int], np.ndarray, np.ndarray]): 
            A tuple containing the following elements:
            - pid (str): Participant ID.
            - nid (str): Nodule lesion ID.
            - irc (Tuple[int, int, int]): A tuple containing the indices (i, r, c) where:
                - i (int): Index along the first dimension.
                - r (int): Row index in the 2D slice.
                - c (int): Column index in the 2D slice.
            - image (np.ndarray): 3D image array.
            - mask (np.ndarray): 3D mask array.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'participant_id' (str): The participant ID.
            - 'nodule_lesion_id' (str): The nodule lesion ID.
            - 'distance' (float): The calculated distance to the nearest non-zero element in the mask.
    """
    pid, nid, irc, image, mask = args
    i, r, c = irc
    mask_slice = mask[i, :, :]
    image_slice = image[i, :, :]

    point = (r, c)
    if mask_slice[point] == 0:
        distance_transform = distance_transform_edt(1 - mask_slice)
        distance_to_lung = distance_transform[point] * -1
    else:
        distance_transform = distance_transform_edt(mask_slice)
        distance_to_lung = distance_transform[point]

    
    return {'participant_id': pid, 'nodule_lesion_id': nid, 'distance': distance_to_lung}

def process_scan(idx: int, study_ids: np.ndarray, data: pd.DataFrame, image_root: str, mask_root: str) -> List[Dict[str, Any]]:
    """
    Processes a single scan to calculate the wall distance for nodules.
    Args:
        idx (int): Index of the study ID to process.
        study_ids (np.ndarray): Array of study IDs.
        data (pd.DataFrame): DataFrame containing nodule information.
        image_root (str): Root directory where the images are stored.
        mask_root (str): Root directory where the masks are stored.
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the calculated distances for each nodule.
    """
    try:
        study_id = study_ids[idx]
        scan_id = f'{study_id}_Y0_BASELINE_A'
        nodules = data.query('participant_id == @study_id').to_dict(orient='records')
        
        image_path = Path(image_root, study_id, f'{scan_id}.mhd')
        image, origin, voxel_size, orientation = load_scan(image_path)

        mask_path = os.path.join(mask_root, f'{scan_id}.zip')
        mask = build_lung_masks(mask_path)
        
        results = [
            calculate_distance(
                (nodule['participant_id'],
                nodule['nodule_lesion_id'],
                xyz2irc_wrapper(nodule, origin, voxel_size, orientation),
                image,
                mask)
            )
            for nodule in nodules
        ]
    except Exception as e:
        results = [
            {
                'participant_id': nodule['participant_id'],
                'nodule_lesion_id': nodule['nodule_lesion_id'], 
                'distance': -99
            }
            for nodule in nodules
        ]
        
    return results

def main(input_file: str, image_root: str, mask_root: str, workers: int = 1) -> None:
    """
    Main function to calculate wall distance for nodules.

    Args:
        input_file (str): Path to the input CSV file containing nodule information.
        image_root (str): Root directory containing the image files.
        mask_root (str): Root directory containing the mask files.
        workers (int, optional): Number of worker processes to use for parallel processing. Defaults to 1.

    Returns:
        None
    """

    recode = {
        'radiology_report_nodule_lesion_id': 'nodule_lesion_id',
        'radiology_report_nodule_x_coordinate': 'nodule_x_coordinate',
        'radiology_report_nodule_y_coordinate': 'nodule_y_coordinate',
        'radiology_report_nodule_z_coordinate': 'nodule_z_coordinate',
        'radiology_report_nodule_diameter_mm': 'nodule_diameter_mm'
    }

    data = (
        pd.read_csv(
            input_file,
            usecols=['participant_id'] + list(recode.keys()),
        )
        .rename(columns=recode)
    )
    study_ids = data['participant_id'].unique()

    # Reduce the study_ids based on the segmentations available
    study_ids = [
        study_id
        for study_id in study_ids
        if os.path.exists(os.path.join(mask_root, f'{study_id}_Y0_BASELINE_A.zip'))
    ]

    print(f"Processing {len(study_ids)} studies")

    partial_process_scan = partial(
        process_scan,
        study_ids=study_ids,
        data=data,
        image_root=image_root,
        mask_root=mask_root
    )

    with mp.Pool(workers) as pool:
        all_results = pool.map(partial_process_scan, range(len(study_ids)))

    all_results = [result for sublist in all_results for result in sublist]

    output_path = 'nodule_distances.csv'
    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Calculate nodule distances to lung wall.')
    parser.add_argument('--input-file', required=True, help='Path to input file containing nodules and scans')
    parser.add_argument('--image-root', required=True, help='Root directory containing images by study_id')
    parser.add_argument('--mask-root', required=True, help='Directory containing lung masks by scan_id')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers to use')

    args = parser.parse_args()
    main(args.input_file, args.image_root, args.mask_root, args.workers)
