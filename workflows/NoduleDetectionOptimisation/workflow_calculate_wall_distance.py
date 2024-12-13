
from pathlib import Path
import SimpleITK as sitk
import tempfile
import zipfile
from matplotlib import pyplot as plt
from metaflow import FlowSpec, step, Parameter
import pandas as pd
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from metaflow import FlowSpec, step, Parameter
import numpy as np
import os
import json
import pandas as pd

def xyz2irc(coord_xyz, origin, voxel_size, orientation=np.array([[1,0,0],[0,1,0],[0,0,1]])):

    origin_a = np.array(origin)
    voxel_size_a = np.array(voxel_size)
    coord_a = np.array(coord_xyz)

    cri_a = ((coord_a - origin_a) @ np.linalg.inv(orientation)) / voxel_size_a
    
    # it can only be whole numbers as irc
    cri_a = np.round(cri_a)
    return (int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))

class NoduleDistanceFlow(FlowSpec):

    input_file = Parameter('input-file', help='Path to input file containing nodules and scans')
    image_root = Parameter('image-root', help='Root directory containing images by study_id')
    mask_root = Parameter('mask-root', help='Directory containing lung masks by scan_id')
    
    @step
    def start(self):
        # Load the input file
        recode = {
            'radiology_report_nodule_lesion_id': 'nodule_lesion_id',
            'radiology_report_nodule_x_coordinate': 'nodule_x_coordinate',
            'radiology_report_nodule_y_coordinate': 'nodule_y_coordinate',
            'radiology_report_nodule_z_coordinate': 'nodule_z_coordinate',
            'radiology_report_nodule_diameter_mm': 'nodule_diameter_mm'
        }

        self.data = (
            pd.read_csv(
            self.input_file,
            usecols=['participant_id'] + list(recode.keys()),
            )
            .rename(columns=recode)
        )
        # self.study_ids = self.data['participant_id'].unique()[:1]
        self.study_ids = [
            'summit-2225-stn',
            'summit-2227-pjw',
            'summit-2252-etc', 
            'summit-2257-hrd',
            'summit-2276-nzk',
            'summit-2323-kha',
            'summit-2336-xcb',
            'summit-2344-yhh',
            'summit-2354-cfc'
        ]
        self.next(self.process_scans, foreach='study_ids')
    
    @step
    def process_scans(self):
        try:
            # Process nodules for each scan
            study_id = self.input
            scan_id =  f'{study_id}_Y0_BASELINE_A'
            nodules = self.data.query('participant_id == @study_id').to_dict(orient='records')
            
            # Load image to get origin, voxel size, and orientation
            image_path = Path(self.image_root, study_id, f'{scan_id}.mhd')
            image, origin, voxel_size, orientation = self._load_scan(image_path)

            # Load lung mask
            mask_path = os.path.join(self.mask_root, f'{scan_id}.zip')
            mask = self._build_lung_masks(mask_path)
            
            # Calculate distances
            self.results = [
                {
                    'participant_id': nodule['participant_id'], 
                    'nodule_lesion_id': nodule['nodule_lesion_id'], 
                    'distance': self._calculate_distance(
                        nodule['participant_id'],
                        nodule['nodule_lesion_id'],
                        self._xyz2irc(
                            nodule, 
                            origin, 
                            voxel_size,
                            orientation
                        ),
                        image,
                        mask
                    )
                }
                for idx, nodule in enumerate(nodules)
            ]

        except Exception as e:

            print(f"Error processing scan {study_id}: {e}")

            self.results = [
                {
                    'participant_id': nodule['participant_id'], 
                    'nodule_lesion_id': nodule['nodule_lesion_id'], 
                    'distance': -99
                }
                for nodule in nodules
            ]
    
        self.next(self.aggregate_results)

    @step
    def aggregate_results(self, inputs):
        # Combine results from all scans
        self.all_results = sum((inp.results for inp in inputs), [])
        self.next(self.end)
    
    @step
    def end(self):
        # Save or print final results
        output_path = 'nodule_distances.csv'
        df = pd.DataFrame(self.all_results)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    def _build_lung_masks(self, segmentation_path):

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

    def _load_scan(self, image_path):
        metadata = sitk.ReadImage(image_path)
        image = np.array(sitk.GetArrayFromImage(metadata), dtype=np.float32)
        origin = metadata.GetOrigin()
        voxel_size = metadata.GetSpacing()
        orientation = np.array(metadata.GetDirection()).reshape(3,3)        
        return image, origin, voxel_size, orientation
        
    def _xyz2irc(self, nodule, origin, voxel_size, orientation):
        x, y, z = nodule['nodule_x_coordinate'], nodule['nodule_y_coordinate'], nodule['nodule_z_coordinate']
        return xyz2irc((x, y, z), origin, voxel_size, orientation)

    def _calculate_distance(self, pid, nid, irc, image, mask):
        """
        Calculate the distance of a nodule to the lung wall using the lung mask.
        """
        i, r, c = irc
        mask = mask[i, :, :]
        image = image[i, :, :]

        point = (r, c)  # Example point outside the lung region
        if mask[point] == 0:
            distance_transform = distance_transform_edt(1 - mask)
        else:
            distance_transform = distance_transform_edt(mask)

        distance_to_lung = distance_transform[point]

        print(f"Distance from the point {point} to the lung mask: {distance_to_lung} mm")    

        fig, ax =  plt.subplots(1, 2, figsize=(15, 5))
        ax[0].set_title(f"Binary Mask (Slice) - {distance_to_lung:.2f} mm")
        ax[0].imshow(mask[:, :], cmap='gray')
        ax[0].scatter(c, r, color='blue', label='Nodule Center', s=100)

        ax[1].set_title("Distance Transform (Slice)")
        ax[1].imshow(image[:, :], cmap='gray')
        rect = plt.Rectangle((c - 5, r - 5), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
        ax[1].add_patch(rect)

        plt.savefig(f"results/images/{pid}_{nid}_distance.png")

        return distance_to_lung

if __name__ == "__main__":
    NoduleDistanceFlow()
