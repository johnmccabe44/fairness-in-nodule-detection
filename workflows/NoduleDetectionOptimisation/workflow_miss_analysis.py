import datetime
import json
import logging
from math import e
from tempfile import TemporaryDirectory
import tempfile
import zipfile
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedFormatter
import numpy as np
from metaflow import FlowSpec, step, IncludeFile, Parameter, conda_base
import numpy as np
import os
import pandas as pd
from pathlib import Path
import scipy.stats as stats
import sys
import nibabel as nib
from scipy.ndimage import distance_transform_edt
import subprocess


if os.path.basename(os.getcwd()).upper() == 'SOTAEVALUATIONNODULEDETECTION':
    sys.path.append('utilities')
    sys.path.append('notebooks')
else:
    sys.path.append('../../utilities')
    sys.path.append('../../notebooks')

from summit_utils import SummitScan, xyz2irc, XyzTuple
from evaluation import noduleCADEvaluation
import sys
import os


METADATA_COLUMNS = [
    'name',
    'col',
    'row',
    'index',
    'diameter',
    'management_plan',
    'gender',
    'ethnic_group',
    'nodule_lesion_id',
    'nodule_type',
    'nodule_diameter_mm'
]

def cleanup(s):
    if s == 'CALCIFIED':
         return 'Consolidation'
    
    s = s.replace('_', ' ')
    s = ' '.join(word.capitalize() for word in s.split())
    return s

def load_data(workspace_path, model, flavour, actionable):

    print(model, flavour, actionable)

    # Load the data
    if model == 'grt123':

        if not Path(f'{workspace_path}/models/grt123/bbox_result/trained_summit/summit/{flavour}/{flavour}_metadata..csv').exists():
            subprocess.run(
                [
                    'scp', 
                    f'jmccabe@little:/cluster/project2/SUMMIT/cache/sota/grt123/bbox_result/trained_summit/summit/{flavour}/{flavour}_metadata.csv',
                    f'{workspace_path}/models/grt123/bbox_result/trained_summit/summit/{flavour}/{flavour}_metadata.csv'
                ],
                check=True
            )

        annotations = pd.read_csv(
            f'{workspace_path}/models/grt123/bbox_result/trained_summit/summit/{flavour}/{flavour}_metadata.csv',
            usecols=METADATA_COLUMNS)

        if not Path(f'{workspace_path}/models/grt123/bbox_result/trained_summit/summit/{flavour}/{flavour}_predictions.csv').exists():
            subprocess.run(
                [
                    'scp', 
                    f'jmccabe@little:/cluster/project2/SUMMIT/cache/sota/grt123/bbox_result/trained_summit/summit/{flavour}/{flavour}_predictions.csv',
                    f'{workspace_path}/models/grt123/bbox_result/trained_summit/summit/{flavour}/{flavour}_predictions.csv'
                ],
                check=True
            )

        results = (
            pd.read_csv(
                f'{workspace_path}/models/grt123/bbox_result/trained_summit/summit/{flavour}/{flavour}_predictions.csv',
                usecols=['name', 'col', 'row', 'index', 'diameter', 'threshold']
            )
            .rename(columns={'threshold': 'threshold_original'})
            .assign(threshold=lambda x: 1 / (1 + np.exp(-x['threshold_original'])))
        )
        
    elif model == 'detection':

        recode = {
            'scan_id' : 'name',
            'nodule_x_coordinate' : 'row',
            'nodule_y_coordinate' : 'col',
            'nodule_z_coordinate' : 'index',
            'nodule_diameter_mm' : 'diameter',
            'management_plan' : 'management_plan',
            'gender' : 'gender',
            'ethnic_group' : 'ethnic_group',
            'nodule_lesion_id' : 'nodule_lesion_id',
            'nodule_type' : 'nodule_type'
        }

        annotations = (
            pd.read_csv(f'{workspace_path}/metadata/summit/{flavour}/test_metadata.csv', usecols=recode.keys())
            .rename(columns=recode)
        )

        if not Path(f'{workspace_path}/models/detection/result/trained_summit/summit/{flavour}/result_{flavour}.json').exists():
            subprocess.run(
                [
                    'scp', 
                    f'jmccabe@little:/home/jmccabe/jobs/SOTAEvaluationNoduleDetection/models/detection/result/trained_summit/result_{flavour}.json',
                    f'{workspace_path}/models/detection/result/trained_summit/summit/{flavour}/result_{flavour}.json'
                ],
                check=True
            )

        prediction_json = json.load(open(f'{workspace_path}/models/detection/result/trained_summit/summit/{flavour}/result_{flavour}.json','r'))
        
        results_json, cnt = {}, 0
        for itm in prediction_json['test']:
            for bdx, box in enumerate(itm['box']):
                results_json[cnt] = {
                    'name' : itm['image'].split('/')[-1].replace('.nii.gz', ''),
                    'row' : box[0],
                    'col' : box[1],
                    'index' : box[2],
                    'diameter' : box[3],
                    'threshold' : itm['score'][bdx]
                }
                cnt += 1
        
        results = pd.DataFrame.from_dict(results_json, orient='index')
        print(results.head())

    elif model == 'ticnet':


        if not Path(f'{workspace_path}/models/ticnet/annotations/summit/{flavour}/{flavour}_metadata.csv').exists():
            subprocess.run(
                [
                    'scp', 
                    f'jmccabe@little:/cluster/project2/SUMMIT/cache/sota/ticnet/summit/bboxes/{flavour}/{flavour}_metadata.csv',
                    f'{workspace_path}/models/ticnet/annotations/summit/{flavour}/{flavour}_metadata.csv'
                ],
                check=True
            )

        annotations = pd.read_csv(
            f'{workspace_path}/models/ticnet/annotations/summit/{flavour}/{flavour}_metadata.csv',
            usecols=METADATA_COLUMNS
        )

        results = (
            pd.read_csv(
                f'{workspace_path}/models/ticnet/results/trained_summit/summit/{flavour}/submission_rpn.csv'
            )
            .rename(
                columns={
                    'seriesuid' : 'name',
                    'coordX' : 'row',
                    'coordY' : 'col',
                    'coordZ' : 'index',
                    'diameter_mm' : 'diameter',
                    'probability' : 'threshold'
                }
            )
        )

    scan_metadata = (
        pd.read_csv(
            f'{workspace_path}/metadata/summit/{flavour}/test_scans_metadata.csv',
            usecols=[
                'Y0_PARTICIPANT_DETAILS_main_participant_id',
                'participant_details_gender',
                'lung_health_check_demographics_race_ethnicgroup'
            ]
        )
        .rename(
            columns={
                'Y0_PARTICIPANT_DETAILS_main_participant_id': 'StudyId',
                'participant_details_gender' : 'gender',
                'lung_health_check_demographics_race_ethnicgroup' : 'ethnic_group'
            }
        )
        .assign(
            Name=lambda x: x['StudyId'] + '_Y0_BASELINE_A'
        )
    )

    # Reduce the annotations to only actionable cases
    if actionable:
        annotations = annotations[annotations['management_plan'].isin(['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1'])]
        annotations_excluded = annotations[annotations['management_plan']=='RANDOMISATION_AT_YEAR_1']
    else:
        annotations = annotations
        annotations_excluded = annotations.drop(annotations.index)

    return annotations, results, scan_metadata, annotations_excluded

def get_thresholds(analysis_data, operating_points=[0.125, 0.25, 0.5, 1, 2, 4, 8]):
    """
    Get the thresholds at different FPPs

    Args:
    analysis_data: tuple, tuple of fps, sens, threshold

    Returns:
    thresh_values: list of float, list of thresholds at different FPPs

    """
    fps = analysis_data[0]
    threshold = analysis_data[2]

    thresh_values = {}
    for fpps in operating_points:
        idx = np.abs(fps - fpps).argmin()
        thresh_values[fpps] = threshold[idx]

    return thresh_values

def miss_anaysis_at_fpps(scans_path, annotations_path, exclusions_path, predictions_path, thresholds):

    missed_metadata = {}

    for operating_point, threshold in thresholds.items():

        predictions = pd.read_csv(predictions_path)
        predictions_at_operating_point = predictions[predictions.threshold > threshold]

        with TemporaryDirectory() as temp_dir:
            predictions_at_operating_point.to_csv(f'{temp_dir}/predictions.csv', index=False)
            missed_annotations = noduleCADEvaluation(
                annotations_filename=annotations_path,
                annotations_excluded_filename=exclusions_path,
                seriesuids_filename=scans_path,
                results_filename=f'{temp_dir}/predictions.csv',
                filter='Missed Annotations',
                outputDir=f'{temp_dir}/results',
                perform_bootstrapping=False,
                show_froc=False
            )

            misses = (
                pd.read_csv(
                f'{temp_dir}/results/nodulesWithoutCandidate_predictions.txt',
                header=None
                )
                .rename(columns={0:'name',1: 'idx', 2:'col',3:'row',4:'index',5:'diameter',6:'threshold'})
                .assign(miss=True)
            )

            annotations = pd.read_csv(annotations_path)

            df = pd.merge(
                misses, 
                annotations, 
                on=['name','row','col','index','diameter'], 
                how='right'
            )

            df['miss'] = df['miss'].fillna(False)

            print(f'Missed Annotations at {threshold} FPPs:', sum(df.miss))

            missed_metadata[operating_point] = df

    return missed_metadata

def get_voxel_coords(scan_path, xyz):
    
    if not os.path.exists(scan_path):
        print(f'{scan_path} does not exist')
        return None

    scan = SummitScan.load_scan(scan_path)

    return xyz2irc(
        xyz,
        scan.origin,
        scan.voxel_size,
        scan.orientation
    )

def display_nodules(scan_id, scan_path, data):

    print(scan_path)
    if not os.path.exists(scan_path):
        print(f'{scan_path} does not exist')
        return None
        
    scan = SummitScan.load_scan(scan_path)

    # Get the number of nodules
    nodule_cnt = data.shape[0]

    # Create a new figure based on the number of nodules
    fig, axs = plt.subplots(nodule_cnt, 1, figsize=( 7, nodule_cnt * 7))
    fig.suptitle(f"{scan_id} ({nodule_cnt} nodules)")
    
    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]  # Convert single Axes to a list for consistency

    for i, (idx, row) in enumerate(data.iterrows()):

        img = scan.image[row['idx'], :, :]
        axs[i].imshow(img, cmap='gray')

        if row["nodule_type"] == 'CALCIFIED':
            diameter = 15
        else:
            diameter = row['nodule_diameter_mm']

        # Add a rectangle centered on irc
        rect = plt.Rectangle(
            (
                row['col'] - diameter,
                row['row'] - diameter
            ),
            diameter * 2,
            diameter * 2,
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )

        axs[i].add_patch(rect)
        axs[i].set_title(f"{row['nodule_type']} ({row['nodule_diameter_mm']}mm)")

    plt.savefig(f'results/images/{scan_id}_nodules.png')

def build_lung_masks(segmentation_path, output_path):

    if not os.path.exists(segmentation_path):
        print(f'{segmentation_path} does not exist')
        return None

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
        
        combined_mask_nifti = nib.Nifti1Image(combined_mask, mask.affine)
        nib.save(combined_mask_nifti, output_path)


def calculate_distance_from_mask(mask_path, scan_id, idx, row, col, nodule_type, nodule_diameter, nodule_lesion_id):

    if not os.path.exists(mask_path):
        print(f'{mask_path} does not exist')
        return None

    mask = nib.load(mask_path).get_fdata().astype(np.uint8)
    mask = np.transpose(mask, (2, 1, 0))
    mask = mask[idx, :, :]

    distance_transform = distance_transform_edt(mask)

    # Define the point (in voxel coordinates)
    point = (col, row)  # Example point outside the lung region
    if mask[point] == 0:
        xdistance_transform = distance_transform_edt(1 - mask)
        distance_to_lung = xdistance_transform[point] * -1
    else:
        distance_to_lung = distance_transform[point]

    # Get the distance from the point to the nearest lung voxel
    print(f"Distance from the point {point} to the lung mask: {distance_to_lung} mm")    

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title(f"Binary Mask (Slice), {nodule_type} ({nodule_diameter}mm)")
    plt.imshow(mask[:, :], cmap='gray')
    plt.scatter(row, col, color='blue', label='Nodule Center', s=100)

    plt.subplot(1, 2, 2)
    plt.title("Distance Transform (Slice)")
    plt.imshow(distance_transform[:, :], cmap='hot')

    plt.savefig(f"results/images/{scan_id}_{nodule_lesion_id}_distance.png")

    return distance_to_lung

class MissedNodulesFlow(FlowSpec):
    """
    Calculate the free response operating characteristic (FROC) scores for each demographic group
    """

    actionable = Parameter(
        'actionable', 
        type=bool, 
        help='Only include actionable cases', 
        default=True
    )
    flavour = Parameter(
        'flavour',
        type=str,
        help='Dataset flavour',
        default='optimisation'
    )
    scan_path = Parameter(
        'scan_path', 
        type=str, 
        help='Path to the scan data', 
        default='/Users/john/Projects/SOTAEvaluationNoduleDetection/data/summit/scans'
    )
    segmentation_path = Parameter(
        'segmentation_path', 
        type=str, 
        help='Path to the segmentation data', 
        default='/Users/john/Projects/SOTAEvaluationNoduleDetection/data/summit/segmentations'
    )
    workspace_path = Parameter(
        'workspace_path', 
        type=str, 
        help='Path to the workspace data', 
        default='/Users/john/Projects/SOTAEvaluationNoduleDetection'
    )
    
    @step
    def start(self):
    
        self.models = ['grt123', 'detection', 'ticnet']
        self.next(self.get_missed_annotations, foreach='models')

    @step
    def get_missed_annotations(self):

        self.model = self.input

        print(f'Processing {self.model} model')

        annotations, results, scan_metadata, annotations_excluded = load_data(self.workspace_path, self.model, self.flavour, self.actionable)

        # Reduce the annotations to only actionable cases
        annotations['diameter_cats'] = pd.cut(
            annotations['diameter'], 
            bins=[0, 6, 8, 30, 40, 999], 
            labels=['0-6mm', '6-8mm', '8-30mm', '30-40mm', '40+mm']
        )

        if self.actionable:
            annotations = annotations[annotations['management_plan'].isin(['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1'])]
            annotations_excluded = annotations[annotations['management_plan']=='RANDOMISATION_AT_YEAR_1']
        else:
            annotations = annotations
            annotations_excluded = annotations.drop(annotations.index)

        scans = scan_metadata['Name']

        with TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            scans.to_csv(temp_dir_path / 'scans.csv', index=False)
            annotations.to_csv(temp_dir_path / 'annotations.csv', index=False)
            annotations_excluded.to_csv(temp_dir_path / 'exclusions.csv', index=False)
            results.to_csv(temp_dir_path / 'predictions.csv', index=False)
            output_path = temp_dir_path / 'miss_analysis'

            froc_metrics = noduleCADEvaluation(
            annotations_filename=temp_dir_path / 'annotations.csv',
            annotations_excluded_filename=temp_dir_path / 'exclusions.csv', 
            seriesuids_filename=temp_dir_path / 'scans.csv',
            results_filename=temp_dir_path / 'predictions.csv',
            filter=f'Model: {self.model}, \nDataset: {self.flavour}, \nActionable Only: {self.actionable}',
            outputDir=output_path
            )

            thresholds = get_thresholds(froc_metrics, operating_points=[0.125, 2, 256])

            self.missed_metadata = miss_anaysis_at_fpps(
                scans_path=temp_dir_path / 'scans.csv',
                annotations_path=temp_dir_path / 'annotations.csv',
                exclusions_path=temp_dir_path / 'exclusions.csv',
                predictions_path=temp_dir_path / 'predictions.csv',
                thresholds=thresholds
            )

        self.next(self.join)

    @step
    def join(self, inputs):

        self.results = {}
        for input in inputs:
            self.results[input.model] = input.missed_metadata

        self.next(self.calculate_distances)

    @step
    def calculate_distances(self):

        Path('results/images').mkdir(exist_ok=True)
        Path('segmentations').mkdir(exist_ok=True)

        missed_nodule_counts = {}
        for model in self.results.keys():

            model_missed_metadata = self.results[model]
            for operating_point in [0.125]:

                missed_metadata = model_missed_metadata[operating_point]

                for idx, row in missed_metadata[missed_metadata.miss].iterrows():
                    uid = f"{row['name']}_{row['nodule_lesion_id']}"

                    if uid not in missed_nodule_counts:
                        missed_nodule_counts[uid] = []

                    missed_nodule_counts[uid].append(model)        

        missed_nodule_counts = {k: '|'.join(v) for k, v in missed_nodule_counts.items()}

        any = (
            pd.DataFrame(missed_nodule_counts.items(), columns=['uid', 'models'])
            .assign(grt123=lambda x: x.models.str.contains('grt123'))
            .assign(detection=lambda x: x.models.str.contains('detection'))
            .assign(ticnet=lambda x: x.models.str.contains('ticnet'))
            .filter(['uid', 'grt123', 'detection', 'ticnet'])
        )

        test_data = (
            pd.read_csv(f'{self.workspace_path}/metadata/summit/{self.flavour}/test_metadata.csv')
            .assign(uid=lambda x: x.scan_id + '_' + x.nodule_lesion_id.astype(str))
            )

        # 1. all
        any_data = any.merge(test_data, on='uid')

        voxel_coords = {}
        for idx, row in any_data.iterrows():
            
            scan_id = row['scan_id']
            study_id = scan_id.split('_')[0]

            scan_path = f'{self.scan_path}/{study_id}/{scan_id}.mhd'

            voxel_coords[idx] = get_voxel_coords(
                scan_path,
                XyzTuple(
                    row['nodule_x_coordinate'],
                    row['nodule_y_coordinate'],
                    row['nodule_z_coordinate']
                )
            )

        voxel_coords_df = pd.DataFrame(voxel_coords).T
        voxel_coords_df.columns = ['idx', 'row', 'col']
        any_data = any_data.join(voxel_coords_df)

        for scan_id, grp in any_data.groupby('scan_id'):

            study_id = scan_id.split('_')[0]
            scan_path = f'{self.scan_path}/{study_id}/{scan_id}.mhd'

            display_nodules(scan_id, scan_path, grp)

        Path('segmentations').mkdir(exist_ok=True)
        for scan_id in any_data['scan_id'].unique():
            study_id = scan_id.split('_')[0]

            segmentation_path = f'{self.segmentation_path}/{scan_id}.zip'
            
            build_lung_masks(segmentation_path, f"segmentations/{scan_id}.nii.gz")


        distances_to_lung = {}
        for idx, row in any_data.iterrows():
            distances_to_lung[idx] =  calculate_distance_from_mask(
                f"segmentations/{row['scan_id']}.nii.gz",
                row['scan_id'], 
                row['idx'], 
                row['col'], 
                row['row'], 
                row['nodule_type'], 
                row['nodule_diameter_mm'],
                row['nodule_lesion_id']
            )
      
        any_data = any_data.join(pd.Series(distances_to_lung, name='distance_to_lung'))

        self.any_data = any_data

        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    MissedNodulesFlow()