import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

if os.path.basename(os.getcwd()).upper() == 'SOTAEVALUATIONNODULEDETECTION':
    sys.path.append('utilities')
    sys.path.append('notebooks')
else:
    sys.path.append('../../utilities')
    sys.path.append('../../notebooks')

from evaluation import noduleCADEvaluation

METADATA_COLUMNS = [
    'nodule_lesion_id',
    'nodule_type',
    'nodule_x_coordinate',
    'nodule_y_coordinate',
    'nodule_z_coordinate',
    'nodule_diameter_mm',
    'management_plan',
    'gender',
    'ethnic_group',    
    'name',
    'col',
    'row',
    'index',
    'diameter',    
]

def scp_file(remote_path, local_path):
    if not Path(local_path).exists():
        if sys.platform == "darwin":  # macOS
            subprocess.run(['scp', '-P', '2222', remote_path, local_path], check=True)
        else:
            subprocess.run(['scp', remote_path, local_path], check=True)

def load_csv(file_path, usecols=None, rename_cols=None):
    df = pd.read_csv(file_path, usecols=usecols)
    if rename_cols:
        df = df.rename(columns=rename_cols)
    return df

def load_data(workspace_path, model, flavour, actionable):
    """
    Load data for the specified model and flavour.

    Args:
    workspace_path (str): Path to the workspace data.
    model (str): Model name ('grt123', 'detection', or 'ticnet').
    flavour (str): Dataset flavour.
    actionable (bool): Whether to include only actionable cases.

    Returns:
    tuple: A tuple containing annotations, results, scan_metadata, and annotations_excluded.
    """

    print(model, flavour, actionable)

    if model == 'grt123':
        metadata_path = f'{workspace_path}/models/grt123/bbox_result/trained_summit/summit/{flavour}/{flavour}_metadata.csv'
        predictions_path = f'{workspace_path}/models/grt123/bbox_result/trained_summit/summit/{flavour}/{flavour}_predictions.csv'
        remote_metadata_path = f'jmccabe@little:/cluster/project2/SUMMIT/cache/sota/grt123/bbox_result/trained_summit/summit/{flavour}/{flavour}_metadata.csv'
        remote_predictions_path = f'jmccabe@little:/cluster/project2/SUMMIT/cache/sota/grt123/bbox_result/trained_summit/summit/{flavour}/{flavour}_predictions.csv'

        with ThreadPoolExecutor() as executor:
            executor.submit(scp_file, remote_metadata_path, metadata_path)
            executor.submit(scp_file, remote_predictions_path, predictions_path)

        annotations = load_csv(metadata_path, usecols=METADATA_COLUMNS)
        results = (
            load_csv(predictions_path, usecols=['name', 'col', 'row', 'index', 'diameter', 'threshold'])
            .rename(columns={'threshold': 'threshold_original'})
            .assign(threshold=lambda x: 1 / (1 + np.exp(-x['threshold_original'])))
        )

    elif model == 'detection':
        recode = {
            'scan_id': 'name',
            'nodule_x_coordinate': 'row',
            'nodule_y_coordinate': 'col',
            'nodule_z_coordinate': 'index',
            'nodule_diameter_mm': 'diameter',
            'management_plan': 'management_plan',
            'gender': 'gender',
            'ethnic_group': 'ethnic_group',
            'nodule_lesion_id': 'nodule_lesion_id',
            'nodule_type': 'nodule_type'
        }

        annotations = load_csv(f'{workspace_path}/metadata/summit/{flavour}/test_metadata.csv', usecols=recode.keys(), rename_cols=recode)
        results_path = f'{workspace_path}/models/detection/result/trained_summit/summit/{flavour}/result_{flavour}.json'
        remote_results_path = f'jmccabe@little:/home/jmccabe/jobs/SOTAEvaluationNoduleDetection/models/detection/result/trained_summit/result_{flavour}.json'

        scp_file(remote_results_path, results_path)

        with open(results_path, 'r') as f:
            prediction_json = json.load(f)

        results_json = {
            f'{bdx}_{cnt}': {
                'name': itm['image'].split('/')[-1].replace('.nii.gz', ''),
                'row': box[0],
                'col': box[1],
                'index': box[2],
                'diameter': box[3],
                'threshold': itm['score'][bdx]
            }
            for cnt, itm in enumerate(prediction_json['test'])
            for bdx, box in enumerate(itm['box'])
        }

        results = pd.DataFrame.from_dict(results_json, orient='index')

    elif model == 'ticnet':
        metadata_path = f'{workspace_path}/models/ticnet/annotations/summit/{flavour}/{flavour}_metadata.csv'
        predictions_path = f'{workspace_path}/models/ticnet/results/trained_summit/summit/{flavour}/submission_rpn.csv'
        remote_metadata_path = f'jmccabe@little:/cluster/project2/SUMMIT/cache/sota/ticnet/summit/bboxes/{flavour}/{flavour}_metadata.csv'
        remote_predictions_path = f'jmccabe@little://home/jmccabe/jobs/SOTAEvaluationNoduleDetection/models/ticnet/results/summit/{flavour}/res/120/FROC/submission_rpn.csv'

        scp_file(remote_metadata_path, metadata_path)

        annotations = load_csv(metadata_path, usecols=METADATA_COLUMNS)
        results = load_csv(predictions_path).rename(columns={
            'seriesuid': 'name',
            'coordX': 'row',
            'coordY': 'col',
            'coordZ': 'index',
            'diameter_mm': 'diameter',
            'probability': 'threshold'
        })

    scan_metadata = load_csv(
        f'{workspace_path}/metadata/summit/{flavour}/test_scans_metadata.csv',
        usecols=[
            'Y0_PARTICIPANT_DETAILS_main_participant_id',
            'participant_details_gender',
            'lung_health_check_demographics_race_ethnicgroup'
        ],
        rename_cols={
            'Y0_PARTICIPANT_DETAILS_main_participant_id': 'StudyId',
            'participant_details_gender': 'gender',
            'lung_health_check_demographics_race_ethnicgroup': 'ethnic_group'
        }
    ).assign(Name=lambda x: x['StudyId'] + '_Y0_BASELINE_A')

    # Add a column for the nodule diameter categories
    annotations['diameter_cats'] = pd.cut(
        annotations['diameter'], 
        bins=[0, 6, 8, 30, 40, 999], 
        labels=['0-6mm', '6-8mm', '8-30mm', '30-40mm', '40+mm']
    )

    # Reduce the annotations to actionable cases, setting exclusions as a separate dataframe
    if actionable:
        annotations_excluded = annotations[annotations['management_plan'] == 'RANDOMISATION_AT_YEAR_1']
        annotations = annotations[annotations['management_plan'].isin(['3_MONTH_FOLLOW_UP_SCAN', 'URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1'])]
    else:
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

        with tempfile.TemporaryDirectory() as temp_dir:
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
