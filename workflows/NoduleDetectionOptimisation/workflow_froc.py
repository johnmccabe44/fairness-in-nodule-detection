import datetime
import json
import logging
from math import e
from re import M
from tempfile import TemporaryDirectory
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

METADATA_COLUMNS = ['name', 'col', 'row', 'index', 'diameter', 'management_plan', 'gender', 'ethnic_group', 'nodule_lesion_id', 'nodule_type']

if os.path.basename(os.getcwd()).upper() == 'SOTAEVALUATIONNODULEDETECTION':
    sys.path.append('utilities')
    sys.path.append('notebooks')
else:
    sys.path.append('../../utilities')
    sys.path.append('../../notebooks')

from FairnessInNoduleDetectionAlgorithms.utils import caluclate_cpm_from_bootstrapping, display_plots_with_error_bars
from summit_utils import *
from evaluation import noduleCADEvaluation
import sys
import os

def load_data(workspace_path, model, flavour, actionable):

    print(model, flavour, actionable)

    # Load the data
    if model == 'grt123':
        annotations = pd.read_csv(
            f'{workspace_path}/models/grt123/bbox_result/trained_summit/summit/{flavour}/{flavour}_metadata.csv',
            usecols=METADATA_COLUMNS)

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

class FROCFlow(FlowSpec):
    """
    Calculate the free response operating characteristic (FROC) scores for each demographic group
    """

    actionable = Parameter('actionable', type=bool, help='Only include actionable cases', default=True)
    n_bootstraps = Parameter('bootstraps', help='Number of bootstraps to perform', default=1000)

    if os.path.basename(os.getcwd()).upper() == 'SOTAEVALUATIONNODULEDETECTION':
        workspace_path = Path(os.getcwd()).as_posix()
    else:
        workspace_path = Path(os.getcwd()).parent.parent.as_posix()
    

    @step
    def start(self):
    
        self.models_and_flavours = [
            (model, flavour)
            for model in ['grt123', 'detection', 'ticnet']
            for flavour in ['optimisation', 'optimisation_actionable']
        ]

        self.next(self.calculate_froc, foreach='models_and_flavours')
    
    @step
    def calculate_froc(self):

        self.model, self.flavour = self.input

        annotations, results, scan_metadata, annotations_excluded = load_data(self.workspace_path, self.model, self.flavour, self.actionable)

        output_dir = f'results/summit/{self.model}/{self.flavour}/{"Actionable" if self.actionable else "All"}/FROC'

        scans = scan_metadata['Name']

        annotations = annotations[annotations['name'].isin(scans.values)]
        exclusions = annotations_excluded[annotations_excluded['name'].isin(scans.values)]
        predictions = results[results['name'].isin(scans.values)]

        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            scans.to_csv(temp_dir / 'scans.csv', index=False)
            annotations.to_csv(temp_dir / 'annotations.csv', index=False)
            exclusions.to_csv(temp_dir / 'exclusions.csv', index=False)
            predictions.to_csv(temp_dir / 'predictions.csv', index=False)
            output_path = Path(f'{output_dir}')

            self.froc_metrics = noduleCADEvaluation(
                annotations_filename=temp_dir / 'annotations.csv',
                annotations_excluded_filename=temp_dir / 'exclusions.csv', 
                seriesuids_filename=temp_dir / 'scans.csv',
                results_filename=temp_dir / 'predictions.csv',
                filter=f'Model: {self.model}, \nDataset: {self.flavour}, \nActionable Only: {self.actionable}',
                outputDir=output_path
            )

        self.cpm_data, self.cpm_summary = caluclate_cpm_from_bootstrapping(output_path / 'froc_predictions_bootstrapping.csv')
        self.cpm_data.set_index('fps', inplace=True)

        self.boot_metrics = (
            pd.read_csv(output_path / 'froc_predictions_bootstrapping.csv')
            .rename(columns={
                'FPrate': 'FPRate',
                'Sensivity[Mean]': 'Sensitivity',
                'Sensivity[Lower bound]': 'LowSensitivity',
                'Sensivity[Upper bound]': 'HighSensitivity'
            })
        )

        self.bootstap_results = np.load(output_path / 'bootstrap_sensitivites.npy')
        self.next(self.join)

    @step
    def join(self, inputs):

        self.froc_metrics = {f'{inp.model}_{inp.flavour}' : inp.froc_metrics for inp in inputs}
        self.cpm_data = {f'{inp.model}_{inp.flavour}' : inp.cpm_data for inp in inputs}
        self.cpm_summary = {f'{inp.model}_{inp.flavour}' : inp.cpm_summary for inp in inputs}
        self.boot_metrics = {f'{inp.model}_{inp.flavour}' : inp.boot_metrics for inp in inputs}
        self.bootstap_results = {f'{inp.model}_{inp.flavour}' : inp.bootstap_results for inp in inputs}
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    FROCFlow()