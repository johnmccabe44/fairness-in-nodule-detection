import datetime
import json
import logging
from math import e
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

if os.path.basename(os.getcwd()).upper() == 'SOTAEVALUATIONNODULEDETECTION':
    sys.path.append('utilities')
    sys.path.append('notebooks')
else:
    sys.path.append('../utilities')
    sys.path.append('../notebooks')

from FairnessInNoduleDetectionAlgorithms.utils import (
    get_thresholds, 
    miss_anaysis_at_fpps
)

from summit_utils import *
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

class MissedNodulesFlow(FlowSpec):
    """
    Calculate the free response operating characteristic (FROC) scores for each demographic group
    """

    flavour = Parameter('flavour', help='Flavour to evaluate', default='test_balanced')
    actionable = Parameter('actionable', type=bool, help='Only include actionable cases', default=True)

    if os.path.basename(os.getcwd()).upper() == 'SOTAEVALUATIONNODULEDETECTION':
        workspace_path = Path(os.getcwd()).as_posix()
    else:
        workspace_path = Path(os.getcwd()).parent.as_posix()
    
    @step
    def start(self):
    
        self.models = ['grt123', 'detection', 'ticnet']
        self.next(self.get_missed_annotations, foreach='models')

    @step
    def get_missed_annotations(self):

        self.model = self.input

        print(f'Processing {self.model} model')

        # Load the data
        if self.model == 'grt123':
            annotations = pd.read_csv(
                f'{self.workspace_path}/models/grt123/bbox_result/trained_summit/summit/{self.flavour}/{self.flavour}_metadata.csv',
                usecols=METADATA_COLUMNS
            )

            predictions = (
                pd.read_csv(
                    f'{self.workspace_path}/models/grt123/bbox_result/trained_summit/summit/{self.flavour}/{self.flavour}_predictions.csv',
                    usecols=['name', 'col', 'row', 'index', 'diameter', 'threshold']
                )
                .rename(columns={'threshold': 'threshold_original'})
                .assign(threshold=lambda x: 1 / (1 + np.exp(-x['threshold_original'])))
            )

        elif self.model == 'detection':
            annotations = pd.read_csv(
                f'{self.workspace_path}/models/detection/result/trained_summit/summit/{self.flavour}/annotations.csv',
                usecols=METADATA_COLUMNS
            )
            
            predictions = pd.read_csv(
                f'{self.workspace_path}/models/detection/result/trained_summit/summit/{self.flavour}/predictions.csv',
                usecols=['name', 'col', 'row', 'index', 'diameter', 'threshold']
            )

        elif self.model == 'ticnet':

            _annotations = (
                pd.read_csv(f'/Users/john/Projects/TiCNet-main/annotations/summit/{self.flavour}/test_metadata.csv').rename(columns={
                    'seriesuid' : 'name',
                    'coordX' : 'row',
                    'coordY' : 'col',
                    'coordZ' : 'index',
                    'diameter_mm' : 'diameter',
                    'probability' : 'threshold'
                })
                .assign(name_counter=lambda df: df.groupby('name').cumcount() + 1)
            )

            _metadata = (
                pd.read_csv(f'/Users/john/Projects/SOTAEvaluationNoduleDetection/metadata/summit/{self.flavour}/test_metadata.csv')
                .assign(name=lambda df: df.participant_id + '_Y0_BASELINE_A')
                .assign(name_counter=lambda df: df.groupby('name').cumcount() + 1)
            )

            annotations = pd.merge(_metadata, _annotations, on=['name', 'name_counter'], how='outer')[METADATA_COLUMNS]

            assert annotations.shape[0] == _metadata.shape[0], 'Mismatch in number of annotations'

            predictions = pd.read_csv(f'/Users/john/Projects/TiCNet-main/results/summit/{self.flavour}/res/110/FROC/submission_rpn.csv').rename(
                columns={
                    'seriesuid' : 'name',
                    'coordX' : 'row',
                    'coordY' : 'col',
                    'coordZ' : 'index',
                    'diameter_mm' : 'diameter',
                    'probability' : 'threshold'
                }
            )

        scan_metadata = pd.read_csv(f'{self.workspace_path}/metadata/summit/{self.flavour}/test_scans_metadata.csv',
                                        usecols=['Y0_PARTICIPANT_DETAILS_main_participant_id', 'participant_details_gender','lung_health_check_demographics_race_ethnicgroup']).rename(
                                        columns={'Y0_PARTICIPANT_DETAILS_main_participant_id': 'StudyId', 'participant_details_gender' : 'gender', 'lung_health_check_demographics_race_ethnicgroup' : 'ethnic_group'}).assign(
                                        Name=lambda x: x['StudyId'] + '_Y0_BASELINE_A')
        training_data_path = f'{self.workspace_path}/metadata/summit/{self.flavour}/training_metadata.csv'

        # Reduce the annotations to only actionable cases
        annotations['diameter_cats'] = pd.cut(
            annotations['nodule_diameter_mm'], 
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
            predictions.to_csv(temp_dir_path / 'predictions.csv', index=False)
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
                model=self.model,
                experiment=self.flavour,
                scans_metadata=scan_metadata,
                scans_path=temp_dir_path / 'scans.csv',
                annotations_path=temp_dir_path / 'annotations.csv',
                exclusions_path=temp_dir_path / 'exclusions.csv',
                predictions_path=temp_dir_path / 'predictions.csv',
                thresholds=thresholds,
                output_path=output_path
            )

        self.next(self.join)

    @step
    def join(self, inputs):

        self.results = {}
        for input in inputs:
            self.results[input.model] = input.missed_metadata

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    MissedNodulesFlow()