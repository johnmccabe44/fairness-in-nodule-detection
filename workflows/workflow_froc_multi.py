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

from FairnessInNoduleDetectionAlgorithms.utils import caluclate_cpm_from_bootstrapping, display_plots_with_error_bars
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
    'nodule_type'
]

class FROCFlow(FlowSpec):
    """
    Calculate the free response operating characteristic (FROC) scores for each demographic group
    """

    model = Parameter('model', help='Model to evaluate')
    flavour = Parameter('flavour', help='Flavour to evaluate', default='test_balanced')
    actionable = Parameter('actionable', type=bool, help='Only include actionable cases')
    n_bootstraps = Parameter('bootstraps', help='Number of bootstraps to perform', default=1000)
    exclude_outliers = Parameter('exclude_outliers', type=bool, help='Exclude outliers from the bootstrapping')

    if os.path.basename(os.getcwd()).upper() == 'SOTAEVALUATIONNODULEDETECTION':
        workspace_path = Path(os.getcwd()).as_posix()
    else:
        workspace_path = Path(os.getcwd()).parent.as_posix()
    


    @step
    def start(self):
        

        print(self.model, self.flavour, self.actionable, self.exclude_outliers)

        self.output_dir = f'results/summit/resample/{self.model}/{self.flavour}/{"Actionable" if self.actionable else "All"}/{"Excluded" if self.exclude_outliers else "Included"}/FROC'

        # Load the data
        if self.model == 'grt123':
            annotations = pd.read_csv(
                f'{self.workspace_path}/models/grt123/bbox_result/trained_summit/summit/{self.flavour}/{self.flavour}_metadata.csv',
                usecols=METADATA_COLUMNS
            )

            self.results = (
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
            
            self.results = pd.read_csv(
                f'{self.workspace_path}/models/detection/result/trained_summit/summit/{self.flavour}/predictions.csv',
                usecols=['name', 'col', 'row', 'index', 'diameter', 'threshold']
            )

        elif self.model == 'ticnet':

            _annotations = (
                pd.read_csv(f'{self.workspace_path}/../TiCNet-main/annotations/summit/{self.flavour}/test_metadata.csv').rename(columns={
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
                pd.read_csv(f'{self.workspace_path}/metadata/summit/{self.flavour}/test_metadata.csv')
                .assign(name=lambda df: df.participant_id + '_Y0_BASELINE_A')
                .assign(name_counter=lambda df: df.groupby('name').cumcount() + 1)
            )

            annotations = pd.merge(_metadata, _annotations, on=['name', 'name_counter'], how='outer')[METADATA_COLUMNS]

            assert annotations.shape[0] == _metadata.shape[0], 'Mismatch in number of annotations'

            self.results = pd.read_csv(f'{self.workspace_path}/../TiCNet-main/results/summit/{self.flavour}/res/110/FROC/submission_rpn.csv').rename(
                columns={
                    'seriesuid' : 'name',
                    'coordX' : 'row',
                    'coordY' : 'col',
                    'coordZ' : 'index',
                    'diameter_mm' : 'diameter',
                    'probability' : 'threshold'
                }
            )


        self.scan_metadata = pd.read_csv(f'{self.workspace_path}/metadata/summit/{self.flavour}/test_scans_metadata.csv',
                                        usecols=['Y0_PARTICIPANT_DETAILS_main_participant_id', 'participant_details_gender','lung_health_check_demographics_race_ethnicgroup']).rename(
                                        columns={'Y0_PARTICIPANT_DETAILS_main_participant_id': 'StudyId', 'participant_details_gender' : 'gender', 'lung_health_check_demographics_race_ethnicgroup' : 'ethnic_group'}).assign(
                                        Name=lambda x: x['StudyId'] + '_Y0_BASELINE_A')

        # Reduce the annotations to only actionable cases
        if self.actionable:
            self.annotations = annotations[annotations['management_plan'].isin(['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1'])]
            self.annotations_excluded = annotations[annotations['management_plan']=='RANDOMISATION_AT_YEAR_1']
        else:
            self.annotations = annotations
            self.annotations_excluded = annotations.drop(annotations.index)


    @step
    def resample(self):

        self.output_dir = self.output_dir
        self.scan_metadata = self.scan_metadata
        self.annotations = self.annotations
        self.annotations_excluded = self.annotations_excluded
        self.results = self.results

        self.resample_groups = ['balance_gender', 'balance_ethnic_group']
        self.next(self.calculate_frocs, foreach='resample_groups')


    @step
    def calculate_frocs(self):

        self.output_dir = self.output_dir / self.input
        self.scan_metadata = self.scan_metadata
        self.annotations = self.annotations
        self.annotations_excluded = self.annotations_excluded
        self.results = self.results

        self.resample_group = input

        if self.resample_group == 'balance_gender':
            self.scan_metadata = resample_to_balance_nodule_counts(self.scan_metadata,'gender',250)

        elif self.resample_group == 'balance_ethnic_group':
            self.scan_metadata = resample_to_balance_nodule_counts(self.scan_metadata,'ethnic_group',150)

        # Define the subsequent slices to be performed
        gender_groups = [('gender','MALE'), ('gender', 'FEMALE')]
        ethnic_groups = [('ethnic_group', 'Asian or Asian British'),('ethnic_group','Black'),('ethnic_group','White')]
        self.demographic_groups = [('all', 'all')] + gender_groups + ethnic_groups
        self.next(self.calculate_froc, foreach='demographic_groups')

    @step
    def calculate_froc(self):


            self.output_dir = sample.output_dir
            self.annotations = sample.annotations
            self.annotations_excluded = sample.annotations_excluded
            self.results = sample.results
            self.resample_group = sample.resample_group
            self.scan_metadata = sample.scan_metadata

            cat, val = self.input

            if cat == 'all':
                scans = self.scan_metadata['Name']
            else:
                scans = self.scan_metadata[self.scan_metadata[cat] == val]['Name']

            annotations = self.annotations[self.annotations['name'].isin(scans.values)]
            exclusions = self.annotations_excluded[self.annotations_excluded['name'].isin(scans.values)]
            predictions = self.results[self.results['name'].isin(scans.values)]

            self.number_scans = scans.shape[0]
            self.number_annotations = annotations.shape[0]
            self.number_exclusions = exclusions.shape[0]
            self.number_predictions = predictions.shape[0]

            with TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                scans.to_csv(temp_dir / 'scans.csv', index=False)
                annotations.to_csv(temp_dir / 'annotations.csv', index=False)
                exclusions.to_csv(temp_dir / 'exclusions.csv', index=False)
                predictions.to_csv(temp_dir / 'predictions.csv', index=False)
                output_path = Path(f'{self.output_dir}/{val}')

                self.froc_metrics = noduleCADEvaluation(
                    annotations_filename=temp_dir / 'annotations.csv',
                    annotations_excluded_filename=temp_dir / 'exclusions.csv', 
                    seriesuids_filename=temp_dir / 'scans.csv',
                    results_filename=temp_dir / 'predictions.csv',
                    filter=f'Model: {self.model}, \nDataset: {self.flavour}, \nDemographic: {cat} - {val}, \nActionable Only: {self.actionable}',
                    outputDir=output_path
                )

            self.cat = cat
            self.val = val
            _, self.cpm_summary = caluclate_cpm_from_bootstrapping(output_path / 'froc_predictions_bootstrapping.csv')

        self.next(self.join_froc)

    @step
    def join(self, inputs):

        self.cpm_summary = {inp.val : inp.cpm_summary for inp in inputs}
        self.next(self.join_sample)

    @step
    def join_sample(self, inputs):

        self.
        

    @step
    def end(self):
        pass


if __name__ == '__main__':
    FROCFlow()