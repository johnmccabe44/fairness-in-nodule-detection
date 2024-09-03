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

sys.path.append('../utilities')
sys.path.append('../notebooks')

from FairnessInNoduleDetectionAlgorithms.utils import caluclate_cpm_from_bootstrapping, display_plots_with_error_bars
from summit_utils import *
from evaluation import noduleCADEvaluation


class FROCFlow(FlowSpec):
    """
    Calculate the free response operating characteristic (FROC) scores for each demographic group
    """

    model = Parameter('model', help='Model to evaluate')
    flavour = Parameter('flavour', help='Flavour to evaluate')
    actionable = Parameter('actionable', help='Only include actionable cases', default=True)
    n_bootstraps = Parameter('bootstraps', help='Number of bootstraps to perform', default=1000)

    workspace_path = Path(os.getcwd()).parent.as_posix()
    
    @step
    def start(self):
        
        # Load the data
        if self.model == 'grt123':
            self.annotations = pd.read_csv(f'{self.workspace_path}/models/grt123/bbox_result/trained_summit/summit/{self.flavour}/{self.flavour}_metadata.csv', usecols=['name', 'col', 'row', 'index', 'diameter', 'management_plan', 'gender', 'ethnic_group'])
            self.annotations_excluded = pd.read_csv(f'{self.workspace_path}/data/summit/metadata/grt123_annotations_excluded_empty.csv', usecols=['name', 'col', 'row', 'index', 'diameter'])
            self.results = pd.read_csv(f"{self.workspace_path}/models/grt123/bbox_result/trained_summit/summit/{self.flavour}/{self.flavour}_predictions.csv", usecols=['name', 'col', 'row', 'index', 'diameter', 'threshold']).rename(columns={'threshold': 'threshold_original'}).assign(threshold=lambda x: 1 / (1 + np.exp(-x['threshold_original'])))
            
        elif self.model == 'detection':
            self.annotations = pd.read_csv(f'{self.workspace_path}/models/detection/result/trained_summit/summit/{self.flavour}/annotations.csv', usecols=['name', 'col', 'row', 'index', 'diameter', 'management_plan','gender','ethnic_group'])
            self.annotations_excluded = pd.read_csv(f'{self.workspace_path}/data/summit/metadata/detection_annotations_excluded_empty.csv', usecols=['name', 'col', 'row', 'index', 'diameter'])
            self.results_filename = pd.read_csv(f"{self.workspace_path}/models/detection/result/trained_summit/summit/{self.flavour}/predictions.csv", usecols=['name', 'col', 'row', 'index', 'diameter', 'threshold'])

        self.scan_metadata = pd.read_csv(f'{self.workspace_path}/metadata/summit/{self.flavour}/test_scans_metadata.csv',
                                        usecols=['Y0_PARTICIPANT_DETAILS_main_participant_id', 'participant_details_gender','lung_health_check_demographics_race_ethnicgroup']).rename(
                                        columns={'Y0_PARTICIPANT_DETAILS_main_participant_id': 'StudyId', 'participant_details_gender' : 'gender', 'lung_health_check_demographics_race_ethnicgroup' : 'ethnic_group'}).assign(
                                        Name=lambda x: x['StudyId'] + '_Y0_BASELINE_A')

        # Reduce the annotations to only actionable cases
        if self.actionable:
            self.annotations = self.annotations[self.annotations['management_plan'].isin(['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1'])]

        # Define the subsequent slices to be performed
        gender_groups = [('gender','MALE'), ('gender', 'FEMALE')]
        ethnic_groups = [('ethnic_group', 'Asian or Asian British'),('ethnic_group','Black'),('ethnic_group','White')]

        if self.flavour == 'test_balanced':
            self.demographic_groups = [('all', 'all')] + gender_groups + ethnic_groups

        elif self.flavour == 'male_only':
            self.demographic_groups = [('all', 'all')] + ethnic_groups

        elif self.flavour == 'balanced_white_only':
            self.demographic_groups = [('all', 'all')] + gender_groups

        self.next(self.calculate_froc, foreach='demographic_groups')

    @step
    def calculate_froc(self):
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
            output_path = Path(f'results/FairnessInNoduleDetectionAlgorithms/FROC/{val}')

            self.froc_metrics = noduleCADEvaluation(
                annotations_filename=temp_dir / 'annotations.csv',
                annotations_excluded_filename=temp_dir / 'exclusions.csv', 
                seriesuids_filename=temp_dir / 'scans.csv',
                results_filename=temp_dir / 'predictions.csv',
                filter=f'Model: {self.model}, Dataset: {self.flavour}, Demographic: {cat} - {val}, Actionable Only: {self.actionable}',
                outputDir=output_path
            )

        self.cat = cat
        self.val = val
        self.cpm_data = caluclate_cpm_from_bootstrapping(output_path / 'froc_predictions_bootstrapping.csv').set_index('fps')
        self.boot_metrics = (
            pd.read_csv(output_path / 'froc_predictions_bootstrapping.csv')
            .rename(columns={
                'FPrate': 'FPRate',
                'Sensivity[Mean]': 'Sensitivity',
                'Sensivity[Lower bound]': 'LowSensitivity',
                'Sensivity[Upper bound]': 'HighSensitivity'
            })
        )

        self.next(self.join)

    @step
    def join(self, inputs):

        self.froc_metrics = {inp.val : inp.froc_metrics for inp in inputs}
        self.cpm_data = {inp.val : inp.cpm_data for inp in inputs}
        self.boot_metrics = {inp.val : inp.boot_metrics for inp in inputs}

        self.next(self.charting)

    @step
    def charting(self):
        self.froc_metrics = self.froc_metrics
        self.cpm_data = self.cpm_data
        self.boot_metrics = self.boot_metrics

        # Define charts to be generated
        self.demographic_groups = [
            ('gender' , ['MALE','FEMALE']),
            ('ethnic_group' , ['Asian or Asian British','Black','White'])
        ]

        self.next(self.chart, foreach='demographic_groups')

    @step
    def chart(self):
        demographic_group = self.input[0]
        demographic_categories = self.input[1]


        display_plots_with_error_bars(
            model=self.model,
            flavour=self.flavour, 
            actionable=self.actionable,
            protected_group=demographic_group,
            categories=demographic_categories,
            sensitivity_data=self.cpm_data,
            output_path=Path(f'results/FairnessInNoduleDetectionAlgorithms/FROC/images')
            )

        self.next(self.join_chart)

    @step
    def join_chart(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    FROCFlow()