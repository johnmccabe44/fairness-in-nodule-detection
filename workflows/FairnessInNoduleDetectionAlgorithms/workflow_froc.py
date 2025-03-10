import datetime
import json
import logging
import os
import sys
from math import e
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedFormatter
from metaflow import FlowSpec, IncludeFile, Parameter, conda_base, step

sys.path.append('/Users/john/Projects/SOTAEvaluationNoduleDetection/utilities')
sys.path.append('/Users/john/Projects/SOTAEvaluationNoduleDetection/notebooks')

import os
import sys

from evaluation import noduleCADEvaluation
from FairnessInNoduleDetectionAlgorithms.utils import (
    calculate_cpm_from_bootstrapping, display_plots_with_error_bars)
from summit_utils import *
from utils import load_data


class FROCFlow(FlowSpec):
    """
    Calculate the free response operating characteristic (FROC) scores for each demographic group
    """

    dataset = Parameter('dataset', help='Dataset to evaluate', default='lsut')
    model = Parameter('model', help='Model to evaluate', default='ticnet')
    flavour = Parameter('flavour', help='Flavour to evaluate', default='test_balanced')
    actionable = Parameter('actionable', type=bool, help='Only include actionable cases', default=True)
    n_bootstraps = Parameter('bootstraps', help='Number of bootstraps to perform', default=1000)

    workspace_path = '/Users/john/Projects/SOTAEvaluationNoduleDetection'
    
    @step
    def start(self):

        print(self.model, self.flavour, self.actionable)

        self.output_dir = f'results/{self.dataset}/{self.model}/{self.flavour}/{"Actionable" if self.actionable else "All"}/FROC'

        self.annotations, self.results, self.scan_metadata, self.annotations_excluded = load_data(
            self.workspace_path, self.model, self.dataset, self.flavour, self.actionable
        )

        # Define the subsequent slices to be performed
        gender_groups = {
            'summit' : [('gender','MALE'), ('gender', 'FEMALE')],
            'lsut' : [('gender','Male'), ('gender', 'Female')],
        }

        ethnic_groups = {
            'summit' : [('ethnic_group', 'Asian or Asian British'),('ethnic_group','Black'),('ethnic_group','White')],
            'lsut' : [('ethnic_group', 'Other'),('ethnic_group','White')],
        }

        if self.flavour == 'test_balanced':
            self.demographic_groups = [('all', 'all')] + gender_groups[self.dataset] + ethnic_groups[self.dataset]

        elif self.flavour == 'male_only':
            self.demographic_groups = [('all', 'all')] + ethnic_groups[self.dataset]

        elif self.flavour == 'white_only':
            self.demographic_groups = [('all', 'all')] + gender_groups[self.dataset]

        self.next(self.calculate_froc, foreach='demographic_groups')

    @step
    def calculate_froc(self):
        cat, val = self.input

        self.output_dir = self.output_dir

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
        self.cpm_data, self.cpm_summary = calculate_cpm_from_bootstrapping(output_path / 'froc_predictions_bootstrapping.csv')
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

        self.output_dir = inputs[0].output_dir

        self.froc_metrics = {inp.val : inp.froc_metrics for inp in inputs}
        self.cpm_data = {inp.val : inp.cpm_data for inp in inputs}
        self.cpm_summary = {inp.val : inp.cpm_summary for inp in inputs}

        if self.dataset == 'summit':
            order_by = ['MALE','FEMALE','Asian or Asian British','Black','White','all']
            
        elif self.dataset == 'lsut':
            order_by = ['Male', 'Female', 'White', 'Other', 'all']

        pd.DataFrame.from_dict(self.cpm_summary, orient='index').reindex(order_by).to_csv(f'{self.output_dir}/cpm_summary.csv')

        self.boot_metrics = {inp.val : inp.boot_metrics for inp in inputs}
        self.bootstap_results = {inp.val : inp.bootstap_results for inp in inputs}

        self.next(self.charting)

    @step
    def charting(self):

        self.output_dir = self.output_dir

        self.froc_metrics = self.froc_metrics
        self.cpm_data = self.cpm_data
        self.boot_metrics = self.boot_metrics
        self.bootstap_results = self.bootstap_results

        groupings = {
            'summit' : {
                'gender' : ['MALE','FEMALE'],
                'ethnic_group' : ['Asian or Asian British','Black','White']
            },
            'lsut' : {
                'gender' : ['Male', 'Female'],
                'ethnic_group' : ['White','Other']
            }
        }

        # Define charts to be generated
        if self.flavour == 'test_balanced':            
            self.demographic_groups = [
                ('gender' , groupings[self.dataset]['gender']),
                ('ethnic_group' , groupings[self.dataset]['ethnic_group'])
            ]
        elif self.flavour == 'male_only':
            self.demographic_groups = [
                ('ethnic_group' , groupings[self.dataset]['ethnic_group'])
            ]

        elif self.flavour == 'white_only':
            self.demographic_groups = [
                ('gender' , groupings[self.dataset]['gender'])
            ]

        self.next(self.chart, foreach='demographic_groups')

    @step
    def chart(self):

        self.output_dir = self.output_dir

        demographic_group = self.input[0]
        demographic_categories = self.input[1]

        display_plots_with_error_bars(
            model=self.model,
            flavour=self.flavour, 
            actionable=self.actionable,
            protected_group=demographic_group,
            categories=demographic_categories,
            sensitivity_data=self.cpm_data,
            bootstrap_results=self.bootstap_results,
            output_path=Path(f'{self.output_dir}/images')
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