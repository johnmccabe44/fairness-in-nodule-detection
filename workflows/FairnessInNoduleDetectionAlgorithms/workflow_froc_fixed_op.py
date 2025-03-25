import datetime
import json
import logging
import os
import sys
from gc import get_threshold
from math import e
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import scipy.stats as stats
from cv2 import mean
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedFormatter
from metaflow import FlowSpec, IncludeFile, Parameter, conda_base, step
from requests import get

if sys.platform == "darwin":
    sys.path.append('/Users/john/Projects/SOTAEvaluationNoduleDetection/utilities')
    sys.path.append('/Users/john/Projects/SOTAEvaluationNoduleDetection/notebooks')
elif sys.platform == "linux":
    sys.path.append('/home/jmccabe/Projects/SOTAEvaluationNoduleDetection/utilities')
    sys.path.append('/home/jmccabe/Projects/SOTAEvaluationNoduleDetection/notebooks')
else:
    raise EnvironmentError("Unsupported platform")



import os
import sys
import time

from evaluation import noduleCADEvaluation
from FairnessInNoduleDetectionAlgorithms.utils import (
    calculate_cpm_from_bootstrapping, check_ci_across_categories,
    display_plots_with_error_bars, get_thresholds)
from summit_utils import *
from tqdm import tqdm
from utils import load_data


def is_matched(annotation, prediction):
    annotation_center = np.array([annotation['index'], annotation['row'], annotation['col']])
    prediction_center = np.array([prediction['index'], prediction['row'], prediction['col']])
    distance = np.linalg.norm(annotation_center - prediction_center)
    return distance <= annotation['diameter'] / 2

def get_boostrapped_sensitivity(scans, scan_sensitivity_data, n_bootstraps):

    start_time = time.time()

    sensitivity_dict = {threshold: [] for threshold in scan_sensitivity_data[scans[0]].keys()}

    for bootstrap in range(n_bootstraps):
        bootstrapped_scans = np.random.choice(scans, len(scans), replace=True)

        results = {threshold: {'tp': 0, 'fn': 0} for threshold in scan_sensitivity_data[scans[0]].keys()}
        for scan in bootstrapped_scans:
            for threshold in scan_sensitivity_data[scan].keys():
                results[threshold]['tp'] += scan_sensitivity_data[scan][threshold]['tp']
                results[threshold]['fn'] += scan_sensitivity_data[scan][threshold]['fn']

        sensitivity = {threshold: results[threshold]['tp'] / (results[threshold]['tp'] + results[threshold]['fn']) for threshold in results.keys()}
        for threshold in sensitivity.keys():
            sensitivity_dict[threshold].append(sensitivity[threshold])

    mean_sens = {threshold: np.mean(sensitivity_dict[threshold]) for threshold in sensitivity_dict.keys()}
    lower_sens = {threshold: np.percentile(sensitivity_dict[threshold], 2.5) for threshold in sensitivity_dict.keys()}
    upper_sens = {threshold: np.percentile(sensitivity_dict[threshold], 97.5) for threshold in sensitivity_dict.keys()}

    sensitivity_summary = {
        threshold: {
            'mean_sens': mean_sens[threshold],
            'lower_sens': lower_sens[threshold],
            'upper_sens': upper_sens[threshold]
        }
        for threshold in mean_sens.keys()
    }

    end_time = time.time()
    print(f"Time taken to run get_boostrapped_sensitivity: {end_time - start_time} seconds")

    return sensitivity_summary

def get_scan_sensitivity_data(scans, annotations, predictions, thresholds):


    sensitivity_per_scan = {scan: {threshold: {'tp': 0, 'fn': 0} for threshold in thresholds} for scan in scans}

    for scan in scans:
        scan_annotations = annotations[annotations['name'] == scan]
        scan_predictions = predictions[predictions['name'] == scan]

        for threshold in thresholds:
            filtered_predictions = scan_predictions[scan_predictions['threshold'] >= threshold]

            for _, annotation in scan_annotations.iterrows():
                matched = False
                for _, prediction in filtered_predictions.iterrows():
                    if is_matched(annotation, prediction):
                        matched = True
                        sensitivity_per_scan[scan][threshold]['tp'] += 1
                        break
                if not matched:
                    sensitivity_per_scan[scan][threshold]['fn'] += 1

    return sensitivity_per_scan

class FROCFlow(FlowSpec):
    """
    Calculate the free response operating characteristic (FROC) scores for each demographic group
    """

    dataset = Parameter('dataset', help='Dataset to evaluate', default='summit')
    model = Parameter('model', help='Model to evaluate', default='grt123')
    flavour = Parameter('flavour', help='Flavour to evaluate', default='white_only')
    actionable = Parameter('actionable', type=bool, help='Only include actionable cases', default=True)
    n_bootstraps = Parameter('bootstraps', help='Number of bootstraps to perform', default=1000)

    if sys.platform == "darwin":
        workspace_path = '/Users/john/Projects/SOTAEvaluationNoduleDetection'
    elif sys.platform == "linux":
        workspace_path = '/home/jmccabe/Projects/SOTAEvaluationNoduleDetection'
    
    @step
    def start(self):

        print(self.model, self.flavour, self.actionable)

        self.output_dir = f'{self.workspace_path}/workflows/FairnessInNoduleDetectionAlgorithms/results/{self.dataset}/{self.model}/{self.flavour}/{"Actionable" if self.actionable else "All"}/FROC'

        self.annotations, self.results, self.scan_metadata, self.annotations_excluded = load_data(
            self.workspace_path, self.model, self.dataset, self.flavour, self.actionable
        )

        self.next(self.calculate_froc)

    @step
    def calculate_froc(self):

        self.output_dir = self.output_dir

        scans = self.scan_metadata['Name']

        self.scan_metadata = self.scan_metadata
        self.annotations = self.annotations[self.annotations['name'].isin(scans.values)]
        self.exclusions = self.annotations_excluded[self.annotations_excluded['name'].isin(scans.values)]
        self.predictions = self.results[self.results['name'].isin(scans.values)]

        self.number_scans = scans.shape[0]
        self.number_annotations = self.annotations.shape[0]
        self.number_exclusions = self.exclusions.shape[0]
        self.number_predictions = self.predictions.shape[0]

        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            scans.to_csv(temp_dir / 'scans.csv', index=False)
            self.annotations.to_csv(temp_dir / 'annotations.csv', index=False)
            self.exclusions.to_csv(temp_dir / 'exclusions.csv', index=False)
            self.predictions.to_csv(temp_dir / 'predictions.csv', index=False)
            output_path = temp_dir

            froc_metrics = noduleCADEvaluation(
                annotations_filename=temp_dir / 'annotations.csv',
                annotations_excluded_filename=temp_dir / 'exclusions.csv', 
                seriesuids_filename=temp_dir / 'scans.csv',
                results_filename=temp_dir / 'predictions.csv',
                filter='temp',
                outputDir=output_path
            )

        self.thresholds = get_thresholds(froc_metrics)

        self.scan_sensitivity_data = get_scan_sensitivity_data(scans.values, self.annotations, self.predictions, self.thresholds)

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
            self.demographic_groups = gender_groups[self.dataset] + ethnic_groups[self.dataset]
            
        elif self.flavour == 'male_only':
            self.demographic_groups = ethnic_groups[self.dataset]

        elif self.flavour == 'white_only':
            self.demographic_groups = gender_groups[self.dataset]

        self.next(self.calculate_demographic_splits, foreach='demographic_groups')

    @step
    def calculate_demographic_splits(self):

        self.group = self.input[0]
        self.category = self.input[1]

        scans = self.scan_metadata[self.scan_metadata[self.group] == self.category]['Name'].values

        self.bootstrapped_sensitivity = get_boostrapped_sensitivity(
            scans, self.scan_sensitivity_data, self.n_bootstraps
        )

        self.next(self.join)

    @step
    def join(self, inputs):
        
        self.bootstrapped_sensitivity = {}
        for inp in inputs:
            self.bootstrapped_sensitivity[inp.category] = inp.bootstrapped_sensitivity

        with open(f'{self.workspace_path}/workflows/FairnessInNoduleDetectionAlgorithms/results/{self.dataset}/{self.model}_{self.flavour}_froc_fixed_thresholds.json', 'w') as f:
            f.write(json.dumps(self.bootstrapped_sensitivity))    

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    FROCFlow()