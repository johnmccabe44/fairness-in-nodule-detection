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
from cv2 import mean
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedFormatter
from metaflow import FlowSpec, IncludeFile, Parameter, conda_base, step
from requests import get
from sklearn.metrics import confusion_matrix

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

from evaluation import noduleCADEvaluation
from FairnessInNoduleDetectionAlgorithms.utils import get_thresholds
from summit_utils import *
from utils import load_data


def is_match(annotation, prediction):

    annotation_row, annotation_col, annotation_index, annotation_diameter = annotation
    prediction_row, prediction_col, prediction_index, _ = prediction

    radius = annotation_diameter / 2

    distance = np.sqrt(
        (annotation_row - prediction_row) ** 2 +
        (annotation_col - prediction_col) ** 2 +
        (annotation_index - prediction_index) ** 2
    )

    return distance <= radius

def calculate_fairness_metrics(scan_results, n_bootstrap=1000):

    scans = list(scan_results.keys())
    fairness_results = []
    for fpps_rate in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
        tp, tn, fp, fn = 0, 0, 0, 0            
        
        for scan in scans:
            scan_data = scan_results[scan][fpps_rate]

            tp += scan_data['TP']
            tn += scan_data['TN']
            fp += scan_data['FP']
            fn += scan_data['FN']

        # Calculate tpr, fpr, ppv, pred_rate
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        pred_rate = (tp + fp) / len(scans) if len(scans) > 0 else 0
        fairness_results.append({
            'fpps_rate' : scan_results[scan][fpps_rate]['fpps_rate'],
            'threshold' : scan_results[scan][fpps_rate]['threshold'],            
            'tpr' : tpr,
            'fpr' : fpr,
            'ppv' : ppv,
            'pred_rate' : pred_rate,
            'tp' : tp,
            'tn' : tn,
            'fp' : fp,
            'fn' : fn,
            'counts' : float(sum([tp, tn, fp, fn]))
        })

    # Calculate the confidence intervals
    fairness_data = pd.DataFrame(fairness_results)

    return fairness_data

def calculate_bootstrapped_fairness_metrics(scan_results, n_bootstrap=1000):

    scans = list(scan_results.keys())
    
    bootstrap_results = []
    for bootstrap in range(n_bootstrap):
        # For each scan check whether it is TP, TN, FP, FN
        bootstrap_scans = np.random.choice(scans, size=len(scans), replace=True)

        for fpps_rate in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
            tp, tn, fp, fn = 0, 0, 0, 0            

            for scan in bootstrap_scans:
                scan_data = scan_results[scan][fpps_rate]

                tp += scan_data['TP']
                tn += scan_data['TN']
                fp += scan_data['FP']
                fn += scan_data['FN']

            # Calculate tpr, fpr, ppv, pred_rate
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            pred_rate = (tp + fp) / len(bootstrap_scans) if len(bootstrap_scans) > 0 else 0
            bootstrap_results.append({
                'fpps_rate' : scan_results[scan][fpps_rate]['fpps_rate'],
                'threshold' : scan_results[scan][fpps_rate]['threshold'],
                'bootstrap_id' : bootstrap,                
                'tpr' : tpr,
                'fpr' : fpr,
                'ppv' : ppv,
                'pred_rate' : pred_rate,
                'counts' : float(sum([tp, tn, fp, fn]))
            })

    # Calculate the confidence intervals
    bootstrap_data = pd.DataFrame(bootstrap_results)

    return bootstrap_data

class FairnessFlow(FlowSpec):
    """
    Calculate the free response operating characteristic (FROC) scores for each demographic group
    """

    dataset = Parameter('dataset', help='Dataset to evaluate', default='summit')
    model = Parameter('model', help='Model to evaluate', default='grt123')
    flavour = Parameter('flavour', help='Flavour to evaluate', default='test_balanced')
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

        self.next(self.get_thresholds)

    @step
    def get_thresholds(self):
        """
        Get the thresholds for each demographic group
        """

        scans = self.scan_metadata['Name']

        annotations = self.annotations[self.annotations['name'].isin(scans.values)]
        exclusions = self.annotations_excluded[self.annotations_excluded['name'].isin(scans.values)]
        predictions = self.results[self.results['name'].isin(scans.values)]

        self.number_scans = scans.shape[0]
        self.number_annotations = annotations.shape[0]
        self.number_exclusions = exclusions.shape[0]
        self.number_predictions = predictions.shape[0]

        with TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            scans.to_csv(temp_dir_path / 'scans.csv', index=False)
            annotations.to_csv(temp_dir_path / 'annotations.csv', index=False)
            exclusions.to_csv(temp_dir_path / 'exclusions.csv', index=False)
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

        fpps_rates = [0.125, 0.25, 0.5, 1, 2, 4, 8]

        self.thresholds = [
            (fpps_rate, fpps_thresh)
            for fpps_rate, fpps_thresh in zip(fpps_rates, get_thresholds(froc_metrics))
        ]

        self.next(self.intiate_fairness_loop)

    @step
    def intiate_fairness_loop(self):
        """
        Calculate the FROC scores for each demographic group
        """

        # For each scan check whether it is TP, TN, FP, FN
        self.scan_results = {
            scan['Name']: {
                'gender': scan['gender'],
                'ethnic_group': scan['ethnic_group'],
                **{
                    fpps_rate: {
                        'fpps_rate': fpps_rate,
                        'threshold': threshold,
                        'TP': 0,
                        'TN': 0,
                        'FP': 0,
                        'FN': 0,
                        'Annotations': 0,
                        'Predictions': 0,
                        'Matches': 0
                    }
                    for fpps_rate, threshold in self.thresholds
                }
            }
            for _, scan in self.scan_metadata.iterrows()
        }

        for (fpps_rate, threshold) in self.thresholds:
            
            scans = self.scan_metadata['Name']
            annotations = self.annotations[self.annotations['name'].isin(scans.values)]
            predictions = self.results[self.results['name'].isin(scans.values)&(self.results['threshold'] > threshold)]

            annotations_dict = annotations.groupby('name').apply(lambda x: x[['col', 'row', 'index', 'diameter']].values.tolist()).to_dict()
            predictions_dict = predictions.groupby('name').apply(lambda x: {'boxes': x[['col', 'row', 'index', 'diameter']].values.tolist(), 'scores': x['threshold'].values.tolist()}).to_dict()


            for scan in scans:
                annotations = annotations_dict.get(scan, [])
                predictions = predictions_dict.get(scan, {'boxes': [], 'scores': []})['boxes']

                self.scan_results[scan]['Annotations'] = len(annotations)
                self.scan_results[scan]['Predictions'] = len(predictions)

                annotation_matches = {adx : False for adx in range(len(annotations))}
                for adx, annotation in enumerate(annotations):                    
                    for prediction in predictions:
                        if is_match(annotation, prediction):
                            annotation_matches[adx] = True
                            break

                if annotations:
                    self.scan_results[scan][fpps_rate]['Matches'] = sum(annotation_matches.values())
                    if any(annotation_matches.values()):
                        self.scan_results[scan][fpps_rate]['TP'] = 1
                    else:
                        self.scan_results[scan][fpps_rate]['FN'] = 1

                else:
                    if predictions:
                        self.scan_results[scan][fpps_rate]['FP'] = 1
                    else:
                        self.scan_results[scan][fpps_rate]['TN'] = 1


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
                self.map_categories = gender_groups[self.dataset] + ethnic_groups[self.dataset]

            elif self.flavour == 'male_only':
                self.map_categories = ethnic_groups[self.dataset]

            elif self.flavour == 'white_only':
                self.map_categories = gender_groups[self.dataset]

        self.next(self.calculate_fairness, foreach='map_categories')

    @step
    def calculate_fairness(self):
        """
        Calculate the FROC scores for each demographic group
        """

        self.group, self.cat = self.input

        scan_results = {
            key : scan_data
            for key, scan_data in self.scan_results.items()
            if scan_data[self.group] == self.cat
        }

        self.fairness_data = calculate_fairness_metrics(scan_results)


        self.bootstrap_data = calculate_bootstrapped_fairness_metrics(
            scan_results,
            n_bootstrap=1000
        )

        self.next(self.join_fairness)

    @step
    def join_fairness(self, inputs):
        """
        Join the results from the different demographic groups
        """

        fairness_data = pd.concat([
            inp.fairness_data.assign(
                dataset=self.dataset,
                model=self.model,
                flavour=self.flavour,
                actionable=self.actionable,
                group=inp.group,
                cat=inp.cat,
            )
            for inp in inputs
        ])

        fairness_data.to_csv(
            f'{self.workspace_path}/workflows/FairnessInNoduleDetectionAlgorithms/results/{self.dataset}/{self.model}_{self.flavour}_fairness_data.csv',
            index=False
        )

        bootstrap_data = pd.concat([
            inp.bootstrap_data.assign(
                dataset=self.dataset,
                model=self.model,
                flavour=self.flavour,
                actionable=self.actionable,
                group=inp.group,
                cat=inp.cat,
            )
            for inp in inputs
        ])

        bootstrap_data.to_csv(
            f'{self.workspace_path}/workflows/FairnessInNoduleDetectionAlgorithms/results/{self.dataset}/{self.model}_{self.flavour}_bootstrap_data.csv',
            index=False
        )

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    flow = FairnessFlow()



        # @step
        # def calculate_fairness(self):
        #     """
        #     Calculate the FROC scores for each demographic group
        #     """

        #     # Get the demographic group
    
        #     group, cat = self.input

        #     if cat == 'all':
        #         scans = self.scan_metadata['Name']
        #     else:
        #         scans = self.scan_metadata[self.scan_metadata[group] == cat]['Name']

            

        #     self.next(self.join)