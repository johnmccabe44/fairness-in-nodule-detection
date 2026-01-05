import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import scipy.stats as stats
from evaluation import noduleCADEvaluation
from FairnessInNoduleDetectionAlgorithms.utils import (get_thresholds,
                                                       miss_anaysis_at_fpps)
from metaflow import FlowSpec, Parameter, step

sys.append('/Users/john/Projects/SOTAEvaluationNoduleDetection/utilities')

from summit_utils import *
from utils import load_data

from models.ticnet import dataset


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
    dataset = Parameter('dataset', help='Dataset to evaluate', default='lsut')
    workspace_path = '/Users/john/Projects/SOTAEvaluationNoduleDetection'
    
    @step
    def start(self):
    
        # self.models = ['grt123', 'detection', 'ticnet']
        self.models = ['grt123', 'detection']
        self.next(self.get_missed_annotations, foreach='models')

    @step
    def get_missed_annotations(self):

        self.model = self.input

        print(f'Processing {self.model} model')

        annotations, predictions, scan_metadata, annotations_excluded = load_data(
            self.workspace_path, self.model, self.dataset, self.flavour, self.actionable
        )

        # Reduce the annotations to only actionable cases
        diameter_var = 'nodule_diameter_mm' if self.dataset == 'summit' else 'diameter_mm'

        annotations['diameter_cats'] = pd.cut(
            annotations[diameter_var], 
            bins=[0, 6, 8, 30, 40, 999], 
            labels=['0-6mm', '6-8mm', '8-30mm', '30-40mm', '40+mm']
        )

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