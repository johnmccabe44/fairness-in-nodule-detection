import os
import sys
from math import e
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from metaflow import FlowSpec, Parameter, conda_base, step
from scipy.ndimage import distance_transform_edt

if os.path.basename(os.getcwd()).upper() == 'SOTAEVALUATIONNODULEDETECTION':
    sys.path.append('utilities')
    sys.path.append('notebooks')
else:
    sys.path.append('../../utilities')
    sys.path.append('../../notebooks')

import os
import sys

from evaluation import noduleCADEvaluation
from utils import get_thresholds, load_data, miss_anaysis_at_fpps


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
    workspace_path = Parameter(
        'workspace_path', 
        type=str, 
        help='Path to the workspace data', 
        default='/Users/john/Projects/SOTAEvaluationNoduleDetection'
    )
    
    @step
    def start(self):
    
        self.models = ['grt123', 'detection', 'ticnet']
        # self.models = ['detection']
        self.next(self.get_missed_annotations, foreach='models')

    @step
    def get_missed_annotations(self):

        self.model = self.input

        print(f'Processing {self.model} model')

        annotations, results, scan_metadata, annotations_excluded = load_data(self.workspace_path, self.model, self.flavour, self.actionable)

        scans = scan_metadata['Name'].values

        with TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            np.savetxt(temp_dir_path / 'scans.csv', scans, delimiter=',', fmt='%s')
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

        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    MissedNodulesFlow()