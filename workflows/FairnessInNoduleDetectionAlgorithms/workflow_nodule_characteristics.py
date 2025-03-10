import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from metaflow import FlowSpec, Parameter, conda_base, step

if sys.platform == "darwin":
    sys.path.append('/Users/john/Projects/SOTAEvaluationNoduleDetection/utilities')
    sys.path.append('/Users/john/Projects/SOTAEvaluationNoduleDetection/notebooks')
elif sys.platform == "linux":
    sys.path.append('/home/jmccabe/Projects/SOTAEvaluationNoduleDetection/utilities')
    sys.path.append('/home/jmccabe/Projects/SOTAEvaluationNoduleDetection/notebooks')
else:
    raise EnvironmentError("Unsupported platform")

from evaluation import noduleCADEvaluation
from FairnessInNoduleDetectionAlgorithms.utils import (get_thresholds,
                                                       miss_anaysis_at_fpps)
from summit_utils import *
from utils import load_data


def cleanup(s):
    if s == 'CALCIFIED':
         return 'Consolidation'
    
    s = s.replace('_', ' ')
    s = ' '.join(word.capitalize() for word in s.split())
    return s

class NoduleCharacteristicsFlow(FlowSpec):
    """
    Calculate the free response operating characteristic (FROC) scores for each demographic group
    """

    model = Parameter('model', help='Model to evaluate', default='ticnet')
    flavour = Parameter('flavour', help='Flavour to evaluate', default='test_balanced')
    actionable = Parameter('actionable', type=bool, help='Only include actionable cases', default=True)
    dataset = Parameter('dataset', help='Dataset to evaluate', default='lsut')
    n_bootstraps = Parameter('bootstraps', help='Number of bootstraps to perform', default=1000)

    if sys.platform == "darwin":
        workspace_path = '/Users/john/Projects/SOTAEvaluationNoduleDetection'
    elif sys.platform == "linux":
        workspace_path = '/home/jmccabe/Projects/SOTAEvaluationNoduleDetection'
    
    
    @step
    def start(self):
        
        self.output_dir = f'{self.workspace_path}/workflows/FairnessInNoduleDetectionAlgorithms/results/{self.dataset}/{self.model}/{self.flavour}/{"Actionable" if self.actionable else "All"}/FROC'

        self.annotations, self.results, self.scan_metadata, self.annotations_excluded = load_data(
            self.workspace_path, self.model, self.dataset, self.flavour, self.actionable
        )

        self.training_data_path = f'{self.workspace_path}/metadata/summit/{self.flavour}/training_metadata.csv'

        # Reduce the annotations to only actionable cases
        self.annotations['diameter_cats'] = pd.cut(
            self.annotations['nodule_diameter_mm'], 
            bins=[0, 6, 8, 30, 40, 999], 
            labels=['0-6mm', '6-8mm', '8-30mm', '30-40mm', '40+mm']
        )

        self.next(self.miss_analysis)

    @step
    def miss_analysis(self):

        self.output_dir = self.output_dir
        self.training_data_path = self.training_data_path

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

            thresholds = get_thresholds(froc_metrics)

            self.missed_metadata = miss_anaysis_at_fpps(
                model=self.model,
                experiment=self.flavour,
                scans_metadata=self.scan_metadata,
                scans_path=temp_dir_path / 'scans.csv',
                annotations_path=temp_dir_path / 'annotations.csv',
                exclusions_path=temp_dir_path / 'exclusions.csv',
                predictions_path=temp_dir_path / 'predictions.csv',
                thresholds=thresholds,
                output_path=self.output_dir
            )

        self.annotations = annotations
        self.next(self.nodule_characteristics)

    @step
    def nodule_characteristics(self):

        diameter_cats = [0, 6, 8, 30, 40, 999]
        
        diameter_lbs = [
            '0-6mm',
            '6-8mm',
            '8-30mm',
            '30-40mm',
            '40+mm'
        ]        

        nodule_characteristics = {
           'diameter_cats': diameter_lbs,
            'nodule_type': ['SOLID', 'NON_SOLID', 'PART_SOLID', 'CALCIFIED', 'PERIFISSURAL']
        }

        colors = ['red','blue','green','orange','purple','brown','pink']

        for ivx, (var, order) in enumerate(nodule_characteristics.items()):

            training_data = (
                pd.read_csv(self.training_data_path)
                .assign(diameter_cats=lambda df: 
                    pd.cut(
                        df['nodule_diameter_mm'], 
                        bins=diameter_cats,
                        labels=diameter_lbs
                    )            
                ))

            total_vc = training_data[var].value_counts().sort_index().rename('Total Annotations')

            operating_points = ['0.125', '0.25', '0.5', '1', '2', '4', '8']

            results = []

            for idx, metadata in enumerate(self.missed_metadata):
                results.append(
                    pd.crosstab(
                        metadata[var], 
                        metadata['miss'],
                        normalize='index'
                    )[False]
                    .rename(f'{operating_points[idx]}')
                    .reindex(order)
                    
                )

            df = pd.concat(results, axis=1).fillna(0).round(2).merge(total_vc.to_frame(), left_index=True, right_index=True)

            print(df)

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
            total_vc = total_vc[total_vc.index.isin(df.index)].reindex(df.index)

            for isx, column in enumerate(df.T):    
                ax.plot(df.T[column], label=cleanup(column), linestyle=line_styles[isx], color=colors[isx])
                    
            ax.set_xticklabels(labels=df.columns, rotation=45)
            ax.legend()    
            ax.set_xlabel('False positives per scan')
            ax.set_ylim(0, 1)
            ax.set_ylabel('% of detections')
            ax.grid(visible=True, which='both')

            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/images/nodule_characteristics_{var}.png')

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.bar([cleanup(idx) for idx in total_vc.index], total_vc, alpha=0.5, label=cleanup(column), color=colors)
            ax.set_xticklabels(labels=[cleanup(idx) for idx in total_vc.index], rotation=45)
            ax.set_xlabel('Total Number of Category in Training Data')

            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/images/nodule_characteristics_total_{var}.png')

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    NoduleCharacteristicsFlow()