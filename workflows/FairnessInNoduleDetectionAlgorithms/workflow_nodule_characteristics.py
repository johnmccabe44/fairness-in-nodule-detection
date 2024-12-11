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

class NoduleCharacteristicsFlow(FlowSpec):
    """
    Calculate the free response operating characteristic (FROC) scores for each demographic group
    """

    model = Parameter('model', help='Model to evaluate', default='grt123')
    flavour = Parameter('flavour', help='Flavour to evaluate', default='test_balanced')
    actionable = Parameter('actionable', type=bool, help='Only include actionable cases', default=False)
    n_bootstraps = Parameter('bootstraps', help='Number of bootstraps to perform', default=1000)

    if os.path.basename(os.getcwd()).upper() == 'SOTAEVALUATIONNODULEDETECTION':
        workspace_path = Path(os.getcwd()).as_posix()
    else:
        workspace_path = Path(os.getcwd()).parent.as_posix()
    

    @step
    def start(self):
        
        self.output_dir = f'results/summit/{self.model}/{self.flavour}/{"Actionable" if self.actionable else "All"}/FROC'

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

            self.results = pd.read_csv(f'/Users/john/Projects/TiCNet-main/results/summit/{self.flavour}/res/110/FROC/submission_rpn.csv').rename(
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
        self.training_data_path = f'{self.workspace_path}/metadata/summit/{self.flavour}/training_metadata.csv'


        # Reduce the annotations to only actionable cases

        annotations['diameter_cats'] = pd.cut(
            annotations['nodule_diameter_mm'], 
            bins=[0, 6, 8, 30, 40, 999], 
            labels=['0-6mm', '6-8mm', '8-30mm', '30-40mm', '40+mm']
        )

        if self.actionable:
            self.annotations = annotations[annotations['management_plan'].isin(['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1'])]
            self.annotations_excluded = annotations[annotations['management_plan']=='RANDOMISATION_AT_YEAR_1']
        else:
            self.annotations = annotations
            self.annotations_excluded = annotations.drop(annotations.index)

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