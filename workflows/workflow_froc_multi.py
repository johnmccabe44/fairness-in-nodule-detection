import copy
from math import e
import math
from tempfile import TemporaryDirectory
from networkx import diameter
import numpy as np
from metaflow import FlowSpec, step, IncludeFile, Parameter, conda_base
import numpy as np
import os
import pandas as pd
from pathlib import Path
import scipy.stats as stats
import sys

from torch import ge, threshold_



if os.path.basename(os.getcwd()).upper() == 'SOTAEVALUATIONNODULEDETECTION':
    sys.path.append('utilities')
    sys.path.append('notebooks')
else:
    sys.path.append('../utilities')
    sys.path.append('../notebooks')

from FairnessInNoduleDetectionAlgorithms.utils import (
    caluclate_cpm_from_bootstrapping,
    display_plots_with_error_bars,
    calculate_cpm
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
    'nodule_type'
]

def get_nodule_level_metrics(scans_metadata, annotations, exclusions, predictions, threshold):
    """
    Calculate the free response operating characteristic (FROC) scores for the given data
    """

    # Define the output path

    predictions_tracker = predictions.index.to_list()
    nodules_found = {adx:0 for adx in annotations.index}
    excluded_found = {edx:0 for edx in exclusions.index}

    for _, scan_metadata in scans_metadata.iterrows():

        seriesuid = scan_metadata['name']

        # Get the annotations for the current scan
        annotations_local = annotations[annotations['name'] == seriesuid]
        exclusions_local = exclusions[exclusions['name'] == seriesuid]
        predictions_local = predictions[predictions['name'] == seriesuid]

        for adx, annotation in annotations_local.iterrows():

            x = annotation['row']
            y = annotation['col']
            z = annotation['index']

            diameter = annotation['diameter'] if annotation['diameter'] > 0 else 10

            for pdx, prediction in predictions_local.iterrows():

                x2 = prediction['row']
                y2 = prediction['col']
                z2 = prediction['index']

                dist = math.pow(x - x2, 2) + math.pow(y - y2, 2) + math.pow(z - z2, 2) 

                if dist < diameter:
                    if prediction['threshold'] >= threshold:

                        if pdx not in predictions_tracker:
                            print(f'Prediction {pdx} already aligned with nodule!')
                        else:
                            predictions_tracker.remove(pdx)

                        nodules_found[adx] += 1
                       
        for edx, exclusion in exclusions_local.iterrows():
                
                x = exclusion['row']
                y = exclusion['col']
                z = exclusion['index']
    
                diameter = exclusion['diameter'] if exclusion['diameter'] > 0 else 10
    
                for pdx, prediction in predictions_local.iterrows():
    
                    x2 = prediction['row']
                    y2 = prediction['col']
                    z2 = prediction['index']
    
                    dist = math.pow(x - x2, 2) + math.pow(y - y2, 2) + math.pow(z - z2, 2) 
    
                    if dist < diameter:
                        if prediction['threshold'] >= threshold:
    
                            if pdx not in predictions_tracker:
                                print(f'Prediction {pdx} already aligned with nodule!')
                            else:
                                predictions_tracker.remove(pdx)
    
                            excluded_found[edx] += 1
                        
    return nodules_found, excluded_found, predictions_tracker

def get_operating_point_thresholds(froc_predictions_path, operating_points):

    predictions = pd.read_csv(froc_predictions_path, header=None, names=['fps', 'sensitivity', 'threshold'])
    fps_itp = np.linspace(0.125, 8, num=10000)

    sens_itp = np.interp(fps_itp, predictions['fps'], predictions['sensitivity'])
    threshold_itp = np.interp(fps_itp, predictions['fps'], predictions['threshold'])

    idxs = []
    for fps_value in operating_points:
        idxs.append(np.abs(fps_itp - fps_value).argmin())

    return threshold_itp[idxs]

class FROCSamplingFlow(FlowSpec):
    """
    Calculate the free response operating characteristic (FROC) scores for each demographic group
    """

    model = Parameter('model', help='Model to evaluate', default='grt123')
    flavour = Parameter('flavour', help='Flavour to evaluate', default='test_balanced')
    actionable = Parameter('actionable', type=bool, help='Only include actionable cases', default=True)
    n_bootstraps = Parameter('bootstraps', help='Number of bootstraps to perform', default=1000)

    if os.path.basename(os.getcwd()).upper() == 'SOTAEVALUATIONNODULEDETECTION':
        workspace_path = Path(os.getcwd()).as_posix()
    else:
        workspace_path = Path(os.getcwd()).parent.as_posix()

    @step
    def start(self):
        

        print(self.model, self.flavour, self.actionable)

        self.output_dir = Path(f'results/summit/resample/{self.model}/{self.flavour}/{"Actionable" if self.actionable else "All"}/FROC')

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
                # .rename(columns={'threshold': 'threshold_original'})
                # .assign(threshold=lambda x: 1 / (1 + np.exp(-x['threshold_original'])))
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
                                        usecols=[
                                            'Y0_PARTICIPANT_DETAILS_main_participant_id',
                                            'participant_details_gender',
                                            'lung_health_check_demographics_race_ethnicgroup'
                                        ]).rename(
                                        columns={
                                            'Y0_PARTICIPANT_DETAILS_main_participant_id': 'StudyId',
                                            'participant_details_gender' : 'gender',
                                            'lung_health_check_demographics_race_ethnicgroup' : 'ethnic_group'
                                        }).assign(name=lambda x: x['StudyId'] + '_Y0_BASELINE_A')

        # Reduce the annotations to only actionable cases
        if self.actionable:
            self.annotations = annotations[annotations['management_plan'].isin(['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1'])]
            self.annotations_excluded = annotations[annotations['management_plan']=='RANDOMISATION_AT_YEAR_1']
        else:
            self.annotations = annotations
            self.annotations_excluded = annotations.drop(annotations.index)


        self.next(self.define_sampling)

    @step
    def define_sampling(self):

        self.output_dir = self.output_dir
        self.scan_metadata = self.scan_metadata
        self.annotations = self.annotations
        self.annotations_excluded = self.annotations_excluded
        self.results = self.results

        self.resample_groups = [
            (i, 'random', 500) for i in range(1)
        ]
        self.next(self.resample, foreach='resample_groups')

    @step
    def resample(self):

        
        self.scan_metadata = self.scan_metadata
        self.annotations = self.annotations
        self.annotations_excluded = self.annotations_excluded
        self.results = self.results
        self.group, self.resample_group, self.number_scans = self.input

        self.output_dir = self.output_dir / str(self.group) / self.resample_group

        print(f'Resampling to {self.resample_group}')

        self.scan_metadata = self.scan_metadata #.sample(frac=1, replace=False, random_state=self.group)

        # Define the subsequent slices to be performed
        self.demographic_groups = [('all', 'all')]
        self.next(self.calculate_froc, foreach='demographic_groups')

    @step
    def calculate_froc(self):
        self.output_dir = self.output_dir
        self.scan_metadata = self.scan_metadata
        self.annotations = self.annotations
        self.annotations_excluded = self.annotations_excluded
        self.results = self.results
        self.group = self.group
        self.resample_group = self.resample_group
        self.number_scans = self.number_scans

        self.category, self.value = self.input

        if self.category == 'all':
            scans_metadata = self.scan_metadata
        else:
            scans_metadata = self.scan_metadata[self.scan_metadata[self.category] == self.value]

        self.scans = scans_metadata['name']
        self.annotations = self.annotations[self.annotations['name'].isin(self.scans.values)]
        self.exclusions = self.annotations_excluded[self.annotations_excluded['name'].isin(self.scans.values)]
        self.predictions = self.results[self.results['name'].isin(self.scans.values)]

        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            self.scans.to_csv(temp_dir / 'scans.csv', index=False)
            self.annotations.to_csv(temp_dir / 'annotations.csv', index=False)
            self.exclusions.to_csv(temp_dir / 'exclusions.csv', index=False)
            self.predictions.to_csv(temp_dir / 'predictions.csv', index=False)
            output_path = Path(f'{self.output_dir}/{self.value}')

            print("Output Path: ", output_path)

            self.froc_metrics = noduleCADEvaluation(
                annotations_filename=temp_dir / 'annotations.csv',
                annotations_excluded_filename=temp_dir / 'exclusions.csv', 
                seriesuids_filename=temp_dir / 'scans.csv',
                results_filename=temp_dir / 'predictions.csv',
                filter=f'Model: {self.model}, \nDataset: {self.flavour}, \nDemographic: {self.category} - {self.value}, \nActionable Only: {self.actionable}',
                perform_bootstrapping=False,
                outputDir=output_path
            )


        operating_points = [0.125, 0.25, 0.5, 1, 2, 4, 8]
        # operating_points = [0.125]

        operating_point_thresholds = get_operating_point_thresholds(
            output_path / 'froc_predictions.txt',
            operating_points
        )

        print('*'*100)
        print(operating_point_thresholds)
        print('*'*100)

        annotations = copy.deepcopy(self.annotations)
        exclusions = copy.deepcopy(self.exclusions)
        predictions = copy.deepcopy(self.predictions)

        for operating_point, threshold in zip(operating_points, operating_point_thresholds):
            nodules_found, excluded_found, prediction_tacker = get_nodule_level_metrics(
                scans_metadata,
                self.annotations,
                self.exclusions,
                self.predictions,
                threshold
            )

            annotations[f'detected_{str(operating_point).replace(".", "_")}'] = self.annotations.index.map(nodules_found)
            exclusions[f'detected_{str(operating_point).replace(".", "_")}'] = self.annotations_excluded.index.map(excluded_found)
            predictions[f'fp_{str(operating_point).replace(".", "_")}'] = self.predictions.index.isin(prediction_tacker)

        self.annotations = annotations
        self.exclusions = exclusions
        self.predictions = predictions
        self.next(self.join_froc)

    @step
    def join_froc(self, inputs):


        self.predictions = {}
        self.annotations = {}
        self.exclusions = {}

        for inp in inputs:
            self.annotations[inp.resample_group] = inp.annotations
            self.exclusions[inp.resample_group] = inp.exclusions
            self.predictions[inp.resample_group] = inp.predictions

        self.next(self.join_sample)

    @step
    def join_sample(self, inputs):

        self.annotations = {}
        self.exclusions = {}
        self.predictions = {}

        for inp in inputs:
            self.annotations.update(inp.annotations)
            self.exclusions.update(inp.exclusions)
            self.predictions.update(inp.predictions)

        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    FROCSamplingFlow()