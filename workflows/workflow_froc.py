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

        self.output_dir = f'results/summit/{self.model}/{self.flavour}/{"Actionable" if self.actionable else "All"}/{"Excluded" if self.exclude_outliers else "Included"}/FROC'

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

        # Remove the scans that are not in the metadata
        if self.exclude_outliers:
            self.scan_metadata = self.scan_metadata[~self.scan_metadata['Name'].isin([
        'summit-2626-hgp_Y0_BASELINE_A', 'summit-3339-ktr_Y0_BASELINE_A',
       'summit-3634-kct_Y0_BASELINE_A', 'summit-3679-cmk_Y0_BASELINE_A',
       'summit-4242-bec_Y0_BASELINE_A', 'summit-4345-ctj_Y0_BASELINE_A',
       'summit-6244-vvj_Y0_BASELINE_A', 'summit-7236-yph_Y0_BASELINE_A',
       'summit-7328-thh_Y0_BASELINE_A', 'summit-7347-vgb_Y0_BASELINE_A',
       'summit-7658-wmk_Y0_BASELINE_A', 'summit-8994-kpf_Y0_BASELINE_A',
       'summit-9333-wbc_Y0_BASELINE_A'
                ])]

        # Reduce the annotations to only actionable cases
        if self.actionable:
            self.annotations = annotations[annotations['management_plan'].isin(['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1'])]
            self.annotations_excluded = annotations[annotations['management_plan']=='RANDOMISATION_AT_YEAR_1']
        else:
            self.annotations = annotations
            self.annotations_excluded = annotations.drop(annotations.index)


        # Define the subsequent slices to be performed
        gender_groups = [('gender','MALE'), ('gender', 'FEMALE')]
        ethnic_groups = [('ethnic_group', 'Asian or Asian British'),('ethnic_group','Black'),('ethnic_group','White')]

        if self.flavour == 'test_balanced':
            self.demographic_groups = [('all', 'all')] + gender_groups + ethnic_groups

        elif self.flavour == 'male_only':
            self.demographic_groups = [('all', 'all')] + ethnic_groups

        elif self.flavour == 'white_only':
            self.demographic_groups = [('all', 'all')] + gender_groups

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
        self.cpm_data, self.cpm_summary = caluclate_cpm_from_bootstrapping(output_path / 'froc_predictions_bootstrapping.csv')
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

        pd.DataFrame.from_dict(self.cpm_summary, orient='index').reindex(
            ['MALE','FEMALE','Asian or Asian British','Black','White','all']
        ).to_csv(f'{self.output_dir}/cpm_summary.csv')

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

        # Define charts to be generated
        if self.flavour == 'test_balanced':            
            self.demographic_groups = [
                ('gender' , ['MALE','FEMALE']),
                ('ethnic_group' , ['Asian or Asian British','Black','White'])
            ]
        elif self.flavour == 'male_only':
            self.demographic_groups = [
                ('ethnic_group' , ['Asian or Asian British','Black','White'])
            ]

        elif self.flavour == 'white_only':
            self.demographic_groups = [
                ('gender' , ['MALE','FEMALE'])
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