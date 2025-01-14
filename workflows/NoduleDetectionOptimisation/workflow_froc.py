import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from metaflow import FlowSpec, Parameter, step

if os.path.basename(os.getcwd()).upper() == 'SOTAEVALUATIONNODULEDETECTION':
    sys.path.append('utilities')
    sys.path.append('notebooks')
else:
    sys.path.append('../../utilities')
    sys.path.append('../../notebooks')

from evaluation import noduleCADEvaluation
from FairnessInNoduleDetectionAlgorithms.utils import \
    caluclate_cpm_from_bootstrapping
from utils import load_data


class FROCFlow(FlowSpec):
    """
    Calculate the free response operating characteristic (FROC) scores for each demographic group
    """

    actionable = Parameter('actionable', type=bool, help='Only include actionable cases', default=True)
    n_bootstraps = Parameter('bootstraps', help='Number of bootstraps to perform', default=1000)

    if os.path.basename(os.getcwd()).upper() == 'SOTAEVALUATIONNODULEDETECTION':
        workspace_path = Path(os.getcwd()).as_posix()
    else:
        workspace_path = Path(os.getcwd()).parent.parent.as_posix()
    

    @step
    def start(self):
    
        self.models_and_flavours = [
            (model, flavour)
            for model in ['detection']
            for flavour in ['optimisation']
        ]

        self.next(self.calculate_froc, foreach='models_and_flavours')
    
    @step
    def calculate_froc(self):

        self.model, self.flavour = self.input

        annotations, results, scan_metadata, annotations_excluded = load_data(self.workspace_path, self.model, self.flavour, self.actionable)

        output_path = Path(f'results/summit/{self.model}/{self.flavour}/{"Actionable" if self.actionable else "All"}/FROC')
        output_path.mkdir(parents=True, exist_ok=True)

        scans = scan_metadata['Name']

        annotations = annotations[annotations['name'].isin(scans.values)]
        exclusions = annotations_excluded[annotations_excluded['name'].isin(scans.values)]
        predictions = results[results['name'].isin(scans.values)]


        scans.to_csv(output_path / 'scans.csv', index=False)
        annotations.to_csv(output_path / 'annotations.csv', index=False)
        exclusions.to_csv(output_path / 'exclusions.csv', index=False)
        predictions.to_csv(output_path / 'predictions.csv', index=False)


        self.froc_metrics = noduleCADEvaluation(
            annotations_filename=output_path / 'annotations.csv',
            annotations_excluded_filename=output_path / 'exclusions.csv', 
            seriesuids_filename=output_path / 'scans.csv',
            results_filename=output_path / 'predictions.csv',
            filter=f'Model: {self.model}, \nDataset: {self.flavour}, \nActionable Only: {self.actionable}',
            outputDir=output_path
        )

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

        self.froc_metrics = {f'{inp.model}_{inp.flavour}' : inp.froc_metrics for inp in inputs}
        self.cpm_data = {f'{inp.model}_{inp.flavour}' : inp.cpm_data for inp in inputs}
        self.cpm_summary = {f'{inp.model}_{inp.flavour}' : inp.cpm_summary for inp in inputs}
        self.boot_metrics = {f'{inp.model}_{inp.flavour}' : inp.boot_metrics for inp in inputs}
        self.bootstap_results = {f'{inp.model}_{inp.flavour}' : inp.bootstap_results for inp in inputs}
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    FROCFlow()