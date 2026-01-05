import json
import sys

from metaflow import FlowSpec, IncludeFile, Parameter, conda_base, step

if sys.platform == "darwin":
    sys.path.append('/Users/john/Projects/SOTAEvaluationNoduleDetection/utilities')
    sys.path.append('/Users/john/Projects/SOTAEvaluationNoduleDetection/notebooks')
elif sys.platform == "linux":
    sys.path.append('/home/jmccabe/Projects/SOTAEvaluationNoduleDetection/utilities')
    sys.path.append('/home/jmccabe/Projects/SOTAEvaluationNoduleDetection/notebooks')
else:
    raise EnvironmentError("Unsupported platform")

from evaluation import (calculate_ci, noduleCADEvaluation,
                        pool_bootstrap_evaluation)
from FairnessInNoduleDetectionAlgorithms.utils import (
    calculate_cpm_from_bootstrapping, display_plots_with_error_bars)
from summit_utils import *
from utils import load_data


class MAPScoreFlow(FlowSpec):
    """
    Calculate the mean average precision (mAP) scores for each demographic group
    """

    dataset = Parameter('dataset', help='Dataset to evaluate', default='summit')
    iou_threshold = Parameter('iou_threshold', help='IoU threshold', default=0.1)
    model = Parameter('model', help='Model to evaluate', default='detection')
    flavour = Parameter('flavour', help='Flavour to evaluate', default='test_balanced')
    actionable = Parameter('actionable', help='Only include actionable cases', default=True)
    
    if sys.platform == "darwin":
        workspace_path = '/Users/john/Projects/SOTAEvaluationNoduleDetection'
    elif sys.platform == "linux":
        workspace_path = '/home/jmccabe/Projects/SOTAEvaluationNoduleDetection'
    
    @step
    def start(self):
        
        self.n_boostraps = 1000

        print(f'Running evaluation Model: {self.model}, Flavour: {self.flavour}')

        self.next(self.load_data)

    @step
    def load_data(self):

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
            self.map_categories = [('all', 'all')] + gender_groups[self.dataset] + ethnic_groups[self.dataset]

        elif self.flavour == 'male_only':
            self.map_categories = [('all', 'all')] + ethnic_groups[self.dataset]

        elif self.flavour == 'white_only':
            self.map_categories = [('all', 'all')] + gender_groups[self.dataset]

        self.next(self.calculate_maps, foreach='map_categories')
    
    @step
    def calculate_maps(self):

        group, cat = self.input

        if cat == 'all':
            scans = self.scan_metadata['Name']
        else:
            scans = self.scan_metadata[self.scan_metadata[group] == cat]['Name']

        annotations = self.annotations[self.annotations['name'].isin(scans.values)]
        predictions = self.results[self.results['name'].isin(scans.values)]

        annotations_dict = annotations.groupby('name').apply(lambda x: x[['col', 'row', 'index', 'diameter']].values.tolist()).to_dict()
        predictions_dict = predictions.groupby('name').apply(lambda x: {'boxes': x[['col', 'row', 'index', 'diameter']].values.tolist(), 'scores': x['threshold'].values.tolist()}).to_dict()

        self.mAP = pool_bootstrap_evaluation(
            scans,
            annotations_dict,
            predictions_dict,
            n_bootstrap=self.n_boostraps,
            iou_threshold=self.iou_threshold,
            workers=4
        )

        self.mean_mAP = calculate_ci(self.mAP)
        self.group = group
        self.cat = cat
        self.next(self.join_maps)

    @step
    def join_maps(self, inputs):

        print('Workspace path:',  self.workspace_path)

        self.mean_mAPs = {}
        self.mAPs = {}
        
        for inp in inputs:
            self.mAPs[f'{inp.group}_{inp.cat}'] = inp.mAP
            self.mean_mAPs[f'{inp.group}_{inp.cat}'] = inp.mean_mAP


        # write out results
        with open(f'{self.workspace_path}/workflows/FairnessInNoduleDetectionAlgorithms/results/{self.dataset}/{self.model}_{self.flavour}_{self.iou_threshold}_mAPs.json', 'w') as f:
            f.write(json.dumps(self.mAPs))    

        with open(f'{self.workspace_path}/workflows/FairnessInNoduleDetectionAlgorithms/results/{self.dataset}/{self.model}_{self.flavour}_{self.iou_threshold}_mean_mAPs.json', 'w') as f:
            f.write(json.dumps(self.mean_mAPs))    

        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    MAPScoreFlow()