import json
import numpy as np
from metaflow import FlowSpec, step, IncludeFile, Parameter, conda_base
import numpy as np
import os
import pandas as pd
from pathlib import Path
import scipy.stats as stats
import sys

from sympy import use
from torch import ge

sys.path.append('../utilities')

from evaluation import calculate_ci, pool_bootstrap_evaluation


def read_predictions(workspace_path, model, flavour):
    if model == 'grt123':
        return pd.read_csv(
            f'{workspace_path}/models/grt123/bbox_result/trained_summit/summit/{flavour}/{flavour}_predictions.csv',
            usecols=['name', 'col', 'row', 'index', 'diameter', 'threshold']
        ).rename(columns={'threshold': 'threshold_original'}).assign(threshold=lambda x: 1 / (1 + np.exp(-x['threshold_original'])))
    else:
        return pd.read_csv(
            f'{workspace_path}/models/detection/result/trained_summit/summit/{flavour}/predictions.csv',
            usecols=['name', 'col', 'row', 'index', 'diameter', 'threshold']
        )
    
def read_ground_truths(workspace_path, model, flavour, actionable=True):
    if model == 'grt123':
        ground_truths = pd.read_csv(
            f'{workspace_path}/models/grt123/bbox_result/trained_summit/summit/{flavour}/{flavour}_metadata.csv',
            usecols=['name', 'col', 'row', 'index','diameter', 'gender', 'ethnic_group', 'management_plan']
        )
    else:
        ground_truths = pd.read_csv(
            f'{workspace_path}/models/detection/result/trained_summit/summit/{flavour}/annotations.csv',
            usecols=['name', 'col', 'row', 'index','diameter', 'gender', 'ethnic_group', 'management_plan']
        )

    if actionable:
        actionable = ground_truths['management_plan'].isin(['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1'])
        ground_truths = ground_truths[actionable]
        

    return ground_truths

class MAPScoreFlow(FlowSpec):
    """
    Calculate the mean average precision (mAP) scores for each demographic group
    """

    model = Parameter('model', help='Model to evaluate')
    flavour = Parameter('flavour', help='Flavour to evaluate')
    actionable = Parameter('actionable', help='Only include actionable cases', default=True)
    workspace_path = Path(os.getcwd()).parent.as_posix()
    
    @step
    def start(self):
        
        self.n_boostraps =1000

        print(f'Running evaluation Model: {self.model}, Flavour: {self.flavour}')

        self.next(self.load_data)

    @step
    def load_data(self):
        self.predictions = read_predictions(self.workspace_path, self.model, self.flavour)
        self.ground_truths = read_ground_truths(self.workspace_path, self.model, self.flavour, self.actionable)

        print('Loaded data')
        print('Ground truths:', self.ground_truths.columns)
        print('Predictions:', self.predictions.columns)


        gender_categories = [
            ('gender', 'MALE'),
            ('gender', 'FEMALE')
        ]

        ethnic_group_categories = [
            ('ethnic_group', 'Asian or Asian British'),
            ('ethnic_group', 'Black'),
            ('ethnic_group', 'White')
        ]

        if self.flavour == 'test_balanced':
            self.map_categories = gender_categories + ethnic_group_categories

        elif self.flavour == 'male_only':
            self.map_categories = ethnic_group_categories

        elif self.flavour == 'balanced_white_only':
            self.map_categories = gender_categories

        self.next(self.calculate_maps, foreach='map_categories')
    
    @step
    def calculate_maps(self):

        var, cat = self.input

        ground_truths = self.ground_truths[self.ground_truths[var]==cat]
        predictions = self.predictions[self.predictions['name'].isin(ground_truths['name'].values)]

        gt_annotations = ground_truths.groupby('name').apply(lambda x: x[['col', 'row', 'index', 'diameter']].values.tolist()).to_dict()
        pred_annotations = predictions.groupby('name').apply(lambda x: {'boxes': x[['col', 'row', 'index', 'diameter']].values.tolist(), 'scores': x['threshold'].values.tolist()}).to_dict()

        self.mAP = pool_bootstrap_evaluation(gt_annotations,
                                                pred_annotations,
                                                len(gt_annotations), 
                                                n_bootstrap=self.n_boostraps,
                                                workers=4)

        self.mean_mAP = calculate_ci(self.mAP)
        self.cat = cat
        self.next(self.join_maps)

    @step
    def join_maps(self, inputs):

        print('Workspace path:',  self.workspace_path)

        self.mean_mAPs = {}
        self.mAPs = {}
        
        for inp in inputs:
            self.mAPs[inp.cat] = inp.mAP
            self.mean_mAPs[inp.cat] = inp.mean_mAP


        # write out results
        with open(f'{self.workspace_path}/workflows/results/{self.model}_{self.flavour}_mAPs.json', 'w') as f:
            f.write(json.dumps(self.mAPs))    

        with open(f'{self.workspace_path}/workflows/results/{self.model}_{self.flavour}_mean_mAPs.json', 'w') as f:
            f.write(json.dumps(self.mean_mAPs))    

        self.next(self.calculate_ttests)

    @step
    def calculate_ttests(self):

        self.mAPs = self.mAPs
        self.mean_mAPs = self.mean_mAPs

        print('self.mAPs:', self.mAPs.keys())


        gender_categories = [('MALE','FEMALE')]
        ethnic_group_categories = [
            ('Asian or Asian British','Black'),
            ('Asian or Asian British','White'),
            ('Black','White')]

        if self.flavour == 'test_balanced':
            self.matched_pairs = gender_categories + ethnic_group_categories

        elif self.flavour == 'male_only':
            self.matched_pairs = ethnic_group_categories

        elif self.flavour == 'balanced_white_only':
            self.matched_pairs = gender_categories
        
        self.next(self.calculate_ttest, foreach='matched_pairs')

    @step
    def calculate_ttest(self):
        cat1, cat2 = self.input

        print(f'Running t-test for {cat1} vs {cat2}')

        t_stat, p_value = stats.ttest_ind(self.mAPs[cat1], self.mAPs[cat2])

        
        self.ttest_stats = {'t_stat': t_stat, 'p_value': p_value}
        self.key = f'{cat1}Vs{cat2}'

        self.next(self.join_ttests)

    @step
    def join_ttests(self, inputs):

        self.ttest_stats = {}      
        for inp in inputs:
            self.ttest_stats[inp.key] = inp.ttest_stats


        with open(f'{self.workspace_path}/workflows/results/{self.model}_{self.flavour}_ttest_stats.json', 'w') as f:
            f.write(json.dumps(self.ttest_stats))

        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    MAPScoreFlow()