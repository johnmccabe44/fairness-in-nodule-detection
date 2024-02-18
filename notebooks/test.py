import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,LogFormatter,StrMethodFormatter,FixedFormatter
import json
import numpy as np
import pandas as pd
import sys
import importlib
import shutil
import math
import os
from pathlib import Path
import warnings

sys.path.append('/Users/john/Projects/SOTAEvaluationNoduleDetection/utilities')

from evaluation import noduleCADEvaluation


warnings.simplefilter('ignore')

MIN_THRESHOLD = -10000000


def get_luna_detect_predictions(predictions_json_path):

    with open(predictions_json_path,'r') as f:
        predictions_json = json.load(f)

    images = []
    predictions_dict = {}
    idx = 0
    for image in predictions_json['test']:
        name = image['image'].split('/')[-1][:-7]
        
        images.append(name)

        for box, score in zip(image['box'], image['score']):
            prediction = {}
            prediction['threshold'] = score
            prediction['index'] = box[2]
            prediction['col'] = box[1]
            prediction['row'] = box[0]
            prediction['diameter'] = np.max(box[3:])
            prediction['name'] = name

            predictions_dict[idx] = prediction
            idx+=1

    predictions = pd.DataFrame.from_dict(predictions_dict,orient='index')

    return images, predictions


images, predictions = get_luna_detect_predictions('/Users/john/Projects/SOTAEvaluationNoduleDetection/models/detection/result/trained_luna/luna/result_luna16_fold0.json')

print(predictions.head(10))


annotations = pd.read_csv('/Users/john/Projects/SOTAEvaluationNoduleDetection/output/metadata/luna/test_fold1_metadata.csv')
print(annotations.head(10))

annotations = annotations[annotations['scan_id'].isin(images)]



exclusions = (
    pd.read_csv('/Users/john/Projects/SOTAEvaluationNoduleDetection/scans/luna16/metadata/annotations_excluded.csv')
    .rename(columns={
        'seriesuid':'scan_id',
        'coordX':'nodule_x_coordinate',
        'coordY':'nodule_y_coordinate',
        'coordZ':'nodule_z_coordinate',
        'diameter_mm':'nodule_diameter_mm'
        })
)

exclusions = exclusions[exclusions['scan_id'].isin(images)]


scans = pd.read_csv('/Users/john/Projects/SOTAEvaluationNoduleDetection/output/metadata/luna/test_fold1_scans.csv')
scans = scans[scans['scan_id'].isin(images)]


annotations_filepath = '/Users/john/Projects/SOTAEvaluationNoduleDetection/models/detection/result/trained_luna/luna/fold0_annotations.csv'
exclusions_filepath = '/Users/john/Projects/SOTAEvaluationNoduleDetection/models/detection/result/trained_luna/luna/fold0_exclusions.csv'
scanlist_filepath = '/Users/john/Projects/SOTAEvaluationNoduleDetection/models/detection/result/trained_luna/luna/fold0_scanlist.csv'
predictions_filepath = '/Users/john/Projects/SOTAEvaluationNoduleDetection/models/detection/result/trained_luna/luna/fold0_predictions.csv'
results_dir = Path('/Users/john/Projects/SOTAEvaluationNoduleDetection/output/results/detection/trained_luna/fold0')
results_dir.mkdir(parents=True, exist_ok=True)

rename_dic = {
    'scan_id':'name',
    'nodule_x_coordinate':'row',
    'nodule_y_coordinate':'col',
    'nodule_z_coordinate':'index',
    'nodule_diameter_mm':'diameter'
}

annotations.rename(columns=rename_dic).to_csv(annotations_filepath, index=False)
exclusions.rename(columns=rename_dic).to_csv(exclusions_filepath, index=False)
scans.to_csv(scanlist_filepath, index=False)
predictions.to_csv(predictions_filepath, index=False)


fps_itp, sens_itp = noduleCADEvaluation(
    annotations_filename=annotations_filepath,
    annotations_excluded_filename=exclusions_filepath,
    seriesuids_filename=scanlist_filepath,
    results_filename=predictions_filepath,
    outputDir=results_dir,
    filter='detection_trained_luna_evaluated_on_luna',
)