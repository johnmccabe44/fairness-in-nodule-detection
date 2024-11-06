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

sys.path.append('/home/jmccabe/SOTAEvaluationNoduleDetection/utilities')

from evaluation import noduleCADEvaluation


warnings.simplefilter('ignore')

MIN_THRESHOLD = -10000000

cwd = '/home/jmccabe/SOTAEvaluationNoduleDetection'

scanlist_filepath = Path(cwd, 'output/metadata/reduced/test_metadata.csv')

annotations_filepath = Path(cwd, 'models/grt123/bbox_result/summit/metadata.csv')
predictions_filepath = Path(cwd, 'models/grt123/bbox_result/summit/predictions.csv')
annotations_exclude_filepath = Path(cwd, 'models/grt123/bbox_result/summit/exclusions.csv')

results_dir = Path(cwd, 'output/results/grt123/trained_luna/summit')
results_dir.mkdir(parents=True, exist_ok=True)


fps_itp, sens_itp = noduleCADEvaluation(
    annotations_filename=annotations_filepath,
    annotations_excluded_filename=annotations_exclude_filepath,
    seriesuids_filename=scanlist_filepath,
    results_filename=predictions_filepath,
    outputDir=results_dir,
    filter='trained_luna_evaluated_on_luna',
)