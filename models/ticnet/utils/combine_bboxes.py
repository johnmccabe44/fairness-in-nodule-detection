import argparse
from functools import partial
from multiprocessing import Pool
import math
import numpy as np
import os
import pandas as pd
from pathlib import Path
import tqdm
from typing import List
import warnings

warnings.filterwarnings("ignore")

MIN_THRESHOLD = -10000000

class ShapeDifferentException(Exception):
    pass
class TooHighMetricException(Exception):
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments for collating the outputs from grt123')

    parser.add_argument(
        '--name',
        type=str, 
        help='Name of the model', 
        default='optimisation'
    )

    parser.add_argument(
        '--metadata-path', 
        type=str, 
        help='Path to where the metadata files live', 
        default='/home/jmccabe/Projects/SOTAEvaluationNoduleDetection/metadata/summit/optimisation'
    )

    parser.add_argument(
        '--bbox-result-path', 
        type=str, 
        help='Path to where the predictions live', 
        default='/home/jmccabe/Projects/SOTAEvaluationNoduleDetection/models/ticnet/cache/summit/bboxes/optimisation'
    )

    args = parser.parse_args()
    return args 


def main():

    args = parse_arguments()

    # Load the metadata files
    metadata = pd.concat(
        [
            pd.read_csv(os.path.join(args.metadata_path, f'{ds}_metadata.csv'))
            for ds in ['training', 'validation', 'test', 'holdout']
        ]
    )
    metadata['idx'] = metadata.groupby('scan_id').cumcount() + 1
    print(metadata.head())

    # Coombine all numpy bboxes into a csv file
    box_metadata = []
    for npy_file in Path(args.bbox_result_path).rglob('*.npy'):
        bboxes = np.load(npy_file)
        for idx, bbox in enumerate(bboxes):
            scan_id = npy_file.stem.replace('_bboxes', '')
            box_metadata.append({'scan_id': scan_id, 'idx': idx, 'bbox': bbox})

    # Convert bboxes dictionary to a DataFrame
    bbox_df = pd.DataFrame(box_metadata)
    bbox_df[['index', 'col', 'row', 'diameter']] = pd.DataFrame(bbox_df['bbox'].tolist(), index=bbox_df.index)
    bbox_df['idx'] = bbox_df.groupby('scan_id').cumcount() + 1

    # Merge the metadata and bbox dataframes
    combined_df = pd.merge(metadata, bbox_df, on=['scan_id', 'idx'])

    # check that the dimaeters are the same within 20% of each other
    combined_df['diameter_check'] = combined_df['nodule_diameter_mm'] / combined_df['diameter']

    combined_df.to_csv(f'{args.bbox_result_path}/annotations.csv')

if __name__ == '__main__':
    main()
