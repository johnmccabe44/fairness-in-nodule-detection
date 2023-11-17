import argparse
from functools import partial
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
from pathlib import Path
from typing import List
import sys

from layers import nms,iou
from models.grt123.main import parse_arguments

class ShapeDifferentException(Exception):
    pass
class TooHighMetricException(Exception):
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments for collating the outputs from grt123')

    parser.add_argument('--metadata-path', type=str, help='Path to list of scnas to combine')

    parser.add_argument('--metadata-stem', type=str, help='Stem of file that determines the scans and metadata file.')

    parser.add_argument('--bbox_result_path', type=str, help='Path to where the predictions live')

    parser.add_argument('--output_path', type=str, help='Output path where the final analysis files will live')

    parser.add_argument('--threshold', type=float, help='Threshold to filter the predictions by i.e. keep anything with prediction > threshold')

    parser.add_argument('--workers', type=int, help='Number of workers used in multi-processing, assume cores * 2')

def merge_lbl_and_metadata(metadata, lbb, scan_id):
    """
        The lbl data only contains adjusted irc coordindates
        and we need to attach nodule details such as type and
        brock score to check for patterns. This is done by
        assuming the read order for generating the lbl is the
        same therefore repeating the process and adding additional
        variables to the adjusted irc data.
    """
    stem = scan_id.split('_',1)[0]

    nodule_metadata = metadata[metadata.main_participant_id==stem]

    if nodule_metadata.shape[0] == 0 and np.array_equal(lbb, [[0,0,0,0]]):
        return None
    
    if nodule_metadata.shape[0] == 0 and not np.array_equal(lbb, [[0,0,0,0]]):
        raise ShapeDifferentException(f'Label and metadata mismatch for stem. md:{metadata.shape[0]}, lbb:{lbb.shape[0]}, {lbb}')
    
    if nodule_metadata.shape[0] > 0 and np.array_equal(lbb, [[0,0,0,0]]):
        raise ShapeDifferentException(f'Label and metadata mismatch for stem. md:{metadata.shape[0]}, lbb:{lbb.shape[0]}, {lbb}')
    
    if nodule_metadata.shape[0] != lbb.shape[0]:
        raise ShapeDifferentException(f'Label and metadata mismatch for stem. md:{metadata.shape[0]}, lbb:{lbb.shape[0]}, {lbb}')
    
    d_mean = np.mean(lbb[:,3] / nodule_metadata.nodule_diameter_mm)
    d_std = np.std(lbb[:,3] / nodule_metadata.nodule_diameter_mm)

    if d_mean > 1 or d_std > 0.0001:
        raise TooHighMetricException(f'Mean is too high for the spacing: {scan_id}')

    nodule_metadata.loc[:,['index','row','col','diameter']] = lbb
    nodule_metadata.loc[:,'threshold'] = MIN_THRESHOLD
    nodule_metadata.loc[:,'name'] = scan_id

    return nodule_metadata[['name','threshold','index', 'row','col','diameter','nodule_type','nodule_brock_score', 'management_plan']]

def load_pbb(idx: int, ppb_paths: List[Path], threshold=-1):
    """
        load the candidates for this scan
        use nms to de-duplicate and apply threshold
    """
    pbb_path = ppb_paths[idx]
    pbb = np.load(pbb_path)
    pbb = nms(pbb, 0.05)
    pbb = pbb[pbb[:,0]>threshold]
    if pbb.shape[0]>0:
        return (
            pd.DataFrame(pbb, columns=['threshold','index', 'row', 'col','diameter'])
            .assign(name=pbb_path.name.split('_pbb')[0])
        )
    else:
        return None

def combine_pbb(scans: List[str], bbox_path: Path, threshold: float, workers: int):
    """
        load the candidates for this scan
        use nms to de-duplicate and apply threshold
    """

    pbb_paths = [
        Path(root, fil)
        for root, _, files in os.walk(bbox_path)
        for fil in files 
        if fil.endswith('_pbb.npy') and fil.split('_pbb')[0] in scans
    ]

    pbb_dfs = []
    if workers>1:
        N = len(pbb_paths)
        partial_read_pbb = partial(load_pbb, pbb_paths=pbb_paths, threshold=threshold)
        with Pool(workers) as pool:
            pbb_dfs = pool.map(partial_read_pbb, range(N))
    else:
        for idx in range(len(N)):
            pbb_dfs.append(load_pbb(idx, pbb_paths, threshold))

    combined_pbb = pd.concat([df for df in pbb_dfs if df])
    return combined_pbb.reset_index().drop('level_0', axis=1)

def combine_metadata(scans: List[str], metadata: pd.DataFrame, bbox_path: Path):
    """
        combine the lbl outputs with the original metadata
        this allows for analysis to include the profile and
        identification of nodules that were not missed

    """
    return pd.concat([
        merge_lbl_and_metadata(
            metadata, 
            np.load(Path(bbox_path,scan_id + '_lbb.npy')),
            scan_id)
        for scan_id in scans.scan_id
        if os.path.exists(Path(bbox_path,scan_id + '_lbb.npy'))
    ]).reset_index().rename(columns={'level_0' : 'id'})

def main(scans : List[str], metadata: pd.DataFrame, bbox_path: Path, threshold: float, workers: int):
    
    combined_predictions = combine_pbb(scans, bbox_path, threshold, workers)
        
    combined_nodule_data = combine_metadata(scans, metadata, bbox_path, workers)



if __name__ == '__main__':
    args = parse_arguments()

    scan_path = Path(args.metadata_path, args.metadata_stem + '_scans.csv')
    scans = pd.read_csv(scan_path)['scan_id'].tolist()

    metadata_path = Path(args.metadata_path, args.metadata_stem + '_metadata.csv')
    metadata = pd.read_csv(metadata_path)

    bbox_path = Path(args.bbox_path)

    output_path = Path(args.output_path)

    threshold = args.threshold
    
    workers = args.workers


    main(scans, metadata_path, bbox_path, output_path, threshold, workers)
