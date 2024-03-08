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

from layers import nms,iou

MIN_THRESHOLD = -10000000

class ShapeDifferentException(Exception):
    pass
class TooHighMetricException(Exception):
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments for collating the outputs from grt123')

    parser.add_argument('--name', type=str, help='Name of the model')

    parser.add_argument('--is-folds', action='store_true', help='If the data is in folds')

    parser.add_argument('--scan-name', type=str, help='Name of the scan file')

    parser.add_argument('--metadata-name', type=str, help='Name of the metadata file')

    parser.add_argument('--bbox-result-path', type=str, help='Path to where the predictions live')

    parser.add_argument('--output-path', type=str, help='Output path where the final analysis files will live')

    parser.add_argument('--threshold', type=float, help='Threshold to filter the predictions by i.e. keep anything with prediction > threshold')

    parser.add_argument('--workers', type=int, help='Number of workers used in multi-processing, assume cores * 2')

    args = parser.parse_args()
    return args 

def load_pbb(idx: int, pbb_paths: List[Path], threshold=-1):
    """
        load the candidates for this scan
        use nms to de-duplicate and apply threshold
    """
    try:
        pbb_path = pbb_paths[idx]
        pbb = np.load(pbb_path)
        pbb = pbb[pbb[:,0]>threshold]    
        pbb = nms(pbb, 0.05)

        if pbb.shape[0]>0:
            return (
                pd.DataFrame(pbb, columns=['threshold','index', 'row', 'col','diameter'])
                .assign(name=pbb_path.name.split('_pbb')[0])
            )
        else:
            return pd.DataFrame()
    except Exception as err:
        print(f'Error occured processing:{pbb_paths[idx]}, Error:{err}', flush=True)
        return pd.DataFrame()    

def combine_pbb(scan_ids: List[str], bbox_path: Path, threshold: float, workers: int):
    """
        load the candidates for this scan
        use nms to de-duplicate and apply threshold
    """

    pbb_paths = [
        Path(root, fil)
        for root, _, files in os.walk(bbox_path)
        for fil in files 
        if fil.endswith('_pbb.npy') and fil.split('_pbb')[0] in scan_ids
    ]
    N = len(pbb_paths)

    print(f'Found {N} pbb files to process')

    if workers>1:
        partial_read_pbb = partial(load_pbb, pbb_paths=pbb_paths, threshold=threshold)
        
        with Pool(workers) as pool:
            pbb_dfs = list(tqdm.tqdm(pool.imap(partial_read_pbb, range(N)),total=N))

    else:
        
        pbb_dfs = []
        for idx in tqdm.tqdm(range(N), total=N):
            pbb_dfs.append(load_pbb(idx, pbb_paths, threshold))


    for idx, df in enumerate(pbb_dfs):
        if df.shape[0] == 0:
            continue

        if idx == 0:
            total_pbb = df
        else:
            total_pbb = pd.concat([total_pbb, df])

    return total_pbb.reset_index().drop('level_0', axis=1)

def merge_lbl_and_metadata(idx:int, lbb_paths: List[Path], metadata: pd.DataFrame):
    """
        The lbl data only contains adjusted irc coordindates
        and we need to attach nodule details such as type and
        brock score to check for patterns. This is done by
        assuming the read order for generating the lbl is the
        same therefore repeating the process and adding additional
        variables to the adjusted irc data.
    """

    try: 
        lbb = np.load(lbb_paths[idx])
        scan_id = lbb_paths[idx].name.split('_lbb.npy')[0]
        stem = lbb_paths[idx].name.split('_',1)[0]

        nodule_metadata = metadata[metadata.scan_id==stem]

        if nodule_metadata.shape[0] == 0 and np.array_equal(lbb, [[0,0,0,0]]):
            return None
        
        if nodule_metadata.shape[0] == 0 and not np.array_equal(lbb, [[0,0,0,0]]):
            raise ShapeDifferentException(f'Label and metadata mismatch for stem. md:{metadata.shape[0]}, lbb:{lbb.shape[0]}, {lbb}')
        
        if nodule_metadata.shape[0] > 0 and np.array_equal(lbb, [[0,0,0,0]]):
            raise ShapeDifferentException(f'Label and metadata mismatch for stem. md:{metadata.shape[0]}, lbb:{lbb.shape[0]}, {lbb}')
        
        if nodule_metadata.shape[0] != lbb.shape[0]:
            raise ShapeDifferentException(f'Label and metadata mismatch for stem. md:{metadata.shape[0]}, lbb:{lbb.shape[0]}, {lbb}')
        
        d_mean = np.mean(lbb[:,3] / nodule_metadata.diameter_mm)
        d_std = np.std(lbb[:,3] / nodule_metadata.diameter_mm)

        #if not math.isclose(d_mean, 1, abs_tol=1e-4) or not math.isclose(d_std, 0, abs_tol=1e-4):
        #    raise TooHighMetricException(f'Mean is too high for the spacing: {scan_id}, {d_mean}, {d_std}')

        nodule_metadata.loc[:,['index','row','col','diameter']] = lbb
        nodule_metadata.loc[:,'threshold'] = MIN_THRESHOLD
        nodule_metadata.loc[:,'name'] = scan_id

        return nodule_metadata
    except Exception as err:
        print(f'Error occured processing:{lbb_paths[idx]}, Error:{err}', flush=True)
        return pd.DataFrame()

def combine_metadata(scan_ids: List[str], metadata: pd.DataFrame, bbox_path: Path, workers: int):
    """
        combine the lbl outputs with the original metadata
        this allows for analysis to include the profile and
        identification of nodules that were not missed

    """

    lbb_paths = [
        Path(root, fil)
        for root, _, files in os.walk(bbox_path)
        for fil in files
        if fil.endswith('_lbb.npy') and fil.split('_lbb')[0] in scan_ids
    ]

    N = len(lbb_paths)
    if workers>1:

        partial_merge_lbl_and_metadata = partial(merge_lbl_and_metadata, 
                                                 lbb_paths=lbb_paths, 
                                                 metadata=metadata)
        
        with Pool(workers) as pool:
            md_and_lbb = list(tqdm.tqdm(pool.imap(partial_merge_lbl_and_metadata, range(N)),total=N))

    else:
            md_and_lbb = [
                merge_lbl_and_metadata(idx, lbb_paths, metadata)
                for idx in range(N)
            ]

    return pd.concat(md_and_lbb).reset_index().rename(columns={'level_0' : 'id'})

def main(name: str, scan_ids : List[str], metadata: pd.DataFrame, bbox_path: Path, output_path: Path, threshold: float, workers: int):
    
    combine_pbb(scan_ids, bbox_path, threshold, workers).to_csv(Path(output_path, f'{name}_predictions.csv'), index=False)
    
    combine_metadata(scan_ids, metadata, bbox_path, workers).to_csv(Path(output_path, f'{name}_metadata.csv'), index=False)


if __name__ == '__main__':
    args = parse_arguments()

    if args.is_folds:
        scans = pd.concat([
            pd.read_csv(scan_file_path)
            for scan_file_path in Path(args.metadata_path).glob(f'{args.scan_name}')
        ]).scan_id.values
    else:
        scan_path = Path(args.scan_name)
        scans = pd.read_csv(scan_path).scan_id.values

    if args.is_folds:
        metadata = pd.concat([
            pd.read_csv(metadata_file_path)
            for metadata_file_path in Path(args.metadata_path).glob(f'{args.metadata_name}')
        ])
    else:
        metadata_path = Path(args.metadata_name)
        metadata = pd.read_csv(metadata_path)

    bbox_path = Path(args.bbox_result_path)

    output_path = Path(args.output_path)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    threshold = args.threshold
    
    workers = args.workers

    main(args.name, scans, metadata, bbox_path, output_path, threshold, workers)
