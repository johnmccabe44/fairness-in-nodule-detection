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

    parser.add_argument('--name', type=str, help='Name of the model')

    parser.add_argument('--metadata-path', type=str, help='Path to where the metadata files live')

    parser.add_argument('--scan-name', type=str, help='Name of the scan file')

    parser.add_argument('--metadata-name', type=str, help='Name of the metadata file')

    parser.add_argument('--bbox-result-path', type=str, help='Path to where the predictions live')
    
    parser.add_argument('--output-path', type=str, help='Output path where the final analysis files will live')

    parser.add_argument('--workers', type=int, help='Number of workers used in multi-processing, assume cores * 2')

    args = parser.parse_args()
    return args 

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
        scan_id = lbb_paths[idx].name.split('_bboxes.npy')[0]

        nodule_metadata = metadata[metadata.scan_id==scan_id]

        if nodule_metadata.shape[0] == 0 and np.array_equal(lbb, [[0,0,0,0]]):
            return None
        
        if nodule_metadata.shape[0] == 0 and not np.array_equal(lbb, [[0,0,0,0]]):
            raise ShapeDifferentException(f'Label and metadata mismatch for stem. md:{nodule_metadata.shape[0]}, lbb:{lbb.shape[0]}, {scan_id}')
        
        if nodule_metadata.shape[0] > 0 and np.array_equal(lbb, [[0,0,0,0]]):
            raise ShapeDifferentException(f'Label and metadata mismatch for stem. md:{nodule_metadata.shape[0]}, lbb:{lbb.shape[0]}, {scan_id}')
        
        if nodule_metadata.shape[0] != lbb.shape[0]:
            raise ShapeDifferentException(f'Label and metadata mismatch for stem. md:{nodule_metadata.shape[0]}, lbb:{lbb.shape[0]}, {scan_id}')
        
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
        if fil.endswith('_bboxes.npy') and fil.split('_bboxes')[0] in scan_ids
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

def main(name: str, scan_ids : List[str], metadata: pd.DataFrame, bbox_path: Path, output_path: Path, workers: int):
    
    combine_metadata(scan_ids, metadata, bbox_path, workers).to_csv(Path(output_path, f'{name}_metadata.csv'), index=False)


if __name__ == '__main__':
    args = parse_arguments()

    scan_path = Path(args.scan_name)
    scans = pd.read_csv(scan_path).scan_id.values

    metadata_path = Path(args.metadata_name)
    metadata = pd.read_csv(metadata_path)

    bbox_path = Path(args.bbox_result_path)

    output_path = Path(args.output_path)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    workers = args.workers

    main(args.name, scans, metadata, bbox_path, output_path, workers)
