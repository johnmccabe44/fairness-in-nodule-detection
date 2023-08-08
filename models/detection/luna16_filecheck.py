from functools import partial
import json
import logging
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
import sys
from typing import List

def check_integrity(idx: int, mhd_paths: List[Path]):

    mhd_path = mhd_paths[idx]
    
    try:
        # read in the scan
        metadata = sitk.ReadImage(mhd_path)
        _ = np.array(sitk.GetArrayFromImage(metadata), dtype=np.int16)
        logging.info(f'{mhd_path},1')
        return True

    except Exception as err:

        logging.info(f'{mhd_path},0')
        return False

def main(mhd_root: Path, sub_processes: int):
    """
        Programme orchastrator
        - gets list of luna16 mhds from the decathalon json files
        - checks whether the file can be opened with sitk mhd reader
        - saves the outcome to a file so that it can be checked
    """

    mhd_paths = [mhd_path for mhd_path in mhd_root.iterdir() if mhd_path.as_posix().endswith('.mhd')]

    N = len(mhd_paths)
    partial_check_integrity = partial(check_integrity, 
                                      mhd_paths=mhd_paths)

    with Pool(sub_processes) as pool:
        _ = pool.map(partial_check_integrity, range(N))

if __name__ == '__main__':
    """
        Programme entry point
    """

    logging.basicConfig(filename='sota-detection-checks.log', level=logging.INFO)
    mhd_root        = Path(sys.argv[1])
    sub_processes   = int(sys.argv[2])

    main(mhd_root=mhd_root, sub_processes=sub_processes)
    
                    
