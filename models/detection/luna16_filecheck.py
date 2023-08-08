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

class SimpleITKLogger(sitk.LoggerBase):
    """
    Adapts SimpleITK messages to be handled by a Python Logger object.

    Allows using the logging module to control the handling of messages coming
    from ITK and SimpleTK. Messages such as debug and warnings are handled by
    objects derived from sitk.LoggerBase.

    The LoggerBase.SetAsGlobalITKLogger method must be called to enable
    SimpleITK messages to use the logger.

    The Python logger module adds a second layer of control for the logging
    level in addition to the controls already in SimpleITK.

    The "Debug" property of a SimpleITK object must be enabled (if
    available) and the support from the Python "logging flow" hierarchy
    to handle debug messages from a SimpleITK object.

    Warning messages from SimpleITK are globally disabled with
    ProcessObject:GlobalWarningDisplayOff.
    """

    def __init__(
        self, logger: logging.Logger = logging.getLogger("SimpleITK")
    ):
        """
        Initializes with a Logger object to handle the messages emitted from
        SimpleITK/ITK.
        """
        super(SimpleITKLogger, self).__init__()
        self._logger = logger

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger):
        self._logger = logger

    def __enter__(self):
        self._old_logger = self.SetAsGlobalITKLogger()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._old_logger.SetAsGlobalITKLogger()
        del self._old_logger

    def DisplayText(self, s):
        # Remove newline endings from SimpleITK/ITK messages since the Python
        # logger adds during output.
        self._logger.info(s.rstrip())

    def DisplayErrorText(self, s):
        self._logger.error(s.rstrip())

    def DisplayWarningText(self, s):
        self._logger.warning(s.rstrip())

    def DisplayGenericOutputText(self, s):
        self._logger.info(s.rstrip())

    def DisplayDebugText(self, s):
        self._logger.debug(s.rstrip())

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

    # Enable all debug messages for all ProcessObjects, and procedures
    sitk.ProcessObject.GlobalDefaultDebugOn()

    # Construct a SimpleITK logger to Python Logger adaptor
    sitkLogger = SimpleITKLogger()

    # Configure ITK to use the logger adaptor
    sitkLogger.SetAsGlobalITKLogger()
    
    logging.basicConfig(filename='sota-detection-checks.log', level=logging.INFO)
    mhd_root        = Path(sys.argv[1])
    sub_processes   = int(sys.argv[2])

    main(mhd_root=mhd_root, sub_processes=sub_processes)
    
                    
