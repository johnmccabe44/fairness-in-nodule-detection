# DSB2017 to Fairness-in-Nodule-Detection: Migration Summary

This document details all changes made to migrate the original DSB2017 codebase to the modernized fairness-in-nodule-detection project.

## Table of Contents

1. [PyTorch Modernization & GPU Compatibility](#1-pytorch-modernization--gpu-compatibility)
2. [Python 3 Integer Division Fixes](#2-python-3-integer-division-fixes)
3. [Preprocessing Pipeline Updates](#3-preprocessing-pipeline-updates)
4. [Configuration & Path Management](#4-configuration--path-management)
5. [Data Loading & Compatibility](#5-data-loading--compatibility)
6. [Function Signature Updates](#6-function-signature-updates)
7. [Configuration Parameter Changes](#7-configuration-parameter-changes)
8. [Error Handling & Logging](#8-error-handling--logging)
9. [Utility Functions](#9-utility-functions)
10. [Import & Dependency Updates](#10-import--dependency-updates)

---

## 1. PyTorch Modernization & GPU Compatibility

### Issue
- Old code used deprecated PyTorch syntax (`.data[0]`, `.cuda()`, `Variable`, `volatile=True`)
- Code was hard-coded to specific GPU devices
- Used old CUDA async syntax

### Changes
- Replace `loss.data[0]` with `loss.data`
- Replace `Variable(tensor).cuda()` with `tensor.to(device, non_blocking=True)`
- Remove `torch.cuda.set_device(0)` and use device abstraction
- Replace `volatile=True` with `torch.no_grad()` context (implicitly via eval mode)

### Affected Files
- `training/classifier/trainval_classifier.py`
- `training/classifier/trainval_detector.py`
- `training/detector/main.py`
- `training/detector/layers.py`
- `layers.py`

### Example Changes

**Before:**
```python
coord = Variable(coord).cuda()
x = Variable(x).cuda()
loss_val = loss.data[0]
```

**After:**
```python
coord = coord.to(device, non_blocking=True)
x = x.to(device, non_blocking=True)
loss_val = loss.data
```

---

## 2. Python 3 Integer Division Fixes

### Issue
- Python 2 used `/` for integer division; Python 3 requires `//` or `int()`
- Slicing operations require integer indices

### Changes
- Wrap division operations with `int()`: `int(featshape[2]/2-1)`
- Update array slicing to use proper integer types

### Affected Files
- `models/grt123/net_classifier.py`
- `models/grt123/split_combine.py`
- `data_classifier.py`
- `data_detector.py`
- `preprocessing/full_prep.py`

### Example Changes

**Before:**
```python
centerFeat = self.pool(noduleFeat[:,:,featshape[2]/2-1:featshape[2]/2+1,
                                  featshape[3]/2-1:featshape[3]/2+1,
                                  featshape[4]/2-1:featshape[4]/2+1])
side_len /= stride
margin /= stride
```

**After:**
```python
centerFeat = self.pool(noduleFeat[:,:,int(featshape[2]/2-1):int(featshape[2]/2+1),
                                  int(featshape[3]/2-1):int(featshape[3]/2+1),
                                  int(featshape[4]/2-1):int(featshape[4]/2+1)])
side_len = int(side_len/stride)
margin = int(margin/stride)
```

---

## 3. Preprocessing Pipeline Updates

### SUMMIT Dataset Support

**New Functions:**
- `full_prep_summit()` - Handle SUMMIT dataset structure
- `step1_python_summit()` - SUMMIT-specific image loading
- `load_metaio_scan()` - Load `.mhd` format files using SimpleITK
- `savenpy_summit()` - SUMMIT-specific preprocessing

### Metadata Handling

- Updated `savenpy()` and `savenpy_luna()` to accept metadata from CSV files (pandas)
- Annotations now loaded from CSV with columns: `scan_id`, `nodule_x_coordinate`, `nodule_y_coordinate`, `nodule_z_coordinate`, `nodule_diameter_mm`
- Added `filename` parameter to `label_mapping()` for better error reporting
- Added `use_existing` flag to skip re-preprocessing files

### Affected Files
- `training/prepare.py`
- `preprocessing/full_prep.py`
- `preprocessing/step1.py`
- `training/detector/data.py`

### Example Changes

**step1_python function signature update:**
```python
# Before
return case_pixels, bw1, bw2, spacing

# After
return case_pixels, bw1, bw2, spacing, origin
```

**New SUMMIT support:**
```python
def step1_python_summit(case_path):
    case_pixels, spacing, origin, orientation = load_metaio_scan(case_path)
    # ... processing ...
    return case_pixels, bw1, bw2, spacing, origin
```

---

## 4. Configuration & Path Management

### Issue
- Hard-coded paths specific to original environment
- Configuration file referenced non-existent paths

### Changes
- Migrate to command-line arguments for data paths
- Use `Path` objects instead of string concatenation
- Support both local paths and dataset-specific structures

### Affected Files
- `training/config_training.py`
- `training/detector/main.py`
- `config_submit.py`
- `main.py`

### Example Changes

**config_training.py:**
```python
# Before
config = {'stage1_data_path':'/work/DataBowl3/stage1/stage1/',
          'luna_raw':'/work/DataBowl3/luna/raw/',
          ...}

# After
config = {
    'luna_segment': '/Users/john/Projects/SOTAEvaluationNoduleDetection/scans/luna16/segmentations',
    'luna_data': '/Users/john/Projects/SOTAEvaluationNoduleDetection/scans/luna16/mhd',
    'preprocess_result_path': '/Users/john/Projects/SOTAEvaluationNoduleDetection/models/grt123/prep_result',
    'bbox_path': '/Users/john/Projects/SOTAEvaluationNoduleDetection/models/grt123/bbox_result',
    'use_existing': True,
    'recreate_labels': True,
    'n_worker_preprocessing': 1,
    'preprocessing_backend': 'python',
    'test': cwd
}
```

---

## 5. Data Loading & Compatibility

### Issue
- File paths were split on delimiters that may not exist
- DICOM loading used deprecated `dicom` library
- Collections type checking outdated for Python 3.10+

### Changes
- Use `os.path.basename()` for safer path parsing
- Replace `import dicom` with `import pydicom`
- Update `collections.Iterable` to `collections.abc.Iterable`
- Add `use_existing` flag to skip re-preprocessing files

### Affected Files
- `test_detect.py`
- `data_detector.py`
- `preprocessing/step1.py`

### Example Changes

**Path parsing:**
```python
# Before
name = data_loader.dataset.filenames[i_name].split('-')[0].split('/')[-1]
shortname = name.split('_clean')[0]

# After
name = os.path.basename(data_loader.dataset.filenames[i_name]).split('_clean')[0]
shortname = name
```

**Collections import:**
```python
# Before
from collections import Iterable
elif isinstance(batch[0], Iterable):

# After
from collections.abc import Iterable
elif isinstance(batch[0], collections.abc.Iterable):
```

**DICOM import:**
```python
# Before
import dicom
slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

# After
import pydicom
slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path) if s.endswith('.dcm')]
```

---

## 6. Function Signature Updates

### Device Parameter Addition

Added `device` parameter to training/validation/testing functions:

- `train_casenet(epoch, model, data_loader, optimizer, **device**, args)`
- `val_casenet(epoch, model, data_loader, **device**, args)`
- `test_casenet(model, testset, **device**)`
- `train_nodulenet(..., **device**, ...)`
- `validate_nodulenet(..., **device**)`
- `test_nodulenet(..., **device**)`
- `test_detect(..., **device**)`

### Filename Parameter Addition

- `label_mapping(..., **filename**)` - For error reporting during label generation

### Example Changes

```python
# Before
def train_casenet(epoch, model, data_loader, optimizer, args):
    # ...

# After
def train_casenet(epoch, model, data_loader, optimizer, device, args):
    # ...
```

---

## 7. Configuration Parameter Changes

### Size Limit Adjustments

- `config['sizelim']` reduced from `6.0 mm` to `1.0 mm` (more sensitive detection)
- In alternative: changed to `5.0 mm`

### Config Structure

- Removed dataset-specific hardcoded paths
- Added `use_existing` and `recreate_labels` flags
- Added `n_worker_preprocessing` parameter

### Example Changes

```python
# Before
config['sizelim'] = 6. #mm

# After
config['sizelim'] = 1. #mm
```

---

## 8. Error Handling & Logging

### Issue
- Missing error handling for edge cases
- Limited debug information

### Changes
- Added try-except blocks in preprocessing functions
- Added filename to assertion errors
- Added GPU memory debugging functions (`print_gpu_stats()`)
- Added flush=True to print statements for better logging

### Affected Files
- `training/prepare.py`
- `preprocessing/full_prep.py`
- `training/detector/main.py`

### Example Changes

```python
# Enhanced error handling in savenpy_luna()
try:
    # ... processing code ...
except Exception as e:
    print(e, flush=True)
    print(name+' preprocessing failed', flush=True)

# Enhanced assertion with filename
assert input_size[i] % stride == 0, f'Error: index: {i}, input_size: {input_size[i]} is not divisible by stride:{stride}, file:{filename}'

# GPU monitoring
def print_gpu_stats(device, msg, debug=False):
    """Prints GPU memory statistics for the specified device."""
    if device.type == 'cuda' and debug==True:
        # ... GPU memory checks ...
```

---

## 9. Utility Functions

### New Functions Added

- `load_scan_list()` - Support for `.npy` and `.csv` scan list formats
- `print_gpu_stats()` - Monitor GPU memory usage
- `parse_arguments()` - Command-line argument parsing
- `get_scanlist()` - Filter scans by preprocessed file existence
- `run_prepare()`, `run_detect()`, `run_classify()` - Modularized pipeline stages
- `load_metaio_scan()` - Load `.mhd` format files

### Example Changes

```python
def load_scan_list(path_to_scan_list):
    """Load a list of scan IDs from a file (supports .npy and .csv)."""
    if path_to_scan_list.as_posix().endswith('.npy'):
        return np.load(path_to_scan_list)
    
    if path_to_scan_list.as_posix().endswith('.csv'):
        with open(path_to_scan_list, 'r') as f:
            return [scan_id for scan_id in f.read().split('\n')]
    
    return []

def get_scanlist(scanlist_path, prep_result_path):
    """Filter scans by preprocessed file existence."""
    return [
        scan_id
        for scan_id in pandas.read_csv(scanlist_path)['scan_id'].tolist()
        if os.path.exists(os.path.join(prep_result_path, scan_id + '_clean.npy'))
    ]
```

---

## 10. Import & Dependency Updates

### Replaced/Updated Imports

```python
# Old
import dicom
from collections import Iterable

# New
import pydicom
from collections.abc import Iterable
```

### New Dependencies

- `pathlib.Path` for path manipulation
- `argparse` for CLI arguments
- `pynvml` for GPU monitoring
- `pandas` for metadata handling
- `SimpleITK` for medical image I/O

### Example Changes

```python
# Before
import sys
sys.path.append('../')

# After
from pathlib import Path
import argparse
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
import pandas
import SimpleITK as sitk
```

---

## Summary of Benefits

✅ **Python 3 Compatibility** - Removed all Python 2 specific code  
✅ **GPU Flexibility** - Support for CPU/GPU and multi-GPU setups  
✅ **Dataset Flexibility** - Support for LUNA16 and SUMMIT datasets  
✅ **Error Reporting** - Better debugging with filenames in error messages  
✅ **Metadata Management** - CSV-based annotation handling  
✅ **Preprocessing Efficiency** - Skip existing preprocessed files  
✅ **Modular Pipeline** - Independent preparation, detection, classification steps  
✅ **Improved Logging** - Better visibility into preprocessing and training steps  
✅ **Device Abstraction** - CPU/GPU agnostic code  
✅ **Path Management** - Robust path handling with pathlib

---

## Migration Checklist

- [x] PyTorch API modernization
- [x] Python 3 compatibility
- [x] GPU device abstraction
- [x] SUMMIT dataset support
- [x] Metadata CSV handling
- [x] Configuration refactoring
- [x] Error handling improvements
- [x] CLI argument parsing
- [x] Modular pipeline structure
- [x] Import updates
- [x] Logging enhancements
- [x] Path handling improvements
ß