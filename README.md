# Fairness in Pulmonary Nodule Detection

## Overview

This project investigates fairness in pulmonary nodule detection by training and evaluating three open-source deep learning models across multiple datasets with different demographic stratifications. The goal is to assess how model performance varies across different patient populations and identify potential biases in automated nodule detection systems.

## Models

- **GRT123** - https://github.com/lfz/DSB2017/tree/master
- **MONAI Detection** - https://github.com/Project-MONAI/tutorials/tree/main/detection
- **TiCNet** - https://github.com/MIAinCS/TiCNet

Each model is treated as an independent detection pipeline and evaluated under identical dataset splits and fairness protocols.

## Datasets

- **LUNA** - https://luna16.grand-challenge.org/
    - Medical imaging dataset in metaIO format
    - 888 scans with annotations and lung segmentations (also in metaIO format)
    - Metadata:
        - Annotations (annotations.csv):
            - `seriesuid`: Unique series identifier
            - `coordX`, `coordY`, `coordZ`: 3D coordinates of annotation points
            - `diameter_mm`: Annotated region diameter in millimeters
            - `participant_details_gender` : participant gender/sex
            - `lung_health_check_demographics_race_ethnicgroup`: ethnic group
            - `nodule_count`, `nodule_count_cats` : nodules per scan
            
        - Exclusions (annotations_excluded.csv):
            - `seriesuid`: Unique series identifier
            - `coordX`, `coordY`, `coordZ`: 3D coordinates of annotation points (real world)
            - `diameter_mm`: Annotated region diameter in millimeters


- **SUMMIT** - 
    - Medical imaging dataset in DICOM/ metaIO format
    - Authorised access only
    - Metadata:
        - scan_metadata
            - `participant_id` : Unique subject identifier e.g. summit-1234-abc
            - `` : 
        - nodule_metadata
            - `participant_id` : Unique subject identifier e.g. summit-1234-abc
            - `scan_id` : Unique scan identifier e.g. summit-1234-abc_Y0_BASELINE_A
            - `nodule_x_coordinate`, `nodule_y_coordinate`, `nodule_z_coordinate` : 3D coordinate of annotation points (real world)
            - `nodule_diameter_mm` : Annotated region diameter in millimeters
        - nodule_metadata_excludes
            - `participant_id` : Unique subject identifier e.g. summit-1234-abc
            - `scan_id` : Unique scan identifier e.g. summit-1234-abc_Y0_BASELINE_A
            - `nodule_x_coordinate`, `nodule_y_coordinate`, `nodule_z_coordinate` : 3D coordinate of annotation points (real world)
            - `nodule_diameter_mm` : Annotated region diameter in millimeters

    - Datasets used:
        - Test Balanced (Total 5,940 Trn:5079, Val:267, Tst:594)
        - Male Only (Total 2,098 Trn: 1,573, Val: 105, Tst: 420)
        - White Only slice (Total 4,386 Trn: 3,364 Val: 224, Tst: 798)
        
- **LSUT** - 

## Project Structure

```
models/                    – upstream model implementations
datasets/                  – raw data and metadata
preprocessed/              – model-specific preprocessed outputs
workflow/                  – Metaflow workflows for result aggregation
notebooks/                 – analysis and plotting
results/                   – final metrics and figures
```

## Workflow

All models follow the same high-level experimental structure:

1. **Dataset-specific preprocessing** – Prepare imaging data and metadata for model ingestion
2. **Model training using upstream scripts** – Train detection models on dataset splits
3. **Inference on held-out test sets** – Generate predictions on withheld evaluation data
4. **Aggregation of detection outputs** – Collect and standardize results across models
5. **Fairness analysis and visualisation** – Evaluate performance disparities across demographic groups

The exact commands differ by model and are intentionally not unified.

## Running the Models

All models are executed according to their original upstream implementations. Refer to the individual `README.md` files in each model directory (`models/grt123`, `models/detection`, `models/ticnet`) for complete documentation.

To facilitate running across multiple datasets and demographic slices, additional parameters have been added to the original model scripts. The commands below show the high-level structure; exact parameters vary by model and dataset. Details of the modifications can be found in the `MODIFICATIONS.md` file located in each model directory.

### Model: grt123

**Phase 1: Preprocessing**
```bash
python prepare.py --dataset [luna|summit|lsut] --scanlist-path <path> --metadata-path <path> --datapath <path> --preprocess_result_path <path> [--n_worker_preprocessing 10]
```

**Phase 2: Training**
```bash
python main.py --save-dir <fold/dataset> --data-dir <preprocessed_path> --metadata-dir <metadata_path>
```

**Phase 3: Inference**
```bash
python main.py --prep-result-path <path> --bbox-result-path <path> --scanlist-path <path> --detector <path> --detector-parm <path> --run-detect
```

### Model: detection

**Phase 1: Preprocessing**
```bash
python detection_prepare_images.py --environment-file <env_config> --config-file <model_config>
```

**Phase 2: Training and Inference**
```bash
python detection_training.py --environment-file <env_config> --config-file <model_config> --data-base-dir <path> [--epochs 300 --batch-size 4]
python detection_testing.py --environment-file <env_config> --config-file <model_config>
```

### Model: ticnet

**Phase 1: Preprocessing**
```bash
python preprocess.py --scan-ids <path> --scans-root <path> --segmentations-root <path> --save-path <path>
python generate_lung_mask.py --scan-ids <path> --scans-path <path> --segmentation-path <path>
python cvrt_annos_to_npy.py --flavour <dataset> --annotations-file <path> [--additional-args]
```

**Phase 2: Training and Inference**
```bash
python train.py --batch-size 4 --epochs 300 --out-dir <path> --train-set-list <path> --val-set-list <path> --data-dir <path> --bbox-dir <path>
python test.py --mode eval --weight <path> --preprocessed-dir <path> --bbox-dir <path> --out-dir <path> --test-set-name <path>
```
> **Note:** Inference outputs are aggregated and evaluated using Metaflow workflows described in the [Reproducibility](#reproducibility) section.


## Reproducibility

This repository supports reproducibility to the maximum extent permitted by data
governance and ethical approvals.

### Public Dataset

Experiments on **LUNA16** are fully reproducible.

- LUNA16 is publicly available via the Grand Challenge
- Official challenge splits are used
- All preprocessing, training, inference, and evaluation steps can be rerun using
  the scripts provided in this repository

### Restricted Datasets

Experiments on **SUMMIT** and **LSUT** use restricted-access clinical data.

Due to governance and ethical constraints:
- Raw imaging data cannot be shared
- Scan lists, annotations, and dataset splits cannot be redistributed

Full end-to-end reproduction of these experiments is only possible for users
with authorised access to the underlying datasets.

### Result Aggregation and Evaluation

Inference outputs are generated using Metaflow workflows located in the `workflow/`
directory. A separate analysis notebook located in `notebooks/` combines these outputs to produce all
plots, tables, and summary metrics reported in this work.

These workflows and the notebook ensure that results are deterministic, auditable,
and reproducible at the metric and analysis level.

### Reproducibility Under Authorised Access

For users with appropriate approvals, the pipelines in this repository can be
used to reproduce all reported results by applying the documented preprocessing,
training, inference, and evaluation steps to locally resolved data.

All experiments use fixed dataset partitions and do not involve per-slice
hyperparameter tuning.

### Summary

- LUNA16 experiments are fully reproducible
- SUMMIT and LSUT experiments require authorised data access
- No restricted data or metadata are redistributed
- Experimental methodology and evaluation are fully documented
