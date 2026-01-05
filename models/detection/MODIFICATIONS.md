# Changes Relative to MONAI Detection Tutorial

## Overview

This repository is based on the MONAI Detection Tutorial as available at:
https://github.com/Project-MONAI/tutorials.

It focuses on training, data loading, transforms, inference, configuration, and logging improvements.

> **Note on Source Code**
>
> Code sourced from [MONAI tutorials detection repository](https://github.com/Project-MONAI/tutorials), downloaded on 6 September 2023. The MONAI repository has undergone numerous updates since this date which have not been incorporated into this implementation. Users are advised to consult the current MONAI documentation for the latest methodological advances and API changes.


## Highlights

- Logging over print, with configurable levels
- CLI additions for reproducible runs (epochs, batch size, workers, resume)
- Proper split handling via datalists and base directory
- Configurable training loop (epochs, start epoch) and checkpoint resume
- Safer AMP usage across CPU/CUDA
- Data pipeline refactor to box-aware transforms
- SUMMIT-focused environment/config defaults
- More robust inference and visualization behavior

---

## Logging & Diagnostics

- Add global logging configuration to stream to stdout at DEBUG level.
- Replace `print(...)` with `logging.info(...)` and `logging.debug(...)` for epoch/step/loss timing and metrics.
- Log hardware/training parameters at startup: number of GPUs, batch size, workers.

```python
# Added near imports
import logging, sys

# Before
# print("epoch", epoch)

# After: configure and use logging
logging.basicConfig(
  level=logging.DEBUG,
  format="%(asctime)s [%(levelname)s] %(message)s",
  handlers=[logging.StreamHandler(sys.stdout)],
)
logging.info(f"Number of gpus:{args.gpus}, batch size: {args.batch_size}, args.workers: {args.workers}")
```

## CLI & Configuration

- New arguments:
  - `--data-base-dir` (base folder for NIfTI input)
  - `--epochs`, `--batch-size`, `--gpus`, `--workers`
  - `--resume`, `--start-epoch`
- Simplify JSON loading: `json.load(open(...))`.
- Validation interval updated from 5 to 10 by default.

```python
# Added CLI args
parser.add_argument("-d", "--data-base-dir", help="base folder where niftis are stored")
parser.add_argument("-x", "--epochs", type=int, default=300)
parser.add_argument("-b", "--batch-size", type=int, default=1)
parser.add_argument("-g", "--gpus", type=int, default=1)
parser.add_argument("-w", "--workers", type=int, default=7)
parser.add_argument("-r", "--resume", action="store_true", default=False)
parser.add_argument("-s", "--start-epoch", type=int, default=0)

# Before
env_dict = json.load(open(args.environment_file, "r"))
config_dict = json.load(open(args.config_file, "r"))

# Validation cadence
val_interval = 10  # was 5
```

## Data Lists & Splits

- Training loader now uses full `training` set from datalist (no implicit 95/5 split).
- Validation is loaded explicitly from datalist (`data_list_key="validation"`).
- In some utilities, dataset key becomes configurable via `args.dataset_name`.
- Add support to iterate `training`, `validation`, and `test` sets in preparation flows.

```python
# Before (implicit 95/5 split)
train_ds = CacheDataset(data=train_data[: int(0.95 * len(train_data))], ...)
val_ds = CacheDataset(data=train_data[int(0.95 * len(train_data)) :], ...)

# After (explicit lists)
train_ds = CacheDataset(data=train_data, ...)
validation_data = load_decathlon_datalist(
  args.data_list_file_path, is_segmentation=True, data_list_key="validation", base_dir=args.data_base_dir
)
val_ds = CacheDataset(data=validation_data, ...)
```

## DataLoader Defaults

- `batch_size` from `1` to `args.batch_size`.
- `num_workers` from `7` to `args.workers`.
- `pin_memory=False` (was `torch.cuda.is_available()`).

```python
# Before
train_loader = DataLoader(train_ds, batch_size=1, num_workers=7, pin_memory=torch.cuda.is_available())

# After
train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=False)
```

## Training & Checkpointing

- Add resume functionality:
  - If `--resume`, load TorchScript model from `env_dict["model_path"]` with `_last.pt` suffix.
  - Start epochs from `--start-epoch`.
- `max_epochs` from fixed `300` to `args.epochs`.
- Use `torch.amp.GradScaler(device=device)` and `torch.amp.autocast(device_type=device.type)` for AMP.
- Replace epoch/step printing with logging and include timing.

```python
# Resume logic
if args.resume:
  start_epoch = args.start_epoch
  net = torch.jit.load(env_dict["model_path"][:-3] + "_last.pt")
else:
  start_epoch = 0
  # build backbone/feature_extractor/net ...

# Epochs
max_epochs = args.epochs  # was fixed 300
for epoch in range(start_epoch, max_epochs):
  logging.info(f"epoch {epoch + 1}/{max_epochs}")

# AMP
scaler = torch.amp.GradScaler(device=device)
with torch.amp.autocast(device_type=device.type):
  loss = compute_loss(...)

logging.debug(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
```

## Model Construction

- Build RetinaNet backbone/feature extractor as before when not resuming.
- Compute anchors and size divisibility from feature extractor.
- TorchScript the constructed detector for train-time usage.

```python
num_anchors = anchor_generator.num_anchors_per_location()[0]
size_divisible = [s * 2 * 2 ** max(args.returned_layers) for s in feature_extractor.body.conv1.stride]
net = torch.jit.script(
  RetinaNet(
    spatial_dims=args.spatial_dims,
    num_classes=len(args.fg_labels),
    num_anchors=num_anchors,
    feature_extractor=feature_extractor,
    size_divisible=size_divisible,
  )
)
```

## Validation & Visualization

- Switch to `logging.info` for validation timing and metric summaries.
- Visualize a random validation sample each validation phase (random index), rather than the first.
- Use device-agnostic autocast for validation (`torch.amp.autocast(device_type=...)`).

## Inference Pipeline

- `data_list_key` becomes configurable via `args.dataset_name` in some scripts.
- Results dictionary keyed by dataset name instead of hard-coded `validation`.
- Access image filename via MONAI metadata: `inference_data_i["image"].meta["filename_or_obj"]`.
- Wrap per-batch inference in try/except to surface file-specific errors.

```python
results_dict = {args.dataset_name: []}  # was {"validation": []}

try:
  inference_img_filenames = [d["image"].meta["filename_or_obj"] for d in inference_data]
  use_inferer = not all([d["image"][0, ...].numel() < np.prod(patch_size) for d in inference_data])
  inference_inputs = [d["image"].to(device) for d in inference_data]
  with torch.amp.autocast(device_type=device.type):
    inference_outputs = detector(inference_inputs, use_inferer=use_inferer)
  # post transforms and packing results
except Exception as err:
  print(f"Error: {err} in file: {inference_img_filenames}")
```

## Dataset/Environment Setup (SUMMIT)

- Environment/config defaults updated from LUNA16 to SUMMIT:
  - Default environment JSON: `environment_summit_prepare.json`
  - Default training config JSON: `config_train_summit_16g.json`
- Environment generator accepts a `location` argument (`cluster`/`mac`) and sets raw/resampled base dirs accordingly.
- Output artifacts (model path, event dir, inference JSON) names switched from `luna16_*` to `summit_*`.
- Fold iteration reduced from `range(10)` to `range(1)` in the SUMMIT setup example.

Concrete example:

```python
# detection_prepare_env_files.py
location = sys.argv[1]
if location == 'cluster':
  raw_data_base_dir = "/cluster/project2/SummitLung50"
  resampled_data_base_dir = "/cluster/project2/SUMMIT/cache/summit_detection"
if location == 'mac':
  raw_data_base_dir = "/Users/john/Projects/SOTAEvaluationNoduleDetection/scans/lung50"
  resampled_data_base_dir = "/Users/john/Projects/SOTAEvaluationNoduleDetection/scans/nifti"

downloaded_datasplit_dir = "SUMMIT_datasplit"
out_file = "config/environment_summit_prepare.json"

for fold in range(1):
  out_file = f"config/environment_summit_fold{fold}.json"
  env_dict["model_path"] = os.path.join(out_trained_models_dir, f"model_summit_fold{fold}.pt")
  env_dict["tfevent_path"] = os.path.join(out_tensorboard_events_dir, f"summit_fold{fold}")
  env_dict["out_inference_result_file"] = os.path.join(out_inference_result_dir, f"result_summit_fold{fold}.json")
```

## Visualization Robustness

- When no GT boxes are present, visualize a default region and warn instead of failing.
- Change GT box color from green to blue to improve colorblind accessibility.

Concrete example:

```python
# visualize_image.py
if len(gt_boxes) > 0:
  draw_box = gt_boxes[gt_box_index, :]
else:
  draw_box = [5, 5, 5, 5, 5, 5]
  print('Selected image has no GT box, visualize the whole image instead.')

# Color change
cv2.rectangle(img, pt1, pt2, color=(0, 0, 255))  # was (0, 255, 0)
```

---

## Notes

- Device handling is consistently abstracted via `device.type` across AMP and scaler usage.
- Various minor style consolidations (import formatting, metadata key usage) were applied for clarity and resilience.