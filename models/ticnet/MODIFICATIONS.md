# Changes: TICNet Updates

This document summarizes updates captured in `models/ticnet/ticnet_diff.txt`. It covers model execution, dataset I/O, configuration, preprocessing, annotation conversion, evaluation, and training. Concrete before/after examples are included.

## Highlights

- Remove ad-hoc `data_parallel(...)` wrappers; call modules directly and use `nn.DataParallel` at construction when needed.
- Switch image I/O from NRRD (`.nrrd`) to NumPy (`.npy`).
- Add `bbox_dir` to decouple labels from images across dataset, train, and eval.
- More robust cropping with context-rich error diagnostics.
- CLI-first preprocessing and annotation conversion with mapping-driven column handling.
- Device-agnostic evaluation; optional multi-GPU via `DataParallel`.
- Config tweaks: `blacklist`, `epoch_save`, and new `BBOX_DIR`.

---

## Model Forward Path

- Replace custom `data_parallel(self.module, x)` calls with direct module calls; rely on `nn.DataParallel` if multiple GPUs are available.

```python
# Before
features, feat_4 = data_parallel(self.feature_net, inputs)
self.rpn_logits_flat, self.rpn_deltas_flat = data_parallel(self.rpn, fs)
self.rcnn_logits, self.rcnn_deltas = data_parallel(self.rcnn_head, rcnn_crops)

# After
features, feat_4 = self.feature_net(inputs)
self.rpn_logits_flat, self.rpn_deltas_flat = self.rpn(fs)
self.rcnn_logits, self.rcnn_deltas = self.rcnn_head(rcnn_crops)
```

---

## Configuration Tweaks

- Defaults adjusted:
  - `blacklist`: `[]` → `['scan_id']`
  - `epoch_save`: `1` → `10`
  - add `BBOX_DIR`: `''`

```python
config = {
    # ...
    'blacklist': ['scan_id'],
    'epoch_save': 10,
    'BBOX_DIR': '',
}
```

---

## Dataset: BboxReader and Cropping

- Signature change: add `bbox_dir` and support `.csv` and `.txt` set lists.
- Read bboxes from `bbox_dir` instead of `data_dir`.
- Swap image I/O from NRRD to NumPy `.npy`.
- Initialize empty GT tensors instead of `None` (device-friendly).
- Cropping improvements:
  - Accept `filename` for better error messages.
  - Ensure `target[3]` (diameter) has a positive floor (set to 5 if 0).
  - Wrap `np.pad` in try/except with full context logging.

```python
# __init__ signature
# Before
def __init__(self, data_dir, set_name, cfg, mode='train', split_combiner=None):
    ...
# After
def __init__(self, data_dir, bbox_dir, set_name, cfg, mode='train', split_combiner=None):
    self.bbox_dir = bbox_dir
```

```python
# Set list handling & bbox loading
# Before
if set_name.endswith('.csv'):
    ...
l = np.load(os.path.join(data_dir, f'{fn}_bboxes.npy'))
# After
if set_name.endswith('.csv') or set_name.endswith('.txt'):
    ...
l = np.load(os.path.join(bbox_dir, f'{fn}_bboxes.npy'))
```

```python
# Image I/O
# Before (NRRD)
img, _ = nrrd.read(os.path.join(self.data_dir, f'{path_to_img}_seg.nrrd'))
# After (NumPy)
img = np.load(os.path.join(self.data_dir, f'{path_to_img}.npy'))
```

```python
# GT placeholders
# Before
truth_bboxes = None
truth_labels = None
# After
truth_bboxes = torch.empty(0, 4, dtype=torch.float32)
truth_labels = torch.empty(0, dtype=torch.long)
```

```python
# Crop call & sanity check
sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes, isScale, is_random_crop, filename=filename)
if sample.shape != (1, 128, 128, 128):
    print(filename, sample.shape)
```

```python
# Inside crop: enforce diameter & robust padding
if target[3] == 0:
    target[3] = 5
try:
    crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)
except Exception as e:
    print('ERROR')
    print('filename:', filename, flush=True)
    print('crop shape:', crop.shape, flush=True)
    print('pad:', pad, flush=True)
    print('crop size:', crop_size, flush=True)
    print('start:', start, flush=True)
    print('imgs shape:', imgs.shape, flush=True)
    raise e
```

---

## Preprocessing Pipeline (Masks/Volumes)

- Refactor to CLI-driven, logged, parallelizable process with resilient path resolution:
  - Resolve either flat or study-based directories for scans/masks.
  - Skip already processed scans unless `--overwrite`.
  - Use a worker pool (`--workers`) or sequential path.
  - Accumulate per-scan logs and emit via a central logger.

```python
# Args & logging
parser.add_argument('--scan-ids', nargs='+', type=Path)
parser.add_argument('--scans-root', type=Path)
parser.add_argument('--segmentations-root', type=Path)
parser.add_argument('--save-path', type=Path)
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--batch-size', type=int, default=-1)
parser.add_argument('--batch-number', type=int, default=1)
parser.add_argument('--workers', type=int, default=1)

logger = setup_logging()

# Path resolution
scan_path = scans_root / f'{scan_id}.mhd' if (scans_root / f'{scan_id}.mhd').exists() else scans_root / study_id / f'{scan_id}.mhd'
mask_path = mask_root / f'{scan_id}.mhd' if (mask_root / f'{scan_id}.mhd').exists() else mask_root / study_id / f'{scan_id}.mhd'

# Early exits
if not scan_path.exists():
    return {'level': logging.ERROR, 'created': time.time(), 'msg': f'Scan not found: {scan_id}'}
if not mask_path.exists():
    return {'level': logging.ERROR, 'created': time.time(), 'msg': f'Mask not found: {scan_id}'}
if not overwrite and os.path.exists(os.path.join(save_dir, f'{scan_id}.npy')):
    return {'level': logging.INFO, 'created': time.time(), 'msg': f'{scan_id} already processed'}
```

---

## Annotations Conversion

- Replace config-based script with a flexible CLI and mapping file (`mappings.json`).
- Apply `rename_columns`, `assign_columns`, and `drop_columns` as defined in mappings.
- Save transformed annotations under `--transformed-annotations-dir / --flavour`.
- Save per-scan bboxes to `--bbox-dir` using preprocessed metadata (`origin`, `spacing`, `ebox`).
- Optional `--throttle` for subset processing; write `${dataset}_scans.txt` for traceability.

```python
# CLI
parser.add_argument('--flavour', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--annotations-file', type=Path)
parser.add_argument('--annotations-excluded-file', type=Path)
parser.add_argument('--mappings-file', type=Path)
parser.add_argument('--scan-id-file', type=Path)
parser.add_argument('--preprocessed-dir', type=Path)
parser.add_argument('--bbox-dir', type=Path)
parser.add_argument('--transformed-annotations-dir', type=Path)
parser.add_argument('--throttle', type=int)

# Mapping application
annotations_data = pd.read_csv(
    annotations_file,
    usecols=mappings['rename_columns'].keys() if mappings['rename_columns'] else None,
)
if 'rename_columns' in mappings:
    annotations_data = annotations_data.rename(columns=mappings['rename_columns'])
if 'assign_columns' in mappings:
    annotations_data = annotations_data.assign(
        **{mappings['assign_columns']['column']: eval(mappings['assign_columns']['formula'])}
    )
if 'drop_columns' in mappings:
    annotations_data = annotations_data.drop(columns=mappings['drop_columns'])

# Output
transformed_annotations_file = output_path / annotations_file.name
with open(transformed_annotations_file, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"])
    # ... write rows ...

# Per-scan bbox output
np.save(os.path.join(bbox_dir, f'{uid}_bboxes.npy'), np.array(new_annos))
```

---

## Evaluation (`test.py`)

- CLI expanded to explicit paths: `--preprocessed-dir`, `--bbox-dir`, `--annotations-path`, `--annotations-excluded-path`.
- Device-agnostic; wrap with `nn.DataParallel` if multiple GPUs.
- `BboxReader` now requires `bbox_dir`.
- External `noduleCADEvaluation` calls commented (can be re-enabled by user context).

```python
# Device & DataParallel
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Dataset wiring
dataset = BboxReader(data_dir, bbox_dir, test_set_name, net_config, mode='eval')

# Eval signature & device tensors
def eval(net, dataset, annotations_path=None, annotations_excluded_path=None, save_dir=None):
    input = input.unsqueeze(0).to(device)
    truth_bboxes = truth_bboxes.to(device)
    truth_labels = truth_labels.to(device)
```

---

## Training (`train.py`)

- Add `--bbox-dir` and pass through to `BboxReader` for train/val.
- Keep stdout logging (remove prior redirection to file via `Logger`).

```python
parser.add_argument('--bbox-dir', default=train_config['BBOX_DIR'], type=str, metavar='OUT', help='Path to bboxes')

# Dataset wiring
if mode == 'train':
    dataset = BboxReader(args.data_dir, args.bbox_dir, set_name, net_config, mode='train')
else:
    dataset = BboxReader(args.data_dir, args.bbox_dir, set_name, net_config, mode='val')

# Logger redirection disabled
# logfile = os.path.join(args.out_dir, 'log_train.txt')
# sys.stdout = Logger(logfile)
```

---

## Miscellaneous

- Import order tidied; remove hard-coded `CUDA_VISIBLE_DEVICES` lines (left commented).
- Add small debug prints for unexpected sample shapes during dataset iteration.

```python
# Was
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# Now
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
```

---

## Notes

- These changes make TICNet pipelines more portable (paths/CLIs), robust (error handling), and flexible (device and multi-GPU). Introducing `bbox_dir` decouples labels from images and aligns preprocessing and annotations conversion.
