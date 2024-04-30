
import sys
import data
from importlib import import_module
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_scan_list(path_to_scan_list):
    if path_to_scan_list.as_posix().endswith('.csv'):
        with open(path_to_scan_list, 'r') as f:
            return [
                scan_id 
                for scan_id in f.read().split('\n')
            ]

    return []

def main(datadir, metadata_dir):


    print('Loading model', flush=True)
    model = import_module('res18')
    config, net, loss, get_pbb = model.get_model()

    print('Loading training data', flush=True)

    trn_dataset = data.DataBowl3Detector(
        datadir,
        load_scan_list(Path(metadata_dir, 'training_scans.csv')),
        config,
        phase = 'train')

    print('Loading validation data', flush=True)

    val_dataset = data.DataBowl3Detector(
        datadir,
        load_scan_list(Path(metadata_dir, 'validation_scans.csv')),
        config,
        phase = 'val')
    
    print('Training data size:', trn_dataset.__len__(), flush=True)
    print('Validation data size:', val_dataset.__len__(), flush=True)

    print('Starting training', flush=True)
    
    for epoch in tqdm(range(100)):
        for idx in range(trn_dataset.__len__()):
            trn_dataset.__getitem__(idx)

    print('Starting validation', flush=True)
    for idx in tqdm(range(val_dataset.__len__())):
        val_dataset.__getitem__(idx)

if __name__ == '__main__':
    data_dir = sys.argv[1]
    metadata_dir = sys.argv[2]


    print('Data dir:', data_dir, flush=True)
    print('Metadata dir:', metadata_dir, flush=True)

    print('Starting test', flush=True)
    main(data_dir, metadata_dir)
