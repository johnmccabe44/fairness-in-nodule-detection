
import sys
import data
from importlib import import_module
import logging
from pathlib import Path
from torch.utils.data import DataLoader
import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('test_data_loader.log'),
        logging.StreamHandler()
    ])

def load_scan_list(path_to_scan_list):
    if path_to_scan_list.as_posix().endswith('.csv'):
        with open(path_to_scan_list, 'r') as f:
            return [
                scan_id 
                for scan_id in f.read().split('\n')
            ]

    return []

def main(datadir, metadata_dir):

    model = import_module('res18')
    config, net, loss, get_pbb = model.get_model()

    trn_dataset = data.DataBowl3Detector(
        datadir,
        load_scan_list(Path(metadata_dir, 'training_scans.csv')),
        config,
        phase = 'train')

    val_dataset = data.DataBowl3Detector(
        datadir,
        load_scan_list(Path(metadata_dir, 'validation_scans.csv')),
        config,
        phase = 'val')
    
    for epoch in tqdm(range(100)):
        for idx in trn_dataset.scan_list:
            pass

    for idx in tqdm(range(val_dataset.__len__())):
        pass

if __name__ == '__main__':
    data_dir = sys.argv[1]
    metadata_dir = sys.argv[2]

    main(data_dir, metadata_dir)
