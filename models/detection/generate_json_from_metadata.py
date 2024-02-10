# Generates the Decathalon Challenge annotations from the defaul metadata generted from the proejct.
# The original version had 10 folds with each fold having a different training set to validation set
# even though fold0 contained all scans. Therefore fold0 was used in the preparation of the scans
# prior to training.
import argparse
import datetime
import logging
import sys
import pandas as pd
from pathlib import Path
import json
import argparse

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'generate_json_from_metadata_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'),
        logging.StreamHandler()
    ]
)

def main(name, cache_path, metadata_path, output_path, max_scans_per_ds={'training': None, 'validation': None, 'test': None}):
    
    dataset_json = {'training' : [], 'validation' : [], 'test' : []}

    for data_split in dataset_json.keys():


        if Path(metadata_path, data_split + '_scans.csv').exists():
            scans = pd.read_csv(Path(metadata_path, data_split + '_scans.csv'))
            metadata = pd.read_csv(Path(metadata_path, data_split + '_metadata.csv'))

            if max_scans_per_ds[data_split] is not None:
                logging.info(f'Limiting the number of nodules to {max_scans_per_ds[data_split]}')
                max = max_scans_per_ds[data_split] if max_scans_per_ds[data_split] < scans.shape[0] else scans.shape[0]
            else:
                logging.info('Not limiting the number of nodules')
                max = scans.shape[0]

            for cnt, scan_id in enumerate(scans.scan_id.tolist()):
                study_id = scan_id.split('_',1)[0]

                if not Path(cache_path, study_id, f'{scan_id}.nii.gz').exists():
                    logging.warning(f'{scan_id} not found')

                scan_item = {
                    'box' : [], 
                    'image' : f'{study_id}/{scan_id}.mhd',
                    'label' : []
                }

                for idx, row in metadata[metadata.participant_id==study_id].iterrows():
                    scan_item['box'].append(
                        [
                            row.nodule_x_coordinate,
                            row.nodule_y_coordinate,
                            row.nodule_z_coordinate,
                            row.nodule_diameter_mm,
                            row.nodule_diameter_mm,
                            row.nodule_diameter_mm
                        ])
                    scan_item['label'].append(0)

                dataset_json[data_split].append(scan_item)

                if cnt == max:
                    break

        else:
            logging.warning(f'No {data_split} scans found')
            
    # Create the output path
    Path(output_path, 'mhd_original').mkdir(parents=True, exist_ok=True)
    with open(Path(output_path, 'mhd_original', f'dataset_{name}.json'),'w') as f:
        json.dump(dataset_json, f, indent=4)

    with open(Path(output_path, f'dataset_{name}.json'), 'w') as f:
        json.dump(json.loads(json.dumps(dataset_json).replace('.mhd','.nii.gz')),f,indent=4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name of the dataset")
    parser.add_argument("--cache-path", help="Path to the cache")
    parser.add_argument("--metadata-path", help="Path to the metadata")
    parser.add_argument("--output-path", help="Path to the output")
    parser.add_argument("--max-training-scans", type=int, help="Maximum number of scans for training")
    parser.add_argument("--max-validation-scans", type=int, help="Maximum number of scans for validation")
    parser.add_argument("--max-test-scans", type=int, help="Maximum number of scans for test")

    args = parser.parse_args()

    name = args.name
    cache_path = args.cache_path
    metadata_path = args.metadata_path
    output_path = args.output_path
    max_scans_per_ds = {
        'training': args.max_training_scans,
        'validation': args.max_validation_scans,
        'test': args.max_test_scans
    }

    main(name, cache_path, metadata_path, output_path, max_scans_per_ds)