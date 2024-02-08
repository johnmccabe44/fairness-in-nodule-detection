# Generates the Decathalon Challenge annotations from the defaul metadata generted from the proejct.
# The original version had 10 folds with each fold having a different training set to validation set
# even though fold0 contained all scans. Therefore fold0 was used in the preparation of the scans
# prior to training.

import sys
import pandas as pd
from pathlib import Path
import json


def main(name, cache_path, metadata_path, output_path):
    
    dataset_json = {'training' : [], 'validation' : [], 'test' : []}

    i = j = 0
    first = True
    for data_split in dataset_json.keys():

        scans = pd.read_csv(Path(metadata_path, data_split + '_scans.csv'))
        metadata = pd.read_csv(Path(metadata_path, data_split + '_metadata.csv'))
        
        for scan_id in scans.scan_id.tolist():
            study_id = scan_id.split('_',1)[0]

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
            i += 1


    # Create the output path
    output_path = Path(output_path, 'mhd_original').mkdir(parents=True, exist_ok=True)
    with open(Path(output_path, 'mhd_original', f'dataset_{name}.json'),'w') as f:
        json.dump(dataset_json, f, indent=4)

    with open(Path(output_path, f'dataset_{name}.json'), 'w') as f:
        json.dump(json.loads(json.dumps(dataset_json).replace('.mhd','.nii.gz')),f,indent=4)

if __name__ == '__main__':

    name            = sys.argv[1]  
    cache_path      = sys.argv[2]
    metadata_path   = sys.argv[3]
    output_path     = sys.argv[4]



    main(name, cache_path, metadata_path, output_path)
