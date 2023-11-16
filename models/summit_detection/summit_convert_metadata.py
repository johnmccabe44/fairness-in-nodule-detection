# Generates the Decathalon Challenge annotations from the defaul metadata generted from the proejct.
# The original version had 10 folds with each fold having a different training set to validation set
# even though fold0 contained all scans. Therefore fold0 was used in the preparation of the scans
# prior to training.

import sys
import pandas as pd
from pathlib import Path
import json


def main(cache_path, metadata_path):
    
    data_splits = ['training','validation']
    summit_subset = {'training' : [], 'validation' : []}
    summit_datasplits = {'training' : [], 'validation' : []}

    i = j = 0
    first = True
    for data_split in data_splits:

        scans = pd.read_csv(Path(metadata_path, data_split + '_scans.csv'))
        metadata = pd.read_csv(Path(metadata_path, data_split + '_metadata.csv'))
        
        for scan_id in scans.scan_id.tolist():
            study_id = scan_id.split('_',1)[0]

            scan_item = {
                'box' : [], 
                'image' : f'{study_id}/{scan_id}.mhd',
                'label' : []
            }

            for idx, row in metadata[metadata.main_participant_id==study_id].iterrows():
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

            summit_datasplits[data_split].append(scan_item)
            i += 1

            if not Path(cache_path,scan_item['image'].replace('mhd','nii.gz')).exists():
                summit_subset[data_split].append(scan_item)
                j+= 1

    with open('SUMMIT_datasplit/mhd_original/dataset_fold0.json','w') as f:
        json.dump(summit_datasplits, f, indent=4)

    with open('SUMMIT_datasplit/dataset_fold0.json', 'w') as f:
        json.dump(json.loads(json.dumps(summit_datasplits).replace('.mhd','.nii.gz')),f,indent=4)

    with open('SUMMIT_datasplit/mhd_original/dataset_fold0_subset.json','w') as f:
        json.dump(summit_subset, f)

    with open('SUMMIT_datasplit/dataset_fold0_subset.json', 'w') as f:
        json.dump(json.loads(json.dumps(summit_subset).replace('.mhd','.nii.gz')),f,indent=4)


if __name__ == '__main__':

    cache_path      = sys.argv[1]
    metadata_path   = sys.argv[2]

    main(cache_path, metadata_path)
