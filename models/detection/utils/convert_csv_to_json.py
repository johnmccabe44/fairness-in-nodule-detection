import json
import sys

import pandas as pd


def convert_csv_to_json(scans_path, metadata_path, json_path, dataset, scan_type):

    scans = pd.read_csv(scans_path)
    metadata = pd.read_csv(metadata_path)


    dataset_json = {'training' : [], 'validation' : [], 'test' : []}

    for scan_id in scans['scan_id'].values:
        df = metadata[metadata['scan_id'] == scan_id]

        if df.empty:

            dataset_json[dataset].append({
                'image': f'{scan_id}/{scan_id}{scan_type}',
                'box': [],
                'label' : []
            })

        else:
            for scan_id, group in df.groupby('scan_id'):
                dataset_json[dataset].append({
                    'image': f'{scan_id}/{scan_id}{scan_type}',
                    'box': group[[
                        'nodule_x_coordinate',
                        'nodule_y_coordinate',
                        'nodule_z_coordinate',
                        'nodule_diameter_mm',
                        'nodule_diameter_mm',
                        'nodule_diameter_mm'
                        ]].values.tolist(),
                    'label' : [0] * len(group)
                })

    with open(f'{json_path}/dataset.json', 'w') as f:
        json.dump(dataset_json, f)

if __name__ == '__main__':
    convert_csv_to_json(
        '/Users/john/Projects/SOTAEvaluationNoduleDetection/metadata/lsut/tranche1_scans.csv',
        '/Users/john/Projects/SOTAEvaluationNoduleDetection/metadata/lsut/tranche1_metadata.csv',
        '/Users/john/Projects/SOTAEvaluationNoduleDetection/models/detection/datasplits/lsut/mhd',
        'test',
        '.mhd'
    )