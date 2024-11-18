import argparse
import csv
import json
import shutil
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys

from tqdm import tqdm


def parse_args():

    parser = argparse.ArgumentParser(description='Convert annotations to numpy array')

    parser.add_argument(
        '--flavour',
        type=str, 
        help='Flavour'
    )

    parser.add_argument(
        '--dataset',
        type=str, 
        help='Dataset'
    )
    
    parser.add_argument(
        '--annotations-file',
        type=Path, 
        help='Path to annotations'
    )

    parser.add_argument(
        '--annotations-excluded-file',
        type=Path, 
        help='Path to annotations to be excluded'
    )

    parser.add_argument(
        '--mappings-file',
        type=Path,
        help='Path to mappings file'
    )

    parser.add_argument(
        '--scan-id-file',
        type=Path,
        help='List of scan ids'
    )

    parser.add_argument(
        '--preprocessed-dir', 
        type=Path, 
        help='Path to preprocessed data'
    )

    parser.add_argument(
        '--bbox-dir',
        type=Path,
        help='Path to save bounding boxes'
    )

    parser.add_argument(
        '--transformed-annotations-dir', 
        type=Path, 
        help='Path to save annotations'
    )

    parser.add_argument(
        '--throttle',
        type=int,
        help='Throttle the number of annotations to be converted'
    )

    args = parser.parse_args()
    return args

def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def get_anno_dict(annotations_file, mappings):
    annotations_data = pd.read_csv(annotations_file, usecols=mappings['rename_columns'].keys() if mappings['rename_columns'] else None)

    if 'rename_columns' in mappings.keys():
        annotations_data = annotations_data.rename(columns=mappings['rename_columns'])

    if 'assign_columns' in mappings.keys():
        annotations_data = annotations_data.assign(**{mappings['assign_columns']['column']: eval(mappings['assign_columns']['formula'])})

    if 'drop_columns' in mappings.keys():
        annotations_data = annotations_data.drop(columns=mappings['drop_columns'])

    uid = annotations_data["seriesuid"]
    data = annotations_data[["coordX", "coordY", "coordZ", "diameter_mm"]]
    data = np.array(data)
    annotations_dict = dict([(id, []) for id in uid])
    for i in range(len(uid)):
        annotations_dict[uid[i]].append(data[i])

    return annotations_dict

def generate_label(annos_dict, scan_ids, preprocessed_dir, bbox_dir):
    
    for uid in tqdm(scan_ids):
        
        origin = np.load(preprocessed_dir / f'{uid}_origin.npy')
        spacing = np.load(preprocessed_dir / f'{uid}_spacing.npy')
        ebox = np.load(preprocessed_dir / f'{uid}_ebox.npy')

        new_annos = []
        if uid in annos_dict.keys():
            annos = annos_dict[uid]
            for anno in annos:
                anno[[0, 1, 2]] = anno[[2, 1, 0]]
                coord = anno[:-1]
                new_coord = worldToVoxelCoord(coord, origin, spacing) * spacing - ebox
                new_coord = np.append(new_coord, anno[-1])
                new_annos.append(new_coord)
            annos_dict[uid] = new_annos

        np.save(os.path.join(bbox_dir, '%s_bboxes.npy' % (uid)), np.array(new_annos))

def annotation_to_npy(annotations_file, scan_ids, preprocessed_dir, bbox_dir, output_path, mappings):


    transformed_annotations_file = output_path / annotations_file.name

    if Path(transformed_annotations_file).exists():
        print(f'File {transformed_annotations_file} already exists. Skipping conversion')
        return

    annos_dict = get_anno_dict(annotations_file, mappings)

    generate_label(annos_dict, scan_ids, preprocessed_dir, bbox_dir)

    

    try:
        with open(transformed_annotations_file, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"])

            for uid in annos_dict.keys():
                for annos in annos_dict[uid]:
                    writer.writerow([uid, annos[2], annos[1], annos[0], annos[3]])

    except:
        print("Unexpected error:", sys.exc_info()[0])

def annotation_exclude_to_npy(annotations_excluded_dir, scan_ids, preprocessed_dir, output_path, mappings):
    
    try:
        annos_exclude_dict = get_anno_dict(annotations_excluded_dir, mappings)
    except:
        print("Unexpected error 1:", sys.exc_info()[0])


    for uid in tqdm(scan_ids):
        if uid in annos_exclude_dict.keys():
            annos = annos_exclude_dict[uid]
            origin = np.load(preprocessed_dir + '/' + uid + '_origin.npy')
            spacing = np.load(preprocessed_dir + '/' + uid + '_spacing.npy')
            ebox = np.load(preprocessed_dir + '/' + uid + '_ebox.npy')
            new_annos_exclude = []
            for anno in annos:
                anno[[0, 1, 2]] = anno[[2, 1, 0]]
                coord = anno[:-1]
                new_coord = worldToVoxelCoord(coord, origin, spacing) * spacing - ebox
                new_coord = np.append(new_coord, anno[-1])
                new_annos_exclude.append(new_coord)
            annos_exclude_dict[uid] = new_annos_exclude

    transformed_annotations_exclude_file = output_path / annotations_excluded_dir.name

    if Path(transformed_annotations_exclude_file).exists():
        print(f'File {transformed_annotations_exclude_file} already exists. Skipping conversion')
        return

    with open(transformed_annotations_exclude_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"])

        for uid in annos_exclude_dict.keys():
            for annos in annos_exclude_dict[uid]:
                writer.writerow([uid, annos[2], annos[1], annos[0], annos[3]])


if __name__ == '__main__':

    args = parse_args()

    annotations_path = (args.transformed_annotations_dir / args.flavour)
    annotations_path.mkdir(parents=True, exist_ok=True)

    bbox_path = args.bbox_dir
    bbox_path.mkdir(parents=True, exist_ok=True)

    scan_ids = pd.read_csv(args.scan_id_file).iloc[:,0].tolist()
    with open(annotations_path / f'{args.dataset}_scans.txt' , 'w') as file:
        for scan_id in scan_ids:
            file.write(str(scan_id) + '\n')

    if args.throttle:
        scan_ids = scan_ids[:args.throttle]

    mappings = json.loads(open(args.mappings_file, 'r').read())

    annotation_to_npy(
        args.annotations_file,
        scan_ids,
        args.preprocessed_dir,
        args.bbox_dir,
        annotations_path,
        mappings
    )

    if args.annotations_excluded_file:
        annotation_exclude_to_npy(
            args.annotations_excluded_file,
            scan_ids,
            args.preprocessed_dir,
            annotations_path,
            mappings
        )
    
