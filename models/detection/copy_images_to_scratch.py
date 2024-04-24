import argparse
import json
from pathlib import Path
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasplit-json', type=str, help='Path to the datasplit.json file')
    parser.add_argument('--dataset', type=str, help='Name of the dataset')
    parser.add_argument('--src-data-dir', type=str, help='Source directory')
    parser.add_argument('--dest-data-dir', type=str, help='Destination directory')
    parser.add_argument('--throttle', type=int, default=-1, help='Number of images to copy')
    return parser.parse_args()

def main(datasplit_json, dataset, source_dir, destination_dir, throttle=-1):

    # Load the datasplit.json file
    with open(datasplit_json, 'r') as f:
        datasplit = json.load(f)

    # Iterate over the images in the datasplit
    for image_json in datasplit[dataset]:

        parent_folder, nifti_file = image_json['image'].split('/')
                
        # Construct the source and destination paths
        source_path = f'{source_dir}/{parent_folder}/{nifti_file}'

        Path(f'{destination_dir}/{parent_folder}').mkdir(parents=True, exist_ok=True)
        destination_path = f'{destination_dir}/{parent_folder}/{nifti_file}'
        
        # Copy the image from source to destination
        shutil.copy2(source_path, destination_path)

        if throttle > 0:
            throttle -= 1
            if throttle == 0:
                break

if __name__ == '__main__':
    args = parse_arguments()
    datasplit_json = args.datasplit_json
    dataset = args.dataset
    source_dir = args.src_data_dir
    destination_dir = args.dest_data_dir
    throttle = args.throttle

    main(datasplit_json, dataset, source_dir, destination_dir, throttle)