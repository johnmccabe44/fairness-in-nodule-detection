import numpy as np
from pathlib import Path
import SimpleITK as sitk
import sys

def check_integrity(mhd_path: Path):

    if not mhd_path.exists():
        print(f'{mhd_path} doesnt exist.')
        return False
    
    try:
        # read in the scan
        metadata = sitk.ReadImage(mhd_path)
        image = np.array(sitk.GetArrayFromImage(metadata), dtype=np.int16)
        return True

    except Exception as err:

        print(f'{mhd_path}, error reading: {err.__str__()}')
        return False
    
def main(path_to_mhds):

    with open('mhd_checks.csv', 'w') as f: 
        for mhd_path in Path(path_to_mhds).iterdir():
            if check_integrity(mhd_path):
                f.write(f'{mhd_path},1')
            else:
                f.write(f'{mhd_path},0')


if __name__ == '__main__':

    path_to_mhds = sys.argv[1]
    main(path_to_mhds)
