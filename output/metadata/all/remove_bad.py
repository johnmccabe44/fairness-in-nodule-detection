import pandas as pd
from pathlib import Path
import os
import shutil
import sys

def main(prep_result_path):
    trn_scans = pd.read_csv('training_scans.csv')['scan_id'].tolist()
    val_scans = pd.read_csv('validation_scans.csv')['scan_id'].tolist()


    Path(prep_result_path, 'junk').mkdir(parents=True, exist_ok=True)

    for fil in os.listdir(prep_result_path):
        if fil.find('clean')>-1:
            scan_id = fil.split('_clean',1)[0]
            if not scan_id in trn_scans and not scan_id in val_scans:
                try:
                    shutil.move(
                        Path(prep_result_path, fil),
                        Path(prep_result_path,'junk',fil)
                    )

                except Exception as err:
                    print(f'Error: failed to moved {fil}, error: {err}')

                try:
                    shutil.move(
                        Path(prep_result_path, fil.replace('clean','label')),
                        Path(prep_result_path,'junk',fil.replace('clean','label'))
                    )

                except Exception as err:
                    print(f'Error: failed to moved {fil}, error: {err}')                


if __name__ == '__main__':
    prep_result_path = sys.argv[1]
    main(prep_result_path)
