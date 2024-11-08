from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

def main(prepresult_dir):

    list_label_paths = list(Path(prepresult_dir).rglob("*label.npy"))

    bboxes = {}

    try:
        for label_path in tqdm(list_label_paths):

            name = label_path.stem.replace('_label', '')

            labels = np.load(label_path)

            for label in labels:

                for i, l in enumerate(label):

                    key = f'{name}_{i}'
                    if i:
                        bboxes[key] = {
                            'scan_id': name,
                            'index': i,
                            'row': l[0],
                            'col': l[1],
                            'diameter': l[2]
                        }

        bboxes = pd.DataFrame.from_dict(bboxes, orient='index')
        bboxes.to_csv(Path(prepresult_dir, 'bboxes.csv'), index=False)
    except Exception as e:
        print(e)
        print(label_path)
        print(label)
        
if __name__ == "__main__":
    data_dir = sys.argv[1]

    main(data_dir)