import numpy as np
import pandas as pd
from pathlib import Path
import sys

def combine_exclusions(exclusions_path):
    label_paths = Path(exclusions_path).glob('*label.npy')

    exclusions = {}

    idx = 0
    for label_path in label_paths:

        labels = np.load(label_path)

        print(f'Processing {label_path.stem}, found {labels.shape[0]} exclusions', flush=True)
        for label in labels:
            exclusion = {}

            for i, l in enumerate(label):
                if i == 0:
                    exclusion['scan_id'] = label_path.stem.split('_label')[0]
                    exclusion[i] = l
                else:
                    exclusion[i] = l

            exclusions[idx] = exclusion
            idx += 1

    exclusions = (
        pd.DataFrame.from_dict(exclusions, orient='index')
        .rename(columns={0:'index', 1:'row', 2:'col', 3:'diameter'})
    )

    return exclusions

def main(exclusions_path : str):
    exclusions_path = Path(exclusions_path)
    exclusions = combine_exclusions(exclusions_path)
    exclusions.to_csv(Path(exclusions_path, 'exclusions.csv'), index=False)

if __name__ == '__main__':
    exclusions_path = sys.argv[1]
    main(exclusions_path)