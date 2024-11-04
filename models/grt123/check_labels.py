from pathlib import Path
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm

def main(data_dir):

    combined_results = {}
    for label_path in tqdm(Path(data_dir).rglob('*_label.npy')):
        label = np.load(label_path)
        if label.shape[0] != 1:
            for i, l in enumerate(label):
                combined_results[label_path.stem.replace('_label','') + '_' + str(i)] = l

    combined_df = pd.DataFrame.from_dict(combined_results, orient='index')
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'index': 'id'}, inplace=True)
    return combined_df


if __name__ == '__main__':
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/Users/john/Projects/SOTAEvaluationNoduleDetection/models/grt123/prep_result/summit')
    df = main(data_dir)
    df.to_csv('combined_labels.csv', index=False)
