from pathlib import Path
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

def main(data_dir):

    shapes = {}

    npy_paths = list(Path(data_dir).rglob("summit*Y0_BASELINE_A.npy"))

    for npy_path in tqdm(npy_paths):
        npy = np.load(npy_path)
        shapes[npy_path.stem] = npy.shape

    pd.DataFrame.from_dict(shapes, orient='index').to_csv("ticnet_shapes.csv")

if __name__ == "__main__":
    data_dir = sys.argv[1]
    main(data_dir)
