import logging
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import sys
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    filename='preprocessing.log',
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_size(path):
    return np.load(path).shape

def main(prepresult_path):

    list_clean_files = list(Path(prepresult_path).rglob('*clean.npy'))

    with Pool(8) as p:
        sizes = list(tqdm(p.imap(get_size, list_clean_files), total=len(list_clean_files)))


    for filename, size in zip(list_clean_files, sizes):
        logging.info(f'{filename.stem}: {size}')

if __name__ == '__main__':
    prepresult_path = sys.argv[1]
    main(prepresult_path)