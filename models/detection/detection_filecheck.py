import pandas as pd
from pathlib import Path
import sys

def main(listings_local_path, listings_remote_path):

    local = pd.read_csv(listings_local_path, header=None).rename(columns={0: 'size', 1: 'id'})
    remote = pd.read_csv(listings_remote_path, header=None).rename(columns={0: 'size', 1: 'id'})

    merged = pd.merge(remote, local, on='id',  how='left', suffixes=('_local', '_remote')).assign(different_size=lambda x: x['size_local'] != x['size_remote'])

    merged.to_csv('differences.csv')

if __name__ == '__main__':

    listings_local_path = sys.argv[1]
    listings_remote_path = sys.argv[2]

    main(listings_local_path, listings_remote_path)
