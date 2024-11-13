import argparse
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from dataset.bbox_reader import BboxReader
from torch.utils.data import DataLoader, ConcatDataset
from config import data_config, train_config, net_config
from dataset.collate import train_collate

parser = argparse.ArgumentParser(description='PyTorch Detector')


parser.add_argument(
    '--batch-size',
    default=train_config['batch_size'],
    type=int,
    metavar='N',
    help='batch size'
)

parser.add_argument(
    '--train-set-list',
    default=['/home/jmccabe/Projects/TiCNet-main/annotations/summit/test_balanced/training_scans.csv'],
    nargs='+',
    type=str,
    help='train set paths list'
)

parser.add_argument(
    '--data-dir',
    default='/home/jmccabe/Projects/TiCNet-main/cache/ticnet/summit/preprocessed', 
    type=str, 
    metavar='OUT',
    help='path to load data'
)

parser.add_argument(
    '--num-workers',
    default=1,
    type=int,
    metavar='N',
    help='number of data loading workers'
)

def main():

    args = parser.parse_args()

    label_types = train_config['label_types']

    train_dataset_list = []
    for i in range(len(args.train_set_list)):
        set_name = args.train_set_list[i]
        label_type = label_types[i]

        assert label_type == 'bbox', 'DataLoader not support'
        dataset = BboxReader(args.data_dir, set_name, net_config, mode='train')
        train_dataset_list.append(dataset)


        train_loader = DataLoader(ConcatDataset(train_dataset_list), batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, collate_fn=train_collate, drop_last=True)
    

    for i in tqdm(range(1, 100), desc='Total', ncols=100):
        batch_size = train_config['batch_size']
        with tqdm(enumerate(train_loader), total=len(train_loader), desc='[Train %d]' % i, ncols=100) as t:
            for j, (input, truth_box, truth_label) in t:
                input = Variable(input).cuda()
                truth_box = np.array(truth_box, dtype=object)
                truth_label = np.array(truth_label, dtype=object)


if __name__ == '__main__':

    main()