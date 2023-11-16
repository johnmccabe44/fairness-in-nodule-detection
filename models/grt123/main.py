import argparse

from pathlib import Path
from preprocessing import full_prep, full_prep_summit
from config_submit import config as config_submit

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from layers import acc
from data_detector import DataBowl3Detector,collate
from data_classifier import DataBowl3Classifier

from utils import *
from split_combine import SplitComb
from test_detect import test_detect
from importlib import import_module
import pandas


def parse_arguments():
    """
    Pull arguments from command line, drives whether to pick up SUMMIT structure
    and which metadata file to get the list of scan ids from.

    NOTE: The metadata file should be a list of scan_ids e.g. summit-1234-abc_Y0_BASELINE_A
    this will be parsed into the folder structure of data_dir/studyid/scanid.mhd

    """
    parser = argparse.ArgumentParser(description='XNAT downloader and cacher')

    parser.add_argument('--datapath',
                        metavar='DATAPATH',
                        help='Location of the raw scans on disk',
                        type=str)

    parser.add_argument('--prep-result-path',
                        metavar='PREP-RESULT-PATH',
                        help='Location of where the _clean numpy files will be saved to',
                        type=str)

    parser.add_argument('--bbox-result-path',
                        metavar='BBOX-RESULT-PATH',
                        help='Loaction of where the predictions and labels will be saved to',
                        type=str)

    parser.add_argument('--n-gpu',
                        metavar='N-GPU',
                        help='Number of GPUs to use',
                        type=int,
                        default=1)

    parser.add_argument('--n-worker-preprocessing',
                        metavar='N-WORKER-PREPROCESSING',
                        help='Number of workers to preprocess data',
                        type=int,
                        default=8)

    parser.add_argument('--use-exsiting-preprocessing',
                        help='Use any existing preprocess _clean.npy files',
                        action='store_true')

    parser.add_argument('--run-prepare',
                        help='Run the preparation',
                        action='store_true')

    parser.add_argument('--run-detect',
                        help='Run the nodule detection arm',
                        action='store_true')

    parser.add_argument('--run-classify',
                        help='Run the nodule classification arm',
                        action='store_true')

    parser.add_argument('--summit',
                        help='Whether this is using SUMMIT imaging structure',
                        action='store_true')

    parser.add_argument('--scanlist-path',
                        metavar='SCANLIST-PATH',
                        help='Path to the list of scans to use',
                        type=str)

    parser.add_argument('--metadata-path',
                        metavar='METADATA-PATH',
                        help='Path to the nodule metadata',
                        type=str)    

    args = parser.parse_args()
    return args    

def test_casenet(model,testset, device):
    data_loader = DataLoader(
        testset,
        batch_size = 1,
        shuffle = False,
        num_workers = 32,
        pin_memory=True)
    #model = model.cuda()
    model.eval()
    predlist = []
    
    #     weight = torch.from_numpy(np.ones_like(y).float().cuda()
    for i,(x,coord) in enumerate(data_loader):

        #coord = Variable(coord).cuda()
        #x = Variable(x).cuda()
        coord = coord.to(device, non_blocking=True)
        x = x.to(device, non_blocking=True)

        nodulePred,casePred,_ = model(x,coord)
        predlist.append(casePred.data.cpu().numpy())
        #print([i,data_loader.dataset.split[i,1],casePred.data.cpu().numpy()])

    predlist = np.concatenate(predlist)
    return predlist  

def main(datapath, prep_result_path, bbox_result_path, n_gpu, n_worker_preprocessing, use_exsiting_preprocessing, 
         run_prepare, run_detect, run_classify, summit, scanlist_path=None, metadata_path=None):
    
    #datapath = os.path.join(config_submit['datapath'], 'mhd' if summit else 'dicom')
    #prep_result_path = os.path.join(config_submit['preprocess_result_path'], 'mhd' if summit else 'dicom')
    #bbox_result_path = os.path.join(config_submit['bbox_result_path'], 'mhd' if summit else 'dicom')
    
    #run_prepare = config_submit['run_prepare']
    #run_detect = config_submit['run_detect']
    #run_classify = config_submit['run_classify']


    if run_prepare:

        if summit:
            testsplit = full_prep_summit(data_path=datapath,
                                        prep_folder=prep_result_path,
                                        scanlist_path=scanlist_path,
                                        n_worker=n_worker_preprocessing,
                                        use_existing=use_exsiting_preprocessing,
                                        metadata_path=metadata_path)
        else:
            testsplit = full_prep(datapath,
                                prep_result_path,
                                n_worker=n_worker_preprocessing,
                                use_existing=use_exsiting_preprocessing,
                                metadata_path=metadata_path)

    else:
        testsplit = os.listdir(datapath)


    testsplit = [
        fil.split('_clean')[0]
        for fil in os.listdir(prep_result_path)
        if fil.endswith('_clean.npy')
    ]
    
    # check whether gpu is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if run_detect:

        nodmodel = import_module(config_submit['detector_model'].split('.py')[0])
        config1, nod_net, loss, get_pbb = nodmodel.get_model()
        checkpoint = torch.load(config_submit['detector_param'])
        nod_net.load_state_dict(checkpoint['state_dict'])

        #torch.cuda.set_device(0)
        #nod_net = nod_net.cuda()
        #cudnn.benchmark = True
        #nod_net = DataParallel(nod_net)

        nod_net = nod_net.to(device)
        if use_cuda:
            cudnn.benchmark = True
            nod_net = DataParallel(nod_net)

        if not os.path.exists(bbox_result_path):
            Path(bbox_result_path).mkdir(parents=True, exist_ok=True)

        #testsplit = [f.split('_clean')[0] for f in os.listdir(prep_result_path) if '_clean' in f]
 
        margin = 32
        sidelen = 144
        config1['datadir'] = prep_result_path
        split_comber = SplitComb(sidelen,config1['max_stride'],config1['stride'],margin,pad_value= config1['pad_value'])

        dataset = DataBowl3Detector(testsplit,config1,phase='test',split_comber=split_comber)
        test_loader = DataLoader(dataset,batch_size = 1,
                                shuffle = False,
                                num_workers = args.n_worker_preprocessing,
                                pin_memory=False,
                                collate_fn=collate)

        test_detect(test_loader, nod_net, get_pbb, bbox_result_path, config1, n_gpu=n_gpu)


    if run_classify:
        casemodel = import_module(config_submit['classifier_model'].split('.py')[0])
        casenet = casemodel.CaseNet(topk=5)
        config2 = casemodel.config
        checkpoint = torch.load(config_submit['classifier_param'], encoding='latin1')
        casenet.load_state_dict(checkpoint['state_dict'])

        # torch.cuda.set_device(0)
        # casenet = casenet.cuda()
        # cudnn.benchmark = True
        # casenet = DataParallel(casenet)
        casenet = casenet.to(device)
        if use_cuda:
            casenet = DataParallel(casenet)

        filename = config_submit['outputfile']
        
        config2['bboxpath'] = bbox_result_path
        config2['datadir'] = prep_result_path

        dataset = DataBowl3Classifier(testsplit, config2, phase = 'test')
        predlist = test_casenet(casenet,dataset, device).T
        df = pandas.DataFrame({'id':testsplit, 'cancer':predlist})
        df.to_csv(filename,index=False)

if __name__ == '__main__':
    args = parse_arguments()

    main(datapath=args.datapath, 
         prep_result_path=args.prep_result_path,
         bbox_result_path=args.bbox_result_path,
         n_gpu=args.n_gpu,
         n_worker_preprocessing=args.n_worker_preprocessing,
         use_exsiting_preprocessing=args.use_exsiting_preprocessing,
         run_prepare=args.run_prepare,
         run_detect=args.run_detect,
         run_classify=args.run_classify,
         summit=args.summit,
         scanlist_path=args.scanlist_path,
         metadata_path=args.metadata_path
    )
