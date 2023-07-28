config = {
    'datapath':'/Users/john/Projects/SOTAEvaluationNoduleDetection/scans',
    'preprocess_result_path':'./prep_result',
    'bbox_result_path' : './bbox_result',
    'outputfile':'prediction.csv',
    'detector_model':'net_detector',
    'detector_param':'./model/detector.ckpt',
    'classifier_model':'net_classifier',
    'classifier_param':'./model/classifier.ckpt',
    'n_gpu':1,
    'n_worker_preprocessing':None,
    'use_exsiting_preprocessing':False,
    'run_prepare': True,
    'run_detect': False,
    'run_classify': False
}
