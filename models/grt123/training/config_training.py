config = {
      'stage1_annos_path':[
                  './detector/labels/label_job5.csv',
                  './detector/labels/label_job4_2.csv',
                  './detector/labels/label_job4_1.csv',
                  './detector/labels/label_job0.csv',
                  './detector/labels/label_qualified.csv'
      ],
      'stage1_data_path':'/work/DataBowl3/stage1/stage1/',

      'luna_raw':'/work/DataBowl3/luna/raw/',
      'luna_segment':'/work/DataBowl3/luna/seg-lungs-LUNA16/',
      'luna_data':'/work/DataBowl3/luna/allset',
      'luna_abbr':'./detector/labels/shorter.csv',
      'luna_label':'./detector/labels/lunaqualified.csv',
      
      'datapath' : '/Users/john/Projects/SOTAEvaluationNoduleDetection/scans/lung50',
      'metadata_path' : '/Users/john/Projects/SOTAEvaluationNoduleDetection/output/metadata',

      'preprocess_result_path':'/Users/john/Projects/SOTAEvaluationNoduleDetection/models/grt123/prep_result',       
      'bbox_path':'/Users/john/Projects/SOTAEvaluationNoduleDetection/models/grt123/bbox_result',

      'use_existing' : True,
      
      'n_worker_preprocessing' : 4,
      'preprocessing_backend':'python'
      }
