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
      
      'datapath' : '../../../scans/mhd',
      'scanlist_path':'../../../output/metadata/training_scans.csv',
      'metadata_path' : '../../../output/metadata/training_metadata.csv',

      'preprocess_result_path':'../prep_result/trn',       
      'bbox_path':'../bbox_result/trn',
      
      'n_worker_preprocessing' : 1,
      'preprocessing_backend':'python'
      }
