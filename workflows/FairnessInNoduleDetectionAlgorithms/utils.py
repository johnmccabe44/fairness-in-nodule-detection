import numpy as np
import pandas as pd
from cv2 import threshold
import json
from pathlib import Path

def convert_to_csv_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    predictions = []
    for scan in data['test']:

        name = Path(scan['image']).stem.replace('.nii', '')

        for label, box, score in zip(scan['label'], scan['box'], scan['score']):
            predictions.append({
                'name' : name,
                'row' : box[0],
                'col' : box[1],
                'index' : box[2],
                'diameter' : max(box[3], box[4], box[5]),
                'threshold' : score
            })

    return pd.DataFrame(predictions)
     
def is_actionable(row):
    
    if row['Nod_type'] == 'SN' and row['nodule_diameter_mm'] >= 6:
        return True
    
    elif row['Nod_type'] == 'PSN' and row['nodule_diameter_mm'] >= 8:
        return True
    
    elif row['Nod_type'] == 'pGGN' and row['nodule_diameter_mm'] >= 10:
        return True
    
    else:
        return False

def load_data(workspace_path, model, dataset, flavour, actionable):

    if dataset == 'summit':

        # Load the data
        if model == 'grt123':
            annotations = pd.read_csv(
                f'{workspace_path}/models/grt123/bbox_result/trained_summit/summit/{flavour}/{flavour}_metadata.csv',
                usecols=[
                    'name',
                    'col',
                    'row',
                    'index',
                    'diameter',
                    'management_plan',
                    'gender',
                    'ethnic_group',
                    'nodule_lesion_id',
                    'nodule_type'
                ]
            )

            results = (
                pd.read_csv(
                    f'{workspace_path}/models/grt123/bbox_result/trained_summit/summit/{flavour}/{flavour}_predictions.csv',
                    usecols=['name', 'col', 'row', 'index', 'diameter', 'threshold']
                )
                .rename(columns={'threshold': 'threshold_original'})
                .assign(threshold=lambda x: 1 / (1 + np.exp(-x['threshold_original'])))
            )
            
        elif model == 'detection':
            annotations = pd.read_csv(
                f'{workspace_path}/models/detection/result/trained_summit/summit/{flavour}/annotations.csv',
                usecols=[
                    'name',
                    'col',
                    'row',
                    'index',
                    'diameter',
                    'management_plan',
                    'gender',
                    'ethnic_group',
                    'nodule_lesion_id',
                    'nodule_type'
                ]
            )
            
            results = pd.read_csv(
                f'{workspace_path}/models/detection/result/trained_summit/summit/{flavour}/predictions.csv',
                usecols=['name', 'col', 'row', 'index', 'diameter', 'threshold']
            )

        elif model == 'ticnet':

            _annotations = (
                pd.read_csv(f'{workspace_path}/../TiCNet-main/annotations/summit/{flavour}/test_metadata.csv')
                .rename(columns={
                    'seriesuid' : 'name',
                    'coordX' : 'row',
                    'coordY' : 'col',
                    'coordZ' : 'index',
                    'diameter_mm' : 'diameter',
                    'probability' : 'threshold'
                })
                .assign(name_counter=lambda df: df.groupby('name').cumcount() + 1)
            )

            _metadata = (
                pd.read_csv(f'{workspace_path}/metadata/summit/{flavour}/test_metadata.csv')
                .assign(name=lambda df: df.participant_id + '_Y0_BASELINE_A')
                .assign(name_counter=lambda df: df.groupby('name').cumcount() + 1)
            )

            annotations = (
                pd.merge(_metadata, _annotations, on=['name', 'name_counter'], how='outer')
                .filter([
                    'name',
                    'col',
                    'row',
                    'index',
                    'diameter',
                    'management_plan',
                    'gender',
                    'ethnic_group',
                    'nodule_lesion_id',
                    'nodule_type'
                ])
            )

            assert annotations.shape[0] == _metadata.shape[0], 'Mismatch in number of annotations'

            results = (
                pd.read_csv(f'{workspace_path}/../TiCNet-main/results/summit/{flavour}/res/110/FROC/submission_rpn.csv')
                .rename(
                    columns={
                        'seriesuid' : 'name',
                        'coordX' : 'row',
                        'coordY' : 'col',
                        'coordZ' : 'index',
                        'diameter_mm' : 'diameter',
                        'probability' : 'threshold'
                    }
                )
            )

        scan_metadata = (
            pd.read_csv(
                f'{workspace_path}/metadata/summit/{flavour}/test_scans_metadata.csv',
                usecols=['Y0_PARTICIPANT_DETAILS_main_participant_id', 'participant_details_gender','lung_health_check_demographics_race_ethnicgroup']
            )
            .rename(
                columns={
                    'Y0_PARTICIPANT_DETAILS_main_participant_id': 'StudyId', 
                    'participant_details_gender' : 'gender', 
                    'lung_health_check_demographics_race_ethnicgroup' : 'ethnic_group'
                }
            )
            .assign(Name=lambda x: x['StudyId'] + '_Y0_BASELINE_A')
        )

        # Reduce the annotations to only actionable cases
        if actionable:
            annotations = annotations[annotations['management_plan'].isin(['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1'])]
            annotations_excluded = annotations[annotations['management_plan']=='RANDOMISATION_AT_YEAR_1']
        else:
            annotations = annotations
            annotations_excluded = annotations.drop(annotations.index)

    elif dataset == 'lsut':

        lsut_nodule_type_mapping = {
            'SN' : 'SOLID',
            'PSN' : 'PART_SOLID',
            'pGGN' : 'NON_SOLID'
        }

        if model == 'grt123':
            annotations = (
                pd.read_csv(
                    f'{workspace_path}/models/grt123/bbox_result/trained_summit/lsut/{flavour}/lsut_{flavour}_metadata.csv'
                )
                .assign(nodule_type=lambda df: df['Nod_type'].map(lsut_nodule_type_mapping))
                .assign(actionable=lambda df: df.apply(lambda row: is_actionable(row), axis=1))
                .filter(['name', 'col', 'row', 'index', 'diameter', 'nodule_diameter_mm', 'nodule_type', 'actionable'])
            )

            results = (
                pd.read_csv(
                    f'{workspace_path}/models/grt123/bbox_result/trained_summit/lsut/{flavour}/lsut_{flavour}_predictions.csv',
                    usecols=['name', 'col', 'row', 'index', 'diameter', 'threshold']
                )
                .rename(columns={'threshold': 'threshold_original'})
                .assign(threshold=lambda x: 1 / (1 + np.exp(-x['threshold_original'])))
            )

        elif model == 'detection':
            annotations = (
                pd.read_csv(f'{workspace_path}/metadata/lsut/tranche1_metadata.csv')
                .rename(columns={
                    'scan_id' : 'name',
                    'nodule_x_coordinate' : 'row',
                    'nodule_y_coordinate' : 'col',
                    'nodule_z_coordinate' : 'index'                                          
                })
                .assign(diameter=lambda df: df['nodule_diameter_mm'])
                .assign(nodule_type=lambda df: df['Nod_type'].map(lsut_nodule_type_mapping))
                .assign(actionable=lambda df: df.apply(lambda row: is_actionable(row), axis=1))
                .filter([
                    'name',
                    'row',
                    'col',
                    'index',
                    'diameter',
                    'nodule_type',
                    'nodule_diameter_mm',
                    'actionable'
                ])
            )

            results = convert_to_csv_from_json(f'{workspace_path}/models/detection/result/trained_summit/{dataset}/result_{flavour}.json')

        elif model == 'ticnet':
            _annotations = (
                pd.read_csv(f'{workspace_path}/models/ticnet/annotations/lsut/{flavour}/tranche1_metadata.csv')
                .rename(columns={
                    'seriesuid' : 'name',
                    'coordX' : 'row',
                    'coordY' : 'col',
                    'coordZ' : 'index',
                    'diameter_mm' : 'diameter'
                })
                .assign(threshold=-10000000)
                .assign(name_counter=lambda df: df.groupby('name').cumcount() + 1)
            )

            _metadata = (
                pd.read_csv(f'{workspace_path}/metadata/lsut/tranche1_metadata.csv')
                .rename(columns={'scan_id' : 'name'})
                .assign(nodule_type=lambda df: df['Nod_type'].map(lsut_nodule_type_mapping))
                .assign(actionable=lambda df: df.apply(lambda row: is_actionable(row), axis=1))
                .assign(name_counter=lambda df: df.groupby('name').cumcount() + 1)
            )

            annotations = (
                pd.merge(_metadata, _annotations, on=['name', 'name_counter'], how='outer')
                .filter([
                    'name',
                    'col',
                    'row',
                    'index',
                    'diameter',
                    'nodule_diameter_mm',
                    'nodule_type',
                    'actionable'
                ])
            )

            assert annotations.shape[0] == _metadata.shape[0], 'Mismatch in number of annotations'

            results = (
                pd.read_csv(f'{workspace_path}/models/ticnet/results/trained_summit/lsut/{flavour}/res/120/FROC/submission_rpn.csv')
                .rename(
                    columns={
                        'seriesuid' : 'name',
                        'coordX' : 'row',
                        'coordY' : 'col',
                        'coordZ' : 'index',
                        'diameter_mm' : 'diameter',
                        'probability' : 'threshold'
                    }
                )
            )

        lsut_ethnicity_mapping = {
            'White' : 'White',
            'Black/ African/ Caribbean/ Black British' : 'Other',
            'Other ethnic group' : 'Other',
            'Prefers not to say' : 'Other',
            'Asian or Asian British' : 'Other',
            'Mixed or multiple ethnic groups' : 'Other'

        }

        scan_metadata = (
            pd.read_csv(
                f'{workspace_path}/metadata/lsut/tranche1_scan_metadata.csv',
                usecols=['ScananonID','clinic_gender','clinic_ethnicity']
            )
            .rename(
                columns={
                    'ScananonID': 'StudyId',
                    'clinic_gender' : 'gender',
                }
            )
            .assign(ethnic_group=lambda df: df['clinic_ethnicity'].map(lsut_ethnicity_mapping))
            .assign(Name=lambda x: x['StudyId'])
        )

        if actionable:
            annotations = annotations[annotations['actionable']]
            annotations_excluded = annotations[~annotations['actionable']]
        else:
            annotations = annotations
            annotations_excluded = annotations.drop(annotations.index)

    return annotations, results, scan_metadata, annotations_excluded