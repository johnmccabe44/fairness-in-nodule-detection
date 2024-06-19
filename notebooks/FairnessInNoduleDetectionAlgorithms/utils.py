import json
from operator import is_
from cv2 import norm, normalize
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import warnings
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter
import matplotlib.pyplot as plt
import subprocess
from pyparsing import col
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.simplefilter('ignore')

sys.path.append('../../utilities')
sys.path.append('../../models/grt123')
sys.path.append('../../models/grt123/training')
sys.path.append('../../models/grt123/preprocessing/')

# from layers import nms,iou
from summit_utils import *
from evaluation import noduleCADEvaluation

workspace_path = Path(os.getcwd()).parent.parent

def caluclate_cpm_from_bootstrapping(file_path):
    metrics = pd.read_csv(file_path)

    fps = metrics['FPrate']
    mean_sens = metrics['Sensivity[Mean]']
    low_mean_sens = metrics['Sensivity[Lower bound]']
    high_mean_sens = metrics['Sensivity[Upper bound]']

    idxs = []
    for fps_value in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
        idxs.append(np.abs(fps - fps_value).argmin())
        
    fps = fps[idxs]
    mean_sens = mean_sens[idxs]
    low_sens = low_mean_sens[idxs]
    high_sens = high_mean_sens[idxs]

    
    df = pd.DataFrame({'fps': fps, 'mean_sens': mean_sens, 'low_sens': low_sens, 'high_sens': high_sens}).apply(lambda x: np.round(x,3))
    mean_cpm = df['mean_sens'].mean()
    low_cpm = df['low_sens'].mean()
    high_cpm = df['high_sens'].mean()
    summary_cpm = df.apply(lambda row: f'{row["mean_sens"]} ({row["low_sens"]} - {row["high_sens"]})', axis=1)

    display(summary_cpm.to_frame().T)
    display(df)
    print('Mean Sensitivity:', np.round(mean_cpm,2), 'Low Sensitivity:', np.round(low_cpm,2), 'High Sensitivity:', np.round(high_cpm,3))

    return df

def show_metrics(file_path):
    metrics = pd.read_csv(file_path, skiprows=6, sep=':').rename(columns={0:'Metric',1:'Value'}).round(3)

    display(metrics)

def protected_group_analysis(protected_group, scan_metadata, annotations, exclusions, predictions, output_path):

    output_path = Path(output_path)

    analysis_dict = {}
    summary_dict = {}

    for cat in scan_metadata[protected_group].unique():
        if cat is np.nan:
            continue

        if cat is None:
            continue

            
        print('='*50)
        print(f'Protected Group: {cat}')
        print('='*50)

        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            temp_scans = scan_metadata[scan_metadata[protected_group] == cat]['name']
            temp_annotations = annotations[annotations['name'].isin(temp_scans.values)]
            temp_exclusions = exclusions[exclusions['name'].isin(temp_scans.values)]
            temp_predictions = predictions[predictions['name'].isin(temp_scans.values)]

            temp_scans.to_csv(temp_dir / 'scans.csv', index=False)
            temp_annotations.to_csv(temp_dir / 'annotations.csv', index=False)
            temp_exclusions.to_csv(temp_dir / 'exclusions.csv', index=False)
            temp_predictions.to_csv(temp_dir / 'predictions.csv', index=False)


            order = [
                nodule_type 
                for nodule_type in ['SOLID','PERIFISSURAL','PART_SOLID','NON_SOLID','CALCIFIED'] 
                if nodule_type in temp_annotations.nodule_type.unique()
            ]
            
            display(temp_annotations.nodule_type.value_counts(normalize=True).to_frame().T[order].apply(lambda x: round(x*100)))

            result = noduleCADEvaluation(
                annotations_filename=temp_dir / 'annotations.csv',
                annotations_excluded_filename=temp_dir / 'exclusions.csv',
                seriesuids_filename=temp_dir / 'scans.csv',
                results_filename=temp_dir / 'predictions.csv',
                filter=f'-{cat}',
                outputDir=output_path / cat,
            )

            summary_dict[cat] = caluclate_cpm_from_bootstrapping(output_path / cat / 'froc_predictions_bootstrapping.csv').set_index('fps')
            show_metrics(output_path / cat / 'CADAnalysis.txt')

            analysis_dict[cat] = (
                pd.read_csv(output_path / cat / 'froc_predictions_bootstrapping.csv')
                .rename(columns={
                    'FPrate': 'FPRate',
                    'Sensivity[Mean]': 'Sensitivity',
                    'Sensivity[Lower bound]': 'LowSensitivity',
                    'Sensivity[Upper bound]': 'HighSensitivity'
                })
            )

    fig1 = plt.figure()
    ax = plt.gca()
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    for idx, cat in enumerate(analysis_dict.keys()):

        metrics = analysis_dict[cat]        
        plt.plot(metrics['FPRate'], metrics['Sensitivity'], ls='--', color=colors[idx],label=cat)
        plt.plot(metrics['FPRate'], metrics['LowSensitivity'], ls=':', color=colors[idx], alpha=0.05)
        plt.plot(metrics['FPRate'], metrics['HighSensitivity'], ls=':', color=colors[idx], alpha=0.05)
        ax.fill_between(metrics['FPRate'], metrics['LowSensitivity'], metrics['HighSensitivity'], alpha=0.05)

    xmin = 0.125
    xmax = 8
    plt.xlim(xmin, xmax)
    plt.ylim(0, 1)
    plt.xlabel('Average number of false positives per scan')
    plt.ylabel('Sensitivity')
    #plt.legend(loc='lower right')
    plt.title(f'FROC performance - {protected_group}')
    plt.xscale('log', base=2)
    ax.xaxis.set_major_formatter(FixedFormatter([0.125,0.25,0.5,1,2,4,8]))

    # set your ticks manually
    ax.xaxis.set_ticks([0.125,0.25,0.5,1,2,4,8])
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    plt.legend(loc='lower right')
    plt.grid(visible=True, which='both')
    plt.tight_layout()

    return summary_dict

def set_ethnic_group_is_actionable(row):
    
    if row['lung_health_check_demographics_race_ethnicgroup'] == 'White' and row['radiology_report_management_plan_final'] in ['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1']:        
        return 'White'
    
    elif row['lung_health_check_demographics_race_ethnicgroup'] == 'Black' and row['radiology_report_management_plan_final'] in ['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1']:        
        return 'Black'

    elif row['lung_health_check_demographics_race_ethnicgroup'] == 'Asian or Asian British' and row['radiology_report_management_plan_final'] in ['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1']:    
        return 'Asian or Asian British'   
    
    else:
        return None

def set_ethnic_group_is_not_actionable(row):
    
    if row['lung_health_check_demographics_race_ethnicgroup'] == 'White' and row['radiology_report_management_plan_final'] == 'RANDOMISATION_AT_YEAR_1':        
        return 'White'
    
    elif row['lung_health_check_demographics_race_ethnicgroup'] == 'Black' and row['radiology_report_management_plan_final'] == 'RANDOMISATION_AT_YEAR_1':        
        return 'Black'

    elif row['lung_health_check_demographics_race_ethnicgroup'] == 'Asian or Asian British' and row['radiology_report_management_plan_final'] == 'RANDOMISATION_AT_YEAR_1':    
        return 'Asian or Asian British'    

    else:
        return None

def set_gender_is_actionable(row):

    if row['participant_details_gender'] == 'MALE' and row['radiology_report_management_plan_final'] in ['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1']:        
        return 'Male'
    
    elif row['participant_details_gender'] == 'FEMALE' and row['radiology_report_management_plan_final'] in ['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1']:        
        return 'Female'
    
    else:
        return None

def set_gender_is_not_actionable(row):

    if row['participant_details_gender'] == 'MALE' and row['radiology_report_management_plan_final'] == 'RANDOMISATION_AT_YEAR_1':        
        return 'Male'
    
    elif row['participant_details_gender'] == 'FEMALE' and row['radiology_report_management_plan_final'] == 'RANDOMISATION_AT_YEAR_1':        
        return 'Female'
    
    else:
        return None
    
def set_imdrank_is_actionable(row):
    if row['IMDRank_tertile'] == 'Low' and row['radiology_report_management_plan_final'] in ['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1']:        
        return 'Low_IMDRank_Actionable'
    
    elif row['IMDRank_tertile'] == 'Medium' and row['radiology_report_management_plan_final'] in ['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1']:        
        return 'Medium_IMDRank_Actionable'

    elif row['IMDRank_tertile'] == 'High' and row['radiology_report_management_plan_final'] in ['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1']:        
        return 'High_IMDRank_Actionable'
    
    else:
        return None
    
def set_smoking_pack_years_is_actionable(row):
    if row['smoking_pack_years_cats'] == 'Low (15-35)' and row['radiology_report_management_plan_final'] in ['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1']:        
        return 'Low_Pack_Years_Actionable'

    elif row['smoking_pack_years_cats'] == 'Medium (35-50)' and row['radiology_report_management_plan_final'] in ['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1']:        
        return 'Medium_Pack_Years_Actionable'
    
    elif row['smoking_pack_years_cats'] == 'High (50+)' and row['radiology_report_management_plan_final'] in ['3_MONTH_FOLLOW_UP_SCAN','URGENT_REFERRAL', 'ALWAYS_SCAN_AT_YEAR_1']:        
        return 'High_Pack_Years_Actionable'
                        
    else:
        return None

def get_thresholds(analysis_data):
    """
    Get the thresholds at different FPPs

    Args:
    analysis_data: tuple, tuple of fps, sens, threshold

    Returns:
    thresh_values: list of float, list of thresholds at different FPPs

    """
    fps = analysis_data[0]
    sens = analysis_data[1]
    threshold = analysis_data[2]

    idxs = []
    fps_values = []
    sens_values = []
    thresh_values = []
    for fpps in [0.125, 0.25, 0.5, 1, 2, 4, 8]:

        idx = np.abs(fps - fpps).argmin()
        idxs.append(idx)
        fps_values.append(fps[idx])
        sens_values.append(sens[idx])
        thresh_values.append(threshold[idx])

    return thresh_values

def miss_anaysis_at_fpps(model, scans_metadata, scans_path, annotations_path, exclusions_path, predictions_path, thresholds, show_froc=False):
    """
    Get the missed annotations at different FPPs

    Args:
    scans_path: str, path to the scans csv
    annotations_path: str, path to the annotations csv
    exclusions_path: str, path to the exclusions csv
    predictions_path: str, path to the predictions csv
    thresholds: list of float, list of thresholds to evaluate

    Returns:
    missed_metadata: list of pd.DataFrame, list of missed annotations at each threshold

    """

    missed_metadata = []

    for idx, threshold in enumerate(thresholds):
        predictions = pd.read_csv(predictions_path)
        predictions_at_operating_point = predictions[predictions.threshold > threshold]

        with TemporaryDirectory() as temp_dir:
            predictions_at_operating_point.to_csv(f'{temp_dir}/predictions.csv', index=False)
            missed_annotations = noduleCADEvaluation(
                annotations_filename=annotations_path,
                annotations_excluded_filename=exclusions_path,
                seriesuids_filename=scans_path,
                results_filename=f'{temp_dir}/predictions.csv',
                filter='Missed Annotations',
                outputDir=f'{temp_dir}/results',
                perform_bootstrapping=False,
                show_froc=show_froc
            )

            misses = (
                pd.read_csv(
                f'{temp_dir}/results/nodulesWithoutCandidate_predictions.txt',
                header=None
                )
                .rename(columns={0:'name',1: 'idx', 2:'col',3:'row',4:'index',5:'diameter',6:'threshold'})
                .assign(miss=True)
            )
            annotations = pd.read_csv(annotations_path)

            merge_keys = ['name','row','col','index','diameter']

            df = pd.merge(misses, annotations, on=merge_keys, how='right')
            df['miss'] = df['miss'].fillna(False)
            df = df.merge(
                scans_metadata[['name','is_actionable','smoking_pack_years','IMDRank_tertile','IMDRank_quintile']],
                on='name',
                how='left'
            )

            df.to_csv(f'{workspace_path}/notebooks/FairnessInNoduleDetectionAlgorithms/hits_and_misses/{model}/hits_and_misses_{idx}', index=False)

            print(f'Missed Annotations at {threshold} FPPs:', sum(df.miss))
            missed_metadata.append(df)

    return missed_metadata

def get_protected_group_by_nodule_type(missed_metadata):

    hit_and_miss_dict = {}
    for idx, miss_metadata in enumerate(missed_metadata):

        hit_and_miss_dict[idx] = {'all':None, 'actionable':None}
        hit_and_miss_dict[idx]['all'] = {'gender':None,'ethnic_group':None}
        hit_and_miss_dict[idx]['actionable'] = {'gender':None,'ethnic_group':None}

        x_gender = (
            pd.crosstab(
                [
                    miss_metadata['gender']
                ],
                [
                    miss_metadata['miss'],                    
                    miss_metadata['nodule_type']
                ],
                margins=True
                )
            )
        
        hit_and_miss_dict[idx]['all']['gender'] = x_gender.to_dict()
        
        x_ethnics = (
            pd.crosstab(
                [
                    miss_metadata['ethnic_group']
                ],
                [
                    miss_metadata['miss'],
                    miss_metadata['nodule_type']
                ],
                margins=True
                )
            )
        
        hit_and_miss_dict[idx]['all']['ethnic_group'] = x_ethnics.to_dict()

        miss_metadata = miss_metadata[miss_metadata['is_actionable'] == 'Actonable']

        x_gender = (
            pd.crosstab(
                [
                    miss_metadata['gender']
                ],
                [
                    miss_metadata['miss'],                    
                    miss_metadata['nodule_type']
                ],
                margins=True
                )
            )
        
        hit_and_miss_dict[idx]['actionable']['gender'] = x_gender.to_dict()
        
        x_ethnics = (
            pd.crosstab(
                [
                    miss_metadata['ethnic_group']
                ],
                [
                    miss_metadata['miss'],
                    miss_metadata['nodule_type']
                ],
                margins=True
                )
            )
        
        hit_and_miss_dict[idx]['actionable']['ethnic_group'] = x_ethnics.to_dict()

    return hit_and_miss_dict
    
# False Positive Analysis
def is_distance_match(annotation, prediction):
    """
    Check if the annotation and prediction are a match

    Args:
    annotation: pd.Series, annotation
    prediction: pd.Series, prediction

    Returns:
    bool, True if the annotation and prediction are a match
    """
    x = float(annotation['row'])
    y = float(annotation['col'])
    z = float(annotation['index'])
    d = float(annotation['diameter'])
    radiusSquared = pow((d / 2.0), 2.0)
    dx = float(prediction['row'])
    dy = float(prediction['col'])
    dz = float(prediction['index'])
    distanceSquared = pow((dx - x), 2.0) + pow((dy - y), 2.0) + pow((dz - z), 2.0)
    return distanceSquared <= radiusSquared

def iou_calc(annotation, prediction):
    # Calculate the coordinates of the intersection rectangle
    x1 = max(annotation['row'], prediction['row'])
    y1 = max(annotation['col'], prediction['col'])
    z1 = max(annotation['index'], prediction['index'])

    x2 = min(annotation['row'] + annotation['diameter'], prediction['row'] + prediction['diameter'])
    y2 = min(annotation['col'] + annotation['diameter'], prediction['col'] + prediction['diameter'])
    z2 = min(annotation['index'] + annotation['diameter'], prediction['index'] + prediction['diameter'])

    # Calculate the area of intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1) * max(0, z2 - z1 + 1)

    # Calculate the area of annotation and prediction rectangles
    annotation_area = annotation['diameter'] * annotation['diameter'] * annotation['diameter']
    prediction_area = prediction['diameter'] * prediction['diameter'] * prediction['diameter']
    
    # Calculate the union area
    union_area = annotation_area + prediction_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area

    return iou
    
def false_positive_analysis(thresholds, predictions, annotations):
    """
    Get the false positives at different thresholds

    Args:
    thresholds: list of float, list of thresholds to evaluate
    predictions: pd.DataFrame, dataframe of predictions
    annotations: pd.DataFrame, dataframe of annotations

    Returns:
    false_postives_fpps: dict, dict of false positives at each threshold
    """

    false_postives_fpps = {}
    operating_points = ['0.125', '0.25', '0.5', '1', '2', '4', '8']
    for tdx, threshold in enumerate(thresholds):

        threshold_predictions = predictions[predictions['threshold'] > threshold]

        print('predictions:', len(threshold_predictions), 'at threshold:', threshold)

        distance_false_positives = {}
        iou_false_positives = {}

        for name in predictions['name'].unique():
            
            temp_annotations = annotations[annotations['name'] == name]
            temp_predictions = threshold_predictions[threshold_predictions['name'] == name]

            for pdx, prediction in temp_predictions.iterrows():
                distance_false_positives[pdx] = True

                for adx, annotation in temp_annotations.iterrows():
                    iou_false_positives[pdx] = iou_calc(annotation, prediction)
                    if is_distance_match(annotation, prediction):                        
                        distance_false_positives[pdx] = False
                        break





        print('distance false positives:', sum(pd.Series(distance_false_positives)), 'at threshold:', threshold)

        threshold_predictions['distance_false_positive'] = distance_false_positives
        threshold_predictions['iou_false_positive'] = iou_false_positives

        false_postives_fpps[operating_points[tdx]] = threshold_predictions            
                
    return false_postives_fpps

def copy_numpy_from_cluster(scan_id):
    """
    Copy the scan from the cluster

    Args:
    scan_id: str, scan id

    Returns:
    None
    """
    study_id = scan_id.split('_')[0]

    scan_path = f'/cluster/project2/SUMMIT/cache/sota/grt123/prep_result/summit/{scan_id}*'
    if not os.path.exists(f'{workspace_path}/models/grt123/prep_result/summit/{scan_id}_clean.npy'):
        command = [
            "scp",
            f"jmccabe@little:{scan_path}",
            f"{workspace_path}/models/grt123/prep_result/summit/."
        ]
        result = subprocess.run(command)
        print(result.stdout)
    else:
        print(f'{scan_id} already exists')

def show_numpy_candidate_location(scan_id, row, col, index, diameter, distance_false_positive, iou_false_positive):
    """
    Show the candidate location in the scan

    Args:
    scan_id: str, scan id
    row: int, row
    col: int, col
    index: int, index
    diameter: float, diameter

    Returns:
    None
    """
    scan_name = scan_id + '_clean.npy'
    scan_path = f'{workspace_path}/models/grt123/prep_result/summit/{scan_name}'
    scan = np.load(scan_path)
    scan = scan[0]


    label_name = scan_id + '_label.npy'
    label_path = f'{workspace_path}/models/grt123/prep_result/summit/{label_name}'
    label = np.load(label_path)
    print(label)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(scan[int(index)], cmap='gray')

    if distance_false_positive:
        color = 'b'
    else:
        color = 'r'


    print(f'{scan_id} - {index} {row} {col} {diameter}\nFalse Positive: {distance_false_positive}\nIoU False Positive: {iou_false_positive}')
    circle = plt.Circle((int(col), int(row)), diameter * .75, color=color, fill=False)
    ax.add_artist(circle)
    plt.title(f'{scan_id} - {index} {row} {col} {diameter}\nFalse Positive: {distance_false_positive}\nIoU False Positive: {iou_false_positive}')
    plt.show()

def combine_predictions(predictions_json_path, dataset_name, use_nms=True):
    """
    Get predictions from a json file

    Args:
    predictions_json_path: str, path to the json file

    Returns:
    images: list of str, list of image names
    predictions: pd.DataFrame, dataframe of predictions
    
    """
    predictions_json_path = Path(predictions_json_path)

    predictions_list = []

    for prediction_json_path in predictions_json_path.glob('*json'):
        
        print(prediction_json_path)

        with open(prediction_json_path,'r') as f:
            predictions_json = json.load(f)

        idx = 0
        for image_cnt, image in enumerate(predictions_json[dataset_name]):
            name = image['image'].split('/')[-1][:-7]
            
            image_predictions_dict = {}
            for box, score in zip(image['box'], image['score']):
                prediction = {}
                prediction['threshold'] = score
                prediction['index'] = box[2]
                prediction['col'] = box[1]
                prediction['row'] = box[0]
                prediction['diameter'] = np.max(box[3:])
                prediction['name'] = name

                image_predictions_dict[idx] = prediction
                idx+=1
            
            image_predictions = pd.DataFrame.from_dict(image_predictions_dict, orient='index')
            predictions_list.append(image_predictions)

    predictions = pd.concat(predictions_list, ignore_index=True)
    predictions.to_csv(predictions_json_path / 'predictions.csv', index=False)

    return predictions_json_path / 'predictions.csv'

def show_mhd_candidate_location(scan_id, scan, row, col, index, diameter, distance_false_positive, iou_false_positive):
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(scan[int(index)], cmap='gray')

    if distance_false_positive:
        color = 'b'
    else:
        color = 'r'


    print(f'{scan_id} - {index} {row} {col} {diameter}\nFalse Positive: {distance_false_positive}\nIoU False Positive: {iou_false_positive}')
    circle = plt.Circle((int(col), int(row)), diameter * .5, color=color, fill=False)
    ax.add_artist(circle)
    plt.title(f'{scan_id} - {index} {row} {col} {diameter}\nFalse Positive: {distance_false_positive}\nIoU False Positive: {iou_false_positive}')
    plt.show()    


def copy_scan_from_cluster(scan_id):
    study_id = scan_id.split('_')[0]
   # now copy the src file
    if not os.path.exists(f"{workspace_path}/data/summit/scans/{study_id}/{scan_id}.mhd"):
        os.makedirs(f"{workspace_path}/data/summit/scans/{study_id}", exist_ok=True)

        command = [
            "scp",
            f"jmccabe@little:/cluster/project2/SummitLung50/{study_id}/{scan_id}.*",
            f"{workspace_path}/data/summit/scans/{study_id}/."
        ]
        result = subprocess.run(command)
        print(result.stdout)
    else:
        print(f'{scan_id} already exists')



def display_plots_with_error_bars(model, flavour, protected_group, sensitivity_data, order=None):

    if order is None:
        order = sensitivity_data.keys()
    else:
        order = [cat for cat in order if cat in sensitivity_data.keys()]

    cat_increments = {}
    increment = -0.1 * (len(order) - 1) / 2
    for idx, cat in enumerate(order):
        cat_increments[cat] = increment + 0.1 * idx


    # Example summary statistics (mean, low, high) for Male and Female at each FPPI level
    fppi_levels = [0.125, 0.25, 0.5, 1, 2, 4, 8]

    means = {}
    errors = {}

    for cat in order:
        means[cat] = np.array(sensitivity_data[cat]['mean_sens'])
        errors[cat] = np.array([
            (
                sensitivity_data[cat].loc[fppi, 'mean_sens'] - sensitivity_data[cat].loc[fppi, 'low_sens'],
                sensitivity_data[cat].loc[fppi, 'high_sens'] - sensitivity_data[cat].loc[fppi, 'mean_sens']
            )
            for fppi in fppi_levels
        ]).T

    # Plotting side-by-side scatter plots with error bars
    plt.figure(figsize=(12, 6))

    bar_width = 0.1
    index = np.arange(len(fppi_levels))

    for idx, cat in enumerate(means.keys()):
        plt.errorbar(index + cat_increments[cat], means[cat], yerr=errors[cat], fmt='o',label=cat, capsize=5)

    plt.xlabel('False Positives Per Scan')
    plt.ylabel('Sensitivity')
    # plt.title(f'Sensitivity Comparison Between {protected_group}')
    plt.xticks(index, fppi_levels)
    plt.legend()
    plt.ylim(0.0, 1.0)  # Adjust ylim based on your data range

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'/Users/john/Projects/SOTAEvaluationNoduleDetection/miccai_fair_ai/{model}_{flavour}_{protected_group}.png')
    plt.show() 

