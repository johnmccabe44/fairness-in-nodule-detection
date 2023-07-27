
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
from typing import NamedTuple


class Ircd(NamedTuple):
    index :     float
    row :       float
    col :       float
    diameter :  float

    def __str__(self):
        return '_'.join([
            str(int(self.index)),
            str(int(self.row)),
            str(int(self.col)),
            str(int(self.diameter))
        ])
    
class NoduleType:
    SOLID = "SOLID"
    PERIFISSURAL = "PERIFISSURAL"
    NON_SOLID = "NON_SOLID"
    CALCIFIED = "CALCIFIED"
    PART_SOLID = "PART_SOLID"
    ENDOBRONCHIAL = "ENDOBRONCHIAL"

def baseline_algorithm(row):

    nodule_type = str(row['nodule_type'])
    description = str(row['nodule_reliable_segment'])
    volume_mm3 = float(row['nodule_size_volume_cub_mm'])
    malignancy_probability = float(row['nodule_brock_score'])
    major_axis_mm = float(row['nodule_diameter_mm'])
    major_axis_core_mm = float(row['nodule_subsolid_major_axis_diameter'])

    if nodule_type == NoduleType.SOLID:

        if description.lower().startswith('u'):
            if major_axis_mm < 6:
                return 'RANDOMISATION_AT_YEAR_1'

            if major_axis_mm >=6 and major_axis_mm < 8:
                return '3_MONTH_FOLLOW_UP_SCAN'

            if major_axis_mm >= 8 and malignancy_probability < 10:
                return '3_MONTH_FOLLOW_UP_SCAN'

            if major_axis_mm >= 8 and malignancy_probability >= 10:
                return 'URGENT_REFERRAL'

            return 'ERROR-1'

        else:
        
            if volume_mm3 < 80:
                return 'RANDOMISATION_AT_YEAR_1'

            if volume_mm3 >= 80 and volume_mm3 < 300:
                return '3_MONTH_FOLLOW_UP_SCAN'

            if volume_mm3 > 300:
                
                if malignancy_probability < 10:
                    return '3_MONTH_FOLLOW_UP_SCAN'

                if malignancy_probability >= 10:
                    return 'URGENT_REFERRAL'

            return 'ERROR-2'

    if nodule_type == NoduleType.PART_SOLID:

        if major_axis_core_mm < 8:
            return '3_MONTH_FOLLOW_UP_SCAN'

        if major_axis_core_mm >= 8:
            return '3_MONTH_FOLLOW_UP_SCAN'

        return 'ERROR-3'

    if nodule_type == NoduleType.NON_SOLID:
        
        if major_axis_mm < 5:
            return 'RANDOMISATION_AT_YEAR_1'

        if major_axis_mm >= 5:
            return 'ALWAYS_SCAN_AT_YEAR_1'

        return 'ERROR'

    if nodule_type == NoduleType.CALCIFIED:
        return '3_MONTH_FOLLOW_UP_SCAN'

    if nodule_type == NoduleType.ENDOBRONCHIAL:
        return '3_MONTH_FOLLOW_UP_SCAN'

    if nodule_type == NoduleType.PERIFISSURAL:
        return 'RANDOMISATION_AT_YEAR_1'

    return 'ERROR-4'


def extract_nodules(prep_path: str, output_path: str, scan_id : str, 
                    ircd : Ircd, nodule_type : str, nodule_site : str):
    
    if os.path.exists(Path(prep_path, scan_id+'_clean.npy')):
        img = np.load(Path(prep_path, scan_id+'_clean.npy'))

        fig = plt.figure()
        ax = plt.gca()
        plt.imshow(img[0, ircd.index, :, :])
        
        rect = Rectangle((int(ircd.col-((ircd.diameter+3)//2)),
                        int(ircd.row-((ircd.diameter+3)//2))),
                        width=ircd.diameter+6,
                        height=ircd.diameter+6,
                        linewidth=1,
                        edgecolor='r',
                        facecolor='none')
        
        ax.add_patch(rect)
        ax.set_title(f'Missed Nodule: {scan_id}, Location: {ircd}, Type:{nodule_type}, Site:{nodule_site}')

        Path(output_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(output_path, '_'.join([scan_id, ircd.__str__(),'.png'])))
        plt.close('all')
        return True
    else:
        return False


def collate_missed_nodule_images(metadata_path, misses_path, prep_result_path, output_path):

    metadata = pd.read_csv(metadata_path).rename(columns={'index' : 'nodule_index'})
    misses = (
            pd.read_csv(misses_path, header=None)
            .rename(columns={
                0:'name',
                1:'idx',
                2:'col',
                3:'row',
                4:'index',
                5:'diameter',
                6:'candidate_idx'}
            ))
    
    misses = misses.merge(metadata, left_on='idx', right_index=True)

    for idx, missed_nodule in misses.iterrows():

        #try:
        
        ircd = Ircd(index=int(missed_nodule['index']),
                        row=int(missed_nodule['row']),
                        col=int(missed_nodule['col']),
                        diameter=int(missed_nodule['diameter']))

        if extract_nodules(prep_path=prep_result_path, 
                        output_path=output_path,
                        scan_id=missed_nodule['name'],
                        ircd=ircd,
                        nodule_type=missed_nodule.nodule_type,
                        nodule_site=missed_nodule.nodule_site):
        
            print(f'Finished copying over image for nodule id: {idx}')

        #except Exception as err:
        #    print(f'Error processing nodule id: {idx}, error: {err}')


if __name__ == '__main__':

    action          = sys.argv[1]

    if action == 'collate_missed_nodule_images':
        metadata_path       = sys.argv[2]
        misses_path         = sys.argv[3]
        prep_path           = sys.argv[4]
        output_path         = sys.argv[5]
    
        collate_missed_nodule_images(metadata_path=metadata_path,
                                     misses_path=misses_path,
                                     prep_result_path=prep_path,
                                     output_path=output_path)