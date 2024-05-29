
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
from collections import namedtuple
from typing import NamedTuple
import SimpleITK as sitk

IrcTuple = namedtuple('IrcTuple',['index','row','col'])

XyzTuple = namedtuple('XyzTuple',['x','y','z'])


def xyz2irc(coord_xyz, origin, voxel_size, orientation=np.array([[1,0,0],[0,1,0],[0,0,1]])):

    origin_a = np.array(origin)
    voxel_size_a = np.array(voxel_size)
    coord_a = np.array(coord_xyz)

    cri_a = ((coord_a - origin_a) @ np.linalg.inv(orientation)) / voxel_size_a
    
    # it can only be whole numbers as irc
    cri_a = np.round(cri_a)
    return IrcTuple(index=int(cri_a[2]), row=int(cri_a[1]), col=int(cri_a[0]))

def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    # coords_xyz = (direction_a @ (idx * vxSize_a)) + origin_a
    return XyzTuple(*coords_xyz)

class SummitScan:
    """
    Author: John McCabe
    Description:  

    Attributes:
        pixel_array
        voxel_size
        origin
        orientation
    """
    def __init__(self, scan_uid, metadata, image) -> None:
        super().__init__()
        self.scan_uid = scan_uid
        self.metadata = metadata
        self.image = image

        # Pull out the salient bits of info needed
        self.origin = self.metadata.GetOrigin()
        self.voxel_size = self.metadata.GetSpacing()
        self.orientation = np.array(self.metadata.GetDirection()).reshape(3,3)

    @classmethod
    def load_scan(cls, path, type='MetaImageIO'):
        """
        Loads the scan from raw. Keeps all properties as part of the slices. 
        """

        # unique identifier can be found from file name
        scan_uid = os.path.basename(path).split('.')[0]

        # read in the scan
        if type == 'MetaImageIO':
            metadata = sitk.ReadImage(path)
            image = np.array(sitk.GetArrayFromImage(metadata), dtype=np.float32)

        return cls(scan_uid, metadata, image)
   
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

def collate_metadata(mhd_path):
    """
    Collates all of the metadata for a particular reconstruction and saves as 
    a csv file at root of the reconstruction project store on the CS cluster
    """
    metadata = {}
    for root, _, files in os.walk(mhd_path):
        for fil in files:
            if fil.endswith('.mhd'):

                md = {}
                with open(os.path.join(root, fil),'r') as f:
                    contents = f.read().split('\n')
                    
                    for attr in contents:
                        if attr:
                            key, val = attr.split(' = ')

                            if key in ['ObjectType','AnatomicalOrientation','ElementType']:
                                md[key]=val

                            if key == 'NDims':
                                md[key]=int(val)

                            if key in ['BinaryData', 'BinaryDataByteOrderMSB', 'CompressedData']:
                                md[key]=(val=='True')

                            if key == 'TransformMatrix':
                                try:
                                    md[key]=np.array(val.split(' '),float).reshape([3,3])
                                except Exception as err:
                                    print(f'Error: {err}, {fil}, transformation matrix')
                                    md[key] = pd.NA

                            if key in ['Offset', 'CenterOfRotation', 'ElementSpacing', 'DimSize']:
                                md[key]=np.array(val.split(' '), np.float64)

                    metadata[os.path.join(root, fil)] = md

    df = pd.DataFrame.from_dict(metadata, orient='index').reset_index()

    df['scan_id'] = df['index'].str.split('/').str[-1]
    df['StudyId'] = df['scan_id'].str.split('_').str[0]
    df[['x-offset', 'y-offset', 'z-offset']] = df.Offset.to_list()
    df[['x-spacing', 'y-spacing','z-spacing']] = df.ElementSpacing.to_list()
    df[['x-pixels', 'y-pixels', 'slices']] = df.DimSize.to_list()
    df.to_csv(os.path.join(mhd_path, 'metadata.csv'))

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
        
    if action == 'collate_metadata':
            mhd_path       = sys.argv[2]
            collate_metadata(mhd_path)