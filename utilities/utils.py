
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
from time import strftime
import pandas as pd
from collections import namedtuple
import numpy as np
import matplotlib.patches as patches
from typing import List

IrcTuple = namedtuple('IrcTuple',['index','row','col'])

XyzTuple = namedtuple('XyzTuple',['x','y','z'])

DisplayTuple = namedtuple('DisplayTuple',['nodule_type','scan_id', 'image', 'xyz_coords', 'irc_coords', 'diameter'])

def display_images(nodule_type_images : List[DisplayTuple]):

    for idx, display_tuple in enumerate(nodule_type_images):
        image = display_tuple.image
        xyz = display_tuple.xyz_coords
        irc = display_tuple.irc_coords
        diameter = display_tuple.diameter

        fig, ax = plt.subplots(nrows=1,ncols=1)
        fig.set_size_inches(7,7)

        ax.imshow(image[irc.index,:,:])
        rect = patches.Rectangle((int(irc.col-((diameter+3)//2)), int(irc.row-((diameter+3)//2))),
                                 width=diameter+3,
                                 height=diameter+3,
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        
        ax.add_patch(rect)
        ax.set_title(display_tuple.nodule_type + '-' + display_tuple.scan_id + '-' + str(irc) + '-' + str(xyz))
        plt.show()

def xyz2irc(coord_xyz, origin, voxel_size, orientation):

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
   
def display_original_nodules(scan_path, stem):
    
    scan = SummitScan.load_scan(scan_path)

    nodules = pd.read_csv('/Users/john/Projects/SOTAEvaluationNoduleDetection/output/metadata/test_metadata.csv')
    nodules = nodules[nodules.main_participant_id==stem]

    print(nodules.shape)

    nodule_type_images = []
    for idx, row in nodules.iterrows():

        nodule_xyz = XyzTuple(x=row['nodule_x_coordinate'],
                                y=row['nodule_y_coordinate'],
                                z=row['nodule_z_coordinate'])
        
        nodule_irc = xyz2irc(coord_xyz=nodule_xyz,
                                origin=scan.origin,
                                voxel_size=scan.voxel_size,
                                orientation=scan.orientation)

        
        nodule_diameter = row['nodule_diameter_mm'] if row['nodule_diameter_mm'] else row['nodule_subsolid_major_axis_diameter']
        
        nodule_type_images.append(DisplayTuple(nodule_type='NA',
                                    scan_id=stem,
                                    image=scan.image,
                                    xyz_coords=nodule_xyz,
                                    irc_coords=nodule_irc,
                                    diameter=int(nodule_diameter //scan.voxel_size[1]) ))

    display_images(nodule_type_images)

def display_processed_nodules(stem):

    clean = np.load(f'/Users/john/Projects/SOTAEvaluationNoduleDetection/models/DSB2017-master/prep_result/mhd/{stem}_Y0_BASELINE_A_clean.npy')

    print(clean.shape)
    nodules = np.load(f'/Users/john/Projects/SOTAEvaluationNoduleDetection/models/DSB2017-master/prep_result/mhd/{stem}_Y0_BASELINE_A_label.npy')

    nodule_type_images = []
    for nodule in nodules:

        xyz = XyzTuple(x=0, y=0, z=0)
        irc = IrcTuple(index=int(nodule[0]), col=int(nodule[2]), row=int(nodule[1]))
        d = nodule[3]

        nodule_type_images.append(DisplayTuple(nodule_type='na',
                                               scan_id='stem',
                                               image=clean[0,...],
                                               xyz_coords=xyz,
                                               irc_coords=irc,
                                               diameter=d))

    display_images(nodule_type_images)
