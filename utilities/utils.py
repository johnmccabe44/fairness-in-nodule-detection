
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
from time import strftime
import pandas as pd
from collections import namedtuple
import numpy as np
import matplotlib.patches as patches
from pathlib import Path
from typing import List
import netrc
from zipfile import ZipFile
import xnat
import sys

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

def save_to_disk(downloaded_file, output_path, scan_id, metaio):
    # unzip to temp directory
    with ZipFile(downloaded_file) as zip_file:
        zip_file.extractall(output_path)

        if metaio:
            # pull out the folder where the dcms live
            dcm_dir = os.path.dirname([dcm_file for dcm_file in Path(output_path).rglob('*.dcm')][0])

            # read in the dicom series
            reader = sitk.ImageSeriesReader()
            file_names = reader.GetGDCMSeriesFileNames(dcm_dir)
            reader.SetFileNames(file_names)
            sitk_image = reader.Execute()

            # write to MetaIO with compression
            writer = sitk.ImageFileWriter()
            writer.SetImageIO('MetaImageIO')
            writer.SetFileName(os.path.join(output_path, scan_id))
            writer.UseCompressionOn()
            writer.Execute(sitk_image)

        os.remove(downloaded_file)

def download_from_xnat(scan_path, study_id, scan_id, metaio, overwrite=False):
    """
    downloads a scan from XNAT; VPN must be running in order to download.

    creds are stored in netrc

    output is a zip file, to be useful must be either unzipped and/or converted
    to a different format

    """
    creds = netrc.netrc()
    remote_machine = 'https://covid19-xnat.cs.ucl.ac.uk'
    auth_tokens = creds.authenticators(remote_machine)
    
    download_path = scan_path / study_id
    download_path.mkdir(exist_ok=True, parents=True)
    download_file = Path(download_path, scan_id+'.zip')
    metaio_file = Path(download_path, scan_id+'.mhd')

    if not Path.exists(metaio_file) or overwrite:
        with xnat.connect(remote_machine, user=auth_tokens[0], password=auth_tokens[2]) as xnat_session:
            xnat_project = xnat_session.projects['summit_lung50']
            xnat_project.experiments[scan_id].download(download_file)
            save_to_disk(download_file, download_path, scan_id, metaio)

def collate_metadata(scan_path):

    cnt = 0
    metadata = {}
    for root, _, files in os.walk(scan_path):
        for fil in files:
            if fil.endswith('.mhd'):

                if cnt % 100 == 0:
                    print(f'{cnt}')
                else:
                    print('.}',end="")

                cnt += 1
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
                                    md[key]=np.array(val.split(' '),int).reshape([3,3])
                                except:
                                    print(f'error: {fil}, transformation matrix')
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
    df.to_csv(os.path.join(scan_path, 'metadata.csv'))







if __name__ == '__main__':

    action = sys.argv[1]

    if action == 'download_from_xnat':

        scan_path = Path(sys.argv[2])
        scan_id = sys.argv[3]
        study_id = scan_id.split('_',1)[0]

        download_from_xnat(scan_path=scan_path,
                           study_id=study_id,
                           scan_id=scan_id,
                           metaio=True)
    if action == 'xyz2irc':

        xyz = np.array(sys.argv[2].split(','),dtype=np.float64)
        origin = np.array(sys.argv[3].split(','),dtype=np.float64)
        spacing = np.array(sys.argv[4].split(','),dtype=np.float64)

        print(xyz2irc(xyz, origin, spacing))


    if action == 'collate_metadata':

        scan_path = sys.argv[2]

        collate_metadata(scan_path)