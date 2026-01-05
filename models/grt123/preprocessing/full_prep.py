import os
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas
from scipy.io import loadmat
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from skimage import measure
from skimage.morphology import convex_hull_image
from step1 import step1_python, step1_python_summit

#import h5py

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>2*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask

# def savenpy(id):
id = 1

def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def resample(imgs, spacing, new_spacing,order = 2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')

def savenpy(id, filelist, prep_folder, data_path, use_existing=True, metadata_path=None):      
    resolution = np.array([1,1,1])
    scan_id = filelist[id]
    if use_existing:
        if os.path.exists(os.path.join(prep_folder,scan_id+'_label.npy')) and os.path.exists(os.path.join(prep_folder,scan_id+'_clean.npy')):
            print(scan_id+' had been done')
            return
    try:

        if metadata_path:
            metadata = pandas.read_csv(metadata_path)[[
                'scan_id',
                'nodule_x_coordinate',
                'nodule_y_coordinate',
                'nodule_z_coordinate',
                'nodule_diameter_mm'
            ]]
            metadata = metadata[metadata.scan_id==scan_id]
            label = metadata.to_numpy()
            label = label[:,[3, 1, 2, 4]].astype('float')
        else:
            label = []

        im, m1, m2, spacing, origin = step1_python(os.path.join(data_path,scan_id))
        Mask = m1+m2
        
        newshape = np.round(np.array(Mask.shape)*spacing/resolution)
        xx,yy,zz= np.where(Mask)
        box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
        box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        box = np.floor(box).astype('int')
        margin = 5
        extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
        extendbox = extendbox.astype('int')



        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2
        Mask = m1+m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        im[np.isnan(im)]=-2000
        sliceim = lumTrans(im)
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = sliceim*extramask>bone_thresh
        sliceim[bones] = pad_value
        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        sliceim = sliceim2[np.newaxis,...]
        np.save(os.path.join(prep_folder,scan_id+'_clean'),sliceim)


        if len(label)==0:
            label2 = np.array([[0,0,0,0]])
        elif len(label[0])==0:
            label2 = np.array([[0,0,0,0]])
        elif label[0][0]==0:
            label2 = np.array([[0,0,0,0]])
        else:
            haslabel = 1
            labels = []
            for idx in range(len(label)):
                cri = label[idx]

                
                cri[:3] = (label[idx][:3][[0,2,1]] - origin[[0,2,1]]) / spacing[[0,2,1]]
                cri[:3] = cri[:3] * (spacing[[0,2,1]] / resolution[[0,2,1]])
                cri[:3] = cri[:3] - extendbox[:,0]

                cri[3] = cri[3] * (spacing[1] / resolution[1])
                
                labels.append(cri)

            #label2 = np.copy(label).T
            #label2[:3] = label2[:3][[0,2,1]] - np.expand_dims(origin[[0,2,1]],1)
            #label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
            #label2[3] = label2[3]*spacing[1]/resolution[1]
            #label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
            #label2 = label2[:4].T

        np.save(os.path.join(prep_folder,scan_id+'_label'), np.array(labels))
    except:
        print('bug in '+scan_id)
        raise
    print(scan_id+' done')

def savenpy_summit(id, scanpath_list, prep_folder, use_existing=True, metadata_path=None):     
    """
    JM: Added for SUMMIT dataset pre-processing.
    Preprocess and save CT scan data and nodule labels to numpy files.
    This is a sub-function of the full_prep_summit function that handles the conversion
    and storage of individual scan data to .npy format for efficient loading during training.
    The function performs the following operations:
    1. Loads CT scan and mask data from the specified scan path
    2. Retrieves nodule metadata (coordinates and diameter) from a CSV file using pandas
    3. Resamples the scan to a standard resolution [1, 1, 1]
    4. Applies Hounsfield unit transformation and bone removal
    5. Extracts region of interest based on lung masks
    6. Transforms nodule label coordinates to match the resampled and cropped volume
    7. Saves processed scan and labels as separate .npy files
    Args:
        id (int): Index into scanpath_list to identify which scan to process
        scanpath_list (list): List of pathlib.Path objects pointing to scan files
        prep_folder (str): Directory path where output .npy files will be saved
        use_existing (bool, optional): If True, skip processing if output files already exist. Defaults to True.
        metadata_path (str, optional): Path to CSV file containing nodule metadata. Expected columns:
            - 'scan_id' (str): Identifier matching scan filename stem
            - 'nodule_x_coordinate' (float): X coordinate in mm
            - 'nodule_y_coordinate' (float): Y coordinate in mm
            - 'nodule_z_coordinate' (float): Z coordinate in mm
            - 'nodule_diameter_mm' (float): Nodule diameter in millimeters
            Metadata is filtered using pandas boolean indexing: metadata[metadata.scan_id==scan_id]
            Defaults to None, which results in empty label array.
    Returns:
        None
    Outputs:
        - {scan_id}_clean.npy: Preprocessed CT scan volume with shape (1, H, W, D)
        - {scan_id}_label.npy: Array of nodule labels with shape (N, 4) where columns are [z, x, y, diameter]
    """

    resolution = np.array([1,1,1])
    scan_path = scanpath_list[id]

    scan_id = scan_path.stem
    ext = scan_path.suffix

    print(f'Preparing ... {scan_id}', flush=True)

    if use_existing:
        if os.path.exists(os.path.join(prep_folder,scan_id+'_label.npy')) and os.path.exists(os.path.join(prep_folder,scan_id+'_clean.npy')):
            print(scan_id+' had been done')
            return
    try:
        if metadata_path:
            metadata = pandas.read_csv(metadata_path)[[
                'scan_id',
                'nodule_x_coordinate',
                'nodule_y_coordinate',
                'nodule_z_coordinate',
                'nodule_diameter_mm'
            ]]
            metadata = metadata[metadata.scan_id==scan_id]
            label = metadata.to_numpy()
            label = label[:,[3, 2, 1, 4]].astype('float')
        else:
            label = []

        
        if not os.path.exists(scan_path):
            print('Not downloaded:',scan_path)
            return
        
        im, m1, m2, spacing, origin = step1_python_summit(scan_path)
        Mask = m1+m2
        
        newshape = np.round(np.array(Mask.shape)*spacing/resolution)
        xx,yy,zz= np.where(Mask)
        box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
        box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        box = np.floor(box).astype('int')
        margin = 5
        extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
        extendbox = extendbox.astype('int')

        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2
        Mask = m1+m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        im[np.isnan(im)]=-2000
        sliceim = lumTrans(im)
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = sliceim*extramask>bone_thresh
        sliceim[bones] = pad_value
        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        sliceim = sliceim2[np.newaxis,...]
        np.save(os.path.join(prep_folder,scan_id+'_clean'),sliceim)

        if len(label)==0:
            labels = np.array([[0,0,0,0]])
        elif len(label[0])==0:
            labels = np.array([[0,0,0,0]])
        elif label[0][0]==0:
            labels = np.array([[0,0,0,0]])
        else:
            haslabel = 1
            labels = []
            for idx in range(len(label)):
                cri = label[idx]                
                cri[:3] = (label[idx][:3] - origin) / spacing
                cri[:3] = cri[:3] * (spacing / resolution)
                cri[:3] = cri[:3] - extendbox[:,0]
                cri[3] = cri[3] * (spacing[1] / resolution[1])
                labels.append(cri)

        try:
            np.save(os.path.join(prep_folder,scan_id+'_label'), labels)

        except:
            print('error in saving label', flush=True)
            print(label, flush=False)
            np.save(os.path.join(prep_folder,scan_id+'_label'), [[0,0,0,0]])

    except Exception as err:
        print(f'bug in {scan_id}, error:{err.__str__()}')

    print(scan_id+' done')
    
def full_prep(data_path,prep_folder,n_worker = None, use_existing=True, metadata_path=None):
    warnings.filterwarnings("ignore")
    if not os.path.exists(prep_folder):
        os.mkdir(prep_folder)
            
    print('starting preprocessing')

    filelist = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,f))]
    N = len(filelist)

    if n_worker:
        pool = Pool(n_worker)
        partial_savenpy = partial(savenpy,
                                filelist=filelist,
                                prep_folder=prep_folder,
                                data_path=data_path,
                                use_existing=use_existing,
                                metadata_path=metadata_path)


        _ = pool.map(partial_savenpy,range(N))
        pool.close()
        pool.join()

    else:
        for idx in range(N):
            _ = savenpy(idx, 
                        filelist=filelist,
                        prep_folder=prep_folder,
                        data_path=data_path,
                        use_existing=use_existing,
                        metadata_path=metadata_path)
            
    print('end preprocessing')
    return filelist

def full_prep_summit(data_path, prep_folder, scanlist_path, n_worker = None, use_existing=True, metadata_path=None):
    """
    Preprocess SUMMIT scan data and save as numpy arrays.
    This function loads scan files, validates them against a scanlist, and converts them
    to numpy format for model training. Supports parallel processing for efficiency.
    Note: Added by JM to support preparation of SUMMIT scans.
    Args:
        data_path (str): Path to the root directory containing scan files (.mhd format).
        prep_folder (str): Path to the output folder where preprocessed numpy files will be saved.
        scanlist_path (str): Path to CSV file containing 'scan_id' column with valid scan identifiers.
        n_worker (int, optional): Number of worker processes for parallel processing. 
                                   If > 1, uses multiprocessing Pool. Defaults to None (sequential processing).
        use_existing (bool, optional): If True, skips preprocessing for already existing files. Defaults to True.
        metadata_path (str, optional): Path to metadata file for additional scan information. Defaults to None.
    Returns:
        list: List of Path objects pointing to all located scan files (.mhd) matching the scanlist.
    """


    warnings.filterwarnings("ignore")

    if not os.path.exists(prep_folder):
        os.mkdir(prep_folder)

    print('starting preprocessing')
    
    scan_ids = pandas.read_csv(scanlist_path)['scan_id'].tolist()

    scan_paths = [
        scan_path
        for scan_path in Path(data_path).rglob('*.mhd')
        if scan_path.is_file() and scan_path.name.replace('.mhd', '') in scan_ids
    ]

    N = len(scan_paths)

    if n_worker>1:
        pool = Pool(n_worker)
        partial_savenpy = partial(savenpy_summit,
                                scanpath_list=scan_paths,
                                prep_folder=prep_folder,
                                use_existing=use_existing,
                                metadata_path=metadata_path)


        _ = pool.map(partial_savenpy,range(N))
        pool.close()
        pool.join()

    else:
        for idx in range(N):
            _ = savenpy_summit(idx,
                               scanpath_list=scan_paths,
                               prep_folder=prep_folder,
                               use_existing=use_existing,
                               metadata_path=metadata_path)

    print('end preprocessing')
    return scan_paths    
