import os
import warnings
import json
from pathlib import Path
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

import nibabel as nib
from neuromaps import images 

from . import lgr
from .utils import set_log
from .parcellate import Parcellater


def parcellate_data(data, 
                    data_labels=None,
                    data_space=None, 
                    parcellation=None, 
                    parc_labels=None,
                    parc_space=None,
                    parc_hemi=None,
                    resampling_target="data",
                    drop_background_parcels=True,
                    min_num_valid_datapoints=None, 
                    min_fraction_valid_datapoints=None,
                    return_parc=False,
                    dtype=None,
                    n_proc=1,
                    verbose=True,
                    ignore_zero_division_warning=True):
    """
    Parcellates given imaging data using a specified parcellation.

    Parameters
    ----------
    parcellation : str, os.PathLike, nib.Nifti1Image, nib.GiftiImage, or tuple
        The parcellation image or surfaces, where each region is identified by a unique integer ID.
    parc_labels : list
        Labels for the parcellation regions.
    parc_space : str
        The space in which the parcellation is defined.
    parc_hemi : list of str
        Hemispheres to consider for parcellation, e.g., ["L", "R"].
    resampling_target : {'data', 'parcellation'}
        Specifies which image gives the final shape/size.
    data : list, pd.DataFrame, pd.Series, or np.ndarray
        The imaging data to be parcellated.
    data_labels : list
        Labels for the input data.
    data_space : str
        The space in which the input data is defined.
    drop_background_parcels : bool
        Whether to drop parcels that contain only background intensity.
    min_num_valid_datapoints : int, optional
        Minimum number of valid datapoints required per parcel.
    min_fraction_valid_datapoints : float, optional
        Minimum fraction of valid datapoints required per parcel.
    n_proc : int
        Number of processors to use for parallel processing.
    dtype : data-type
        Desired data type of the output.

    Returns
    -------
    pd.DataFrame
        Parcellated data in a DataFrame.

    Raises
    ------
    TypeError
        If the input data type is not recognized.
    ValueError
        If the resampling target is invalid.

    Notes
    -----
    This function handles different types of input data, including lists, DataFrames, Series, and ndarrays.
    It also manages different parcellation formats and resampling targets.
    """
    verbose = set_log(lgr, verbose)
    
    ## put data into list
    if isinstance(data, Path):
        data = str(data)
    if isinstance(data, (str, tuple, nib.Nifti1Image, nib.GiftiImage)):
        data = [data]
    
    ## case list
    if isinstance(data, list):
        lgr.info("Input type: list, assuming imaging data.")

        # load parcellation
        if parcellation is None:
            lgr.critical_raise("If input 'data' is list, 'parcellation' must not be None!",
                               TypeError)
        if isinstance(parcellation, Path):
            parcellation = str(parcellation)
        if isinstance(parcellation, str):
            if parcellation.endswith(".nii") | parcellation.endswith(".nii.gz"):
                parcellation = images.load_nifti(parcellation)
            elif parcellation.endswith(".gii") | parcellation.endswith(".gii.gz"):
                parcellation = images.load_gifti(parcellation)
                if parc_hemi is None:
                    lgr.warning("Input is single GIFTI image but 'hemi' is not given. Assuming left!")
                    parc_hemi = "left"
            else:
                lgr.error(f"Argument 'parcellation' of type string, but no path ('{parcellation}')!")
        elif isinstance(parcellation, nib.GiftiImage):      
            parcellation = images.load_gifti(parcellation) 
        elif isinstance(parcellation, nib.Nifti1Image):      
            parcellation = images.load_nifti(parcellation) 
        elif isinstance(parcellation, tuple):
            parcellation = (images.load_gifti(parcellation[0]),
                            images.load_gifti(parcellation[1])) 
        else:
            lgr.critical(f"Parcellation data type not recognized! ({type(parcellation)})")
        
        # catch problems
        if (data_space.lower() in ["mni", "mni152"]) & \
            (parc_space.lower() not in ["mni", "mni152"]) & \
            (resampling_target=="data"):
                lgr.warning("Data in MNI space but parcellation in surface space and "
                            "'resampling_target' is 'data'! Cannot resample surface to MNI: "
                            "Setting 'resampling_target' to 'parcellation'.")
                resampling_target = "parcellation"
            
        # number of parcels
        if isinstance(parcellation, nib.Nifti1Image):
            parc_data = parcellation.get_fdata()
        elif isinstance(parcellation, nib.GiftiImage):
            parc_data = parcellation.darrays[0].data
        elif isinstance(parcellation, tuple):
            parc_data = np.c_[parcellation[0].darrays[0].data, parcellation[1].darrays[0].data]
        else:
            lgr.error("Something is wrong with the loaded parcellation image!")
        parc_idc = np.trim_zeros(np.unique(parc_data))    
        parc_n = len(parc_idc)
          
        # modified neuromaps parcellater: can deal with str, path, nifti, gifti, tuple
        parcellater = Parcellater(
            parcellation=parcellation, 
            space=parc_space,
            resampling_target=resampling_target,
            hemi=parc_hemi
        ).fit()
        
        # data extraction function
        def extract_data(file):
            
            # apply parcellater
            kwargs = dict(
                data=file, 
                space=data_space,
                ignore_background_data=True,
                background_value=None,
                fill_dropped=True,
                background_parcels_to_nan=drop_background_parcels,
                min_num_valid_datapoints=min_num_valid_datapoints,
                min_fraction_valid_datapoints=min_fraction_valid_datapoints
            )
            # apply parcellater
            if ignore_zero_division_warning:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "invalid value encountered in divide", RuntimeWarning)
                    file_parc = parcellater.transform(**kwargs).squeeze()
            else:
                file_parc = parcellater.transform(**kwargs).squeeze()
                
            # return data and dropped parcels
            return (file_parc, parcellater._parc_idc_dropped, parcellater._parc_idc_bg, 
                    parcellater._parc_idc_excl)
        
        # extract data (in parallel)
        lgr.info(f"Parcellating imaging data.")
    
        # run  
        data_parc_list = Parallel(n_jobs=n_proc)(
            delayed(extract_data)(f) for f in tqdm(
                data, desc=f"Parcellating ({n_proc} proc)", disable=not verbose)
        )
        # collect data
        data_parc = np.zeros((len(data), parc_n))
        nan_parcels = {"drop": set(), "bg": set(), "excl": set()}
        for i, par_out in enumerate(data_parc_list):
            data_parc[i, :] = par_out[0]
            nan_parcels["drop"].update(set(par_out[1]))
            nan_parcels["bg"].update(set(par_out[2]))
            nan_parcels["excl"].update(set(par_out[3]))
            
        # dropped parcels
        if len(nan_parcels["drop"]) > 0:
            lgr.warning(f"Combined across images, up to {len(nan_parcels['drop'])} parcel(s) were dropped "
                        "after resampling of the parcellation! Data was replaced with nan values."
                        "Try a coarser parcellation or set 'resampling_target' = 'parcellation' to "
                        f"avoid this behavior ({[int(i) for i in nan_parcels['drop']]}).")
            
        # background intensity parcels
        if drop_background_parcels:
            lgr.info(f"Combined across images, {len(nan_parcels['bg'])} parcel(s) had only background "
                     f"intensity and were set to nan ({[int(i) for i in nan_parcels['bg']]}).")
        
        # below parcel threshold parcels
        if min_num_valid_datapoints or min_fraction_valid_datapoints:
            msg = (f"Combined across images, {len(nan_parcels['excl'])} parcels were dropped "
                   f"due to exclusion criteria: ")
            if min_num_valid_datapoints and min_fraction_valid_datapoints:
                msg += (f"min. n = {min_num_valid_datapoints} and "
                        f"{min_fraction_valid_datapoints * 100}% non-background datapoints.")
            elif min_num_valid_datapoints:
                msg += f"min. n = {min_num_valid_datapoints} non-background datapoints."
            elif min_fraction_valid_datapoints:
                msg += f"min. {min_fraction_valid_datapoints * 100}% non-background datapoints."
            msg += f" ({[int(i) for i in nan_parcels['excl']]})."
            lgr.info(msg)
                     
        # output dataframe
        if data_labels is None:
            try:
                if isinstance(data[0], tuple):
                    data_labels = [os.path.basename(f[0]).replace(".gii","").replace(".gz","") \
                        for f in data]
                else:
                    data_labels = [os.path.basename(f).replace(".nii","").replace(".gz","") \
                        for f in data]
            except:
                data_labels = list(range(len(data)))
        df_parc = pd.DataFrame(
            data=data_parc, 
            index=data_labels,
            columns=parc_labels
        )
    
    ## case array
    elif isinstance(data, np.ndarray):
        lgr.info("Input type: ndarray, assuming parcellated data with shape "
                 "(n_files/subjects/etc, n_parcels).")
        if len(data.shape)==1:
            data = data[np.newaxis, :]
        df_parc = pd.DataFrame(
            data=data,
            index=data_labels,
            columns=parc_labels
        )
            
    ## case dataframe
    elif isinstance(data, pd.DataFrame):
        lgr.info("Input type: DataFrame, assuming parcellated data with shape "
                 "(n_files/subjects/etc, n_parcels).")
        df_parc = pd.DataFrame(
            data=data.values,
            index=data_labels if data_labels is not None else data.index,
            columns=parc_labels if parc_labels is not None else data.columns
        )
    
    ## case series
    elif isinstance(data, pd.Series):
        lgr.info("Input type: Series, assuming parcellated data with shape (1, n_parcels).")
        df_parc = pd.DataFrame(
            data=data.values,
            index=parc_labels if parc_labels is not None else data.index, 
            columns=data_labels if data_labels is not None else [data.name],
        )
        df_parc = df_parc.T
    
    ## case not defined
    else:
        lgr.critical_raise(f"Can't import from data with type {type(data)}!",
                           TypeError)
        
    ## check for nan's
    if df_parc.isnull().any(axis=None):
        lgr.warning("Parcellated data contains nan values!")
 
    ## return data array
    if return_parc:
        return df_parc.astype(dtype), parcellation
    else:
        return df_parc.astype(dtype)


def read_json(json_path):
    if isinstance(json_path, (str, Path)):
        with open(json_path) as f:
            json_dict = json.load(f)
    else:
        try:
            json_dict = dict(json_path)
        except ValueError:
            print("Provide path to json-like file or dict-like object!")
    return json_dict


def write_json(json_dict, json_path):
    if isinstance(json_path, (str, Path)):
        json_path = Path(json_path)
        with open(json_path, "w") as f:
            json.dump(json_dict, f)
    else:
        print("Provide path-like object for argument 'json_path'")
    return json_path


def load_img(img):
    # to tuple
    if isinstance(img, (str, Path, nib.Nifti1Image, nib.GiftiImage)):
        img = (img,)
    elif isinstance(img, list):
        img = tuple(img)
    elif isinstance(img, tuple):
        pass
    else:
        raise ValueError("Input must be path, list, tuple or image object")
    # load
    img_load = []
    for i in img:
        # return if image, to string if path
        if isinstance(i, (nib.Nifti1Image, nib.GiftiImage)):
            img_load.append(i)
            continue
        elif isinstance(i, Path):
            i = str(i)
        # load 
        if i.endswith(".nii") or i.endswith(".nii.gz"):
            i = images.load_nifti(i)
        elif i.endswith(".gii") or i.endswith(".gii.gz"):
            i = images.load_gifti(i)
        else:
            raise ValueError("File format not supported. Path must end with .nii(.gz) or .gii(.gz)")
        img_load.append(i)
    # return as tuple if two, or 
    return img_load[0] if len(img_load) == 1 else tuple(img_load)


def load_labels(labels, concat=True, header=None, index=None):
    # to tuple
    if isinstance(labels, (str, Path, list, np.ndarray, pd.Series)):
        labels = (labels,)
    elif isinstance(labels, tuple):
        pass
    else:
        raise ValueError("Input must be path, list, ndarray or Series")
    # load
    labels_load = []
    for l in labels:
        # return if array/list, to string if path
        if isinstance(l, (list, np.ndarray, pd.Series)):
            labels_load.append(list(l))
            continue
        elif isinstance(l, Path):
            l = str(l)
        # load 
        try:
            l = pd.read_csv(l, header=header, index_col=index).iloc[:,0].to_list()
        except:
            raise ValueError("File format not supported. Provide path to csv-like text file.")
        labels_load.append(l)
    # return as tuple if two, or list of one
    if len(labels_load) == 1:
        labels_load = labels_load[0]
    else:
        if concat:
            labels_load = labels_load[0] + labels_load[1]
        else:
            labels_load = tuple(labels_load)
    return labels_load


def load_distmat(distmat):
    # to tuple
    if isinstance(distmat, (str, Path, np.ndarray, pd.DataFrame)):
        distmat = (distmat,)
    elif isinstance(distmat, list):
        distmat = tuple(distmat)
    elif isinstance(distmat, tuple):
        pass
    else:
        raise ValueError("Input must be path, list, tuple, ndarray, or DataFrame")
    # load
    distmat_load = []
    for d in distmat:
        # return if array, to string if path
        if isinstance(d, (np.ndarray, pd.DataFrame)):
            distmat_load.append(np.array(d))
            continue
        elif isinstance(d, Path):
            d = str(d)
        # load 
        try:
            d = pd.read_csv(d, header=None, index_col=None).values
        except:
            raise ValueError("File format not supported. Provide path to csv-like text file.")
        distmat_load.append(d)
    # return as tuple if two, or as array if one 
    return distmat_load[0] if len(distmat_load) == 1 else tuple(distmat_load)
