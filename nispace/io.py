import os

from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from neuromaps import images 

from tqdm.auto import tqdm

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
                    verbose=True):
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
            file_parc = parcellater.transform(
                data=file, 
                space=data_space,
                ignore_background_data=True,
                background_value=None,
                fill_dropped=True,
                background_parcels_to_nan=drop_background_parcels,
                min_num_valid_datapoints=min_num_valid_datapoints,
                min_fraction_valid_datapoints=min_fraction_valid_datapoints
            ).squeeze()
            
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
            lgr.warning(f"Combined across images, up to {len(nan_parcels['drop'])} parcels were dropped "
                        "after resampling of the parcellation! Data was replaced with nan values. "
                        "Try a coarser parcellation or set 'resampling_target' = 'parcellation' to "
                        f"avoid this behavior (indices: {[int(i) for i in nan_parcels['drop']]}).")
            
        # background intensity parcels
        if drop_background_parcels:
            lgr.info(f"Combined across images, {len(nan_parcels['bg'])} parcels had only background "
                     f"intensity and were set to nan (indices: {[int(i) for i in nan_parcels['bg']]}).")
        
        # below parcel threshold parcels
        if min_num_valid_datapoints or min_fraction_valid_datapoints:
            msg = (f"Combined across images, {len(nan_parcels['excl'])} parcels were dropped due to "
                   f"exclusion criteria.")
            if min_num_valid_datapoints:
                msg += f" Minimum {min_num_valid_datapoints} non-background datapoints."
            if min_fraction_valid_datapoints:
                msg += f" Minimum {min_fraction_valid_datapoints * 100}% non-background datapoints."
            msg += f" (Indices: {[int(i) for i in nan_parcels['excl']]})."
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
