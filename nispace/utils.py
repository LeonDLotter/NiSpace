import pathlib
import sys
import numpy as np
import pandas as pd
import re
import copy
import logging
from colorlog import ColoredFormatter
from nibabel.nifti1 import Nifti1Image
from nibabel.gifti import GiftiImage
from nilearn import image
from nilearn._utils.ndimage import get_border_data
from pathlib import Path
import json
from neuromaps import images


class CriticalRaiseLogger(logging.Logger):
    def critical_raise(self, message, error=Exception):
        """
        Log a critical message and raise an error.

        Parameters:
        - message: Message to log
        - error: Exception class to raise
        """
        self.critical(message)
        raise error(message)
    
    
def _init_lgr(lgr_name="", datefmt="%d/%m/%y %H:%M:%S"):
    logging.setLoggerClass(CriticalRaiseLogger) 
    logger = logging.getLogger(lgr_name)
    logger.setLevel(logging.INFO)
    
    if not logger.hasHandlers():
        fmt = f"%(log_color)s%(levelname)s | {'%(asctime)s | ' if datefmt != '' else ''}%(name)s: %(message)s"
        formatter = ColoredFormatter(fmt, datefmt=datefmt)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
    

def set_log(lgr, verbose=True):
    # standard level: INFO
    if verbose == True:
        lgr.setLevel(logging.INFO)
        return True
    # quiet level: 60 (> CRITICAL)
    elif verbose in [False, None, 0]:
        lgr.setLevel(60)
        return False
    # custom level
    else:
        lgr.setLevel(verbose)
        return True
    

def _rm_ext(path, ext=[".txt", ".csv", ".nii", ".gii", ".gz"]):
    return re.sub("|".join(ext), "", path)

    
def _lower_strip_ws(string):
    if isinstance(string, str):
        return string.lower().replace(" ", "")
    else:
        raise TypeError("Provide string input!")
    

def _get_df_string(kind, xdimred=None, ytrans=None, method=None, stat=None, xsea=False, perm=None, norm=False, mc=None):
    
    if kind=="ytrans":
        df_str = f"ytrans-{ytrans}"
        
    elif kind=="xdimred":
        df_str = f"xdimred-{xdimred}"
        
    elif kind=="coloc":
        if (method is not None) & (stat is not None):
            df_str = f"xdimred-{xdimred}_ytrans-{ytrans}_coloc-{method}_stat-{stat}_xsea-{xsea}"
        else:
            raise ValueError("Provide both method and stat!")
        
    elif kind=="null":
        if (method is not None) & (perm is not None):
            if "sets" in perm:
                xsea = True
            df_str = f"xdimred-{xdimred}_ytrans-{ytrans}_coloc-{method}_xsea-{xsea}_perm-{perm}"
        else:
            raise ValueError("Provide both method and perm!")
        
    elif kind=="p":
        if (method is not None) & (stat is not None) & (perm is not None):
            if "sets" in perm:
                xsea = True
            df_str = (f"xdimred-{xdimred}_ytrans-{ytrans}_coloc-{method}_stat-{stat}_xsea-{xsea}_"
                      f"perm-{perm}_norm-{norm}_mc-{mc}")
        else:
            raise ValueError("Provide method, stat, and perm!")
    
    else:
        raise ValueError(f"Kind {kind} not defined!")
        
    return _lower_strip_ws(df_str)
        

def _del_from_tuple(tpl, elem):
    lst = list(tpl)
    lst.remove(elem)
    return tuple(lst)
    
    
def nan_detector(*arrays):
    # Create an initial mask filled with False, with length equal to the first dimension of the first array
    nan_mask = np.full(arrays[0].shape[0], False)
    
    # Iterate over each array to update the mask where any NaN values are found
    for array in arrays:
        # Use np.isnan to check for NaN values and np.any along axis=1 if the array is 2D to reduce it to 1D
        if array.ndim > 1:
            nan_mask |= np.isnan(array).any(axis=1)
        else:
            nan_mask |= np.isnan(array)
    
    return nan_mask


def remove_nan(data, which="col"):
    
    if isinstance(data, np.ndarray):
        axis = 0 if which=="col" else 1 # 0 drops cols, 1 drops rows
        data = data[np.isnan(data.any(axis=axis))]
    
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        axis = 1 if which=="col" else 0 # 1 drops cols, 0 drops rows
        data = data.dropna(axis=axis)
        
    return data


def fill_nan(data, idx, idx_label=None, which="col"):
    
    data_nan = np.array(data)
        
    if which.startswith("row"):
        for i in idx:
            data_nan = np.insert(data_nan, i, np.zeros((1,data.shape[1])), axis=0)
            data_nan[i,:] = np.nan
                       
    elif which.startswith("col"):
        for i in idx:
            data_nan = np.insert(data_nan, i, np.zeros((1,data.shape[0])), axis=1)
            data_nan[:,i] = np.nan
    
    if isinstance(data, pd.DataFrame):
        
        if idx_label is None:
            idx_label = ["nan"]*len(idx)
            
        if which.startswith("row"):
            index = list(data.index)
            for i_label, i in enumerate(idx):
                index[i:i] = [idx_label[i_label]]
            data_nan = pd.DataFrame(data=data_nan, index=index, columns=data.columns)
            
        elif which.startswith("col"):
            columns = list(data.columns)
            for i_label, i in enumerate(idx):
                columns[i:i] = [idx_label[i_label]]
            data_nan = pd.DataFrame(data=data_nan, index=data.index, columns=columns)

    return data_nan.astype(data.dtype if not isinstance(data.dtype, int) else float)


def print_arg_pairs(**kwargs):
    if len(kwargs) == 0:
        return ""
    else:
        max_len = [max(len(str(value)), len(str(label))) for label, value in kwargs.items()]
        label_row, value_row = "| ", "| "
        for (label, value), max_len in zip(kwargs.items(), max_len):
            label_row += f"{label}{'':<{max_len - len(str(label))}} | ".upper()
            value_row += f"{value}{'':<{max_len - len(str(value))}} | "
        return label_row + "\n" + value_row


def get_column_names(df_or_series, force_list=False):
    """
    Get column names from a DataFrame, the name from a Series, or None if input is a numpy array.
    
    Parameters:
    df_or_series (pd.DataFrame, pd.Series, or np.ndarray): The DataFrame, Series, or numpy array to 
    get the names from.
    
    Returns:
    list or None: List of column names if input is a DataFrame, str if input is a Series, 
        or None if input is a numpy array. If force_list is True, will always return a list.
    """
    if isinstance(df_or_series, pd.DataFrame):
        names = df_or_series.columns.tolist()
    elif isinstance(df_or_series, pd.Series):
        names = df_or_series.name
    elif isinstance(df_or_series, np.ndarray):
        names = None
    else:
        raise TypeError("Input must be a pandas DataFrame, Series, or numpy array")
    if force_list:
        if not isinstance(names, list):
            names = [names]
    return names


def lower(str_list):
    if isinstance(str_list, str):
        return str_list.lower()
    elif isinstance(str_list, list):
        return [s.lower() if isinstance(s, str) else s for s in str_list]


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


def get_background_value(img, border_size=2):
    
    data = images.load_data(img).squeeze()
    background = None
    
    # for now only volumetric implemented
    if len(data.shape) == 3:
        # taken from nilearn.masking.compute_background_mask
        if np.isnan(get_border_data(data, border_size)).any():
            background = np.nan
        else:
            background = np.median(get_border_data(data, border_size))

    return background


def load_img(img):
    # to tuple
    if isinstance(img, (str, pathlib.Path, Nifti1Image, GiftiImage)):
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
        if isinstance(i, (Nifti1Image, GiftiImage)):
            img_load.append(i)
            continue
        elif isinstance(i, pathlib.Path):
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
    if isinstance(labels, (str, pathlib.Path, list, np.ndarray, pd.Series)):
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
        elif isinstance(l, pathlib.Path):
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
    if isinstance(distmat, (str, pathlib.Path, np.ndarray, pd.DataFrame)):
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
        elif isinstance(d, pathlib.Path):
            d = str(d)
        # load 
        try:
            d = pd.read_csv(d, header=None, index_col=None).values
        except:
            raise ValueError("File format not supported. Provide path to csv-like text file.")
        distmat_load.append(d)
    # return as tuple if two, or as array if one 
    return distmat_load[0] if len(distmat_load) == 1 else tuple(distmat_load)


def parc_vect_to_vol(vect, parc):
    # check data
    if isinstance(vect, (list, set, tuple, pd.Series)):
        vect = np.array(vect)
    elif isinstance(vect, (np.ndarray, pd.DataFrame())):
        if len(vect.shape) > 1:
            print("Input vector should be 1d-array/list-like. Will flatten and see for the best.")
        vect = np.array(vect).flatten()
    else:
        raise ValueError("Input vector should be 1d-array or list-like.")
    # load data
    parc = image.load_img(parc)
    parc_3darr = parc.get_fdata().astype(int)
    # create empty 3d array
    vect_3darr = np.zeros_like(parc_3darr).astype(vect.dtype)
    # get unique parcels
    parc_idc = np.trim_zeros(np.unique(parc_3darr)).astype(int)
    # check length
    if len(parc_idc) != len(vect):
        raise ValueError("Number of parcels and length of input vector must match.")
    # create mapping
    mapping = {idx: vect[i] for i, idx in enumerate(parc_idc)}
    # assign values
    for idx, val in mapping.items():
        vect_3darr[parc_3darr == idx] = val
    # return image
    return image.new_img_like(parc, vect_3darr)


def relabel_gifti_parc(parc, new_labels=None):
    
    if not isinstance(parc, GiftiImage):
        raise ValueError("'parc' must be a GiftiImage!")
    
    # get data and labels excluding zero
    data = parc.agg_data()
    data_labels = np.trim_zeros(np.unique(data))
    
    # get new labels
    if new_labels is None:
        new_labels = np.arange(len(data_labels)) + 1
    if not isinstance(new_labels, (np.ndarray, pd.Series, list, set)):
        raise ValueError("'new_labels' must be None or 1d array-like!")
    new_labels = np.array(new_labels, dtype=data.dtype).flatten()
    if len(new_labels) != len(data_labels):
        raise ValueError("'new_labels' must be the same length as the number of parcels in 'parc'!")
        
    # reassign labels
    data_relabeled = np.zeros_like(data, dtype=data.dtype)
    for data_label, new_label in zip(data_labels, new_labels):
        data_relabeled[data == data_label] = new_label
    
    # put back into gifti
    parc_relabeled = copy.deepcopy(parc)
    parc_relabeled.darrays[0].data = data_relabeled
    
    return parc_relabeled
