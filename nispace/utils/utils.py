import sys
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import re
import copy
import logging
from contextlib import contextmanager
from colorlog import ColoredFormatter
from numba import njit

import nibabel as nib
from nilearn import image
from nilearn._utils.ndimage import get_border_data
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
    
    
def _init_lgr(lgr_name="nispace", datefmt="%d/%m/%y %H:%M:%S"):
    logging.setLoggerClass(CriticalRaiseLogger)
    logger = logging.getLogger(lgr_name)
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        fmt = f"%(log_color)s%(levelname)s | {'%(asctime)s | ' if datefmt != '' else ''}%(name)s: %(message)s"
        formatter = ColoredFormatter(fmt, datefmt=datefmt)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Don't propagate to root — prevents double output when the calling
        # application has its own root handler configured.
        logger.propagate = False

    return logger


_quiet_ctx = threading.local()


def set_log(lgr, verbose=True):
    if getattr(_quiet_ctx, 'active', False):
        return False
    root = logging.getLogger("nispace")
    if verbose == True:
        root.setLevel(logging.INFO)
        return True
    elif verbose in [False, None, 0]:
        root.setLevel(60)
        return False
    else:
        root.setLevel(verbose)
        return True


@contextmanager
def _quiet():
    """Temporarily silence the nispace logger; set_log calls inside are no-ops."""
    root = logging.getLogger("nispace")
    old = root.level
    root.setLevel(60)
    _quiet_ctx.active = True
    try:
        yield
    finally:
        root.setLevel(old)
        _quiet_ctx.active = False
    

def _rm_ext(path, ext=[".txt", ".csv", ".nii", ".gii", ".gz"]):
    return re.sub("|".join(ext), "", path)

    
def _lower_strip_ws(string):
    if isinstance(string, str):
        return string.lower().replace(" ", "")
    else:
        raise TypeError("Provide string input!")
    

_DF_STRING_FIELDS = ["xdimred", "ytrans", "coloc", "stat", "xsea", "perm", "mc"]

def _parse_df_string(df_str):
    """Reverse of _get_df_string: parse a key string back into its component fields.

    Returns a dict with any subset of: xdimred, ytrans, coloc, stat, xsea, perm, mc.
    """
    result = {}
    for i, field in enumerate(_DF_STRING_FIELDS):
        marker = f"{field}-"
        if marker not in df_str:
            continue
        idx = df_str.index(marker)
        value_start = idx + len(marker)
        value_end = len(df_str)
        for next_field in _DF_STRING_FIELDS[i + 1:]:
            pos = df_str.find(f"_{next_field}-", value_start)
            if pos != -1:
                value_end = pos
                break
        result[field] = df_str[value_start:value_end]
    return result


def _parse_bool(s):
    """Convert "true"/"false" string to bool, pass other values through."""
    if s == "true":
        return True
    if s == "false":
        return False
    return s


def _get_df_string(kind, xdimred=None, ytrans=None, method=None, stat=None, xsea=False, perm=None, mc=None):
    
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
                      f"perm-{perm}_mc-{mc}")
        else:
            raise ValueError("Provide method, stat, and perm!")

    elif kind=="z":
        if (method is not None) & (stat is not None) & (perm is not None):
            if "sets" in perm:
                xsea = True
            df_str = (f"xdimred-{xdimred}_ytrans-{ytrans}_coloc-{method}_stat-{stat}_xsea-{xsea}_"
                      f"perm-{perm}")
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
    if data_nan.dtype == int:
        data_nan = data_nan.astype(float)
        
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

    return data_nan

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


def mean_by_set_df(df, mean_by_set=True, weighted=True, mean_median="mean"):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if "set" not in df.index.names:
        mean_by_set = False
        
    if weighted == True:
        if "weight" not in df.index.names:
            weighted = False
             
    # grouping by set
    if mean_by_set:
        grouped = df.groupby(level="set", sort=False) 
    else:
        grouped = [("", df)]

    df_mean = []
    for name, group in grouped:
        if weighted:
            weights = group.index.get_level_values('weight')
            if mean_median == 'mean':
                weighted_avg = np.ma.average(np.ma.array(group.values, mask=np.isnan(group)), 
                                             weights=weights, axis=0)
                result = pd.DataFrame(weighted_avg.reshape(1, -1), columns=group.columns)
            elif mean_median == 'median':
                # Weighted median is not directly supported, so we need a custom implementation
                result = group.apply(lambda x: np.nanmedian(np.repeat(x.values, weights)), axis=0).to_frame().T
        else:
            if mean_median == 'mean':
                result = group.mean(axis=0).to_frame().T
            elif mean_median == 'median':
                result = group.median(axis=0).to_frame().T
        result.index = pd.Index([name], name="map")
        df_mean.append(result)

    # Concatenate all results
    df_mean = pd.concat(df_mean)

    return df_mean


def get_column_names(df_or_series, force_list=False):
    """
    Get column names from a DataFrame, the name from a Series, or None if input is a numpy array.
    
    Parameters:
    df_or_series (pd.DataFrame, pd.Series, or np.ndarray): The DataFrame, Series, or numpy array to 
    get the names from.
    
    Returns:
    list or None: List of column names if input is a DataFrame, str if input is a Series, or None 
    if input is a numpy array. If force_list is True, will always return a list.
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


@njit
def vect_to_vol_arr(vect, parc_arr, parc_idc, bg_value=0):
    parc_arr_1d = parc_arr.flatten().astype(vect.dtype)
    vect_arr_1d = np.full_like(parc_arr_1d, bg_value, dtype=vect.dtype)
    parc_idc = parc_idc.astype(vect.dtype)
    if len(parc_idc) != len(vect):
        raise ValueError(f"vect ({vect.shape}) and parc_idc ({parc_idc.shape}) don't match.")
    for i, idx in enumerate(parc_idc):
        vect_arr_1d[parc_arr_1d==idx] = vect[i]
    return vect_arr_1d.reshape(parc_arr.shape)

@njit
def vol_to_vect_arr(vol_arr, parc_arr, parc_idc, bg_value=np.nan):
    vol_arr2d = vol_arr.flatten()
    parc_arr2d = parc_arr.flatten().astype(vol_arr.dtype)
    parc_idc = parc_idc.astype(vol_arr.dtype)
    vect = np.zeros(len(parc_idc), dtype=vol_arr.dtype)
    filter_bg = not np.isnan(bg_value)
    for i, idx in enumerate(parc_idc):
        in_parcel = parc_arr2d == idx
        not_nan = ~np.isnan(vol_arr2d)
        if filter_bg:
            idc = in_parcel & not_nan & (vol_arr2d != bg_value)
        else:
            idc = in_parcel & not_nan
        vals = vol_arr2d[idc]
        vect[i] = vals.mean() if len(vals) > 0 else np.nan
    return vect
    

def parc_vect_to_vol(vect, parc):
    # check data
    if isinstance(vect, (list, set, tuple, pd.Series)):
        vect = np.array(vect)
    elif isinstance(vect, (np.ndarray, pd.DataFrame)):
        if len(vect.shape) > 1:
            print("Input vector should be 1d-array/list-like. Will flatten and hope for the best.")
        vect = np.array(vect).flatten()
    else:
        raise ValueError("Input vector should be 1d-array or list-like.")
    # load data
    parc = image.load_img(parc)
    parc_arr = parc.get_fdata()
    parc_idc = np.trim_zeros(np.unique(parc_arr))
    # get volume
    vol_arr = vect_to_vol_arr(vect, parc_arr, parc_idc)
    # return image
    return image.new_img_like(parc, vol_arr)


def relabel_gifti_parc(parc, new_labels=None):
    
    if not isinstance(parc, nib.GiftiImage):
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


def relabel_nifti_parc(parc, new_order=None, new_labels=None, dtype=None):
    
    parc_orig = images.load_nifti(parc)
    data_orig = parc_orig.get_fdata()
    if dtype is None:
        dtype = data_orig.dtype
    if new_order is None:
        new_order = np.trim_zeros(np.unique(data_orig)).astype(dtype)
    if new_labels is None:
        new_labels = np.arange(len(new_order)).astype(dtype) + 1
    if len(new_order) != len(new_labels) != len(np.unique(data_orig)):
        raise ValueError("'new_order' and 'new_labels' must be the same length as the number of parcels in 'parc'!")
    if not all(np.isin(new_order, np.unique(data_orig))):
        raise ValueError("'new_order' must be a subset of the parcels in 'parc'!")
    
    parc_relabeled = np.zeros_like(data_orig, dtype=dtype)
    for label_orig, label_new in zip(new_order, new_labels):
        parc_relabeled[data_orig == label_orig] = label_new
    parc_relabeled = image.new_img_like(parc_orig, parc_relabeled)
    
    return parc_relabeled


def merge_parcellations(parcellations, labels=None, quick=False):
    if not isinstance(parcellations, list):
        raise ValueError("parcellations must be a list")
    if labels is None:
        labels = [None] * len(parcellations)
    if not isinstance(labels, list):
        raise ValueError("labels must be a list")
    if len(parcellations) != len(labels):
        raise ValueError("parcellations and labels must have the same length")
    if not np.equal(*[type(p) for p in parcellations]):
        raise ValueError("all parcellations must be of the same type")
    if not isinstance(parcellations[0], nib.Nifti1Image):
        raise NotImplementedError("Parcellation merging currently only implemented for Nifti1Image")
    if not np.allclose(*[parc.shape for parc in parcellations]):
        raise ValueError("all parcellations must have the same shape")
    
    arr_merged = np.zeros_like(parcellations[0].get_fdata(), dtype=np.int32)
    labels_merged = pd.Series(dtype=str)
    
    if quick:
        # arrays
        arr1 = parcellations[0].get_fdata()
        arr2 = parcellations[1].get_fdata()
        # take first parcellation
        arr_merged = arr1.copy()
        # highest index of first parcellation
        arr1_idx_max = np.max(arr1)
        # nonzero voxels in second parcellation
        arr2_nonzero = arr2 > 0
        # all nonzero voxels in arr2 to zero 
        arr_merged[arr2_nonzero] = 0
        # add second parcellation
        arr_merged[arr2_nonzero] = arr2[arr2_nonzero] + arr1_idx_max
        # return
        return image.new_img_like(parcellations[0], arr_merged)       
    
    # slow approach with relabeling
    i = 1
    for ii, (parc, labs) in enumerate(zip(parcellations, labels)):
        arr = parc.get_fdata()
        idc = np.trim_zeros(np.unique(arr))
        if labs is None:
            labs = idc
        if len(idc) != len(labs):
            raise ValueError(f"parcellation at position {ii} has {len(idc)} indices, but {len(labs)} labels")
        for idx, l in zip(idc, labs):
            arr_merged[arr == idx] = i
            labels_merged.loc[i] = l
            i += 1
    return image.new_img_like(parcellations[0], arr_merged), labels_merged


def correlate_hemispheres(img, mask=None):
    if isinstance(img, (str, Path, nib.Nifti1Image)):
        #raise NotImplementedError("Nifti1Image input not implemented yet!")
        img = images.load_nifti(img)
        dat = img.get_fdata()
    elif isinstance(img, np.ndarray):
        #raise NotImplementedError("Numpy array input not implemented yet!")
        dat = np.squeeze(img)
    elif isinstance(img, (tuple, list)):
        if isinstance(img[0], (nib.GiftiImage, Path, str)):
            dat = (images.load_gifti(img[0]).agg_data(), images.load_gifti(img[1]).agg_data())
        elif isinstance(img[0], np.ndarray):
            dat = (np.squeeze(img[0]), np.squeeze(img[1]))
        else:
            raise ValueError("If input is a tuple, it must be a size-2 tuple of numpy arrays or (path to) two GiftiImages!")
    else:
        raise ValueError("Input must be (path to) a Nifti1Image, numpy array, or size-2 tuple of numpy arrays or GiftiImages!")
    
    if isinstance(dat, tuple):
        a = dat[0].copy()
        b = dat[1].copy()
    else:
        #xyz0 = image.coord_transform(0, 0, 0, np.linalg.inv(affine))
        if mask is None:
            mask = ~(np.isclose(dat, 0) | np.isnan(dat))
        elif isinstance(mask, (str, Path, nib.Nifti1Image)):
            mask = images.load_nifti(mask).get_fdata()
        # original data
        a = dat[mask]
        # flipped across x-axis
        b = dat[::-1, :, :][mask]
        
    return np.corrcoef(a.flatten(), b.flatten())[0,1]


def mirror_nifti(img, affine=None, direction="left_to_right", match_r=False, mask=None):
    if isinstance(img, (np.ndarray)):
        dat = np.squeeze(np.array(img))
        return_array = True
        if affine is None:
            raise ValueError("Affine must be provided if input is array")
    else:
        img = images.load_nifti(img)
        return_array = False
        dat = img.get_fdata()
        affine = img.affine
        
    if len(dat.shape) != 3:
        raise ValueError("Input must be a 3D array or Nifti1Image")
    
    # get coordinates of voxel (0,0,0)
    xyz0 = image.coord_transform(0, 0, 0, np.linalg.inv(affine))
    
    # get left hemisphere
    dat_lh = dat.copy()
    dat_lh[:int(xyz0[0])] = 0
    
    # get right hemisphere
    dat_rh = dat.copy()
    dat_rh[int(xyz0[0]):] = 0
    
    # mirror
    if direction == "right_to_left":
        dat_mirr = dat_lh.copy()
        dat_mirr[:int(xyz0[0])] = dat_lh[::-1, :, :][:int(xyz0[0])]
    elif direction == "left_to_right":
        dat_mirr = dat_rh.copy()
        dat_mirr[int(xyz0[0]):] = dat_rh[::-1, :, :][int(xyz0[0]):]
    elif direction in ["average", "bilateral"]:
        dat_mirr = (dat + dat_lh[::-1, :, :] + dat_rh[::-1, :, :]) / 2
    elif direction == "switch":
        dat_mirr = np.zeros_like(dat)
        dat_mirr[int(xyz0[0]):] = dat_rh[::-1, :, :][int(xyz0[0]):]
        dat_mirr[:int(xyz0[0])] = dat_lh[::-1, :, :][:int(xyz0[0])]
    elif direction == "drop_left":
        dat_mirr = dat_lh
    elif direction == "drop_right":
        dat_mirr = dat_rh
    else:
        raise ValueError(f"Invalid direction: {direction}")
        
    # n
    if return_array:
        return dat_mirr
    else:
        return image.new_img_like(img, dat_mirr)
    
    
def mirror_gifti(img, direction="left_to_right", match_r=False, mask=None):
    if not isinstance(img, tuple):
        raise ValueError("Input must be a tuple of two GiftiImages or arrays!")
    
    if isinstance(img[0], np.ndarray):
        dat = (np.squeeze(img[0]), np.squeeze(img[1]))
        return_array = True
    elif isinstance(img[0], nib.GiftiImage):
        dat = (images.load_gifti(img[0]).agg_data(), images.load_gifti(img[1]).agg_data())
        return_array = False
    else:
        raise ValueError("Input must be a tuple of two GiftiImages or arrays!")
            
    # mirror
    if direction == "left_to_right":
        dat_mirr = (dat[0], dat[0].copy())
    elif direction == "right_to_left":
        dat_mirr = (dat[1].copy(), dat[1])
    elif direction in ["average", "bilateral"]:
        dat_mirr = ((dat[0] + dat[1]) / 2, (dat[0] + dat[1]) / 2)
        
    # return
    if return_array:
        return dat_mirr
    else:
        return (nib.GiftiImage(darrays=dat_mirr[0]), nib.GiftiImage(darrays=dat_mirr[1]))
    

@njit
def _corr_vector(data_1d, correlation=1, seed=None):  
    
    # standardize input
    mu, sigma = np.nanmean(data_1d), np.nanstd(data_1d)
    data_1d = (data_1d - mu) / sigma
    
    # generate random noise with same length as input
    if seed is not None:
        np.random.seed(seed)
    epsilon = np.random.normal(0, 1, len(data_1d))
    epsilon = (epsilon - np.mean(epsilon)) / np.std(epsilon)
    
    # correlated vector using the formula:
    # output = ρ * input + √(1-ρ²) * ε
    output = correlation * data_1d + np.sqrt(1 - correlation**2) * epsilon
    
    # rescale to original scale
    output = output * sigma + mu
    
    return output

def correlated_vector(data_1d, correlation=1, seed=None):    
    # if correlation is 1, return the original vector
    if correlation == 1:
        return data_1d.copy()
    
    # to array
    data_1d = np.array(data_1d).squeeze()

    # get correlated vector
    output = _corr_vector(data_1d, correlation=correlation, seed=seed)
    
    return output
