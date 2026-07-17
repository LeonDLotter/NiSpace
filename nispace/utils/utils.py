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
    """Set the "nispace" root logger's level and return whether INFO is enabled.

    A no-op (returns False immediately) while inside a :func:`_quiet` context
    manager block (``_quiet_ctx.active``), so temporary silencing can't be
    overridden by a nested `verbose=True` call. `lgr` is accepted for call-site
    symmetry with module-level loggers but is not itself used — the level is
    always set on the shared "nispace" root logger.

    Parameters
    ----------
    lgr : logging.Logger
        Unused; kept for signature compatibility with callers that pass their
        module-level logger.
    verbose : bool, int, or None, default=True
        ``True``: set level to INFO. ``False``/``None``/``0``: set level to
        WARNING. Any other value is passed directly to ``Logger.setLevel``
        (e.g. a numeric logging level or level name).

    Returns
    -------
    bool
        Whether INFO-level messages are (now) enabled, or False if silenced
        by an active `_quiet` context.
    """
    if getattr(_quiet_ctx, 'active', False):
        return False
    root = logging.getLogger("nispace")
    if verbose == True:
        root.setLevel(logging.INFO)
        return True
    elif verbose in [False, None, 0]:
        root.setLevel(logging.WARNING)
        return False
    else:
        root.setLevel(verbose)
        return root.level <= logging.INFO


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
    

_DF_STRING_FIELDS = ["xdimred", "ytrans", "coloc", "stat", "xsea", "perm", "pooled", "mc"]

def _parse_df_string(df_str):
    """Reverse of _get_df_string: parse a key string back into its component fields.

    Returns a dict with any subset of: xdimred, ytrans, coloc, stat, xsea, perm, pooled, mc.
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


def _get_df_string(kind, xdimred=None, ytrans=None, method=None, stat=None, xsea=False,
                   perm=None, pooled_p=False, mc=None, engine=None, pooled=False, signed=False):

    if kind=="ytrans":
        df_str = f"ytrans-{ytrans}"

    elif kind=="xdimred":
        df_str = f"xdimred-{xdimred}"

    elif kind=="coloc":
        if (method is not None) & (stat is not None):
            df_str = f"xdimred-{xdimred}_ytrans-{ytrans}_coloc-{method}_stat-{stat}_xsea-{xsea}"
        else:
            raise ValueError("Provide both method and stat!")

    elif kind=="influence":
        if (method is not None) & (stat is not None) & (engine is not None):
            df_str = (f"xdimred-{xdimred}_ytrans-{ytrans}_infl-{method}_stat-{stat}_"
                      f"xsea-{xsea}_engine-{engine}_signed-{signed}")
            if pooled:
                df_str += f"_pooled-{pooled}"
        else:
            raise ValueError("Provide method, stat, and engine!")

    elif kind=="contribution":
        if (method is not None) & (stat is not None):
            df_str = f"xdimred-{xdimred}_ytrans-{ytrans}_contrib-{method}_stat-{stat}_xsea-{xsea}"
            if pooled:
                df_str += f"_pooled-{pooled}"
        else:
            raise ValueError("Provide method and stat!")

    elif kind=="null":
        if (method is not None) & (perm is not None):
            if "sets" in perm:
                xsea = True
            df_str = f"xdimred-{xdimred}_ytrans-{ytrans}_coloc-{method}_xsea-{xsea}_perm-{perm}"
            if pooled_p:
                df_str += f"_pooled-{pooled_p}"
        else:
            raise ValueError("Provide both method and perm!")

    elif kind=="p":
        if (method is not None) & (stat is not None) & (perm is not None):
            if "sets" in perm:
                xsea = True
            df_str = (f"xdimred-{xdimred}_ytrans-{ytrans}_coloc-{method}_stat-{stat}_xsea-{xsea}_"
                      f"perm-{perm}")
            if pooled_p:
                df_str += f"_pooled-{pooled_p}"
            df_str += f"_mc-{mc}"
        else:
            raise ValueError("Provide method, stat, and perm!")

    elif kind=="z":
        if (method is not None) & (stat is not None) & (perm is not None):
            if "sets" in perm:
                xsea = True
            df_str = (f"xdimred-{xdimred}_ytrans-{ytrans}_coloc-{method}_stat-{stat}_xsea-{xsea}_"
                      f"perm-{perm}")
            if pooled_p:
                df_str += f"_pooled-{pooled_p}"
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
    """Build a combined 1D NaN mask across one or more arrays sharing dim-0 length.

    Plain numpy (no numba). For any array with more than 1 dimension, NaN is
    reduced across `axis=1` (any NaN in the row marks it). This is the
    pre-masking step used before calling the NaN-intolerant numba functions
    in ``stats/coloc.py`` (``pearson``, ``mlr``, etc.): callers in
    ``core/colocalize.py``/``core/region_influence.py`` build
    ``parcel_mask = ~nan_detector(X_T, y)`` and index with it before passing
    data into those functions.

    Parameters
    ----------
    *arrays : np.ndarray
        One or more arrays with the same length along axis 0.

    Returns
    -------
    np.ndarray of bool
        Shape ``(n,)``; True where any input array has NaN at that position.
    """
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


def dedupe_rows(X):
    """Deduplicate rows of a 2D array, NaN-safe.

    Uses pandas (``DataFrame.duplicated``/``groupby(dropna=False)``), not
    ``np.unique(axis=0, return_inverse=True)`` -- the latter does not merge
    NaN-containing duplicate rows (each NaN-containing row is kept as a
    separate "unique" row even if otherwise identical), which matters here
    since dropped/background parcels are commonly NaN throughout NiSpace.

    Parameters
    ----------
    X : np.ndarray
        2D array, shape ``(n_rows, n_cols)``.

    Returns
    -------
    X_unique : np.ndarray
        Deduplicated rows of `X`, in order of first appearance.
    inverse_idx : np.ndarray of int
        Shape ``(n_rows,)`` such that ``X_unique[inverse_idx]`` reconstructs `X`.
    """
    df = pd.DataFrame(X)
    # with sort=False, group codes are assigned in order of first appearance, so
    # group k IS the row index into X_unique below -- inverse_idx falls out directly
    inverse_idx = df.groupby(list(df.columns), dropna=False, sort=False).ngroup().to_numpy()
    X_unique = X[~df.duplicated(keep="first").to_numpy()]
    return X_unique, inverse_idx


def remove_nan(data, which="col"):
    """Drop rows or columns containing NaN from an array/DataFrame/Series.

    Plain numpy/pandas (no numba). Not currently called anywhere in NiSpace
    (dead code) — kept as a public utility.

    Parameters
    ----------
    data : np.ndarray, pd.DataFrame, or pd.Series
        Input data.
    which : {"col", "row"}, default="col"
        Which axis to drop entries containing NaN along.

    Returns
    -------
    Same type as `data`, with NaN-containing rows/columns removed.
    """
    if isinstance(data, np.ndarray):
        axis = 0 if which=="col" else 1 # 0 drops cols, 1 drops rows
        data = data[np.isnan(data.any(axis=axis))]
    
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        axis = 1 if which=="col" else 0 # 1 drops cols, 0 drops rows
        data = data.dropna(axis=axis)
        
    return data


def fill_nan(data, idx, idx_label=None, which="col"):
    """Re-insert all-NaN rows/columns at given positions (inverse of dropping them).

    Plain numpy/pandas (no numba). Used in ``api.py``'s dimensionality-reduction
    path (`X_reduction`) to restore parcels/maps that were dropped (as
    all-NaN) before PCA/ICA/FA, so the reduced output keeps the original
    column/row layout with NaN at the dropped positions.

    Parameters
    ----------
    data : array-like or pd.DataFrame
        Input data to insert NaN rows/columns into.
    idx : array-like of int
        Positions (in the *output*, post-insertion, array) at which to
        insert a new all-NaN row/column.
    idx_label : list of str, optional
        For DataFrame input, labels to assign to the newly inserted
        index/column entries. Defaults to ``"nan"`` for each.
    which : str, default="col"
        Whether to insert rows (any string starting with ``"row"``) or
        columns (starting with ``"col"``).

    Returns
    -------
    np.ndarray or pd.DataFrame
        `data` with NaN rows/columns inserted at `idx`. Integer input is
        upcast to float (NaN requires a float dtype).
    """
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
    """Format keyword arguments as an aligned two-row key/value log table.

    Plain string formatting (no numba/numpy). Used throughout ``api.py`` to
    build ``lgr.info(...)`` summary messages (e.g. reduction/transform
    settings actually used for a given call).

    Parameters
    ----------
    **kwargs
        Arbitrary label=value pairs to display.

    Returns
    -------
    str
        Two-line string: uppercased, pipe-delimited labels on the first
        line, corresponding values aligned beneath on the second. Empty
        string if no kwargs are given.
    """
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
    """(Weighted) mean/median aggregation of a DataFrame grouped by a "set" index level.

    Plain pandas/numpy (no numba). Used in ``api.py``'s set-aggregation path
    (aggregating multiple maps within a named "set" of X reference maps into
    a single representative map). NaN is excluded via ``np.ma``/``nanmedian``.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a "set" level in its index for grouping to take effect,
        and (if `weighted=True`) a "weight" level with per-row weights.
    mean_by_set : bool, default=True
        Group by the "set" index level. Silently disabled (treated as False)
        if `df.index` has no "set" level.
    weighted : bool, default=True
        Weight rows by the "weight" index level. Silently disabled if `df.index`
        has no "weight" level.
    mean_median : {"mean", "median"}, default="mean"
        Aggregation statistic. Weighted median is computed via a
        value-repetition workaround (``np.repeat`` by integer weight), not a
        true weighted-median algorithm.

    Returns
    -------
    pd.DataFrame
        One row per set (or a single row if `mean_by_set=False`), indexed by
        a "map" level.

    Raises
    ------
    ValueError
        If `df` is not a pandas DataFrame.
    """
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
    """Lowercase a string, or each string element of a list (non-strings passed through).

    Plain Python (no numpy/numba). Not currently called anywhere in NiSpace
    (dead code) — everywhere lowercasing is actually needed, `api.py` uses
    Python's builtin ``str.lower()`` directly instead.

    Parameters
    ----------
    str_list : str or list
        Input string, or list of strings/other objects.

    Returns
    -------
    str or list
        Lowercased string, or list with string elements lowercased.
    """
    if isinstance(str_list, str):
        return str_list.lower()
    elif isinstance(str_list, list):
        return [s.lower() if isinstance(s, str) else s for s in str_list]


def get_background_value(img, border_size=2):
    """Auto-detect a volumetric image's background value from its border voxels.

    Adapted from ``nilearn.masking.compute_background_mask``. Only
    implemented for 3D/volumetric input — for anything else (e.g. surface
    data), this silently returns ``None`` rather than raising. Used by
    :class:`~nispace.parcellate.Parcellater` to resolve ``background_value="auto"``
    (see [[project_parcellation_bg_handling]]).

    Parameters
    ----------
    img : image-like
        Input accepted by ``neuromaps.images.load_data``.
    border_size : int, default=2
        Border thickness (in voxels) sampled to estimate the background.

    Returns
    -------
    float, np.nan, or None
        The detected background value (border median), ``np.nan`` if any
        border voxel is NaN, or ``None`` if `img` is not 3D.
    """
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
    """Broadcast a per-parcel value vector back into a full volume array.

    Numba-compiled; requires plain ``np.ndarray`` inputs. Inverse operation
    of parcel-mean extraction (e.g. :func:`vol_to_vect_arr`): each voxel is
    assigned its parcel's value via a label lookup. No NaN-specific handling
    (a NaN in `vect` is simply assigned as-is). Used in ``plotting.py`` to
    reconstruct volume/vertex arrays for rendering.

    Parameters
    ----------
    vect : np.ndarray
        Per-parcel values, shape ``(n_parcels,)``, in the same order as
        `parc_idc`.
    parc_arr : np.ndarray
        Parcellation label array (any shape; flattened internally).
    parc_idc : np.ndarray
        Parcel label values corresponding to each entry of `vect`.
    bg_value : scalar, default=0
        Value assigned to voxels not belonging to any parcel in `parc_idc`.

    Returns
    -------
    np.ndarray
        Same shape as `parc_arr`, with each voxel set to its parcel's value
        from `vect` (or `bg_value`).

    Raises
    ------
    ValueError
        If `vect` and `parc_idc` have different lengths.
    """
    parc_arr_1d = parc_arr.flatten().astype(vect.dtype)
    vect_arr_1d = np.full_like(parc_arr_1d, bg_value, dtype=vect.dtype)
    parc_idc = parc_idc.astype(vect.dtype)
    if len(parc_idc) != len(vect):
        raise ValueError(f"vect ({vect.shape}) and parc_idc ({parc_idc.shape}) don't match.")
    for i, idx in enumerate(parc_idc):
        vect_arr_1d[parc_arr_1d==idx] = vect[i]
    return vect_arr_1d.reshape(parc_arr.shape)

def _resolve_bg_array(bg_spec, auto_value=np.nan):
    """Resolve a background_value spec to a sorted float64 array for vol_to_vect_arr.

    Parameters
    ----------
    bg_spec : list
        Already-normalised list (str 'auto', None, or float entries).
    auto_value : float
        Pre-computed auto-detected background value; used wherever 'auto'/None
        appear in bg_spec. Ignored (not inserted) when NaN.
    """
    values = set()
    for item in bg_spec:
        if item in (None, "auto"):
            if not np.isnan(auto_value):
                values.add(float(auto_value))
        else:
            values.add(float(item))
    return np.array(sorted(values), dtype=np.float64)


@njit
def vol_to_vect_arr(vol_arr, parc_arr, parc_idc, bg_values):
    """Aggregate vol_arr into parcel means, excluding NaN and any values in bg_values.

    Parameters
    ----------
    bg_values : np.ndarray of float64
        Values to exclude in addition to NaN. Pass an empty array for NaN-only
        exclusion (equivalent to nanmean).
    """
    vol_arr2d = vol_arr.flatten()
    parc_arr2d = parc_arr.flatten().astype(vol_arr.dtype)
    parc_idc = parc_idc.astype(vol_arr.dtype)
    vect = np.zeros(len(parc_idc), dtype=vol_arr.dtype)
    for i, idx in enumerate(parc_idc):
        in_parcel = parc_arr2d == idx
        not_nan = ~np.isnan(vol_arr2d)
        idc = in_parcel & not_nan
        for bg in bg_values:
            idc = idc & (vol_arr2d != bg)
        vals = vol_arr2d[idc]
        vect[i] = vals.mean() if len(vals) > 0 else np.nan
    return vect
    

def parc_vect_to_vol(vect, parc):
    """Convert a per-parcel value vector into a full volumetric NIfTI image.

    Thin nilearn/nibabel wrapper around :func:`vect_to_vol_arr`: loads `parc`,
    determines its (nonzero, sorted) parcel labels, broadcasts `vect` back
    onto the voxel grid, and returns a proper ``Nifti1Image``. Unlike most
    functions in this module, this one is a real end-user-facing utility
    (used directly in ``docs/nb_examples/drug_challenge.ipynb``), not just an
    internal helper.

    Parameters
    ----------
    vect : array-like
        1D per-parcel values, length must match the number of nonzero labels
        in `parc`.
    parc : image-like
        Parcellation image, loaded via ``nilearn.image.load_img``.

    Returns
    -------
    nibabel.Nifti1Image
        Volume with each parcel's voxels set to its value from `vect`.

    Raises
    ------
    ValueError
        If `vect` is not 1D or list-like.
    """
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
    """Reassign a surface parcellation's integer labels to a new label sequence.

    Plain numpy/nibabel (no numba). Used in ``core/parcellation.py`` to
    harmonize labels when combining a cortex surface parcellation with a
    subcortex volume parcellation into one combined ("+") parcellation.

    Parameters
    ----------
    parc : nib.GiftiImage
        Input surface parcellation (single hemisphere).
    new_labels : array-like of int, optional
        New label values, one per existing (nonzero) parcel, in the order of
        ``np.unique`` of the existing labels. Defaults to ``1..n_parcels``.

    Returns
    -------
    nib.GiftiImage
        Deep copy of `parc` with labels reassigned.

    Raises
    ------
    ValueError
        If `parc` is not a GiftiImage, if `new_labels` is not 1D array-like,
        or if its length doesn't match the number of existing parcels.
    """
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
    """Reassign a volumetric parcellation's integer labels to a new label sequence.

    Plain numpy/nibabel (no numba). Volumetric counterpart of
    :func:`relabel_gifti_parc`; used in ``core/parcellation.py`` for the same
    combined cortex+subcortex label-harmonization step.

    Parameters
    ----------
    parc : image-like
        Input volumetric parcellation, loaded via ``neuromaps.images.load_nifti``.
    new_order : array-like of int, optional
        Existing label values to keep, in the order they should be relabeled.
        Defaults to all nonzero labels present in `parc`, sorted.
    new_labels : array-like of int, optional
        New label values corresponding to `new_order`. Defaults to
        ``1..len(new_order)``.
    dtype : optional
        Output label dtype. Defaults to the input image's dtype.

    Returns
    -------
    nibabel.Nifti1Image
        Volume with labels reassigned.

    Raises
    ------
    ValueError
        If `new_order`/`new_labels` lengths don't match the number of
        existing parcels, or if `new_order` isn't a subset of the labels
        actually present in `parc`.
    """
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
    """Combine two or more NIfTI parcellation images into one labeled volume.

    Plain numpy/nibabel (no numba). Only implemented for ``nib.Nifti1Image``
    input. Used in ``datasets.py`` and ``core/parcellation.py`` to merge a
    cortex and a subcortex parcellation into one combined image — always via
    the fast 2-image path (`quick=True`) in practice.

    Parameters
    ----------
    parcellations : list of nib.Nifti1Image
        Images to merge; all must be the same type and shape.
    labels : list, optional
        Per-parcellation label sequences to assign (only used by the slow,
        `quick=False` path). Defaults to each parcellation's own existing
        label values.
    quick : bool, default=False
        If True, take the first parcellation's labels as-is and offset the
        second parcellation's nonzero labels by the first's max label (only
        supports exactly 2 parcellations; returns just the merged image). If
        False, sequentially relabels every parcel across all parcellations
        starting at 1 (supports any number of parcellations; also returns a
        `labels_merged` Series mapping new label -> original label).

    Returns
    -------
    nibabel.Nifti1Image, or (nibabel.Nifti1Image, pd.Series) if `quick=False`
        Merged parcellation image (and, for the slow path, a Series mapping
        each new integer label to its original source label).

    Raises
    ------
    ValueError
        If `parcellations`/`labels` aren't lists, have mismatched lengths, or
        the parcellations aren't uniformly typed/shaped.
    NotImplementedError
        If `parcellations[0]` is not a ``nib.Nifti1Image``.
    """
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
    """Correlate a volumetric image with its own left-right mirror image.

    Plain numpy/nibabel (no numba). Not currently called anywhere in NiSpace
    (dead code) — a sibling of the equally-dead
    :func:`nispace.nulls.correlate_hemis_parc` (voxel-level here vs.
    parcel-level there). Kept as a public utility, e.g. for sanity-checking
    whole-brain left-right symmetry of a volume before/without parcellation.

    Parameters
    ----------
    img : str, Path, nib.Nifti1Image, np.ndarray, or size-2 tuple/list
        Volumetric image (single 3D array/image, mirrored internally via
        `mask`/x-axis flip) or a size-2 tuple/list of two pre-split
        hemisphere arrays/GiftiImages (used as-is, no mirroring).
    mask : str, Path, nib.Nifti1Image, or np.ndarray, optional
        Boolean/loadable mask restricting which voxels enter the
        correlation. Only used for single-image input; defaults to
        excluding NaN and exact-zero voxels.

    Returns
    -------
    float
        Pearson correlation between the (masked, flattened) original and
        its left-right-flipped counterpart.

    Raises
    ------
    ValueError
        If `img` is not one of the supported types/shapes.
    """
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
    """Mirror/flip a volumetric image across the (affine-derived) x=0 hemisphere boundary.

    Plain numpy/nibabel (no numba). Not currently called anywhere in NiSpace
    (dead code) — kept as a public utility. `match_r`/`mask` parameters are
    accepted but currently unused by the implementation.

    Parameters
    ----------
    img : np.ndarray or image-like
        Input volume. If an array, `affine` must be provided.
    affine : np.ndarray, optional
        4x4 affine; required when `img` is an array, otherwise taken from
        `img` itself.
    direction : str, default="left_to_right"
        One of ``"left_to_right"``/``"right_to_left"`` (overwrite one
        hemisphere with the mirrored other), ``"average"``/``"bilateral"``
        (average both mirrored hemispheres), ``"switch"`` (swap
        hemispheres), or ``"drop_left"``/``"drop_right"`` (zero out one
        hemisphere).
    match_r : bool, optional
        Currently unused.
    mask : optional
        Currently unused.

    Returns
    -------
    np.ndarray or nibabel.Nifti1Image
        Mirrored volume, same type as the input (array in, array out).

    Raises
    ------
    ValueError
        If `img` is an array without `affine`, if `img` is not 3D, or if
        `direction` is not one of the supported modes.
    """
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
    """Mirror/flip a bilateral surface image pair across hemispheres.

    Plain numpy/nibabel (no numba). Surface counterpart of :func:`mirror_nifti`,
    with fewer supported modes (no per-hemisphere drop/switch). Not currently
    called anywhere in NiSpace (dead code) — kept as a public utility.
    `match_r`/`mask` parameters are accepted but currently unused.

    Parameters
    ----------
    img : tuple of (nib.GiftiImage, nib.GiftiImage) or (np.ndarray, np.ndarray)
        (left, right) hemisphere data.
    direction : str, default="left_to_right"
        ``"left_to_right"``: copy LH data to RH. ``"right_to_left"``: copy RH
        data to LH. ``"average"``/``"bilateral"``: average both hemispheres,
        assigned to both outputs.
    match_r : bool, optional
        Currently unused.
    mask : optional
        Currently unused.

    Returns
    -------
    tuple
        (left, right) mirrored pair, same type as the input (arrays in,
        arrays out; GiftiImages in, GiftiImages out).

    Raises
    ------
    ValueError
        If `img` is not a tuple, or its elements are neither ndarrays nor
        GiftiImages.
    """
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
    
    # standardize input (population std, ddof=0: standardizing the given vector
    # to itself, not estimating a population parameter from a sample -- hand-rolled
    # since numba's np.nanstd doesn't accept a ddof kwarg)
    mu = np.nanmean(data_1d)
    sigma = np.sqrt(np.nanmean((data_1d - mu) ** 2))
    data_1d = (data_1d - mu) / sigma

    # generate random noise with same length as input
    if seed is not None:
        np.random.seed(seed)
    epsilon = np.random.normal(0, 1, len(data_1d))
    eps_mu = np.mean(epsilon)
    eps_sigma = np.sqrt(np.mean((epsilon - eps_mu) ** 2))
    epsilon = (epsilon - eps_mu) / eps_sigma
    
    # correlated vector using the formula:
    # output = ρ * input + √(1-ρ²) * ε
    output = correlation * data_1d + np.sqrt(1 - correlation**2) * epsilon
    
    # rescale to original scale
    output = output * sigma + mu
    
    return output

def correlated_vector(data_1d, correlation=1, seed=None):
    """Generate a vector correlated with `data_1d` at a target Pearson r.

    Public wrapper around the numba-jitted :func:`_corr_vector`
    (``output = correlation * standardized_input + sqrt(1-correlation**2) * noise``,
    rescaled back to the input's mean/SD). Not currently called anywhere in
    NiSpace (dead code) — kept as a public utility, e.g. for generating
    synthetic test data with a known ground-truth correlation.

    Parameters
    ----------
    data_1d : array-like
        1D input vector.
    correlation : float, default=1
        Target Pearson correlation with `data_1d`. If exactly 1, `data_1d`
        is returned unchanged (no noise added).
    seed : int, optional
        Random seed for the noise draw.

    Returns
    -------
    np.ndarray
        1D vector of the same length as `data_1d`, correlated with it at
        (approximately) `correlation`.
    """
    # if correlation is 1, return the original vector
    if correlation == 1:
        return data_1d.copy()

    # to array
    data_1d = np.array(data_1d).squeeze()

    # get correlated vector
    output = _corr_vector(data_1d, correlation=correlation, seed=seed)

    return output


def apply_transform(img, mni_from=None, mni_to=None, transform=None, order=3, res=None):
    """Apply an ANTs/ITK composite transform between MNI152 template spaces.

    Two operation modes:

    * **MNI mode**: pass ``mni_from`` and ``mni_to``. The required templateflow
      ``.h5`` is fetched automatically. If both spaces are equal the input image
      is returned unchanged.
    * **Transform mode**: pass a path (str / Path) to an ANTs composite ``.h5``
      file. ``mni_from`` / ``mni_to`` are ignored; the target grid is inferred
      from the filename (templateflow ``tpl-`` convention) when possible, or
      falls back to the embedded displacement-field grid.

    The output voxel resolution is controlled by ``res``. When ``res=None``
    the voxel size of the input image is used (rounded to 1, 2, or 3 mm).

    Parameters
    ----------
    img : str, Path, or nibabel.SpatialImage
        Input image to resample.
    mni_from : str, optional
        Source MNI space (e.g. ``'MNI152NLin6Asym'``, ``'MNI152NLin2009cAsym'``).
    mni_to : str, optional
        Target MNI space.
    transform : str or Path, optional
        Path to an ANTs/ITK composite ``.h5`` transform file.
    order : int
        Spline interpolation order passed to ``nitransforms.resampling.apply``
        (0 = nearest neighbour, 1 = trilinear, 3 = cubic spline). Default 3.
    res : int, str, or None
        Output resolution in mm. Accepted forms: ``1``, ``2``, ``3`` or
        ``"1mm"``, ``"2mm"``, ``"3mm"``. If None, inferred from the input
        image's voxel size (rounded and clamped to 1–3 mm).

    Returns
    -------
    nibabel.Nifti1Image
        Resampled image in the target space.
    """
    try:
        from nitransforms.io.itk import ITKCompositeH5
        from nitransforms import TransformChain, linear, DenseFieldTransform
        from nitransforms.resampling import apply as _nt_apply
    except ImportError as exc:
        raise ImportError(
            "apply_transform requires 'nitransforms'. "
            "Install with: pip install nitransforms"
        ) from exc

    import re
    import warnings

    # Load image if path given
    if isinstance(img, (str, Path)):
        img = nib.load(str(img))

    if transform is not None:
        # Transform mode: load h5 directly
        h5_path = str(transform)
        # Try to parse target space from templateflow filename (tpl-XXX_from-...)
        target_space = None
        m = re.match(r"tpl-([^_]+)_", Path(h5_path).name)
        if m:
            target_space = m.group(1)
    else:
        # MNI mode: validate and fetch transform from templateflow
        if mni_from is None or mni_to is None:
            raise ValueError("Provide either 'transform' or both 'mni_from' and 'mni_to'.")
        if mni_from == mni_to:
            return img
        import templateflow.api as tflow
        h5_path = tflow.get(**{"template": mni_to, "from": mni_from, "extension": "h5"})
        if not h5_path:
            raise ValueError(
                f"No templateflow transform found from '{mni_from}' to '{mni_to}'."
            )
        h5_path = str(h5_path)
        target_space = mni_to

    # Determine output resolution
    _available_res = [1, 2, 3]
    vox_size = float(np.min(np.abs(img.header.get_zooms()[:3])))
    if res is None:
        res = min(_available_res, key=lambda r: abs(r - vox_size))
    else:
        if isinstance(res, str):
            res = int(res.lower().replace("mm", ""))
        if res not in _available_res:
            raise ValueError(f"res={res} not supported. Choose from {_available_res}.")
        if res < round(vox_size):
            import warnings as _warnings
            _warnings.warn(
                f"Requested output resolution ({res}mm) is finer than the input image "
                f"voxel size ({vox_size:.1f}mm). The output grid will be denser but "
                f"effective resolution remains limited by the input.",
                UserWarning, stacklevel=2,
            )

    # Build reference image from nispace template (defines output grid)
    # Deferred import avoids circular dependency (datasets.py imports utils.py)
    reference = None
    if target_space is not None:
        try:
            from nispace.datasets import fetch_template
            tpl_path = fetch_template(target_space, res=f"{res}mm", desc="brainmask", verbose=False)
            reference = nib.load(str(tpl_path))
        except Exception:
            pass  # fall through to warp-grid fallback below

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="nitransforms")

        # Load composite h5: part 0 = affine, part 1 = displacement field
        parts = ITKCompositeH5.from_filename(h5_path)
        chain = TransformChain(
            [linear.Affine(parts[0].to_ras()), DenseFieldTransform(parts[1], is_deltas=True)]
        )

        if reference is None:
            # Fallback: use the displacement field's own spatial grid
            warp_nii = parts[1]
            reference = nib.Nifti1Image(
                np.zeros(warp_nii.shape[:3], dtype=np.uint8), warp_nii.affine, warp_nii.header
            )

        return _nt_apply(chain, img, reference=reference, order=order)


def apply_mni_mask(img, mask_desc, space, resampling_target="data"):
    """Apply a named MNI mask template to a NIfTI image, setting masked-out voxels to NaN.

    Parameters
    ----------
    img : nib.Nifti1Image
        Stat map to mask.
    mask_desc : str
        Template descriptor, e.g. ``"cortexmask"``, ``"subcortexmask"``, ``"brainmask"``.
    space : str
        MNI space key, e.g. ``"MNI152NLin2009cAsym"``.
    resampling_target : {"data", "mask"}
        ``"data"`` (default): resample mask to *img* grid (preserves img resolution).
        ``"mask"``: resample *img* to mask grid.

    Returns
    -------
    nib.Nifti1Image
        Image with voxels outside the mask set to NaN.
    """
    from nilearn.image import resample_to_img
    # Deferred import avoids circular dependency (datasets.py imports utils.py)
    from nispace.datasets import fetch_template
    mask_path = fetch_template(space, desc=mask_desc, verbose=False)
    mask_img = nib.load(str(mask_path))
    if resampling_target == "data":
        mask_r = resample_to_img(mask_img, img, interpolation="nearest",
                                  force_resample=True, copy_header=True)
        mask_arr = mask_r.get_fdata() > 0
        arr = img.get_fdata().copy()
        arr[~mask_arr] = np.nan
        return image.new_img_like(img, arr, copy_header=True)
    else:
        img_r = resample_to_img(img, mask_img, interpolation="continuous",
                                 force_resample=True, copy_header=True)
        arr = img_r.get_fdata().copy()
        arr[mask_img.get_fdata() == 0] = np.nan
        return image.new_img_like(img_r, arr, copy_header=True)
