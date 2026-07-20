import os
import warnings
import json
import dill, pickle, blosc, gzip
from pathlib import Path
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

import nibabel as nib
from neuromaps import images 

import logging
lgr = logging.getLogger(__name__)
from .utils.utils import set_log
from .parcellate import Parcellater

# ==================================================================================================
# DEPRECATION MESSAGE STRINGS
# ==================================================================================================

_DEPR_IGNORE_BACKGROUND_DATA = (
    "'ignore_background_data' is deprecated and will be removed in the first non-dev "
    "release. Use 'background_value' instead: pass background_value=False to disable "
    "background exclusion (equivalent to ignore_background_data=False), or a scalar/"
    "list/'auto' to enable it (equivalent to ignore_background_data=True)."
)
_DEPR_DROP_BACKGROUND_PARCELS = (
    "'drop_background_parcels' is deprecated and will be removed in the first non-dev "
    "release. Use 'report_background_parcels' instead (same meaning, now named "
    "consistently across Parcellater and parcellate_data())."
)


def parcellate_data(data,
                    data_labels=None,
                    data_space=None,
                    parcellation=None,
                    parc_labels=None,
                    parc_space=None,
                    parc_hemi=None,
                    resampling_target="data",
                    background_value="auto",
                    report_background_parcels=False,
                    min_num_valid_datapoints=None,
                    min_fraction_valid_datapoints=None,
                    return_parc=False,
                    dtype=None,
                    n_proc=1,
                    verbose=True,
                    ignore_zero_division_warning=True,
                    ignore_background_data=None,
                    drop_background_parcels=None):
    """
    Parcellates given imaging data using a specified parcellation.

    Parameters
    ----------
    data : list, dict, pd.DataFrame, pd.Series, or np.ndarray
        The imaging data to be parcellated. Lists/dicts are treated as paths or
        loaded nifti/gifti images (dict values become the data, dict keys become
        `data_labels`); DataFrames/Series/ndarrays are treated as already-parcellated
        data of shape (n_files, n_parcels).
    data_labels : list, optional
        Labels for the input data. If not given, derived from file basenames (list
        input) or from the DataFrame/Series index/name (already-parcellated input).
    data_space : str
        The space in which the input data is defined.
    parcellation : str, os.PathLike, nib.Nifti1Image, nib.GiftiImage, tuple, or Parcellation
        The parcellation image or surfaces, where each region is identified by a
        unique integer ID. A :class:`~nispace.core.parcellation.Parcellation` object
        is also accepted; `parc_labels`/`parc_hemi`/`parc_space` are then taken from
        its active space unless explicitly overridden. Required (non-None) when
        `data` is a list/dict.
    parc_labels : list
        Labels for the parcellation regions.
    parc_space : str
        The space in which the parcellation is defined.
    parc_hemi : list of str
        Hemispheres to consider for parcellation, e.g., ["L", "R"].
    resampling_target : {'data', 'parcellation'}
        Specifies which image gives the final shape/size.
    background_value : float, list, set, array, 'auto', or False
        Value(s) to treat as background, or ``False`` to disable background
        exclusion entirely (background/zero is then treated as real data --
        never masked, never triggers the empty-mean-to-NaN path; NaN is
        still always excluded regardless of this parameter). Accepts:

        - ``'auto'`` (default): auto-detect from border voxels (volumetric)
          or medial wall median (surface), combined with exact ``0.0`` --
          equivalent to ``['auto', 0.0]``.
        - float (e.g. ``0.0``): exclude that specific value only.
        - list/set/array: any combination of floats and the ``'auto'``/
          ``None`` sentinel.
        - ``False``: disable background exclusion entirely.
    report_background_parcels : bool
        Whether to explicitly flag (and log) parcels whose raw data was
        entirely background -- every non-NaN raw voxel/vertex in the parcel
        matches `background_value`. Such parcels are already NaN via
        empty-mean aggregation regardless of this flag, so it only affects
        whether they're recorded/logged, not the returned values. Always a
        no-op when `background_value=False`, since in that mode
        `background_value` may label real, meaningful data (e.g. binary Y
        cluster-coverage maps, where an all-zero parcel is a genuine
        0%-overlap result, not missing background) that must never be
        flagged here. Default: False
    min_num_valid_datapoints : int, optional
        Minimum number of valid datapoints required per parcel.
    min_fraction_valid_datapoints : float, optional
        Minimum fraction of valid datapoints required per parcel.
    return_parc : bool, default False
        If True, also return the loaded parcellation image (nifti/gifti/tuple).
    dtype : data-type, optional
        Desired data type of the output.
    n_proc : int, default 1
        Number of processors to use for parallel processing (list/dict input only).
    verbose : bool, default True
        Whether to print progress/info messages.
    ignore_zero_division_warning : bool, default True
        Whether to suppress numpy's "invalid value encountered in divide" warning
        raised when a parcel's mean is computed from zero valid datapoints.
    ignore_background_data : bool, optional
        Deprecated. Use `background_value` instead -- pass
        ``background_value=False`` for what used to be
        ``ignore_background_data=False``. Default: None (not set)
    drop_background_parcels : bool, optional
        Deprecated. Use `report_background_parcels` instead (same meaning).
        Default: None (not set)

    Returns
    -------
    pd.DataFrame
        Parcellated data of shape (n_files, n_parcels).
    pd.DataFrame, nib.Nifti1Image or nib.GiftiImage or tuple
        If `return_parc=True`, also returns the loaded parcellation image.

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

    # deprecation shims -- both legacy params are forwarded through to
    # Parcellater.transform() as-is (it's the single source of truth for
    # old-vs-new precedence); only drop_background_parcels needs resolving
    # here since parcellate_data() invented that name itself (Parcellater's
    # own name for the same flag is report_background_parcels).
    if ignore_background_data is not None:
        lgr.warning(_DEPR_IGNORE_BACKGROUND_DATA)
    if drop_background_parcels is not None:
        lgr.warning(_DEPR_DROP_BACKGROUND_PARCELS)
        report_background_parcels = drop_background_parcels

    # unpack Parcellation object into flat args (lazy import avoids circular dependency)
    from .core.parcellation import Parcellation
    if isinstance(parcellation, Parcellation):
        # bilateral surface parcellating is not yet supported
        if getattr(parcellation, "_bilateral", False) and parcellation._space is not None:
            active_space = parcellation._space or ""
            if "mni" not in active_space.lower():
                raise NotImplementedError(
                    "Surface parcellating with a bilateral Parcellation is not yet supported. "
                    "Use an MNI space or fetch pre-parcellated data via fetch_reference(). # TODO"
                )
        parc_labels = parc_labels if parc_labels is not None else parcellation._labels
        parc_hemi   = parc_hemi   if parc_hemi   is not None else parcellation._hemi
        parc_space  = parc_space  if parc_space  is not None else parcellation._space
        # _image_obj requires an active space; if none is set (pre-parcellated data path),
        # set to None — list inputs will raise their own error, DataFrame inputs don't need it
        parcellation = parcellation._image_obj if parcellation._space is not None else None

    ## put data into list
    if isinstance(data, Path):
        data = str(data)
    if isinstance(data, (str, tuple, nib.Nifti1Image, nib.GiftiImage)):
        data = [data]
        
    ## case list
    if isinstance(data, (list, dict)):
        if isinstance(data, dict):
            lgr.info("Input type: dict, assuming (img_name, img) pairs for imaging data.")
            data_labels = list(data.keys()) if data_labels is None else data_labels
            data = list(data.values())
        else:
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
        if ("mni" in data_space.lower()) & \
            ("mni" not in parc_space.lower()) & \
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
            space="mni152" if "mni" in parc_space.lower() else parc_space,
            resampling_target=resampling_target,
            hemi=parc_hemi
        ).fit()
        
        # data extraction function
        def extract_data(file):
            
            # apply parcellater
            kwargs = dict(
                data=file,
                space="mni152" if "mni" in data_space.lower() else data_space,
                hemi=parc_hemi,
                background_value=background_value,
                fill_dropped=True,
                report_background_parcels=report_background_parcels,
                min_num_valid_datapoints=min_num_valid_datapoints,
                min_fraction_valid_datapoints=min_fraction_valid_datapoints,
                ignore_background_data=ignore_background_data,
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
        lgr.info(
            f"Background (bg) handling: background_value={background_value!r}"
            + (f", ignore_background_data={ignore_background_data!r} (deprecated override)"
               if ignore_background_data is not None else "")
            + f"; reporting bg-only parcels: {report_background_parcels}"
        )
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
        if report_background_parcels:
            lgr.info(f"Combined across images, {len(nan_parcels['bg'])} parcel(s) had only background "
                     f"intensity (already nan via empty-mean aggregation) "
                     f"({[int(i) for i in nan_parcels['bg']]}).")
        
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
        if isinstance(data_labels, str):
            data_labels = [data_labels]
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
    """
    Load a JSON file, or pass through a dict-like object as a dict.

    Parameters
    ----------
    json_path : str, os.PathLike, or dict-like
        Path to a JSON file, or an already-loaded dict-like object.

    Returns
    -------
    dict
    """
    if isinstance(json_path, (str, Path)):
        with open(json_path) as f:
            json_dict = json.load(f)
    else:
        try:
            json_dict = dict(json_path)
        except (TypeError, ValueError):
            lgr.critical_raise("Provide path to json-like file or dict-like object!", ValueError)
    return json_dict


def write_json(json_dict, json_path):
    """
    Write a dict to a JSON file (pretty-printed, indent=4).

    Parameters
    ----------
    json_dict : dict
        Dictionary to serialize.
    json_path : str or os.PathLike
        Destination file path.

    Returns
    -------
    Path
        The resolved destination path.
    """
    if isinstance(json_path, (str, Path)):
        json_path = Path(json_path)
        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=4)
    else:
        lgr.critical_raise(
            f"Provide path-like object for argument 'json_path', not {type(json_path)}!",
            ValueError
        )
    return json_path


def load_img(img, override_file_format=False):
    """
    Load one or more nifti/gifti images, passing through already-loaded objects.

    Parameters
    ----------
    img : str, os.PathLike, nib.Nifti1Image, nib.GiftiImage, list, or tuple
        A single image (path or loaded object), or a list/tuple of up to 2 such
        elements (e.g. left/right hemisphere gifti paths). Also accepts ``.curv``
        FreeSurfer morphometry files, wrapped into a `nib.GiftiImage`.
    override_file_format : {False, '.nii', '.nii.gz', '.gii', '.gii.gz'}, default False
        If given, rename+reinterpret path inputs with an unrecognized extension as
        this format before loading, instead of raising.

    Returns
    -------
    nib.Nifti1Image, nib.GiftiImage, or tuple
        A single loaded image, or a tuple of loaded images if `img` had 2 elements.

    Raises
    ------
    ValueError
        If `override_file_format` is invalid, an element's type/extension is
        unsupported, or `img` is not a path/list/tuple/image object.
    """
    # check override_file_format
    if override_file_format not in [False, ".nii", ".nii.gz", ".gii", ".gii.gz"]:
        raise ValueError("'override_file_format' must be False, '.nii', '.nii.gz', '.gii' or '.gii.gz'")
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
        # if preloaded image, continue
        if isinstance(i, (nib.Nifti1Image, nib.GiftiImage)):
            pass
        # if string, load
        elif isinstance(i, (str, Path)):
            i = str(i)
            if i.endswith(".nii") or i.endswith(".nii.gz"):
                i = images.load_nifti(i)
            elif override_file_format in [".nii", ".nii.gz"]:
                i = Path(i).rename(Path(i).with_suffix(override_file_format))
                i = images.load_nifti(i)
            elif i.endswith(".gii") or i.endswith(".gii.gz"):
                i = images.load_gifti(i)
            elif override_file_format in [".gii", ".gii.gz"]:
                i = Path(i).rename(Path(i).with_suffix(override_file_format))
                i = images.load_gifti(i)
            elif i.endswith(".curv"):
                i = nib.GiftiImage(
                    darrays=[nib.gifti.GiftiDataArray(data=nib.freesurfer.read_morph_data(i))]
                )
            else:
                raise ValueError(f"File format of '{i}' not supported. Path must end with .nii(.gz) or .gii(.gz)")
        else:
            raise ValueError(f"Type {type(i)} not supported! Provide nifti, gifti or path to image file.")
        img_load.append(i)
    # return as tuple if two, or 
    return img_load[0] if len(img_load) == 1 else tuple(img_load)


def load_labels(labels, concat=True, header=None, index=None):
    """
    Load one or more label lists from paths, arrays, or Series.

    Parameters
    ----------
    labels : str, os.PathLike, list, np.ndarray, pd.Series, or tuple
        A single label source, or a tuple of up to 2 such elements (e.g. left/right
        hemisphere label files). list/ndarray/Series elements are passed through as
        plain lists; str/Path elements are read as the first column of a csv-like
        text file.
    concat : bool, default True
        If `labels` has 2 elements, whether to concatenate them into a single flat
        list (True) or return them as a 2-tuple (False).
    header : int, optional
        Row index to use as column header when reading a csv-like file; forwarded
        to ``pandas.read_csv``.
    index : int, optional
        Column index to use as the row index when reading a csv-like file;
        forwarded to ``pandas.read_csv``.

    Returns
    -------
    list or tuple
        A single flat list if `labels` had 1 element (or 2 with `concat=True`);
        otherwise a 2-tuple of lists.

    Raises
    ------
    ValueError
        If `labels` is not a path/list/ndarray/Series/tuple thereof, or a path
        element can't be read as a csv-like file.
    """
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
    """
    Load one or more distance matrices from paths, arrays, or DataFrames.

    Parameters
    ----------
    distmat : None, str, os.PathLike, np.ndarray, pd.DataFrame, list, or tuple
        A single distance matrix source, or a tuple of up to 2 such elements (e.g.
        left/right hemisphere matrices). ``None`` (or a tuple/list of all ``None``)
        is passed through unchanged. array/DataFrame elements are converted to
        plain ndarrays; str/Path elements are read as a headerless csv-like file.

    Returns
    -------
    np.ndarray, tuple, or None
        A single ndarray if `distmat` had 1 element; otherwise a 2-tuple of
        ndarrays. `None` is passed through unchanged.

    Raises
    ------
    ValueError
        If `distmat` is not a path/list/tuple/ndarray/DataFrame thereof, or a path
        element can't be read as a csv-like file.
    """
    # catch None content
    if distmat is None or (isinstance(distmat, tuple) and all([d is None for d in distmat])):
        return distmat
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


def load_spinmat(spinmat):
    """
    Load one or more spin-permutation index/weight arrays from paths or arrays.

    Parameters
    ----------
    spinmat : None, str, os.PathLike, np.ndarray, list, or tuple
        A single spin matrix source, or a tuple of up to 2 such elements (e.g.
        left/right hemisphere spins). ``None`` (or a tuple/list of all ``None``) is
        passed through unchanged. ``.npz`` paths are read via their ``"data"`` key;
        other array file paths are memory-mapped (``mmap_mode='c'``) rather than
        fully loaded into memory.

    Returns
    -------
    np.ndarray, tuple, or None
        A single array if `spinmat` had 1 element; otherwise a 2-tuple of arrays.
        `None` is passed through unchanged.

    Raises
    ------
    ValueError
        If `spinmat` is not a path/ndarray/list/tuple thereof, or an element's type
        is unsupported.
    """
    if spinmat is None or (isinstance(spinmat, tuple) and all(s is None for s in spinmat)):
        return spinmat
    if isinstance(spinmat, (str, Path, np.ndarray)):
        spinmat = (spinmat,)
    elif isinstance(spinmat, list):
        spinmat = tuple(spinmat)
    elif isinstance(spinmat, tuple):
        pass
    else:
        raise ValueError("Input must be path, ndarray, or list/tuple thereof")
    loaded = []
    for s in spinmat:
        if isinstance(s, np.ndarray):
            loaded.append(s)
        elif isinstance(s, (str, Path)):
            p = Path(s)
            if p.suffix == ".npz":
                f = np.load(p, allow_pickle=False)
                loaded.append(f["data"])
            else:
                loaded.append(np.load(p, allow_pickle=False, mmap_mode='c'))
        else:
            raise ValueError(f"Unsupported spinmat element type: {type(s)}")
    return loaded[0] if len(loaded) == 1 else tuple(loaded)


def to_pickle(obj, filepath, use_dill=False):
    """
    Pickle, compress, and save to a file.

    Parameters
    ----------
    obj : object
        Any python object to be pickled.
    filepath : str
        File path destination; must end in ``.pkl``, ``.pkl.gz``, or ``.pkl.blosc``
        (determines the compression method used).
    use_dill : bool, default False
        Whether to pickle with ``dill`` instead of the standard ``pickle`` module
        (needed for objects `pickle` can't handle, e.g. local functions/lambdas).

    Raises
    ------
    ValueError
        If `filepath`'s extension is not one of the supported formats.
    """
    
    # use dill instead of pickle
    if use_dill:
        pkl = dill
    else:
        pkl = pickle
        
    # save
    if filepath.endswith(".pkl"):
        with open(filepath, "wb") as f:
            pkl.dump(obj, f)
    elif filepath.endswith(".pkl.gz"):
        with gzip.open(filepath, "wb") as f:
            pkl.dump(obj, f)
    elif filepath.endswith(".pkl.blosc"):
        arr = pkl.dumps(obj, -1)
        with open(filepath, "wb") as f:
            s = 0
            while s < len(arr):
                e = min(s + blosc.MAX_BUFFERSIZE, len(arr))
                carr = blosc.compress(arr[s:e], typesize=8)
                f.write(carr)
                s = e
    else:
        raise ValueError(f"Unsupported file extension of path: {filepath}")


def from_pickle(filepath, use_dill=False):
    """
    Unpickle a python object saved with :func:`to_pickle`.

    Parameters
    ----------
    filepath : str
        Path to a ``.pkl``, ``.pkl.gz``, or ``.pkl.blosc`` file (extension
        determines the decompression method used).
    use_dill : bool, default False
        Whether to unpickle with ``dill`` instead of the standard ``pickle``
        module; must match what was used to save the file.

    Returns
    -------
    object
        The unpickled python object.

    Raises
    ------
    ValueError
        If `filepath`'s extension is not one of the supported formats.
    """
    
    # use dill instead of pickle
    if use_dill:
        pkl = dill
    else:
        pkl = pickle
        
    # load
    if filepath.endswith(".pkl"):
        with open(filepath, "rb") as f:
            return pkl.load(f)
    elif filepath.endswith(".pkl.gz"):
        with gzip.open(filepath, "rb") as f:
            return pkl.load(f)
    elif filepath.endswith(".pkl.blosc"):
        arr = []
        buffsize = blosc.MAX_BUFFERSIZE
        with open(filepath, "rb") as f:
            while buffsize > 0:
                try:
                    carr = f.read(buffsize)
                except (OverflowError, MemoryError):
                    buffsize = buffsize // 2
                    continue

                if len(carr) == 0:
                    break
                arr.append(blosc.decompress(carr))

        if buffsize == 0:
            raise RuntimeError("Could not determine a buffer size.")

        return pkl.loads(b"".join(arr))
    else:
        raise ValueError(f"Unsupported file extension of path: {filepath}")
    
    
def read_msigdb_json(json_path):
    """
    Read MSigDB gene set JSON file and return a "clean" dictionary of gene sets as expected for
    collection files in NiSpace.
    MSigDB json files can be found at https://www.gsea-msigdb.org/gsea/msigdb/human/genesets.jsp
    after selecting a specific gene set.
    
    Parameters
    ----------
    json_path : str, os.PathLike
        Path to MSigDB gene set JSON file.
        
    Returns
    -------
    dict
        Dictionary of gene sets. Keys are gene set names, values are lists of gene symbols.
        Both keys and values are sorted alphabetically.
    """
    in_dict = read_json(json_path)
    gene_sets = sorted( in_dict.keys() )
    out_dict = {
        gene_set: sorted( in_dict[gene_set]["geneSymbols"] ) 
        for gene_set in gene_sets
    }
    return out_dict