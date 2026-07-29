import copy
from typing import List, Union, Sequence, Literal, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

import logging
lgr = logging.getLogger(__name__)
from .io import parcellate_data, to_pickle, from_pickle
from .core.parcellation import Parcellation
from .core.reduce_x import _reduce_dimensions
from .core.transform_y import _dummy_code_groups, _num_code_subjects, _get_transform_fun
from .core.colocalize import _get_colocalize_fun, _sort_colocs, _get_coloc_stats, _rank_regress, _xsea_aggregate
from .core.correlate_within_region import correlate_within_region_core, _CWR_METHODS
from .core.permute import (_get_null_maps, _get_exact_p_values, _get_correct_mc_method,
                               _EMPIRICAL_MC_METHODS, _resolve_permute_combo,
                               _resolve_permute_mode_settings)
from .core.nullmaps import NullMaps
from .core.plot import _plot_categorical
from .core.constants import _COLOC_METHODS, _SPACE_DEFAULT_VOL, _COLOC_METHODS_UNIVARIATE
from .datasets import fetch_parcellation, fetch_reference, _check_parcellation
from .nulls import get_distance_matrix, _SPIN_METHODS, _DISTMAT_FREE_METHODS, _parse_null_method
from .stats.coloc import beta, elasticnet, lasso, mlr, partialpearson, pearson, rank2d, ridge
from .stats.misc import (mc_correction, residuals_nan, zscore_df, permute_groups,
                          compute_meff, meff_sidak_correction,
                          maxT_correction, step_maxT_correction, _null_stats_to_array,
                          null_to_p, rho_to_z, z_to_rho)
from .stats.effectsize import rzscore_nan, zscore_nan
from .cv import _get_dist_dep_splits, _get_rand_splits
from .plotting import nice_stats_labels, brainplot
from .utils.utils import (set_log, _quiet, fill_nan, _get_df_string, _lower_strip_ws, mean_by_set_df,
                          get_column_names, lower, print_arg_pairs,
                          _parse_df_string, _parse_bool, dedupe_rows)


def _cwr_null_p(obs, null):
    """Two-tailed empirical p per column: fraction of |null| >= |obs|, floor-clipped
    to avoid an exact-0 p (matches null_to_p's floor convention, stats/misc.py).
    Used for correlate_within_region()'s raw per-parcel p and, with the same floor,
    for get_within_region_correlations()'s maxT/step_maxT correction -- so
    'corrected p >= raw p' holds by construction, not just approximately.

    A NaN obs (e.g. a parcel/subject that was entirely missing, so rho itself is
    undefined) must produce a NaN p, not a fake "significant" one: `NaN >= x` is
    always False in numpy, so the naive fraction would silently evaluate to 0.0
    and get floor-clipped to `1/n_perm` -- an artificially tiny "significant"
    p-value for a statistic that was never computed. Guarded explicitly here."""
    n_perm = null.shape[0]
    with np.errstate(invalid="ignore"):
        p = (np.abs(null) >= np.abs(obs)[np.newaxis, :]).mean(axis=0)
    floor = max(np.finfo(float).eps, 1.0 / n_perm)
    p = np.clip(p, floor, 1.0 - floor)
    return np.where(np.isnan(obs), np.nan, p)


_CWR_OMNIBUS_STATS = ("rho", "absrho", "rho2")


def _cwr_omnibus_aggregate(rho, omnibus_stat, already_z=False):
    """Aggregate per-parcel rho (last axis) into one number per row: signed mean
    ("rho"), mean absolute value ("absrho"), or mean squared value ("rho2").
    Applied identically to the observed per-parcel rho (1D) and to each
    permutation's per-parcel null (2D, (n_perm, n_parcels)) so the two are
    directly comparable via _cwr_null_p -- for "absrho"/"rho2" that two-tailed
    |.| comparison collapses to a one-tailed test since both are already >= 0.

    Uses nanmean, not mean: a single entirely-missing parcel (rho == NaN for
    that parcel, in every permutation too) should be excluded from the average,
    not silently NaN out the whole omnibus statistic over one bad parcel.

    already_z : bool, default False
        Whether `rho` is already Fisher-z (correlate_within_region()'s own
        r_to_z, default True as of the same convention colocalize() uses).
        Normalized back to raw rho first (z_to_rho) so every branch below
        always works from a known, consistent scale regardless of the
        caller's r_to_z choice -- "absrho"/"rho2" must operate on the actual
        correlation coefficient (|rho|/rho**2), not |z|/z**2, which would be
        a different, not-directly-meaningful quantity.

    "rho" never averages raw (bounded, non-additive) correlation coefficients
    directly -- each parcel's rho is Fisher-z-transformed first, averaged on
    that scale, then converted back to a correlation for interpretability
    (standard meta-analytic combination; raw-r averaging is downward-biased in
    magnitude, worse for parcels with larger |rho|). Applying the same
    (monotonic, odd) round-trip to both the observed statistic and every null
    draw leaves the permutation p-value from `_cwr_null_p` unchanged either
    way -- this only fixes what the reported "rho" number itself means."""
    if already_z:
        rho = z_to_rho(rho)
    with np.errstate(invalid="ignore"):
        if omnibus_stat == "rho":
            return z_to_rho(np.nanmean(rho_to_z(rho), axis=-1))
        elif omnibus_stat == "absrho":
            return np.nanmean(np.abs(rho), axis=-1)
        elif omnibus_stat == "rho2":
            return np.nanmean(rho ** 2, axis=-1)
    if omnibus_stat not in _CWR_OMNIBUS_STATS:
        lgr.critical_raise(
            f"'omnibus_stat' must be one of {_CWR_OMNIBUS_STATS}, got '{omnibus_stat}'.",
            ValueError
        )


def _match_maps(index, queries):
    """Return sorted unique integer positions in *index* matching any of *queries*.

    Matching priority:
    1. Exact equality with the index value
    2. For tuple index values: exact element match or partial string match on any element
    3. For scalar index values: exact or partial string match
    """
    if isinstance(queries, (str, int)):
        queries = [queries]
    keep = []
    for i, val in enumerate(index):
        for q in queries:
            q_str = str(q)
            if q == val:
                keep.append(i)
                break
            if isinstance(val, tuple):
                parts = [str(v) for v in val]
                if q in val or any(q_str == p or q_str in p for p in parts):
                    keep.append(i)
                    break
            else:
                s = str(val)
                if q_str == s or q_str in s:
                    keep.append(i)
                    break
    return sorted(set(keep))


# ==================================================================================================
# DEPRECATION MESSAGE STRINGS
# ==================================================================================================

_DEPR_RETURN_SELF = (
    "In the first non-dev release, all NiSpace object methods will return the "
    "object itself by default. Set NiSpace(return_self=True) to disable this warning."
)
_DEPR_COMBAT_KEEP = (
    "'combat_keep' is deprecated and will be ignored. All regression covariates "
    "are now automatically protected during ComBat harmonization."
)
_DEPR_P_FROM_AVERAGE_Y_COLOC = (
    "'p_from_average_y_coloc' is deprecated and will be removed in the first "
    "non-dev release. Use 'pooled_p' instead."
)
_DEPR_SORT_COLOCS = (
    "'sort_colocs' is deprecated and will be removed in the first non-dev release. "
    "Use sort_by='coloc' instead."
)
_DEPR_L2RMAP = (
    "'parcellation_l2rmap' is deprecated and will be removed in the first non-dev release. "
    "Left-to-right parcel mapping is no longer supported. The parameter is silently ignored."
)
_DEPR_IGNORE_BACKGROUND_DATA = (
    "'ignore_background_data' is deprecated and will be removed in the first non-dev "
    "release. Use 'background_value' instead: pass background_value=False to disable "
    "background exclusion (equivalent to ignore_background_data=False), or a scalar/"
    "list/'auto'/per-role dict to enable it (equivalent to ignore_background_data=True)."
)


def _resolve_per_role(value, role, default="auto"):
    """Resolve a background_value spec (scalar/False/dict) for one X/Y/Z role.

    `value` is either a plain scalar/list/'auto'/False (applied uniformly to
    every role, passed through unchanged) or a per-role dict like
    ``{"y": False}`` (keys are exactly "x"/"y"/"z"; a role absent from the
    dict falls back to `default`, never to some other implicit value).
    """
    return value.get(role, default) if isinstance(value, dict) else value

# ==================================================================================================
# DEFINE CLASS
# ==================================================================================================

class NiSpace:
    """
    Main analysis object for spatial colocalization / imaging-transcriptomics-style
    workflows between a set of X (predictor) maps and a set of Y (target) maps,
    with optional Z covariate maps. Import via ``from nispace import NiSpace``.

    Typical usage follows a fixed pipeline of method calls, each acting on and
    updating the same object:

    1. Construct with `x`/`y`/(optional `z`) and a `parcellation`, then call
       :meth:`fit` to parcellate/validate the data.
    2. Optionally reduce/clean/transform the data: :meth:`reduce_x`,
       :meth:`clean_y`, :meth:`transform_y`, :meth:`transform_z`.
    3. Compute colocalization statistics between X and Y with :meth:`colocalize`.
    4. Optionally decompose a colocalization result region-by-region with
       ``nispace.diagnostics.regional_influence`` / ``regional_contribution``.
    5. Assess significance via permutation testing with :meth:`permute`, then
       :meth:`correct_p` for multiple comparisons and/or
       :meth:`normalize_colocalizations` against the null distribution.
    6. Retrieve results with the `get_*` methods (`get_x`, `get_y`, `get_z`,
       `get_colocalizations`, `get_p_values`, ...), visualize with :meth:`plot` /
       :meth:`plot_brain`, and persist the object with :meth:`to_pickle` /
       :meth:`from_pickle` / :meth:`copy`.

    Attributes
    ----------
    `NiSpace` has no public instance attributes. All state (data, colocalization
    results, null distributions, p-values, parcellation, and internal settings) is
    stored on private, underscore-prefixed attributes and is not meant to be
    accessed directly — use the `get_*` methods instead.
    """

    def __init__(self, 
                 x: Union[np.ndarray, pd.DataFrame, pd.Series, 
                          List[Union[str, Path, nib.Nifti1Image, nib.GiftiImage]],
                          Dict[str, Union[str, Path, nib.Nifti1Image, nib.GiftiImage]]], 
                 y: Union[np.ndarray, pd.DataFrame, pd.Series, 
                          List[Union[str, Path, nib.Nifti1Image, nib.GiftiImage]],
                          Dict[str, Union[str, Path, nib.Nifti1Image, nib.GiftiImage]]] = None, 
                 z: Union[Literal["gm", "wm", "csf", "veins", "arteries"],
                          List[Literal["gm", "wm", "csf", "veins", "arteries"]],
                          np.ndarray, pd.DataFrame, pd.Series,
                          List[Union[str, Path, nib.Nifti1Image, nib.GiftiImage]],
                          Dict[str, Union[str, Path, nib.Nifti1Image, nib.GiftiImage]]] = None,
                 x_labels: Sequence[str] = None, 
                 y_labels: Sequence[str] = None, 
                 z_labels: Sequence[str] = None, 
                 data_space: Literal["MNI152NLin6Asym", "MNI152NLin2009cAsym", "fsaverage", "fsLR"] = _SPACE_DEFAULT_VOL,
                 standardize: Union[Literal["x", "y", "z", "xy", "xz", "yz", "xyz"], bool] = "xz", 
                 drop_nan: bool = False,    
                 parcellation: Union[str, Path, nib.Nifti1Image, nib.GiftiImage] = None, 
                 parcellation_labels: Sequence[str] = None, 
                 parcellation_space: Literal["MNI152NLin6Asym", "MNI152NLin2009cAsym", "fsaverage", "fsLR"] = _SPACE_DEFAULT_VOL,
                 parcellation_hemi: Union[Literal["R", "L"], Sequence[Literal["L", "R"]]] = ["L", "R"], 
                 parcellation_symmetric: bool = False,
                 parcellation_l2rmap: pd.DataFrame = None,
                 parcellation_idc_lh: Sequence[int] = None,
                 parcellation_idc_rh: Sequence[int] = None,
                 parcellation_idc_sc: Sequence[int] = None,
                 parcellation_dist_mat: Union[np.ndarray, pd.DataFrame] = None,
                 parcellation_spin_mat: np.ndarray = None,
                 load_dist_mat: bool = True,
                 load_spin_mat: bool = True,
                 resampling_target: Literal["data", "parcellation"] = "data",
                 n_proc: int = 1,
                 seed: int = None,
                 verbose: bool = True,
                 dtype: Union[type, str] = np.float32,
                 return_self: bool = True,
                 binary_y: bool = False,
                 **kwargs):
        """
        Initialize the NiSpace object. 
        On initialization, the parameters are only stored. Processing is done with NiSpace.fit().
        
        Parameters
        ----------
        x : array-like of shape(n_reference, n_parcels) or shape(n_parcels) or len(n_reference) list-like of image data
            The reference maps (e.g., pet or mRNA data). Can be a numpy array, pandas DataFrame, 
            pandas Series, a list or a dictionary containing (paths to) image objects. If a 
            dictionary, the keys are the names of the maps, but will be overridden by x_labels.
        y : array-like of shape(n_target, n_parcels) or shape(n_parcels) or len(n_target) list-like of image data, optional
            The target data (i.e., usually your maps of interest). Data types can be the same as 
            for x. Default is None. If None, NiSpace will create a copy of the reference maps to 
            evaluate reference map-to-map intercorrelations.
        z : str, list of str, or array-like, optional
            Covariate maps to regress out of X and/or Y before colocalization. Can be one of the
            shortcut strings ``"gm"``, ``"wm"``, ``"csf"``, ``"veins"``, or ``"arteries"`` (or a
            list of several) to automatically fetch the corresponding tissue probability map (TPM)
            from the NiSpace data library. Alternatively, accepts a numpy array, pandas
            DataFrame/Series, or a list of image paths/objects. Default is None.
            When Z is provided and :meth:`colocalize` is called with its default
            ``regress_z=True``, Z is regressed from X and Y before computing the correlation —
            making ``colocalize("spearman")`` operationally equivalent to
            ``colocalize("partialspearman")``. The ``partial*`` method names are self-documenting
            aliases that additionally raise an error if Z is missing. Pass ``regress_z=False`` to
            :meth:`colocalize` to suppress regression even when Z is set.
        x_labels : sequence of str, optional
            Labels for the x data. Default is None. If None and x is DataFrame or Series, the 
            labels are taken from x's index (DataFrame) or name (Series).
        y_labels : sequence of str, optional
            Labels for the y data. Default is None. If None and y is DataFrame or Series, the 
            labels are taken from y's index (DataFrame) or name (Series).
        z_labels : sequence of str, optional
            Labels for the z data. Default is None. If None and len(z) == len(y) any y is DataFrame 
            or Series, the labels are taken from y's (not z's) index (DataFrame) or name (Series).
        data_space : str, optional
            The space in which the (x,y,z) data is defined; passed to neuromaps. Should be one of
            "MNI152NLin6Asym", "MNI152NLin2009cAsym", "fsaverage" or "fsLR". 
        standardize : str or bool, optional
            Whether to standardize the parcellated (x,y,z) data within each map across parcels. 
            Default is "xz". If True, will standardize all data. If (combination of) "x", "y", 
            or "z", will standardize only the respective data array.
        drop_nan : bool, optional
            Whether to drop NaN values. Default is False. NaN values should be handled case-by-case 
            in all following analyses, so False is a good choice. 
        parcellation : str, Path, Nifti1Image, or GiftiImage, optional
            The parcellation image to use. Default is None. Required in following cases: 
            1) if image (paths) are passed to x, y, or z; 2) if z is a TPM shortcut string; 3) if "map" permutation
            is used in NiSpace.permute(); 4) if distance-based cross-validation is used. Cases 3/4
            apply even if initial data is passed pre-parcellated in arrays.
        parcellation_labels : sequence of str, optional
            Labels for the parcellation. Default is None. If None and input (x,y) data is DataFrame or 
            Series, 
        parcellation_space : str, optional
            The space in which the parcellation is defined. See data_space for details.
        parcellation_hemi : sequence of str, optional
            The hemispheres in which the parcellation if defined if parcellation_space is 
            "fsaverage" or "fslr". Default is ["L", "R"].
        parcellation_dist_mat : array-like of shape(n_parcels, n_parcels), optional
            The distance matrix for the parcellation. Default is None. Required as described in 
            cases 3) and 4) for parcellation.
        resampling_target : str, optional
            The target for resampling. Options: "data" (default) or "parcellation". If "data", the
            parcellation is resampled to the data space before application (nearest neighbor). 
            If "parcellation", the data is resampled to the parcellation space (linear). 
            Resampling only works in MNI -> MNI and MNI -> mni/fsaverage/fslr direction; if, e.g., 
            input data is in MNI space and parcellation is in fsaverage, resampling_target will be
            forced to "parcellation", as MNI -> fsaverage/fslr transformation is not supported.
        n_proc : int, optional
            The number of processes to use in joblib parallelization. Default is 1. -1 will use as
            many processes as cores are detected.
        seed : int, optional
            Default random seed, used by :meth:`reduce_x`, :meth:`colocalize`, and
            :meth:`permute` whenever their own ``seed`` argument is left at
            ``None``. Passing ``seed`` directly to one of those methods
            overrides this default for that call only.
        verbose : bool, optional
            Whether to print (a lot of) verbose output. Default is True.
        dtype : data-type, optional
            The data type to use for the arrays. Default is np.float32 to save memory.

        Returns
        -------
        None
        """
        
        self._x = x
        self._x_with_self = False
        self._y = y
        self._z = z
        self._x_lab = x_labels
        self._y_lab = y_labels
        self._z_lab = z_labels
        if isinstance(data_space, str):
            data_space = [data_space] * 3
        elif isinstance(data_space, (list, tuple)) & len(data_space)==1:
            data_space = data_space * 3
        elif isinstance(data_space, (list, tuple)) & len(data_space)==3:
            pass
        else:
            lgr.critical_raise("'data_space' must be a string, a list with len==1 or a list with "
                               f"len==3! Is {type(data_space)}.",
                               ValueError)
        self._data_space = data_space
        if "parc" in kwargs and parcellation is None:
            parcellation = kwargs.pop("parc")
        # Convert tuple/list combined parcellation to "+" string before type dispatch
        if isinstance(parcellation, (tuple, list)) and all(isinstance(p, str) for p in parcellation):
            parcellation = "+".join(parcellation)
        if parcellation_l2rmap is not None:
            lgr.warning(_DEPR_L2RMAP)
            # TODO (first non-dev release): remove parcellation_l2rmap param entirely
        # custom image/path parcellations are built immediately; integrated strings
        # and already-constructed Parcellation objects are handled in fit()
        if parcellation is None:
            self._parc = None
        elif not isinstance(parcellation, (str, Parcellation)):
            # image or path → build Parcellation immediately
            self._parc = Parcellation.from_path(
                source=parcellation,
                space=parcellation_space,
                labels=parcellation_labels,
                dist_mat=parcellation_dist_mat,
                spin_mat=parcellation_spin_mat,
                symmetric=parcellation_symmetric,
                hemi=parcellation_hemi,
            )
        else:
            # string name or existing Parcellation object → resolved in fit()
            self._parc = {
                "parc": parcellation,
                "labels": parcellation_labels,
                "space": parcellation_space,
                "hemi": parcellation_hemi,
                "symmetric": parcellation_symmetric,
                "idc_lh": parcellation_idc_lh,
                "idc_rh": parcellation_idc_rh,
                "idc_sc": parcellation_idc_sc,
            }
        self._parc_dist_mat = {
            "null_maps": parcellation_dist_mat
        }
        if not isinstance(parcellation_dist_mat, tuple):
            self._parc_dist_mat["cv"] = parcellation_dist_mat
        self._parc_spin_mat = parcellation_spin_mat
        self._load_dist_mat = load_dist_mat
        self._load_spin_mat = load_spin_mat
        self._resampl_target = resampling_target
        self._n_proc = n_proc
        self._seed = seed
        self._drop_nan = drop_nan
        self._dtype = dtype
        self._verbose = verbose
        if standardize == True:
            self._zscore = "xyz"
        elif standardize in [None, False]:
            self._zscore = ""
        else:
            self._zscore = standardize
        self._transform_count = 0
        
        # empty data storage dicts
        self._X_dimred = {}
        self._dimred = {}
        self._Y_trans = {}
        self._colocs = {}
        self._colocs_fun = {}
        self._coloc_kwargs_by_method = {}
        self._nulls = {
            "_colocs": {}
        }
        self._p_colocs = {}
        self._z_colocs = {}
        self._corr_within = {}
        self._p_corr_within = {}

        # defaults for get functions (IMPORTANT: this determines what coloc and get function will do!)
        self._last_settings = {
            "method": None,
            "X_reduction": False,
            "Y_transform": False,
            "xsea": False,
            "rank": False,
            "zy_matched": False,
            "regress_z": None,
            "mc_method": None,
            "z_method": "robust",
            "pooled_p": False,
            "cwr_method": None,
        }
        
        # deprecation adjustment
        self._return_self = return_self

        # binary Y mode
        self._binary_y = binary_y
        if binary_y:
            lgr.info("binary_y=True: Expecting Y input derived from binary maps "
                     "(e.g., cluster or network maps); background_value=False "
                     "for Y parcellation (unless explicitly overridden via "
                     "background_value={'y': ...} in .fit()).")
        if binary_y and "y" in self._zscore:
            self._zscore = self._zscore.replace("y", "")
            lgr.warning(
                "binary_y=True: Y data will not be z-standardized regardless of the "
                "'standardize' setting. Z-scoring would destroy the [0,1] range of "
                "parcellated binary maps."
            )

    # FIT ==========================================================================================
    
    def fit(self, **kwargs):
        """
        "Fit" the NiSpace class instance, i.e., check input and apply parcellation if necessary.
        Input and parameters are set on initialization.

        Parameters
        ----------
        **kwargs
            Any keyword argument accepted by :func:`parcellate_data` can be
            passed here and will override that function's defaults. The most
            commonly needed ones are:

            background_value : float, list, set, array, 'auto', False, or dict
                Value(s) treated as background, or ``False`` to disable
                background exclusion entirely (background/zero is then real
                data, e.g. for binary/coverage-style maps -- NaN is still
                always excluded regardless). ``'auto'`` (default) auto-detects
                a border-voxel/medial-wall value and combines it with exact
                ``0.0``. Also accepts a **per-role dict**,
                ``{"x": ..., "y": ..., "z": ...}``, to set X/Y/Z
                independently -- any of the above per key; roles absent from
                the dict fall back to ``'auto'``. Default: ``'auto'``

                ``NiSpace(binary_y=True)`` automatically applies
                ``background_value=False`` to Y only (an all-zero parcel in a
                binary/fractional cluster-coverage map is a genuine 0%-overlap
                result, not missing background); this default backs off only
                if the dict passed here explicitly contains a ``"y"`` key --
                a plain top-level scalar/list meant for X/Z does not affect it.
            report_background_parcels : bool
                Whether to explicitly flag (and log) all-background parcels.
                Such parcels are already NaN via empty-mean aggregation
                regardless of this flag, so it only affects whether they're
                recorded/logged, not the returned values. Always a no-op for
                a role resolved to ``background_value=False`` (e.g. Y under
                ``binary_y=True``). Default: False
            min_num_valid_datapoints : int, optional
                Minimum number of valid (non-background, non-NaN) datapoints
                required per parcel; parcels below this are set to NaN.
                Default: None
            min_fraction_valid_datapoints : float, optional
                Minimum fraction of valid (non-background, non-NaN)
                datapoints, relative to the parcel's total size in the
                resampled parcellation, required per parcel. Default: None

            The deprecated ``ignore_background_data``/``drop_background_parcels``
            kwargs are still accepted (forwarded through, with a deprecation
            warning) but bypass the per-role ``background_value`` dict/
            ``binary_y`` resolution above entirely -- use ``background_value``
            instead.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        verbose = set_log(lgr, self._verbose)
        lgr.info("*** NiSpace.fit() - Data extraction and preparation. ***")
        
        # TODO (first non-dev release): remove return_self parameter and all _return_self branches
        #   throughout api.py; methods should unconditionally return self
        if not self._return_self:
            lgr.warning(_DEPR_RETURN_SELF)
    
        ## handle parcellation
        if self._parc is not None:
            # integrated parcellation
            if isinstance(self._parc, dict) and isinstance(self._parc["parc"], str):
                # check if parcellation is an integrated parcellation
                parc_integrated = _check_parcellation(self._parc["parc"], force_str=True, raise_not_found=False)
                if parc_integrated is not None:
                    # fetch full multi-space Parcellation (space=None → Parcellation object)
                    parc_obj = fetch_parcellation(
                        parcellation=parc_integrated,
                        hemi=self._parc["hemi"],
                        return_dist_mat=self._load_dist_mat,
                        return_spin_mat=self._load_spin_mat,
                        verbose=verbose,
                    )
                    # activate space only when raw image data needs parcellating;
                    # DataFrames/Series/ndarrays are already parcellated — everything
                    # else (lists, paths, image objects, "gm" string, …) is raw
                    needs_parcellating = any(
                        not isinstance(d, (pd.DataFrame, pd.Series, np.ndarray))
                        for d in [self._x, self._y, self._z] if d is not None
                    )
                    if needs_parcellating:
                        active_space = parc_obj.get_image_for_dataspace(self._parc["space"])
                        parc_obj.set_active_space(active_space)
                    self._parc = parc_obj
                    # populate dist_mat dict from Parcellation — use null space explicitly
                    # (parc._space may be None when data is already parcellated)
                    _ns_result = parc_obj.get_null_space()
                    _null_dm_space = _ns_result[1][0] if isinstance(_ns_result[0], tuple) else _ns_result[0]
                    dm = parc_obj.get_dist_mat(space=_null_dm_space, compute_if_missing=False)
                    self._parc_dist_mat["null_maps"] = dm
                    if not isinstance(dm, tuple):
                        self._parc_dist_mat["cv"] = dm
                    if self._parc_spin_mat is None:
                        _cx_space = _ns_result[0][0] if isinstance(_ns_result[0], tuple) else _ns_result[0]
                        self._parc_spin_mat = parc_obj.get_spin_mat(space=_cx_space)

            # custom parcellation (string file path not matched as integrated)
            if not isinstance(self._parc, Parcellation):
                self._parc = Parcellation.from_path(
                    source=self._parc["parc"],
                    space=self._parc["space"],
                    labels=self._parc["labels"],
                    dist_mat=self._parc_dist_mat["null_maps"],
                    symmetric=self._parc["symmetric"],
                    hemi=self._parc["hemi"],
                )

        ## extract input data
        # background_value resolution: a plain scalar/list/'auto'/False applies
        # uniformly to X/Y/Z (as merged into _input_kwargs below); a per-role
        # dict ({"x": ..., "y": ..., "z": ...}) is resolved independently per
        # role via _resolve_per_role, with roles absent from the dict falling
        # back to 'auto'. Legacy ignore_background_data (if explicitly passed)
        # takes precedence and skips this resolution entirely -- forwarded
        # uniformly to X/Y/Z exactly as before the background_value redesign.
        # TODO (first non-dev release): remove legacy ignore_background_data kwarg
        #   support and the _legacy_bg branches below
        _legacy_bg = "ignore_background_data" in kwargs
        if _legacy_bg:
            lgr.warning(_DEPR_IGNORE_BACKGROUND_DATA)
        _bg_spec = kwargs.get("background_value", "auto")

        _input_kwargs = dict(
            parcellation=self._parc,
            resampling_target=self._resampl_target,
            n_proc=self._n_proc,
            verbose=verbose,
            dtype=self._dtype,
        ) | kwargs
        if not _legacy_bg:
            _input_kwargs["background_value"] = _resolve_per_role(_bg_spec, "x")

        # reference data -> usually e.g. PET atlases
        lgr.info("Checking input data for 'x' (should be, e.g., PET data):")
        self._X = parcellate_data(
            self._x,
            data_labels=self._x_lab,
            data_space=self._data_space[0],
            **_input_kwargs
        )
        lgr.info(f"Got 'x' data for {self._X.shape[0]} x {self._X.shape[1]} parcels.")

        # target data -> usually e.g. subject data or group-level outcome data
        if self._y is None:
            lgr.warning("No 'y' data detected. Will use 'X' as both reference and target data!")
            self._Y = self._X.copy()
            self._x_with_self = True
            if isinstance(self._zscore, str):
                if "x" in self._zscore and not "y" in self._zscore:
                    self._zscore += "y"
        else:
            lgr.info("Checking input data for 'y' (should be, e.g., subject data):")
            _input_kwargs_y = _input_kwargs.copy()
            if _legacy_bg:
                if self._binary_y:
                    _input_kwargs_y.setdefault("ignore_background_data", False)
            else:
                _y_val = _resolve_per_role(_bg_spec, "y")
                _explicit_y = isinstance(_bg_spec, dict) and "y" in _bg_spec
                if self._binary_y and not _explicit_y:
                    _y_val = False
                _input_kwargs_y["background_value"] = _y_val
            self._Y = parcellate_data(
                self._y,
                data_labels=self._y_lab,
                data_space=self._data_space[1],
                **_input_kwargs_y
            )
        lgr.info(f"Got 'y' data for {self._Y.shape[0]} x {self._Y.shape[1]} parcels.")
        
        # data to control correlations for
        if self._z is not None:
            lgr.info("Checking input data for z (should be, e.g., grey matter probability):")
            _TPM_SHORTCUTS = {"gm", "wm", "csf", "veins", "arteries"}
            _z_list = [self._z] if isinstance(self._z, str) else (
                list(self._z) if isinstance(self._z, list) else None)
            _z_data_space = self._data_space[2]
            if _z_list is not None and all(
                    isinstance(s, str) and s.lower() in _TPM_SHORTCUTS for s in _z_list):
                _z_list = [s.lower() for s in _z_list]
                # integrated parcellation -> fetch the precomputed per-parcellation TPM table
                # directly (faster, and already correctly background-treated -- safer than
                # re-parcellating raw images under any background_value default/override).
                # Only custom (non-integrated) parcellations fall back to raw TPM images,
                # fetched in a fixed canonical volumetric space -- not '_data_space[2]', which
                # describes user-supplied z data; there is none here, this is a fetched
                # reference map -- and resampled by the normal parcellate_data() call below.
                _parc_name = None
                if isinstance(self._parc, Parcellation):
                    _parc_name = ((self._parc._cx_name or "") + (self._parc._sc_name or "")
                                  if self._parc._is_combined else self._parc._name)
                _parc_integrated = _check_parcellation(_parc_name, raise_not_found=False) \
                    if _parc_name else None
                if _parc_integrated is not None:
                    lgr.info(f"Fetching pre-parcellated TPM reference map(s) for z: {_z_list} "
                             f"(parcellation: '{_parc_integrated}').")
                    self._z = fetch_reference("tpm", maps=_z_list, parcellation=self._parc,
                                              hemi=self._parc._selected_hemi, standardize_parcellated=False,
                                              print_references=False, verbose=verbose)
                else:
                    lgr.info(f"Fetching TPM reference map(s) for z: {_z_list} "
                             f"(space: '{_SPACE_DEFAULT_VOL}').")
                    self._z = fetch_reference("tpm", maps=_z_list, space=_SPACE_DEFAULT_VOL,
                                              print_references=False, verbose=verbose)
                    _z_data_space = _SPACE_DEFAULT_VOL
                if self._z_lab is None:
                    self._z_lab = _z_list
            _input_kwargs_z = _input_kwargs.copy()
            if not _legacy_bg:
                _input_kwargs_z["background_value"] = _resolve_per_role(_bg_spec, "z")
            self._Z = parcellate_data(
                self._z,
                data_labels=self._z_lab,
                data_space=_z_data_space,
                **_input_kwargs_z
            )
            lgr.info(f"Got 'z' data for {self._Z.shape[0]} x {self._Z.shape[1]} parcels.")
            
        else:
            self._Z = None
        
        ## check parcel number
        if self._X.shape[1] != self._Y.shape[1]:
            lgr.critical_raise("Got differing numbers of parcels in 'x' & 'y' data!", 
                               ValueError)
        if self._Z is not None:
            if self._X.shape[1] != self._Z.shape[1]:
                lgr.critical_raise("Got differing numbers of parcels in 'x'/'y' & 'z' data!", 
                                   ValueError)          
        
        # ## check distance matrix
        # if self._parc_dist_mat["null_maps"] is not None:
        #     dist_mat = load_distmat(self._parc_dist_mat["null_maps"])
        #     if not isinstance(dist_mat, tuple):
        #         dist_mat = dist_mat, 
        #     for d in dist_mat:
        #         if d.shape[0] != d.shape[1] != (self._X.shape[1] if len(dist_mat) == 1 
        #                                         else self._X.shape[1] / 2):
        #             lgr.warning(f"Provided distance matrix shape {d.shape} is not symmetric or "
        #                         f"does not fit with number of parcels in data ({self._X.shape[1]})!"
        #                         " Ignoring provided matrix.")
        #             dist_mat = None
        #             break
        #     self._parc_dist_mat["null_maps"] = dist_mat[0] if len(dist_mat) == 1 else dist_mat
            
        ## check data indices
        if all(self._X.columns != self._Y.columns):
            lgr.warning("Parcel labels (column names) differ between 'x' & 'y' dataframes! "
                        "Using 'x' labels for both.")
            self._Y.columns = self._X.columns.copy()
        if self._Z is not None:
            if all(self._X.columns != self._Z.columns):
                lgr.warning("Parcel labels (column names) differ between 'x'/'y' & 'z' dataframes! "
                            "Using 'x' labels for both.")
                self._Z.columns = self._X.columns.copy()
        
        ## deal with nan's
        self._nan_bool = pd.concat([self._X, self._Y, self._Z], axis=0).isnull().any(axis=0)
        self._no_nan = np.array(~self._nan_bool)
        # case remove nan parcels completely
        if self._drop_nan==True:
            lgr.warning(f"Dropping {np.sum(self._nan_bool)} parcels with nan's. "
                        "This might lead to problems with null map generation!")
            self._X = self._X.loc[:, self._no_nan]
            self._Y = self._Y.loc[:, self._no_nan]
            if self._Z is not None:
                self._Z = self._Z.loc[:, self._no_nan]
            self._nan_bool = pd.concat([self._X, self._Y, self._Z], axis=0).isnull().any(axis=0)
            self._no_nan = np.array(~self._nan_bool)
            
        # get column (parcel) indices and labels with nan's
        self._nan_cols = list(np.where(self._nan_bool==True)[0])
        self._nan_labs = list(self._nan_bool[self._nan_bool].index)
            
        ## parcel number
        self._n_parcels = self._X.shape[1]
        
        ## update data labels
        self._x_lab = self._X.index
        self._y_lab = self._Y.index
        
        ## z-standardization
        if self._zscore:
            self._zscore = self._zscore.lower()
            if "x" in self._zscore:
                lgr.info("Z-standardizing 'X' data.")
                self._X = zscore_df(self._X, along="rows")
            if "y" in self._zscore:
                lgr.info("Z-standardizing 'Y' data.")
                self._Y = zscore_df(self._Y, along="rows")
            if ("z" in self._zscore) & (self._Z is not None):
                lgr.info("Z-standardizing 'Z' data.")
                self._Z = zscore_df(self._Z, along="rows")

        ## warn on accidental duplicate rows (e.g. the same reference/subject map included
        # twice) -- cheap diagnostic, not a dedup path; NaN-safe via pandas .duplicated()
        for _name, _data in (("X", self._X), ("Y", self._Y), ("Z", self._Z)):
            if _data is not None:
                _n_dup = int(_data.duplicated().sum())
                if _n_dup:
                    if (_name == "X") and ("set" in _data.index.names):
                        continue
                    else:
                        lgr.warning(f"{_n_dup} duplicate row(s) found in '{_name}' data -- if "
                                    "unintended, check your input for accidentally repeated maps.")

        ## return complete object
        return self


    # REDUCE DIMENSIONS ============================================================================
    
    def reduce_x(self, reduction, 
                 mean_by_set=False, weighted_mean=False,
                 n_components=None, min_ev=None, fa_method="minres", fa_rotation="promax",
                 seed=None, store=True, verbose=None):
        """
        Reduce the X data to a smaller number of maps/components before
        colocalization -- either by aggregating (mean/median, optionally per
        ``"set"``) or by a proper dimensionality reduction (PCA/ICA/FA). The
        result is stored under ``reduction`` and picked up by
        :meth:`get_x`/:meth:`colocalize`/etc. via their ``X_reduction`` argument.

        Parameters
        ----------
        reduction : str
            One of:

            * ``"mean"`` / ``"median"`` -- parcel-wise mean/median across X maps
              (optionally per ``"set"``, see ``mean_by_set``).
            * ``"pca"`` -- principal component analysis (``sklearn.PCA``).
            * ``"ica"`` -- independent component analysis (``sklearn.FastICA``);
              has no explained-variance concept, so ``min_ev`` has no effect.
            * ``"fa"`` -- factor analysis (requires the optional
              ``factor_analyzer`` package).

            Any other value logs an error and returns ``None`` rather than
            raising.
        mean_by_set : bool, default False
            For ``"mean"``/``"median"``, group X maps by their ``"set"``
            MultiIndex level before aggregating (silently disabled if X has no
            ``"set"`` level).
        weighted_mean : bool, default False
            For ``"mean"``/``"median"``, weight maps by a ``"weight"``
            MultiIndex level on X (silently disabled if absent).
        n_components : int, optional
            Number of components to keep for ``"pca"``/``"ica"``/``"fa"``.
            Ignored if not applicable. Defaults to the maximum possible 
            (one component per parcel) if neither this nor ``min_ev`` is given.
        min_ev : float, optional
            Minimum cumulative explained-variance (EV) fraction; if given, overrides
            ``n_components`` for ``"pca"``/``"fa"`` by picking the smallest
            sufficient number of components/factors, which's cumulative EV exceeds 
            ``min_ev``. Ignored if not applicable.
        fa_method : str, default "minres"
            Factor-extraction method, forwarded to ``factor_analyzer.FactorAnalyzer``. 
            Only used for ``"fa"``.
        fa_rotation : str, default "promax"
            Rotation method, forwarded to ``factor_analyzer.FactorAnalyzer``.
            Only used for ``"fa"``.
        seed : int, optional
            Random seed. Only used by ``"ica"``. Defaults to the seed set at
            init (``NiSpace(seed=...)``) if not given here.
        store : bool, default True
            Store the reduced X (accessible via ``get_x(X_reduction=reduction)``)
            and, for ``"pca"``/``"ica"``/``"fa"``, its per-component metadata
            (explained variance, loadings), and remember ``reduction`` as the
            "last used" X reduction for subsequent calls.
        verbose : bool, optional
            Print progress messages. Defaults to the value set at init.

        Returns
        -------
        pandas.DataFrame or tuple
            For ``"mean"``/``"median"``: the reduced X DataFrame. For
            ``"pca"``/``"ica"``/``"fa"``: a tuple ``(X_reduced, ev, loadings)``,
            where ``ev`` is the per-component explained variance (``None`` for
            ``"ica"``) and ``loadings`` is a DataFrame of each original
            parcel/map's association with each retained component.
        """
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        lgr.info("*** NiSpace.reduce_x() - X dimensionality reduction. ***")
        seed = seed if seed is not None else self._seed

        ## check if fit was run
        self._check_fit()
        if self._X.shape[0] <= 1:
            lgr.critical_raise(f"For X dimensionality reduction, X data has to be more than "
                               f"one map ({self._X.shape[0]})!",
                               ValueError)   
        
        ## get X data (so this function can be run on direct X input data)
        _X = self._X
 
        ## case mean or median
        if reduction.lower() in ["mean", "median"]:
            lgr.info(f"Calculating parcelwise{' weighted' if weighted_mean else ''} "
                     f"{reduction}{' by set' if mean_by_set else ''} of X data.")
            
            if weighted_mean & ("weight" not in _X.index.names):
                lgr.error("DataFrame must have 'weight' in its MultiIndex for weighted calculations")
                weighted_mean = False
            if mean_by_set & ("set" not in _X.index.names):
                lgr.error("DataFrame must have 'set' in its MultiIndex for set-wise calculations")
                mean_by_set = False

            _X_reduced = mean_by_set_df(_X, mean_by_set, weighted_mean, reduction)
            
        ## case PCA / case ICA / case FA
        elif reduction.lower() in ["pca", "ica", "fa"]:
            lgr.info(f"Calculating {reduction.upper()} on X data.")
            _X_reduced, ev, loadings = _reduce_dimensions(
                data=_X.values[:, self._no_nan].T, 
                method=reduction, 
                n_components=n_components, 
                min_ev=min_ev,
                fa_method=fa_method, 
                fa_rotation=fa_rotation,
                seed=seed
            )
            # save
            _X_reduced = fill_nan(
                data=pd.DataFrame(
                    data=_X_reduced.T, 
                    index=[f"c{i}" for i in range(_X_reduced.shape[1])], 
                    columns=_X.iloc[:, self._no_nan].columns, 
                    dtype=self._dtype
                ),
                idx=self._nan_cols, 
                idx_label=self._nan_labs, 
                which="col"
            )
            loadings = pd.DataFrame(
                data=loadings, 
                columns=_X_reduced.index, 
                index=_X.index, 
                dtype=self._dtype
            )
            ev = pd.Series(
                data=ev,
                name="ev",
                index=[f"c{i}" for i in range(_X_reduced.shape[0])],
                dtype=self._dtype
            )         
        
        ## case not defined
        else:
            lgr.error(f"Dimensionality reduction '{reduction}' not defined!")
            return None
                        
        ## save and return     
        if store:
            self._X_dimred[_get_df_string(kind="xdimred", xdimred=reduction)] = _X_reduced
            if reduction in ["pca", "ica", "fa"]:
                self._dimred[reduction] = dict(
                    method=reduction,
                    n_components=_X_reduced.shape[0],
                    min_ev=min_ev,
                    loadings=loadings
                )
                if reduction in ["pca", "fa"]:
                    self._dimred[reduction]["ev"] = ev
                if reduction=="fa":
                    self._dimred[reduction]["fa_method"] = fa_method
                    self._dimred[reduction]["fa_rotation"] = fa_rotation
            self._set_last(X_reduction=reduction)

            ## return
            if self._return_self:
                return self
            
        if reduction in ["pca", "ica", "fa"]:
            return _X_reduced, ev, loadings
        else:
            return _X_reduced
        
    
    # CLEAN ========================================================================================
    
    def clean_y(self, how,
                covariates_within=None,
                covariates_between=None,
                protect=None,
                within_y_specific=False,
                combat=False, combat_protect=None, combat_keep=None,
                combat_train=None, combat_model=None, combat_kwargs=None,
                plot_design_between=False,
                n_proc=None, replace=True, verbose=None):
        """
        Regress covariates out of Y, "within" (across parcels, per map) and/or
        "between" (across maps/subjects, per parcel), with optional ComBat site
        harmonization for the between-subject case.

        "Within" regression removes a per-parcel confound from each Y map
        individually -- e.g. regressing a grey-matter probability map out of an
        MRI map so that the result reflects tissue-corrected signal rather than 
        partial-volume effects. "Between" regression removes subject/map-level 
        confounds shared across parcels -- e.g. age, sex, or scan site -- fit 
        and applied jointly across all parcels via one design matrix. 
        The two are independent and can be combined in one call.

        Parameters
        ----------
        how : str or list of str
            Which regression(s) to perform: ``"within"``, ``"between"``, or both.
        covariates_within : array-like or "z", optional
            Per-parcel covariate map(s) to regress out of each Y map. Only used if
            ``"within" in how``; ignored (with no regression performed) if
            ``None``. The literal string ``"z"``/``"Z"`` regresses the Z data
            provided at :meth:`fit` instead of an explicit array (raises if no Z
            was provided). A single covariate map is broadcast to every Y map
            unless ``within_y_specific=True``, in which case one covariate map per
            Y row is expected.
        covariates_between : array-like, Series, or DataFrame, optional
            Subject/map-level covariate(s) to regress out across parcels. Only
            used if ``"between" in how``; ignored if ``None``. Categorical columns
            (object/string/categorical dtype, or a column literally named
            ``"site"``) are one-hot encoded; continuous columns are used as-is.
        protect : array-like, Series, or DataFrame, optional
            Covariates to hold constant (partial out) while regressing
            ``covariates_between``, without themselves being removed from Y --
            typically group/subject design columns that should not be regressed
            away. Only relevant with ``"between"``.
        within_y_specific : bool, default False
            If True, ``covariates_within`` supplies one covariate map per Y row
            instead of a single map broadcast to all rows.
        combat : bool, default False
            Apply ComBat harmonization during the between-subject step. Requires
            a ``"site"`` column in ``covariates_between``; otherwise silently
            disabled with a warning. Requires the optional ``neuroHarmonize``
            package.
        combat_protect : array-like, Series, or DataFrame, optional
            Additional covariates ComBat should protect (preserve biological
            variance for) without using them as regression covariates.
        combat_keep : optional
            Deprecated and ignored; all regression covariates are now
            automatically protected during ComBat harmonization.
        combat_train : array-like of bool, optional
            Boolean vector marking a training subset: if valid, ComBat is fit only
            on this subset and applied to the rest. Ignored (full-sample fit) if
            the length doesn't match or values aren't boolean-like.
        combat_model : optional
            A previously fitted ComBat model to apply (rather than refit). If
            ``None``, a fresh model is fit and stored (together with the
            covariates used) on the object for later reuse.
        combat_kwargs : dict, optional
            Additional keyword arguments forwarded to neuroHarmonize's
            ``harmonizationLearn``.
        plot_design_between : bool, default False
            Plot the between-subject design matrix (diagnostic only, no effect on
            the result).
        n_proc : int, optional
            Number of parallel processes for the per-parcel/per-subject
            regression loops. Defaults to the value set at init.
        replace : bool, default True
            Overwrite ``self`` 's stored Y with the cleaned result. If False, the
            cleaned data is computed and returned but the object's Y is left
            untouched.
        verbose : bool, optional
            Print progress messages. Defaults to the value set at init.

        Returns
        -------
        pandas.DataFrame
            The cleaned Y data (same shape, columns, and index as the input Y).
        """
        from .core.clean_y import _clean_y_within, _clean_y_between
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        lgr.info("*** NiSpace.clean_y() - Y covariate regression. ***")
        self._check_fit()
        # TODO (first non-dev release): remove combat_keep parameter entirely
        if combat_keep is not None:
            lgr.warning(_DEPR_COMBAT_KEEP)
        n_proc = self._n_proc if n_proc is None else n_proc
        combat_kwargs = {} if combat_kwargs is None else combat_kwargs

        if isinstance(how, str):
            how = [how]
        if not isinstance(how, (list, tuple, set)) or \
                not all(h in ["within", "between"] for h in how):
            lgr.critical_raise(
                f"'how' must be (list of) 'within' and/or 'between', got {how!r}.", ValueError)

        Y = self._Y
        Y_arr = np.array(Y)
        did_within = did_between = False

        # within: regression across parcels per map
        if "within" in how and covariates_within is not None:
            lgr.info("Performing covariate regression within map/subjects (e.g., grey matter maps).")
            Y_arr, used_z, did_within = _clean_y_within(
                Y_arr=Y_arr,
                covariates_within=covariates_within,
                Z=self._Z,
                n_maps=Y.shape[0],
                n_parcels=Y.shape[1],
                within_y_specific=within_y_specific,
                n_proc=n_proc,
                dtype=self._dtype,
                verbose=verbose,
            )
            if used_z:
                self._clean_y_z = True

        # between: ComBat harmonization and/or regression across subjects
        if "between" in how and covariates_between is not None:
            lgr.info("Performing covariate regression between maps/subjects (e.g., age, sex, site).")
            Y_arr, combat_model, combat_covariates = _clean_y_between(
                Y_arr=Y_arr,
                covariates_between=covariates_between,
                n_subjects=Y.shape[0],
                protect=protect,
                combat=combat,
                combat_protect=combat_protect,
                combat_train=combat_train,
                combat_model=combat_model,
                combat_kwargs=combat_kwargs,
                plot_design_between=plot_design_between,
                n_proc=n_proc,
                dtype=self._dtype,
                verbose=verbose,
            )
            did_between = True
            if combat_model is not None:
                self._clean_y_combat_model = combat_model
                self._clean_y_combat_cov = combat_covariates

        if not did_within and not did_between:
            lgr.warning("No covariate regression performed! Set 'how' to 'between' and/or 'within' "
                        "and provide covariate arrays through 'covariates_{within|between}'!")

        Y = pd.DataFrame(Y_arr, columns=Y.columns, index=Y.index, dtype=self._dtype)
        if replace:
            self._Y = Y
        if self._return_self:
            return self
        return Y
    
    
    # TRANSFORM Y ==================================================================================    
        
    def transform_y(self, transform, groups=None, subjects=None, Y=None,
                    Y_name="Y", store=True, verbose=None):
        """
        Apply a group-comparison or aggregation formula to Y, turning per-subject
        or per-group raw maps into a single comparison/summary map (e.g. an effect
        size, a z-score, or a group mean) that colocalize() can then use in place
        of the raw Y data.

        Parameters
        ----------
        transform : str
            Formula string, e.g. ``"hedges(a,b)"``, ``"zscore(a,b)"``,
            ``"mean(y)"``. Supported formulas (``y`` = the whole input; ``a``/
            ``b`` = the two groups defined by ``groups``, smaller/
            alphabetically-first value -> ``a``):

            * ``y`` -- identity (no-op passthrough)
            * ``mean(y)``, ``median(y)``, ``std(y)``, ``var(y)`` -- summary
              statistic across rows, per parcel
            * ``elemdiff(a,b)`` / ``a-b`` -- elementwise ``a - b`` (paired,
              requires ``subjects``)
            * ``meandiff(a,b)`` / ``mean(a)-mean(b)`` -- difference of means
            * ``center(a,b)`` / ``a-mean(b)`` -- ``a`` centered on ``b``'s mean
            * ``cohen(a,b)`` -- Cohen's d, independent groups
            * ``pairedcohen(a,b)`` -- Cohen's d, paired/dependent groups
              (requires ``subjects``)
            * ``hedges(a,b)`` -- Hedges' g (bias-corrected Cohen's d)
            * ``zscore(a)`` / ``zscore(a,b)`` -- z-score of ``a`` against itself
              or against reference group ``b``
            * ``rzscore(a)`` / ``rzscore(a,b)`` -- robust (median/MAD) z-score;
              warns if the reference group has few observations (n<20/n<30)
            * ``prc(a,b)`` -- percent change ``(a-b)/a*100`` (paired, requires
              ``subjects``)
            * ``logfc(a,b)`` -- log fold-change (auto-shifted to stay defined for
              data that can be negative, e.g. already z-scored/residualized)
            * ``centile(a)`` / ``centile(a,b)`` -- percentile rank of ``a``
              within the reference distribution

            Note: ``"pairedhedges(a,b)"`` (paired/bias-corrected analogue of
            ``"hedges(a,b)"``) is not implemented -- calling it raises
            ``ValueError``.
        groups : array-like, optional
            2-level grouping vector, one entry per Y row. Required by any formula
            referencing ``a``/``b``; not needed for ``y``-only formulas (e.g.
            ``"mean(y)"``). Rows with NaN group labels are dropped with a
            warning.
        subjects : array-like, optional
            Subject/pair identifiers, one per Y row, used to match rows across
            groups ``a``/``b`` for paired formulas (``elemdiff``, ``pairedcohen``,
            ``prc``). Each ID must appear exactly once per group. If omitted for a
            paired formula, matched row order within each group is assumed
            (with a warning).
        Y : DataFrame, optional
            Data to transform. Defaults to the object's own Y data (``self._Y``);
            passing an explicit DataFrame lets this method operate on other data
            (this is how :meth:`transform_z` reuses it for Z).
        Y_name : str, default "Y"
            Cosmetic label used in log messages only (e.g. ``"Z"`` when called
            from :meth:`transform_z`); has no effect on the computation.
        store : bool, default True
            Store the transformed data and the resolved ``groups``/``subjects``
            on the object (so that :meth:`colocalize`, :meth:`get_y`, and
            :meth:`permute` can later default to this transform), and remember it
            as the "last" Y transform for future ``Y_transform=None`` calls.
        verbose : bool, optional
            Print progress messages. Defaults to the value set at init.

        Returns
        -------
        pandas.DataFrame
            The transformed data: one row per aggregate statistic for
            aggregate formulas (e.g. ``hedges``, ``cohen``), or one row per
            subject/map for row-preserving formulas (e.g. ``zscore``,
            ``centile``).
        """
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        lgr.info(f"*** NiSpace.transform_{Y_name.lower()}() - {Y_name} transformation and comparison. ***")
        
        ## check if fit was run
        self._check_fit()

        if self._binary_y and Y is None:
            lgr.warning(
                "binary_y=True: transform_y() is not meaningful for binary Y maps. "
                "Binary maps represent a fixed observed contrast and should not be "
                "transformed. Proceeding, but consider whether this is intended."
            )

        ## Y data
        if Y is None:
            _Y = self._Y
        else:
            _Y = Y
        if _Y.shape[0] <= 1:
            lgr.critical_raise(f"For {Y_name} transformation, data has to consist of more than "
                               f"one map ({_Y.shape[0]})!",
                               ValueError)            
        
        ## prepare groups
        if groups is not None:
            lgr.info("Groups/sessions vector provided, ensuring dummy-coding.")
            groups = np.array(groups).squeeze()
            # check if right length
            if len(groups) != _Y.shape[0]:
                lgr.critical_raise(f"Length of 'groups' ({len(groups)}) does not match length of "
                                   f"{Y_name} data ({_Y.shape[0]})!",
                                   ValueError)
            # drop nans
            groups_nanidc = pd.Series(groups).isnull().values
            groups_nonan = groups[~groups_nanidc]
            if len(groups)!=len(groups_nonan):
                lgr.warning(f"Variable 'group' contains {groups_nanidc.sum()} NaN values. "
                            f"These {Y_name} data will be dropped.")
            # get dummy-coded group variable
            groups_nonan_dummy = _dummy_code_groups(groups_nonan)
            
            # prepare Y
            _Y_nonan = _Y.loc[~groups_nanidc, :]
            
        else:
            _Y_nonan = _Y
            groups_nanidc, groups_nonan_dummy = None, None
            
        ## prepare subjects
        if subjects is not None and groups is not None:
            lgr.info("Subjects vector provided, validating.")
            subjects = np.array(subjects).squeeze()
            if len(subjects) != _Y.shape[0]:
                lgr.critical_raise(f"Length of 'subjects' ({len(subjects)}) does not match length of "
                                   f"{Y_name} data ({_Y.shape[0]})!",
                                   ValueError)
            subjects_nonan = subjects[~groups_nanidc]
            subjects_nonan_num = _num_code_subjects(subjects_nonan)
        else:
            subjects_nonan_num = None
                
        ## prepare formula
        transform = _lower_strip_ws(transform)
            
        ## apply the transform formula to the Y data
        lgr.info(f"Applying {Y_name} transform '{transform}'.")
        # get formula interpreter to evaluate string formulas and output dataframes
        apply_transform, paired = _get_transform_fun(transform, return_df=True, return_paired=True,
                                                     dtype=self._dtype, ignore_nan_warnings=True)
        # small-sample warnings for reference-group-normed transforms
        if any(t in transform for t in ("rzscore", "rzscores", "centile", "centiles")):
            if groups_nonan_dummy is not None:
                n_ref = (groups_nonan_dummy == 1).sum()
            else:
                n_ref = len(_Y_nonan)
            if "centile" in transform:
                if n_ref < 20:
                    lgr.warning(f"centile: reference group has only n={n_ref} subjects. "
                                f"Percentile ranks are coarse at small n — consider "
                                f"'zscore(a,b)' or, for a single summary map, 'hedges(a,b)'.")
                elif n_ref < 30:
                    lgr.info(f"centile: reference group has n={n_ref} subjects. "
                             f"Percentile ranks may be coarse at this sample size.")
            else:
                if n_ref < 20:
                    lgr.warning(f"rzscore: reference group has only n={n_ref} subjects. "
                                f"MAD-based normalization is unreliable at small n — consider "
                                f"'zscore(a,b)' or, for a single summary map, 'hedges(a,b)'.")
                elif n_ref < 30:
                    lgr.info(f"rzscore: reference group has n={n_ref} subjects. "
                             f"MAD estimates can be variable at this sample size.")
        # paired comparison but no subjects vector
        if paired and subjects is None:
            lgr.warning("The transform performs a paired comparison but argument 'subjects' was not "
                        "provided! Will assume that subjects order is equal in both sessions. To ensure "
                        "results validity, provide a 'subjects' vector.")
            subjects_nonan_num = np.zeros_like(groups_nonan_dummy)
            for g in [0, 1]:
                subjects_nonan_num[groups_nonan_dummy==g] = np.arange((groups_nonan_dummy==g).sum())
        # apply transform
        _Y_trans = apply_transform(y=_Y_nonan, groups=groups_nonan_dummy, subjects=subjects_nonan_num)
        _Y_trans = _Y_trans.astype(self._dtype)
        # test number of columns (= n parcels) of output
        if _Y.shape[1]!=_Y_trans.shape[1]:
            lgr.critical_raise(f"Transformed {Y_name} data of shape {_Y_trans.shape} "
                               f"(assuming {_Y_trans.shape[1]} parcels); "
                               f"this does not fit with input {Y_name} of shape {_Y.shape}!",
                               ValueError)
        
        ## save
        if store:
            # save groups/subjects
            self._groups = groups
            self._groups_nan_idc = groups_nanidc
            self._groups_no_nan = groups_nonan_dummy
            if subjects is not None:
                self._subjects = subjects
                self._subjects_no_nan = subjects_nonan_num
            # save transformed y
            df_str = _get_df_string("ytrans", ytrans=transform)
            self._Y_trans[df_str] = _Y_trans
            self._set_last(Y_transform=transform)

            ## return
            if self._return_self:
                return self
        return _Y_trans 

    
    # TRANSFORM Z ==================================================================================
    
    def transform_z(self, transform="Y", groups="Y", subjects="Y",
                    replace=True, verbose=None):
        """
        Apply a :meth:`transform_y`-style formula to the object's Z data instead
        of Y. A thin wrapper: internally calls ``transform_y(transform,
        Y=self._Z, Y_name="Z", store=False)``, so the transform is never
        remembered as the "last" Y transform and no per-transform history is
        kept for Z -- ``replace=True`` (the default) simply overwrites ``self``'s
        Z with the result.

        Parameters
        ----------
        transform : str, default "Y"
            Formula string, see :meth:`transform_y` for the full list. Unlike
            ``groups``/``subjects`` below, ``"Y"`` here is **not** a sentinel for
            "reuse the last Y transform" -- it is parsed as a literal formula,
            which normalizes (case-insensitively) to the identity formula, i.e.
            the default is "no transformation", not "whatever transform_y() last
            used".
        groups : array-like or "Y", default "Y"
            Grouping vector for the formula (see :meth:`transform_y`). The
            literal string ``"Y"`` (case-insensitive) is a real sentinel here:
            it reuses whichever ``groups`` vector was set by the last
            :meth:`transform_y` call (or ``None`` if none was set). Any other
            value is passed through unchanged.
        subjects : array-like or "Y", default "Y"
            Subject/pair identifiers for the formula (see :meth:`transform_y`).
            Same ``"Y"``-sentinel mechanism as ``groups``, reusing the last
            :meth:`transform_y` call's ``subjects``.
        replace : bool, default True
            Overwrite ``self`` 's stored Z with the transformed result. If False,
            the transformed data is computed and returned but the object's Z is
            left untouched.
        verbose : bool, optional
            Print progress messages. Defaults to the value set at init.

        Returns
        -------
        pandas.DataFrame
            The transformed Z data -- see :meth:`transform_y`'s Returns for the
            row-shape convention (aggregate vs. row-preserving formulas).
        """
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)

        # take groups from Y
        if isinstance(groups, str):
            if groups.lower() == "y":
                if hasattr(self, "_groups"):
                    groups = self._groups
                else:
                    groups = None
        
        # take subjects from Y
        if isinstance(subjects, str):
            if subjects.lower() == "y":
                if hasattr(self, "_subjects"):
                    subjects = self._subjects
                else:
                    subjects = None
        
        # apply y transform function to z data
        _Z_trans = self.transform_y(transform, groups=groups, subjects=subjects, Y=self._Z,
                                    Y_name="Z", store=False, verbose=verbose)
        
        # replace z data & return
        if replace:
            self._Z = _Z_trans
        
        ## return
        if self._return_self:
            return self
        return _Z_trans
    
    
    # COLOCALIZE ===================================================================================

    def colocalize(self, method=None, X_reduction=None, Y_transform=None, xsea=None,
                   xsea_aggregation_method="mean",
                   regress_z=True, zy_matched=False,
                   X=None, Y=None, Z=None,
                   store=True, n_proc=None, seed=None, verbose=None,
                   dist_mat_kwargs=None,
                   force_dict=False,
                   **kwargs):
        """
        Compute colocalization statistics between each X map (or set, if XSEA is
        active) and each Y map, optionally regressing Z out of X and/or Y first.
        This is the core computation step of the NiSpace pipeline, feeding
        :meth:`permute`, :meth:`correct_p`, and ``nispace.diagnostics.regional_influence``/
        ``regional_contribution``.

        Parameters
        ----------
        method : str, optional
            Colocalization method. Defaults to the last method used in
            :meth:`colocalize` (raises if none has ever been set). One of:

            * ``"pearson"`` -- Pearson correlation
            * ``"spearman"`` -- Pearson correlation on ranks
            * ``"partialpearson"`` -- Pearson with Z regressed out
            * ``"partialspearman"`` -- Spearman with Z regressed out (Z is also
              ranked, a "standard" partial-Spearman)
            * ``"mi"`` -- mutual information
            * ``"slr"`` -- simple linear regression (one X predictor at a time)
            * ``"mlr"`` -- multiple linear regression (all X maps as joint
              predictors). ``r2`` is well-calibrated for significance testing
              even under correlated predictors, but the per-predictor ``beta``
              is not (see ``"pls"`` below for the same caveat) -- correlated
              predictors share credit for the same explained variance, which
              shrinks each individual ``beta`` and can make ``beta``-based
              significance testing severely underpowered. Use ``beta`` for
              descriptive/interpretive ranking only.
            * ``"dominance"`` -- dominance analysis (partitions R² across
              predictors)
            * ``"pls"`` -- partial least squares regression. Also returns
              ``score_r`` (signed Pearson correlation between the component-1
              latent score and Y -- unlike ``r2``, which is always >= 0, this
              retains sign) and ``weight`` (unit-norm component-1 PLS weight
              vector per X predictor -- the "gene weight" reported in
              imaging-transcriptomics PLS studies), both reflecting component
              1 only regardless of ``n_components``. Both use a **median-vote
              sign convention** (reporting-only -- doesn't affect any
              p-value, which depends on magnitude alone): positive iff the
              *median* per-predictor correlation with Y is positive, i.e.
              "generally, high X corresponds to high Y, and vice versa" (the
              same phenotype-anchored idea GSEA uses to label a gene set
              enriched "positively"/"negatively", rather than an internal
              PCA/PLS solver convention), keeping ``score_r``'s sign
              meaningful and comparable across different XSEA sets/fits. For
              XSEA-style set testing, ``weight`` is nonetheless empirically
              **anti-conservative** (unlike ``mlr``'s ``beta``, which is
              conservative/underpowered) -- it behaves like an aggregated
              per-predictor correlation, not a joint model coefficient, so it
              inherits the naive intercorrelation-driven false-positive
              inflation that ``r2``/``score_r`` avoid. Prefer ``r2``/
              ``score_r`` for significance testing; ``beta``/``weight`` for
              descriptive/interpretive ranking, or for non-XSEA per-map fits
              where no set-level aggregation is involved.
            * ``"pcr"`` -- principal component regression. Also returns
              ``score_r`` (signed Pearson correlation between the first
              principal component's scores and Y, same median-vote sign
              convention as ``"pls"`` above), reflecting component 1 only
              regardless of ``n_components``.
            * ``"lasso"``, ``"ridge"``, ``"elasticnet"`` -- regularized
              regression with spatial (parcel-fold) cross-validation

        X_reduction : str, optional
            Label of a previously computed X dimensionality reduction (see
            :meth:`reduce_x`) to use instead of the raw X data. Defaults to the
            last one used (or the raw X data if none has been used).
        Y_transform : str, optional
            Label of a previously computed Y transform (see :meth:`transform_y`)
            to use instead of the raw Y data. Defaults to the last one used (or
            the raw Y data if none has been used). If this transform has not
            been computed yet, it is run automatically (using ``groups``/
            ``subjects`` from ``**kwargs`` if given) with a warning.
        xsea : bool, optional
            Aggregate X maps into sets before colocalizing (X-Set Enrichment
            Analysis) -- requires X to have a ``"set"`` MultiIndex level.
            Defaults to the last value used. Combined with a correlation
            ``method`` (``"pearson"``/``"spearman"``), this requires Fisher-z
            transformed correlations -- ``r_to_z=False`` in ``**kwargs`` is
            overridden to ``True`` with a warning.
        xsea_aggregation_method : str, default "mean"
            How to aggregate per-set colocalization statistics across a set's
            members when ``xsea`` is active: ``"mean"``, ``"median"``,
            ``"absmean"``, ``"absmedian"``, ``"weightedmean"``, or
            ``"weightedabsmean"`` (the weighted variants require a ``"weight"``
            MultiIndex level on X; fall back to unweighted with a warning if
            missing).
        regress_z : bool, default True
            Regress Z out of X and/or Y before colocalizing (requires Z to have
            been provided at :meth:`fit`; a no-op otherwise). Forced on for
            ``partial*`` methods. Defaults to the last value used.
        zy_matched : bool, default False
            Treat Z as having one map per Y row (rather than a single/shared Z
            used for every Y row) -- e.g. per-subject nuisance maps matched to
            per-subject Y maps. Incompatible with ``partial*`` methods (falls
            back to the corresponding non-partial method with a warning).
            Defaults to the last value used.
        X, Y, Z : array-like or DataFrame, optional
            Explicit data overriding the object's own fitted X/Y/Z (or the
            resolved ``X_reduction``/``Y_transform``). Rarely needed.
        store : bool, default True
            Store the result on the object (accessible via
            :meth:`get_colocalizations`), and remember ``method``,
            ``X_reduction``, ``Y_transform``, ``xsea``, ``regress_z``, and
            ``zy_matched`` as the "last used" settings for subsequent calls with
            unset (``None``) arguments.
        n_proc : int, optional
            Number of parallel processes (one per Y row). Defaults to the value
            set at init.
        seed : int, optional
            Random seed forwarded to the regularized-regression methods'
            cross-validation splitting. Not persisted across calls. Defaults to
            the seed set at init (``NiSpace(seed=...)``) if not given here.
        verbose : bool, optional
            Print progress messages. Defaults to the value set at init.
        dist_mat_kwargs : dict, optional
            Only used for ``method in {"lasso", "ridge", "elasticnet"}``.
            Recognized keys: ``parcel_tr_te_splits`` (pre-computed spatial CV
            splits), ``euclidean_dist_mat`` (pre-computed distance matrix),
            ``parcel_train_pct`` (default 0.75). Remaining keys are forwarded to
            the internal distance-matrix computation.
        force_dict : bool, default False
            Always return a dict even when the method produces a single
            statistic (e.g. ``"pearson"``'s ``rho``).
        **kwargs
            ``groups``, ``subjects`` : optional
                Forwarded to :meth:`transform_y` if ``Y_transform`` needs to be
                auto-run (see above); unused otherwise.
            ``rank`` : bool, optional
                Rank-transform X (and Y) before colocalizing. Forced True for
                ``"spearman"``/``"partialspearman"``; otherwise defaults to
                False. Deliberately *not* resolved from a prior call's setting
                (unlike the parameters above), since inheriting it across a
                ``method`` change would silently mislabel results.

            Other recognized keys are forwarded to the underlying
            colocalization function: ``r_to_z`` (bool, default True -- Fisher-z
            transform correlation coefficients), ``r_equal_one`` (default
            ``"raise"`` -- behavior when a correlation is exactly 1),
            ``adj_r2`` (bool, default True -- adjusted vs. raw R² for
            slr/mlr/dominance/pcr), ``mlr_individual`` (bool, default False --
            compute per-predictor unique-R² contributions for ``"mlr"``),
            ``n_components`` (int, default 1 -- for ``"pls"``/``"pcr"``),
            ``n_neighbors`` (for ``"mi"``), and sklearn ``Lasso``/``Ridge``/
            ``ElasticNet`` keyword arguments for the regularized methods.

        Returns
        -------
        pandas.DataFrame or dict of pandas.DataFrame
            X labels (or set names, if ``xsea``) as columns, Y labels as rows.
            A dict of ``{stat: DataFrame}`` is returned when the method produces
            more than one statistic (e.g. ``"mlr"``) or when ``force_dict=True``.
            For ``method in {"pearson", "spearman", "partialpearson",
            "partialspearman"}``, the ``"rho"`` stat is Fisher-z-transformed
            (``numpy.arctanh``) by default (``r_to_z=True`` in ``**kwargs``) --
            it is *not* a raw, bounded [-1, 1] correlation coefficient unless
            ``r_to_z=False`` was passed. The key/column is always named
            ``"rho"`` regardless of this setting, so the scale must be tracked
            separately (e.g. via ``self._coloc_kwargs_by_method[method]
            ["r_to_z"]``) -- every other stat (``"r2"``, ``"mi"``, ``"beta"``,
            ...) is unaffected by ``r_to_z``.
        """
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        lgr.info("*** NiSpace.colocalize() - Estimating X & Y colocalizations. ***")
        
        # kwargs
        dist_mat_kwargs = {} if dist_mat_kwargs is None else dist_mat_kwargs
        
        ## check if fit was run 
        self._check_fit()
        
        ## settings
        n_proc = self._n_proc if n_proc is None else n_proc
        seed = self._seed if seed is None else seed
        dtype = self._dtype

        ## settings
        # NOTE: `rank` is intentionally NOT resolved via _get_last -- unlike X_reduction/
        # Y_transform (which legitimately should persist across calls), rank must default
        # to a function of *this call's* method, not whatever a previous, possibly
        # different-method call last used. Falling back to the stored last value here
        # was a real bug: colocalize(method="pearson") run after colocalize(method=
        # "spearman") would silently inherit rank=True and compute Spearman's correlation
        # while labeling it "pearson".
        rank = kwargs.pop("rank", None)
        method, X_reduction, Y_transform, xsea, zy_matched, regress_z = self._get_last(
            method=method,
            X_reduction=X_reduction,
            Y_transform=Y_transform,
            xsea=xsea,
            zy_matched=zy_matched,
            regress_z=regress_z,
        )
        if method is None:
            coloc_methods = ", ".join(list(_COLOC_METHODS.keys()))
            lgr.critical_raise(f"No colocalization method defined! Supported:\n{coloc_methods}",
                               ValueError)
        else:
            lgr.info(f"Running '{method}' colocalization" + \
                     (f" on '{X_reduction}'-reduced X data" if X_reduction else "") + \
                     (f" with '{Y_transform}' transform" if Y_transform else "") + ".")
        if rank is None:
            rank = False
        if "spearman" in method:
            rank = True

        if self._binary_y and method in {"spearman", "partialspearman"}:
            lgr.warning(
                f"binary_y=True: Ranked colocalization method '{method}' is not recommended "
                "for binary Y data. Parcellated binary maps have many tied zeros; rank "
                "transforms degrade sensitivity. Use 'pearson' for an approximate "
                "point-biserial correlation."
            )

        ## get X and Y data (so this function can be run on direct X & Y input data)
        # NOTE: this X/Y/Z resolution block is mirrored (not shared) by
        # nispace/diagnostics.py's _resolve_coloc_data() helper, used by
        # regional_influence()/regional_contribution()/local_colocalization(). Deliberately
        # not unified with that helper -- this block also auto-runs transform_y(), mutates
        # XSEA state, and handles the regularized-regression CV-split branch below, none of
        # which _resolve_coloc_data() needs to reproduce.
        # X
        if X is None:
            if not X_reduction:
                X = self._X
            else:
                with _quiet():
                    X = self.get_x(X_reduction=X_reduction)
        X_arr = np.array(X, dtype=dtype)
        X_weights = None
        if xsea:
            lgr.info("Will perform X-set enrichment analysis (XSEA).")
            if not isinstance(X, pd.DataFrame):
                lgr.critical_raise("XSEA requires X data to be a pandas DataFrame!",
                                   TypeError)
            if not isinstance(X.index, pd.MultiIndex):
                lgr.critical_raise("XSEA requires X data to have a MultiIndex!",
                                   TypeError)
            if "set" not in X.index.names:
                lgr.critical_raise("XSEA requires X data to have a MultiIndex with a 'set' level!",
                                   ValueError)
            if "weighted" in xsea_aggregation_method and "weight" not in X.index.names:
                lgr.warning("Weighted XSEA requires X data to have a MultiIndex with a 'weight' level! "
                            "Will not use weights.")
                xsea_aggregation_method = xsea_aggregation_method.replace("weighted", "")
            X_arr = {set_name: np.array(set_X, dtype=self._dtype) 
                     for set_name, set_X in X.groupby(level="set", sort=False)}
            if "weighted" in xsea_aggregation_method:
                X_weights = {set_name: np.array(set_X.index.get_level_values("weight"), dtype=self._dtype) 
                             for set_name, set_X in X.groupby(level="set", sort=False)}
            lgr.info(f"Using {len(X_arr)} sets with between "
                     f"{X.index.get_level_values('set').value_counts().min()} and "
                     f"{X.index.get_level_values('set').value_counts().max()} samples. "
                     f"Aggregating within-set colocalizations with: {xsea_aggregation_method}.")
            if ("spearman" in method or "pearson" in method):
                if "r_to_z" in kwargs:
                    if kwargs["r_to_z"] is False:
                        lgr.warning("XSEA with correlation colocalization requires Fisher's Z "
                                    "transform! Will set 'r_to_z' = True.")
                        kwargs["r_to_z"] = True
            self._xsea = True
            self._xsea_aggregation_method = xsea_aggregation_method
            
        # Y
        groups = kwargs.pop("groups", None)
        subjects = kwargs.pop("subjects", None)
        if Y is None:
            if not Y_transform:
                Y = self._Y
            else:
                if not self._check_transform(ytrans=Y_transform, raise_error=False):
                    lgr.warning(f"Y transform '{Y_transform}' was not run before. Running now.")
                    self.transform_y(Y_transform, groups, subjects)
                with _quiet():
                    Y = self.get_y(Y_transform=Y_transform)
        Y_arr = np.array(Y, dtype=dtype)

        # Z
        if Z is None:
            Z = self._Z
        # if regress_z is True, we regress Z from X and Y, if None or False, we don't regress Z
        if Z is None or regress_z is None or regress_z == False:
            regress_z = ""
        if regress_z == True:
            regress_z = "xy"
        # partialspearman and partialpearson entail full Z regression
        if regress_z and "partial" in method:
            if Z is None:
                lgr.error(f"Provide Z data for method '{method}'! Using method "
                          f"'{method.replace('partial', '')}' instead.") 
                method = method.replace("partial", "")
                regress_z = ""
            elif hasattr(self, "_clean_y_z"):
                lgr.warning("It seems, Z-from-Y-regression was performed before. Will add X-from-Z-regression.")
                regress_z = "x"
            else:
                regress_z = "xy"
        # if zy_matched is True, we regress each Z from each Y, we cannot regress from X in that case
        if regress_z and zy_matched:
            if Z.shape[0] != Y_arr.shape[0]:
                lgr.error(f"Number of Z maps ({Z.shape[0]}) must equal number of Y maps " + \
                          f"({Y_arr.shape[0]}) if 'zy_matched' is True! Will not perform Z regression.")
                zy_matched, regress_z = False, ""
            elif "partial" in method:
                lgr.warning(f"Method '{method}' is not compatible with matched Z->Y regression. "
                            f"Will perfom semi-partial '{method.replace('partial', '')}' correlation.")
                method = method.replace("partial", "")
                regress_z = "y"
            else:
                regress_z = "y"
        # if regression was performed in .clean_y(), we avoid to touch y again
        if hasattr(self, "_clean_y_z"):
            lgr.warning(f"It seems, Z-from-Y-regression was performed before. "
                        f"{'Will only perform Z-from-X-regression.' if regress_z else 'Skipping regression.'}")
            regress_z = regress_z.replace("y", "")
        # if regress_z is still not False, we proceed
        Z_arr = np.array(Z, dtype=dtype) if regress_z else None
        # standard partial Spearman correlation ranks X, Y, AND Z before partial-
        # correlating (not just X/Y) -- without this, Z stays on its raw scale while
        # X/Y are ranked, giving a materially different (non-standard) result whenever
        # Z's distribution isn't already rank-equivalent to raw (e.g. skewed Z).
        if rank and Z_arr is not None:
            Z_arr = rank2d(Z_arr.T).T
            
        ## Preranking and regression
        if rank or regress_z:
            if rank:
                lgr.info("Pre-ranking X and Y data.")
            if regress_z:
                lgr.info(f"Regressing {Z_arr.shape[0]} {'Y-matched ' if zy_matched else ''}Z maps from "
                         f"{regress_z.upper()} data.")
            # X
            X_arr = _rank_regress(
                arr=X_arr, 
                rank=rank, 
                regress="x" in regress_z, 
                z=Z_arr, 
                zy_matched=zy_matched, 
                verbose=verbose
            )
            # Y
            Y_arr = _rank_regress(
                arr=Y_arr, 
                rank=rank, 
                regress="y" in regress_z, 
                z=Z_arr, 
                zy_matched=zy_matched, 
                verbose=verbose
            )
            
        ## special case regularized regression: we need euclidean distance matrices
        parcel_tr_te_splits, parcel_train_pct = None, None
        if (method in ["lasso", "ridge", "elasticnet"]):
            
            parcel_tr_te_splits = dist_mat_kwargs.pop("parcel_tr_te_splits", None)
            euclidean_dist_mat = dist_mat_kwargs.pop("euclidean_dist_mat", None)
            parcel_train_pct = dist_mat_kwargs.pop("parcel_train_pct", 0.75)
            
            if parcel_tr_te_splits is None:
                lgr.info("Fetching euclidean distance matrix for regularized regression "
                         "colocalization with n(parcel)-fold CV.")
                
                if self._zscore:
                    lgr.warning("Input data was Z-standardized, which might lead to leakage in CV!")
                
                if euclidean_dist_mat is None:
                    euclidean_dist_mat = self._get_dist_mat(
                        dist_mat_type="cv", 
                        n_proc=n_proc,
                        **dist_mat_kwargs
                    )
                    
                if any([s in (self._parc._space or "").lower() for s in ["mni", "fsa"]]):
                    lgr.info("Calculating distance-dependent parcel splits.")
                    parcel_tr_te_splits = _get_dist_dep_splits(
                        dist_mat=euclidean_dist_mat[np.ix_(self._no_nan, self._no_nan)],
                        train_pct=parcel_train_pct
                    )
                else:
                    lgr.warning("Calculating random parcel splits as parcellation space not supported.")
                    parcel_tr_te_splits = _get_rand_splits(
                        n_obs=int(self._no_nan.sum()),
                        train_pct=parcel_train_pct,
                        seed=seed
                    )

        # save colocalization settings
        self._coloc_kwargs = dict(
            xsea=xsea,
            xsea_method=xsea_aggregation_method,
            parcel_train_pct=parcel_train_pct,
            parcel_tr_te_splits=parcel_tr_te_splits,
            parcel_mask_regularized=self._no_nan.copy(),
            **kwargs
        )
        if self._x_with_self:
            self._coloc_kwargs["r_equal_one"] = np.nan
        
        ## get function to perform colocalization for one y vector/row (= per subject), needed for parallelization
        _y_colocalize = _get_colocalize_fun(
            method=method,
            seed=seed, 
            verbose=verbose, 
            dtype=dtype, 
            **self._coloc_kwargs
        )

        ## run actual prediction using joblib.Parallel
        _colocs_list = Parallel(n_jobs=n_proc)(
            delayed(_y_colocalize)(X_arr, Y_arr[i_y, :], X_weights) \
                for i_y in tqdm(
                    range(Y.shape[0]), 
                    desc=f"Colocalizing ({method}, {n_proc} proc)", 
                    disable=not verbose
                )
        )
        
        ## sort output with helper function, return as df
        _colocs = _sort_colocs(
            method=method, 
            xsea=xsea,
            y_colocs_list=_colocs_list, 
            n_X=len(X_arr),
            n_Y=Y.shape[0],
            return_df=True, 
            labs_X=X.index if not xsea else X_arr.keys(), 
            labs_Y=Y.index, 
            #n_components=n_components,
            dtype=dtype
        )
        
        ## store & return
        if store:
            # save output
            for stat in _colocs:
                df_str = _get_df_string("coloc", xdimred=X_reduction, ytrans=Y_transform, 
                                        method=method, stat=stat, xsea=xsea)
                self._colocs[df_str] = _colocs[stat]
            # save coloc. function
            self._colocs_fun[method] = _y_colocalize
            # save per-method settings (needed by diagnostics.regional_influence(), which --
            # unlike permute() -- can't rely solely on the closure since it does its own math
            # rather than calling _y_colocalize). rank/regress_z/zy_matched are the fully
            # *resolved* values (post the partial-method-fallback/zy_matched-mismatch
            # branches above) -- diagnostics.regional_influence() reads them back verbatim
            # rather than re-deriving them, since re-deriving independently is exactly what
            # caused the rank staleness bug fixed above.
            self._coloc_kwargs_by_method[method] = self._coloc_kwargs.copy()
            self._coloc_kwargs_by_method[method]["rank"] = rank
            self._coloc_kwargs_by_method[method]["regress_z"] = regress_z
            self._coloc_kwargs_by_method[method]["zy_matched"] = zy_matched
            # save last settings
            self._set_last(
                method=method,
                X_reduction=X_reduction,
                Y_transform=Y_transform,
                xsea=xsea,
                rank=rank,
                zy_matched=zy_matched,
                regress_z=regress_z,
            )
            
            ## return
            if self._return_self:
                return self
        # return dict of dfs
        if force_dict or len(_colocs) > 1:
            return _colocs
        else:
            return _colocs[list(_colocs.keys())[0]]


    # CORRELATE WITHIN REGION ======================================================================

    def correlate_within_region(self, X=None, Y=None, method="pearson",
                                X_reduction=None, Y_transform=None,
                                n_perm=1000, seed=None, r_to_z=True, store=True, verbose=None):
        """
        Per-parcel, across-subject correlation between X and Y -- the transpose
        of :meth:`colocalize` (which correlates across parcels, within a
        subject/map). For each parcel independently: do maps with a higher X
        value at this parcel also have a higher Y value at this parcel, across
        the set of X/Y maps (e.g. subjects)?

        By default, operates on the object's stored X/Y (via :meth:`get_x`/
        :meth:`get_y`, respecting ``X_reduction``/``Y_transform``). Pass ``X``/
        ``Y`` directly to override with different data -- including a 1D,
        subject-length vector (e.g. an external covariate like age), which is
        broadcast against every parcel of the other (2D) side.

        If the object's stored X/Y is used (not overridden) and was
        constructed with ``standardize`` including ``"x"``/``"y"``, a warning
        is raised: that z-scores each *map* across its own parcels, which is
        the right normalization for :meth:`colocalize`'s across-parcel axis,
        but distorts the across-subject axis this method actually correlates
        along (each map/subject would get its own rescaling before the
        per-parcel comparison).

        A binary (two-level, e.g. 0/1) 1D vector is a common special case of
        this: with ``method="pearson"``, the per-parcel ``rho`` is exactly the
        point-biserial correlation, which converts deterministically to the
        classic pooled-variance (Student's) two-sample t-statistic via
        ``t = rho * sqrt((n-2)/(1-rho**2))`` -- so it recovers the same
        per-parcel effect ranking as an independent-samples t-test. The
        default permutation null (subject/group-label pairing shuffled, group
        sizes preserved since the labels themselves aren't resampled) is
        exactly the classical nonparametric permutation test for two
        independent samples -- valid without the normality/equal-variance
        assumptions the parametric t-test needs, since it only relies on
        exchangeability under the true null. Combined with ``maxT``/
        ``step_maxT`` (see :meth:`get_within_region_correlations`), this gives
        a mass-univariate, FWER-controlled group-difference test per parcel --
        distinct from :meth:`transform_y`'s ``hedges(a,b)``/:meth:`colocalize`
        route, which tests whether the *shape* of a group-difference map
        matches other maps, not per-parcel significance of the difference
        itself. Note that ``rho``'s sign depends on which group is coded
        higher.

        The null distribution is built by permuting map/subject identity
        (which X row pairs with which Y row) -- not a spatial/spin null, since
        parcels are not the resampled axis here. The same permutation is
        applied consistently across all parcels within one iteration, which is
        what makes maxT-style FWER correction (via
        :meth:`get_within_region_correlations`) valid.

        Parameters
        ----------
        X, Y : array-like, DataFrame, Series, or None
            Override data. 2D input must be shape (n_subjects, n_parcels); 1D
            input must be length n_subjects (broadcast across parcels). A
            (n_subjects, 1) 2D input (e.g. a single-column DataFrame) is
            treated the same as 1D. At least one of the (possibly-defaulted)
            X/Y must be 2D with more than one column. Defaults (``None``) to
            the object's stored ``get_x()``/``get_y()`` output.
        method : {"pearson", "spearman"}, default "pearson"
        X_reduction : str, optional
            Which stored X reduction to use when ``X`` is not given directly
            (see :meth:`reduce_x`). Defaults to the last one used, or raw X.
        Y_transform : str, optional
            Which stored Y transform to use when ``Y`` is not given directly
            (see :meth:`transform_y`). Defaults to the last one used, or raw Y.
        n_perm : int, default 1000
            Number of map/subject-identity permutations for the null. 0/None
            skips the null (rho only, no p-values).
        seed : int, optional
            Defaults to the seed set at init (``NiSpace(seed=...)``).
        r_to_z : bool, default True
            Fisher-z-transform (``numpy.arctanh``) the returned/stored ``rho``
            (and its null) -- **to align with the rest of the toolbox's
            convention** (:meth:`colocalize`'s own ``r_to_z=True`` default):
            a correlation coefficient computed anywhere in NiSpace is on the
            same scale by default, whether or not it ever gets aggregated.
            This does not change any p-value (the comparison is invariant to
            this monotonic transform) -- it only changes the scale ``"stat"``
            itself is expressed on. :meth:`get_within_region_correlations_omnibus`
            transparently accounts for this scale when averaging across
            parcels, regardless of what ``r_to_z`` was used here.
        store : bool, default True
            Store the result (accessible via :meth:`get_within_region_correlations`)
            and remember these settings as "last used".
        verbose : bool, optional
            Print progress messages. Defaults to the value set at init.

        Returns
        -------
        NiSpace or pandas.DataFrame
            ``self`` if ``self._return_self`` (default), else the observed
            per-parcel correlation as a one-row DataFrame.
        """
        loglevel = lgr.getEffectiveLevel()
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        lgr.info("*** NiSpace.correlate_within_region() - per-parcel, across-subject correlation. ***")

        self._check_fit()
        seed = seed if seed is not None else self._seed

        if method not in _CWR_METHODS:
            lgr.critical_raise(f"'method' must be one of {_CWR_METHODS}, got '{method}'!",
                               ValueError)

        # resolve X: explicit override, or the object's stored X
        if X is None:
            if "x" in self._zscore:
                lgr.warning(
                    "X was Z-standardized (standardize='...x...'), which z-scores each map "
                    "across parcels -- correlate_within_region() correlates across maps/"
                    "subjects instead, so this rescales each one differently before that "
                    "comparison and distorts the result. Consider NiSpace(standardize=...) "
                    "without 'x', or pass a raw X= override here."
                )
            with _quiet():
                X_df = self.get_x(X_reduction=X_reduction)
            x_arr, x_index, x_columns = X_df.values, X_df.index, X_df.columns
        else:
            x_arr = np.asarray(X.values if isinstance(X, (pd.Series, pd.DataFrame)) else X,
                               dtype=float)
            x_index = X.index if isinstance(X, (pd.Series, pd.DataFrame)) else None
            x_columns = X.columns if isinstance(X, pd.DataFrame) else None

        # resolve Y: explicit override, or the object's stored Y
        if Y is None:
            if "y" in self._zscore:
                lgr.warning(
                    "Y was Z-standardized (standardize='...y...'), which z-scores each map "
                    "across parcels -- correlate_within_region() correlates across maps/"
                    "subjects instead, so this rescales each one differently before that "
                    "comparison and distorts the result. Consider NiSpace(standardize=...) "
                    "without 'y', or pass a raw Y= override here."
                )
            with _quiet():
                Y_df = self.get_y(Y_transform=Y_transform)
            y_arr, y_index, y_columns = Y_df.values, Y_df.index, Y_df.columns
        else:
            y_arr = np.asarray(Y.values if isinstance(Y, (pd.Series, pd.DataFrame)) else Y,
                               dtype=float)
            y_index = Y.index if isinstance(Y, (pd.Series, pd.DataFrame)) else None
            y_columns = Y.columns if isinstance(Y, pd.DataFrame) else None

        # subject/map alignment: index-based when available, else positional (with a warning)
        if x_arr.shape[0] != y_arr.shape[0]:
            lgr.critical_raise(f"X has {x_arr.shape[0]} subjects/maps, Y has {y_arr.shape[0]} "
                               "-- counts must match.",
                               ValueError)
        if x_index is not None and y_index is not None:
            if list(x_index) != list(y_index):
                lgr.warning("X and Y subject/map labels do not match. Pairing is done "
                            "positionally (same row order assumed for X and Y).")
        else:
            lgr.warning("X and/or Y has no subject/map index; assuming positional order "
                        "matches (same row order for X and Y).")

        parcel_labels = x_columns if x_columns is not None else y_columns

        rho, null = correlate_within_region_core(x_arr, y_arr, method=method, n_perm=n_perm,
                                                  seed=seed, r_to_z=r_to_z)

        if parcel_labels is None:
            parcel_labels = [f"parcel{i}" for i in range(len(rho))]
        _rho_df = pd.DataFrame([rho], index=[method], columns=parcel_labels, dtype=self._dtype)

        if store:
            X_reduction, Y_transform = self._get_last(X_reduction=X_reduction,
                                                       Y_transform=Y_transform)
            _key = _get_df_string("corrwithin", xdimred=X_reduction, ytrans=Y_transform,
                                  method=method)
            self._corr_within[_key] = _rho_df
            if null is not None:
                self._nulls.setdefault("corr_within", {})[_key] = {
                    "null_dist": null,
                    "n_perm": n_perm,
                    "seed": seed,
                    "r_to_z": r_to_z,
                }
                p = _cwr_null_p(rho, null)
                self._p_corr_within[_key] = pd.DataFrame([p], index=[method],
                                                         columns=parcel_labels, dtype=self._dtype)
            else:
                # avoid stale p/null from a prior call under the same key (same
                # xdimred/ytrans/method) contaminating a fresh n_perm=0 result
                self._p_corr_within.pop(_key, None)
                self._nulls.get("corr_within", {}).pop(_key, None)
            self._set_last(cwr_method=method, X_reduction=X_reduction, Y_transform=Y_transform)

            lgr.setLevel(loglevel)
            if self._return_self:
                return self
            return _rho_df

        lgr.setLevel(loglevel)
        return _rho_df

    # ----------------------------------------------------------------------------------------------

    def get_within_region_correlations(self, method=None, mc_method="step_maxT",
                                    X_reduction=None, Y_transform=None, verbose=None):
        """
        Retrieve per-parcel, across-subject correlation results from
        :meth:`correlate_within_region`, optionally multiple-comparison
        corrected across parcels. See :meth:`get_within_region_correlations_omnibus`
        for a single global test across all parcels instead of one p-value per
        parcel.

        Parameters
        ----------
        method : {"pearson", "spearman"}, optional
            Defaults to the last one used.
        mc_method : {"fdr_bh", "bonferroni", "holm", "maxT", "step_maxT"}, optional
            Multiple-comparison correction across parcels. Defaults to
            ``"step_maxT"`` -- the recommended, proven-calibrated choice (see
            below and ``bench5-1_region_correlation_fpr.ipynb``); pass
            ``None`` explicitly to get uncorrected results only
            (``"p_corr"`` then stays ``None``). Requires
            :meth:`correlate_within_region` to have been run with
            ``n_perm > 0`` -- raises ``KeyError`` otherwise, including under
            this default, since silently falling back to uncorrected results
            would hide that no null was ever computed. Unlike :meth:`correct_p`
            (which is specific to :meth:`colocalize` results), this dispatches
            directly to the underlying ``nispace.stats.misc`` primitives.

            ``"maxT"``/``"step_maxT"`` (:cite:`westfall1993`) control the
            family-wise error rate using the *same* subject-permutation null
            already generated by :meth:`correlate_within_region` (requires
            ``n_perm > 0`` there): for each permutation, the **maximum**
            ``|rho|`` across *all* parcels is taken, giving one "how extreme
            can the single most extreme parcel get under H0" draw per
            permutation. A parcel's corrected p-value is the fraction of
            these per-permutation maxima that meet or exceed its own
            observed ``|rho|``. Because the max is taken jointly across
            parcels within each permutation, whatever correlation exists
            between parcels' test statistics (e.g. from spatial
            autocorrelation in X/Y) is preserved automatically -- no
            independence assumption is made, unlike ``"bonferroni"``/
            ``"fdr_bh"``. ``"step_maxT"`` refines this by excluding
            already-more-extreme parcels from the max at each step, which
            can reject more parcels than plain ``"maxT"`` when several true
            effects are present -- but the two are mathematically identical
            on whether *any* parcel is rejected (both reduce to the same
            top-ranked-parcel computation).

            ``"meff"`` (Sidak correction via an effective-number-of-
            independent-tests estimate) is **not supported** here: it needs
            a parcel-parcel correlation matrix estimated from only
            ``n_subjects`` observations, typically far fewer than
            ``n_parcels`` for this method, which badly underestimates true
            dimensionality and is anti-conservative (see
            ``bench5-1_region_correlation_fpr.ipynb``). Use ``"maxT"``/
            ``"step_maxT"`` instead.
        X_reduction, Y_transform : str, optional
            Which stored X/Y to have used. Defaults to the last one used.
        verbose : bool, optional
            Print progress messages. Defaults to the value set at init.

        Returns
        -------
        dict
            ``{"stat_type": "rho", "mc_method": mc_method, "stat": DataFrame,
            "p": DataFrame or None, "p_corr": DataFrame or None}``. ``"p"`` is
            ``None`` if :meth:`correlate_within_region` was run with
            ``n_perm=0``; ``"p_corr"`` is ``None`` unless ``mc_method`` is
            given. ``"stat"`` is on whatever scale that call's own ``r_to_z``
            produced (Fisher-z by default, as of that method's own default) --
            same key name (``"rho"``) either way, see :meth:`correlate_within_region`.
        """
        loglevel = lgr.getEffectiveLevel()
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)

        cwr_method, X_reduction, Y_transform = self._get_last(
            cwr_method=method, X_reduction=X_reduction, Y_transform=Y_transform
        )
        _key = _get_df_string("corrwithin", xdimred=X_reduction, ytrans=Y_transform,
                              method=cwr_method)

        try:
            rho_df = self._corr_within[_key]
        except KeyError:
            available = "\n".join(list(self._corr_within.keys()))
            lgr.critical_raise(
                f"No correlate_within_region result for method='{cwr_method}', "
                f"X_reduction='{X_reduction}', Y_transform='{Y_transform}'. Did you run "
                f"NiSpace.correlate_within_region()? Available:\n{available}",
                KeyError
            )

        has_p = _key in self._p_corr_within
        if mc_method is not None and not has_p:
            lgr.critical_raise(
                f"No p-values stored for '{_key}' -- 'mc_method' requires n_perm > 0 to "
                "have been used in correlate_within_region().",
                KeyError
            )

        p_df, p_corr_df = None, None
        if has_p:
            p_df = self._p_corr_within[_key]

            if mc_method is not None:
                if mc_method in ("maxT", "step_maxT"):
                    null_entry = self._nulls.get("corr_within", {}).get(_key)
                    if null_entry is None:
                        lgr.critical_raise(
                            f"No null distribution stored for '{_key}' -- maxT/step_maxT "
                            "correction requires n_perm > 0 in correlate_within_region().",
                            KeyError
                        )
                    null_dist = null_entry["null_dist"]  # (n_perm, n_parcels)
                    obs_abs = np.abs(rho_df.values)[0]      # (n_parcels,)
                    null_abs = np.abs(null_dist)            # (n_perm, n_parcels)
                    # a parcel with no observed rho (entirely missing data) has no
                    # defined statistic to correct -- exclude it from the maxT/
                    # step_maxT family entirely (so it can't distort other parcels'
                    # max-statistic null either) and give it NaN, not a fake p from
                    # 'NaN >= x' silently evaluating to False everywhere
                    valid = ~np.isnan(obs_abs)
                    counts = np.full(obs_abs.shape, np.nan)
                    if valid.any():
                        obs_v = obs_abs[valid]
                        null_v = null_abs[:, valid]
                        if mc_method == "maxT":
                            with np.errstate(invalid="ignore"):
                                null_max = np.nanmax(null_v, axis=1)   # (n_perm,)
                            counts_v = np.mean(null_max[:, np.newaxis] >= obs_v[np.newaxis, :], axis=0)
                        else:  # step_maxT
                            order = np.argsort(obs_v)[::-1]
                            obs_s = obs_v[order]
                            # NaN would poison np.maximum.accumulate's running max for every
                            # step after it appears -- substitute -inf so a sporadic per-
                            # permutation NaN (too few valid pairs in that one shuffle) can
                            # never win the max, without breaking the accumulation
                            null_s = np.where(np.isnan(null_v[:, order]), -np.inf, null_v[:, order])
                            null_rev_cummax = np.maximum.accumulate(null_s[:, ::-1], axis=1)[:, ::-1]
                            p_s = np.maximum.accumulate(np.mean(null_rev_cummax >= obs_s[np.newaxis, :], axis=0))
                            counts_v = np.empty_like(p_s)
                            counts_v[order] = p_s
                        counts[valid] = counts_v
                    # same floor-clip convention as the raw per-parcel p (_cwr_null_p) so
                    # 'corrected >= raw' holds by construction, not just approximately
                    n_perm_used = null_entry["n_perm"]
                    floor = max(np.finfo(float).eps, 1.0 / n_perm_used)
                    with np.errstate(invalid="ignore"):
                        p_corr = np.clip(counts, floor, 1.0 - floor)
                    p_corr_df = pd.DataFrame([p_corr], index=p_df.index, columns=p_df.columns,
                                             dtype=self._dtype)

                elif mc_method == "meff":
                    # Deliberately unsupported here, not just unimplemented: with
                    # n_subjects typically far fewer than n_parcels, the parcel-parcel
                    # correlation matrix meff needs is rank-deficient (rank capped at
                    # n_subjects-1 regardless of orientation), so meff badly
                    # underestimates true effective dimensionality and gives an
                    # anti-conservative (too liberal) correction -- confirmed empirically
                    # across n_subjects=10-100 in bench5-1_region_correlation_fpr.ipynb
                    # (FWER 11-55% at nominal alpha=0.05, never converging to nominal in
                    # that range). 'maxT'/'step_maxT' have the same n_perm>0 prerequisite
                    # and are proven well-calibrated -- use those instead.
                    lgr.critical_raise(
                        "'meff' correction is not supported for correlate_within_region() -- "
                        "it is anti-conservative (too liberal) whenever n_subjects is far "
                        "fewer than n_parcels, which is the typical regime for this method. "
                        "Use 'maxT' or 'step_maxT' instead (same n_perm>0 prerequisite, "
                        "proven well-calibrated across n_subjects=10-100 -- see "
                        "bench5-1_region_correlation_fpr.ipynb).",
                        ValueError
                    )

                else:
                    # alpha only affects the reject mask (not returned here, see
                    # get_within_region_correlations_omnibus discussion), not the
                    # corrected p-values themselves for fdr_bh/bonferroni/holm.
                    # statsmodels.multipletests NaNs out EVERY output the moment a
                    # single input p is NaN (a missing parcel), not just that
                    # parcel's own -- mask out before calling it, reinsert after
                    p_vals = p_df.values[0]
                    nan_mask = np.isnan(p_vals)
                    p_corr_vals = np.full(p_vals.shape, np.nan)
                    if not nan_mask.all():
                        p_corr_valid, _ = mc_correction(p_vals[~nan_mask], alpha=0.05,
                                                        method=mc_method, dtype=self._dtype)
                        p_corr_vals[~nan_mask] = p_corr_valid
                    p_corr_df = pd.DataFrame([p_corr_vals], index=p_df.index, columns=p_df.columns,
                                             dtype=self._dtype)

        out = {"stat_type": "rho", "mc_method": mc_method, "stat": rho_df, "p": p_df,
               "p_corr": p_corr_df}

        lgr.info(f"Returning correlate_within_region results: \n"
                 f"{print_arg_pairs(method=cwr_method, X_reduction=X_reduction, Y_transform=Y_transform, mc_method=mc_method)}")
        lgr.setLevel(loglevel)

        return out

    # ----------------------------------------------------------------------------------------------

    def get_within_region_correlations_omnibus(self, omnibus_stat="absrho", method=None,
                                             X_reduction=None, Y_transform=None, verbose=None):
        """
        Single global test from :meth:`correlate_within_region`: "are region
        values more correlated between X and Y, on average across all
        parcels, than expected by chance?" -- one p-value for the whole
        analysis, as opposed to :meth:`get_within_region_correlations`'s one
        p-value per parcel.

        Reuses the same subject-permutation null already generated by
        :meth:`correlate_within_region` (requires ``n_perm > 0`` there):
        ``omnibus_stat`` aggregates the per-parcel ``rho`` into one number,
        and the same aggregation is applied to each permutation's per-parcel
        null to build a null distribution of that one number, against which
        the observed aggregate is compared (same floor-clipped empirical
        p-value convention as :meth:`get_within_region_correlations`'s raw
        p).

        Parameters
        ----------
        omnibus_stat : {"rho", "absrho", "rho2"}, default "absrho"
            How to aggregate the per-parcel ``rho`` values into one number.

            - ``"rho"`` -- signed mean, computed via Fisher-z averaging (each
              parcel's rho is z-transformed, averaged, then converted back to
              a correlation -- never a raw arithmetic mean of bounded rho
              values, which is downward-biased for larger |rho|; standard
              meta-analytic combination). Most powerful if you expect a
              consistent-direction relationship across parcels (mirrors
              :meth:`paired_colocalization`'s ``pooled_p="mean"``, which pools
              on the same Fisher-z scale by default), but a
              sign-heterogeneous true effect (positive in some regions,
              negative in others) can cancel out and hide it.
            - ``"absrho"`` (default) -- mean absolute value. Robust to
              sign-heterogeneity; matches the ``|rho|`` convention already
              used by ``"maxT"``/``"step_maxT"`` in
              :meth:`get_within_region_correlations`.
            - ``"rho2"`` -- mean squared value ("average variance
              explained"). More powerful than ``"absrho"`` when the true
              effect is concentrated in a few strongly-correlated parcels
              rather than spread thinly across most of them, at the cost of
              being more sensitive to a single outlier parcel.
        method : {"pearson", "spearman"}, optional
            Defaults to the last one used.
        X_reduction, Y_transform : str, optional
            Which stored X/Y to have used. Defaults to the last one used.
        verbose : bool, optional
            Print progress messages. Defaults to the value set at init.

        Returns
        -------
        dict
            ``{"stat_type": omnibus_stat, "stat": float, "p": float}``.

        Raises
        ------
        KeyError
            If the requested combination was never computed, or was computed
            with ``n_perm=0`` (no null to test the omnibus statistic
            against).
        """
        loglevel = lgr.getEffectiveLevel()
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)

        if omnibus_stat not in _CWR_OMNIBUS_STATS:
            lgr.critical_raise(
                f"'omnibus_stat' must be one of {_CWR_OMNIBUS_STATS}, got '{omnibus_stat}'.",
                ValueError
            )

        cwr_method, X_reduction, Y_transform = self._get_last(
            cwr_method=method, X_reduction=X_reduction, Y_transform=Y_transform
        )
        _key = _get_df_string("corrwithin", xdimred=X_reduction, ytrans=Y_transform,
                              method=cwr_method)

        try:
            rho_df = self._corr_within[_key]
        except KeyError:
            available = "\n".join(list(self._corr_within.keys()))
            lgr.critical_raise(
                f"No correlate_within_region result for method='{cwr_method}', "
                f"X_reduction='{X_reduction}', Y_transform='{Y_transform}'. Did you run "
                f"NiSpace.correlate_within_region()? Available:\n{available}",
                KeyError
            )

        null_entry = self._nulls.get("corr_within", {}).get(_key)
        if null_entry is None:
            lgr.critical_raise(
                f"No null distribution stored for '{_key}' -- the omnibus test requires "
                "n_perm > 0 in correlate_within_region().",
                KeyError
            )

        # rho_df/null_dist are already Fisher-z whenever correlate_within_region()'s own
        # r_to_z was True (its default) -- _cwr_omnibus_aggregate normalizes back to raw
        # rho internally either way, so this is correct regardless of that setting
        _already_z = null_entry.get("r_to_z", True)
        stat_obs = _cwr_omnibus_aggregate(rho_df.values[0], omnibus_stat, already_z=_already_z)
        null_agg = _cwr_omnibus_aggregate(null_entry["null_dist"], omnibus_stat, already_z=_already_z)
        p = _cwr_null_p(np.array([stat_obs]), null_agg[:, np.newaxis])[0]

        out = {"stat_type": omnibus_stat, "stat": float(stat_obs), "p": float(p)}

        lgr.info(f"Returning correlate_within_region omnibus test: \n"
                 f"{print_arg_pairs(method=cwr_method, X_reduction=X_reduction, Y_transform=Y_transform, omnibus_stat=omnibus_stat)}")
        lgr.setLevel(loglevel)

        return out

    # PERMUTE ======================================================================================

    def permute(self, what, method=None, X_reduction=None, Y_transform=None, xsea=None,
                n_perm=10000,
                maps_which="X", maps_nulls=None, maps_method=None, dist_mat=None,
                sets_X_background=None,
                p_tails=None, pooled_p="auto", p_from_average_y_coloc=None,
                n_proc=None, seed=None, store=True, verbose=None, force_dict=False,
                **kwargs):
        """
        Estimate exact non-parametric p-values via permutation testing.

        Parameters
        ----------
        what : str or list of str
            What to permute. One or more of:
            ``"maps"`` — spatially constrained null maps for X and/or Y brain maps;
            ``"groups"`` — Y group labels (requires ``Y_transform``); :cite:`dukart2021`
            ``"sets"`` — X set membership labels (requires XSEA);
            ``"pairs"`` — within-pair colocalization against a between-pair null
            (SPICE test; requires N matched maps in both X and Y). :cite:`weinstein2021` Pairs can
            be subjects, studies, tracer targets, or any unit for which one map
            exists in each modality.
            Allowed combinations for multi-element lists: ``["maps", "groups"]``,
            ``["maps", "sets"]``, ``["groups", "sets"]``. Three-way simultaneous
            permutation is not supported and falls back to ``["groups", "sets"]``.
            ``"pairs"`` cannot be combined with other modes.
        method : str, optional
            Colocalization method. Defaults to the method used in the last
            :meth:`colocalize` call.
        X_reduction : str, optional
            X dimensionality-reduction label. Defaults to last used.
        Y_transform : str, optional
            Y transformation label. Defaults to last used.
        xsea : bool, optional
            Whether to run in XSEA mode. Defaults to last used.
        n_perm : int, optional
            Number of permutations. Default is 10000.
        maps_which : str or list of str, optional
            Which data to generate null maps for: ``"X"``, ``"Y"``, or
            ``["X", "Y"]``. Default is ``"X"``.
        maps_nulls : dict, optional
            Pre-computed null maps as ``{map_name: array(n_perm, n_parcels)}``.
            Bypasses null map generation entirely when provided and valid.
        maps_method : str, optional
            Null map generation method. Auto-selected from the parcellation when
            not set (default: ``"moran"`` for all parcellation types). Options:
            ``"moran"`` / ``"msr"``, ``"variomoran"`` / ``"variomsr"``,
            ``"cornblath"`` / ``"spin"`` (surface only), ``"alexander_bloch"``,
            ``"burt2018"``, ``"burt2020"``, ``"random"``.
        dist_mat : array-like of shape (n_parcels, n_parcels), optional
            Pre-computed geodesic distance matrix. Generated from the
            parcellation if not provided (and required by the null method).
        sets_X_background : array-like of shape (n_maps, n_parcels), optional
            Background X map pool for set permutation. If not provided, the
            unique observed X maps are used as the background.
        p_tails : str or dict, optional
            P-value tail(s). ``"two"``, ``"upper"``, or ``"lower"``. Can be a
            dict keyed by statistic name (e.g. ``{"rho": "two"}``). Defaults to
            method-appropriate tails.
        pooled_p : str or bool, optional
            How to aggregate across Y maps before computing p-values:
            ``"mean"`` or ``"median"`` (average first, one p-value per X map),
            ``False`` (one p-value per Y×X pair),
            ``"auto"`` (default) — ``False`` for single-Y, ``"mean"`` otherwise.
            For ``what="groups"``, ``pooled_p`` is not a free choice — it always
            answers a group-level question and is forced to ``"mean"``
            regardless of what is passed (with a warning if the requested value
            conflicts), including when ``"auto"`` would otherwise resolve to
            ``False``.
            For ``what="pairs"``, within-pair coupling is always aggregated
            across pairs; ``"mean"`` and ``"median"`` are both valid and control
            the aggregation function; ``False`` falls back to ``"mean"``.
            For a correlation-based ``method`` (``"pearson"``/``"spearman"``/
            ``"partialpearson"``/``"partialspearman"``), this pooling always
            averages/medians the ``"rho"`` values on the Fisher-z scale --
            never raw, bounded correlation coefficients -- regardless of
            whether :meth:`colocalize`'s own ``r_to_z`` was ``True`` (the
            default; already Fisher-z, pooled as-is) or ``False`` (raw rho is
            transformed on the fly for this pooling step only; the stored,
            unpooled ``"rho"`` itself is unaffected and stays raw). This is
            not a free choice -- averaging raw correlation coefficients is
            never statistically valid, so there is no way to opt out of it.
        p_from_average_y_coloc : str or bool, optional
            Deprecated. Use ``pooled_p`` instead.
        n_proc : int, optional
            Number of parallel processes. Defaults to the value set at init.
        seed : int, optional
            Random seed for reproducibility. Defaults to the seed set at init
            (``NiSpace(seed=...)``) if not given here.
        store : bool, optional
            Store p-values and z-scores in the object. Default is True.
        verbose : bool, optional
            Print progress messages. Defaults to the value set at init.
        force_dict : bool, optional
            Always return a dict even when the result has a single statistic.

        Other Parameters
        ----------------
        maps_centroids : bool
            ``maps_*`` kwargs → null map generation (:func:`generate_null_maps`).
            Use parcel centroids for geodesic distance matrix. Default False.
        maps_parc_resample : int
            Voxel size (mm) to resample parcellation before distance-matrix
            computation. Default 2.
        maps_lr_mirror_dist_mat : bool
            Mirror left-hemisphere distance matrix to the right. Default False.
        maps_split_hemi : bool or None
            Generate null maps separately per hemisphere. Default None.
        maps_split_cxsc : bool
            Generate null maps separately for cortex and subcortex. Default False.
        maps_cx_sc_minmax_scale : bool
            Min–max scale cortex and subcortex null maps before merging.
            Default False.
        maps_procedure : str
            Moran randomisation procedure: ``"singleton"`` (default) or ``"all"``.
        maps_joint : bool
            Moran joint randomisation. Default True.
        distmat_centroids : bool
            ``distmat_*`` kwargs → distance-matrix generation (``_get_dist_mat``).
            Use centroids for CV distance matrix. Default False.
        distmat_parc_resample : int
            Resampling voxel size for CV distance matrix. Default 2.
        groups_paired : bool or "auto"
            ``groups_*`` kwargs → group-label permutation (:func:`permute_groups`).
            Paired permutation (requires subjects vector). ``"auto"`` infers
            pairing from the Y transform. Default ``"auto"``.
        groups_strategy : str
            Permutation strategy: ``"shuffle"`` (default), ``"proportional"``, or ``"draw"``.
            Remaining kwargs (no prefix) are forwarded to :meth:`colocalize`.

        Returns
        -------
        p_values : DataFrame or dict of DataFrames
            P-values indexed by Y labels × X labels.  A dict is returned when
            the colocalization method produces multiple statistics or when
            ``force_dict=True``.

        """
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        lgr.info("*** NiSpace.permute() - Estimate exact non-parametric p values. ***")
        seed = seed if seed is not None else self._seed

        ## check if fit was run
        self._check_fit()

        ## check for allowed permutation combinations
        # check what variable (validated first: the "maps" membership check right
        # below assumes 'what' is already a list, and would raise a confusing
        # TypeError instead of this ValueError for e.g. an int/float 'what')
        if isinstance(what, str):
            what = [what]
        elif isinstance(what, (list, tuple)):
            what = list(what)
        else:
            lgr.critical_raise(f"'what' must be list, tuple, or string, not {type(what)}",
                               ValueError)

        ## map permutation: raise early only when parc is genuinely needed
        if "maps" in what and self._parc is None:
            _has_nulls    = maps_nulls is not None or (self._nulls.get("maps_null") is not None)
            _has_dist     = dist_mat is not None
            _is_spin      = maps_method in _SPIN_METHODS if maps_method else False
            _is_dist_free = maps_method in _DISTMAT_FREE_METHODS if maps_method else False
            if not (_has_nulls or (_has_dist and not _is_spin) or _is_dist_free):
                lgr.critical_raise(
                    "Map null map generation requires a parcellation. Provide one via "
                    "NiSpace(parcellation=...), or supply pre-computed null maps (maps_nulls=) "
                    "or a distance matrix (dist_mat=) with a non-spin null method.",
                    ValueError,
                )
        what = sorted(what)
        if self._binary_y and "groups" in what:
            lgr.critical_raise(
                "binary_y=True: Group label permutation (permute='groups') cannot be used "
                "with binary Y data. Binary maps are a fixed observed result (e.g., a "
                "meta-analytic cluster); scrambling group labels is not a valid null model. "
                "Use permute='maps' with maps_which='X', or supply NiMARE coordinate-sampling "
                "null maps via maps_nulls= and maps_which='Y'.",
                ValueError,
            )
        # check maps_which variable
        if maps_which:
            if isinstance(maps_which, str):
                maps_which = [m for m in maps_which]
            maps_which = sorted(maps_which)
            if maps_which not in [["X"], ["Y"], ["X", "Y"]]:
                lgr.critical_raise(f"'maps_which' has to be 'X', 'Y', or ['X', 'Y'] not '{maps_which}'",
                                   ValueError)
        # validate/resolve the requested combination of permutation targets against
        # the central registry (raises for anything not defined there)
        try:
            what, perm_info, maps_which, combo_warnings = _resolve_permute_combo(what, maps_which)
        except ValueError as e:
            lgr.critical_raise(str(e), ValueError)
        for w in combo_warnings:
            lgr.warning(w)
        lgr.info(f"Permutation of: {perm_info}.")
            
        ## settings
        _rank_kwarg = kwargs.pop("rank", None)
        method, X_reduction, Y_transform, xsea, rank, zy_matched, regress_z = self._get_last(
            method=method,
            X_reduction=X_reduction,
            Y_transform=Y_transform,
            xsea=xsea,
            rank=None,
            zy_matched=None,
            regress_z=None,
        )
        if _rank_kwarg is not None:
            rank = _rank_kwarg
        if "spearman" in method:
            rank = True

        # specific settings via kwargs
        # distance matrix generation 
        dist_mat_kwargs = {
            "dist_mat_type": "null_maps", 
            "centroids": False,
            "parc_resample": 2
        } 
        for k in [k for k in kwargs.keys() if k.startswith("distmat_")]:
            dist_mat_kwargs[k.removeprefix("distmat_")] = kwargs.pop(k)
        # null maps generation
        maps_separate_sc = kwargs.pop("maps_separate_sc", False)
        maps_kwargs = {
            "nispace_nulls": self._nulls,
            "use_existing": True,
            "null_maps": maps_nulls,
            "null_method": maps_method,
            "spin_mat": None,
            "lr_mirror_dist_mat": False,
            "parc_resample": 2,
            "parc_name": self._parc._name if self._parc else None,
        }
        for k in [k for k in kwargs.keys() if k.startswith("maps_")]:
            maps_kwargs[k.removeprefix("maps_")] = kwargs.pop(k)
        # resolve null_method and null_space from parcellation
        if self._parc is not None:
            null_space_result = self._parc.get_null_space()
            _is_split_strategy = isinstance(null_space_result[0], tuple)
            if _is_split_strategy:
                (cx_null_space, _), (sc_null_space, _) = null_space_result
                # null_space = MNI sc space for image loading; cx surface handled in _get_null_maps
                null_space = sc_null_space
            else:
                null_space = null_space_result[0]
            if maps_kwargs["null_method"] is None:
                if _is_split_strategy:
                    _, cx_m = null_space_result[0]
                    _, sc_m = null_space_result[1]
                    maps_kwargs["null_method"] = (cx_m, sc_m)
                    lgr.info(f"Using default split null method (cx='{cx_m}', sc='{sc_m}') "
                             f"(cx space: '{cx_null_space}', sc space: '{sc_null_space}').")
                else:
                    maps_kwargs["null_method"] = null_space_result[1]
                    lgr.info(f"Using default null method '{null_space_result[1]}' "
                             f"(parcellation null space: '{null_space}').")
            # ensure null_space is loaded and fitted so backward-compat properties work in _get_null_maps
            self._parc._ensure_image_loaded(null_space)
            if null_space not in self._parc._hemi_dict:
                self._parc._fit_space(null_space)
            if self._parc._space is None:
                self._parc._space = null_space
            # pass precomputed spin matrix for any spin method (generate_null_maps validates shape)
            _cx_m = maps_kwargs["null_method"][0] if isinstance(maps_kwargs["null_method"], tuple) else maps_kwargs["null_method"]
            if _cx_m in _SPIN_METHODS:
                cached = self._parc_spin_mat or self._parc.get_spin_mat(null_space)
                if cached is not None:
                    maps_kwargs["spin_mat"] = cached
        else:
            if maps_kwargs["null_method"] is None:
                maps_kwargs["null_method"] = "moran"
                lgr.info("No parcellation set; defaulting null method to 'moran'.")
        # groups permutation
        groups_kwargs = {
            "paired": "auto",
            "strategy": "shuffle",
        }
        for k in [k for k in kwargs.keys() if k.startswith("groups_")]:
            groups_kwargs[k.removeprefix("groups_")] = kwargs.pop(k)
        # colocalization
        coloc_kwargs = {
            "xsea": xsea,
            "n_proc": n_proc,
            "seed": seed,
        } | kwargs
        
        ## merge with settings from current NiSpace object        
        n_proc = n_proc if n_proc is not None else self._n_proc
        dtype = self._dtype
        
        ## check if colocalize was run
        xsea = True if ("sets" in what) or (xsea == True) else False
        if not self._check_colocalize(method, None, X_reduction, Y_transform, xsea,
                                      raise_error=False):
            lgr.warning(f"'{method}' colocalization was not run before. Running now.")
            self.colocalize(method, X_reduction, Y_transform, **coloc_kwargs)
            # colocalize() resolves rank/regress_z/zy_matched through its own conditional
            # logic (partial methods, zy_matched, clean_y interactions, etc.).  Refresh
            # local variables so the null-map pre-ranking/regression block below uses
            # exactly the same settings as the observed colocalization just computed.
            rank       = self._last_settings["rank"]
            regress_z  = self._last_settings["regress_z"] or ""
            zy_matched = self._last_settings["zy_matched"]
            if "spearman" in method:
                rank = True

        ## get observed data
        # X
        if not X_reduction:
            _X_obs = self._X
        else:
            lgr.info(f"Loading dimensionality-reduced X data, reduction method = '{X_reduction}'.")
            with _quiet():
                _X_obs = self.get_x(X_reduction=X_reduction)
        _X_obs_arr = np.array(_X_obs, dtype=dtype)
        if xsea:
            if self._xsea:
                _X_obs_arr = {set_name: np.array(set_X, dtype=dtype) 
                              for set_name, set_X in _X_obs.groupby(level="set", sort=False)}
            else:
                lgr.warning("Input 'what' contains 'sets' or 'xsea' was set to True but it seems "
                            "XSEA was not run before. Will not perform XSEA (permutation).")
                what.remove("sets")
                xsea = False
        # Y
        _Y_obs = self._Y
        _Y_obs_arr = np.array(_Y_obs, dtype=dtype)
        if Y_transform:
            lgr.info(f"Loading transformed Y data, transform = '{Y_transform}'.")
            with _quiet():
                _Y_trans_obs = self.get_y(Y_transform=Y_transform)
            _Y_trans_obs_arr = np.array(_Y_trans_obs, dtype=dtype)
        # Z
        _Z_obs = self._Z
        _Z_obs_arr = np.array(_Z_obs, dtype=dtype)

        # TODO (first non-dev release): remove p_from_average_y_coloc parameter
        if p_from_average_y_coloc is not None:
            lgr.warning(_DEPR_P_FROM_AVERAGE_Y_COLOC)
            pooled_p = p_from_average_y_coloc

        ## get observed colocalizations as numpy arrays
        lgr.info(f"Loading observed colocalizations (method = '{method}').")
        with _quiet():
            _colocs_obs = self.get_colocalizations(
                method,
                X_reduction=X_reduction,
                Y_transform=Y_transform,
                xsea=xsea,
                force_dict=True,
            )
        _colocs_obs = {stat: np.array(df, dtype=dtype) for stat, df in _colocs_obs.items()}

        # whether the stored "rho" (pearson/spearman/partial*) is already Fisher-z (the
        # colocalize() call's own r_to_z, default True) -- needed below wherever "rho" gets
        # pooled (mean/median) across Y maps or pairs: averaging *raw* bounded correlation
        # coefficients is never valid, so pooling always forces the Fisher-z scale for
        # "rho" regardless of r_to_z, applying rho_to_z() on the fly only when the stored
        # value isn't already on that scale (r_to_z=False) -- never a double-transform.
        _rho_already_z = self._coloc_kwargs_by_method.get(method, {}).get("r_to_z", True)

        ## pairs permutation (SPICE) — early return, bypasses null-map generation
        if what == ["pairs"]:
            # validation
            if self._X.shape[0] != self._Y.shape[0]:
                lgr.critical_raise(
                    f"what='pairs' requires X and Y to have the same number of maps "
                    f"(got X: {self._X.shape[0]}, Y: {self._Y.shape[0]}). "
                    "Fit NiSpace with matched per-pair X and Y data.",
                    ValueError,
                )
            if self._X.shape[0] < 3:
                lgr.critical_raise(
                    f"what='pairs' requires at least 3 pairs (got {self._X.shape[0]}).",
                    ValueError,
                )
            if Y_transform:
                lgr.critical_raise(
                    "what='pairs' cannot be combined with Y_transform. "
                    "Fit NiSpace with raw per-pair Y maps, or use what='groups' instead.",
                    ValueError,
                )
            if self._X.index.tolist() != self._Y.index.tolist():
                lgr.warning(
                    "X and Y index labels do not match. Pair matching is done positionally "
                    "(same row order assumed for X and Y)."
                )
            # pooled_p resolution
            if pooled_p == "auto" or pooled_p is True:
                pooled_p = "mean"
            elif pooled_p is False:
                lgr.warning(
                    "pooled_p=False is not supported for what='pairs' (within-pair "
                    "coupling is always aggregated across pairs). Falling back to 'mean'."
                )
                pooled_p = "mean"
            elif pooled_p not in ("mean", "median"):
                pooled_p = "mean"
            self._nulls["pooled_p"] = pooled_p
            # get the N×N coloc matrix (first stat key)
            _stat = next(iter(_colocs_obs))
            _mat = _colocs_obs[_stat]   # (N, N) float32 array
            # never average raw (bounded, non-additive) correlation coefficients: pool on
            # the Fisher-z scale regardless of r_to_z, converting only if the stored "rho"
            # isn't already on that scale -- see _rho_already_z above
            _pool_as_z = _stat == "rho" and not _rho_already_z
            if _pool_as_z:
                _mat = rho_to_z(_mat)
            _N = _mat.shape[0]
            _agg = np.median if pooled_p == "median" else np.mean
            _observed = float(_agg(np.diag(_mat)))
            lgr.info(
                f"Pairs permutation: N={_N}, observed within-pair {_stat} "
                f"({pooled_p}) = {_observed:.4f}"
                f"{' (Fisher-z scale)' if _pool_as_z else ''}."
            )
            # build null key for cache lookup
            _perm = "pairs"
            _null_key = _get_df_string(
                "null",
                xdimred=X_reduction,
                ytrans=Y_transform,
                method=method,
                xsea=xsea,
                perm=_perm,
                pooled_p=pooled_p,
            )
            # cache check
            _pairs_null_entry = self._nulls.get("pairs_null", {}).get(_null_key)
            if (_pairs_null_entry is not None
                    and _pairs_null_entry.get("n_perm", 0) >= n_perm
                    and _pairs_null_entry.get("null_method") == "pairs"):
                lgr.info("Using cached pairs permutation null distribution.")
                _null_dist = _pairs_null_entry["null_dist"]
            else:
                # vectorized null: shuffle row indices for each permutation
                rng = np.random.default_rng(seed)
                _sigmas = np.argsort(rng.random((_N, n_perm)), axis=0).T   # (n_perm, N)
                _null_dist = _agg(_mat[_sigmas, np.arange(_N)], axis=1).astype(dtype)  # (n_perm,)
            # p-value
            _p_val = float(null_to_p(_observed, _null_dist, tail="upper"))
            lgr.info(f"Pairs permutation p-value ({pooled_p}): {_p_val:.4f}.")
            # build output DataFrame
            _p_df = pd.DataFrame(
                [[_p_val]], index=["within_pair"], columns=["all"], dtype=dtype
            )
            _p_data = {_stat: _p_df}
            # store
            if store:
                self._nulls.setdefault("pairs_null", {})[_null_key] = {
                    "null_dist":   _null_dist,
                    "observed":    _observed,
                    "n_perm":      n_perm,
                    "null_method": "pairs",
                    "stat":        _stat,
                }
                _p_key = _get_df_string(
                    "p",
                    xdimred=X_reduction,
                    ytrans=Y_transform,
                    method=method,
                    stat=_stat,
                    xsea=xsea,
                    perm=_perm,
                    pooled_p=pooled_p,
                )
                self._p_colocs[_p_key] = _p_df
                self._set_last(
                    method=method,
                    X_reduction=X_reduction,
                    Y_transform=Y_transform,
                    xsea=xsea,
                    rank=rank,
                    zy_matched=zy_matched,
                    regress_z=regress_z,
                    perm=_perm,
                    pooled_p=pooled_p,
                )
                if self._return_self:
                    return self
                if force_dict or len(_p_data) > 1:
                    return _p_data
                return _p_df
            else:
                if force_dict or len(_p_data) > 1:
                    return _p_data, _null_dist
                return _p_df, _null_dist

        ## resolve pooled_p: "auto" -> decide based on the number of rows that are actually
        # being colocalized (i.e. the Y_transform output, not the raw per-subject Y -- for a
        # group comparison this is usually a single group-difference map even though the raw
        # Y has one row per subject), "median"/"mean" -> calculate p based on mean/median
        # colocalization across Y rows, False -> calculate p for every Y row individually,
        # anything else -> defaults to mean
        _n_y_rows = next(iter(_colocs_obs.values())).shape[0]

        # what="groups" always answers a group-level question, so pooled_p isn't a
        # free choice for it -- it's forced by the mode. Returns pooled_p unchanged
        # (still possibly "auto") for any other what (maps/sets/pairs), where
        # pooled_p remains a free, meaningful choice resolved by the generic
        # auto-resolution block below.
        pooled_p, mode_warning = _resolve_permute_mode_settings(what, pooled_p)
        if mode_warning:
            lgr.warning(mode_warning)

        if pooled_p:
            if pooled_p == "auto":
                if _n_y_rows == 1 or self._x_with_self:
                    pooled_p = False
                elif _n_y_rows > 1:
                    pooled_p = "mean"
            elif pooled_p not in ["mean", "median"]:
                pooled_p = "mean"
            if pooled_p:
                lgr.info("Will calculate p values for mean colocalization across Y maps. Set "
                         "'pooled_p=False' to compute p values for each Y map individually.")
            self._nulls["pooled_p"] = pooled_p

        # get average prediction values of all y if requested (no-op when there is only one
        # row to begin with, e.g. a single group-difference map from Y_transform)
        if pooled_p and _n_y_rows > 1:
            for stat in _colocs_obs.keys():
                _vals = _colocs_obs[stat]
                # never average raw (bounded, non-additive) correlation coefficients --
                # pool "rho" on the Fisher-z scale regardless of r_to_z (see
                # _rho_already_z above); this pooled copy only ever feeds the p-value
                # comparison below (_get_exact_p_values), never displayed/stored itself,
                # so no back-transform is needed here
                if stat == "rho" and not _rho_already_z:
                    _vals = rho_to_z(_vals)
                if pooled_p == "median":
                    _colocs_obs[stat] = np.nanmedian(_vals, axis=0)[np.newaxis, :]
                else:
                    _colocs_obs[stat] = np.nanmean(_vals, axis=0)[np.newaxis, :]
        
        ## prepare permuted data as prerequisite for null colocalization runs
        _X_null, _Y_null, _Z_null = None, None, None
        
        # case permute X/Y brain maps
        if "maps" in what:
            # iterate map datasets to permute
            for XY in maps_which:
                lgr.info(f"Generating permuted {XY} maps.")
                
                # if no null maps & no distance matrix given, generate distance matrix
                # Skip for: pure spin; split+spin where sc dist_mat comes from parc.get_sc_dist_mat()
                _cx_m_check, _sc_m_check = _parse_null_method(maps_kwargs["null_method"])
                _skip_distmat = (
                    # pure spin or split+spin where sc dist_mat comes from get_sc_dist_mat()
                    (_cx_m_check in _SPIN_METHODS and (
                        _sc_m_check is None or (
                            self._parc is not None
                            and self._parc._sc_dist_mat_spec is not None
                        )
                    ))
                    # split non-spin where both component dist_mats can be lazy-loaded
                    or (_sc_m_check is not None
                        and self._parc is not None
                        and self._parc._cx_dist_mat_spec is not None
                        and self._parc._sc_dist_mat_spec is not None)
                    # dist-mat-free method (e.g. random): no dist_mat ever needed
                    or (_cx_m_check in _DISTMAT_FREE_METHODS
                        and _sc_m_check in _DISTMAT_FREE_METHODS | {None})
                )
                if maps_nulls is None and dist_mat is None and not _skip_distmat:
                    dist_mat = self._get_dist_mat(**dist_mat_kwargs)
                
                # get null maps, will not generate new maps if already existing and use of
                # existing is requested
                if XY=="X":
                    # xsea: gene sets are frequently redundant (e.g. GO hierarchy) -- dedupe
                    # by value before generation so a gene repeated across sets gets ONE null
                    # surrogate per permutation (reused everywhere it appears), not one
                    # independent draw per (set, gene) occurrence. `_inverse_idx_X` maps each
                    # original _X_obs row back to its row in the deduplicated frame; used below
                    # to re-expand the (smaller) null array back into per-set arrays.
                    _inverse_idx_X = None
                    if isinstance(_X_obs_arr, dict):
                        data_obs = _X_obs.drop_duplicates()
                        _, _inverse_idx_X = dedupe_rows(_X_obs.values)
                    else:
                        data_obs = _X_obs
                    standardize_nulls = True if "x" in self._zscore else False
                elif XY=="Y":
                    if Y_transform:
                        data_obs = _Y_trans_obs
                    else:
                        data_obs = _Y_obs
                    standardize_nulls = True if "y" in self._zscore else False
                maps_nulls, new_spin_mat = _get_null_maps(
                    data_obs=data_obs,
                    dist_mat=dist_mat,
                    parc=self._parc,
                    #parc_kwargs=self._parc_info,
                    #standardize=False,
                    standardize=standardize_nulls,
                    n_perm=n_perm,
                    seed=seed,
                    n_proc=n_proc,
                    dtype=dtype,
                    verbose=verbose,
                    permute_which=XY,
                    **maps_kwargs
                )

                # tag null maps with X/Y identity and store
                maps_nulls.null_which = XY
                self._nulls["maps_null"] = maps_nulls
                # promote newly generated spin matrix to instance attribute for reuse
                if new_spin_mat is not None and self._parc_spin_mat is None:
                    self._parc_spin_mat = new_spin_mat

                # sort null map data into lists of length n_perm, each element being one
                # permuted array of observed values
                lgr.debug("Sorting null map data into arrays.")
                if XY=="X":
                    _X_null = maps_nulls.perm_list(dtype=self._dtype)
                    # case: xsea requested: re-sort into a list of dicts of set-wise arrays
                    if isinstance(_X_obs_arr, dict):
                        idc_set = np.array(_X_obs.index.get_level_values("set"))
                        # _inverse_idx_X maps original (set, gene) row positions to rows in
                        # the deduplicated null array generated above -- genes shared across
                        # sets are gathered from the same underlying null draw
                        _X_null = [{set_name: null[_inverse_idx_X[idc_set == set_name], :]
                                    for set_name in _X_obs_arr.keys()}
                                   for null in _X_null]
                elif XY=="Y":
                    _Y_null = maps_nulls.perm_list(dtype=self._dtype)
            
        # case permute Y groups
        if ("groups" in what) and Y_transform:
            lgr.info(f"Generating permuted Y groups.")

            # clean transform name early so cache comparison is consistent
            Y_transform = _lower_strip_ws(Y_transform)

            # check for cached group permutation null maps
            _groups_null_cached = self._nulls.get("groups_null")
            if (_groups_null_cached is not None and maps_kwargs.get("use_existing", True)
                    and _groups_null_cached.null_method == Y_transform
                    and _groups_null_cached.n_perm >= n_perm):
                lgr.info("Using cached group permutation null maps.")
                _needed_labels = list(_Y_trans_obs.index)
                _Y_null = _groups_null_cached.subset(_needed_labels).perm_list(dtype=self._dtype)
            else:
                # get groups without nan values
                groups = self._groups_no_nan
                if hasattr(self, "_subjects_no_nan"):
                    subjects = self._subjects_no_nan
                else:
                    subjects = None

                # Y values without nan values in group vector
                _Y_obs_arr_nonan = _Y_obs_arr[~self._groups_nan_idc, :]

                ## prepare formula & transform function
                # TODO: get stored transform function
                apply_transform, paired = _get_transform_fun(Y_transform, return_df=False,
                                                             return_paired=True, dtype=dtype,
                                                             ignore_nan_warnings=True)

                # paired permutations?
                if groups_kwargs["paired"] not in ["auto", True, False]:
                    lgr.warning("Argument 'groups_paired' must be of boolean type or 'auto' not "
                                f"'{groups_kwargs['paired']}'! Setting to 'auto'.")
                    groups_kwargs["paired"] = "auto"
                if groups_kwargs["paired"] == "auto":
                    groups_kwargs["paired"] = paired

                # get list of permuted group labels
                lgr.info(f"Permuting groups/sessions vector, strategy: "
                         f"{'paired' if groups_kwargs['paired'] else 'unpaired'}, {groups_kwargs['strategy']}.")
                groups_null = permute_groups(
                    groups=groups,
                    subjects=subjects,
                    n_perm=n_perm,
                    n_proc=n_proc,
                    seed=seed,
                    verbose=verbose,
                    **groups_kwargs
                )

                # get permuted group comparison results
                # parallelization function
                def par_fun(group_null):
                    # apply transform with random groups
                    Y_null = apply_transform(y=_Y_obs_arr_nonan, groups=group_null, subjects=subjects)
                    return Y_null
                # run in parallel
                _Y_null = Parallel(n_jobs=n_proc)(
                    delayed(par_fun)(g) for g in tqdm(
                        groups_null,
                        desc=f"Null transformations ({method}, {n_proc} proc)", disable=not verbose
                    )
                )

                # wrap in NullMaps and cache for reuse across coloc method changes
                # _Y_null is list of n_perm arrays, each (n_contrasts, n_parcels)
                # stack → (n_perm, n_contrasts, n_parcels); transpose → NullMaps invariant
                self._nulls["groups_null"] = NullMaps(
                    np.stack(_Y_null).transpose(1, 0, 2),
                    labels=list(_Y_trans_obs.index),
                    null_method=Y_transform,
                    null_type="group",
                )
        
        # case permute Y groups but no comparison is provided
        elif ("groups" in what) & (not Y_transform):
            lgr.critical_raise("Provide a comparison ('Y_transform') to perform group permutation!",
                               ValueError)

        # case X Set Enrichment Analysis: permute X sets
        if "sets" in what:
            lgr.info("Generating permuted X sets.")
            if sets_X_background is None:
                sets_X_background = _X_obs.drop_duplicates(ignore_index=True).values
                lgr.warning(f"No X background dataset provided. Will use "
                            f"{sets_X_background.shape[0]} unique X maps as background.")
            else:
                if not isinstance(sets_X_background, (np.ndarray, pd.DataFrame)):
                    lgr.critical_raise(f"X background maps must be of type np.ndarray or "
                                       f"pd.DataFrame, not {type(sets_X_background)}!",
                                       TypeError)
                if sets_X_background.shape[1] != _X_obs.shape[1]:
                    lgr.critical_raise(f"X background maps of wrong shape {sets_X_background.shape}!",
                                       ValueError)
                lgr.info(f"Will use {sets_X_background.shape[0]} provided background maps.")
                sets_X_background = np.array(sets_X_background, dtype=dtype)
                if "x" in self._zscore:
                    lgr.info("Z-standardizing X background maps.")
                    sets_X_background = zscore_df(sets_X_background, along="rows", force_df=False)
                
            # get permuted X sets
            set_sizes = [set_X.shape[0] for set_X in _X_obs_arr.values()]
            set_names = list(_X_obs_arr.keys())
            bg_size = sets_X_background.shape[0]
            # get permuted indices
            rng = np.random.default_rng(seed)
            _X_null = [
                {name: rng.choice(bg_size, size=size, replace=False) 
                 for name, size in zip(set_names, set_sizes)} 
                for _ in tqdm(range(n_perm), desc="Permuting X set indices", disable=not verbose)
            ] 
            # function to get permuted data from indices. necessary to handle large X set arrays
            def _xsea_perm_data(i):
                return {name: sets_X_background[idc, :] for name, idc in _X_null[i].items()}
            
        # catch case in which xsea is performed (i.e., x array is dict) but sets are not permuted
        elif "sets" not in what and isinstance(_X_obs_arr, dict):
            lgr.info("Running X Set Enrichment Analysis (XSEA) without set permutation.")
            # function to get _X_null data, only necessary for compatibility with the above
            def _xsea_perm_data(i):
                return _X_null[i]
            
        # handle weighted XSEA
        X_weights = None
        if isinstance(_X_obs_arr, dict):
            if "weighted" in self._xsea_aggregation_method:
                if isinstance(_X_obs_arr, dict):
                    X_weights = {set_name: np.array(set_X.index.get_level_values("weight"), dtype=self._dtype) 
                                for set_name, set_X in _X_obs.groupby(level="set", sort=False)}
        
        ## pre-rank and regress
        if rank or regress_z:
            if rank:
                lgr.info("Pre-ranking X and Y (null) data.")
            if regress_z:
                lgr.info(f"Regressing {_Z_obs_arr.shape[0]} {'Y-matched ' if zy_matched else ''}Z "
                         f"maps from (null) {regress_z.upper()} data.")
            # X observed
            _X_obs_arr = _rank_regress(
                arr=_X_obs_arr, 
                rank=rank, 
                regress="x" in regress_z, 
                z=_Z_obs_arr, 
                zy_matched=zy_matched, 
                verbose=verbose
            )
            # X null
            if not ("sets" in what and isinstance(_X_obs_arr, dict)):
                _X_null = _rank_regress(
                    arr=_X_null, 
                    rank=rank, 
                    regress="x" in regress_z, 
                    z=_Z_obs_arr, 
                    zy_matched=zy_matched, 
                    verbose=verbose,
                    n_proc=n_proc
                )
            # XSEA background
            sets_X_background = _rank_regress(
                arr=sets_X_background, 
                rank=rank, 
                regress="x" in regress_z, 
                z=_Z_obs_arr, 
                zy_matched=zy_matched, 
                verbose=verbose,
            )
            # Y observed
            _Y_obs_arr = _rank_regress(
                arr=_Y_obs_arr, 
                rank=rank, 
                regress="y" in regress_z, 
                z=_Z_obs_arr, 
                zy_matched=zy_matched, 
                verbose=verbose,
            )
            # Y null
            _Y_null = _rank_regress(
                arr=_Y_null, 
                rank=rank, 
                regress="y" in regress_z, 
                z=_Z_obs_arr, 
                zy_matched=zy_matched, 
                verbose=verbose,
                n_proc=n_proc
            )
        
        ## check what permuted dataframes we have, if we dont have them, copy observed data (!)
        if (not _X_null) & (not _Y_null) & (not _Z_null):
            lgr.critical_raise("No permuted data generated. Supported permutations ('what') are: "
                               "'maps', 'groups', and 'sets'.",
                               ValueError)
        # X
        if not _X_null:
            _X_null = [_X_obs_arr] * n_perm
        # Y
        if not _Y_null:
            _Y_null = [_Y_trans_obs_arr if "_Y_trans_obs" in locals() else _Y_obs_arr] * n_perm
        
        ## run null colocalizations
        # function to perform colocalization for one y vector (= per subject); see NiSpace.colocalize()
        # the function was saved by NiSpace.colocalize()
        _y_colocalize = self._colocs_fun[method]
        
        # function to perform colocalization for one X/Y/Z null array
        xsea = True if isinstance(_X_null[0], dict) else False
        #n_components = self._coloc_kwargs["n_components"]
        def par_fun(X_null, Y_null, X_weights=None):
            # run colocalization
            null_colocs_list = [
                _y_colocalize(X_null, Y_null[i_y, :], X_weights)
                for i_y in range(Y_null.shape[0])
            ]
            # sort output with helper function, return as array
            null_colocs = _sort_colocs(
                method=method, 
                xsea=xsea,
                y_colocs_list=null_colocs_list, 
                n_X=len(X_null),
                n_Y=Y_null.shape[0],
                #n_components=n_components,
                return_df=False,
                dtype=dtype
            )
            # average colocalization if requested (no-op when there is only one row)
            if pooled_p and _n_y_rows > 1:
                for stat in null_colocs:
                    _vals = null_colocs[stat]
                    # same Fisher-z-before-pooling forcing as the observed side above --
                    # must match it exactly, or obs/null end up compared on different
                    # scales
                    if stat == "rho" and not _rho_already_z:
                        _vals = rho_to_z(_vals)
                    if pooled_p == "median":
                        null_colocs[stat] = np.nanmedian(_vals, axis=0)[np.newaxis, :]
                    else:
                        null_colocs[stat] = np.nanmean(_vals, axis=0)[np.newaxis, :]
            # return
            return null_colocs
        
        # xsea null aggregation fast path: for methods that score each gene independently
        # (see _COLOC_METHODS_UNIVARIATE), the per-set statistic is a plain post-hoc
        # reduction over per-gene values -- so those can be precomputed ONCE (per unique
        # gene, not per set-occurrence-per-permutation) and reused via array lookup,
        # instead of recomputing raw correlations inside `par_fun` for every permutation.
        # Restricted to single-mode "sets"/"maps"(Y) xsea calls -- see plan for rationale;
        # everything else (multivariate methods, what-combos, non-xsea) uses `par_fun` as
        # before, unchanged.
        _fast_xsea_sets = xsea and method in _COLOC_METHODS_UNIVARIATE and what == ["sets"]
        _fast_xsea_mapsY = (xsea and method in _COLOC_METHODS_UNIVARIATE
                             and what == ["maps"] and maps_which == ["Y"])

        if _fast_xsea_sets or _fast_xsea_mapsY:
            lgr.info("Using precomputed per-gene statistics for XSEA null aggregation.")
            set_names_fast = list(_X_obs_arr.keys())
            xsea_method = self._xsea_aggregation_method
            weighted = "weighted" in xsea_method

            # plain (non-xsea) per-row colocalization function, built with the exact same
            # kwargs used to build self._colocs_fun[method] -- reuses the existing, tested
            # pearson/mutualinfo/r2 numerics (incl. r_to_z handling) rather than
            # reimplementing them here
            _plain_kwargs = {k: v for k, v in self._coloc_kwargs_by_method[method].items()
                              if k not in ("rank", "regress_z", "zy_matched")}
            _plain_kwargs["xsea"] = False
            _y_colocalize_plain = _get_colocalize_fun(
                method=method, seed=seed, verbose=False, dtype=dtype, **_plain_kwargs
            )

            stat_key = None
            if _fast_xsea_sets:
                # Y is fixed (not permuted) in pure "sets" mode -- one background
                # stat matrix serves every permutation; only the drawn indices vary
                Y_fixed = _Y_null[0]  # (n_Y_rows, n_parcels), same for all i
                bg_unique, bg_inverse_idx = dedupe_rows(sets_X_background)
                stat_bg_unique = np.zeros((Y_fixed.shape[0], bg_unique.shape[0]), dtype=dtype)
                for i_y in range(Y_fixed.shape[0]):
                    res = _y_colocalize_plain(bg_unique, Y_fixed[i_y, :])
                    if stat_key is None:
                        stat_key = next(iter(res))
                    stat_bg_unique[i_y] = res[stat_key]
                stat_lookup = stat_bg_unique[:, bg_inverse_idx]  # (n_Y, bg_size)

                set_stats = []
                for set_name in set_names_fast:
                    # (n_perm, set_size) drawn background indices for this set
                    idx_matrix = np.stack([_X_null[i][set_name] for i in range(n_perm)])
                    gathered = stat_lookup[:, idx_matrix]  # (n_Y, n_perm, set_size)
                    w = X_weights[set_name] if weighted else None
                    set_stats.append(_xsea_aggregate(
                        gathered, xsea_method, weights=w, axis=-1,
                        rho_scale=(stat_key == "rho" and not _rho_already_z),
                    ))
                full = np.stack(set_stats, axis=-1).astype(dtype)  # (n_Y, n_perm, n_sets)

            else:  # _fast_xsea_mapsY: X sets fixed (observed), Y is nulled per permutation
                idc_set = np.array(_X_obs.index.get_level_values("set"))
                X_unique, inverse_idx = dedupe_rows(np.vstack(list(_X_obs_arr.values())))
                set_member_idx = {name: inverse_idx[idc_set == name] for name in set_names_fast}

                n_y_rows_null = _Y_null[0].shape[0]
                stat_uniq = np.zeros((n_y_rows_null, n_perm, X_unique.shape[0]), dtype=dtype)
                for i in tqdm(range(n_perm), desc=f"Null colocalizations ({method}, precomputed)",
                              disable=not verbose):
                    for i_y in range(n_y_rows_null):
                        res = _y_colocalize_plain(X_unique, _Y_null[i][i_y, :])
                        if stat_key is None:
                            stat_key = next(iter(res))
                        stat_uniq[i_y, i] = res[stat_key]

                set_stats = []
                for set_name in set_names_fast:
                    gathered = stat_uniq[:, :, set_member_idx[set_name]]  # (n_Y, n_perm, set_size)
                    w = X_weights[set_name] if weighted else None
                    set_stats.append(_xsea_aggregate(
                        gathered, xsea_method, weights=w, axis=-1,
                        rho_scale=(stat_key == "rho" and not _rho_already_z),
                    ))
                full = np.stack(set_stats, axis=-1).astype(dtype)  # (n_Y, n_perm, n_sets)

            # pooled_p reduction, matching par_fun's post-processing exactly (no-op if
            # pooled_p is falsy or there's only one Y row to begin with)
            if pooled_p and _n_y_rows > 1:
                reducer = np.nanmedian if pooled_p == "median" else np.nanmean
                # same Fisher-z-before-pooling forcing as the non-xsea pooled_p sites
                # above -- "full" here is already the XSEA per-set aggregate (see
                # _xsea_aggregate above, which now separately guarantees the
                # gene-to-set aggregation step never averages raw rho either, via its
                # own rho_scale= param). _xsea_aggregate's rho_scale handling always
                # round-trips back to whatever scale "gathered" started on (raw when
                # _rho_already_z is False, since it z-transforms-then-back; untouched
                # when _rho_already_z is True, since rho_scale is then False and
                # gathered was already z) -- so "full"'s scale still mirrors
                # _rho_already_z exactly, same condition as every other pooled_p site
                if stat_key == "rho" and not _rho_already_z:
                    full = rho_to_z(full)
                full = reducer(full, axis=0, keepdims=True).astype(dtype)

            _colocs_null = [{stat_key: full[:, i, :]} for i in range(n_perm)]

        # run in parallel
        elif not xsea:
            _colocs_null = Parallel(n_jobs=n_proc)(
                delayed(par_fun)(_X_null[i], _Y_null[i])
                for i in tqdm(
                    range(n_perm),
                    desc=f"Null colocalizations ({method}, {n_proc} proc)", disable=not verbose
                )
            )
        else:
            _colocs_null = Parallel(n_jobs=n_proc)(
                delayed(par_fun)(_xsea_perm_data(i), _Y_null[i], X_weights)
                for i in tqdm(
                    range(n_perm),
                    desc=f"Null colocalizations ({method}, {n_proc} proc)", disable=not verbose
                )
            )
            
        ## calculate exact p values
        # get values
        p_data, p_tails_resolved = _get_exact_p_values(
            method=method,
            xsea_aggr=self._xsea_aggregation_method if xsea else None,
            colocs_obs=_colocs_obs,
            colocs_null=_colocs_null,
            p_tails=p_tails,
            verbose=verbose,
            dtype=dtype
        )
        # to dataframe
        for stat in p_data.keys():
            # column names
            if p_data[stat].shape[1] == 1:
                cols = [stat]
            elif p_data[stat].shape[1] == _X_obs.shape[0]:
                cols = _X_obs.index
            elif isinstance(_X_obs_arr, dict) and p_data[stat].shape[1] == len(_X_obs_arr):
                cols = list(_X_obs_arr.keys())
            else:
                lgr.critical_raise(f"p value array of wrong shape ({p_data[stat].shape})!",
                                   ValueError)
            # index names
            if (pooled_p in ["mean", "median"]) and (_n_y_rows > 1):
                rows = [pooled_p]
            elif "_Y_trans_obs" in locals():
                rows = _Y_trans_obs.index
            else:
                rows = _Y_obs.index
            p_data[stat] = pd.DataFrame(p_data[stat], columns=cols, index=rows)
        
        # save and return
        if store:    
            perm = "".join(what).replace("maps", "".join(maps_which)+"maps")
            for stat in p_data:
                df_str = _get_df_string(
                    "p",
                    xdimred=X_reduction,
                    ytrans=Y_transform,
                    method=method,
                    stat=stat,
                    xsea=xsea,
                    perm=perm,
                    pooled_p=pooled_p,
                )
                self._p_colocs[df_str] = p_data[stat]
            df_str = _get_df_string(
                "null",
                xdimred=X_reduction,
                ytrans=Y_transform,
                method=method,
                xsea=xsea,
                perm=perm,
                pooled_p=pooled_p,
            )
            self._nulls["_colocs"][df_str] = _colocs_null
            self._nulls[f"p_tails_{df_str}"] = p_tails_resolved
            self._set_last(
                method=method,
                X_reduction=X_reduction,
                Y_transform=Y_transform,
                xsea=xsea,
                rank=rank,
                zy_matched=zy_matched,
                regress_z=regress_z,
                perm=perm,
                pooled_p=pooled_p,
            )
            ## return
            if self._return_self:
                return self
            # return dict of dfs
            if force_dict or len(p_data) > 1:
                return p_data
            else:
                return p_data[list(p_data.keys())[0]]
        else:
            if force_dict or len(p_data) > 1:
                return p_data, _colocs_null
            else:
                k = list(p_data.keys())[0]
                return p_data[k], _colocs_null[k]
    
    
    # CORRECT ======================================================================================

    def correct_p(self, mc_method="meff",
                  mc_alpha=0.05, mc_dimension="array", coloc_method=None, store=True, verbose=None):
        """
        Apply a multiple-comparisons correction to the uncorrected p-values
        previously computed by :meth:`permute`, storing the corrected result
        under its own key (so several ``mc_method`` corrections of the same
        permutation result can coexist and be retrieved separately via
        :meth:`get_p_values`/:meth:`get_corrected_p_values`).

        Parameters
        ----------
        mc_method : str, default "meff"
            Correction method. One of:

            * ``"meff"`` / ``"meff_galwey"`` (default) -- Šidák correction using
              an effective number of independent tests estimated from the
              eigenvalues of X's (and, for ``mc_dimension="array"`` with
              multiple Y rows, also Y's) correlation matrix. :cite:`galwey2009`
            * ``"meff_li_ji"`` -- same Šidák-correction scheme, with an
              alternative eigenvalue-based effective-N estimator. :cite:`liji2005`
            * ``"maxT"`` -- single-step max-statistic FWER correction from the
              permutation null computed by :meth:`permute`; requires the null
              colocalization distributions to still be available (i.e. not
              dropped via ``save_nulls=False``). :cite:`westfall1993`
            * ``"step_maxT"`` -- step-down variant of ``"maxT"``, more powerful
              while preserving FWER control. :cite:`westfall1993`
            * ``"fdr_bh"`` -- Benjamini-Hochberg false discovery rate. :cite:`benjamini1995`
            * ``"bonferroni"``, or any other method name accepted by
              ``statsmodels.stats.multitest.multipletests`` (e.g. ``"holm"``,
              ``"hommel"``, ``"sidak"``, ``"fdr_by"``) -- passed through as-is.

        mc_alpha : float, default 0.05
            Alpha threshold used by the correction.
        mc_dimension : str, default "array"
            Axis over which to correct: ``"array"`` (jointly across all X x Y
            comparisons), ``"x"``/``"columns"`` (per X column), or
            ``"y"``/``"rows"`` (per Y row). Not all combinations are supported by
            every method -- e.g. ``"maxT"``/``"step_maxT"`` don't support
            per-column correction, and for the ``meff`` methods with multiple Y
            rows, ``"array"`` applies a joint X x Y correction that is only
            meaningful when the Y rows are related entities examined together
            (e.g. several disorders' effect-size maps); use ``mc_dimension="y"``
            for independent per-Y-row correction (e.g. individual-subject maps).
        coloc_method : str, optional
            Restrict correction to p-values from one colocalization method
            (useful when several methods' results are stored at once). Defaults
            to correcting all stored uncorrected p-values.
        store : bool, default True
            Store the corrected p-values on the object, and remember
            ``mc_method`` as the "last used" correction (read by
            :meth:`get_corrected_p_values`, :meth:`plot`).
        verbose : bool, optional
            Print progress messages. Defaults to the value set at init.

        Returns
        -------
        dict of pandas.DataFrame
            Corrected p-values keyed by their internal storage key string.

        """
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        lgr.info("*** NiSpace.correct_p() - Correct p values for multiple comparisons. ***")

        # list of all p-value df keys, only uncorrected p-values
        p_strs = [k for k in self._p_colocs if "mc-none" in k]
        if coloc_method is not None:
            p_strs = [s for s in p_strs if f"coloc-{coloc_method}" in s]

        # resolve mc method name (aliases → canonical)
        mc_method = _get_correct_mc_method(mc_method)
        lgr.info(f"Correction method: '{mc_method}', alpha: {mc_alpha}, dimension: '{mc_dimension}'.")

        # mc_dimension → how
        if mc_dimension in ["x", "X", "c", "col", "cols", "column", "columns"]:
            how = "c"
        elif mc_dimension in ["y", "Y", "r", "row", "rows"]:
            how = "r"
        else:
            how = "a"

        # key suffix (strips _ and - so it's safe as a key fragment)
        mc_key = mc_method.replace("_", "").replace("-", "").lower()

        p_corr = dict()

        if mc_method in _EMPIRICAL_MC_METHODS:
            for p_str in p_strs:
                fields   = _parse_df_string(p_str)
                xdimred  = _parse_bool(fields.get("xdimred", False))
                ytrans   = _parse_bool(fields.get("ytrans", False))
                coloc    = fields.get("coloc")
                stat     = fields.get("stat")
                xsea     = _parse_bool(fields.get("xsea", False))
                perm     = fields.get("perm")
                pooled   = _parse_bool(fields.get("pooled", False))

                p_str_mc = p_str.replace("mc-none", f"mc-{mc_key}")
                p_values = self._p_colocs[p_str]

                # ── Meff + Sidak ──────────────────────────────────────────────
                if mc_method in {"meff_galwey", "meff_li_ji"}:
                    if how == "c":
                        lgr.warning("mc_dimension='x'/'c' is not meaningful for 'meff' "
                                    "(Meff is defined over X maps). Ignoring.")
                    meff_variant = "galwey" if mc_method == "meff_galwey" else "li_ji"
                    with _quiet():
                        X_data = np.array(self.get_x(X_reduction=xdimred))
                    # for XSEA: use per-set mean map so Meff reflects set-level independence
                    if xsea and hasattr(self._X.index, "get_level_values"):
                        set_labels = self._X.index.get_level_values("set")
                        X_data = np.array(
                            pd.DataFrame(X_data, index=self._X.index)
                            .groupby(set_labels).mean()
                        )
                    meff_x = compute_meff(X_data, method=meff_variant)
                    lgr.info(f"Meff_X ({meff_variant}) = {meff_x:.2f} "
                             f"(from {X_data.shape[0]} maps).")

                    n_y_rows = p_values.shape[0]
                    if how == "a" and n_y_rows > 1:
                        # joint correction across X and Y: meff_total = meff_X * meff_Y
                        with _quiet():
                            Y_data = np.array(self.get_y(Y_transform=ytrans))
                        meff_y = compute_meff(Y_data, method=meff_variant)
                        meff = meff_x * meff_y
                        lgr.info(f"Meff_Y ({meff_variant}) = {meff_y:.2f} "
                                 f"(from {n_y_rows} maps). Meff_total = {meff:.2f}.")
                        lgr.warning(
                            f"Joint Meff correction (Meff_X × Meff_Y = {meff:.2f}) assumes all "
                            "Y maps are related entities examined together (e.g., disorder effect "
                            "size maps). It is NOT valid for individual subject maps. "
                            "Use mc_dimension='y' for independent per-Y correction."
                        )
                    else:
                        meff = meff_x

                    p_corr[p_str_mc], _ = meff_sidak_correction(
                        p_values, meff=meff, alpha=mc_alpha, dtype=self._dtype
                    )

                # ── Max-T / Step-down Max-T ───────────────────────────────────
                elif mc_method in {"maxT", "step_maxT"}:
                    # warn if scale comparability assumption may be violated
                    if stat in {"beta", "individual"} and "x" not in (self._zscore or ""):
                        lgr.warning(
                            f"maxT on stat='{stat}' assumes regression coefficients are on a "
                            "comparable scale across X maps, but X was not z-scored "
                            f"(standardize='{self._zscore}'). Max-T results may be unreliable. "
                            "Re-run with standardize='x' (or 'xz') to satisfy this assumption."
                        )
                    # maxT corrects across X (columns) per Y row — override "array" default
                    how_maxt = how if how != "a" else "r"
                    if how == "a":
                        lgr.info("mc_dimension not explicitly set to 'y'; defaulting to per-Y "
                                 "(max across X) for maxT.")
                    elif how == "c":
                        lgr.critical_raise(
                            "mc_dimension='x' (per-X column) is not supported for maxT. "
                            "Use mc_dimension='y' (per-Y row, default) or 'array' (global).",
                            ValueError
                        )
                    # get null key and null colocs
                    null_str = _get_df_string(
                        "null", xdimred=xdimred, ytrans=ytrans,
                        method=coloc, xsea=xsea, perm=perm, pooled_p=pooled,
                    )
                    if null_str not in self._nulls["_colocs"]:
                        lgr.critical_raise(
                            f"Null colocalizations for '{null_str}' not found. "
                            f"'{mc_method}' requires the full null distributions. "
                            "Either re-run permute(), or reload the object with save_nulls=True "
                            "(note: correct_p('maxT'/'step_maxT') should be called before saving "
                            "without nulls).",
                            KeyError
                        )
                    null_colocs = self._nulls["_colocs"][null_str]
                    # get observed statistics (not p-values)
                    with _quiet():
                        obs_stats = self.get_colocalizations(
                            method=coloc, stats=[stat],
                            X_reduction=xdimred, Y_transform=ytrans,
                            xsea=xsea, force_dict=True,
                        )[stat]
                    # get tail (use stored resolved p_tails if available, else default)
                    p_tails_stored = self._nulls.get(f"p_tails_{null_str}", {})
                    tail = p_tails_stored.get(stat, "two")
                    correction_fn = maxT_correction if mc_method == "maxT" else step_maxT_correction
                    p_corr[p_str_mc], _ = correction_fn(
                        obs_stats, null_colocs, stat=stat,
                        tail=tail, how=how_maxt,
                        alpha=mc_alpha, dtype=self._dtype
                    )

        else:
            # statsmodels path
            for p_str in p_strs:
                p_str_mc = p_str.replace("mc-none", f"mc-{mc_key}")
                p_corr[p_str_mc], _ = mc_correction(
                    self._p_colocs[p_str],
                    alpha=mc_alpha,
                    method=mc_method,
                    how=how,
                    dtype=self._dtype
                )

        # save and return
        if store:
            for p_str in p_corr:
                self._p_colocs[p_str] = p_corr[p_str]
            self._set_last(mc_method=mc_method)
            if self._return_self:
                return self
        return p_corr

    # ----------------------------------------------------------------------------------------------

    def normalize_colocalizations(self, coloc_method=None, z_method="robust", store=True,
                                  verbose=None):
        """
        Z-score observed colocalization statistics against their null
        permutation distribution (per X column), producing values that are more
        comparable across colocalization methods/maps with different natural
        scales. This is distinct from any z-scoring of the raw input data
        (``standardize=`` at init) -- it normalizes colocalization *output*
        against the null computed by :meth:`permute`, which must have been run
        first. Also distinct from :meth:`colocalize`'s own ``r_to_z`` (Fisher-z
        transform of ``"rho"``, applied once, upstream, at colocalize()-time):
        if that was active (the default), this function's null-relative
        z-score is computed on the *already* Fisher-z ``"rho"`` values, not
        raw correlations -- a "z-score of a z" for correlation-based methods,
        not a Fisher-z itself.

        Parameters
        ----------
        coloc_method : str, optional
            Restrict normalization to one colocalization method's stored null
            results. Defaults to normalizing all of them.
        z_method : {"robust", "standard"}, default "robust"
            ``"robust"`` uses a median/MAD-based z-score (columns with zero MAD
            become NaN); anything else uses a standard mean/SD-based z-score.
            Remembered as the "last used" ``z_method`` for later calls (e.g.
            :meth:`plot`).
        store : bool, default True
            Store the normalized values on the object (accessible via
            :meth:`get_normalized_colocalizations`).
        verbose : bool, optional
            Print progress messages. Defaults to the value set at init.

        Returns
        -------
        NiSpace
            Returns ``self`` (for chaining), regardless of ``store``.
        """
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        lgr.info("*** NiSpace.normalize_colocalizations() - Normalize colocalizations against null "
                 "distribution. ***")
        lgr.info(f"Z-score method: {'robust (median/MAD)' if z_method == 'robust' else 'standard (mean/SD)'}.")

        if not self._nulls["_colocs"]:
            lgr.critical_raise(
                "No null colocalizations found. "
                "Run permute() first, or reload the object with save_nulls=True "
                "(note: normalize_colocalizations() should be called before saving without nulls).",
                ValueError
            )

        score_fn = rzscore_nan if z_method == "robust" else zscore_nan
        self._set_last(z_method=z_method)

        null_keys = list(self._nulls["_colocs"].keys())
        if coloc_method is not None:
            null_keys = [k for k in null_keys if f"coloc-{coloc_method.lower()}" in k]

        for null_str in null_keys:
            fields  = _parse_df_string(null_str)
            xdimred = _parse_bool(fields.get("xdimred", False))
            ytrans  = _parse_bool(fields.get("ytrans", False))
            coloc   = fields.get("coloc")
            xsea    = _parse_bool(fields.get("xsea", False))
            perm    = fields.get("perm")
            pooled  = _parse_bool(fields.get("pooled", False))

            null_colocs = self._nulls["_colocs"][null_str]
            stats = _get_coloc_stats(coloc, permuted_only=True)

            for stat in stats:
                with _quiet():
                    obs_dict = self.get_colocalizations(
                        method=coloc, stats=[stat],
                        X_reduction=xdimred, Y_transform=ytrans,
                        xsea=xsea, force_dict=True,
                    )
                if stat not in obs_dict:
                    continue
                obs_df = obs_dict[stat]

                null_arr = _null_stats_to_array(null_colocs, stat).astype(float)
                z_arr = score_fn(np.array(obs_df, dtype=float), null_arr)
                z_df = pd.DataFrame(z_arr, index=obs_df.index, columns=obs_df.columns,
                                    dtype=self._dtype)

                if store:
                    z_str = _get_df_string(
                        "z",
                        xdimred=xdimred, ytrans=ytrans,
                        method=coloc, stat=stat,
                        xsea=xsea, perm=perm, pooled_p=pooled,
                    )
                    self._z_colocs[z_str] = z_df
                    lgr.info(f"Stored normalized colocalizations: {z_str}")

        if self._return_self:
            return self
        return self


    # PLOT ====================================================================================

    def plot(self, kind="categorical",
             method=None, stats=None,
             X_reduction=None, Y_transform=None,
             xsea=None,
             Y_labels=None, X_labels=None,
             Y_maps=None, X_maps=None,
             values="coloc", mc_method=None,
             plot_nulls=True, annot_p=True, permute_what=None,
             title="auto",
             sort_by=None,        # None | 'coloc'|'abs_coloc'|'z'|'abs_z'|'p' — also enables truncation when n_categories is exceeded
             sort_colocs=False,
             n_categories=50,     # max categories shown; exceeded + sort_by set → truncate top N; exceeded + no sort_by → skip with warning; None = no limit

             colocalizations_dict=None, nulls_dict=None, p_dict=None, pc_dict=None,
             fig=None, ax=None, figsize=None, show=True,
             plot_kwargs=None, nullplot_kwargs=None,
             verbose=None):
        """
        Plot a stored colocalization result as a categorical (per-X-map) plot,
        one figure per requested statistic, optionally overlaid with the null
        permutation distribution and significance annotation.

        Only ``kind="categorical"`` is currently implemented -- ``"correlation"``,
        ``"brain"``, and ``"nullhist"`` are planned but not yet built (passing
        them raises ``NotImplementedError``). For brain-map visualization, use
        the separate :meth:`plot_brain` method instead.

        Parameters
        ----------
        kind : str, default "categorical"
            Only ``"categorical"`` is currently supported.
        method, X_reduction, Y_transform, xsea : optional
            Identify which stored :meth:`colocalize` result to plot; see
            :meth:`colocalize`. Default to the last-used values. Ignored if
            ``colocalizations_dict`` is given directly.
        stats : str or list of str, optional
            Which statistic(s) to plot, one figure each. Defaults to all stats
            found for ``method``.
        Y_maps, X_maps : str or list of str, optional
            Restrict the plot to matching Y/X map labels (exact or substring
            match). ``Y_labels``/``X_labels`` are accepted as legacy aliases
            (used only if the corresponding ``_maps`` argument is not given).
        values : {"coloc", "z", "p"}, default "coloc"
            What to plot: the raw observed statistic (via
            :meth:`get_colocalizations`), the null-normalized z-score (via
            :meth:`get_normalized_colocalizations`, requires
            :meth:`normalize_colocalizations` to have been run), or
            ``-log10(p)`` (via :meth:`get_p_values`; disables ``plot_nulls``
            unconditionally, since null distributions aren't meaningful in
            p-value space).
        mc_method : str, optional
            Which p-value correction to use for annotation/``values="p"``.
            Special values ``"uncorrected"``/``"none"``/``"false"`` force
            uncorrected p-values. Defaults to the last correction used in
            :meth:`correct_p`.
        plot_nulls : bool, default True
            Overlay the null permutation distribution. Requires
            ``permute_what`` to be resolvable (i.e. :meth:`permute` to have
            been run); otherwise disabled with a warning. Always disabled when
            ``values="p"``.
        annot_p : bool or str, default True
            Annotate significance. Also accepts a mode string forwarded to
            ``plotting.print_significance`` (e.g. ``"text"``). Disabled (with a
            warning) under the same condition as ``plot_nulls``.
        permute_what : str, optional
            Which permutation ("what") to pull null distributions/p-values
            from; see :meth:`permute`'s ``what`` argument. Defaults to the
            last-used value. The special value ``"pairs"`` collapses an N x N
            colocalization matrix to its diagonal (matched-pair SPICE-style
            results) before plotting.
        title : str, default "auto"
            Plot title. ``"auto"`` builds one from the method/context (recomputed
            for each statistic when multiple ``stats`` are plotted in one call).
            A custom string is used verbatim for every statistic's plot.
        sort_by : {None, "coloc", "abs_coloc", "z", "abs_z", "p"}, optional
            Sort X categories by the mean (or abs mean) observed value,
            z-score, or p-value across Y rows. Also enables truncation (see
            ``n_categories``) when set.
        sort_colocs : bool, default False
            Deprecated; use ``sort_by="coloc"`` instead.
        n_categories : int, optional, default 50
            Maximum number of X categories to display. If exceeded and
            ``sort_by`` is set, truncates to the top N; if exceeded and
            ``sort_by`` is ``None``, that statistic's plot is skipped entirely
            (with a warning) rather than truncating arbitrarily. ``None``
            disables the limit.
        colocalizations_dict : dict, optional
            Pre-computed ``{stat: DataFrame}`` result (as from
            ``get_colocalizations(force_dict=True)``) to plot directly, bypassing
            all internal fetching.
        nulls_dict, p_dict, pc_dict : optional
            Pre-computed null-distribution / uncorrected-p / corrected-p data,
            bypassing the corresponding internal fetch.
        fig, ax : optional
            Existing matplotlib Figure/Axes to draw into (for building custom
            multi-panel figures). Pass both together.
        figsize : tuple, optional
            Figure size; auto-sized by number of X categories if not given.
        show : bool, default True
            Call ``plt.show()`` after each statistic's plot.
        plot_kwargs, nullplot_kwargs : dict, optional
            Extra keyword arguments forwarded to :func:`nispace.plotting.catplot`
            and :func:`nispace.plotting.nullplot` respectively.
        verbose : bool, optional
            Print progress messages. Defaults to the value set at init.

        Returns
        -------
        tuple or dict of tuple
            ``(fig, ax, plot)`` per statistic; a single tuple if only one
            statistic was plotted, otherwise a dict keyed by stat name.
        """
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        lgr.info("*** NiSpace.plot() - Plot colocalization results. ***")

        if kind in {"correlation", "brain", "nullhist"}:
            lgr.critical_raise(
                f"plot(kind='{kind}') is planned but not yet implemented. "
                "Only kind='categorical' is currently supported. For brain-map "
                "visualization, use plot_brain() instead.",
                NotImplementedError
            )

        # kwargs
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        nullplot_kwargs = {} if nullplot_kwargs is None else nullplot_kwargs

        # check fit
        self._check_fit()

        # settings
        method, X_reduction, Y_transform, xsea, permute_what = self._get_last(
            method=method,
            X_reduction=X_reduction,
            Y_transform=Y_transform,
            xsea=xsea,
            perm=permute_what
        )

        # check if minimum input provided
        if colocalizations_dict is None and method is None:
            lgr.critical_raise("Provide either a method name or a colocalization result!",
                               ValueError)

        # TODO (first non-dev release): remove sort_colocs parameter entirely
        if sort_colocs:
            lgr.warning(_DEPR_SORT_COLOCS)
            if sort_by is None:
                sort_by = "coloc"
            sort_colocs = False

        # p mode never shows null distributions; z mode supports them
        if values == "p":
            plot_nulls = False

        # check nulls/p plot
        if (plot_nulls or annot_p) and not permute_what:
            lgr.warning("if 'plot_nulls' or 'annot_p', provide 'permute_what' ({'groups', "
                        "'{x|y|xy}maps', 'sets'}). Setting 'plot_nulls' and 'annot_p' to False!")
            plot_nulls = False
            annot_p = False

        # resolve mc_method for p mode
        if values == "p":
            if isinstance(mc_method, str) and mc_method.lower() in ("uncorrected", "none", "false"):
                mc_method = None
            elif mc_method is None:
                mc_method = self._last_settings.get("mc_method")
            if mc_method is not None:
                mc_method = _get_correct_mc_method(mc_method)
            if mc_method:
                lgr.info(f"Plotting −log₁₀(p), corrected: {mc_method}.")
            else:
                lgr.info("Plotting −log₁₀(p), uncorrected.")
        elif values == "z":
            lgr.info("Plotting null-normalised z-scores.")

        # get arguments
        check_kwargs = dict(method=method, stats=stats, xdimred=X_reduction,
                            ytrans=Y_transform, xsea=xsea)
        get_kwargs = dict(method=method, stats=stats, X_reduction=X_reduction,
                          Y_transform=Y_transform, xsea=xsea)

        # get colocalization results
        _z_method = self._last_settings.get("z_method", "robust")
        if colocalizations_dict is None:
            self._check_colocalize(**check_kwargs)

            if values == "z":
                _z_method = self._last_settings.get("z_method", "robust")
                lgr.info(f"Z-score normalisation method: "
                         f"{'robust (median/MAD)' if _z_method == 'robust' else 'standard (mean/SD)'}.")
                with _quiet():
                    colocalizations_dict = self.get_normalized_colocalizations(
                        **get_kwargs, force_dict=True,
                    )
                if plot_nulls:
                    with _quiet():
                        _raw = self.get_colocalizations(
                            **get_kwargs, force_dict=True,
                            get_nulls=True, nulls_permute_what=permute_what,
                        )
                    _, nulls_dict = _raw if isinstance(_raw, tuple) else (_raw, None)
                    if nulls_dict is None:
                        lgr.warning("No nulls found. Not plotting null distributions in z mode.")
                        plot_nulls = False
                else:
                    nulls_dict = None

            elif values == "p":
                _mc = mc_method.replace("_", "").replace("-", "") if mc_method else None
                with _quiet():
                    _p_raw = self.get_p_values(
                        **get_kwargs, permute_what=permute_what, mc_method=_mc,
                        force_dict=True,
                    )
                colocalizations_dict = {
                    stat: df.apply(lambda col: -np.log10(col))
                    for stat, df in _p_raw.items()
                }
                nulls_dict = None

            else:  # values == "coloc"
                with _quiet():
                    coloc_dicts = self.get_colocalizations(
                        **get_kwargs,
                        force_dict=True,
                        get_nulls=plot_nulls,
                        nulls_permute_what=permute_what,
                    )
                if isinstance(coloc_dicts, tuple):
                    colocalizations_dict, nulls_dict = coloc_dicts
                else:
                    colocalizations_dict, nulls_dict = coloc_dicts, None
                if plot_nulls and nulls_dict is None:
                    lgr.warning("No nulls found. Not plotting null distributions.")
                    plot_nulls = False

        else:
            if not isinstance(colocalizations_dict, dict):
                lgr.critical_raise("Provide colocalizations as dict as returned by "
                                   "NiSpace.get_colocalizations(force_dict=True)!",
                                   TypeError)
            if nulls_dict:
                if not isinstance(nulls_dict, dict):
                    lgr.error("Provide null colocalizations as dict as returned by NiSpace."
                              "get_colocalizations(force_dict=True, get_nulls=True)!")
                    nulls_dict = None
        
        # pairs mode: replace N×N coloc dict with (N, 1) diagonal view for plotting
        if permute_what == "pairs" and colocalizations_dict is not None:
            for stat, coloc_mat in colocalizations_dict.items():
                _diag_vals = np.diag(np.array(coloc_mat))
                colocalizations_dict[stat] = pd.DataFrame(
                    _diag_vals,
                    index=coloc_mat.index,
                    columns=["within_pair"],
                )
            # nulls_dict already has {"within_pair": (1, n_perm)} from get_colocalizations
            # if user passed custom nulls_dict, leave it as-is

        # Y_labels / X_labels are legacy aliases for Y_maps / X_maps
        if Y_labels is not None and Y_maps is None:
            Y_maps = Y_labels
        if X_labels is not None and X_maps is None:
            X_maps = X_labels

        # restrict to given y maps
        keep_y = keep_x = None
        if Y_maps is not None:
            _ref_df = next(iter(colocalizations_dict.values()))
            keep_y = _match_maps(_ref_df.index, Y_maps)
            if not keep_y:
                lgr.critical_raise(f"No Y maps matching {Y_maps!r} found.", ValueError)
            for stat in colocalizations_dict:
                colocalizations_dict[stat] = colocalizations_dict[stat].iloc[keep_y]
                if nulls_dict is not None:
                    if isinstance(nulls_dict[stat], dict):
                        for null_str in nulls_dict[stat]:
                            nulls_dict[stat][null_str] = nulls_dict[stat][null_str].iloc[keep_y]
                    else:
                        nulls_dict[stat] = nulls_dict[stat].iloc[keep_y]

        # restrict to given x maps
        if X_maps is not None:
            _ref_df = next(iter(colocalizations_dict.values()))
            keep_x = _match_maps(_ref_df.columns, X_maps)
            if not keep_x:
                lgr.critical_raise(f"No X maps matching {X_maps!r} found.", ValueError)
            for stat in colocalizations_dict:
                colocalizations_dict[stat] = colocalizations_dict[stat].iloc[:, keep_x]
                if nulls_dict is not None:
                    if isinstance(nulls_dict[stat], dict):
                        nulls_dict[stat] = {
                            k: v for k, v in nulls_dict[stat].items()
                            if k in colocalizations_dict[stat].columns
                        }

        def _filter_dict(d):
            if d is None:
                return None
            out = {}
            for stat, df in d.items():
                if keep_y is not None:
                    df = df.iloc[keep_y]
                if keep_x is not None:
                    df = df.iloc[:, keep_x]
                out[stat] = df
            return out
        # p_dict / pc_dict are always {stat: DataFrame} — plain iloc is fine

        # auto-fetch p-values for annotation (and sort_by="p")
        # resolve effective mc_method: explicit arg > last stored setting
        _annot_mc = mc_method or self._last_settings.get("mc_method")
        if _annot_mc:
            _annot_mc = _get_correct_mc_method(_annot_mc).replace("_", "").replace("-", "")
        if annot_p is not False and values not in ("p",):
            if p_dict is None:
                try:
                    with _quiet():
                        _fetched = self.get_p_values(
                            **get_kwargs, permute_what=permute_what,
                            mc_method=None, force_dict=True,
                        )
                    if _fetched:
                        p_dict = _fetched
                except Exception:
                    pass
            if pc_dict is None and _annot_mc is not None:
                try:
                    with _quiet():
                        _fetched = self.get_p_values(
                            **get_kwargs, permute_what=permute_what,
                            mc_method=_annot_mc, force_dict=True,
                        )
                    if _fetched:
                        pc_dict = _fetched
                except Exception:
                    pass
        p_dict  = _filter_dict(p_dict)
        pc_dict = _filter_dict(pc_dict)

        # loop over stats
        stats = [s for s in colocalizations_dict if s not in ["intercept"]]
        out = {}
        for stat in stats:

            lgr.info(f"Creating {kind} plot for method {method}, colocalization stat {stat}.")
            if title == "auto":
                _title = f"{nice_stats_labels(method)} colocalization"
                if Y_transform:
                    _title += f" after {nice_stats_labels(Y_transform.replace('(a,b)', ''))} transform"
                if permute_what:
                    _perm_str = nice_stats_labels(permute_what)
                    if values == "z":
                        _title += f"\n(permutation of {_perm_str} | normalized)"
                    elif values == "p":
                        if mc_method:
                            _mc_sub = mc_method.replace("_", "-")
                            _p_suffix = rf"$p_{{\mathrm{{{_mc_sub}}}}}$"
                        else:
                            _p_suffix = r"$p_{\mathrm{uncorrected}}$"
                        _title += f"\n(permutation of {_perm_str} | {_p_suffix})"
                    else:
                        _title += f"\n(permutation of {_perm_str})"
            else:
                _title = title

            # compute column sort order (positional indices) for sort_by
            _valid_sort_by = {"coloc", "z", "p", "abs_coloc", "abs_z"}
            _sort_order = None
            if sort_by is not None and sort_by not in _valid_sort_by:
                lgr.warning(f"sort_by='{sort_by}' is not a valid option "
                            f"({', '.join(sorted(_valid_sort_by))}). Ignoring.")
                sort_by = None
            if sort_by is not None and colocalizations_dict[stat].shape[1] > 1:
                try:
                    if sort_by in ("coloc", "abs_coloc"):
                        _src = colocalizations_dict[stat]
                        sv = _src.mean(axis=0).abs() if sort_by == "abs_coloc" else _src.mean(axis=0)
                        _ascending = False
                    elif sort_by in ("z", "abs_z"):
                        if values == "z":
                            _src = colocalizations_dict[stat]
                        else:
                            with _quiet():
                                _z = self.get_normalized_colocalizations(
                                    **get_kwargs, force_dict=True,)
                            _src = _z.get(stat, colocalizations_dict[stat])
                        sv = _src.mean(axis=0).abs() if sort_by == "abs_z" else _src.mean(axis=0)
                        _ascending = False
                    elif sort_by == "p":
                        if values == "p":
                            sv = colocalizations_dict[stat].mean(axis=0)
                            _ascending = False
                        else:
                            _pd = ((pc_dict or {}).get(stat) or (p_dict or {}).get(stat))
                            if _pd is None:
                                # auto-fetch using same logic as values="p"
                                _mc = mc_method.replace("_", "").replace("-", "") if mc_method else None
                                with _quiet():
                                    _p_fetched = self.get_p_values(
                                        **get_kwargs, permute_what=permute_what,
                                        mc_method=_mc, force_dict=True,
                                    )
                                _pd = _p_fetched.get(stat)
                            if _pd is None:
                                raise ValueError("No p-values available for sort_by='p'.")
                            sv = _pd.mean(axis=0)
                            _ascending = True  # lower p = more significant
                    else:
                        sv = None
                    if sv is not None:
                        sorted_labels = sv.sort_values(ascending=_ascending).index.tolist()
                        orig_labels = list(colocalizations_dict[stat].columns)
                        _sort_order = [orig_labels.index(l) for l in sorted_labels]
                except Exception as e:
                    lgr.warning(f"Could not compute sort order for sort_by='{sort_by}': {e}")

            _n_x = colocalizations_dict[stat].shape[1]
            if n_categories is not None and _n_x > n_categories:
                if sort_by is not None and _sort_order is not None:
                    # sort_by is set → truncation is meaningful, show top N
                    _sort_order = _sort_order[:n_categories]
                    lgr.info(
                        f"Showing top {n_categories} of {_n_x} categories "
                        f"(sorted by '{sort_by}')."
                    )
                else:
                    # no sort_by → arbitrary truncation would be misleading, skip instead
                    lgr.warning(
                        f"Plot skipped: {_n_x} categories exceed n_categories={n_categories}. "
                        f"To show the top {n_categories}, add sort_by='abs_z'. "
                        f"To show all, pass n_categories=None."
                    )
                    continue

            if kind == "categorical":
                fig_ax = _plot_categorical(
                    colocs_df=colocalizations_dict[stat],
                    stat=stat,
                    nulls_dict=nulls_dict,
                    p_df=p_dict[stat] if p_dict is not None else None,
                    pc_df=pc_dict[stat] if pc_dict is not None else None,
                    values=values,
                    mc_method=mc_method or _annot_mc,
                    sort=sort_colocs,
                    sort_order=_sort_order,
                    annot_p=annot_p,
                    z_method=_z_method,
                    fig=fig,
                    ax=ax,
                    title=_title,
                    figsize=figsize,
                    kwargs=plot_kwargs,
                    null_kwargs=nullplot_kwargs
                )

            if show:
                plt.show()
            out[stat] = fig_ax
            
        if len(out) ==1:
            out = out[stat]        
        return out


    # PLOT BRAIN ===================================================================================

    def plot_brain(self,
                   data="Y",
                   maps=None,
                   Y_transform=None,
                   X_reduction=None,
                   kind=None,
                   space=None,
                   surf_mesh="inflated",
                   views=None,
                   cmap=None,
                   vmin=None, vmax=None,
                   shared_colorscale=False,
                   symmetric_cmap="auto",
                   colorbar=True,
                   colorbar_label="",
                   ncols=1,
                   title="auto",
                   n_max=5,
                   figsize=None,
                   show=True,
                   verbose=None,
                   **kwargs):
        """Plot brain maps directly onto surfaces or anatomical volumes.

        Parameters
        ----------
        data : {"Y", "X"} or pd.DataFrame
            Which data to plot. "Y" (default) uses the fitted Y maps, "X" the
            reference maps. A DataFrame can be passed directly.
        maps : str or list, optional
            Subset of maps to plot. Matches against the DataFrame index
            (including MultiIndex levels and tuple entries).
        Y_transform : str, optional
            Y transform to apply when data="Y". Defaults to the last used transform.
        X_reduction : str, optional
            X reduction to apply when data="X". Defaults to the last used reduction.
        kind : str, optional
            Rendering mode: "glass", "slice", or "surface". Defaults to "glass".
        space : str, optional
            Parcellation space.
        surf_mesh : str
            Surface mesh ("inflated", "pial", etc.).
        views : list, optional
            Surface views to render.
        cmap : str
            Colormap.
        vmin, vmax : float, optional
            Colorscale limits.
        shared_colorscale : bool
            Share colorscale across all maps.
        symmetric_cmap : bool
            Force symmetric colorscale around zero.
        colorbar : bool
            Show colorbar.
        colorbar_label : str
            Label for the colorbar title.
        ncols : int
            Number of columns in the subplot grid.
        n_max : int
            Maximum number of maps to plot. Raises an error if exceeded.
        figsize : tuple, optional
            Figure size in inches.
        show : bool
            Call plt.show() after plotting.
        verbose : bool, optional
            Verbose logging. Defaults to the instance setting.
        **kwargs
            Additional keyword arguments forwarded to brainplot() and from
            there to the underlying nilearn plotting functions.

        Returns
        -------
        fig : matplotlib.Figure
        axes : list of matplotlib.Axes
        """
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        lgr.info("*** NiSpace.plot_brain() ***")
        self._check_fit()

        Y_transform, X_reduction = self._get_last(
            Y_transform=Y_transform, X_reduction=X_reduction
        )

        # -- resolve data --
        if isinstance(data, str):
            if data.upper() == "Y":
                lgr.info(f"Plotting Y data (Y_transform='{Y_transform}').")
                with _quiet():
                    data_df = self.get_y(Y_transform=Y_transform)
            elif data.upper() == "X":
                lgr.info(f"Plotting X data (X_reduction='{X_reduction}').")
                with _quiet():
                    data_df = self.get_x(X_reduction=X_reduction)
            else:
                lgr.critical_raise(
                    f"data='{data}' not recognised. Use 'Y', 'X', or a DataFrame.",
                    ValueError,
                )
        elif isinstance(data, pd.DataFrame):
            lgr.info("Plotting custom DataFrame.")
            data_df = data
        else:
            lgr.critical_raise(
                "data must be 'Y', 'X', or a pandas DataFrame.", ValueError
            )

        # -- map selection --
        if maps is not None:
            keep = _match_maps(data_df.index, maps)
            if not keep:
                lgr.critical_raise(
                    f"No maps matching {maps!r} found in the index.", ValueError
                )
            data_df = data_df.iloc[keep]

        # -- safeguard against too many subplots --
        if len(data_df) > n_max:
            lgr.critical_raise(
                f"plot_brain: {len(data_df)} maps selected but n_max={n_max}. "
                "Subset with the maps= argument or increase n_max.",
                ValueError,
            )

        # -- call brainplot --
        fig, axes = brainplot(
            data=data_df,
            parcellation=self._parc,
            kind=kind,
            space=space,
            surf_mesh=surf_mesh,
            views=views,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            shared_colorscale=shared_colorscale,
            symmetric_cmap=symmetric_cmap,
            colorbar=colorbar,
            colorbar_label=colorbar_label,
            ncols=ncols,
            title=title,
            figsize=figsize,
            verbose=verbose,
            **kwargs,
        )

        if show:
            plt.show()
        return fig, axes


    # GET ==========================================================================================
    
    def get_x(self, X_reduction=None, maps=None, squeeze=False, verbose=None, copy=True):
        """
        Retrieve the object's X data: either the raw, fitted X, or a
        previously computed dimensionality reduction (see :meth:`reduce_x`).

        Parameters
        ----------
        X_reduction : str, optional
            Label of the reduction to retrieve (as passed to
            ``reduce_x(reduction=...)``). Defaults to the last one used, or the
            raw X data if none has been used. Raises ``KeyError`` (listing
            available labels) if the requested reduction was never computed.
        maps : str or list of str, optional
            Restrict to matching X map/set labels (exact or substring match).
        squeeze : bool, default False
            If exactly one map remains after any ``maps`` filtering, return it
            as a ``pandas.Series`` instead of a single-row DataFrame.
        verbose : bool, optional
            Print progress messages. Defaults to the value set at init.
        copy : bool, default True
            Return an independent copy rather than a live reference to the
            object's internal data.

        Returns
        -------
        pandas.DataFrame or pandas.Series
        """
        loglevel = lgr.getEffectiveLevel()
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)

        X_reduction = self._get_last(X_reduction=X_reduction)
        if X_reduction is False:
            out = self._X
        else:
            try:
                out = self._X_dimred[_get_df_string("xdimred", xdimred=X_reduction)]
            except KeyError:
                available = "\n".join(list(self._X_dimred.keys()))
                lgr.critical_raise(f"No X dataframe for dimensionality reduction '{X_reduction}' "
                                   f"found! Available: {available}",
                                   KeyError)

        if maps is not None:
            keep = _match_maps(out.index, maps)
            if not keep:
                lgr.critical_raise(f"No maps matching {maps!r} found in X index.", ValueError)
            out = out.iloc[keep]

        if squeeze and len(out) == 1:
            out = out.squeeze()

        lgr.info(f"Returning X dataframe: \n{print_arg_pairs(X_reduction=X_reduction)}")
        lgr.setLevel(loglevel)
        return out.copy() if copy else out
    
    # ----------------------------------------------------------------------------------------------
    
    def get_y(self, Y_transform=None, maps=None, squeeze=False, verbose=None, copy=True):
        """
        Retrieve the object's Y data: either the raw, fitted Y, or a
        previously computed transform (see :meth:`transform_y`).

        Parameters
        ----------
        Y_transform : str, optional
            Label of the transform to retrieve (the formula string passed to
            ``transform_y(transform=...)``). Defaults to the last one used, or
            the raw Y data if none has been used. Raises ``KeyError`` (listing
            available labels) if the requested transform was never computed.
        maps : str or list of str, optional
            Restrict to matching Y map labels (exact or substring match).
        squeeze : bool, default False
            If exactly one map remains after any ``maps`` filtering, return it
            as a ``pandas.Series`` instead of a single-row DataFrame.
        verbose : bool, optional
            Print progress messages. Defaults to the value set at init.
        copy : bool, default True
            Return an independent copy rather than a live reference to the
            object's internal data.

        Returns
        -------
        pandas.DataFrame or pandas.Series
        """
        loglevel = lgr.getEffectiveLevel()
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)

        Y_transform = self._get_last(Y_transform=Y_transform)
        if Y_transform is False:
            out = self._Y
        else:
            try:
                out = self._Y_trans[_get_df_string("ytrans", ytrans=Y_transform)]
            except KeyError:
                available = "\n".join([k.replace("ytrans-", "") for k in self._Y_trans.keys()])
                lgr.critical_raise(f"No Y dataframe for transform '{Y_transform}' found! "
                                   f"Available: {available}",
                                   KeyError)

        if maps is not None:
            keep = _match_maps(out.index, maps)
            if not keep:
                lgr.critical_raise(f"No maps matching {maps!r} found in Y index.", ValueError)
            out = out.iloc[keep]

        if squeeze and len(out) == 1:
            out = out.squeeze()

        lgr.info(f"Returning Y dataframe: \n{print_arg_pairs(Y_transform=Y_transform)}")
        lgr.setLevel(loglevel)
        return out.copy() if copy else out
    
    # ----------------------------------------------------------------------------------------------
         
    def get_z(self, verbose=None, copy=True):
        """
        Retrieve the object's Z (covariate) data, as originally provided at
        :meth:`fit` (or as last overwritten in place by
        :meth:`transform_z`, if used). Unlike :meth:`get_x`/:meth:`get_y`,
        there is no per-transform lookup for Z -- only the current Z is
        available.

        Parameters
        ----------
        verbose : bool, optional
            Print progress messages. Defaults to the value set at init.
        copy : bool, default True
            Return an independent copy rather than a live reference to the
            object's internal data.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ValueError
            If no Z data was ever provided.
        """
        loglevel = lgr.getEffectiveLevel()
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        
        out = self._Z
        if out is None:
            lgr.critical_raise("No Z dataframe found!",
                               ValueError)
            
        lgr.info("Returning Z dataframe.")
        lgr.setLevel(loglevel)
        return out.copy() if copy else out  
    
    # ----------------------------------------------------------------------------------------------
   
    def get_colocalizations(self, method=None, stats=None,
                            X_reduction=None, Y_transform=None, xsea=None,
                            normalized=False, perm=None,
                            get_nulls=False, nulls_permute_what=None, pooled_p=None,
                            force_dict=False, verbose=None, copy=True):
        """
        Retrieve a stored :meth:`colocalize` result (or, with
        ``normalized=True``, a stored :meth:`normalize_colocalizations`
        result -- this is what :meth:`get_normalized_colocalizations` calls
        under the hood).

        Parameters
        ----------
        method, X_reduction, Y_transform, xsea : optional
            Identify which stored result to retrieve; see :meth:`colocalize`.
            All default to the last-used values.
        stats : str or list of str, optional
            Which statistic(s) to retrieve (e.g. ``"rho"``, ``"beta"``).
            Defaults to all stats produced by ``method`` (or, if
            ``normalized=True``, only the subset for which a null distribution
            exists).
        normalized : bool, default False
            Retrieve the null-normalized z-scores from
            :meth:`normalize_colocalizations` instead of the raw observed
            statistics. Raises ``KeyError`` if that hasn't been run.
        perm : str, optional
            Which permutation ("what") the retrieved normalized result was
            computed against; see :meth:`permute`. Only relevant with
            ``normalized=True``. Defaults to the last-used value.
        get_nulls : bool, default False
            Also return the null colocalization distributions alongside the
            observed statistics, as a ``(colocalizations, nulls)`` tuple.
            Requires ``nulls_permute_what`` to identify which permutation's
            nulls to fetch.
        nulls_permute_what : str, optional
            Which permutation's null distributions to fetch when
            ``get_nulls=True`` (e.g. ``"maps"``, ``"groups"``, ``"sets"``,
            ``"pairs"``; see :meth:`permute`'s ``what`` argument).
        pooled_p : bool or str, optional
            Pooling mode used for the stored null distribution being
            retrieved; see :meth:`permute`. Defaults to the last-used value.
        force_dict : bool, default False
            Always return a dict even when only one statistic is retrieved.
        verbose : bool, optional
            Print progress messages. Defaults to the value set at init.
        copy : bool, default True
            Return independent copies rather than live references to the
            object's internal data. Only applies to the non-``normalized``
            path; normalized results are always returned as copies.

        Returns
        -------
        pandas.DataFrame or dict of pandas.DataFrame, or a tuple thereof
            A dict of ``{stat: DataFrame}`` if more than one ``stats`` entry is
            retrieved or ``force_dict=True``, otherwise a single DataFrame.
            If ``get_nulls=True``, a ``(colocalizations, nulls)`` tuple is
            returned instead. See :meth:`colocalize`'s Returns section --
            ``"rho"`` (pearson/spearman/partial*) is on the Fisher-z scale by
            default (that call's ``r_to_z``, not re-resolved here), same key
            name regardless of scale.
        """
        loglevel = lgr.getEffectiveLevel()
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)

        method, X_reduction, Y_transform, xsea = self._get_last(
            method=method,
            X_reduction=X_reduction,
            Y_transform=Y_transform,
            xsea=xsea,
        )

        if normalized:
            perm, pooled_p = self._get_last(perm=perm, pooled_p=pooled_p)
            if stats is None:
                stats = _get_coloc_stats(method, permuted_only=True)
            elif isinstance(stats, str):
                stats = [stats]
            else:
                stats = list(stats).copy()
            out = dict()
            for stat in stats:
                z_str = _get_df_string(
                    "z",
                    xdimred=X_reduction, ytrans=Y_transform,
                    method=method, stat=stat,
                    xsea=xsea, perm=perm, pooled_p=pooled_p,
                )
                if z_str not in self._z_colocs:
                    lgr.critical_raise(
                        f"Normalized colocalizations for '{z_str}' not found. "
                        "Run normalize_colocalizations() first.",
                        KeyError
                    )
                out[stat] = self._z_colocs[z_str].copy()
            if not force_dict and len(out) == 1:
                out = out[stats[0]]
            lgr.info(f"Returning z-scored colocalizations.")
            lgr.setLevel(loglevel)
            return out

        if stats is None:
            stats = _get_coloc_stats(method)
        elif isinstance(stats, str):
            stats = [stats]
        else:
            stats = list(stats).copy()

        coloc_keys = list(self._colocs.keys())

        out = dict()
        for stat in stats:
            coloc_str = _get_df_string(
                "coloc",
                xdimred=X_reduction,
                ytrans=Y_transform,
                method=method,
                stat=stat,
                xsea=xsea
            )
            if coloc_str not in coloc_keys:
                if method=="mlr" and \
                    any([f"stat-{s}" not in coloc_str for s in ["individual", "intercept"]]):
                    stats.remove(stat)
                    continue
                else:
                    available = "\n".join(coloc_keys)
                    lgr.critical_raise(f"Colocalizations for '{coloc_str}' not found! "
                                       f"Available: {available}",
                                       KeyError)
            out[stat] = self._colocs[coloc_str].copy() if copy else self._colocs[coloc_str]
        
        if get_nulls and nulls_permute_what is None:
            lgr.error("If 'get_nulls' is True, 'nulls_permute_what' must not be None!")
            get_nulls = False

        if get_nulls:
            if nulls_permute_what not in ["groups", "groupsxmaps", "groupssets",
                                          "xmaps", "ymaps", "xymaps", "ymapssets",
                                          "sets", "pairs"]:
                lgr.critical_raise("If 'get_nulls' is True, 'nulls_permute_what' must be one of "
                                   "{'groups', '{x|y|xy}maps', 'sets', 'pairs'}!",
                                   ValueError)
            pooled_p = self._get_last(pooled_p=pooled_p)
            out_null = None

            # pairs permutation: null stored as flat array in self._nulls["pairs_null"]
            if nulls_permute_what == "pairs":
                null_str = _get_df_string(
                    "null",
                    xdimred=X_reduction,
                    ytrans=Y_transform,
                    method=method,
                    xsea=xsea,
                    perm="pairs",
                    pooled_p=pooled_p,
                )
                _pairs_cache = self._nulls.get("pairs_null", {}).get(null_str)
                if _pairs_cache is None:
                    lgr.error(
                        f"Pairs null distribution for '{null_str}' not found. "
                        "Run permute(what='pairs') first."
                    )
                else:
                    _null_dist = _pairs_cache["null_dist"]   # (n_perm,)
                    out_null = {
                        stat: {"within_pair": _null_dist[np.newaxis, :]}
                        for stat in stats
                    }

            else:
                null_str = _get_df_string(
                    "null",
                    xdimred=X_reduction,
                    ytrans=Y_transform,
                    method=method,
                    xsea=xsea,
                    perm=nulls_permute_what,
                    pooled_p=pooled_p,
                )
                if null_str not in self._nulls["_colocs"].keys():
                    available = "\n".join(list(self._nulls["_colocs"].keys()))
                    lgr.error(f"Null colocalizations for '{null_str}' not found! Available: {available}")
                else:
                    nulls = self._nulls["_colocs"][null_str].copy()

                    out_null = dict()
                    n_nulls = len(nulls)
                    with _quiet():
                        idx = self.get_p_values(method, nulls_permute_what, _COLOC_METHODS[method][0],
                                                xsea,
                                                pooled_p=pooled_p,
                                                X_reduction=X_reduction,
                                                Y_transform=Y_transform).index
                    for stat in stats:

                        if out[stat].shape[1] == 1:
                            out_null[stat] = pd.DataFrame(
                                {i: nulls[i][stat][:, 0] for i in range(n_nulls)},
                                index=idx
                            )

                        else:
                            out_null[stat] = dict()
                            for i_x, x in enumerate(out[stat].columns):
                                out_null[stat][x] = pd.DataFrame(
                                    {i: nulls[i][stat][:, i_x] for i in range(n_nulls)},
                                    index=idx
                                )
                
        # force return as dict if requested
        if not force_dict:
            if len(out)==1:
                out = out[stats[0]]
                
                if "out_null" in locals():
                    out_null = out_null[stats[0]]
        
        string = print_arg_pairs(method=method, xsea=xsea, X_reduction=X_reduction, 
                                 Y_transform=Y_transform)
        lgr.info(f"Returning colocalizations: \n{string}")
        lgr.setLevel(loglevel)
        return (out, out_null) if get_nulls else out  
    
    # ----------------------------------------------------------------------------------------------
    
    def get_p_values(self, method=None, permute_what=None, stats=None, xsea=None,
                     mc_method=None, pooled_p=None,
                     X_reduction=None, Y_transform=None, force_dict=False, verbose=None, copy=True):
        """
        Retrieve p-values computed by :meth:`permute`, uncorrected by default,
        or a specific multiple-comparisons-corrected result previously produced
        by :meth:`correct_p`.

        Parameters
        ----------
        method, X_reduction, Y_transform, xsea : optional
            Identify which stored colocalization result the p-values belong to;
            see :meth:`colocalize`. All default to the last-used values.
        permute_what : str, optional
            Which permutation ("what") the p-values were computed against, see
            :meth:`permute`'s ``what`` argument. Defaults to the last-used
            value.
        stats : str or list of str, optional
            Which statistic(s)' p-values to retrieve. Defaults to all stats for
            which a permutation null exists.
        mc_method : str, optional
            Which correction to retrieve, as passed to
            ``correct_p(mc_method=...)``. Defaults to ``None``, which retrieves
            the **uncorrected** p-values (the default, and the input
            :meth:`correct_p` itself corrects) -- not the last-used correction;
            for that, use :meth:`get_corrected_p_values`.
        pooled_p : bool or str, optional
            Pooling mode used for the stored result being retrieved; see
            :meth:`permute`. Defaults to the last-used value.
        force_dict : bool, default False
            Always return a dict even when only one statistic is retrieved.
        verbose : bool, optional
            Print progress messages. Defaults to the value set at init.
        copy : bool, default True
            Return independent copies rather than live references to the
            object's internal data.

        Returns
        -------
        pandas.DataFrame or dict of pandas.DataFrame
            A dict of ``{stat: DataFrame}`` if more than one statistic is
            retrieved or ``force_dict=True``, otherwise a single DataFrame.

        Raises
        ------
        KeyError
            If the requested combination was never computed (e.g.
            :meth:`permute` was never run).
        """
        loglevel = lgr.getEffectiveLevel()
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)

        method, X_reduction, Y_transform, xsea, permute_what, pooled_p = self._get_last(
            method=method,
            X_reduction=X_reduction,
            Y_transform=Y_transform,
            xsea=xsea,
            perm=permute_what,
            pooled_p=pooled_p,
        )
        
        if mc_method is not None:
            mc_method = _get_correct_mc_method(mc_method).replace("-", "").replace("_", "")

        self._check_permute(method, permute_what, mc_method, xsea, stats, X_reduction, Y_transform,
                            pooled_p=pooled_p)

        if stats is None:
            stats = _get_coloc_stats(method, permuted_only=True)
        elif isinstance(stats, str):
            stats = [stats]
        
        out = dict()
        for stat in stats:
            p_str = _get_df_string(
                "p",
                xdimred=X_reduction,
                ytrans=Y_transform,
                method=method,
                stat=stat,
                xsea=xsea,
                perm=permute_what,
                pooled_p=pooled_p,
                mc=mc_method,
            )
            if p_str not in self._p_colocs.keys():
                if "coloc-mlr_stat-individual" in p_str:
                    continue
                else:
                    available = "\n".join(list(self._p_colocs.keys()))
                lgr.critical_raise(f"Colocalization p values for '{p_str}' not found. "
                                   f"Available: {available}",
                                   KeyError)
            out[stat] = self._p_colocs[p_str].copy() if copy else self._p_colocs[p_str]
                
        if not force_dict:
            if len(out)==1:
                out = out[list(out.keys())[0]]
        
        string = print_arg_pairs(method=method, permute_what=permute_what, xsea=xsea,
                                 mc_method=mc_method,
                                 X_reduction=X_reduction, Y_transform=Y_transform)
        lgr.info(f"Returning p values: \n{string}")
        lgr.setLevel(loglevel)
        return out

    # ----------------------------------------------------------------------------------------------

    def get_corrected_p_values(self, mc_method=None, **kwargs):
        """
        Shortcut for :meth:`get_p_values` that defaults ``mc_method`` to
        whichever correction was last used in :meth:`correct_p`, instead of
        ``get_p_values``'s own default of retrieving the uncorrected p-values.

        Parameters
        ----------
        mc_method : str, optional
            Defaults to the last correction method used in :meth:`correct_p`.
            Raises ``ValueError`` if :meth:`correct_p` was never run.
        **kwargs
            Forwarded to :meth:`get_p_values` (``method``, ``permute_what``,
            ``stats``, ``xsea``, ``pooled_p``, ``X_reduction``, ``Y_transform``,
            ``force_dict``, ``verbose``, ``copy``).

        Returns
        -------
        pandas.DataFrame or dict of pandas.DataFrame
            See :meth:`get_p_values`.
        """
        mc_method = self._get_last(mc_method=mc_method)
        if mc_method is None:
            lgr.critical_raise(
                "No corrected p values found. Run correct_p() first.",
                ValueError
            )
        mc_method = _get_correct_mc_method(mc_method)
        return self.get_p_values(mc_method=mc_method, **kwargs)

    # ----------------------------------------------------------------------------------------------

    def get_normalized_colocalizations(self, **kwargs):
        """
        Shortcut for ``get_colocalizations(normalized=True, ...)`` -- retrieves
        the null-normalized z-scores produced by
        :meth:`normalize_colocalizations`.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`get_colocalizations` (``method``, ``stats``,
            ``X_reduction``, ``Y_transform``, ``xsea``, ``perm``, ``get_nulls``,
            ``nulls_permute_what``, ``pooled_p``, ``force_dict``, ``verbose``).

        Returns
        -------
        pandas.DataFrame or dict of pandas.DataFrame
            See :meth:`get_colocalizations`.
        """
        return self.get_colocalizations(normalized=True, **kwargs)

    # SAVE, LOAD, COPY =============================================================================

    def to_pickle(self, filepath, save_nulls=True, verbose=None):
        """
        Save the NiSpace object to a pickle file.

        Parameters
        ----------
        filepath : str
            Filepath to save the NiSpace object to.
        save_nulls : bool, optional
            Whether to save the null distributions. Defaults to True. If False, null
            colocalizations are dropped, which substantially reduces file size but prevents
            running correct_p('maxT'), correct_p('step_maxT'), or normalize_colocalizations()
            after reloading. Call those methods before saving if you intend to drop nulls.
        verbose : bool, optional
        """
        loglevel = lgr.getEffectiveLevel()
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        
        # remove nulls (very large depending on number of permutations) if requested
        self_save = self.copy()
        if not save_nulls:
            self_save._nulls = {
                "_colocs": {}
            }

        # save
        to_pickle(self_save, filepath, use_dill=True)
        lgr.debug(f"Saved NiSpace object to {filepath}.")  
        lgr.setLevel(loglevel)

    # ----------------------------------------------------------------------------------------------

    def copy(self, deep=True, verbose=True):
        """
        Duplicate this NiSpace object, e.g. to try an alternative analysis
        branch without mutating the original.

        Parameters
        ----------
        deep : bool, default True
            If True, recursively duplicate everything (all stored X/Y/Z data,
            colocalizations, nulls, p-values, etc.) so the copy shares no
            mutable state with the original. If False, only the top-level
            object is duplicated -- its attributes still reference the same
            underlying dicts/DataFrames as the original, so in-place mutation
            of e.g. a shared dict would affect both.
        verbose : bool, default True
            Print progress messages.

        Returns
        -------
        NiSpace
            The duplicated object.
        """
        loglevel = lgr.getEffectiveLevel()
        try:
            set_log(lgr, verbose)
            if deep==True:
                return copy.deepcopy(self)
            else:
                return copy.copy(self)
        finally:
            lgr.setLevel(loglevel)
            
    # ----------------------------------------------------------------------------------------------

    @staticmethod 
    def from_pickle(filepath, verbose=True):
        """
        Load a NiSpace object from a pickle file.

        Parameters
        ----------
        filepath : str
            Filepath to load the NiSpace object from.
        verbose : bool, optional
            Whether to print verbose output. Defaults to True.

        Returns
        -------
        nispace_object : NiSpace
            The loaded NiSpace object.
        """
        loglevel = lgr.getEffectiveLevel()
        verbose = set_log(lgr, verbose)

        # load
        nispace_object = from_pickle(filepath, use_dill=True)
        lgr.debug(f"Loaded NiSpace object from {filepath}.")

        # migrate legacy null map storage (pre-NullMaps refactor)
        _mn = nispace_object._nulls.get("maps_null")
        if isinstance(_mn, dict):
            lgr.info("Migrating legacy null maps dict to NullMaps.")
            method = nispace_object._nulls.pop("maps_null_method", None)
            nispace_object._nulls["maps_null"] = NullMaps.from_dict(_mn, null_method=method)
        elif isinstance(_mn, NullMaps) and _mn.null_method is None:
            # intermediate pickle: NullMaps present but null_method not yet an attribute
            nispace_object._nulls["maps_null"].null_method = \
                nispace_object._nulls.pop("maps_null_method", None)
        # clean up keys superseded by NullMaps attributes
        nispace_object._nulls.pop("maps_null_which", None)  # now NullMaps.null_which
        nispace_object._nulls.pop("maps_spin", None)        # now _parc_spin_mat

        # backfill storage dicts added in later versions -- pickle restores __dict__
        # directly and bypasses __init__, so an object pickled before one of these was
        # introduced would otherwise be missing it entirely (AttributeError on first use)
        for attr, default in [
            ("_coloc_kwargs_by_method", {}),
        ]:
            if not hasattr(nispace_object, attr):
                setattr(nispace_object, attr, default)

        # return
        lgr.setLevel(loglevel)
        return nispace_object


    # PRIVATE METHODS ==============================================================================
    
    def _check_fit(self, raise_error=True):
        if not (hasattr(self, "_X") | hasattr(self, "_Y")):
            if raise_error:
                lgr.critical_raise("Input data ('X', 'Y') not found. Did you run NiSpace.fit()?!",
                               ValueError)
            else:
                return False
        else:
            return True
    
    # ----------------------------------------------------------------------------------------------
    
    def _check_transform(self, ytrans=False, raise_error=True):
        y_str = _get_df_string("ytrans", ytrans=ytrans)
        lgr.debug(y_str)
        if y_str not in self._Y_trans.keys():
            if raise_error:
                lgr.critical_raise(f"Y transform = '{ytrans}' not found. Did you run "
                                   f"NiSpace.transform_y()?!",
                                   KeyError)
            else:
                return False
        else:
            return True
         
    # ----------------------------------------------------------------------------------------------
    
    def _check_colocalize(self, method, stats=None, xdimred=False, ytrans=False, xsea=False, 
                          raise_error=True):
        if stats is None:
            stats = _get_coloc_stats(method, drop_optional=True)
        elif isinstance(stats, str):
            stats = [stats]
            
        for stat in stats:
            coloc_str = _get_df_string("coloc", xdimred=xdimred, ytrans=ytrans, method=method, 
                                       stat=stat, xsea=xsea) 
            lgr.debug(coloc_str)
            if coloc_str not in self._colocs.keys():
                if raise_error:
                    lgr.critical_raise(
                        f"Colocalizations for method = '{method}', stat = '{stat}', "
                        f"X dimensionality reduction = '{xdimred}', and Y transform = '{ytrans}' "
                        f"not found. Did you run NiSpace.colocalize()?!",
                        KeyError
                    )
                else:
                    return False
                
        return True
    
    # ----------------------------------------------------------------------------------------------
    
    def _check_permute(self, method, permute_what, mc_method=None, xsea=False,
                       stats=None, xdimred=False, ytrans=False, pooled_p=False, raise_error=True):
        if stats is None:
            stats = _get_coloc_stats(method, drop_optional=True, permuted_only=True)
        elif isinstance(stats, str):
            stats = [stats]

        for stat in stats:
            p_str = _get_df_string("p", xdimred=xdimred, ytrans=ytrans, method=method, stat=stat,
                                    perm=permute_what, pooled_p=pooled_p, mc=mc_method, xsea=xsea).lower()
            lgr.debug(p_str)
            if p_str not in self._p_colocs:
                if raise_error:
                    lgr.critical_raise(
                        f"P values for permute_what = '{permute_what}', method = '{method}', "
                        f"stat = '{stat}', xsea = {xsea}, X dimensionality reduction = '{xdimred}', "
                        f"Y transform = '{ytrans}', and mc_method = '{mc_method}' not found. "
                        "Did you run NiSpace.permute()?!",
                        KeyError
                    )
                else:
                    return False
                
        return True
        
    # ----------------------------------------------------------------------------------------------
    
    def _get_last(self, **kwargs):
        out = []
        for arg, value in kwargs.items():
            if not arg in self._last_settings:
                lgr.critical_raise(f"Last setting for '{arg}' not found. "
                                   f"Available: {list(self._last_settings.keys())}")
            else:
                if value is None:
                    value_last = self._last_settings[arg]
                    if isinstance(value_last, str):
                        value_last = value_last.lower()
                    out.append(value_last)
                else:
                    out.append(value)
        return tuple(out) if len(out) > 1 else out[0]
    
    # ----------------------------------------------------------------------------------------------
    
    def _set_last(self, **kwargs):
        for arg, value in kwargs.items():
            self._last_settings[arg] = value
        
    # ----------------------------------------------------------------------------------------------
        
    def _get_dist_mat(self, dist_mat_type, centroids=False, parc_resample=2,
                      n_proc=None, store=True, verbose=None, force_generate=False):
        loglevel = lgr.getEffectiveLevel()
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)

        if self._parc is None:
            lgr.critical_raise(
                "Distance matrix computation requires a parcellation. "
                "Provide one via NiSpace(parcellation=...).",
                ValueError,
            )

        if dist_mat_type not in ["cv", "null_maps"]:
            lgr.critical_raise(f"dist_mat_type = '{dist_mat_type}' not defined",
                               ValueError)
        
        dist_mat_dict = self._parc_dist_mat
        generate_dist_mat = True
        if not force_generate and dist_mat_type in dist_mat_dict:
            dist_mat = dist_mat_dict[dist_mat_type]
            if dist_mat is not None:
                generate_dist_mat = False
            
        if generate_dist_mat:
            _ns_result = self._parc.get_null_space()
            # for combined parcellations get_null_space returns nested tuple — use sc (MNI) space
            null_space = _ns_result[1][0] if isinstance(_ns_result[0], tuple) else _ns_result[0]
            # ensure the null space is loaded and its derived attrs (hemi, idc, …) are computed
            self._parc._ensure_image_loaded(null_space)
            if null_space not in self._parc._hemi_dict:
                self._parc._fit_space(null_space)
            dist_mat = get_distance_matrix(
                parc=self._parc.get_image(null_space),
                parc_space=null_space,
                parc_hemi=self._parc.get_hemi(null_space),
                parc_resample=parc_resample,
                centroids=centroids,
                surf_euclidean=True if dist_mat_type=="cv" else False,
                n_proc=self._n_proc if not n_proc else n_proc,
                verbose=verbose
            )
        
        if store:
            self._parc_dist_mat[dist_mat_type] = dist_mat
            
        lgr.setLevel(loglevel)
        return dist_mat
    
    # ----------------------------------------------------------------------------------------------
    