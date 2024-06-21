import copy
import gzip
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
# opt. dependencies for combat harmonization
try:
    from neuroHarmonize import harmonizationLearn
    from neuroCombat import neuroCombat
    _NEUROHARMONIZE_AVAILABLE = True
except ImportError:
    _NEUROHARMONIZE_AVAILABLE = False

from . import lgr
from .io import parcellate_data, load_distmat
from .modules.reduce_x import _reduce_dimensions
from .modules.transform_y import _dummy_code_groups, _num_code_subjects, _get_transform_fun
from .modules.colocalize import _get_colocalize_fun, _sort_colocs, _get_coloc_stats
from .modules.permute import _get_null_maps, _get_exact_p_values, _get_correct_mc_method
from .modules.plot import _plot_categorical
from .modules.constants import _PARCS, _PARCS_DEFAULT, _COLOC_METHODS
from .datasets import fetch_parcellation, fetch_template
from .nulls import get_distance_matrix
from .stats.coloc import *
from .stats.misc import mc_correction, residuals_nan, zscore_df, permute_groups
from .cv import _get_dist_dep_splits, _get_rand_splits
from .plotting import nice_stats_labels
from .utils import (set_log, fill_nan, _get_df_string, _lower_strip_ws, mean_by_set_df,
                    get_column_names, lower, print_arg_pairs)


# ==================================================================================================
# DEFINE CLASS
# ==================================================================================================

class NiSpace:
    """
    The NiSpace class. Docs under construction.

    Initialize the NiSpace model.
    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The input data to fit the model.
        
    y : array-like of shape (n_samples, n_features), optional
        The target data to fit the model. Default is None.
    
    z : array-like of shape (n_samples, n_features), optional
        Additional data to fit the model. Default is None.
    
    Returns
    -------
    nothing
    """

    def __init__(self, 
                 x, 
                 y=None, 
                 z=None, 
                 x_labels=None, y_labels=None, z_labels=None, 
                 data_space="MNI152", 
                 standardize="xz", 
                 drop_nan=False,    
                 parcellation=None, 
                 parcellation_labels=None, 
                 parcellation_space="MNI152", 
                 parcellation_hemi=["L", "R"], 
                 #parcellation_density="10k",
                 parcellation_dist_mat=None,
                 resampling_target="data",
                 n_proc=1, 
                 verbose=True,
                 dtype=np.float32):
        
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
                               f"len==3! Is {type(data_space)} with len({len(data_space)}).",
                               ValueError)
        self._data_space = data_space
        self._parc = parcellation
        self._parc_info = {
            "labels": parcellation_labels,
            "space": parcellation_space,
            "hemi": parcellation_hemi,
            #"density": parcellation_density,
        }
        self._parc_dist_mat = {
            "null_maps": parcellation_dist_mat
        }
        self._resampl_target = resampling_target
        self._n_proc = n_proc
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
        self._nulls = {
            "_colocs": {}
        }
        self._p_colocs = {}
        
        # defaults for get functions (IMPORTANT: this determines what coloc and get function will do!)
        self._last_settings = {
            "method": None, 
            "X_reduction": False, 
            "Y_transform": False,
            "xsea": False, 
        }
    
    
    # FIT ==========================================================================================
    
    def fit(self):
        """
        "Fit" the NiSpace class instance.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        verbose = set_log(lgr, self._verbose)
        lgr.info("*** NiSpace.fit() - Data extraction and preparation. ***")
        
        ## handle integrated parcellations
        if self._parc is None:
            self._parc = _PARCS_DEFAULT
        if isinstance(self._parc, str):
            if self._parc.lower() in _PARCS:
                try:
                    parc, labels, space, density, dist_mat = fetch_parcellation(
                        parcellation=self._parc,                 
                        return_space=True,
                        return_resolution=True,
                        return_dist_mat=True,
                        return_loaded=True
                    )
                    self._parc = parc
                    self._parc_info["labels"] = labels
                    self._parc_info["space"] = space
                    #self._parc_info["density"] = density
                    self._parc_info["hemi"] = ("L", "R") if space=="fsaverage" else None
                    self._parc_dist_mat["null_maps"] = dist_mat
                    lgr.info("Loaded integrated parcellation with pre-calculated distance matrix.")
                except FileNotFoundError:
                    pass

        ## extract input data
        _input_kwargs = dict(
            parcellation=self._parc, 
            parc_labels=self._parc_info["labels"],
            parc_hemi=self._parc_info["hemi"],
            parc_space=self._parc_info["space"],
            resampling_target=self._resampl_target,
            n_proc=self._n_proc,
            verbose=verbose,
            dtype=self._dtype,
        )
        
        # reference data -> usually e.g. PET atlases
        # TODO: GSEA INPUT MANAGEMENT
        lgr.info("Checking input data for 'x' (should be, e.g., PET data):")
        self._X, self._parc = parcellate_data(
            self._x, 
            data_labels=self._x_lab,
            data_space=self._data_space[0], 
            return_parc=True,
            **_input_kwargs
        )
        lgr.info(f"Got 'x' data for {self._X.shape[0]} x {self._X.shape[1]} parcels.")
        
        # target data -> usually e.g. subject data or group-level outcome data
        if self._y is None:
            lgr.warning("No 'y' data detected. Will use 'X' as both reference and target data!")
            self._Y = self._X
            self._x_with_self = True
        else:
            lgr.info("Checking input data for 'y' (should be, e.g., subject data):")
            self._Y = parcellate_data(
                self._y, 
                data_labels=self._y_lab,
                data_space=self._data_space[1], 
                **_input_kwargs
            )
        lgr.info(f"Got 'y' data for {self._Y.shape[0]} x {self._Y.shape[1]} parcels.")
        
        # data to control correlations for
        if self._z is not None:
            lgr.info("Checking input data for z (should be, e.g., grey matter data):")
            if isinstance(self._z, str):
                if self._z in ["GM", "GMV", "gm", "gmv"]:
                    lgr.info("Using standard grey matter probability map as 'z' for GMV-control.")
                    self._z = [fetch_template("MNI152", desc="gmprob")]
                    self._z_lab = ["gm"]
            self._Z = parcellate_data(
                self._z, 
                data_labels=self._z_lab,
                data_space=self._data_space[2],
                **_input_kwargs
            )
            lgr.info(f"Got 'z' data for {self._Z.shape[0]} x {self._Z.shape[1]} parcels.")
            if self._Z.shape[0] not in [1, self._Y.shape[0]]:
                lgr.warning(f"Z data is not same shape as Y data ({self._Y.shape}) or shape "
                            f"(1,{self._X.shape[1]}). Check if this is intended!: {self._Z.shape}")
            
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
        
        ## check distance matrix
        if self._parc_dist_mat["null_maps"] is not None:
            dist_mat = load_distmat(self._parc_dist_mat["null_maps"])
            if not isinstance(dist_mat, tuple):
                dist_mat = dist_mat, 
            for d in dist_mat:
                if d.shape[0] != d.shape[1] != (self._X.shape[1] if len(dist_mat) == 1 
                                                else self._X.shape[1] / 2):
                    lgr.warning(f"Provided distance matrix shape {d.shape} is not symmetric or "
                                f"does not fit with number of parcels in data ({self._X.shape[1]})!"
                                " Ignoring provided matrix.")
                    dist_mat = None
                    break
            self._parc_dist_mat["null_maps"] = dist_mat[0] if len(dist_mat) == 1 else dist_mat
            
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
                                    
        ## return complete object
        return self


    # REDUCE DIMENSIONS ============================================================================
    
    def reduce_x(self, reduction, 
                 mean_by_set=False, weighted_mean=False,
                 n_components=None, min_ev=None, fa_method="minres", fa_rotation="promax",
                 seed=None, store=True, verbose=None):
        """
        Performs dimensionality reduction on X data.
        Under construction.
        """
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        lgr.info("*** NiSpace.reduce_x() - X dimensionality reduction. ***")
        
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
            lgr.error(f"Dimensionality reduction '{reduction}' not defined!",
                      ValueError)
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
                
        if reduction in ["pca", "ica", "fa"]:
            return _X_reduced, ev, loadings
        else:
            return _X_reduced
        
    
    # CLEAN ========================================================================================
    
    def clean_y(self, how, 
                covariates_within=None, 
                covariates_between=None, 
                combat=False, combat_keep=None, combat_train=None, combat_model=None, combat_kwargs={},
                n_proc=None, replace=True, verbose=None):
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        lgr.info(f"*** NiSpace.clean_y() - Y covariate regression. ***")
        
        ## check if fit was run
        self._check_fit()
        
        ## number of runners
        n_proc = self._n_proc if n_proc is None else n_proc
        
        ## Y data
        Y = self._Y
        Y_arr = np.array(Y)
        
        ## clean
        if isinstance(how, str):
            how = [how]
        msg = (f"Input '{how}' of type '{type(how)}' for argument 'how' not known! " 
               "Must be (list-like of) 'within' for regression within maps/subjects " 
               "(e.g., GM TPM) or 'between' for regression across maps/subjects.")
        if not isinstance(how, (list, tuple, set)):
            lgr.critical_raise(msg, TypeError)
        if not all(h in ["within", "between"] for h in how):
            lgr.critical_raise(msg, ValueError)
        wcov_arr, bcov_arr = None, None
        
        # regression within subjects (across parcels)
        if "within" in how and covariates_within is not None:
            lgr.info("Performing covariate regression within map/subjects (e.g., grey matter maps).")
            # use Z data
            if isinstance(covariates_within, str):
                if covariates_within in ["z", "Z"]:
                    lgr.info("Using Z data for 'within' covariate regression.")
                    if self._Z is not None:
                        wcov_arr = np.array(self._Z)
                        self._clean_y_z = True
                    else:
                        lgr.critical_raise("Provide Z data at initialization for Z regression!",
                                           ValueError)
                else:
                    lgr.error(f"'within' covariate {covariates_within} not defined! "
                              "Pass 'Z' if you want to regress Z data, or provide a custom array.")
            # use other data
            elif isinstance(covariates_within, (np.ndarray, pd.Series, pd.DataFrame)):
                wcov_arr = np.array(covariates_within, dtype=self._dtype)
                # ensure 2d in NiSpace format
                if wcov_arr.ndim == 1:
                    wcov_arr = wcov_arr[np.newaxis, :]
                # check shape
                lgr.info(f"Assuming {wcov_arr.shape[0]} 'within' covariate map(s) for "
                         f"{wcov_arr.shape[1]} parcels.")
                if wcov_arr.shape[1] != Y.shape[1] or wcov_arr.shape[0] not in [1, Y.shape[0]]:
                    lgr.error(f"Covariate data shape ({wcov_arr.shape}) does not match Y data! "
                              f"Must be (n, {Y.shape[1]}) or ({Y.shape},).")
                    wcov_arr = None
            # type not known
            else:
                lgr.critical_raise("Provided 'covariates_within' of type "
                                   f"{type(covariates_within)} not supported!",
                                   TypeError)
            # run
            if wcov_arr is not None:
                # copy data if only one map
                if wcov_arr.shape[0] == 1:
                    wcov_arr = np.tile(wcov_arr, (Y.shape[0], 1))
                Y_partial = Parallel(n_jobs=n_proc)(
                    delayed(residuals_nan)(wcov_arr[i_y, :], Y_arr[i_y, :]) for i_y in tqdm(
                        range(Y.shape[0]), 
                        desc=f"Regressing within covariate(s) on Y ({n_proc} proc)", 
                        disable=not verbose
                )) 
                Y_arr = np.array(Y_partial, dtype=self._dtype)
                
        # regression/harmonization across subjects
        if "between" in how and covariates_between is not None:
            lgr.info("Performing covariate regression between maps/subjects (e.g., age, sex, site).")
            # process data
            if isinstance(covariates_between, (np.ndarray, pd.Series, pd.DataFrame)):
                bcov_arr = np.array(covariates_between, dtype=self._dtype)
                # ensure 2d in NiSpace format
                if bcov_arr.ndim == 1:
                    bcov_arr = bcov_arr[:, np.newaxis]
                # check shape
                lgr.info(f"Assuming {bcov_arr.shape[1]} 'between' covariate(s) for "
                         f"{bcov_arr.shape[0]} maps/subjects.")
                if bcov_arr.shape[0] != Y.shape[0]:
                    lgr.error(f"Covariate data shape does not match Y data! Must be ({Y.shape[0]},n).")
                    bcov_arr = None
                # get cov names as a list; all strings are lowercase only, returns [None] if ndarray
                bcov_names = lower(get_column_names(covariates_between, force_list=True))
                # if combat requested, check for site variable
                if combat:
                    if (bcov_names == [None]) or ("site" not in bcov_names):
                        lgr.warning("For ComBat harmonization, provide between_covariates as a "
                                    "DataFrame with one column named - or Series of name - 'site'.")
                        lgr.warning("Not performing ComBat harmonization.")
                        combat = False
                    elif not _NEUROHARMONIZE_AVAILABLE:
                        lgr.critical_raise("Optional dependency: neuroHarmonize. Run 'pip install neurocombat "
                                           "neuroharmonize' in your environment to use ComBat harmonization.")
                        combat = False
                    else:
                        # split into site covariate and other covariate arrays
                        bcov_site = bcov_arr[:, np.array(bcov_names) == "site"]
                        if combat_keep is not None and isinstance(combat_keep, (str, list, tuple, set, np.ndarray)):
                            if isinstance(combat_keep, str):
                                combat_keep = [combat_keep]
                            else:
                                combat_keep = list(combat_keep)
                            bcov_keep = bcov_arr[:, np.isin(bcov_names, combat_keep)]
                            bcov_arr = bcov_arr[:, ~np.isin(bcov_names, combat_keep + ["site"])]
                        else:
                            lgr.warning("With ComBat harmonization, you might want to define some "
                                        "covariates to retain: pass a list-like via 'combat_keep'.")
                            bcov_keep = None
                            bcov_arr = bcov_arr[:, np.array(bcov_names) != "site"]
                    
            # type not known
            else:
                lgr.critical_raise(f"Provided 'covariates_between' of type "
                                   f"{type(covariates_between)} not supported!",
                                   TypeError)
                            
            # run everything instead of combat 
            if bcov_arr is not None:
                Y_partial = Parallel(n_jobs=n_proc)(
                    delayed(residuals_nan)(bcov_arr, Y_arr[:, i_p]) for i_p in tqdm(
                        range(Y.shape[1]), 
                        desc=f"Regressing {bcov_arr.shape[1]} between covariate(s) on Y ({n_proc} proc)", 
                        disable=not verbose
                )) 
                Y_arr = np.array(Y_partial, dtype=self._dtype).T
                
                # COMBAT
                if combat:
                    lgr.info("Performing combat harmonization, retaining "
                             f"{bcov_keep.shape[1] if bcov_keep is not None else 0} covariates.")
                    # covariate df
                    combat_covariates = pd.DataFrame(bcov_site, columns=["SITE"])
                    if bcov_keep is not None:
                        combat_covariates = pd.concat(
                            [combat_covariates, pd.DataFrame(bcov_keep, columns=combat_keep)], 
                            axis=1
                        )
                    # missings warning
                    Y_arr_isnan = np.isnan(Y_arr)
                    if Y_arr_isnan.any().any():
                        lgr.warning("Detected missing values in Y data, which is not supported with "
                                    "ComBat harmonization. Missing values will be imputed with "
                                    "map-wise medians and replaced by nan after harmonization. "
                                    "CAVE: experimental feature!")
                        Y_arr = np.apply_along_axis(
                            lambda x: np.where(np.isnan(x), np.nanmedian(x), x), 
                            axis=1, 
                            arr=Y_arr,
                        )
                    # y train index
                    if isinstance(combat_train, (list, tuple, set, np.ndarray, pd.Series)):
                        if len(combat_train) == Y.shape[0]:
                            if all(i in {True, False, 0, 1} for i in combat_train):
                                idx_train = np.array(combat_train).astype(bool)
                    if (combat_train is not None) and ("idx_train" not in locals()):
                        lgr.warning(f"'combat_train' must be boolean vector of length {Y.shape[0]}! "
                                    "Setting 'combat_train' to None.")
                        combat_train = None
                    # apply
                    from neuroHarmonize import harmonizationLearn, harmonizationApply
                    if combat_model is None:
                        if combat_train is None:
                            combat_model, Y_arr = harmonizationLearn(
                                data=Y_arr, 
                                covars=combat_covariates,
                                **combat_kwargs
                            )
                        else:
                            lgr.info(f"Training model on {idx_train.sum()} subjects.")
                            temp = np.zeros(Y_arr.shape)
                            combat_model, temp[idx_train, :] = harmonizationLearn(
                                data=Y_arr[idx_train, :], 
                                covars=combat_covariates.loc[idx_train, :],
                                **combat_kwargs
                            )
                            lgr.info(f"Applying model to {(~idx_train).sum()} subjects.")
                            temp[~idx_train, :] = harmonizationApply(
                                data=Y_arr[~idx_train, :],
                                covars=combat_covariates.loc[~idx_train, :],
                                model=combat_model
                            )
                            Y_arr = temp
                    else:
                        Y_arr = harmonizationApply(
                            data=Y_arr,
                            covars=combat_covariates,
                            model=combat_model
                        )
                    # put back the original nan values
                    Y_arr[Y_arr_isnan] = np.nan
                    # store
                    self._clean_y_combat_model = combat_model
                    self._clean_y_combat_cov = combat_covariates
                        
        # done nothing
        if wcov_arr is None and bcov_arr is None:
            lgr.warning("No covariate regression performed! Set 'how' to 'between' and/or 'within "
                        "and provide covariate arrays through 'covariates_{within|between}'!")
        
        # to df
        Y = pd.DataFrame(Y_arr, columns=Y.columns, index=Y.index, dtype=self._dtype)
        
        ## save
        if replace:
            self._Y = Y
        
        return Y
    
    
    # TRANSFORM Y ==================================================================================    
        
    def transform_y(self, transform, groups=None, subjects=None, Y=None,
                    Y_name="Y", store=True, verbose=None):
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        lgr.info(f"*** NiSpace.transform_{Y_name.lower()}() - {Y_name} transformation and comparison. ***")
        
        ## check if fit was run
        self._check_fit()
        
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
                                                     dtype=self._dtype)
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
            
        return _Y_trans 

    
    # TRANSFORM Z ==================================================================================
    
    def transform_z(self, transform="Y", groups="Y", subjects="Y",
                    replace=True, verbose=None):
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
        return _Z_trans
    
    
    # COLOCALIZE ===================================================================================

    def colocalize(self, method=None, X_reduction=None, Y_transform=None, xsea=None, 
                   xsea_aggregation_method="mean",
                   X=None, Y=None, Z=None, 
                   Z_regression=True, 
                   store=True, n_proc=None, seed=None, verbose=None,
                   dist_mat_kwargs={},
                   **kwargs):
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        
        lgr.info("*** NiSpace.colocalize() - Estimating X & Y colocalizations. ***")
        
        ## check if fit was run 
        self._check_fit()
        
        ## settings
        n_proc = self._n_proc if n_proc is None else n_proc
        dtype = self._dtype
        
        ## settings
        method, X_reduction, Y_transform, xsea = self._get_last(
            method=method, 
            X_reduction=X_reduction, 
            Y_transform=Y_transform, 
            xsea=xsea
        )
        if method is None:
            coloc_methods = ", ".join(list(_COLOC_METHODS.keys()))
            lgr.critical_raise(f"No colocalization method defined! Supported: {coloc_methods}", 
                               ValueError)
        else:
            lgr.info(f"Running '{method}' colocalization" + \
                     (f" on '{X_reduction}'-reduced X data" if X_reduction else "") + \
                     (f" with '{Y_transform}' transform" if Y_transform else "") + ".")
        
        ## get X and Y data (so this function can be run on direct X & Y input data)
        # X
        if not X:
            if not X_reduction:
                X = self._X
            else:
                X = self.get_x(X_reduction=X_reduction, verbose=False)
        X_arr = np.array(X, dtype=dtype)
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
            X_arr = {set_name: np.array(set_X, dtype=self._dtype) 
                     for set_name, set_X in X.groupby(level="set", sort=False)}
            lgr.info(f"Using {len(X_arr)} sets with between "
                     f"{X.index.get_level_values('set').value_counts().min()} and "
                     f"{X.index.get_level_values('set').value_counts().max()} samples. "
                     f"Aggregating within-set colocalizations with: {xsea_aggregation_method}.")
            if ("spearman" in method or "pearson" in method):
                if hasattr(kwargs, "r_to_z"):
                    if kwargs["r_to_z"] is False:
                        lgr.warning("XSEA with correlation colocalization requires Fisher's Z "
                                    "transform! Will set 'r_to_z' = True.")
                        kwargs["r_to_z"] = True
            self._xsea = True
            self._xsea_aggregation_method = xsea_aggregation_method
            
        # Y
        groups = kwargs.pop("groups", None)
        subjects = kwargs.pop("subjects", None)
        if not Y:
            if not Y_transform:
                Y = self._Y
            else:
                if not self._check_transform(ytrans=Y_transform, raise_error=True):
                    lgr.warning(f"Y transform '{Y_transform}' was not run before. Running now.")
                    self.transform_y(Y_transform, groups, subjects)
                else:
                    Y = self.get_y(Y_transform=Y_transform, verbose=False)        
        Y_arr = np.array(Y, dtype=dtype)
        
        # Z
        if not Z:
            Z = self._Z
        if Z is None and "partial" in method:
            temp = method.replace('partial', '')
            lgr.error(f"Provide Z data for method '{method}'! Using method '{temp}' instead.") 
            method = temp.copy()
        elif not Z_regression and "partial" in method:
            lgr.warning(f"Method '{method}' entails Z regression, will set 'Z_regression' = True.")
            Z_regression = True
        if Z_regression and Z is not None:
            lgr.info(f"Will regress Z from Y {'during' if 'partial' in method else 'before'} "
                     "colocalization calculation.")
            Z_arr = np.array(Z, dtype=dtype)
            
            # reasons to skip: Z regr. already performed, wrong shape
            msg = ""
            if hasattr(self, "_clean_y_z"):
                msg = "It seems, Z regression was performed using NiSpace.clean_y()."
                if "partial" in method:
                    msg += f" Method '{method}' entails Z regression. This will result in an error."
            elif Z_arr.shape[0] not in [1, Y_arr.shape[0]]:
                msg = f"Number of Z maps ({Z_arr.shape[0]}) must equal number of Y maps " + \
                      f"({Y_arr.shape[0]}) or be 1!"
            if len(msg) > 0:
                if Z_regression != "force":
                    lgr.warning(msg + " Will not perform Z regression.")
                    Z_regression, Z_arr = False, None
                else:
                    lgr.warning(" Forcing Z regression. Check results validity!")
                    
            if Z_arr is not None:
                # if Z is one map, we assume average map (e.g., MNI152 TPM) and repeat it along ax 0
                if Z_arr.shape[0] == 1 and Y_arr.shape[0] > 1:
                    lgr.info("Found one Z map. Will be regressed from every Y.")
                    Z_arr = np.tile(Z_arr, (Y_arr.shape[0], 1))
                # if Z and Y have same amount of maps, will leave it as is
                elif Z_arr.shape[0] == Y_arr.shape[0]:
                    lgr.info("Found equal number of Z and Y maps. Will perform map-wise regression.")
        else:
            Z_regression, Z_arr = False, None
                
        ## special case regularized regression: we need euclidean distance matrices
        parcel_tr_te_splits, parcel_train_pct = None, None
        if (method in ["lasso", "ridge", "elasticnet"]):
            
            parcel_tr_te_splits = dist_mat_kwargs.pop("parcel_tr_te_splits", None)
            euclidean_dist_mat = dist_mat_kwargs.pop("euclidean_dist_mat", None)
            parcel_train_pct = dist_mat_kwargs.pop("parcel_train_pct", 0.75)
            
            if parcel_tr_te_splits is None:
                lgr.info("Fetching euclidean distance matrix for regularized regression "
                         "colocalization with n(parcel)-fold CV.")
                
                if self._zscore != False:
                    lgr.warning("Input data was Z-standardized, which might lead to leakage in CV!")
                
                
                if euclidean_dist_mat is None:
                    euclidean_dist_mat = self._get_dist_mat(
                        dist_mat_type="cv", 
                        n_proc=n_proc,
                        **dist_mat_kwargs
                    )
                    
                if any([s in self._parc_info["space"].lower() for s in ["mni", "fsa"]]):
                    lgr.info("Calculating distance-dependent parcel splits.")
                    self.parcel_tr_te_splits_works = _get_dist_dep_splits(
                        dist_mat=euclidean_dist_mat[np.ix_(self._no_nan, self._no_nan)], 
                        train_pct=parcel_train_pct
                    ) 
                else:
                    lgr.warning("Calculating random parcel splits as parcellation space not supported.")
                    parcel_tr_te_splits = _get_rand_splits(
                        train_pct=parcel_train_pct, 
                        seed=seed
                    ) 
                    
        # save colocalization settings
        self._coloc_kwargs = dict(
            regr_z=Z_regression,
            xsea=xsea,
            xsea_method=xsea_aggregation_method,
            parcel_train_pct=None,
            parcel_tr_te_splits=None,
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
            delayed(_y_colocalize)(X_arr, Y_arr[i_y, :], Z_arr[i_y, :] if Z_arr is not None else None) \
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
            # save last settings
            self._set_last(
                method=method,
                X_reduction=X_reduction,
                Y_transform=Y_transform,
                xsea=xsea
            )
            
        # return dict of dfs
        return _colocs
   
   
    # PERMUTE ======================================================================================
    
    def permute(self, what, method=None, X_reduction=None, Y_transform=None, xsea=None, 
                n_perm=10000, 
                maps_which="X", maps_nulls=None, maps_use_existing_nulls=True, 
                maps_null_method="moran", 
                maps_dist_mat=None, maps_dist_mat_centroids=False, maps_dist_mat_downsample=3,
                groups_perm_paired="auto", groups_perm_strategy="proportional",
                sets_X_background=None,
                p_tails=None, p_from_average_y_coloc="auto",
                n_proc=None, seed=None, store=True, verbose=None,
                **kwargs):
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        lgr.info("*** NiSpace.permute() - Estimate exact non-parametric p values. ***")

        ## check if fit was run
        self._check_fit()
        
        ## check for allowed permutation combinations
        # check what variable
        if isinstance(what, str):
            what = [what]
        elif isinstance(what, (list, tuple)):
            pass
        else:
            lgr.critical_raise(f"'what' must be list, tuple, or string, not {type(what)}",
                               ValueError)
        what = sorted(what)
        # check maps_which variable
        if maps_which:
            if isinstance(maps_which, str):
                maps_which = [maps_which]
            maps_which = sorted(maps_which)
            if maps_which not in [["X"], ["Y"], ["X", "Y"]]:
                lgr.critical_raise(f"'maps_which' has to be 'X', 'Y', or ['X', 'Y'] not '{maps_which}'",
                                   ValueError)
        # case X/Y maps
        if what == ["maps"]:
            perm_info = f"{' & '.join(maps_which)} maps"
        # case Y groups
        elif what == ["groups"]:
            perm_info = "Y groups"
        # case X sets
        elif what == ["sets"]:
            perm_info = "X sets"
        # case X/Y maps and Y groups
        elif what == ["groups", "maps"]:
            if maps_which != ["X"]:
                lgr.warning("Y map permutation not allowed in combination with Y group permutation. "
                            "Will set 'maps_which' = 'X' and permute X maps instead.")
                maps_which = ["X"]
            perm_info = "X maps and Y groups"
        # case X/Y maps and X sets
        elif what == ["maps", "sets"]:
            if maps_which != ["Y"]:
                lgr.warning("X set permutation not allowed in combination with X map permutation. "
                            "Will set 'maps_which' = 'Y' and permute Y maps instead.")
                maps_which = ["Y"]
            perm_info = "X sets and Y maps"
        # case X sets and Y groups
        elif what == ["groups", "sets"]:
            perm_info = "X sets and Y groups"
        # case X/Y maps, X sets, and Y groups
        elif what == ["groups", "maps", "sets"]:
            lgr.warning("Cannot perform simultaneous permutation of sets, maps, and groups. "
                        "Will run permutation of X sets and Y groups instead.")
            what = ["groups", "sets"]
        # case other
        else:
            lgr.critical_raise(f"'what' = '{what}' not defined!",
                               ValueError)
        lgr.info(f"Permutation of: {perm_info}.")
            
        ## settings
        method, X_reduction, Y_transform, xsea = self._get_last(
            method=method, 
            X_reduction=X_reduction, 
            Y_transform=Y_transform, 
            xsea=xsea
        )
        
        ## merge with settings from current NiSpace object        
        n_proc = n_proc if n_proc is not None else self._n_proc
        dtype = self._dtype
        
        ## check if colocalize was run
        xsea = True if ("sets" in what) or (xsea == True) else False
        if not self._check_colocalize(method, None, X_reduction, Y_transform, xsea, 
                                      raise_error=False):
            lgr.warning(f"'{method}' colocalization was not run before. Running now.")
            coloc_kwargs = dict(
                xsea=xsea,
                n_proc=n_proc,
                seed=seed,
                **kwargs
            )
            self.colocalize(method, X_reduction, Y_transform, **coloc_kwargs)
        
        ## get observed data
        # X
        if not X_reduction:
            _X_obs = self._X
        else:
            lgr.info(f"Loading dimensionality-reduced X data, reduction method = '{X_reduction}'.")
            _X_obs = self.get_x(X_reduction=X_reduction, verbose=False)
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
            _Y_trans_obs = self.get_y(Y_transform=Y_transform, verbose=False)
            _Y_trans_obs_arr = np.array(_Y_trans_obs, dtype=dtype)
        # Z
        _Z_obs = self._Z
        _Z_obs_arr = np.array(_Z_obs, dtype=dtype)

        ## get averaging method for p_from_average_y_coloc: "auto" -> decide based on number of Y maps,
        # "median", "mean" -> calculate p based on mean/median colocalization across Y maps, 
        # False -> calculate p for every Y map, anything else -> defaults to mean
        if p_from_average_y_coloc:
            if p_from_average_y_coloc == "auto":
                if _Y_obs.shape[0] > 1:
                    lgr.info("Will calculate p values for mean calculation across Y maps. Set "
                             "'p_from_average_y_coloc' = False to change this behavior.")
                    p_from_average_y_coloc = "mean"
                else:
                    p_from_average_y_coloc = False
            elif p_from_average_y_coloc not in ["mean", "median"]:
                p_from_average_y_coloc = "mean"
            self._nulls["p_from_average_y_coloc"] = p_from_average_y_coloc
                    
        ## get observed colocalizations as numpy arrays
        lgr.info(f"Loading observed colocalizations (method = '{method}').")
        _colocs_obs = self.get_colocalizations(
            method, 
            X_reduction=X_reduction, 
            Y_transform=Y_transform, 
            xsea=xsea,
            force_dict=True,
            verbose=False
        )
        _colocs_obs = {stat: np.array(df, dtype=dtype) for stat, df in _colocs_obs.items()}
                    
        # get average prediction values of all y if requested
        if p_from_average_y_coloc:                
            for stat in _colocs_obs.keys():
                if p_from_average_y_coloc=="median":
                    _colocs_obs[stat] = np.nanmedian(_colocs_obs[stat], axis=0)[np.newaxis, :]
                else:
                    _colocs_obs[stat] = np.nanmean(_colocs_obs[stat], axis=0)[np.newaxis, :]
        
        ## prepare permuted data as prerequisite for null colocalization runs
        _X_null, _Y_null, _Z_null = None, None, None
        
        # case permute X/Y brain maps
        if "maps" in what:
            # iterate map datasets to permute
            for XY in maps_which:
                lgr.info(f"Generating permuted {XY} maps.")
                
                # if no null maps & also no distance matrix given, generate distance matrix
                if (maps_nulls is None) & (maps_dist_mat is None):
                    maps_dist_mat = self._get_dist_mat(
                        dist_mat_type="null_maps", 
                        centroids=maps_dist_mat_centroids,
                        downsample_vol=maps_dist_mat_downsample
                    )
                
                # get null maps, will not generate new maps if already existing and use of 
                # existing is requested
                if XY=="X":
                    data_obs = _X_obs
                    standardize_nulls = True if "x" in self._zscore else False
                elif XY=="Y":
                    if Y_transform:
                        data_obs = _Y_trans_obs
                    else:
                        data_obs = _Y_obs
                    standardize_nulls = True if "y" in self._zscore else False
                maps_nulls = _get_null_maps(
                    data_obs=data_obs, 
                    null_maps=maps_nulls,
                    nispace_nulls=self._nulls, 
                    use_existing_maps=maps_use_existing_nulls, 
                    standardize=standardize_nulls,
                    null_method=maps_null_method,
                    dist_mat=maps_dist_mat,
                    parc=self._parc,
                    parc_kwargs=self._parc_info,
                    n_perm=n_perm, 
                    seed=seed, 
                    n_proc=n_proc, 
                    dtype=dtype,
                    verbose=verbose
                )
                
                # store null maps
                self._nulls["maps_null_method"] = maps_null_method
                self._nulls["maps_null"] = maps_nulls
                self._nulls["maps_null_which"] = XY

                # sort null map data into lists of length n_perm, each element being one 
                # permuted array of observed values 
                lgr.debug("Sorting null map data into arrays.")
                if XY=="X":
                    _X_null = [np.c_[[maps[i, :] for maps in maps_nulls.values()]].astype(self._dtype) 
                               for i in range(n_perm)]
                    # case: xsea requested: re-sort into a list of dicts of set-wise arrays
                    if isinstance(_X_obs_arr, dict):
                        idc_set = np.array(_X_obs.index.get_level_values("set"))
                        _X_null = [{set_name: null[idc_set == set_name, :] 
                                    for set_name in _X_obs_arr.keys()} 
                                   for null in _X_null]
                elif XY=="Y":
                    _Y_null = [np.c_[[maps[i, :] for maps in maps_nulls.values()]].astype(self._dtype) 
                               for i in range(n_perm)]
            
        # case permute Y groups
        if ("groups" in what) and Y_transform:
            lgr.info(f"Generating permuted Y groups.")
            
            # get groups without nan values
            groups = self._groups_no_nan
            if hasattr(self, "_subjects_no_nan"):
                subjects = self._subjects_no_nan
            else:
                subjects = None
            
            # Y values without nan values in group vector
            _Y_obs_arr_nonan = _Y_obs_arr[~self._groups_nan_idc, :]
            
            ## prepare formula & transform function
            Y_transform = _lower_strip_ws(Y_transform)
            apply_transform, paired = _get_transform_fun(Y_transform, return_df=False, 
                                                         return_paired=True, dtype=dtype)
            
            # paired permutations?
            if groups_perm_paired not in ["auto", True, False]:
                lgr.warning("Argument 'groups_perm_paired' must be of boolean type or 'auto' not "
                            f"'{groups_perm_paired}'! Setting to 'auto'.")
                groups_perm_paired = "auto"
            if groups_perm_paired == "auto":
                groups_perm_paired = paired
            
            # get list of permuted group labels
            lgr.info(f"Permuting groups/sessions vector, strategy: "
                     f"{'paired' if groups_perm_paired else 'unpaired'}, {groups_perm_strategy}.")
            groups_null = permute_groups(
                groups=groups, 
                subjects=subjects, 
                strategy=groups_perm_strategy, 
                paired=groups_perm_paired,
                n_perm=n_perm, 
                n_proc=n_proc,
                seed=seed,
                verbose=verbose
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
            
            
        ## check what permuted dataframes we have, if we dont have them, copy observed data (!)
        if (not _X_null) & (not _Y_null) & (not _Z_null):
            lgr.critical_raise("No permuted data generated. Supported permutations ('what') are: "
                               "'maps', 'groups', and 'sets'.",
                               ValueError)
        if not _X_null:
            _X_null = [_X_obs_arr] * n_perm
        if not _Y_null:
            _Y_null = [_Y_trans_obs_arr if "_Y_trans_obs" in locals() else _Y_obs_arr] * n_perm
        if not _Z_null:
            if _Z_obs is not None:
                if _Z_obs_arr.shape[0] == 1 and _Y_obs_arr.shape[0] > 1:
                    _Z_obs_arr = np.tile(_Z_obs_arr, (_Y_obs_arr.shape[0], 1)).astype(self._dtype)
                elif _Z_obs_arr.shape[0] != _Y_obs_arr.shape[0]:
                    lgr.critical_raise(f"Z data of wrong shape ({_Z_obs_arr.shape})!",
                                       ValueError)
                _Z_null = [_Z_obs_arr] * n_perm
            else:
                _Z_null = [None] * n_perm
                
        ## run null colocalizations
        # function to perform colocalization for one y vector (= per subject); see NiSpace.colocalize()
        # the function was saved by NiSpace.colocalize()
        _y_colocalize = self._colocs_fun[method]
        
        # function to perform colocalization for one X/Y/Z null array
        xsea = True if isinstance(_X_null[0], dict) else False
        #n_components = self._coloc_kwargs["n_components"]
        def par_fun(X_null, Y_null, Z_null=None):
            # run colocalization
            if Z_null is None:
                null_colocs_list = [
                    _y_colocalize(X_null, Y_null[i_y, :], None)
                    for i_y in range(Y_null.shape[0])
                ]
            else:
                null_colocs_list = [
                    _y_colocalize(X_null, Y_null[i_y, :], Z_null[i_y, :])
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
            # average colocalization if requested
            if p_from_average_y_coloc:
                for stat in null_colocs:
                    if p_from_average_y_coloc == "median":
                        null_colocs[stat] = np.nanmedian(null_colocs[stat], axis=0)[np.newaxis, :]
                    else:
                        null_colocs[stat] = np.nanmean(null_colocs[stat], axis=0)[np.newaxis, :]
            # return            
            return null_colocs
        
        # run in parallel
        if not xsea:
            _colocs_null = Parallel(n_jobs=n_proc)(
                delayed(par_fun)(_X_null[i], _Y_null[i], _Z_null[i]) 
                for i in tqdm(
                    range(n_perm), 
                    desc=f"Null colocalizations ({method}, {n_proc} proc)", disable=not verbose
                )
            )
        else:
            _colocs_null = Parallel(n_jobs=n_proc)(
                delayed(par_fun)(_xsea_perm_data(i), _Y_null[i], _Z_null[i]) 
                for i in tqdm(
                    range(n_perm), 
                    desc=f"Null colocalizations ({method}, {n_proc} proc)", disable=not verbose
                )
            )
            
        ## calculate exact p values
        # get values
        p_data, p_data_norm = _get_exact_p_values(
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
            if (p_from_average_y_coloc in ["mean", "median"]) & (_Y_obs.shape[0]>1):
                rows = [p_from_average_y_coloc]
            elif "_Y_trans_obs" in locals():
                rows = _Y_trans_obs.index
            else:
                rows = _Y_obs.index
            p_data[stat] = pd.DataFrame(p_data[stat], columns=cols, index=rows)
            p_data_norm[stat] = pd.DataFrame(p_data_norm[stat], columns=cols, index=rows)
        
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
                    perm=perm
                )
                self._p_colocs[df_str] = p_data[stat]
                self._p_colocs[df_str.replace("norm-false", "norm-true")] = p_data_norm[stat]
            df_str = _get_df_string(
                "null", 
                xdimred=X_reduction, 
                ytrans=Y_transform, 
                method=method,
                xsea=xsea,
                perm=perm
            )
            self._nulls["_colocs"][df_str] = _colocs_null
            self._set_last(
                method=method, 
                X_reduction=X_reduction, 
                Y_transform=Y_transform, 
                xsea=xsea,
                perm=perm
            )
            return p_data 
        else:
            return p_data, p_data_norm, _colocs_null
    
    
    # CORRECT ======================================================================================

    def correct_p(self, method=None, 
                  mc_alpha=0.05, mc_method="fdr_bh", mc_dimension="array", store=True, verbose=None):
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        lgr.info("*** NiSpace.correct_p() - Correct p values for multiple comparisons. ***")
        
        # get p data depending
        #p_value_dict = self._p_colocs
        
        # list of all p-value df keys, only uncorrected p-values
        p_strs = [k for k in self._p_colocs if "mc-none" in k]
        if method is not None:
            p_strs = [s for s in p_strs if f"method-{method}" in s]

        # get dimension of array to correct along
        if mc_dimension in ["x", "X", "c", "col", "cols", "column", "columns"]:
            how = "c"
        elif mc_dimension in ["y", "Y", "r", "row", "rows"]:
            how = "r"
        else:
            how = "a"
            
        # mc method
        mc_method = _get_correct_mc_method(mc_method)
        
        # get p values, mc function passes keywords to statsmodels.multitest
        p_corr = dict()
        for p_str in p_strs:
            p_str_mc = p_str.replace("mc-none", f"mc-{mc_method.replace('_', '').replace('-', '')}")
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
        return p_corr
    
    
    # PLOT ====================================================================================

    def plot(self, kind="categorical",
             method=None, stats=None, 
             X_reduction=None, Y_transform=None,
             xsea=None,
             Y_labels=None,
             plot_nulls=True, plot_p=True, permute_what=None,
             title="auto", sort_colocs=False,
             colocalizations_dict=None, nulls_dict=None, p_dict=None, pc_dict=None, mc_method="fdr_bh",
             fig=None, ax=None, figsize=None, show=True,
             plot_kwargs={}, nullplot_kwargs={},
             verbose=None): 
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        lgr.info("*** NiSpace.plot() - Plot colocalization results. ***")
        
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
        
        # check nulls/p plot
        if (plot_nulls or plot_p) and not permute_what:
            lgr.warning("if 'plot_nulls' or 'plot_p', provide 'permute_what' ({'groups', "
                        "'{x|y|xy}maps', 'sets'}). Setting 'plot_nulls' and 'plot_p' to False!")
            plot_nulls, plot_p = False, False
        
        # get arguments
        check_kwargs = dict(method=method, stats=stats, xdimred=X_reduction, 
                            ytrans=Y_transform, xsea=xsea)
        get_kwargs = dict(method=method, stats=stats, X_reduction=X_reduction, 
                          Y_transform=Y_transform, xsea=xsea)
        
        # get colocalization results
        if colocalizations_dict is None:
            self._check_colocalize(**check_kwargs)
            coloc_dicts = self.get_colocalizations(
                **get_kwargs, 
                force_dict=True,
                get_nulls=plot_nulls, 
                nulls_permute_what=permute_what,
                verbose=False
            )
            if isinstance(coloc_dicts, tuple):
                colocalizations_dict, nulls_dict = coloc_dicts
            else:
                colocalizations_dict, nulls_dict = coloc_dicts, None
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
        
        # restrict to given y labels
        if Y_labels is not None:
            if isinstance(Y_labels, str):
                Y_labels = [Y_labels]
            for stat in colocalizations_dict:
                indexer = colocalizations_dict[stat].index.isin(Y_labels)
                colocalizations_dict[stat] = colocalizations_dict[stat].loc[indexer]
                if nulls_dict is not None:
                    for null_str in nulls_dict[stat]:
                        nulls_dict[stat][null_str] = nulls_dict[stat][null_str].loc[indexer]
        
        # # get p values
        # if plot_p and p_dict is None:
        #     if self._check_permute(**check_kwargs, permuted=permute_what, raise_error=False):
        #         p_dict = self.get_p_values(**get_kwargs, permuted=permute_what, force_dict=True)
        #     else:
        #         lgr.error("No p values found. Provide via 'p_dict' or run NiSpace.permute()!")
                
        # # get corrected p values
        # if plot_p and pc_dict is None:
        #     if self._check_permute(**check_kwargs, permuted=permute_what, mc_method=mc_method, raise_error=False):
        #         pc_dict = self.get_p_values(**get_kwargs, permuted=permute_what, mc_method=mc_method, force_dict=True)
        #     else:
        #         lgr.error(f"No corrected p values for mc_method '{mc_method}' found. Provide via "
        #                   "'pc_dict' or run NiSpace.permute() & NiSpace.correct_p()!")
        
        # loop over stats
        stats = [s for s in colocalizations_dict if s not in ["intercept"]]
        out = {}
        for stat in stats:
            
            lgr.info(f"Creating {kind} plot for method {method}, colocalization stat {stat}.")
            if title == "auto":
                title = f"{nice_stats_labels(method)} colocalization"
                if Y_transform:
                    title += f" after {nice_stats_labels(Y_transform.replace('(a,b)', ''))} transform"
                if nulls_dict:
                    title += f"\n(permutation of {nice_stats_labels(permute_what)})"

            if kind == "categorical":
                fig_ax = _plot_categorical(
                    colocs_df=colocalizations_dict[stat],
                    stat=stat,
                    nulls_dict=nulls_dict,
                    p_df=p_dict[stat] if p_dict is not None else None,
                    pc_df=pc_dict[stat] if pc_dict is not None else None,
                    sort=sort_colocs,
                    fig=fig,
                    ax=ax,
                    title=title,
                    figsize=figsize, 
                    kwargs=plot_kwargs,
                    null_kwargs=nullplot_kwargs
                )
                    
            elif kind == "correlation":
                pass
            
            elif kind == "brain":
                pass
            
            elif kind == "nullhist":
                pass
            
            if show:
                plt.show()
            out[stat] = fig_ax
            
        if len(out) ==1:
            out = out[stat]        
        return out
        
        
    # GET ==========================================================================================
    
    def get_x(self, X_reduction=None, verbose=None):
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
                
        lgr.info(f"Returning X dataframe: \n{print_arg_pairs(X_reduction=X_reduction)}")
        lgr.setLevel(loglevel)
        return out      
    
    # ----------------------------------------------------------------------------------------------
    
    def get_y(self, Y_transform=None, verbose=None):
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
                
        lgr.info(f"Returning Y dataframe: \n{print_arg_pairs(Y_transform=Y_transform)}")
        lgr.setLevel(loglevel)
        return out      
    
    # ----------------------------------------------------------------------------------------------
         
    def get_z(self, verbose=None):
        loglevel = lgr.getEffectiveLevel()
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        
        out = self._Z
        if out is None:
            lgr.critical_raise("No Z dataframe found!",
                               ValueError)
            
        lgr.info("Returning Z dataframe.")
        lgr.setLevel(loglevel)
        return out  
    
    # ----------------------------------------------------------------------------------------------
   
    def get_colocalizations(self, method=None, stats=None, 
                            X_reduction=None, Y_transform=None, xsea=None,
                            get_nulls=False, nulls_permute_what=None, force_dict=False,
                            verbose=None): 
        loglevel = lgr.getEffectiveLevel()
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        
        method, X_reduction, Y_transform, xsea = self._get_last(
            method=method, 
            X_reduction=X_reduction, 
            Y_transform=Y_transform, 
            xsea=xsea
        )
        
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
            out[stat] = self._colocs[coloc_str].copy()
        
        if get_nulls and nulls_permute_what is None:
            lgr.error("If 'get_nulls' is True, 'nulls_permute_what' must not be None!")
            get_nulls = False
            
        if get_nulls:
            if nulls_permute_what not in ["groups", "groupsxmaps", "groupssets", 
                                          "xmaps", "ymaps", "xymaps", "ymapssets",
                                          "sets"]:
                lgr.critical_raise("If 'get_nulls' is True, 'nulls_permute_what' must be one of "
                                   "{'groups', '{x|y|xy}maps', 'sets'}!",
                                   ValueError)
            out_null = None
            null_str = _get_df_string(
                "null", 
                xdimred=X_reduction,
                ytrans=Y_transform,
                method=method, 
                xsea=xsea,
                perm=nulls_permute_what
            )
            if null_str not in self._nulls["_colocs"].keys():
                available = "\n".join(list(self._nulls["_colocs"].keys()))
                lgr.error(f"Null colocalizations for '{null_str}' not found! Available: {available}")
            else:
                nulls = self._nulls["_colocs"][null_str].copy()
                
                out_null = dict()
                n_nulls = len(nulls)
                idx = self.get_p_values(method, nulls_permute_what, _COLOC_METHODS[method][0], 
                                        xsea, norm=False,
                                        X_reduction=X_reduction,
                                        Y_transform=Y_transform,
                                        verbose=False).index
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
                     norm=False, mc_method=None, 
                     X_reduction=None, Y_transform=None, force_dict=False, verbose=None): 
        loglevel = lgr.getEffectiveLevel()
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        
        method, X_reduction, Y_transform, xsea, permute_what = self._get_last(
            method=method, 
            X_reduction=X_reduction, 
            Y_transform=Y_transform, 
            xsea=xsea,
            perm=permute_what
        )
        
        self._check_permute(method, permute_what, mc_method, xsea, stats, X_reduction, Y_transform)
        
        if stats is None:
            stats = _get_coloc_stats(method, permuted_only=True)
        elif isinstance(stats, str):
            stats = [stats]
        
        if mc_method is not None:
            mc_method = mc_method.replace("-", "").replace("_", "")
        
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
                norm=norm,
                mc=mc_method,
            )
            if p_str not in self._p_colocs.keys() and "coloc-mlr_stat-individual" not in p_str:
                available = "\n".join(list(self._p_colocs.keys()))
                lgr.critical_raise(f"Colocalization p values for '{p_str}' not found. "
                                   f"Available: {available}",
                                   KeyError)
            out[stat] = self._p_colocs[p_str]
                
        if not force_dict:
            if len(out)==1:
                out = out[list(out.keys())[0]]
        
        string = print_arg_pairs(method=method, permute_what=permute_what, xsea=xsea, 
                                 mc_method=mc_method, norm=norm, 
                                 X_reduction=X_reduction, Y_transform=Y_transform)
        lgr.info(f"Returning p values: \n{string}")
        lgr.setLevel(loglevel)
        return out
        
        
    # SAVE, LOAD, COPY =============================================================================

    def to_pickle(self, filepath, save_nulls=True, verbose=None):
        loglevel = lgr.getEffectiveLevel()
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        
        ext = os.path.splitext(filepath)[1]
        if ext==".gz":
            open_fun = gzip.open
        elif ext in [".pkl", ".pickle"]:
            open_fun = open
        else:
            lgr.critical_raise(f"Filetype *{ext} not known. Supported: '.pbz2', '.pickle', '.pkl'.",
                               ValueError)
        
        # remove nulls (very large depending on number of permutations) if requested
        self_save = self.copy()
        if save_nulls == False:
            self_save.nulls = dict()
        
        # save
        with open_fun(filepath, "wb") as f:
            pickle.dump(self_save, f, pickle.HIGHEST_PROTOCOL)
        lgr.debug(f"Saved NiSpace object to {filepath}.")  
        lgr.setLevel(loglevel)

    # ----------------------------------------------------------------------------------------------

    def copy(self, deep=True, verbose=True):
        set_log(lgr, verbose)
        
        if deep==True:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)        
            
    # ----------------------------------------------------------------------------------------------

    @staticmethod 
    def from_pickle(filepath, verbose=True):
        loglevel = lgr.getEffectiveLevel()
        verbose = set_log(lgr, verbose)
        
        ext = os.path.splitext(filepath)[1]
        if ext==".gz":
            open_fun = gzip.open
        elif ext in [".pkl", ".pickle"]:
            open_fun = open
        else:
            lgr.critical_raise(f"Filetype *{ext} not known. Supported: '.pbz2', '.pickle', '.pkl'.",
                               ValueError)
        
        # load   
        with open_fun(filepath, "rb") as f:
            juspyce_object = pickle.load(f)
        lgr.debug(f"Loaded NiSpace object from {filepath}.")

        # return
        lgr.setLevel(loglevel)
        return juspyce_object


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
                       stats=None, xdimred=False, ytrans=False, raise_error=True):
        if stats is None:
            stats = _get_coloc_stats(method, drop_optional=True, permuted_only=True)
        elif isinstance(stats, str):
            stats = [stats]
 
        for stat in stats:
            p_str = _get_df_string("p", xdimred=xdimred, ytrans=ytrans, method=method, stat=stat,
                                    perm=permute_what, mc=mc_method, xsea=xsea).lower()
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
        
    def _get_dist_mat(self, dist_mat_type, centroids=False, downsample_vol=3, 
                      n_proc=None, store=True, verbose=None):
        loglevel = lgr.getEffectiveLevel()
        verbose = set_log(lgr, self._verbose if verbose is None else verbose)
        
        if dist_mat_type not in ["cv", "null_maps"]:
            lgr.critical_raise(f"dist_mat_type = '{dist_mat_type}' not defined",
                               ValueError)
            
        if hasattr(self, "_parc_dist_mat"):
            if dist_mat_type in self._parc_dist_mat:
                dist_mat = self._parc_dist_mat[dist_mat_type]
                _generate_dist_mat = False
            else:
                _generate_dist_mat = True
        else:
            self._parc_dist_mat = {}
            _generate_dist_mat = True
            
        if _generate_dist_mat:
            dist_mat = get_distance_matrix(
                parc=self._parc, 
                parc_space=self._parc_info["space"],
                parc_hemi=self._parc_info["hemi"],
                #parc_density=self._parc_info["density"],
                downsample_vol=downsample_vol,
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
    