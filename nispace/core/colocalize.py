import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import logging
lgr = logging.getLogger(__name__)
from ..utils.utils import nan_detector
from ..stats.coloc import (pearson, mutualinfo, r2, mlr, dominance, fast_pls1, pcr, lasso, ridge, elasticnet, rank2d)
from ..stats.misc import rho_to_z, z_to_rho, residuals_nan
from .constants import _COLOC_METHODS, _COLOC_METHODS_DROPOPT, _COLOC_METHODS_PERM


def _rank_regress(arr, rank, regress, z=None, zy_matched=False, n_proc=1, verbose=True):
    
    if arr is None or (not rank and not regress):
        return arr
    
    def regress_z_fun(arr, z, zy_matched):
        z = z.astype(arr.dtype)
        if z.shape[0] == 1:
            return np.row_stack([residuals_nan(x=z[0], y=arr[i]) for i in range(arr.shape[0])])
        elif zy_matched:
            return np.row_stack([residuals_nan(x=z[i], y=arr[i]) for i in range(arr.shape[0])])
        else:
            return np.row_stack([residuals_nan(x=z.T, y=arr[i]) for i in range(arr.shape[0])])
    
    # case 1: arr is array, e.g. X or Y arrays
    if isinstance(arr, np.ndarray):
        arr_out = arr
        if rank:
            arr_out = rank2d(arr_out.T).T
        if regress:
            # NOTE: must residualize arr_out (possibly already ranked above), not the
            # original arr -- using `arr` here silently discarded the rank2d() result
            # whenever both rank and regress applied (e.g. partialspearman with Z),
            # making it compute exactly what partialpearson computes instead.
            arr_out = regress_z_fun(arr_out, z, zy_matched)
            
    # case 2: arr is list, e.g., X or Y null arrays
    elif isinstance(arr, list):
        
        # parallelize
        def par_fun(arr_i):
            
            # case 2.1: arr_i is array
            if isinstance(arr_i, np.ndarray):
                if rank:
                    arr_i = rank2d(arr_i.T).T
                if regress:
                    arr_i = regress_z_fun(arr_i, z, zy_matched)
                
            # case 2.2: arr_i is dict
            elif isinstance(arr_i, dict):
                if rank:
                    arr_i = {set_name: rank2d(set_arr.T).T for set_name, set_arr in arr_i.items()}
                if regress:
                    arr_i = {set_name: regress_z_fun(set_arr, z, zy_matched) for set_name, set_arr in arr_i.items()}
            
            # rest
            else:
                raise ValueError(f"Unsupported type: {type(arr_i)}")

            return arr_i
        
        # run in parallel
        arr_out = Parallel(n_jobs=n_proc)(
            delayed(par_fun)(arr_i) 
            for arr_i in tqdm(arr, desc=f"Processing null arrays ({n_proc} proc)", disable=not verbose)
        )
    
    # case 3: arr is dict, e.g., X or Y null arrays
    elif isinstance(arr, dict):
        arr_out = arr
        if rank:
            arr_out = {set_name: rank2d(set_arr.T).T for set_name, set_arr in arr_out.items()}
        if regress:
            # see case 1 note above -- must chain off arr_out, not the original arr
            arr_out = {set_name: regress_z_fun(set_arr, z, zy_matched) for set_name, set_arr in arr_out.items()}
            
    # rest
    else:
        raise ValueError(f"Unsupported type: {type(arr)}")
    
    return arr_out   


_DOMINANCE_MAX_PREDICTORS_DEFAULT = 20


def _check_predictor_count(method, n_predictors, n_obs,
                           dominance_max_predictors=_DOMINANCE_MAX_PREDICTORS_DEFAULT):
    """
    Warn/error about statistically-dangerous or computationally-infeasible predictor
    counts for the multivariate colocalization methods (X used as one joint predictor
    block, all rows fit together -- ``mlr``/``dominance``/``pcr``/``pls``). This is
    independent of any memory concern and independent of call size (n_perm etc.) -- it
    fires once against the observed data, since the underlying statistical/computational
    problem doesn't change across permutations, only the shuffled values do.

    - ``mlr``/``dominance`` (OLS-based): rank-deficient once ``n_predictors >= n_obs``
      -- ``adj_r2`` becomes mathematically degenerate (can hit zero, go negative, or be
      arbitrarily large; see :func:`~nispace.stats.coloc.mlr`'s own docstring). Warns,
      doesn't block -- the fit is still numerically computable, just uninterpretable.
    - ``dominance`` additionally fits ``2**n_predictors - 1`` models -- computationally
      infeasible past ``dominance_max_predictors`` (default 20, i.e. up to ~1M models;
      2**25+ is the clearly-infeasible extreme). Hard error above that cutoff.
    - ``pcr``: dense-eigendecomposes an ``(n_predictors, n_predictors)`` covariance
      matrix per Y-row per permutation -- O(p^2) memory / O(p^3) compute, independent of
      ``n_obs``. Warns above a fixed threshold (500).
    - ``pls``: lighter warning mirroring its own docstring's overfitting caveat near
      ``n_predictors ~ n_obs`` (component-based, not as fragile as OLS, but not immune).
    - ``lasso``/``ridge``/``elasticnet``: no guard -- regularized, designed for
      ``n_predictors >> n_obs``.
    """
    if method == "dominance":
        if n_predictors > dominance_max_predictors:
            lgr.critical_raise(
                f"colocalize(method='dominance'): {n_predictors} predictors requires "
                f"2**{n_predictors}-1 ({2 ** n_predictors - 1:,}) models -- "
                "computationally infeasible. Reduce predictor count (e.g. via "
                "X_reduction) or use 'mlr'/'pcr'/a regularized method instead. Override "
                "via dominance_max_predictors= if you have verified this is intentional "
                "(strongly discouraged above ~25).",
                ValueError,
            )
        if n_predictors >= n_obs:
            lgr.warning(
                f"colocalize(method='dominance'): {n_predictors} predictor(s) >= {n_obs} "
                "usable parcel(s) -- the underlying OLS subset fits are rank-deficient; "
                "R2/adj_r2 become mathematically degenerate. Consider X_reduction, a "
                "regularized method ('pls'/'lasso'/'ridge'/'elasticnet'), or reducing "
                "predictor count."
            )
    elif method == "mlr":
        if n_predictors >= n_obs:
            lgr.warning(
                f"colocalize(method='mlr'): {n_predictors} predictor(s) >= {n_obs} usable "
                "parcel(s) -- the OLS fit is rank-deficient; adj_r2 becomes "
                "mathematically degenerate (can hit zero, go negative, or be arbitrarily "
                "large). Consider X_reduction, a regularized method "
                "('pls'/'lasso'/'ridge'/'elasticnet'), or reducing predictor count."
            )
    elif method == "pcr":
        if n_predictors > 500:
            lgr.warning(
                f"colocalize(method='pcr'): {n_predictors} predictors -- PCA's O(p^2) "
                "memory / O(p^3) compute cost (p=n_predictors) may be substantial, "
                "especially since this repeats per Y-row per permutation. Consider "
                "X_reduction."
            )
    elif method == "pls":
        if n_predictors >= n_obs:
            lgr.warning(
                f"colocalize(method='pls'): {n_predictors} predictor(s) >= {n_obs} usable "
                "parcel(s) -- pls can overfit badly (both real and null draws saturate "
                "toward R2=1) once predictor count approaches/exceeds n_obs. Consider "
                "fewer n_components or X_reduction."
            )


def _get_coloc_stats(method, permuted_only=False, drop_optional=False):
    
    if method in _COLOC_METHODS:
        stats = _COLOC_METHODS[method].copy()
    else:
        lgr.critical_raise(f"Method {method} not defined!", 
                           ValueError)
        
    if permuted_only:
        stats = [stat for stat in stats if stat in _COLOC_METHODS_PERM[method]]
        
    if drop_optional:
        stats = [stat for stat in stats if stat in _COLOC_METHODS_DROPOPT[method]]
    
    return stats


def _get_colocalize_fun(method,
                        xsea=False, xsea_method="mean",
                        r_to_z=True, r_equal_one="raise", adj_r2=True, mlr_individual=False, 
                        parcel_mask_regularized=None, parcel_tr_te_splits=None, parcel_train_pct=None, 
                        n_components=2,
                        seed=None, verbose=False, dtype=np.float32, **kwargs):
   
    ## case pearson / spearman
    if method in ["pearson", "spearman", "partialpearson", "partialspearman"]:
        
        def _y_colocalize(X, y, weights=None):  
            parcel_mask_y = ~np.isnan(y)
            # iterate x (atlases/predictors)
            _colocs = np.zeros(X.shape[0], dtype=dtype)
            for i_x in range(X.shape[0]):
                x = X[i_x, :]
                parcel_mask = parcel_mask_y & ~np.isnan(x)
                _colocs[i_x] = pearson(
                    x=x[parcel_mask], # atlas
                    y=y[parcel_mask], # subject
                ) 
            if r_equal_one == "raise":
                if np.isclose(_colocs, 1).any():
                    raise ValueError(f"'{method}' colocalization equal to 1 detected! Are you "
                                        "correlating data with itself or do you have too few parcels?")
            else:
                _colocs[np.isclose(_colocs, 1)] = r_equal_one
            if r_to_z:
                _colocs = rho_to_z(_colocs)
                
            return {"rho": _colocs}
        
    ## case mi
    elif method == "mi":
        
        def _y_colocalize(X, y, weights=None):  
            parcel_mask_y = ~np.isnan(y)
            # iterate x (atlases/predictors)
            _colocs = np.zeros(X.shape[0], dtype=dtype)
            for i_x in range(X.shape[0]):
                x = X[i_x, :]
                parcel_mask = parcel_mask_y & ~np.isnan(x)
                _colocs[i_x] = mutualinfo(
                    x=x[parcel_mask], # atlas
                    y=y[parcel_mask], # subject
                    seed=seed,
                    **kwargs
                )
                
            return {"mi": _colocs}
            
    ## case slr
    elif method=="slr":
        
        def _y_colocalize(X, y, weights=None):  
            parcel_mask_y = ~np.isnan(y)
              
            # iterate x (atlases/predictors)
            _colocs = np.zeros(X.shape[0], dtype=dtype)
            for i_x in range(X.shape[0]):
                x = X[i_x, :]
                parcel_mask = parcel_mask_y & ~np.isnan(x)
                _colocs[i_x] = r2(
                    x=x[parcel_mask, np.newaxis], # atlas
                    y=y[parcel_mask], # subject
                    adj_r2=adj_r2
                )
                
            return {"r2": _colocs}
                
    ## case mlr
    elif method=="mlr":
        
        def _y_colocalize(X, y, weights=None):   
            X_T = X.T 
            parcel_mask = ~nan_detector(X_T, y)
            
            _colocs = dict()
            _colocs["r2"], params = mlr(
                x=X_T[parcel_mask, :], # atlases
                y=y[parcel_mask], # subject      
                adj_r2=adj_r2,
                intercept=True
            )
            _colocs["intercept"] = params[0]
            _colocs["beta"] = params[1:]
            if mlr_individual:  
                _colocs["individual"] = np.zeros(X.shape[0], dtype=dtype)
                for i_x in range(X.shape[0]):
                    _colocs["individual"][i_x] = r2(
                        x=np.delete(X_T[parcel_mask, :], i_x, axis=1), # atlases
                        y=y[parcel_mask], # subject
                        adj_r2=adj_r2
                    )      
                _colocs["individual"] = _colocs["r2"] - _colocs["individual"]
                
            return _colocs
                
    ## case dominance
    elif method=="dominance":
        
        def _y_colocalize(X, y, weights=None):   
            X_T = X.T 
            parcel_mask = ~nan_detector(X_T, y)
            
            _colocs = dominance(
                x=X_T[parcel_mask, :], # atlases
                y=y[parcel_mask], # subject   
                adj_r2=adj_r2,
            ) # dict with dom stats
            
            return _colocs
    
    ## case pls
    elif method == "pls":
        
        def _y_colocalize(X, y, weights=None):
            X_T = X.T 
            parcel_mask = ~nan_detector(X_T, y)
            
            _colocs = fast_pls1(
                x=X_T[parcel_mask, :], # atlases
                y=y[parcel_mask], # subject    
                n_components=n_components,
                #**kwargs
            )
            
            return _colocs
        
    ## case pcr
    elif method == "pcr":
        
        def _y_colocalize(X, y, weights=None):
            X_T = X.T 
            parcel_mask = ~nan_detector(X_T, y)
            
            _colocs = pcr(
                x=X_T[parcel_mask, :], # atlases
                y=y[parcel_mask], # subject
                adj_r2=adj_r2,
                n_components=n_components,
                #**kwargs
            )
            
            return _colocs
    
    ## case regularized
    elif method in ["lasso", "ridge", "elasticnet"]:       
        # NOTE: will exclude nan's list-wise (!= case-wise as all other methods)
        
        if method=="lasso":
            _pred_fun = lasso
        elif method=="ridge":
            _pred_fun = ridge
        elif method=="elasticnet":
            _pred_fun = elasticnet
            
        def _y_colocalize(X, y, weights=None):
            X_T = X.T 
            
            _colocs = _pred_fun(
                x=X_T[parcel_mask_regularized, :], # atlases
                y=y[parcel_mask_regularized], # subject    
                cv=parcel_tr_te_splits, 
                seed=seed, 
                **kwargs
            )
            
            return _colocs              
        
    ## case not defined
    else:
        lgr.critical_raise(f"Colocalization method '{method}' not defined!",
                           ValueError)
        
    # return colocalization function for one y vector and one X array
    if not xsea:
        return _y_colocalize
    
    # return colocalization function for one y vector and multiple X arrays with
    # results aggregated based on xsea_method
    else:
        # "mean"/"weightedmean" average per-gene values into a per-set score -- never
        # valid on raw (bounded, non-additive) correlation coefficients (see
        # rho_to_z's docstring / project memory "never average raw correlations").
        # Gated on method here (not per-call on `stat`) because for these 4 methods
        # every stat _y_colocalize ever produces is "rho" -- no other stat name is
        # possible, so there's no risk of misapplying this to an unrelated stat.
        # "median"/"absmedian" need no correction: order statistics commute exactly
        # with any monotonic transform (arctanh is one), so z-then-median-then-back
        # is mathematically identical to a raw median regardless. "absmean"/
        # "weightedabsmean" are left as-is, matching
        # get_within_region_correlations_omnibus()'s established precedent of only
        # correcting the signed mean ("rho"), not the abs-magnitude variants.
        _rho_needs_z_fix = (
            method in ("pearson", "spearman", "partialpearson", "partialspearman")
            and not r_to_z
        )
        if xsea_method == "mean":
            if _rho_needs_z_fix:
                def aggr(arr):
                    return z_to_rho(np.nanmean(rho_to_z(arr)))
            else:
                def aggr(arr):
                    return np.nanmean(arr)
        elif xsea_method == "median":
            def aggr(arr):
                return np.nanmedian(arr)
        elif xsea_method == "absmean":
            def aggr(arr):
                return np.nanmean(np.abs(arr))
        elif xsea_method == "absmedian":
            def aggr(arr):
                return np.nanmedian(np.abs(arr))
        elif xsea_method == "weightedmean":
            if _rho_needs_z_fix:
                def aggr(arr, weights):
                    arr_z = rho_to_z(arr)
                    out_z = np.ma.average(np.ma.array(arr_z, mask=np.isnan(arr_z)),
                                          weights=weights, axis=0)
                    return z_to_rho(out_z)
            else:
                def aggr(arr, weights):
                    return np.ma.average(np.ma.array(arr, mask=np.isnan(arr)), weights=weights, axis=0)
        elif xsea_method == "weightedabsmean":
            def aggr(arr, weights):
                return np.ma.average(np.ma.array(np.abs(arr), mask=np.isnan(arr)), weights=weights, axis=0)
        else:
            lgr.critical_raise(f"XSEA aggregation method '{xsea_method}' not defined!",
                               ValueError)
        
        if not "weighted" in xsea_method:
            def _y_colocalize_xsea(X_dict, y, weights=None):
                # get coloc stats as a list of dicts, one dict per X set
                _colocs_xsea = []
                for set_X in X_dict.values():
                    _colocs_xsea.append(_y_colocalize(set_X, y))
                # get aggregated metrics per set
                _colocs = {}
                for stat in _colocs_xsea[0].keys():
                    _colocs[stat] = np.array([aggr(c[stat]) for c in _colocs_xsea], dtype=dtype)
                return _colocs
        else:
            def _y_colocalize_xsea(X_dict, y, weights):
                # get coloc stats as a list of dicts, one dict per X set
                _colocs_xsea = []
                _weights_xsea = []
                for set_name, set_X in X_dict.items():
                    _colocs_xsea.append(_y_colocalize(set_X, y))
                    _weights_xsea.append(weights[set_name])
                # get aggregated metrics per set
                _colocs = {}
                for stat in _colocs_xsea[0].keys():
                    # nothing to aggregate (mostly r2 -> 1 value per set)
                    if _colocs_xsea[0][stat].ndim == 0:
                        _colocs[stat] = np.array([c[stat] for c in _colocs_xsea], dtype=dtype)
                    # weighted aggregation
                    else:
                        _colocs[stat] = np.array([aggr(c[stat], w) for c, w in zip(_colocs_xsea, _weights_xsea)], dtype=dtype)
                return _colocs
            
        return _y_colocalize_xsea


def _xsea_aggregate(arr, xsea_method, weights=None, axis=-1, rho_scale=False):
    """Reduce per-gene stat values to a per-set statistic along `axis`.

    Same formulas as the aggregator closures built inside `_get_colocalize_fun`'s xsea
    branch above, generalized to an arbitrary reduction axis so they can be applied to
    batched lookups (e.g. shape ``(n_Y, n_perm, set_size)``) in one call instead of once
    per set/permutation. Used by the XSEA null-precompute fast paths in
    `NiSpace.permute()` (api.py), which replace the per-permutation `_y_colocalize_xsea`
    loop with vectorized array lookups but must reduce with identical aggregation math.

    rho_scale : bool, default False
        Pass True when `arr` holds raw (non-Fisher-z) pearson/spearman-type
        correlation coefficients (i.e. the caller's method is one of the 4 rho-based
        methods and colocalize()'s r_to_z was False) -- "mean"/"weightedmean" then
        transform to the Fisher-z scale before averaging and back afterward (never a
        raw average of bounded correlation coefficients; see
        `_get_colocalize_fun`'s matching `aggr` closures and project memory "never
        average raw correlations"). No-op for "median"/"absmedian" (order statistics
        commute exactly with the monotonic Fisher-z transform, so the result is
        identical either way) and for "absmean"/"weightedabsmean" (left as raw
        magnitude averages, matching get_within_region_correlations_omnibus()'s
        established precedent of only correcting the signed mean).
    """
    if xsea_method == "mean":
        if rho_scale:
            return z_to_rho(np.nanmean(rho_to_z(arr), axis=axis))
        return np.nanmean(arr, axis=axis)
    elif xsea_method == "median":
        return np.nanmedian(arr, axis=axis)
    elif xsea_method == "absmean":
        return np.nanmean(np.abs(arr), axis=axis)
    elif xsea_method == "absmedian":
        return np.nanmedian(np.abs(arr), axis=axis)
    elif xsea_method == "weightedmean":
        arr_in = rho_to_z(arr) if rho_scale else arr
        out = np.ma.average(np.ma.array(arr_in, mask=np.isnan(arr_in)), weights=weights, axis=axis)
        out = np.ma.filled(out, np.nan)
        return z_to_rho(out) if rho_scale else out
    elif xsea_method == "weightedabsmean":
        out = np.ma.average(np.ma.array(np.abs(arr), mask=np.isnan(arr)), weights=weights, axis=axis)
        return np.ma.filled(out, np.nan)
    else:
        lgr.critical_raise(f"XSEA aggregation method '{xsea_method}' not defined!",
                           ValueError)


def _sort_colocs(method, y_colocs_list, n_X, n_Y, xsea=False,
                 return_df=True, labs_X=None, labs_Y=None, 
                 n_components=None,
                 dtype=np.float32):
    
    ## collect data in arrays
    coloc_arrays = dict()
    
    # empty arrays to copy
    arr_2d = np.zeros((n_Y, n_X), dtype=dtype)
    arr_1d = np.zeros((n_Y, 1), dtype=dtype) if not xsea else arr_2d.copy()
            
    # case MLR: dict with one array per stat
    if method=="mlr":
        
        coloc_arrays["beta"] = arr_2d.copy()
        coloc_arrays["intercept"] = arr_1d.copy() 
        coloc_arrays["r2"] = arr_1d.copy()            
            
        for y, prediction in enumerate(y_colocs_list):
            coloc_arrays["r2"][y] = prediction["r2"]
            coloc_arrays["intercept"][y] = prediction["intercept"]
            coloc_arrays["beta"][y, :] = prediction["beta"]
        
        if "individual" in y_colocs_list[0].keys():
            coloc_arrays["individual"] = arr_2d.copy()
            for y, prediction in enumerate(y_colocs_list):
                coloc_arrays["individual"][y] = prediction["individual"]
    
    # case dominance: dict with one array per dominance stat
    elif method=="dominance":
        
        coloc_arrays["total"] = arr_2d.copy()
        coloc_arrays["individual"] = arr_2d.copy()
        coloc_arrays["relative"] = arr_2d.copy()
        coloc_arrays["sum"] = arr_1d.copy()
        
        for y, prediction in enumerate(y_colocs_list):
            coloc_arrays["total"][y] = prediction["total"]
            coloc_arrays["individual"][y] = prediction["individual"]
            coloc_arrays["relative"][y] = prediction["relative"]
            coloc_arrays["sum"][y] = prediction["sum"]
    
    elif method == "pls":
        coloc_arrays["r2"] = arr_1d.copy()
        coloc_arrays["beta"] = arr_2d.copy()
        coloc_arrays["score_r"] = arr_1d.copy()
        coloc_arrays["weight"] = arr_2d.copy()
        # for i in range(pls_n_components):
        #     coloc_arrays[f"loadings_comp{i}"] = arr_1d.copy()

        for y, prediction in enumerate(y_colocs_list):
            coloc_arrays["r2"][y] = prediction["r2"]
            coloc_arrays["beta"][y, :] = prediction["beta"]
            coloc_arrays["score_r"][y] = prediction["score_r"]
            coloc_arrays["weight"][y, :] = prediction["weight"]
            # for i in range(pls_n_components):
            #     coloc_arrays[f"loadings_comp{i}"][y] = prediction["loadings"][:, i]

    elif method == "pcr":
        coloc_arrays["r2"] = arr_1d.copy()
        coloc_arrays["score_r"] = arr_1d.copy()

        for y, prediction in enumerate(y_colocs_list):
            coloc_arrays["r2"][y] = prediction["r2"]
            coloc_arrays["score_r"][y] = prediction["score_r"]

    # case regularized regression
    elif method in ["lasso", "ridge", "elasticnet"]:
        
        coloc_arrays["beta"] = arr_2d.copy()
        coloc_arrays["r2"] = arr_1d.copy()
        coloc_arrays["alpha"] = arr_1d.copy()
       
        for y, prediction in enumerate(y_colocs_list):
            coloc_arrays["beta"][y, :] = prediction["beta"]
            coloc_arrays["r2"][y] = prediction["r2"]
            coloc_arrays["alpha"][y] = prediction["alpha"]
            
        if method=="elasticnet":
            coloc_arrays["l1ratio"] = arr_1d.copy()
            for y, prediction in enumerate(y_colocs_list):
                coloc_arrays["l1ratio"][y] = prediction["l1ratio"]
    
    # case all others -> correlations and slr
    else:
        stat = list(y_colocs_list[0].keys())[0]
        coloc_arrays[stat] = arr_2d.copy()
        for y, prediction in enumerate(y_colocs_list):
            coloc_arrays[stat][y] = prediction[stat]
    
    ## to dataframe & return
    if return_df:
        coloc_dfs = dict()
        
        for stat, arr in coloc_arrays.items():
            if arr.shape[1] == 1:
                columns = [stat]
            # elif stat=="loadings":
            #     columns = [f"comp_{i}" for i in range(arr.shape[1])]
            else:
                columns = labs_X
                
            coloc_dfs[stat] = pd.DataFrame(
                data=arr, 
                columns=columns,
                index=labs_Y,
                dtype=dtype
            ) 

        return coloc_dfs
    
    else:
        return coloc_arrays
    
    