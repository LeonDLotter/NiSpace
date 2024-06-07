import numpy as np
import pandas as pd

from .. import lgr
from ..utils import nan_detector
from ..stats.coloc import (corr, partialcorr, r2, mlr, dominance, pls, pcr,lasso, ridge, elasticnet)
from ..stats.misc import residuals_nan, rho_to_z
from ..modules.constants import _COLOC_METHODS, _COLOC_METHODS_DROPOPT, _COLOC_METHODS_PERM


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


def _get_colocalize_fun(method, regr_z=True,
                        xsea=False, xsea_method="mean",
                        r_to_z=True, r_equal_one="raise", adj_r2=True, mlr_individual=False, 
                        n_components=10,
                        parcel_mask_regularized=None, parcel_tr_te_splits=None, parcel_train_pct=None, 
                        coloc_method_kwargs=None,
                        seed=None, verbose=False, dtype=np.float32):
   
    ## case (partial) pearson / spearman
    if any(m in method for m in ["pearson", "spearman"]):
        
        # spearman vs pearson
        rank = True if "spearman" in method else False
        
        # case simple correlation
        if "partial" not in method:
            def _y_colocalize(X, y, z=None):  
                if regr_z and z is not None:
                    y = residuals_nan(z, y)
                parcel_mask_y = ~np.isnan(y)
                # iterate x (atlases/predictors)
                _colocs = np.zeros(X.shape[0], dtype=dtype)
                for i_x in range(X.shape[0]):
                    x = X[i_x, :]
                    parcel_mask = parcel_mask_y & ~np.isnan(x)
                    _colocs[i_x] = corr(
                        x=x[parcel_mask], # atlas
                        y=y[parcel_mask], # subject
                        rank=rank
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
        
        # case partial correlation
        else:
            def _y_colocalize(X, y, z):    
                parcel_mask_yz = ~nan_detector(y, z)
                # iterate x (atlases/predictors)
                _colocs = np.zeros(X.shape[0], dtype=dtype)
                for i_x in range(X.shape[0]):
                    x = X[i_x, :]
                    parcel_mask = parcel_mask_yz & ~np.isnan(x)
                    _colocs[i_x] = partialcorr(
                        x=x[parcel_mask], # atlas
                        y=y[parcel_mask], # subject
                        z=z[parcel_mask], # data to partial out
                        rank=rank
                    )
                if r_equal_one == "raise" and np.isclose(_colocs, 1).any():
                    raise ValueError(f"'{method}' colocalization equal to 1 detected! Are you "
                                     "correlating data with itself or do you have too few parcels?")
                else:
                    _colocs[np.isclose(_colocs, 1)] = r_equal_one
                if r_to_z:
                    _colocs = rho_to_z(_colocs)
                    
                return {"rho": _colocs}
            
    ## case slr
    elif method=="slr":
        
        def _y_colocalize(X, y, z=None):  
            if regr_z and z is not None:
                y = residuals_nan(z, y)
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
        
        def _y_colocalize(X, y, z=None):   
            if regr_z and z is not None:
                y = residuals_nan(z, y)
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
        
        def _y_colocalize(X, y, z=None):   
            if regr_z and z is not None:
                y = residuals_nan(z, y) 
            X_T = X.T 
            parcel_mask = ~nan_detector(X_T, y)
            
            _colocs = dominance(
                x=X_T[parcel_mask, :], # atlases
                y=y[parcel_mask], # subject   
                adj_r2=adj_r2,
                verbose=True if verbose=="debug" else False
            ) # dict with dom stats
            
            return _colocs
    
    ## case pls
    elif method == "pls":
        
        def _y_colocalize(X, y, z=None):
            if regr_z and z is not None:
                y = residuals_nan(z, y)
            X_T = X.T 
            parcel_mask = ~nan_detector(X_T, y)
            
            _colocs = pls(
                x=X_T[parcel_mask, :], # atlases
                y=y[parcel_mask], # subject    
                n_components=n_components,
                kwargs=coloc_method_kwargs
            )
            
            return _colocs
        
    ## case pcr
    elif method == "pcr":
        
        def _y_colocalize(X, y, z=None):
            if regr_z and z is not None:
                y = residuals_nan(z, y)
            X_T = X.T 
            parcel_mask = ~nan_detector(X_T, y)
            
            _colocs = pcr(
                x=X_T[parcel_mask, :], # atlases
                y=y[parcel_mask], # subject   
                adj_r2=adj_r2,
                n_components=n_components,
                kwargs=coloc_method_kwargs
            )
            
            return _colocs
    
    ## case regularized
    elif method in ["lasso", "ridge", "elasticnet"]:       
        # NOTE: will exclude nan's list-wise (<-> case-wise as all other methods)
        
        if method=="lasso":
            _pred_fun = lasso
        elif method=="ridge":
            _pred_fun = ridge
        elif method=="elasticnet":
            _pred_fun = elasticnet
            
        def _y_colocalize(X, y, z=None):
            if regr_z and z is not None:
                y = residuals_nan(z, y)
            X_T = X.T 
            
            _colocs = _pred_fun(
                x=X_T[parcel_mask_regularized, :], # atlases
                y=y[parcel_mask_regularized], # subject    
                cv=parcel_tr_te_splits, 
                seed=seed, 
                kwargs=coloc_method_kwargs
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
        if xsea_method == "mean":
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
        else:
            lgr.critical_raise(f"XSEA aggregation method '{xsea_method}' not defined!",
                               ValueError)
            
        def _y_colocalize_xsea(X_dict, y, z=None):
            # get coloc stats as a list of dicts, one dict per X set
            _colocs_xsea = []
            for set_X in X_dict.values():
                _colocs_xsea.append(_y_colocalize(set_X, y, z))
            # get aggregated metrics per set
            _colocs = {}
            for stat in _colocs_xsea[0].keys():
                _colocs[stat] = np.array([aggr(c[stat]) for c in _colocs_xsea], dtype=dtype)
            return _colocs
            
        return _y_colocalize_xsea


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
        # for i in range(pls_n_components):
        #     coloc_arrays[f"loadings_comp{i}"] = arr_1d.copy()
        
        for y, prediction in enumerate(y_colocs_list):
            coloc_arrays["r2"][y] = prediction["r2"]
            coloc_arrays["beta"][y, :] = prediction["beta"]
            # for i in range(pls_n_components):
            #     coloc_arrays[f"loadings_comp{i}"][y] = prediction["loadings"][:, i]
            
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
    
    