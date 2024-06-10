import numpy as np
import pandas as pd
from numba import njit
from itertools import permutations
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from scipy.stats import zscore, norm
from statsmodels.stats.multitest import multipletests


@njit(cache=True)
def np_any_axis1(x):
    """Numba compatible version of np.any(x, axis=1)."""
    out = np.zeros(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_or(out, x[:, i])
    return out 


@njit(cache=True, nogil=True)
def residuals(x, y, decenter=False):
    """Compute residuals for Regression of predictor(s) x on target y. 
    Requires numpy arrays with columns as predictors/target.

    Args:
        x (numpy.ndarray): shape (n_values, n_predictors)
        y (numpy.ndarray): shape (n_values, 1) or (n_values,)
        decenter (bool, optional): add mean of y to residuals before return

    Returns:
        numpy.ndarray: 1D array of residuals w or w/o added mean of y
    """
   
    X = np.column_stack((x, np.ones(x.shape[0], dtype=x.dtype)))
    beta = np.linalg.pinv((X.T).dot(X)).dot(X.T.dot(y))
    y_hat = np.dot(X, beta)
    
    if decenter:
        y_hat += y.mean()
        
    return y - y_hat


@njit(cache=True, nogil=True)
def residuals_nan(x, y, decenter=False):
    """Compute residuals for Regression of predictor(s) x on target y. 
    Requires numpy arrays with columns as predictors/target.

    Args:
        x (numpy.ndarray): shape (n_values, n_predictors)
        y (numpy.ndarray): shape (n_values, 1) or (n_values,)
        decenter (bool, optional): add mean of y to residuals before return

    Returns:
        numpy.ndarray: 1D array of residuals w or w/o added mean of y
    """
    nan_mask = np_any_axis1(np.isnan(np.column_stack((x, y))))
    x_ = x[~nan_mask]
    y_ = y[~nan_mask]
    
    X = np.column_stack((x_, np.ones(x_.shape[0], dtype=x_.dtype)))
    beta = np.linalg.pinv((X.T).dot(X)).dot(X.T.dot(y_))
    y_hat = np.dot(X, beta)
    
    if decenter:
        y_hat += y_.mean()
        
    resid = np.full(y.shape, np.nan, dtype=y.dtype)
    resid[~nan_mask] = y_ - y_hat
        
    return resid


def rho_to_z(array, replace_1=1 - np.finfo(float).eps):
    """Fisher's z-transformation of correlation coefficients."""
    array = np.array(array)
    array[np.isclose(array, 1)] = replace_1
    array_z = np.arctanh(array)
    return array_z


def zscore_df(df, along="cols", force_df=True):
    """Z-standardizes array and returns pandas dataframe.

    Args:
        df (pandas dataframe): input dataframe
        along (str, optional): Either "cols" or "rows". Defaults to "cols".

    Returns:
        pd.DataFrame or pd.Series: standardized dataframe/series
    """    
    
    if along=="cols":
        axis = 0
    elif along=="rows":
        axis = 1
    else:
        raise ValueError(f"Option along=={along} not defined!")
        
    arr_stand = zscore(df, axis=axis, nan_policy="omit")
    
    # DataFrame
    if isinstance(df, pd.DataFrame):
        df_stand = pd.DataFrame(
            data=arr_stand,
            columns=df.columns,
            index=df.index
        )
        
    # Series
    elif isinstance(df, pd.Series):
        if force_df:
            df_stand = pd.DataFrame(
                data=arr_stand,
                columns=[df.name],
                index=df.index
            )
        else:
            df_stand = pd.Series(
                data=arr_stand,
                index=df.index,
                name=df.name
            )
    
    # ndarray
    elif isinstance(df, np.ndarray):
        if force_df:
            df_stand = pd.DataFrame(
                data=arr_stand
            )
        else:
            df_stand = arr_stand
            
    # not defined
    else:
        raise TypeError(f"Input data type {type(df)} not defined!")
    
    return df_stand


def permute_groups(groups, strategy="proportional", paired=False, subjects=None, n_perm=1, 
                   n_proc=1, seed=None, verbose=False):
    
    groups = np.array(groups)
    n = len(groups)
    group_labels, group_sizes = np.unique(groups, return_counts=True)
    n_labels = len(group_labels)
    if verbose == "debug":
        print(f"{n} samples with labels: {group_labels} with sizes {group_sizes}")
    
    if paired:
        # get subjects
        if subjects is None:
            raise ValueError("Paired permutation requires argument 'subjects'!")
        subjects = np.array(subjects)
        unique_subjects = np.unique(subjects)
        n_subjects = len(unique_subjects)
        
        # check input
        if len(subjects) != len(groups):
            raise ValueError("Number of subjects must equal length of group/sessions vector!")
        if not (np.all(group_sizes == group_sizes[0]) and n_subjects == group_sizes[0]):
            raise ValueError("All sessions must have the same number of samples")
        
    groups_perm = []

    # unpaired: permute across the whole set
    if not paired:
        
        # proportional: permuted groups have the same size & proportion of subjects as the original groups
        if "prop" in strategy:
            
            group_idc = {label: np.where(groups==label)[0] for label in group_labels}
                
            def perm_fun(i):
                rng = np.random.default_rng(None if seed is None else seed + i)
                groups_tmp = np.full(groups.shape, np.nan)
                group_idc_tmp = group_idc.copy()
                for label, size in zip(group_labels, group_sizes):
                    if verbose=="debug" and i == 0:
                        print(f"Permuted group '{label}' of size {size}:")
                    
                    for l, s in zip(group_labels, group_sizes):
                        idc = group_idc_tmp[l]
                        fraction = s / n
                        n_sample = np.floor(size * fraction).astype(int)
                        if verbose=="debug" and i == 0:
                            print(f"{n_sample} samples ({n_sample / size * 100:.02f} %) from group '{l}'.")
                        idc_perm = rng.choice(idc, n_sample, replace=False)
                        group_idc_tmp[l] = idc[~np.isin(idc, idc_perm)]
                        groups_tmp[idc_perm] = label
                        
                if np.isnan(groups_tmp).any():
                    groups_tmp[np.isnan(groups_tmp)] = rng.permutation(group_labels)

                if np.isnan(groups_perm).any():
                    raise ValueError("Problem with group assignment")
                else:
                    return groups_tmp.astype(groups.dtype)  
        
        # shuffle: random permutation across the whole set
        elif "shuff" in strategy:
            def perm_fun(i):
                rng = np.random.default_rng(None if seed is None else seed + i)
                return rng.permutation(groups)
            
        # draw: random permutation across the whole set, with replacement
        elif "draw" in strategy:
            def perm_fun(i):
                rng = np.random.default_rng(None if seed is None else seed + i)
                return rng.choice(groups, len(groups), replace=True)
        
        # not defined
        else:
            raise ValueError(f"Unknown unpaired permutation strategy: {strategy}!")
        
    # paired: permute within each subject
    else:
        
        # proportional: permuted sessions are equally sized and consist of 50% of each original session
        # (given two groups, should also work with more)
        if "prop" in strategy:
            # get a df with indices of each subject in the original vector by session/"group label"
            subs_by_session_idc = pd.DataFrame(
                np.zeros((n_subjects, n_labels), dtype=int),
                index=unique_subjects,
                columns=group_labels,
                dtype=int
            )
            for ses in group_labels:
                subs_by_session_idc.loc[unique_subjects, ses] = np.where((groups==ses) & np.isin(subjects, unique_subjects))[0]
            
            session_combinations = list(permutations(group_labels))
            n_samples = np.floor(n_subjects / len(session_combinations)).astype(int)
            
            def perm_fun(i):
                rng = np.random.default_rng(None if seed is None else seed + i)
                sessions_perm = np.full(n, np.nan)
                subs_tmp = unique_subjects.copy()
                for session_labels_perm in session_combinations:
                    subs = rng.choice(subs_tmp, n_samples, replace=False)
                    subs_tmp = subs_tmp[~np.isin(subs_tmp, subs)]
                    sessions_perm[subs_by_session_idc.loc[subs, :]] = session_labels_perm

                if len(subs_tmp) > 0:
                    sessions_perm[subs_by_session_idc.loc[subs_tmp, :]] = rng.permutation(group_labels) 
            
                if np.isnan(sessions_perm).any():
                    raise ValueError("Problem with group assignment")
                else:
                    return sessions_perm.astype(groups.dtype)   
                    
        
        # shuffle: random permutation within subjects
        elif "shuff" in strategy:

            def perm_fun(i):
                rng = np.random.default_rng(None if seed is None else seed + i)
                sessions_perm = groups.copy()
                for sub in np.unique(subjects):
                    sessions_perm[subjects==sub] = rng.permutation(groups[subjects==sub])
                return sessions_perm
                
        # not defined
        else:
            raise ValueError(f"Unknown paired permutation strategy: {strategy}!")
        
    # Run
    groups_perm = Parallel(n_jobs=n_proc)(
        [delayed(perm_fun)(i) 
         for i in tqdm(range(n_perm), 
                       desc=f"Permuting groups ({n_proc} proc)", 
                       disable=not verbose)]
    )
    
    # return permuted group vector(s), as 1d array or list thereof, if n_perm > 1
    if len(groups_perm) == 1:
        groups_perm = groups_perm[0]
    return groups_perm  


def null_to_p(test_value, null_array, tail="two", fit_norm=False):
    """Return p-value for test value(s) against null array.
    
    Adopted from NiMARE v0.0.12: https://zenodo.org/record/6600700
    (NiMARE/nimare/stats.py)
    
    Parameters
    ----------
    test_value : 1D array_like
        Values for which to determine p-value.
    null_array : 1D array_like
        Null distribution against which test_value is compared.
    tail : {'two', 'upper', 'lower'}, optional
        Whether to compare value against null distribution in a two-sided
        ('two') or one-sided ('upper' or 'lower') manner.
        If 'upper', then higher values for the test_value are more significant.
        If 'lower', then lower values for the test_value are more significant.
        Default is 'two'.
    fit_norm : boolean
        Whether to fit a normal distribution to null_array data and compute p values from that.
        Might be useful if the relative order of multiple highly significant p values is of
        interest, but the number of null values cannot be increased sufficiently.
    
    Returns
    -------
    p_value : :obj:`float`
        P-value(s) associated with the test value when compared against the null
        distribution. Return type matches input type (i.e., a float if
        test_value is a single float, and an array if test_value is an array).
    
    Notes
    -----
    P-values are clipped based on the number of elements in the null array, if fit_norm=False.
    In this case, no p-values of 0 or 1 should be produced.
    """
    
    if tail not in {"two", "upper", "lower"}:
        raise ValueError('Argument "tail" must be one of ["two", "upper", "lower"]')

    return_first = isinstance(test_value, (float, int))
    test_value = np.atleast_1d(test_value)
    null_array = np.array(null_array)

    ## empirical p value
    if not fit_norm:
        def compute_p(t, null):
            null = np.sort(null)
            idx = np.searchsorted(null, t, side="left").astype(float)
            return 1 - idx / len(null)
    
    ## fit a normal distribution and get p value
    elif fit_norm:
        def compute_p(t, null):
            mu, sd = norm.fit(null)
            return 1 - norm.cdf(t, mu, sd)
            
    ## calculate p
    if tail == "two":
        p_l = compute_p(test_value, null_array)
        p_r = compute_p(test_value * -1, null_array * -1)
        p = 2 * np.minimum(p_l, p_r)
    elif tail == "lower":
        p = compute_p(test_value * -1, null_array * -1)
    elif tail == "upper":
        p = compute_p(test_value, null_array)

    # ensure p_value in the following range:
    # smallest_value <= p_value <= (1.0 - smallest_value) or 1 if fit_norm
    smallest_value = np.maximum(np.finfo(float).eps, 1.0 / len(null_array)) if not fit_norm else 0
    largest_value = 1 - smallest_value
    result = np.clip(p, a_min=smallest_value, a_max=largest_value)

    return result[0] if return_first else result


def mc_correction(p_array, alpha=0.05, method="fdr_bh", how="array", dtype=None):
    
    # prepare data
    p = np.array(p_array)
    p_shape = p.shape

    ## correct across whole input array -> flattern & reshape
    if how in ["a", "arr", "array"]:
        # flatten row-wise
        p_1d = p.flatten("C") 
        # get corrected p-values
        res = multipletests(p_1d, alpha=alpha, method=method)
        pcor_1d = res[1]
        reject_1d = res[0]
        # reshape to original form
        pcor = np.reshape(pcor_1d, p_shape, "C")
        reject = np.reshape(reject_1d, p_shape, "C")
    
    ## correct across each column/row
    else:
        pcor, reject = np.zeros_like(p, dtype=dtype), np.zeros_like(p, dtype=dtype)
        if how in ["c", "col", "cols", "column", "columns"]:
            for col in range(p.shape[1]):
                res = multipletests(p[:,col], alpha=alpha, method=method)
                pcor[:,col], reject[:,col] = res[1], res[0]
        elif how in ["r", "row", "rows"]:
            for row in range(p.shape[0]):
                res = multipletests(p[row,:], alpha=alpha, method=method)
                pcor[row,:], reject[row,:] = res[1], res[0]
        else:
            print(f"Input how='{how}' not defined!")
          
    ## return as input dtype
    if isinstance(p_array, pd.DataFrame):
        pcor = pd.DataFrame(
            pcor, 
            index=p_array.index, 
            columns=p_array.columns, 
            dtype=dtype)
        reject = pd.DataFrame(
            reject, 
            index=p_array.index, 
            columns=p_array.columns, 
            dtype=dtype)
    elif isinstance(p_array, pd.Series):
        pcor = pd.Series(
            pcor, 
            index=p_array.index, 
            name=p_array.name, 
            dtype=dtype)
        reject = pd.Series(
            reject, 
            index=p_array.index, 
            name=p_array.name, 
            dtype=dtype)     
    return pcor, reject

