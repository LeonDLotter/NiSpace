import logging
import numpy as np
import pandas as pd
from numba import njit
from itertools import permutations
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from scipy.stats import zscore, norm
from statsmodels.stats.multitest import multipletests

lgr = logging.getLogger(__name__)


@njit(cache=True)
def np_any_axis1(x):
    """Numba compatible version of np.any(x, axis=1)."""
    out = np.zeros(x.shape[0], dtype=np.bool_)
    for i in range(x.shape[1]):
        out = np.logical_or(out, x[:, i])
    return out 


@njit(cache=True, nogil=True)
def residuals(x, y, decenter=False):
    """Compute residuals for Regression with dependent variable y and independent variable(s) x.
    Requires numpy arrays with columns as independent variables.

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
    """Compute residuals for Regression with dependent variable y and independent variable(s) x. 
    Requires numpy arrays with columns as independent variables.

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


@njit(cache=True, nogil=True)
def partial_residuals_nan(x_nuisance, x_protect, y):
    """Partial residuals: regress x_nuisance from y while controlling for x_protect.

    Fits a joint model [x_nuisance | x_protect | intercept] so that nuisance
    coefficients are estimated controlling for the protected variables, then
    removes only the nuisance component. The protected effects (e.g. group
    differences) are preserved in the returned values.

    Args:
        x_nuisance (numpy.ndarray): shape (n, p) — confounds to remove
        x_protect  (numpy.ndarray): shape (n, q) — variables to control for but keep
        y          (numpy.ndarray): shape (n,)

    Returns:
        numpy.ndarray: shape (n,), NaN where input had NaN
    """
    nan_mask = np_any_axis1(np.isnan(np.column_stack((x_nuisance, x_protect, y))))
    xn = x_nuisance[~nan_mask]
    xp = x_protect[~nan_mask]
    y_ = y[~nan_mask]

    X_full = np.column_stack((xn, xp, np.ones(xn.shape[0], dtype=xn.dtype)))
    n_nuisance = x_nuisance.shape[1]
    beta = np.linalg.pinv((X_full.T).dot(X_full)).dot(X_full.T.dot(y_))

    resid = np.full(y.shape, np.nan, dtype=y.dtype)
    resid[~nan_mask] = y_ - xn.dot(beta[:n_nuisance])
    return resid


def rho_to_z(array, replace_1=1 - np.finfo(float).eps):
    """Fisher's z-transformation of correlation coefficients."""
    array = np.array(array)
    array[np.isclose(array, 1)] = replace_1
    array_z = np.arctanh(array)
    return array_z


def z_to_rho(array):
    """Inverse Fisher's z-transformation of correlation coefficients."""
    array = np.array(array)
    array_rho = np.tanh(array)
    return array_rho


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
    lgr.debug(f"{n} samples with labels: {group_labels} with sizes {group_sizes}")
    
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

    return_first = isinstance(test_value, (float, int, np.floating, np.integer))
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
    null_array_mean = np.mean(null_array)
    test_value_centered = test_value - null_array_mean
    null_array_centered = null_array - null_array_mean
    if tail == "two":
        p_l = compute_p(test_value_centered, null_array_centered)
        p_r = compute_p(-test_value_centered, -null_array_centered)
        p = 2 * np.minimum(p_l, p_r)
    elif tail == "lower":
        p = compute_p(-test_value_centered, -null_array_centered)
    elif tail == "upper":
        p = compute_p(test_value_centered, null_array_centered)

    # ensure p_value in the following range:
    # smallest_value <= p_value <= (1.0 - smallest_value) or 1 if fit_norm
    smallest_value = np.maximum(np.finfo(float).eps, 1.0 / len(null_array)) if not fit_norm else 0
    largest_value = 1 - smallest_value
    result = np.clip(p, a_min=smallest_value, a_max=largest_value)

    return result[0] if return_first else result


def compute_meff(X, method="galwey"):
    """Effective number of independent tests from data matrix via eigendecomposition.

    Parameters
    ----------
    X : array-like, shape (n_maps, n_features)
    method : "galwey" (default) or "li_ji"
        Galwey (2009) Genet Epidemiol 33:559; Li & Ji (2005) Ann Hum Genet 69:519.

    Returns
    -------
    float
    """
    X = np.atleast_2d(np.array(X, dtype=float))
    # drop parcel columns that contain any NaN (NaN propagates through corrcoef → eigvalsh fails)
    X = X[:, ~np.isnan(X).any(axis=0)]
    if X.shape[1] < 2:
        raise ValueError(f"compute_meff: fewer than 2 non-NaN parcels after dropping NaNs "
                         f"(got {X.shape[1]}).")
    # single map: nothing to decorrelate, no effective reduction
    if X.shape[0] < 2:
        return 1.0
    corr = np.corrcoef(X)
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = eigenvalues[eigenvalues > 0]
    if method == "galwey":
        meff = np.sum(np.sqrt(eigenvalues)) ** 2 / np.sum(eigenvalues)
    elif method == "li_ji":
        meff = np.sum((eigenvalues >= 1).astype(float) + (eigenvalues - np.floor(eigenvalues)))
    else:
        raise ValueError(f"method must be 'galwey' or 'li_ji', not '{method}'")
    return float(meff)


def meff_sidak_correction(p_array, meff, alpha=0.05, dtype=None):
    """Sidak correction using Meff effective number of tests.

    p_corr = 1 - (1 - p)^meff
    """
    p = np.array(p_array, dtype=float)
    p_corr = np.clip(1.0 - (1.0 - p) ** meff, 0.0, 1.0)
    reject = p_corr <= alpha
    if isinstance(p_array, pd.DataFrame):
        p_corr = pd.DataFrame(p_corr, index=p_array.index, columns=p_array.columns, dtype=dtype)
        reject = pd.DataFrame(reject, index=p_array.index, columns=p_array.columns, dtype=dtype)
    elif isinstance(p_array, pd.Series):
        p_corr = pd.Series(p_corr, index=p_array.index, name=p_array.name, dtype=dtype)
        reject = pd.Series(reject, index=p_array.index, name=p_array.name, dtype=dtype)
    return p_corr, reject


def _null_stats_to_array(null_colocs, stat):
    """Stack per-permutation null statistic dicts into (n_perm, n_Y, n_X) array."""
    return np.stack([null_colocs[i][stat] for i in range(len(null_colocs))], axis=0)


def _signed_stats(obs, null_arr, tail):
    """Apply tail direction: return (obs_cmp, null_cmp) ready for >= comparison."""
    if tail == "two":
        return np.abs(obs), np.abs(null_arr)
    elif tail == "upper":
        return obs, null_arr
    elif tail == "lower":
        return -obs, -null_arr
    raise ValueError(f"tail must be 'two', 'upper', or 'lower', not '{tail}'")


def _df_like(array, template, dtype):
    """Wrap array in DataFrame/Series matching template's index/columns."""
    if isinstance(template, pd.DataFrame):
        return pd.DataFrame(array, index=template.index, columns=template.columns, dtype=dtype)
    elif isinstance(template, pd.Series):
        return pd.Series(array, index=template.index, name=template.name, dtype=dtype)
    return array


def maxT_correction(obs_stats, null_colocs, stat, tail="two", how="r", alpha=0.05, dtype=None):
    """Max-T FWER correction (Westfall & Young 1993).

    Parameters
    ----------
    obs_stats : DataFrame or array, shape (n_Y, n_X)
    null_colocs : list[n_perm] of {stat: (n_Y, n_X) array}
    stat : key to extract from null_colocs dicts
    tail : "two" | "upper" | "lower"
    how : "r" — per-Y row, max across X (default); "a" — global max across Y×X
    """
    obs = np.array(obs_stats, dtype=float)
    null_arr = _null_stats_to_array(null_colocs, stat).astype(float)  # (n_perm, n_Y, n_X)
    obs_cmp, null_cmp = _signed_stats(obs, null_arr, tail)

    if how == "r":
        null_max = null_cmp.max(axis=2)  # (n_perm, n_Y)
        # broadcast: null_max[:, y] >= obs_cmp[y, x] for each x
        p = np.mean(null_max[:, :, np.newaxis] >= obs_cmp[np.newaxis, :, :], axis=0)
    elif how == "a":
        null_max = null_cmp.max(axis=(1, 2))  # (n_perm,)
        p = np.mean(null_max[:, np.newaxis, np.newaxis] >= obs_cmp[np.newaxis, :, :], axis=0)
    else:
        raise ValueError(f"how='{how}' not supported for maxT. Use 'r' or 'a'.")

    p = np.clip(p, 0.0, 1.0)
    reject = p <= alpha
    return _df_like(p, obs_stats, dtype), _df_like(reject, obs_stats, dtype)


def step_maxT_correction(obs_stats, null_colocs, stat, tail="two", how="r", alpha=0.05, dtype=None):
    """Step-down Max-T FWER correction (Westfall & Young 1993).

    Enforces monotonicity on sorted max-T p-values for increased power over plain maxT.
    """
    obs = np.array(obs_stats, dtype=float)
    n_Y, n_X = obs.shape
    null_arr = _null_stats_to_array(null_colocs, stat).astype(float)  # (n_perm, n_Y, n_X)
    obs_cmp, null_cmp = _signed_stats(obs, null_arr, tail)

    def _step_down_1d(obs_1d, null_2d):
        # obs_1d: (n,); null_2d: (n_perm, n)
        order = np.argsort(obs_1d)[::-1]
        obs_s = obs_1d[order]
        null_s = null_2d[:, order]  # (n_perm, n)
        # step-down max: null_max_j[perm] = max(null_s[perm, j:])
        # vectorised via reverse cumulative max
        null_rev_cummax = np.maximum.accumulate(null_s[:, ::-1], axis=1)[:, ::-1]  # (n_perm, n)
        p_s = np.mean(null_rev_cummax >= obs_s[np.newaxis, :], axis=0)
        # enforce monotonicity (step-down)
        p_s = np.maximum.accumulate(p_s)
        # restore original order
        p_out = np.empty_like(p_s)
        p_out[order] = p_s
        return p_out

    if how == "r":
        p = np.zeros_like(obs)
        for y in range(n_Y):
            p[y, :] = _step_down_1d(obs_cmp[y, :], null_cmp[:, y, :])
    elif how == "a":
        obs_flat = obs_cmp.flatten()
        null_flat = null_cmp.reshape(len(null_colocs), -1)
        p_flat = _step_down_1d(obs_flat, null_flat)
        p = p_flat.reshape(n_Y, n_X)
    else:
        raise ValueError(f"how='{how}' not supported for step_maxT. Use 'r' or 'a'.")

    p = np.clip(p, 0.0, 1.0)
    reject = p <= alpha
    return _df_like(p, obs_stats, dtype), _df_like(reject, obs_stats, dtype)


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

