import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _welford_1d(arr):
    n = 0
    mean = 0.0
    M2 = 0.0
    for x in arr:
        if not np.isnan(x):
            n += 1
            delta = x - mean
            mean += delta / n
            M2 += delta * (x - mean)
    var = M2 / (n - 1) if n >= 2 else np.nan
    return n, mean, var

# ---------------------------------------------------
# Cohen's d for independent groups
# ---------------------------------------------------
def cohen(a, b):
    a = np.array(a)
    b = np.array(b)

    # Number of elements in each column
    na = a.shape[0]
    nb = b.shape[0]
    dof = na + nb - 2

    # Calculate the pooled standard deviation for each column
    pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1, axis=0) + (nb - 1) * np.var(b, ddof=1, axis=0)) / dof)

    # Calculate Cohen's d for each column
    d = (np.mean(a, axis=0) - np.mean(b, axis=0)) / pooled_std

    return d

def cohen_nan(a, b):
    a = np.array(a)
    b = np.array(b)

    # Number of elements in each column
    na = np.sum(~np.isnan(a), axis=0)
    nb = np.sum(~np.isnan(b), axis=0)
    dof = na + nb - 2

    # Calculate the pooled standard deviation for each column
    pooled_std = np.sqrt(((na - 1) * np.nanvar(a, ddof=1, axis=0) + (nb - 1) * np.nanvar(b, ddof=1, axis=0)) / dof)

    # Calculate Cohen's d for each column
    d = (np.nanmean(a, axis=0) - np.nanmean(b, axis=0)) / pooled_std

    return d

@njit(cache=True, nogil=True)
def cohen_nan_fast(a, b):
    n_cols = a.shape[1]
    d = np.empty(n_cols, dtype=np.float64)
    for j in range(n_cols):
        na, mean_a, var_a = _welford_1d(a[:, j])
        nb, mean_b, var_b = _welford_1d(b[:, j])
        dof = na + nb - 2
        if dof <= 0:
            d[j] = np.nan
        else:
            pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / dof)
            d[j] = (mean_a - mean_b) / pooled_std
    return d


# ---------------------------------------------------
# Cohen's d for dependent groups
# ---------------------------------------------------
def cohen_paired(a, b):
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        raise ValueError("Arrays 'a' and 'b' must have the same shape.")
    
    # Calculate the difference between pairs for each column
    diff = a - b

    # Calculate Cohen's d for each column
    d = np.mean(diff, axis=0) / np.std(diff, ddof=1, axis=0)

    return d

def cohen_paired_nan(a, b):
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        raise ValueError("Arrays 'a' and 'b' must have the same shape.")
    
    # Calculate the difference between pairs for each column
    diff = a - b

    # Calculate Cohen's d for each column
    d = np.nanmean(diff, axis=0) / np.nanstd(diff, ddof=1, axis=0)

    return d

@njit(cache=True, nogil=True)
def cohen_paired_nan_fast(a, b):
    n_rows, n_cols = a.shape
    d = np.empty(n_cols, dtype=np.float64)
    for j in range(n_cols):
        n = 0
        mean_d = 0.0
        M2 = 0.0
        for i in range(n_rows):
            ai, bi = a[i, j], b[i, j]
            if not np.isnan(ai) and not np.isnan(bi):
                diff = ai - bi
                n += 1
                delta = diff - mean_d
                mean_d += delta / n
                M2 += delta * (diff - mean_d)
        if n < 2:
            d[j] = np.nan
        else:
            d[j] = mean_d / np.sqrt(M2 / (n - 1))
    return d


# ---------------------------------------------------
# Hedges g 
# ---------------------------------------------------
def hedges(a, b):
    a = np.array(a)
    b = np.array(b)

    # Calculate Cohen's d for each column
    d = cohen(a, b)

    # Calculate the correction factor for Hedges' g for each column
    na = a.shape[0]
    nb = b.shape[0]
    dof = na + nb - 2
    correction = 1 - (3 / (4 * dof - 1))

    # Calculate Hedges' g for each column
    g = d * correction

    return g

def hedges_nan(a, b):
    a = np.array(a)
    b = np.array(b)

    # Calculate Cohen's d for each column
    d = cohen_nan(a, b)

    # Calculate the correction factor for Hedges' g for each column
    na = np.sum(~np.isnan(a), axis=0)
    nb = np.sum(~np.isnan(b), axis=0)
    dof = na + nb - 2
    correction = 1 - (3 / (4 * dof - 1))

    # Calculate Hedges' g for each column
    g = d * correction

    return g

@njit(cache=True, nogil=True)
def hedges_nan_fast(a, b):
    n_cols = a.shape[1]
    g = np.empty(n_cols, dtype=np.float64)
    for j in range(n_cols):
        na, mean_a, var_a = _welford_1d(a[:, j])
        nb, mean_b, var_b = _welford_1d(b[:, j])
        dof = na + nb - 2
        if dof <= 0:
            g[j] = np.nan
        else:
            pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / dof)
            d = (mean_a - mean_b) / pooled_std
            g[j] = d * (1.0 - 3.0 / (4.0 * dof - 1.0))
    return g


# def hedges_paired(a, b):
#     a = np.array(a)
#     b = np.array(b)
#     if a.shape != b.shape:
#         raise ValueError("Arrays 'a' and 'b' must have the same shape.")

#     # Calculate Cohen's d for each column
#     d = cohen_paired_nan(a, b)

#     # Calculate the correction factor for Hedges' g for each column
#     n = np.sum(~np.isnan(a), axis=0)
#     correction = 1 - (3 / (4 * n - 1))

#     # Calculate Hedges' g for each column
#     g = d * correction

#     return g


# ---------------------------------------------------
# Zscores 
# ---------------------------------------------------
def zscore(a, b=None):
    a = np.array(a)

    if b is not None:
        b = np.array(b)
        z = (a - np.mean(b, axis=0)) / np.std(b, ddof=1, axis=0)
    else:
        z = (a - np.mean(a, axis=0)) / np.std(a, ddof=1, axis=0)

    return z

def zscore_nan(a, b=None):
    a = np.array(a)

    if b is not None:
        b = np.array(b)
        z = (a - np.nanmean(b, axis=0)) / np.nanstd(b, ddof=1, axis=0)
    else:
        z = (a - np.nanmean(a, axis=0)) / np.nanstd(a, ddof=1, axis=0)

    return z

@njit(cache=True, nogil=True)
def _col_stats(arr):
    """Per-column Welford mean and std for a 2D array."""
    n_cols = arr.shape[1]
    means = np.empty(n_cols, dtype=np.float64)
    stds  = np.empty(n_cols, dtype=np.float64)
    for j in range(n_cols):
        _, mean, var = _welford_1d(arr[:, j])
        means[j] = mean
        stds[j]  = np.sqrt(var)   # nan propagates when n < 2
    return means, stds

def zscore_nan_fast(a, b=None):
    ref = a if b is None else b
    means, stds = _col_stats(ref)
    return (a - means) / stds    # numpy broadcast: row-major, SIMD-friendly


# ---------------------------------------------------------------------------
# Robust Zscores
# ---------------------------------------------------------------------------
def rzscore_nan(a, b=None):
    a = np.array(a)
    if b is not None:
        b = np.array(b)
        med = np.nanmedian(b, axis=0)
        mad = np.nanmedian(np.abs(b - med), axis=0)
    else:
        med = np.nanmedian(a, axis=0)
        mad = np.nanmedian(np.abs(a - med), axis=0)
    return (a - med) / (1.4826 * mad)

@njit(cache=True, nogil=True)
def _nanmedian_1d(arr):
    """NaN-safe median of a 1D array via sort."""
    n = 0
    for x in arr:
        if not np.isnan(x):
            n += 1
    if n == 0:
        return np.nan
    valid = np.empty(n, dtype=np.float64)
    k = 0
    for x in arr:
        if not np.isnan(x):
            valid[k] = x
            k += 1
    valid = np.sort(valid)
    mid = n // 2
    if n % 2 == 0:
        return (valid[mid - 1] + valid[mid]) / 2.0
    else:
        return valid[mid]

@njit(cache=True, nogil=True)
def _col_robust_stats(arr):
    """Per-column NaN-safe median and MAD for a 2D array."""
    n_cols = arr.shape[1]
    medians = np.empty(n_cols, dtype=np.float64)
    mads    = np.empty(n_cols, dtype=np.float64)
    for j in range(n_cols):
        col = arr[:, j]
        med = _nanmedian_1d(col)
        medians[j] = med
        mads[j] = _nanmedian_1d(np.abs(col - med))   # NaN propagates, skipped in median
    return medians, mads

def rzscore_nan_fast(a, b=None):
    ref = a if b is None else b
    medians, mads = _col_robust_stats(ref)
    return (a - medians) / (1.4826 * mads)    # numpy broadcast: row-major, SIMD-friendly


# ---------------------------------------------------
# Percent change 
# ---------------------------------------------------
def prc(a, b):
    a = np.array(a, dtype=float)  # Ensure input is a numpy array and convert to float for safe division
    b = np.array(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Arrays 'a' and 'b' must have the same shape.")

    # Calculate percentage change
    # Use np.where to avoid division by zero
    p = np.where(a != 0, (b - a) / a * 100, np.nan)

    return p

@njit(cache=True, nogil=True)
def prc_fast(a, b):
    n_rows, n_cols = a.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        for j in range(n_cols):
            ai = a[i, j]
            result[i, j] = np.nan if ai == 0.0 else (b[i, j] - ai) / ai * 100.0
    return result
