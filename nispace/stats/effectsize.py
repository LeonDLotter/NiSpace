import numpy as np
from numba import njit


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


def hedges_paired(a, b):
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        raise ValueError("Arrays 'a' and 'b' must have the same shape.")

    # Calculate Cohen's d for each column
    d = cohen_paired_nan(a, b)

    # Calculate the correction factor for Hedges' g for each column
    n = np.sum(~np.isnan(a), axis=0)
    correction = 1 - (3 / (4 * n - 1))

    # Calculate Hedges' g for each column
    g = d * correction

    return g


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


def prc(a, b):
    a = np.array(a, dtype=float)  # Ensure input is a numpy array and convert to float for safe division
    b = np.array(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Arrays 'a' and 'b' must have the same shape.")

    # Calculate percentage change
    # Use np.where to avoid division by zero
    p = np.where(a != 0, (b - a) / a * 100, np.nan)

    return p
