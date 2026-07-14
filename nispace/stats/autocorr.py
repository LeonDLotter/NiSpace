import numpy as np
from scipy.spatial.distance import squareform
from nispace.stats.coloc import corr
from numba import njit


def morans_i(data, distmat, normalize=False, local=False, invert_dist=True, nan_policy="drop"):
    """
    Calculates Moran's I from distance matrix `distmat` and brain map `data`
    Adopted from https://github.com/netneurolab/markello_spatialnulls/blob/master/parspin/parspin/spatial.py

    Parameters
    ----------
    data : (N,) array_like
        Brain map vector of interest
    distmat : (N, N) array_like
        Distance matrix between `N` regions / vertices / voxels / whatever
    normalize : bool, optional
        Whether to normalize rows of distance matrix prior to calculation.
        Default: False
    local : bool, optional
        Whether to calculate local Moran's I instead of global. Default: False
    invert_dist : bool, optional
        Whether to invert the distance matrix to generate a weight matrix.
        Default: True

    Returns
    -------
    i : float
        Moran's I, measure of spatial autocorrelation
    """
    # ensure data and distmat are numpy arrays
    data = np.array(data).squeeze()
    distmat = np.array(distmat)
    
    # drop nan
    if nan_policy == "drop":
        notnan = ~np.isnan(data)
        data = data[notnan]
        distmat = distmat[np.ix_(notnan, notnan)]
    elif nan_policy == "raise":
        raise ValueError("NaN values in data")
    elif nan_policy == "propagate":
        pass
    else:
        raise ValueError("Invalid nan_policy")

    # convert distance matrix to weights
    if invert_dist:
        with np.errstate(divide='ignore'):
            distmat = 1 / distmat
    np.fill_diagonal(distmat, 0)

    # normalize rows, if desired
    if normalize:
        distmat /= distmat.sum(axis=-1, keepdims=True)

    # calculate Moran's I
    z = data - data.mean()
    if local:
        with np.errstate(all='ignore'):
            # ddof=0: local Moran's I / LISA is defined with population std,
            # not a sample-estimate choice
            z /= data.std(ddof=0)

    zl = np.squeeze(distmat @ z[:, None])
    den = (z * z).sum()

    if local:
        return (len(data) - 1) * z * zl / den

    return len(data) / distmat.sum() * (z * zl).sum() / den


def morans_i_fast(weightmat, data):
    """Fast global Moran's I calculation from weight matrix and data vector. 
    No checks. NaNs will be dropped. No normalization of weight matrix. """
    
    np.fill_diagonal(weightmat, 0)
    data_nan = np.isnan(data)
    data = data[~data_nan]
    weightmat = weightmat[np.ix_(~data_nan, ~data_nan)]
    z = data - data.mean()
    zl = np.squeeze(weightmat @ z[:, None])
    den = (z * z).sum()
    return len(data) / weightmat.sum() * (z * zl).sum() / den

# def morans_i(data, dist_mat):
#     data = np.array(data)
#     dist_mat = np.array(dist_mat)

#     # Inverse distance weights
#     with np.errstate(divide='ignore'):
#         weight_mat = 1.0 / dist_mat
#     np.fill_diagonal(weight_mat, 0.0)  # No self-weighting
    
#     # Drop nan
#     data_nan = np.isnan(data)
#     data = data[~data_nan]
#     weight_mat = weight_mat[np.ix_(~data_nan, ~data_nan)]
    
#     # Calculate Moran's I
#     n = len(data)
#     data_mean = np.mean(data)
#     data_diff = data - data_mean
#     num = np.sum(weight_mat * np.outer(data_diff, data_diff))
#     denom = np.sum(data_diff ** 2)
#     morans_i = (n / np.sum(weight_mat)) * (num / denom)
    
#     return morans_i


def variogram_sa(data, distmat, n_bins=25, return_variogram=False):
    """Spatial autocorrelation scalar consistent with Burt2018/Burt2020 (brainsmash).

    Both variogram-based null methods implicitly define SA via the empirical
    semivariogram: γ(d) = ½·E[(xᵢ−xⱼ)² | dist(i,j)≈d].  High SA means γ rises
    slowly from zero (nearby parcels nearly identical) toward the sill var(x).

    The scalar returned is 1 − mean(γ_norm) where γ_norm = γ(d) / var(x),
    averaged uniformly across distance bins.  Range [0, 1]; 1 = perfectly
    autocorrelated, 0 = no autocorrelation.  Bins are percentile-based (equal
    number of pairs per bin), matching brainsmash's binning convention.

    Parameters
    ----------
    data : (N,) array_like
    distmat : (N, N) array_like
    n_bins : int
        Number of distance bins.
    return_variogram : bool
        If True, also return (bin_centers, gamma_norm).

    Returns
    -------
    sa : float
    bin_centers : (n_bins,) ndarray  [only if return_variogram=True]
    gamma_norm  : (n_bins,) ndarray  [only if return_variogram=True]
    """
    data = np.asarray(data, dtype=float).ravel()
    distmat = np.asarray(distmat, dtype=float)

    # drop NaN
    notnan = ~np.isnan(data)
    data = data[notnan]
    distmat = distmat[np.ix_(notnan, notnan)]

    # upper triangle (excluding diagonal)
    i_idx, j_idx = np.triu_indices(len(data), k=1)
    d = distmat[i_idx, j_idx]
    sq_diff = (data[i_idx] - data[j_idx]) ** 2

    # remove pairs where distance is 0 or NaN
    valid = (d > 0) & np.isfinite(d)
    d, sq_diff = d[valid], sq_diff[valid]

    # percentile-based bin edges (equal pairs per bin, same as brainsmash)
    edges = np.percentile(d, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)

    gamma, centers = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (d >= lo) & (d < hi)
        if sel.sum() > 1:
            gamma.append(sq_diff[sel].mean() / 2)   # semivariogram value
            centers.append((lo + hi) / 2)

    gamma = np.array(gamma)
    centers = np.array(centers)

    sill = np.var(data, ddof=1)
    gamma_norm = gamma / sill if sill > 0 else np.ones_like(gamma)

    sa = float(1.0 - gamma_norm.mean())

    if return_variogram:
        return sa, centers, gamma_norm
    return sa


def mantel(data, distmat, spearman=False):
    data = np.array(data).squeeze()
    distmat = np.array(distmat)
    if not data.shape[0] == distmat.shape[0] == distmat.shape[1]:
        raise ValueError("Shapes!")
    
    diffmat = np.abs(data[:, None] - data[None, :])
    diffmat_flat = squareform(diffmat, checks=False)
    if not np.allclose(distmat, distmat.T):
        raise ValueError("Distance matrix not symmetric")
    distmat_flat = squareform(distmat)
    
    notnan = ~( np.isnan(diffmat_flat) | np.isnan(distmat_flat) )
    r = corr(diffmat_flat[notnan], distmat_flat[notnan], rank=spearman)
    return r 

@njit   
def mantel_fast(data, flat_distmat, spearman=False):
    
    diffmat = np.abs(data[:, None] - data[None, :])
    diffmat_flat = numba_squareform(diffmat)
    notnan = ~( np.isnan(diffmat_flat) | np.isnan(flat_distmat) )
    r = corr(diffmat_flat[notnan], flat_distmat[notnan], rank=spearman)
    return r

@njit
def numba_squareform(square_mat):
    """Convert square distance matrix to condensed vector form using Numba."""
    n = square_mat.shape[0]
    # Length of output vector is n*(n-1)/2
    vector_length = (n * (n - 1)) // 2
    vector = np.zeros(vector_length)
    
    # Fill the vector using a single loop
    k = 0
    for i in range(n-1):
        for j in range(i+1, n):
            vector[k] = square_mat[i, j]
            k += 1
    return vector
