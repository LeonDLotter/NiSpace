from itertools import combinations
import numpy as np
from numba import njit
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from tqdm.auto import tqdm

from .. import lgr
from ..utils.utils import _del_from_tuple

# for backwards compatibility
@njit(nogil=True)
def rank_array(array):
    return rank1d(array)

@njit(cache=True, nogil=True)
def rank1d(arr):
    """Rank an array. CAVE: Cannot really deal with nan's!"""
    
    _args = arr.argsort()
    ranked = np.empty_like(arr)
    ranked[_args] = np.arange(arr.size)
    
    return ranked

@njit(cache=True, nogil=True)
def rank2d(arr):
    """Rank an array. Handles nan's"""
    
    if arr.ndim == 1:
        return rank1d(arr)
    
    ranked = np.full_like(arr, np.nan)
    for i in range(arr.shape[1]):
        v = arr[:, i]
        nonan = ~np.isnan(v)
        ranked[nonan, i] = rank1d(v[nonan])
    
    return ranked


@njit(cache=True, nogil=True)
def corr(x, y, rank=False):
    """Compute Pearson or Spearman correlation for two 1D arrays."""
    
    if rank:
        x = rank1d(x)
        y = rank1d(y)
    
    m_x = x.mean()
    m_y = y.mean()
    num = np.sum((x - m_x) * (y - m_y))
    den = np.sqrt(np.sum((x - m_x) ** 2) * np.sum((y - m_y) ** 2))
    r = num / den
    
    return r


@njit(cache=True, nogil=True)
def pearson(x, y):
    """Compute Pearson correlation for two 1D arrays."""
    
    m_x = x.mean()
    m_y = y.mean()
    num = np.sum((x - m_x) * (y - m_y))
    den = np.sqrt(np.sum((x - m_x) ** 2) * np.sum((y - m_y) ** 2))
    
    return num / den


@njit(cache=True, nogil=True)
def partialcorr(x, y, z, rank=False):
    """Computes partial correlation between {x} and {y} controlled for {z}

    Args:
        x (array-like): input vector 1
        y (array-like): input vector 2
        z (array-like): input vector to be controlled for
        rank (bool, optional): True or False. Defaults to False. -> Pearson correlation

    Returns:
        rp (float): (ranked) partial correlation coefficient between x and y
    """
    
    if rank:
        x = rank1d(x)
        y = rank1d(y)
        z = rank1d(z)
    
    C = np.column_stack((x, y, z))
    corr = np.corrcoef(C, rowvar=False)
    corr_inv = np.linalg.inv(corr) # the (multiplicative) inverse of a matrix.
    rp = -corr_inv[0,1] / (np.sqrt(corr_inv[0,0] * corr_inv[1,1]))
    
    return rp


@njit(cache=True, nogil=True)
def partialpearson(x, y, z):
    """Computes partial Pearson correlation between {x} and {y} controlled for {z}

    Args:
        x (array-like): input vector 1
        y (array-like): input vector 2
        z (array-like): input array to be controlled for

    Returns:
        rp (float): partial correlation coefficient between x and y
    """
    
    C = np.column_stack((x, y, z))
    corr = np.corrcoef(C, rowvar=False)
    corr_inv = np.linalg.inv(corr) # the (multiplicative) inverse of a matrix.
    rp = -corr_inv[0,1] / (np.sqrt(corr_inv[0,0] * corr_inv[1,1]))
    
    return rp


def mutualinfo(x, y, n_neighbors=3):
    """Compute mutual information between x and y using sklearn.

    Args:
        x (numpy.ndarray): shape (n_values, n_predictors)
        y (numpy.ndarray): shape (n_values, 1) or (n_values,)
        n_neighbors (int, optional): Number of neighbors for MI estimation. Defaults to 3.

    Returns:
        float: mutual information between x and y
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]
    return mutual_info_regression(x, y, discrete_features=False, n_neighbors=n_neighbors)[0]

    
@njit(cache=True, nogil=True)
def mlr(x, y, adj_r2=True, intercept=True):
    """Compute Regression of predictor(s) x on target y. 
    Requires numpy arrays with columns as predictors/target.

    Args:
        x (numpy.ndarray): shape (n_values, n_predictors)
        y (numpy.ndarray): shape (n_values, 1) or (n_values,)
        adj_r2 (bool, optional): Calculate adjusted R2. Defaults to True.
        intercept(bool, optional): Return intercept in leading position of beta array or omit 

    Returns:
        float: (adjusted) R2
        array: parameters, starting with or w/o intercept
    """
    
    n_obs = x.shape[0]
    n_x = x.shape[1]
    
    X = np.column_stack((np.ones(n_obs, dtype=x.dtype), x))
    beta = np.linalg.pinv((X.T).dot(X)).dot(X.T.dot(y))
    y_hat = np.dot(X, beta)
    ss_res = np.sum((y - y_hat)**2)       
    ss_tot = np.sum((y - np.mean(y))**2)   
    rsq = 1 - ss_res / ss_tot  
    
    if adj_r2:
        rsq = 1 - (1 - rsq) * (n_obs - 1) / (n_obs - n_x - 1)
    
    beta = beta.flatten()
    if intercept==False:
        beta = beta[1:]
    
    return (rsq, beta)


@njit(cache=True, nogil=True)
def r2(x, y, adj_r2=True):
    """Compute R2 for Regression of predictor(s) x on target y. 
    Requires numpy arrays with columns as predictors/target.

    Args:
        x (numpy.ndarray): shape (n_values, n_predictors)
        y (numpy.ndarray): shape (n_values, 1) or (n_values,)
        adj_r2 (bool, optional): Calculate adjusted R2. Defaults to True.

    Returns:
        float: (adjusted) R2
    """
    
    n_obs = x.shape[0]
    n_x = x.shape[1]
    
    X = np.column_stack((x, np.ones(n_obs, dtype=x.dtype)))
    beta = np.linalg.pinv((X.T).dot(X)).dot(X.T.dot(y))
    y_hat = np.dot(X, beta)
    ss_res = np.sum((y - y_hat)**2)       
    ss_tot = np.sum((y - np.mean(y))**2)   
    rsq = 1 - ss_res / ss_tot  
    
    if adj_r2:
        rsq = 1 - (1 - rsq) * (n_obs - 1) / (n_obs - n_x - 1)
        
    return rsq
    

@njit(cache=True, nogil=True)
def beta(x, y, intercept=True):
    """Compute beta coefficients for Regression of predictor(s) x on target y. 
    Requires numpy arrays with columns as predictors/target.

    Args:
        x (numpy.ndarray): shape (n_values, n_predictors)
        y (numpy.ndarray): shape (n_values, 1) or (n_values,)
        intercept(bool, optional): Return intercept in leading position of beta array or omit  

    Returns:
        numpy.ndarray: 1D array of beta coefficients (w or w/o intercept)
    """

    X = np.column_stack((np.ones(x.shape[0], dtype=x.dtype), x))
    beta = np.linalg.pinv((X.T).dot(X)).dot(X.T.dot(y)).flatten()

    if intercept==False:
        beta = beta[1:]
    
    return beta


def dominance(x, y, adj_r2=False, verbose=True):

    if verbose: lgr.info(f"Running dominance analysis with {x.shape[1]} "
                         f"predictors and {len(y)} features.")
    
    ## print total rsquare
    rsq_total = r2(x=x, y=y, adj_r2=adj_r2)
    if verbose: lgr.info(f"Full model R^2 = {rsq_total:.03f}")
    dom_stats = dict()
    dom_stats["sum"] = rsq_total
    
    ## get possible predictor combinations
    n_pred = x.shape[1]
    pred_combs = [list(combinations(range(n_pred), i)) for i in range(1, n_pred+1)]
    
    ## calculate R2s
    if verbose: lgr.info("Calculating models...")
    rsqs = dict()
    for len_group in tqdm(pred_combs, desc='Iterating over len groups', disable=not verbose):
        for pred_idc in tqdm(len_group, desc='Inside loop', disable=True):
            rsq = r2(x=x[:, pred_idc], y=y, adj_r2=adj_r2)
            rsqs[pred_idc] = rsq

    ## collect metrics
    # individual dominance
    if verbose: lgr.info("Calculating individual dominance.")
    dom_stats["individual"] = np.zeros((n_pred))    
    for i in range(n_pred):
        dom_stats["individual"][i] = rsqs[(i,)]
    dom_stats["individual"] = dom_stats["individual"].reshape(1, -1)
        
    # partial dominance
    if verbose: lgr.info("Calculating partial dominance.")
    dom_stats["partial"] = np.zeros((n_pred, n_pred-1)) 
    for i in range(n_pred - 1):
        i_len_combs = list(combinations(range(n_pred), i + 2))
        for j_node in range(n_pred):
            j_node_sel = [v for v in i_len_combs if j_node in v]
            reduced_list = [_del_from_tuple(comb, j_node) for comb in j_node_sel]
            diff_values = [rsqs[j_node_sel[i]] - rsqs[reduced_list[i]] for i in range(
                len(reduced_list))]
            dom_stats["partial"][j_node,i] = np.mean(diff_values)
    #dom_stats["partial"] = dom_stats["partial"].mean(axis=1)

    # total dominance
    if verbose: lgr.info("Calculating total dominance.")
    dom_stats["total"] = np.mean(np.c_[dom_stats["individual"].T, dom_stats["partial"]], axis=1)
        
    # relative contribution
    dom_stats["relative"] = dom_stats["total"] / rsq_total
    
    ## sanity check
    if not np.allclose(np.sum(dom_stats["total"]), rsq_total):
        lgr.error(f"Sum of total dominance ({np.sum(dom_stats['total'])}) does not "
                  f"equal full model R^2 ({rsq_total})! ")
    
    return dom_stats


def pls(x, y, n_components=np.inf, **kwargs):
    """
    """
    reg = PLSRegression(
        n_components=np.min([n_components, x.shape[1]]).astype(int),
        **kwargs,
    )
    reg.fit(x, y)
    
    out = {
        "r2": reg.score(x, y),
        "beta": np.squeeze(reg.coef_.T),
        "loadings": reg.x_loadings_,
    }
    
    return out


def pcr(x, y, adj_r2=True, n_components=np.inf, **kwargs):
    """
    """
    n_components = np.min([n_components, x.shape[1]]).astype(int)
    
    x_pcs = PCA(n_components=n_components, **kwargs).fit_transform(x)
    
    rsq = r2(x_pcs, y, adj_r2=adj_r2)
    
    return {"r2": rsq}


def elasticnet(x, y, cv=None, seed=None, **kwargs):
    """
    """
    
    regCV = ElasticNetCV(
        cv=cv,
        random_state=seed,
        **kwargs
    )
    regCV.fit(X=x, y=y)
    
    out = {
        "alpha": regCV.alpha_,
        "l1ratio": regCV.l1_ratio_,
        "r2": regCV.score(x, y),
        "beta": regCV.coef_
    } 
    
    return out


def lasso(x, y, cv=None, seed=None, kwargs={}):
    """
    """
    
    regCV = LassoCV(
        cv=cv,
        random_state=seed,
        **kwargs
    )
    regCV.fit(X=x, y=y)
    
    out = {
        "alpha": regCV.alpha_,
        "r2": regCV.score(x, y),
        "beta": regCV.coef_
    } 
    
    return out
    

def ridge(x, y, cv=None, seed=None, kwargs={}):
    """
    """
    
    regCV = RidgeCV(
        cv=cv,
        **kwargs
    )
    regCV.fit(X=x, y=y)
    
    out = {
        "alpha": regCV.alpha_,
        "r2": regCV.score(x, y),
        "beta": regCV.coef_
    } 
    
    return out


# Numba-accelerated implementation of sklearn-style SIMPLS for a single target
# should return the same as sklearn.cross_decomposition.PLSRegression with ~5x speed-up
@njit(fastmath=True, cache=True)
def _simpls1_loop(X_res, y_res, n_comp):
    """
    SIMPLS deflation when Y has shape (n_samples,)
    Returns W, P, Q, T_norms (x-weights, x-loadings, y-loadings, norms of T).
    """
    n, p = X_res.shape
    W = np.empty((p, n_comp))
    P = np.empty((p, n_comp))
    Q = np.empty(n_comp)
    V = np.empty((n_comp, p)).T # orthonormal basis for deflation, flipped to achieve order="F"
    T_norms = np.empty(n_comp)        

    for a in range(n_comp):
        # cross-covariance vector (instead of matrix when q == 1)
        s = X_res.T @ y_res # shape (p,)
        r = s / np.linalg.norm(s) # first left-singular vector
        
        # sklearn sign convention (svd_flip) 
        if r[np.abs(r).argmax()] < 0.0: # largest‐abs entry must be +ve
            r *= -1.0

        t = X_res @ r
        norm_t = np.linalg.norm(t)
        T_norms[a] = norm_t
        t /= norm_t
        r /= norm_t # make tᵀr == 1

        p = X_res.T @ t
        q = np.dot(y_res, t) # scalar because q == 1

        W[:, a] = r
        P[:, a] = p
        Q[a]    = q

        # orthogonalise p to build V basis
        v = p.copy()
        for j in range(a):
            v -= V[:, j] * np.dot(V[:, j], p)
        v /= np.linalg.norm(v)
        V[:, a] = v

        # deflate X and y
        X_res -= np.outer(t, p)
        y_res -= t * q

    return W, P, Q, T_norms

# full PLS function
def fast_pls1(
    x: np.ndarray,
    y: np.ndarray,
    n_components: int
):
    """
    Fast PLS (SIMPLS) for a single target.

    Parameters
    ----------
    x : (n_samples, n_features) array_like
    y : (n_samples,) or (n_samples, 1) array_like
    n_components : int
        Number of latent components.

    Returns
    -------
    coef : (n_features,) ndarray
        Regression weights in original data units.
    intercept : float
    r2 : float
        Coefficient of determination.
    x_loadings : (n_features, n_components) ndarray
        Same meaning as ``PLSRegression.x_loadings_`` from scikit-learn.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = x.shape

    n_components = np.minimum(n_components, p)

    # centre & scale (matches sklearn default)
    x_mean = x.mean(axis=0)
    y_mean = y.mean()
    xc = x - x_mean
    yc = y - y_mean

    x_std = xc.std(axis=0, ddof=1)
    y_std = yc.std(ddof=1)
    xc /= x_std
    yc /= y_std

    # latent variables via numba loop
    W, P_raw, Q, t_norms = _simpls1_loop(xc.copy(), yc.copy(), n_components)

    # sklearn-style loadings
    x_loadings = P_raw / t_norms

    # coefficients in scaled space, back-transform
    inner = np.linalg.solve(P_raw.T @ W, Q) # (n_components,)
    coef_scaled = W @ inner # (p,)
    coef = coef_scaled * (y_std / x_std)
    intercept = y_mean - x_mean @ coef

    # get R2
    y_pred = x @ coef + intercept
    r2 = 1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - y_mean) ** 2)

    return {
        "r2": r2,
        "beta": coef,
        "loadings": x_loadings,
    }
