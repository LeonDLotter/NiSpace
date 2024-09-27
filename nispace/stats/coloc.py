from itertools import combinations
import numpy as np
from numba import njit
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from .. import lgr
from ..utils.utils import _del_from_tuple


@njit(cache=True, nogil=True)
def rank_array(array):
    """Rank an array. CAVE: Cannot really deal with nan's!"""
    
    _args = array.argsort()
    ranked = np.empty_like(array)
    ranked[_args] = np.arange(array.size)
    
    return ranked


@njit(cache=True, nogil=True)
def corr(x, y, rank=False):
    """Compute Pearson or Spearman correlation for two 1D arrays."""
    
    if rank:
        x = rank_array(x)
        y = rank_array(y)
    
    m_x = x.mean()
    m_y = y.mean()
    num = np.sum((x - m_x) * (y - m_y))
    den = np.sqrt(np.sum((x - m_x) ** 2) * np.sum((y - m_y) ** 2))
    r = num / den
    
    return r


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
        x = rank_array(x)
        y = rank_array(y)
        z = rank_array(z)
    
    C = np.column_stack((x, y, z))
    corr = np.corrcoef(C, rowvar=False)
    corr_inv = np.linalg.inv(corr) # the (multiplicative) inverse of a matrix.
    rp = -corr_inv[0,1] / (np.sqrt(corr_inv[0,0] * corr_inv[1,1]))
    
    return rp

    
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