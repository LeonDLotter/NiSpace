from itertools import combinations
import numpy as np
from numba import njit
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import mutual_info_regression
from tqdm.auto import tqdm

import logging
lgr = logging.getLogger(__name__)
from ..utils.utils import _del_from_tuple

# for backwards compatibility
@njit(nogil=True)
def rank_array(array):
    """Backwards-compatibility alias for `rank1d` (same signature/behavior)."""
    return rank1d(array)

@njit(cache=True, nogil=True)
def rank1d(arr):
    """Rank a 1D array using mid-ranks (average rank) for tied values.

    Parameters
    ----------
    arr : np.ndarray, shape (n,), dtype float
        Numba-jitted: must be a plain 1D `np.ndarray`, not a list/Series.
        **Does not handle NaN** -- strip NaN entries before calling
        (see e.g. `rank2d`, which does this per-column). Constant arrays
        receive identical ranks, i.e. zero variance, which yields NaN
        when the ranks are subsequently correlated.

    Returns
    -------
    ranked : np.ndarray, shape (n,), dtype float64

    Notes
    -----
    Used internally by `rank2d` (per non-NaN column) and by `corr` (when
    `rank=True`). `core/colocalize.py`'s Spearman path ranks data via
    `rank2d` and then calls plain `pearson` on the ranks, rather than
    calling `rank1d`/`corr` directly.
    """

    n = arr.size
    _args = arr.argsort()
    ranked = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        # find the end of the current run of equal values
        j = i + 1
        while j < n and arr[_args[j]] == arr[_args[i]]:
            j += 1
        # assign the average (mid) rank to all tied elements
        mid = (i + j - 1) * 0.5
        for k in range(i, j):
            ranked[_args[k]] = mid
        i = j

    return ranked

@njit(cache=True, nogil=True)
def rank2d(arr):
    """Rank a 2D array column-wise using mid-ranks, skipping NaN per column.

    Parameters
    ----------
    arr : np.ndarray, shape (n_obs, n_features) or (n_obs,), dtype float
        Numba-jitted: must be a plain `np.ndarray`. 1D input is dispatched
        to `rank1d` directly.

    Returns
    -------
    ranked : np.ndarray, same shape as `arr`, dtype float64
        Each column's non-NaN values are ranked independently (mid-ranks
        for ties); NaN positions are left as NaN. This is the one
        NaN-*tolerant* function in this module -- unlike `rank1d`,
        `pearson`, `mlr`, etc., which all require pre-masked input.

    Notes
    -----
    Used by `core/colocalize.py`'s `_rank_regress` to rank X/Y/null arrays
    for Spearman-style correlation, where different columns may have
    different NaN patterns.
    """

    if arr.ndim == 1:
        return rank1d(arr)

    ranked = np.full(arr.shape, np.nan, dtype=np.float64)
    for i in range(arr.shape[1]):
        v = arr[:, i]
        nonan = ~np.isnan(v)
        ranked[nonan, i] = rank1d(v[nonan])

    return ranked


@njit(cache=True, nogil=True)
def corr(x, y, rank=False):
    """Compute Pearson (or, with `rank=True`, Spearman) correlation for two 1D arrays.

    Parameters
    ----------
    x, y : np.ndarray, shape (n,), dtype float
        Numba-jitted: must be plain 1D ``np.ndarray`` instances of equal length.
        **Does not handle NaN** -- strip/mask NaN entries before calling.
    rank : bool, default False
        If True, rank `x`/`y` via `rank1d` first (Spearman); if False,
        compute Pearson directly on the raw values.

    Returns
    -------
    r : float
        NaN if either array has zero variance.

    Notes
    -----
    A more generic rank-optional sibling of `pearson`; used by `nulls.py`
    and `stats/autocorr.py` for spatial-autocorrelation-null comparisons.
    `core/colocalize.py`'s own pearson/spearman colocalization path calls
    `rank2d` + plain `pearson` instead of this function.
    """

    if rank:
        x = rank1d(x)
        y = rank1d(y)

    m_x = x.mean()
    m_y = y.mean()
    num = np.sum((x - m_x) * (y - m_y))
    den = np.sqrt(np.sum((x - m_x) ** 2) * np.sum((y - m_y) ** 2))
    if den == 0.0:
        return np.nan
    return num / den


@njit(cache=True, nogil=True)
def pearson(x, y):
    """Compute Pearson correlation for two 1D arrays.

    Parameters
    ----------
    x, y : np.ndarray, shape (n,), dtype float
        Numba-jitted: must be plain 1D ``np.ndarray`` instances of equal length.
        **Does not handle NaN** -- callers must pre-mask (e.g.
        `x[mask], y[mask]`).

    Returns
    -------
    r : float
        NaN if either array has zero variance.

    Notes
    -----
    The workhorse of `core/colocalize.py`'s `"pearson"`/`"spearman"`
    colocalization path (Spearman is computed by ranking with `rank2d`
    first, then calling this function on the ranks) and of
    `core/reduce_x.py`/`core/region_influence.py`.
    """

    m_x = x.mean()
    m_y = y.mean()
    num = np.sum((x - m_x) * (y - m_y))
    den = np.sqrt(np.sum((x - m_x) ** 2) * np.sum((y - m_y) ** 2))
    if den == 0.0:
        return np.nan
    return num / den


@njit(cache=True, nogil=True)
def partialcorr(x, y, z, rank=False):
    """Closed-form partial correlation between `x` and `y`, controlling for `z`.

    Computed via inversion of the 3-variable correlation matrix, not by
    residualization.

    Parameters
    ----------
    x, y, z : np.ndarray, shape (n,), dtype float
        Numba-jitted: must be plain 1D ``np.ndarray`` instances of equal length.
        **Does not handle NaN** -- strip/mask NaN entries before calling.
    rank : bool, default False
        If True, rank `x`/`y`/`z` via `rank1d` first (partial Spearman); if
        False, use raw values (partial Pearson).

    Returns
    -------
    rp : float
        (Ranked) partial correlation coefficient between `x` and `y`.

    Notes
    -----
    Not on NiSpace's live `colocalize()` code path: `method="partialpearson"`/
    `"partialspearman"` there is computed by residualizing X/Y against Z
    first (`core/colocalize.py`'s `_rank_regress` -> `residuals_nan`) and
    then correlating the residuals with plain `pearson`, not by this
    closed-form formula. Provided as a standalone utility.
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
    """Closed-form partial Pearson correlation between `x` and `y`, controlling for `z`.

    Equivalent to `partialcorr(x, y, z, rank=False)`, without the branch.

    Parameters
    ----------
    x, y, z : np.ndarray, shape (n,), dtype float
        Numba-jitted: must be plain 1D ``np.ndarray`` instances of equal length.
        **Does not handle NaN** -- strip/mask NaN entries before calling.

    Returns
    -------
    rp : float
        Partial correlation coefficient between `x` and `y`.

    Notes
    -----
    Not on NiSpace's live `colocalize()` code path: `method="partialpearson"`
    there is computed by residualizing X/Y against Z first
    (`core/colocalize.py`'s `_rank_regress` -> `residuals_nan`) and then
    correlating the residuals with plain `pearson`, not by this closed-form
    formula. Imported into `api.py`'s namespace but not called there either
    -- provided as a standalone utility.
    """
    
    C = np.column_stack((x, y, z))
    corr = np.corrcoef(C, rowvar=False)
    corr_inv = np.linalg.inv(corr) # the (multiplicative) inverse of a matrix.
    rp = -corr_inv[0,1] / (np.sqrt(corr_inv[0,0] * corr_inv[1,1]))
    
    return rp


def mutualinfo(x, y, n_neighbors=3, seed=None):
    """Compute mutual information between `x` and `y` via sklearn's k-NN estimator.

    Thin wrapper around `sklearn.feature_selection.mutual_info_regression`
    (not numba-jitted).

    Parameters
    ----------
    x : np.ndarray, shape (n_obs,) or (n_obs, 1)
        1D input is reshaped to a column vector.
    y : np.ndarray, shape (n_obs,)
    n_neighbors : int, default 3
        Number of neighbors for the k-NN MI estimator.
    seed : int, optional
        Passed as `mutual_info_regression`'s `random_state`. The estimator adds
        small random noise to break ties, so results are not reproducible
        across calls unless this is set.

    Returns
    -------
    mi : float

    Notes
    -----
    Not NaN-tolerant (sklearn errors on NaN) -- called with pre-masked,
    NaN-free `x[mask]`/`y[mask]` in `core/colocalize.py`'s `"mi"`
    colocalization method.
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]
    return mutual_info_regression(x, y, discrete_features=False, n_neighbors=n_neighbors,
                                   random_state=seed)[0]

    
@njit(cache=True, nogil=True)
def _ols_beta_via_eigh(X, y):
    """OLS coefficients `beta = pinv(X.T @ X) @ (X.T @ y)`, via `eigh` on the Gram matrix.

    Shared by `mlr`/`r2`/`beta` (this module) and `residuals_nan`/`partial_residuals_nan`
    (`stats/misc.py`) -- every numba-jitted OLS fit in NiSpace goes through this one function.
    Uses `np.linalg.eigh`, not `np.linalg.pinv` (SVD-based): numba's SVD implementation can
    intermittently fail to converge (`Internal algorithm failed to converge`) once `X`'s
    columns approach or exceed its rows (`X.T @ X` singular/near-singular) -- observed in
    practice once benchmark sweeps started testing that regime. `eigh` is mathematically
    equivalent for this symmetric-PSD case (verified against `np.linalg.pinv` in tests), more
    numerically robust (different, more stable LAPACK routine), and faster (benchmarked under
    numba: ~1.3x at typical `n_predictors << n_obs` shapes, up to ~3x as columns approach
    rows, with zero call-overhead from being factored out here -- verified separately).
    Extracting this as a shared function costs nothing under numba: cross-`@njit` calls
    compile directly, with no Python-level dispatch.

    When `X`'s columns are rank-deficient relative to its rows, the returned `beta` is a
    genuine minimum-norm solution (the underlying regression problem is ill-posed, not a
    numerical artifact of this implementation) -- callers relying on `adj_r2`-style formulas
    downstream should expect those to become mathematically degenerate in that regime too
    (e.g. division by a term that hits zero or goes negative), by design of the formula itself.

    Parameters
    ----------
    X : np.ndarray, shape (n_obs, n_cols)
    y : np.ndarray, shape (n_obs,)

    Returns
    -------
    beta : np.ndarray, shape (n_cols,)
    """
    XtX = X.T.dot(X)
    Xty = X.T.dot(y)
    eigvals, eigvecs = np.linalg.eigh(XtX)
    cutoff = np.max(np.abs(eigvals)) * 1e-10
    inv_eigvals = np.where(eigvals > cutoff, 1.0 / eigvals, 0.0).astype(X.dtype)
    return eigvecs.dot(inv_eigvals * (eigvecs.T.dot(Xty)))


@njit(cache=True, nogil=True)
def mlr(x, y, adj_r2=True, intercept=True):
    """Multiple linear regression of predictor(s) `x` on target `y` (via pseudo-inverse).

    Parameters
    ----------
    x : np.ndarray, shape (n_obs, n_predictors), dtype float
        Numba-jitted: must be a plain 2D `np.ndarray`.
        **Does not handle NaN** -- callers must pre-mask (e.g.
        `x[mask, :], y[mask]`).
    y : np.ndarray, shape (n_obs,), dtype float
    adj_r2 : bool, default True
        Return the adjusted (rather than raw) R2.
    intercept : bool, default True
        If True, the leading entry of the returned `beta` array is the
        fitted intercept; if False, it's omitted.

    Returns
    -------
    rsq : float
        (Adjusted) R2 of the fit.
    beta : np.ndarray, shape (n_predictors + 1,) or (n_predictors,)
        Regression coefficients, with or without the leading intercept
        per `intercept`.

    Notes
    -----
    Used throughout `core/colocalize.py` (the `"mlr"` colocalization
    method and its per-predictor `"individual"` R2 drops) and
    `core/region_influence.py` (full-model R2 for regional influence).

    Coefficients come from `_ols_beta_via_eigh` (see its docstring for why `eigh`, not
    `pinv`). When `n_predictors >= n_obs`, the fit is genuinely rank-deficient, so `beta`/
    `rsq` reflect a real minimum-norm solution -- `adj_r2` in particular becomes
    mathematically degenerate there (division by `n_obs - n_x - 1`, which hits zero or goes
    negative), by design of adjusted R2 itself, not a bug in this function.
    """

    n_obs = x.shape[0]
    n_x = x.shape[1]

    X = np.column_stack((np.ones(n_obs, dtype=x.dtype), x))
    beta = _ols_beta_via_eigh(X, y)
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
    """R2 of the regression of predictor(s) `x` on target `y` (see `mlr`).

    Same fitting procedure as `mlr` but returns only the R2, without the
    beta coefficients.

    Parameters
    ----------
    x : np.ndarray, shape (n_obs, n_predictors), dtype float
        Numba-jitted: must be a plain 2D `np.ndarray`.
        **Does not handle NaN** -- callers must pre-mask.
    y : np.ndarray, shape (n_obs,), dtype float
    adj_r2 : bool, default True
        Return the adjusted (rather than raw) R2.

    Returns
    -------
    rsq : float

    Notes
    -----
    Used by `core/colocalize.py`'s `"slr"` (single-predictor) colocalization
    method, its `"mlr"` method's per-predictor `"individual"` R2 drops, and
    by `dominance` (combinatorial R2 over predictor subsets).

    Coefficients come from `_ols_beta_via_eigh` (see its docstring) -- called here, notably,
    by `pcr()` for its retained-components joint-R2, where `n_predictors` (retained
    components) can approach `n_obs`.
    """

    n_obs = x.shape[0]
    n_x = x.shape[1]

    X = np.column_stack((x, np.ones(n_obs, dtype=x.dtype)))
    beta = _ols_beta_via_eigh(X, y)
    y_hat = np.dot(X, beta)
    ss_res = np.sum((y - y_hat)**2)       
    ss_tot = np.sum((y - np.mean(y))**2)   
    rsq = 1 - ss_res / ss_tot  
    
    if adj_r2:
        rsq = 1 - (1 - rsq) * (n_obs - 1) / (n_obs - n_x - 1)
        
    return rsq
    

@njit(cache=True, nogil=True)
def beta(x, y, intercept=True):
    """Beta coefficients for the regression of predictor(s) `x` on target `y` (see `mlr`).

    Same fitting procedure as `mlr` but returns only the coefficients,
    without the R2.

    Parameters
    ----------
    x : np.ndarray, shape (n_obs, n_predictors), dtype float
        Numba-jitted: must be a plain 2D `np.ndarray`.
        **Does not handle NaN** -- callers must pre-mask.
    y : np.ndarray, shape (n_obs,), dtype float
    intercept : bool, default True
        If True, the leading entry is the fitted intercept; if False,
        it's omitted.

    Returns
    -------
    beta : np.ndarray, shape (n_predictors + 1,) or (n_predictors,)

    Notes
    -----
    Imported into `api.py`'s namespace; no direct call site found in
    `core/colocalize.py` (which uses `mlr` when both R2 and coefficients
    are needed) -- provided as a standalone coefficients-only utility.

    Coefficients come from `_ols_beta_via_eigh` (see its docstring).
    """

    X = np.column_stack((np.ones(x.shape[0], dtype=x.dtype), x))
    beta = _ols_beta_via_eigh(X, y).flatten()

    if intercept==False:
        beta = beta[1:]
    
    return beta


def dominance(x, y, adj_r2=False, verbose=False):
    """Dominance analysis: decompose R2 into each predictor's average contribution.

    Fits `r2` on every possible predictor subset (`2**n_predictors - 1`
    models) and averages each predictor's marginal R2 contribution across
    subset sizes, giving "individual", "partial", and "total" dominance per
    predictor (the total dominance values sum exactly to the full model's
    R2). Not numba-jitted; cost grows combinatorially with `n_predictors`.

    Parameters
    ----------
    x : np.ndarray, shape (n_obs, n_predictors), dtype float
        **Does not handle NaN** -- callers must pre-mask.
    y : np.ndarray, shape (n_obs,), dtype float
    adj_r2 : bool, default False
        Use adjusted R2 in the underlying `r2` fits.
    verbose : bool, default False
        Print progress (model count, running R2) as fitting proceeds.

    Returns
    -------
    dom_stats : dict
        ``"sum"`` (full-model R2, float), ``"individual"`` (shape
        ``(1, n_predictors)``), ``"partial"`` (shape
        ``(n_predictors, n_predictors - 1)``), ``"total"`` (shape
        ``(n_predictors,)``, sums to ``"sum"``), ``"relative"`` (``"total"``
        normalized to sum to 1).

    Raises
    ------
    ValueError
        If the summed total dominance does not reconstruct the full-model
        R2 within `np.allclose` tolerance (internal consistency check).

    Notes
    -----
    Used by `core/colocalize.py`'s `"dominance"` colocalization method, on
    pre-masked, NaN-free `x`/`y`.

    References
    ----------
    :cite:`azen2003`.
    """

    if verbose: print(f"Dominance analysis with {x.shape[1]} predictors and {len(y)} features.")
    
    ## print total rsquare
    rsq_total = r2(x=x, y=y, adj_r2=adj_r2)
    if verbose: print(f"Full model R^2 = {rsq_total:.03f}")
    dom_stats = dict()
    dom_stats["sum"] = rsq_total
    
    ## get possible predictor combinations
    n_pred = x.shape[1]
    pred_combs = [list(combinations(range(n_pred), i)) for i in range(1, n_pred+1)]
    
    ## calculate R2s
    if verbose: print("Calculating models...")
    rsqs = dict()
    for len_group in tqdm(pred_combs, desc='Iterating over len groups', disable=not verbose):
        for pred_idc in tqdm(len_group, desc='Inside loop', disable=True):
            rsq = r2(x=x[:, pred_idc], y=y, adj_r2=adj_r2)
            rsqs[pred_idc] = rsq

    ## collect metrics
    # individual dominance
    if verbose: print("Calculating individual dominance.")
    dom_stats["individual"] = np.zeros((n_pred))    
    for i in range(n_pred):
        dom_stats["individual"][i] = rsqs[(i,)]
    dom_stats["individual"] = dom_stats["individual"].reshape(1, -1)
        
    # partial dominance
    if verbose: print("Calculating partial dominance.")
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
    if verbose: print("Calculating total dominance.")
    dom_stats["total"] = np.mean(np.c_[dom_stats["individual"].T, dom_stats["partial"]], axis=1)
        
    # relative contribution
    dom_stats["relative"] = dom_stats["total"] / rsq_total
    
    ## sanity check
    if not np.allclose(np.sum(dom_stats["total"]), rsq_total):
        raise ValueError(f"Sum of total dominance ({np.sum(dom_stats['total'])}) does not "
                         f"equal full model R^2 ({rsq_total})! ")
    
    return dom_stats


def pls(x, y, n_components=2, **kwargs):
    """Partial least squares regression of `x` on `y` via scikit-learn's NIPALS `PLSRegression`.

    Parameters
    ----------
    x : np.ndarray, shape (n_obs, n_predictors)
        **Does not handle NaN** -- sklearn errors on NaN input; pre-mask.
    y : np.ndarray, shape (n_obs,) or (n_obs, 1)
    n_components : int, default 2
        Number of latent components; clipped to `n_predictors` if larger. Kept small by default
        -- `pls` overfits badly (both real and null draws saturate toward R2=1) once
        `n_components` becomes a non-trivial fraction of `n_obs`, or once the predictor count
        approaches/exceeds `n_obs`; see `docs/nb_benchmarks/bench4-1_set_resampling_fpr.ipynb`
        for the full story.
    **kwargs
        Forwarded to `sklearn.cross_decomposition.PLSRegression`.

    Returns
    -------
    out : dict
        ``"r2"`` (float), ``"beta"`` (shape ``(n_predictors,)``),
        ``"loadings"`` (``reg.x_loadings_``).

    Notes
    -----
    Reference/cross-check implementation only -- NiSpace's
    `colocalize(method="pls")` actually calls `fast_pls1` (a numba SIMPLS
    implementation, ~5x faster), not this function.
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


def _pca_via_eigh(x, n_components):
    """Top-`n_components` PCA scores of `x`, via eigendecomposition of the (small) predictor
    covariance matrix (`np.linalg.eigh`), not `sklearn.decomposition.PCA`'s SVD-based fit.

    Shared by `pcr()` (this module) and `core/reduce_x.py`'s `method="pca"` path. Deliberate
    choice, not an oversight: sklearn's PCA calls `scipy.linalg.svd` (LAPACK's `gesdd`
    driver), which can intermittently fail to converge (`LinAlgError`) -- a known LAPACK
    flakiness, notably on macOS's Accelerate BLAS backend, observed in practice during a
    permutation-heavy benchmark run (thousands of PCA fits, no single bad predictor set to
    blame). `eigh` uses a different, more robust LAPACK routine (symmetric eigensolver), and
    is also cheaper here since `n_predictors` is typically << `n_obs` in NiSpace's use cases --
    eigendecomposing the small `n_predictors x n_predictors` covariance matrix beats SVD-ing
    the full `n_obs x n_predictors` one. Mathematically equivalent to SVD-based PCA (verified
    against `sklearn.decomposition.PCA` in tests, up to floating-point precision and sign).

    Component 1's sign is fixed to be non-negatively correlated with the mean of `x`'s own
    (centered) columns -- the WGCNA "eigengene" convention: component 1 is typically the
    consensus direction across columns, so this gives a reproducible, interpretable sign
    ("this component increases when the average input column increases") instead of `eigh`'s
    raw sign, which is an implementation detail of the underlying LAPACK routine and not
    guaranteed stable across BLAS backends/platforms/versions. Harmless for `pcr()`, which
    immediately overrides component 1's sign again with its own Y-anchored median-vote
    convention regardless of what it receives; this is what protects `_pca_via_eigh`'s only
    other caller, `core/reduce_x.py`'s `method="pca"` path, which has no such downstream
    override and previously had no sign convention here at all. One caveat, same as any
    mean-based convention: for a genuinely bipartite/antagonistic `x` (about half the columns
    anti-correlated with the rest), the mean is a weak reference and the resulting sign can
    feel arbitrary -- immaterial here (no p-value/calibration depends on it, unlike the
    Y-anchored `score_r` case this was deliberately *not* used for, see project memory).
    Components beyond the first keep `eigh`'s raw sign -- they are orthogonal to component 1
    by construction, so there is no equivalent "overall mean" anchor for them.

    Parameters
    ----------
    x : np.ndarray, shape (n_obs, n_predictors)
        **Does not handle NaN** -- pre-mask.
    n_components : int
        Number of components to retain. Not clipped against `n_predictors` here -- callers
        must pass a valid value (both current callers clip beforehand).

    Returns
    -------
    x_pcs : np.ndarray, shape (n_obs, n_components)
    """
    xc = x - x.mean(axis=0)
    cov = xc.T @ xc
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order, arbitrary per-component sign
    order = np.argsort(eigvals)[::-1][:n_components]
    x_pcs = xc @ eigvecs[:, order]

    if x_pcs.shape[1] > 0:
        x_mean = xc.mean(axis=1)
        if pearson(x_pcs[:, 0], x_mean) < 0.0:
            x_pcs[:, 0] *= -1.0

    return x_pcs


def pcr(x, y, adj_r2=True, n_components=2):
    """Principal component regression: PCA-reduce `x`, then regress on `y` via `r2`.

    PCA comes from `_pca_via_eigh` (see its docstring for why `eigh`, not sklearn's PCA).

    Parameters
    ----------
    x : np.ndarray, shape (n_obs, n_predictors)
        **Does not handle NaN** -- pre-mask.
    y : np.ndarray, shape (n_obs,)
    adj_r2 : bool, default True
        Use adjusted R2 in the underlying `r2` fit.
    n_components : int, default 2
        Number of principal components to retain; clipped to `n_predictors` if larger. Kept
        small by default -- the adjusted-R2 correction's null-draw mean drifts well above 0
        (i.e. under-corrects) once `n_components` approaches the real effective rank of a
        resampled-with-reuse background pool; `n_components=1-2` stayed best-calibrated across
        the full tested range, `n_components` scaled up with the parcellation was the wrong
        intuition. See `docs/nb_benchmarks/bench4-1_set_resampling_fpr.ipynb` for the full story.

    Returns
    -------
    out : dict
        ``r2`` -- the R2 of `y` regressed on the retained PCs.
        ``score_r`` -- signed Pearson correlation between the first principal
        component's scores and y. Unlike ``r2`` (always >= 0), this retains
        sign. Reflects component 1 only, regardless of `n_components` --
        PCA's component 1 is deterministic given `x` alone, so unlike PLS it
        does not need a "same regardless of how many components were
        requested" caveat. Sign uses a **median-vote convention**
        (reporting-only -- does not affect any p-value, which depends on
        magnitude alone): positive iff the *median* per-predictor Pearson
        correlation with y is positive, i.e. "generally, high predictor
        value corresponds to high y, and vice versa" -- rather than
        sklearn's PCA default sign (an arbitrary largest-loading convention,
        not meaningful here). Applied in place to component 1 of the PCA
        transform itself (not just to `score_r`), so it stays consistent for
        any future per-component/per-predictor `pcr` output.

    Notes
    -----
    Used by `core/colocalize.py`'s `"pcr"` colocalization method, on
    pre-masked, NaN-free `x`/`y`.
    """
    n_components = np.min([n_components, x.shape[1]]).astype(int)

    x_pcs = _pca_via_eigh(x, n_components)

    # orient component 1 via the median-vote sign convention (see score_r above), in place on
    # x_pcs itself rather than only in a local copy -- costs nothing (a per-column sign flip
    # cannot change a joint OLS fit's R2) and keeps the reoriented component available for any
    # future per-component/per-predictor pcr output, not just score_r below. Every other
    # retained component's sign stays whatever `eigh` returned (arbitrary but immaterial: OLS
    # absorbs any per-column sign flip into the corresponding coefficient, so joint R2 is
    # unaffected either way -- only component 1 has a reporting-facing `score_r`/sign contract).
    #
    # Unlike PLS's component 1 (w1 ~ X'y, whose *unflipped* direction is provably always
    # non-negatively correlated with y -- see _simpls1_loop), PCA's PC1 is fully Y-blind
    # (chosen purely to maximize x's own variance): its raw sign has no guaranteed relationship
    # to y at all, so "flip only if the vote is negative" is NOT safe here -- verified by a
    # mixed-sign regression test where that shortcut left score_r's sign disagreeing with the
    # majority vote. Must instead explicitly compare PC1's own raw correlation with y against
    # the desired sign and flip only on a mismatch.
    xc = x - x.mean(axis=0)  # cheap to recompute (O(n*p)); _pca_via_eigh's own xc is internal
    yc = y - y.mean()
    per_pred_r = (xc.T @ yc) / (np.sqrt((xc ** 2).sum(axis=0)) * np.sqrt((yc ** 2).sum()))
    desired_sign = 1.0 if np.median(per_pred_r) >= 0.0 else -1.0
    raw_sign = 1.0 if pearson(x_pcs[:, 0], y) >= 0.0 else -1.0
    if raw_sign != desired_sign:
        x_pcs[:, 0] *= -1.0

    rsq = r2(x_pcs, y, adj_r2=adj_r2)

    score_r = float(pearson(x_pcs[:, 0], y))

    return {"r2": rsq, "score_r": score_r}


def elasticnet(x, y, cv=None, seed=None, **kwargs):
    """Elastic-net regularized regression of `x` on `y` via `sklearn.linear_model.ElasticNetCV`.

    Parameters
    ----------
    x : np.ndarray, shape (n_obs, n_predictors)
        **Does not handle NaN** -- sklearn errors on NaN input; pre-mask.
    y : np.ndarray, shape (n_obs,)
    cv : int, cross-validation generator, or None
        Passed to `ElasticNetCV` for selecting `alpha`/`l1_ratio`.
    seed : int, optional
        Passed as `ElasticNetCV`'s `random_state`.
    **kwargs
        Forwarded to `ElasticNetCV`.

    Returns
    -------
    out : dict
        ``"alpha"``/``"l1ratio"`` (selected regularization strength/mix),
        ``"r2"``, ``"beta"`` (shape ``(n_predictors,)``).

    Notes
    -----
    Used by `core/colocalize.py`'s regularized-regression colocalization
    case. Unlike the other coloc.py methods (which exclude NaN case-wise,
    i.e. per predictor combination), the regularized methods
    (`elasticnet`/`lasso`/`ridge`) exclude NaN list-wise across all
    predictors at once before calling this function.
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
    """Lasso-regularized regression of `x` on `y` via `sklearn.linear_model.LassoCV`.

    Parameters
    ----------
    x : np.ndarray, shape (n_obs, n_predictors)
        **Does not handle NaN** -- sklearn errors on NaN input; pre-mask.
    y : np.ndarray, shape (n_obs,)
    cv : int, cross-validation generator, or None
        Passed to `LassoCV` for selecting `alpha`.
    seed : int, optional
        Passed as `LassoCV`'s `random_state`.
    kwargs : dict, default {}
        Forwarded to `LassoCV`.

    Returns
    -------
    out : dict
        ``"alpha"`` (selected regularization strength), ``"r2"``,
        ``"beta"`` (shape ``(n_predictors,)``).

    Notes
    -----
    Used by `core/colocalize.py`'s regularized-regression colocalization
    case, with NaN excluded list-wise (see `elasticnet`'s Notes) before
    calling this function.
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
    """Ridge-regularized regression of `x` on `y` via `sklearn.linear_model.RidgeCV`.

    Parameters
    ----------
    x : np.ndarray, shape (n_obs, n_predictors)
        **Does not handle NaN** -- sklearn errors on NaN input; pre-mask.
    y : np.ndarray, shape (n_obs,)
    cv : int, cross-validation generator, or None
        Passed to `RidgeCV` for selecting `alpha`.
    seed : int, optional
        Unused by `RidgeCV` (which has no `random_state`); accepted for a
        uniform signature with `lasso`/`elasticnet`.
    kwargs : dict, default {}
        Forwarded to `RidgeCV`.

    Returns
    -------
    out : dict
        ``"alpha"`` (selected regularization strength), ``"r2"``,
        ``"beta"`` (shape ``(n_predictors,)``).

    Notes
    -----
    Used by `core/colocalize.py`'s regularized-regression colocalization
    case, with NaN excluded list-wise (see `elasticnet`'s Notes) before
    calling this function.
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

        if a == 0:
            # Median-vote sign convention, reporting-only (doesn't affect any p-value, which
            # depends on magnitude alone): orient the first, undeflated component so it's
            # positive iff the MEDIAN per-predictor Pearson correlation with y is positive --
            # i.e. "generally, high predictor value <-> high y". s[i] here is proportional (by
            # one shared positive constant, since X_res/y_res are already centered+scaled to
            # unit variance at this undeflated first iteration) to predictor i's own
            # correlation with y, so sign(median(s)) already equals sign(median per-predictor
            # correlation) directly -- no separate per-predictor correlation pass needed.
            # Reproduces a plain majority-by-count vote whenever there's a clear majority (a
            # median's sign among untied values equals the majority sign by count), but --
            # unlike a pure count vote -- needs no separate tie-break rule near a 50/50 split
            # (the median naturally uses the boundary values' own magnitude there) and is
            # robust to a single outlier predictor (unlike a sum/mean-based vote).
            if np.median(s) < 0.0:
                r = -r
        else:
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
    Fast PLS via the SIMPLS algorithm for a single target.

    Numba-accelerated (`_simpls1_loop`); matches
    `sklearn.cross_decomposition.PLSRegression` output with ~5x speed-up.
    This is the implementation NiSpace's `colocalize(method="pls")`
    actually calls (not the plain sklearn-based `pls` function above).
    Implements SIMPLS :cite:`dejong1993`.

    Parameters
    ----------
    x : (n_samples, n_features) array_like
        **Does not handle NaN** -- callers must pre-mask.
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
    score_r : float
        Signed Pearson correlation between the first latent component's score
        (``t1 = xc @ w1``, on centered/scaled data) and y. Unlike ``r2``
        (always >= 0), this retains sign. Reflects component 1 only,
        regardless of ``n_components``. ``score_r ** 2 == r2`` holds only when
        ``n_components == 1`` (for ``n_components > 1``, ``r2`` reflects the
        full multi-component fit while ``score_r`` reflects only component 1 --
        expected divergence, not a bug).
    weight : (n_features,) ndarray
        Unit-norm PLS weight vector for component 1, same magnitude as
        ``PLSRegression.x_weights_[:, 0]`` from scikit-learn but re-oriented:
        both ``weight`` and ``score_r`` use a **median-vote sign convention**
        (reporting-only -- does not affect any p-value, which depends on
        magnitude alone): positive iff the *median* per-predictor Pearson
        correlation with y is positive, i.e. "generally, high predictor
        value corresponds to high y, and vice versa" (the same
        phenotype-anchored idea GSEA uses to label a gene set
        "positively"/"negatively" enriched, rather than an internal
        PCA/PLS solver convention). Reproduces a plain majority-by-count
        vote whenever there is a clear majority, degrades gracefully (no
        arbitrary tie-break needed) near a 50/50 split, and is robust to a
        single outlier predictor (unlike a sum/mean-based vote). Distinct
        from ``beta`` (the back-transformed regression coefficient, which --
        like ``mlr``'s ``beta`` -- can be severely underpowered for
        significance testing under multicollinear x, since correlated
        predictors share credit for the same explained variance) and from
        ``x_loadings``/``loadings`` (used to reconstruct x from scores, not
        to compute them).

    References
    ----------
    :cite:`dejong1993`.
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

    # component-1-only outputs, independent of n_components requested above.
    # W[:, 0] as returned by _simpls1_loop is scaled so that t^T r == 1 (an internal
    # deflation convenience, not a meaningful weight magnitude) -- renormalize to unit
    # norm to match sklearn's `x_weights_` magnitude and the imaging-transcriptomics
    # PLS-weight literature. Sign (of both t1 and weight) was already fixed by
    # _simpls1_loop's median-vote convention at component 1; t1 is computed from the
    # pre-renormalization column, staying algebraically consistent with the SIMPLS loop's
    # own scores (Pearson correlation is invariant to positive rescaling either way, so
    # renormalizing weight doesn't affect score_r).
    t1 = xc @ W[:, 0]
    score_r = float(pearson(t1, yc))
    weight = W[:, 0] / np.linalg.norm(W[:, 0])

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
        "score_r": score_r,
        "weight": weight,
    }
