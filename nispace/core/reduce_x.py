import numpy as np
from sklearn.decomposition import FastICA
from statsmodels.multivariate.factor import Factor
from statsmodels.multivariate.factor_rotation import rotate_factors

import logging
lgr = logging.getLogger(__name__)
from ..stats.coloc import pearson, _pca_via_eigh

# factor extraction: NiSpace name -> statsmodels Factor method. Iterated principal-axis
# factoring converges to the minimum-residual (unweighted least squares) solution.
_FA_METHODS = {"minres": "pa", "ml": "ml"}
_FA_ROTATIONS = (None, "varimax", "promax")
# iterated principal axis: statsmodels' default (50) stops before convergence on real data
_FA_PA_MAXITER = 1000


def _kaiser_varimax(loadings):
    """Varimax rotation with Kaiser normalization (rows scaled to unit communality before
    rotating, scaled back after), as in R's ``varimax()`` / ``psych``."""
    h = np.sqrt((loadings ** 2).sum(axis=1, keepdims=True))
    h[h == 0] = 1
    rotated, _ = rotate_factors(loadings / h, "varimax")
    return rotated * h


def _kaiser_promax(loadings, power=4):
    """Promax rotation (Hendrickson & White, 1964) with Kaiser normalization, as in R's
    ``promax()``: varimax, then a least-squares fit to the target ``L * |L|**(power-1)``.

    Returns the pattern loadings and the factor correlation matrix ``phi``.
    """
    h = np.sqrt((loadings ** 2).sum(axis=1, keepdims=True))
    h[h == 0] = 1
    vari, _ = rotate_factors(loadings / h, "varimax")
    target = vari * np.abs(vari) ** (power - 1)
    coef = np.linalg.lstsq(vari, target, rcond=None)[0]
    coef = coef * np.sqrt(np.diag(np.linalg.pinv(coef.T @ coef)))
    coef_inv = np.linalg.inv(coef)
    return (vari @ coef) * h, coef_inv @ coef_inv.T


def _fit_fa(data, n_factors, method, rotation):
    """Factor analysis of ``data`` (observations x variables) on its correlation matrix.

    Returns ``(loadings, phi)``: loadings (variables x factors), each factor's sign set so
    that its loadings sum to >= 0, factors sorted by explained variance; ``phi`` is the
    factor correlation matrix for oblique rotations (``"promax"``), else ``None``.
    """
    fit_kwargs = {"maxiter": _FA_PA_MAXITER} if _FA_METHODS[method] == "pa" else {}
    loadings = Factor(data, n_factor=n_factors, method=_FA_METHODS[method], smc=True)\
        .fit(**fit_kwargs).loadings
    # statsmodels' "ml" returns complex128 loadings with zero imaginary parts on numpy >= 2.5
    loadings = np.real_if_close(loadings, tol=1000)
    if np.iscomplexobj(loadings):
        lgr.critical_raise("Factor analysis returned complex loadings; the factor model does "
                           "not fit these data. Try fewer factors or fa_method='minres'.",
                           ValueError)
    phi = None
    if rotation is not None and loadings.shape[1] > 1:
        if rotation == "varimax":
            loadings = _kaiser_varimax(loadings)
        else:
            loadings, phi = _kaiser_promax(loadings)
    # sign convention (as in R psych / factor_analyzer): column sums positive
    signs = np.sign(loadings.sum(axis=0))
    signs[signs == 0] = 1
    loadings = loadings * signs
    # sort by explained variance (sum of squared loadings), descending
    order = np.argsort(-(loadings ** 2).sum(axis=0), kind="stable")
    loadings = loadings[:, order]
    if phi is not None:
        phi = (phi * np.outer(signs, signs))[np.ix_(order, order)]
    return loadings, phi


def _fa_scores(data, loadings, phi=None):
    """Regression (Thurstone) factor scores: ``Z R^-1 S`` with ``Z`` the column-standardized
    data (ddof=0), ``R`` its correlation matrix and ``S`` the structure matrix
    (``loadings @ phi``; the loadings for orthogonal factors)."""
    z = (data - data.mean(axis=0)) / data.std(axis=0)
    structure = loadings if phi is None else loadings @ phi
    weights = np.linalg.lstsq(np.corrcoef(data, rowvar=False), structure, rcond=None)[0]
    return z @ weights


def _reduce_dimensions(data, method="pca", n_components=None, min_ev=None, 
                       fa_method="minres", fa_rotation="promax",
                       seed=None):
    
    # to array
    data = np.array(data)
    
    # set n_components to max number if min explained variance is given
    n_components = data.shape[1] if (n_components is None) | (min_ev is not None) else n_components
    lgr.info(f"Performing dimensionality reduction using {method} (max components: "
             f"{n_components}, min EV: {min_ev}).")
    
    # case pca
    if method=="pca":
        # run pca with all components -- via _pca_via_eigh (eigh on the small predictor
        # covariance matrix), not sklearn.decomposition.PCA (SVD-based, can intermittently
        # fail to converge; see its docstring). Per-component sign is arbitrary either way
        # (matches any eigendecomposition/SVD), so may differ from a previous sklearn-based
        # run -- the subspace/explained variance is identical, only individual component
        # signs may flip.
        pcs = _pca_via_eigh(data, n_components)
        # ddof=0 on both sides: ratio is invariant to ddof choice as long as it
        # matches; pinned to population convention for consistency/clarity
        ev = np.var(pcs, axis=0, ddof=0) / np.sum(np.var(data, axis=0, ddof=0))
        # find number of components that sum up to total EV of >= min_ev
        if min_ev is not None:
            total_ev = 0
            for i, e in enumerate(ev):
                total_ev += e
                if total_ev>=min_ev:
                    n_components = i+1
                    lgr.info(f"{n_components} PC(s) explain(s) a total variance of "
                             f"{np.sum(ev[:n_components]):.04f} >= {min_ev} ({ev[:n_components]}).")
                    break
        # cut components & ev
        components = pcs[:,:n_components]
        ev = ev[:n_components]
        lgr.info(f"Returning {n_components} principal component(s).")
    
    # case ica
    elif method=="ica":
        components = FastICA(n_components=n_components, random_state=seed, max_iter=1000)\
            .fit_transform(data) 
        ev = None
        lgr.info(f"Returning {n_components} independent component(s).")
 
    # case fa
    elif method=="fa":
        if fa_method not in _FA_METHODS:
            lgr.critical_raise(f"fa_method = '{fa_method}' not supported; use one of "
                               f"{list(_FA_METHODS)}.", ValueError)
        if fa_rotation not in _FA_ROTATIONS:
            lgr.critical_raise(f"fa_rotation = '{fa_rotation}' not supported; use one of "
                               f"{list(_FA_ROTATIONS)}.", ValueError)

        # find the number of unrotated factors whose cumulative EV is >= min_ev
        if min_ev is not None:
            loadings, _ = _fit_fa(data, n_components, fa_method, None)
            ev = np.cumsum((loadings ** 2).sum(axis=0)) / data.shape[1]
            if ev[-1] < min_ev:
                n_components = len(ev)
                lgr.warning(f"Given min EV ({min_ev}) > max possible EV ({ev[-1]:.02f})! "
                            f"Using max factor number ({n_components}).")
            else:
                n_components = int(np.argmax(ev >= min_ev)) + 1
                lgr.info(f"{n_components} factor(s) explain(s) a total variance of "
                         f"{ev[n_components - 1]:.02f} >= {min_ev}.")
        # run actual factor analysis
        loadings, phi = _fit_fa(data, n_components, fa_method, fa_rotation)
        n_components = loadings.shape[1]
        components = _fa_scores(data, loadings, phi).astype(data.dtype)
        # proportion of total variance per factor (sum of squared pattern loadings / n vars)
        ev = (loadings ** 2).sum(axis=0) / data.shape[1]
        lgr.info(f"Returning {n_components} factor(s).")
        
    else:
        lgr.critical_raise(f"method = '{method}' not defined!", ValueError)
    
    # get PCA and ICA "loadings"
    if method in ["pca", "ica"]:
        components = components.astype(data.dtype)
        loadings = np.zeros((data.shape[1], n_components))
        for c in range(n_components):
            for r in range(data.shape[1]):
                loadings[r, c] = pearson(x=data[:, r], y=components[:, c])
    ## return
    return components, ev, loadings