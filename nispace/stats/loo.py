"""Closed-form (case-deletion) leave-one-out diagnostics.

These reproduce, exactly, the result of deleting one observation and
refitting from scratch -- not an approximation -- via the standard
regression-diagnostics identities (Cook & Weisberg 1982; Belsley, Kuh &
Welsch 1980), computed once from the full fit rather than by refitting
n times. Internally always computed in float64 regardless of input dtype
(rank-1 downdates can lose precision at small pivots in float32), matching
the "float64 internally" convention already used by ``rank1d``/``rank2d``
in ``stats/coloc.py``. Cast back to the input dtype at the return boundary.
"""

import numpy as np

import logging
lgr = logging.getLogger(__name__)


def pearson_loo(x, y):
    """Per-point leave-one-out Pearson correlation.

    For each valid position i, returns the Pearson correlation of x and y
    computed with position i excluded -- exact, via a running-sum downdate,
    not a refit. NaN-aware: positions where x or y is NaN are excluded from
    the fit entirely and get NaN in the output; every other position's
    output is computed from the remaining valid positions only.

    Args:
        x (numpy.ndarray): shape (n,)
        y (numpy.ndarray): shape (n,)

    Returns:
        numpy.ndarray: shape (n,), dtype matching x's input dtype
    """
    out_dtype = np.result_type(x, y) if not np.issubdtype(np.asarray(x).dtype, np.floating) \
        else np.asarray(x).dtype

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]

    valid = ~np.isnan(x) & ~np.isnan(y)
    n_valid = int(valid.sum())

    out = np.full(n, np.nan, dtype=np.float64)
    if n_valid < 3:
        return out.astype(out_dtype)

    xv = x[valid]
    yv = y[valid]

    Sx, Sy = xv.sum(), yv.sum()
    Sxx, Syy, Sxy = (xv * xv).sum(), (yv * yv).sum(), (xv * yv).sum()

    n_loo = n_valid - 1
    Sx_i = Sx - xv
    Sy_i = Sy - yv
    Sxx_i = Sxx - xv * xv
    Syy_i = Syy - yv * yv
    Sxy_i = Sxy - xv * yv

    num = n_loo * Sxy_i - Sx_i * Sy_i
    den = np.sqrt((n_loo * Sxx_i - Sx_i ** 2) * (n_loo * Syy_i - Sy_i ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        r_loo = num / den

    out[valid] = r_loo
    return out.astype(out_dtype)


def _hat_diag(Xd, XtX_inv):
    """Leverage (hat-matrix diagonal) for every row of design matrix Xd.

    Args:
        Xd (numpy.ndarray): shape (n, p), design matrix incl. intercept column
        XtX_inv (numpy.ndarray): shape (p, p), (Xd.T @ Xd)^-1

    Returns:
        numpy.ndarray: shape (n,)
    """
    return np.einsum("ij,jk,ik->i", Xd, XtX_inv, Xd)


def mlr_loo(X, y, adj_r2=True):
    """Per-point leave-one-out (adjusted) R^2 for multiple linear regression.

    For each valid position i, returns the R^2 of a regression of y on X
    computed with position i excluded -- exact, via the standard
    hat-matrix/leverage case-deletion identities, not a refit. NaN-aware:
    positions where any column of X or y is NaN are excluded from the fit
    entirely and get NaN in the output.

    Args:
        X (numpy.ndarray): shape (n, n_predictors)
        y (numpy.ndarray): shape (n,)
        adj_r2 (bool, optional): Calculate adjusted R2. Defaults to True.

    Returns:
        numpy.ndarray: shape (n,), dtype matching X's input dtype
    """
    out_dtype = np.asarray(X).dtype if np.issubdtype(np.asarray(X).dtype, np.floating) \
        else np.result_type(X, y)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = X.shape[0]
    n_x = X.shape[1]

    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    n_valid = int(valid.sum())

    out = np.full(n, np.nan, dtype=np.float64)
    if n_valid < n_x + 3:
        return out.astype(out_dtype)

    Xv = X[valid]
    yv = y[valid]

    Xd = np.column_stack((np.ones(n_valid, dtype=np.float64), Xv))
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    beta = XtX_inv @ (Xd.T @ yv)
    y_hat = Xd @ beta
    resid = yv - y_hat

    h = _hat_diag(Xd, XtX_inv)

    sse_full = np.sum(resid ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        sse_loo = sse_full - resid ** 2 / (1.0 - h)

    n_loo = n_valid - 1
    Sy, Syy = yv.sum(), (yv * yv).sum()
    mean_loo = (Sy - yv) / n_loo
    with np.errstate(divide="ignore", invalid="ignore"):
        sst_loo = (Syy - yv ** 2) - n_loo * mean_loo ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        rsq_loo = 1.0 - sse_loo / sst_loo
        if adj_r2:
            rsq_loo = 1.0 - (1.0 - rsq_loo) * (n_loo - 1) / (n_loo - n_x - 1)

    out[valid] = rsq_loo
    return out.astype(out_dtype)
