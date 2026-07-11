"""Low-level tests for nispace.stats.loo -- no NiSpace object involved.

Verifies that the closed-form case-deletion identities in stats/loo.py are
numerically identical to actually deleting a point and refitting from
scratch (via the existing stats/coloc.py primitives, used here purely as an
independent brute-force oracle). Tolerance: 1e-6 for mlr_loo (float64
matrix-inverse arithmetic accumulates a bit more error than the pure
sum-downdate used by pearson_loo, which is held to 1e-8).
"""

import numpy as np
import pytest

from nispace.stats.loo import pearson_loo, mlr_loo
from nispace.stats.coloc import pearson, mlr

PEARSON_ATOL = 1e-8
MLR_ATOL = 1e-6


def _brute_force_pearson_loo(x, y):
    n = x.shape[0]
    return np.array([pearson(np.delete(x, i), np.delete(y, i)) for i in range(n)])


def _brute_force_mlr_loo(X, y, adj_r2=True):
    n = X.shape[0]
    return np.array([
        mlr(np.delete(X, i, axis=0), np.delete(y, i), adj_r2=adj_r2, intercept=True)[0]
        for i in range(n)
    ])


@pytest.mark.parametrize("n", [15, 30, 60, 200])
def test_pearson_loo_matches_brute_force(rng, n):
    x = rng.normal(size=n)
    y = rng.normal(size=n) + 0.5 * x
    loo = pearson_loo(x, y)
    brute = _brute_force_pearson_loo(x, y)
    np.testing.assert_allclose(loo, brute, atol=PEARSON_ATOL)


def test_pearson_loo_nan_handling(rng):
    n = 40
    x = rng.normal(size=n)
    y = rng.normal(size=n) + 0.5 * x
    nan_idx = rng.choice(n, size=5, replace=False)
    x_nan = x.copy()
    x_nan[nan_idx] = np.nan

    loo = pearson_loo(x_nan, y)
    valid = ~np.isnan(x_nan)

    assert np.all(np.isnan(loo[nan_idx]))
    brute = _brute_force_pearson_loo(x_nan[valid], y[valid])
    np.testing.assert_allclose(loo[valid], brute, atol=PEARSON_ATOL)


@pytest.mark.parametrize("n,n_x", [(20, 1), (30, 3), (60, 3), (200, 10)])
def test_mlr_loo_matches_brute_force(rng, n, n_x):
    X = rng.normal(size=(n, n_x))
    beta = rng.normal(size=n_x)
    y = X @ beta + rng.normal(scale=0.3, size=n)
    loo = mlr_loo(X, y, adj_r2=True)
    brute = _brute_force_mlr_loo(X, y, adj_r2=True)
    np.testing.assert_allclose(loo, brute, atol=MLR_ATOL)


def test_mlr_loo_nan_handling(rng):
    n, n_x = 50, 3
    X = rng.normal(size=(n, n_x))
    beta = rng.normal(size=n_x)
    y = X @ beta + rng.normal(scale=0.3, size=n)

    nan_rows = rng.choice(n, size=4, replace=False)
    X_nan = X.copy()
    X_nan[nan_rows[:2], 0] = np.nan
    y_nan = y.copy()
    y_nan[nan_rows[2:]] = np.nan

    loo = mlr_loo(X_nan, y_nan, adj_r2=True)
    valid = ~np.isnan(X_nan).any(axis=1) & ~np.isnan(y_nan)

    assert np.all(np.isnan(loo[~valid]))
    brute = _brute_force_mlr_loo(X_nan[valid], y_nan[valid], adj_r2=True)
    np.testing.assert_allclose(loo[valid], brute, atol=MLR_ATOL)


def test_mlr_loo_near_collinear_stability(rng):
    """Near-collinear predictors stress-test the (X'X)^-1 downdate; confirms
    the float64-internal implementation stays accurate where a naive
    float32 version would be expected to degrade."""
    n = 60
    x1 = rng.normal(size=n)
    x2 = x1 + rng.normal(scale=1e-4, size=n)
    X = np.column_stack([x1, x2])
    beta = np.array([1.0, 1.0])
    y = X @ beta + rng.normal(scale=0.3, size=n)

    loo = mlr_loo(X, y, adj_r2=True)
    brute = _brute_force_mlr_loo(X, y, adj_r2=True)
    np.testing.assert_allclose(loo, brute, atol=1e-4)
