"""Keyword arguments of the regularized colocalization methods (lasso/ridge/elasticnet) reach
the sklearn estimators; ``random_state`` overrides NiSpace's seed instead of colliding with it."""

import numpy as np
import pytest
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from nispace import NiSpace
from nispace.stats.coloc import elasticnet, lasso, ridge

N_PARCELS = 24  # fake_parcellation


@pytest.fixture
def data(rng):
    X = rng.normal(size=(100, 5))
    y = X @ np.array([1.0, 0.0, -0.5, 0.0, 0.3]) + rng.normal(scale=0.5, size=100)
    return X, y


def test_lasso_forwards_kwargs(data):
    X, y = data
    alphas = (0.001, 0.05, 2.0)
    out = lasso(X, y, cv=5, seed=42, alphas=alphas, max_iter=50)
    expected = LassoCV(cv=5, random_state=42, alphas=alphas, max_iter=50).fit(X, y)
    assert out["alpha"] in alphas
    assert np.isclose(out["alpha"], expected.alpha_)
    assert np.allclose(out["beta"], expected.coef_)


def test_ridge_forwards_kwargs(data):
    X, y = data
    alphas = (0.01, 100.0, 1e4)
    out = ridge(X, y, cv=5, alphas=alphas)
    expected = RidgeCV(cv=5, alphas=alphas).fit(X, y)
    assert out["alpha"] in alphas
    assert np.isclose(out["alpha"], expected.alpha_)
    assert np.allclose(out["beta"], expected.coef_)


@pytest.mark.parametrize("fun, est", [(lasso, LassoCV), (elasticnet, ElasticNetCV)])
def test_random_state_overrides_seed(data, fun, est):
    X, y = data
    kw = dict(selection="random", tol=1e-1, max_iter=5)
    out = fun(X, y, cv=5, seed=1, random_state=7, **kw)
    expected = est(cv=5, random_state=7, **kw).fit(X, y)
    assert np.allclose(out["beta"], expected.coef_)


@pytest.mark.parametrize("method", ["lasso", "ridge", "elasticnet"])
def test_colocalize_regularized_kwargs(rng, fake_parcellation, method):
    x = rng.normal(size=(5, N_PARCELS))
    y = rng.normal(size=(2, N_PARCELS))
    nsp = NiSpace(x=x, y=y, parcellation=fake_parcellation, seed=0, verbose=False).fit()
    kw = {"alphas": (0.1, 1.0)} if method == "ridge" else {"max_iter": 500}
    if method == "elasticnet":
        kw["random_state"] = 3
    nsp.colocalize(method, **kw)
    colocs = nsp.get_colocalizations(method, force_dict=True)
    assert np.isfinite(colocs["r2"].values).all()
    if method == "ridge":
        assert set(colocs["alpha"].values.ravel()) <= {0.1, 1.0}
