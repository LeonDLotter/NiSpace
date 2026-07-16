"""Ground-truth tests for nispace.stats.coloc.

Every function here reimplements something an established library already
provides (numba-jitted for speed, or wrapping sklearn directly). Prior tests
only checked internal consistency (e.g. `_fast` vs plain variants, or hand-
written "expected" formulas) -- never an *independent* reference
implementation, so a bug shared between the implementation and a
hand-written test formula would slip through. These tests instead compare
against scipy/sklearn/statsmodels/pingouin, calling each library directly
(not through another nispace wrapper), including NaN handling where the
reference library supports it (`nan_policy="omit"` / manual dropna).
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr, rankdata

from nispace.stats.coloc import (
    pearson, corr, rank1d, rank2d,
    partialpearson, partialcorr,
    mlr, r2, beta, dominance,
    pls, pcr, fast_pls1,
    lasso, ridge, elasticnet, mutualinfo,
)


# ── rank1d / rank2d vs scipy.stats.rankdata ──────────────────────────────────
# nispace's ranks are 0-based (scipy's rankdata is 1-based) -- otherwise
# identical, including its mid-rank tie-breaking convention.

def test_rank1d_matches_scipy_rankdata_average(rng):
    x = rng.normal(size=40)
    x[[2, 5, 10]] = x[0]  # inject ties on purpose
    expected = rankdata(x, method="average") - 1
    assert np.allclose(rank1d(x), expected)


def test_rank2d_matches_scipy_rankdata_columnwise(rng):
    a = rng.normal(size=(30, 3))
    a[[1, 2], 0] = a[0, 0]  # ties in column 0
    expected = np.column_stack([rankdata(a[:, i], method="average") - 1 for i in range(3)])
    assert np.allclose(rank2d(a), expected)


def test_rank2d_with_nan_matches_scipy_on_non_nan_subset(rng):
    a = rng.normal(size=(30, 2))
    a[[0, 5], 0] = np.nan
    result = rank2d(a)
    assert np.isnan(result[0, 0]) and np.isnan(result[5, 0])
    mask = ~np.isnan(a[:, 0])
    expected = rankdata(a[mask, 0], method="average") - 1
    assert np.allclose(result[mask, 0], expected)


# ── pearson / corr vs scipy.stats.pearsonr ───────────────────────────────────

def test_pearson_matches_scipy(rng):
    x = rng.normal(size=100)
    y = 0.6 * x + rng.normal(scale=0.8, size=100)
    expected, _ = pearsonr(x, y)
    assert np.isclose(pearson(x, y), expected)
    assert np.isclose(corr(x, y, rank=False), expected)


# ── Spearman (rank2d + pearson) vs scipy.stats.spearmanr ─────────────────────

def test_spearman_via_rank2d_matches_scipy(rng):
    x = rng.normal(size=100)
    y = 0.6 * x + rng.normal(scale=0.8, size=100)
    expected, _ = spearmanr(x, y)
    ranked = rank2d(np.column_stack([x, y]))
    assert np.isclose(pearson(ranked[:, 0], ranked[:, 1]), expected)
    assert np.isclose(corr(x, y, rank=True), expected)


def test_spearman_with_nan_matches_scipy_listwise_omit(rng):
    # scipy's nan_policy="omit" for spearmanr does listwise deletion (both
    # x and y non-nan); rank2d itself is only per-column NaN-tolerant, so
    # replicate listwise deletion by hand before ranking -- this is also
    # how a real caller must use rank2d/pearson (neither is listwise-aware).
    x = rng.normal(size=60)
    y = 0.6 * x + rng.normal(scale=0.8, size=60)
    x[[2, 5, 10]] = np.nan
    y[[7, 10, 20]] = np.nan
    expected, _ = spearmanr(x, y, nan_policy="omit")

    mask = ~np.isnan(x) & ~np.isnan(y)
    ranked = rank2d(np.column_stack([x[mask], y[mask]]))
    assert np.isclose(pearson(ranked[:, 0], ranked[:, 1]), expected)


# ── partialpearson / partialcorr vs pingouin.partial_corr ────────────────────

def test_partialpearson_matches_pingouin():
    pg = pytest.importorskip("pingouin")
    rng = np.random.default_rng(7)
    n = 200
    x = rng.normal(size=n)
    z = rng.exponential(size=n)  # skewed, so pearson vs spearman actually differ
    y = 0.5 * x + 0.6 * z + rng.normal(scale=0.8, size=n)

    df = pd.DataFrame({"x": x, "y": y, "z": z})
    expected = pg.partial_corr(data=df, x="x", y="y", covar="z", method="pearson")["r"].iloc[0]

    assert np.isclose(partialpearson(x, y, z), expected)
    assert np.isclose(partialcorr(x, y, z, rank=False), expected)


def test_partial_spearman_matches_pingouin():
    pg = pytest.importorskip("pingouin")
    rng = np.random.default_rng(7)
    n = 200
    x = rng.normal(size=n)
    z = rng.exponential(size=n)
    y = 0.5 * x + 0.6 * z + rng.normal(scale=0.8, size=n)

    df = pd.DataFrame({"x": x, "y": y, "z": z})
    expected = pg.partial_corr(data=df, x="x", y="y", covar="z", method="spearman")["r"].iloc[0]

    assert np.isclose(partialcorr(x, y, z, rank=True), expected, atol=1e-6)


def test_partialpearson_with_nan_matches_pingouin_on_complete_cases():
    # partialpearson itself doesn't handle NaN (must pre-mask) -- verify
    # against pingouin computed on the same listwise-complete subset.
    pg = pytest.importorskip("pingouin")
    rng = np.random.default_rng(7)
    n = 200
    x = rng.normal(size=n)
    z = rng.exponential(size=n)
    y = 0.5 * x + 0.6 * z + rng.normal(scale=0.8, size=n)
    x[[1, 2, 3]] = np.nan
    y[[4, 5]] = np.nan

    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    df = pd.DataFrame({"x": x[mask], "y": y[mask], "z": z[mask]})
    expected = pg.partial_corr(data=df, x="x", y="y", covar="z", method="pearson")["r"].iloc[0]

    assert np.isclose(partialpearson(x[mask], y[mask], z[mask]), expected)


# ── mlr / r2 / beta vs sklearn.linear_model.LinearRegression ────────────────

@pytest.fixture
def regression_data(rng):
    n = 150
    x = rng.normal(size=n)
    z = rng.exponential(size=n)
    X = np.column_stack([x, z])
    y = 0.5 * x + 0.6 * z + rng.normal(scale=0.5, size=n)
    return X, y


def test_mlr_matches_sklearn_linear_regression(regression_data):
    from sklearn.linear_model import LinearRegression
    X, y = regression_data
    rsq, betas = mlr(X, y, adj_r2=False, intercept=True)
    lr = LinearRegression().fit(X, y)
    assert np.isclose(rsq, lr.score(X, y))
    assert np.isclose(betas[0], lr.intercept_)
    assert np.allclose(betas[1:], lr.coef_)


def test_r2_matches_sklearn_r2(regression_data):
    from sklearn.linear_model import LinearRegression
    X, y = regression_data
    rsq = r2(X, y, adj_r2=False)
    lr = LinearRegression().fit(X, y)
    assert np.isclose(rsq, lr.score(X, y))


def test_beta_matches_sklearn_coef(regression_data):
    from sklearn.linear_model import LinearRegression
    X, y = regression_data
    betas = beta(X, y, intercept=True)
    lr = LinearRegression().fit(X, y)
    assert np.isclose(betas[0], lr.intercept_)
    assert np.allclose(betas[1:], lr.coef_)


def test_mlr_adj_r2_matches_manual_formula(regression_data):
    # no library exposes adjusted R2 directly on a fit object -- cross-check
    # against the textbook formula applied to sklearn's raw R2.
    from sklearn.linear_model import LinearRegression
    X, y = regression_data
    n_obs, n_x = X.shape
    lr = LinearRegression().fit(X, y)
    expected = 1 - (1 - lr.score(X, y)) * (n_obs - 1) / (n_obs - n_x - 1)
    rsq, _ = mlr(X, y, adj_r2=True, intercept=True)
    assert np.isclose(rsq, expected)


# ── dominance vs independent sklearn per-subset R2 ────────────────────────────

def test_dominance_individual_matches_sklearn_single_predictor_r2(regression_data):
    from sklearn.linear_model import LinearRegression
    X, y = regression_data
    dom = dominance(X, y, adj_r2=False)
    for i in range(X.shape[1]):
        lr = LinearRegression().fit(X[:, [i]], y)
        assert np.isclose(dom["individual"][0, i], lr.score(X[:, [i]], y))


def test_dominance_sum_matches_sklearn_full_model_r2(regression_data):
    from sklearn.linear_model import LinearRegression
    X, y = regression_data
    dom = dominance(X, y, adj_r2=False)
    lr = LinearRegression().fit(X, y)
    assert np.isclose(dom["sum"], lr.score(X, y))
    assert np.isclose(np.sum(dom["total"]), lr.score(X, y))


# ── pls / fast_pls1 vs a fresh, independently-constructed sklearn PLSRegression ─
# (nispace's own `pls()` already wraps sklearn's PLSRegression, so comparing
# `fast_pls1` against `pls()` would be circular -- instantiate sklearn
# directly here instead.)

def test_fast_pls1_matches_direct_sklearn_plsregression(regression_data):
    from sklearn.cross_decomposition import PLSRegression
    X, y = regression_data
    n_comp = 2
    out = fast_pls1(X, y, n_components=n_comp)
    reg = PLSRegression(n_components=n_comp).fit(X, y)
    assert np.isclose(out["r2"], reg.score(X, y))
    assert np.allclose(out["beta"], np.squeeze(reg.coef_.T))


def test_pls_wrapper_matches_direct_sklearn_plsregression(regression_data):
    from sklearn.cross_decomposition import PLSRegression
    X, y = regression_data
    n_comp = 2
    out = pls(X, y, n_components=n_comp)
    reg = PLSRegression(n_components=n_comp).fit(X, y)
    assert np.isclose(out["r2"], reg.score(X, y))
    assert np.allclose(out["beta"], np.squeeze(reg.coef_.T))


# ── pcr vs independent sklearn PCA + LinearRegression pipeline ──────────────

def test_pcr_matches_independent_sklearn_pipeline(regression_data):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    X, y = regression_data
    n_comp = 2
    pcs = PCA(n_components=n_comp).fit_transform(X)
    lr = LinearRegression().fit(pcs, y)
    out = pcr(X, y, adj_r2=False, n_components=n_comp)
    assert np.isclose(out["r2"], lr.score(pcs, y))


# ── lasso / ridge / elasticnet vs independently-constructed sklearn *CV ─────
# (these already call sklearn internally -- this still catches kwarg-
# forwarding/shape/sign bugs in nispace's wrapper by re-deriving the same
# fit from scratch with matched params, independent of the wrapper code.)

@pytest.fixture
def sparse_regression_data(rng):
    n, p = 100, 5
    X = rng.normal(size=(n, p))
    true_beta = np.array([1.0, 0.0, -0.5, 0.0, 0.3])
    y = X @ true_beta + rng.normal(scale=0.5, size=n)
    return X, y


def test_lasso_matches_direct_lassocv(sparse_regression_data):
    from sklearn.linear_model import LassoCV
    X, y = sparse_regression_data
    out = lasso(X, y, cv=5, seed=42)
    expected = LassoCV(cv=5, random_state=42).fit(X, y)
    assert np.isclose(out["alpha"], expected.alpha_)
    assert np.isclose(out["r2"], expected.score(X, y))
    assert np.allclose(out["beta"], expected.coef_)


def test_ridge_matches_direct_ridgecv(sparse_regression_data):
    from sklearn.linear_model import RidgeCV
    X, y = sparse_regression_data
    out = ridge(X, y, cv=5)
    expected = RidgeCV(cv=5).fit(X, y)
    assert np.isclose(out["alpha"], expected.alpha_)
    assert np.allclose(out["beta"], expected.coef_)


def test_elasticnet_matches_direct_elasticnetcv(sparse_regression_data):
    from sklearn.linear_model import ElasticNetCV
    X, y = sparse_regression_data
    out = elasticnet(X, y, cv=5, seed=42)
    expected = ElasticNetCV(cv=5, random_state=42).fit(X, y)
    assert np.isclose(out["alpha"], expected.alpha_)
    assert np.isclose(out["l1ratio"], expected.l1_ratio_)
    assert np.allclose(out["beta"], expected.coef_)


# ── mutualinfo vs direct sklearn mutual_info_regression ──────────────────────

def test_mutualinfo_matches_direct_sklearn(regression_data):
    from sklearn.feature_selection import mutual_info_regression
    X, y = regression_data
    x = X[:, 0]
    mi = mutualinfo(x, y, n_neighbors=3)
    expected = mutual_info_regression(x[:, None], y, discrete_features=False, n_neighbors=3)[0]
    assert np.isclose(mi, expected)