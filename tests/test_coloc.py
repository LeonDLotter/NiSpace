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


def _align_sign(nispace_vector, reference_vector):
    # nispace intentionally uses its own (Y-blind "module eigengene") sign
    # convention, not sklearn's -- align via dominant-direction dot product
    # rather than assume equality, here just to compare *magnitude*.
    return 1.0 if np.dot(nispace_vector, reference_vector) >= 0 else -1.0


def test_fast_pls1_weight_matches_sklearn_x_weights_magnitude(regression_data):
    from sklearn.cross_decomposition import PLSRegression
    X, y = regression_data
    n_comp = 2
    out = fast_pls1(X, y, n_components=n_comp)
    reg = PLSRegression(n_components=n_comp).fit(X, y)
    sk_w0 = reg.x_weights_[:, 0]
    sign = _align_sign(out["weight"], sk_w0)
    assert np.isclose(np.linalg.norm(out["weight"]), 1.0)
    assert np.allclose(out["weight"], sign * sk_w0, atol=1e-6)


def _median_vote_sign(X, y):
    """Reference (test-independent) implementation of the median-vote sign convention:
    positive iff the median per-predictor Pearson correlation with y is positive."""
    r = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
    return 1.0 if np.median(r) >= 0.0 else -1.0


def test_fast_pls1_weight_and_score_r_use_median_vote_sign_convention(regression_data):
    # component 1's score must be oriented so it's positive iff the median per-predictor
    # correlation with y is positive -- regardless of sklearn's own (different, arbitrary)
    # sign choice.
    X, y = regression_data
    out = fast_pls1(X, y, n_components=2)
    expected_sign = _median_vote_sign(X, y)
    assert (out["score_r"] >= 0) == (expected_sign >= 0)
    # score_r's sign must be consistent with weight (same underlying component)
    xc = X - X.mean(axis=0)
    xc /= xc.std(axis=0, ddof=1)
    yc = y - y.mean()
    t1 = xc @ out["weight"]
    assert np.isclose(out["score_r"], np.corrcoef(t1, yc)[0, 1], atol=1e-6)


def test_fast_pls1_score_r_sign_follows_majority_even_against_strongest_predictor(rng):
    # A genuinely mixed-sign case: 3 predictors positively correlated with y, 2 more
    # strongly negatively correlated -- majority (3/5) is positive, so the median-vote
    # convention should report a positive score_r even though the single strongest
    # individual predictor is negative (guards against silently reverting to a
    # magnitude/sum-dominated convention, which could flip this case).
    n = 200
    y = rng.normal(size=n)
    pos = [y * 0.15 + rng.normal(scale=1.0, size=n) for _ in range(3)]
    neg = [-y * 0.6 + rng.normal(scale=1.0, size=n) for _ in range(2)]
    X = np.column_stack(pos + neg)
    out = fast_pls1(X, y, n_components=1)
    assert _median_vote_sign(X, y) > 0.0  # sanity check on the constructed data itself
    assert out["score_r"] > 0.0


def test_fast_pls1_score_r_squared_equals_r2_for_single_component(regression_data):
    X, y = regression_data
    out = fast_pls1(X, y, n_components=1)
    assert np.isclose(out["score_r"] ** 2, out["r2"])


def test_fast_pls1_score_r_squared_diverges_from_r2_for_multi_component(regression_data):
    # NOT a bug: r2 reflects the full multi-component fit, score_r only component 1 --
    # guards against this later being "fixed" into forced equality.
    X, y = regression_data
    out = fast_pls1(X, y, n_components=2)
    assert not np.isclose(out["score_r"] ** 2, out["r2"])


def test_fast_pls1_component1_invariant_to_n_components(regression_data):
    X, y = regression_data
    out1 = fast_pls1(X, y, n_components=1)
    out2 = fast_pls1(X, y, n_components=2)
    assert np.allclose(out1["weight"], out2["weight"])
    assert np.isclose(out1["score_r"], out2["score_r"])


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


def test_pcr_score_r_squared_equals_unadjusted_r2_for_single_component(regression_data):
    # score_r is always component-1-only; with n_components=1 and adj_r2=False, r2 IS
    # the single-predictor R2, so score_r**2 == r2 exactly (same identity as pls).
    X, y = regression_data
    out = pcr(X, y, adj_r2=False, n_components=1)
    assert np.isclose(out["score_r"] ** 2, out["r2"])


def test_pcr_score_r_uses_median_vote_sign_convention(regression_data):
    from sklearn.decomposition import PCA
    X, y = regression_data
    out = pcr(X, y, n_components=2)
    pc1 = PCA(n_components=2).fit_transform(X)[:, 0]
    sign = _median_vote_sign(X, y)
    expected_r = np.corrcoef(sign * pc1, y)[0, 1]
    assert np.isclose(out["score_r"], expected_r, atol=1e-6)


def test_pcr_score_r_sign_follows_majority_even_against_strongest_predictor(rng):
    # same mixed-sign construction as the pls version above: majority (3/5) positive should
    # win even though the single strongest predictor is negative.
    n = 200
    y = rng.normal(size=n)
    pos = [y * 0.15 + rng.normal(scale=1.0, size=n) for _ in range(3)]
    neg = [-y * 0.6 + rng.normal(scale=1.0, size=n) for _ in range(2)]
    X = np.column_stack(pos + neg)
    out = pcr(X, y, n_components=1)
    assert _median_vote_sign(X, y) > 0.0
    assert out["score_r"] > 0.0


def test_pcr_score_r_component1_invariant_to_n_components(regression_data):
    X, y = regression_data
    out1 = pcr(X, y, n_components=1)
    out2 = pcr(X, y, n_components=2)
    assert np.isclose(out1["score_r"], out2["score_r"])


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