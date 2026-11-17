"""Tests for NiSpace.reduce_x() and its core/reduce_x.py backend
(_reduce_dimensions): mean/median (plain + set-grouped + weighted), PCA/ICA,
min_ev component selection, and error paths.

PCA is ground-truthed against a direct, independently-instantiated
sklearn.decomposition.PCA (not through nispace's own wrapper). ICA has no
canonical single "right answer" (FastICA's solution is subject to a random
rotation/sign ambiguity across implementations/seeds), so it's checked only
for shape/well-formedness. Factor analysis ("fa", statsmodels extraction + NiSpace's
Kaiser-normalized varimax/promax) is ground-truthed against reference values from
factor_analyzer 0.5.1, the implementation it replaced, and against rotation invariants.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from nispace import NiSpace


@pytest.fixture
def nsp_reduce(rng):
    n_x, n_parcels = 6, 25
    X = rng.normal(size=(n_x, n_parcels))
    y = X[0] + rng.normal(scale=0.2, size=n_parcels)
    parcel_labels = [f"p{i}" for i in range(n_parcels)]
    x_df = pd.DataFrame(X, index=[f"x{i}" for i in range(n_x)], columns=parcel_labels)
    y_df = pd.DataFrame(y[None, :], index=["y0"], columns=parcel_labels)
    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    return nsp, X


@pytest.fixture
def nsp_reduce_weighted(rng):
    """X with 'set'/'weight' MultiIndex levels, for mean_by_set/weighted_mean."""
    n_parcels = 25
    idx = pd.MultiIndex.from_tuples(
        [("setA", "g0", 1.0), ("setA", "g1", 3.0), ("setB", "g2", 1.0), ("setB", "g3", 1.0)],
        names=["set", "gene", "weight"],
    )
    X = rng.normal(size=(4, n_parcels))
    parcel_labels = [f"p{i}" for i in range(n_parcels)]
    x_df = pd.DataFrame(X, index=idx, columns=parcel_labels)
    y = X[0] + rng.normal(scale=0.2, size=n_parcels)
    y_df = pd.DataFrame(y[None, :], index=["y0"], columns=parcel_labels)
    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    return nsp, X


# ── mean / median ─────────────────────────────────────────────────────────

def test_reduce_x_mean_matches_manual_mean(nsp_reduce):
    nsp, X = nsp_reduce
    out = nsp.reduce_x("mean", verbose=False)
    np.testing.assert_allclose(out.to_numpy().ravel(), X.mean(axis=0), atol=1e-5)


def test_reduce_x_median_matches_manual_median(nsp_reduce):
    nsp, X = nsp_reduce
    out = nsp.reduce_x("median", verbose=False)
    np.testing.assert_allclose(out.to_numpy().ravel(), np.median(X, axis=0), atol=1e-5)


def test_reduce_x_mean_by_set_groups_correctly(nsp_reduce_weighted):
    nsp, X = nsp_reduce_weighted
    out = nsp.reduce_x("mean", mean_by_set=True, verbose=False)
    assert set(out.index) == {"setA", "setB"}
    np.testing.assert_allclose(out.loc["setA"].to_numpy(), X[0:2].mean(axis=0), atol=1e-5)
    np.testing.assert_allclose(out.loc["setB"].to_numpy(), X[2:4].mean(axis=0), atol=1e-5)


def test_reduce_x_weighted_mean_matches_manual_weighted_average(nsp_reduce_weighted):
    nsp, X = nsp_reduce_weighted
    out = nsp.reduce_x("mean", mean_by_set=True, weighted_mean=True, verbose=False)
    expected_setA = (X[0] * 1.0 + X[1] * 3.0) / 4.0
    np.testing.assert_allclose(out.loc["setA"].to_numpy(), expected_setA, atol=1e-5)


# ── PCA ───────────────────────────────────────────────────────────────────

def test_reduce_x_pca_ev_matches_sklearn(nsp_reduce):
    nsp, X = nsp_reduce
    _, ev, _ = nsp.reduce_x("pca", n_components=3, verbose=False)

    data = X.T  # api.py transposes to (n_parcels, n_maps) before PCA
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(data)
    ev_expected = np.var(pcs, axis=0, ddof=0) / np.sum(np.var(data, axis=0, ddof=0))
    np.testing.assert_allclose(ev.to_numpy(), ev_expected, atol=1e-4)


def test_reduce_x_pca_components_correlate_with_sklearn(nsp_reduce):
    nsp, X = nsp_reduce
    x_pca, _, _ = nsp.reduce_x("pca", n_components=3, verbose=False)

    data = X.T
    pcs = PCA(n_components=3).fit_transform(data)
    # PCA components can differ in sign across implementations -- compare abs corr
    for i in range(3):
        r = np.corrcoef(x_pca.to_numpy()[i], pcs.T[i])[0, 1]
        assert abs(r) > 0.999


def test_reduce_x_pca_min_ev_selects_sufficient_components(nsp_reduce):
    nsp, X = nsp_reduce
    x_pca, ev, _ = nsp.reduce_x("pca", min_ev=0.9, verbose=False)
    assert ev.sum() >= 0.9
    # one fewer component must NOT reach the threshold (min_ev picks the smallest sufficient n)
    if x_pca.shape[0] > 1:
        assert ev.iloc[:-1].sum() < 0.9


def test_reduce_x_pca_loadings_shape(nsp_reduce):
    nsp, X = nsp_reduce
    _, _, loadings = nsp.reduce_x("pca", n_components=2, verbose=False)
    assert loadings.shape == (X.shape[0], 2)


# ── ICA ───────────────────────────────────────────────────────────────────

def test_reduce_x_ica_runs_with_no_explained_variance(nsp_reduce):
    nsp, X = nsp_reduce
    x_ica, ev, loadings = nsp.reduce_x("ica", n_components=2, seed=0, verbose=False)
    assert x_ica.shape == (2, X.shape[1])
    assert ev.isna().all()
    assert np.isfinite(loadings.to_numpy()).all()


# ── error paths ───────────────────────────────────────────────────────────

def test_reduce_x_unknown_reduction_logs_error_and_returns_none(nsp_reduce, caplog):
    """Regression test: reduce_x() used to call lgr.error(msg, ValueError) for
    an unrecognized reduction name -- passing an exception class as a
    %-style logging arg, which crashed logging's own string formatting
    (visible as a spurious "Logging error"/TypeError traceback on every
    call, even though the function itself didn't raise). Fixed by dropping
    the stray second argument. Per the docstring, this path must log an
    error and return None, not raise."""
    result = nsp_reduce[0].reduce_x("not_a_real_reduction", verbose=False)
    assert result is None


def test_reduce_x_single_map_raises(rng):
    n_parcels = 20
    x_df = pd.DataFrame(rng.normal(size=(1, n_parcels)), index=["x0"],
                        columns=[f"p{i}" for i in range(n_parcels)])
    y_df = pd.DataFrame(rng.normal(size=(1, n_parcels)), index=["y0"],
                        columns=[f"p{i}" for i in range(n_parcels)])
    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    with pytest.raises(ValueError):
        nsp.reduce_x("mean", verbose=False)


# ── FA (statsmodels extraction, own rotations) ───────────────────────────
# Replaced the optional factor_analyzer dependency (unmaintained; 0.5.1 breaks on
# scikit-learn >= 1.8). Reference values below: factor_analyzer 0.5.1,
# FactorAnalyzer(n_factors=2, method=..., rotation=...) on _fa_data(), loadings_,
# get_factor_variance()[1] and transform()[:3], rounded to 4 decimals.

_FA_REF = {
    ("minres", "promax"): dict(
        loadings=[[0.7746, 0.0278], [0.9617, -0.3021], [0.5604, 0.2445],
                  [-0.0085, 0.718], [-0.1905, 0.8184], [0.3365, 0.482]],
        ev=[0.3314, 0.2616],
        scores=[[-0.2104, -0.563], [1.2868, 0.8558], [-2.4127, -0.9913]]),
    ("ml", "varimax"): dict(
        loadings=[[0.7558, 0.2027], [0.8686, -0.0774], [0.6068, 0.364],
                  [0.1487, 0.7067], [0.0058, 0.7423], [0.4402, 0.5473]],
        ev=[0.3183, 0.2549],
        scores=[[-0.0579, -0.5868], [1.1736, 0.6077], [-2.3739, -0.4685]]),
}


def _fa_data():
    rng = np.random.default_rng(42)
    F = rng.normal(size=(80, 2))
    L = np.array([[.8, 0], [.7, .1], [.6, .2], [0, .8], [.1, .7], [.3, .5]])
    return F @ L.T + rng.normal(scale=.6, size=(80, 6))


@pytest.mark.parametrize("method,rotation", list(_FA_REF))
def test_fa_matches_factor_analyzer_reference(method, rotation):
    from nispace.core.reduce_x import _reduce_dimensions
    ref = _FA_REF[(method, rotation)]
    scores, ev, loadings = _reduce_dimensions(_fa_data(), "fa", n_components=2,
                                              fa_method=method, fa_rotation=rotation)
    # different optimizers (statsmodels vs scipy minimize): agree to ~1e-3
    np.testing.assert_allclose(loadings, ref["loadings"], atol=2e-3)
    np.testing.assert_allclose(ev, ref["ev"], atol=2e-3)
    np.testing.assert_allclose(scores[:3], ref["scores"], atol=3e-3)


@pytest.mark.parametrize("method", ["minres", "ml"])
def test_fa_rotations_preserve_common_variance(method):
    """Orthogonal rotation: L_r L_r' == L L'; oblique (promax): P phi P' == L L'."""
    from nispace.core.reduce_x import _fit_fa, _kaiser_varimax, _kaiser_promax
    D = _fa_data()
    L0, _ = _fit_fa(D, 2, method, None)
    # rotate one fitted solution (separate ML fits differ by ~1e-6)
    Lv = _kaiser_varimax(L0)
    Lp, phi_p = _kaiser_promax(L0)
    np.testing.assert_allclose(Lv @ Lv.T, L0 @ L0.T, atol=1e-10)
    np.testing.assert_allclose(Lp @ phi_p @ Lp.T, L0 @ L0.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(phi_p), 1, atol=1e-10)
    assert _fit_fa(D, 2, method, "varimax")[1] is None
    # sign convention and variance order of the fitted (rotated) solutions
    for L in (L0, _fit_fa(D, 2, method, "varimax")[0], _fit_fa(D, 2, method, "promax")[0]):
        assert (L.sum(axis=0) >= 0).all()
        ss = (L ** 2).sum(axis=0)
        assert (np.diff(ss) <= 1e-12).all()


def test_fa_unrotated_minres_equals_statsmodels_pa():
    from statsmodels.multivariate.factor import Factor
    from nispace.core.reduce_x import _fit_fa
    D = _fa_data()
    L, _ = _fit_fa(D, 2, "minres", None)
    L_sm = Factor(D, n_factor=2, method="pa", smc=True).fit(maxiter=1000).loadings
    L_sm = L_sm * np.sign(L_sm.sum(axis=0))
    np.testing.assert_allclose(L, L_sm, atol=1e-12)


def test_reduce_x_fa_runs_and_returns_expected_shapes(nsp_reduce):
    nsp, X = nsp_reduce
    x_fa, ev, loadings = nsp.reduce_x("fa", n_components=2, verbose=False)
    assert x_fa.shape == (2, X.shape[1])
    assert ev.shape == (2,)
    assert loadings.shape == (X.shape[0], 2)
    assert np.isfinite(x_fa.to_numpy()).all()


def test_reduce_x_fa_min_ev_picks_smallest_sufficient_number():
    from nispace.core.reduce_x import _reduce_dimensions, _fit_fa
    D = _fa_data()
    L, _ = _fit_fa(D, D.shape[1], "minres", None)
    cum = np.cumsum((L ** 2).sum(axis=0)) / D.shape[1]
    target = (cum[0] + cum[1]) / 2   # between the 1- and 2-factor cumulative EV
    scores, ev, loadings = _reduce_dimensions(D, "fa", min_ev=target)
    assert scores.shape[1] == 2


@pytest.mark.parametrize("kwargs", [{"fa_method": "principal"}, {"fa_rotation": "oblimin"}])
def test_reduce_x_fa_unsupported_options_raise(nsp_reduce, kwargs):
    with pytest.raises(ValueError):
        nsp_reduce[0].reduce_x("fa", n_components=2, verbose=False, **kwargs)
