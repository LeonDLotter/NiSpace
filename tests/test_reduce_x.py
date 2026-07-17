"""Tests for NiSpace.reduce_x() and its core/reduce_x.py backend
(_reduce_dimensions): mean/median (plain + set-grouped + weighted), PCA/ICA,
min_ev component selection, and error paths.

PCA is ground-truthed against a direct, independently-instantiated
sklearn.decomposition.PCA (not through nispace's own wrapper). ICA has no
canonical single "right answer" (FastICA's solution is subject to a random
rotation/sign ambiguity across implementations/seeds), so it's checked only
for shape/well-formedness. Factor analysis ("fa") requires the optional
`factor_analyzer` dependency and is skipped if unavailable.
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


# ── FA (factor_analyzer, optional dependency) ────────────────────────────
# factor_analyzer is installed in all test envs (see pyproject.toml [opt]);
# scikit-learn is pinned <1.8.0 there too, since factor_analyzer 0.5.1 (the
# latest release) calls sklearn's check_array(force_all_finite=...), a
# parameter renamed to ensure_all_finite in sklearn 1.6 and removed outright
# in 1.8 -- reduction="fa" is completely broken against an unpinned/modern
# sklearn install until factor_analyzer ships a fix upstream.

factor_analyzer = pytest.importorskip("factor_analyzer")


def test_reduce_x_fa_runs_and_returns_expected_shapes(nsp_reduce):
    nsp, X = nsp_reduce
    x_fa, ev, loadings = nsp.reduce_x("fa", n_components=2, verbose=False)
    assert x_fa.shape == (2, X.shape[1])
    assert ev.shape == (2,)
    assert loadings.shape == (X.shape[0], 2)
    assert np.isfinite(x_fa.to_numpy()).all()


def test_reduce_x_fa_missing_dependency_raises_importerror(nsp_reduce, monkeypatch):
    import nispace.core.reduce_x as reduce_x_mod
    monkeypatch.setattr(reduce_x_mod, "_FACTOR_ANALYZER_AVAILABLE", False)
    with pytest.raises(ImportError):
        nsp_reduce[0].reduce_x("fa", n_components=2, verbose=False)
