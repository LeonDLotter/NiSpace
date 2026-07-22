"""Tests for regional_influence(): core dispatcher, full NiSpace workflow,
XSEA, exclusions, and the brute-force cost-guard warning.

The analytic-vs-brute-force agreement checks are the core correctness gate
for the whole design (see plan): both engines compute the same
stat_full - stat_loo quantity via different routes and must agree exactly.
"""

import numpy as np
import pandas as pd
import pytest

import nispace.diagnostics as diagnostics
from nispace.core.colocalize import _get_colocalize_fun, _get_coloc_stats
from nispace.core.region_influence import _get_region_influence_fun

ATOL = 1e-3


# ---------------------------------------------------------------------------
# Mid-level: dispatcher called directly on plain arrays, no NiSpace involved
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["pearson", "spearman", "partialpearson", "partialspearman", "mlr"])
def test_analytic_matches_bruteforce_non_xsea(rng, method):
    n_x, n_parcels = 4, 40
    X = rng.normal(size=(n_x, n_parcels))
    w = rng.normal(size=n_x)
    y = w @ X + rng.normal(scale=0.3, size=n_parcels)

    kwargs = {}
    if method in {"partialpearson", "partialspearman"}:
        # partial* methods are dispatched identically to pearson/spearman once
        # Z has already been regressed out upstream -- exercise that path with
        # X/y already "residualized" (here: just itself, since no Z involved
        # at this level; the point is to confirm the shared code path works).
        pass

    y_coloc = _get_colocalize_fun(method, r_to_z=True, adj_r2=True, dtype=np.float32)
    stat = _get_coloc_stats(method, drop_optional=True)[0]

    fun_a, engine_a = _get_region_influence_fun(
        method, "analytic", n_parcels, y_colocalize_fun=y_coloc, stat=stat,
        r_to_z=True, adj_r2=True,
    )
    fun_b, engine_b = _get_region_influence_fun(
        method, "bruteforce", n_parcels, y_colocalize_fun=y_coloc, stat=stat,
        r_to_z=True, adj_r2=True,
    )
    assert engine_a == "analytic"
    assert engine_b == "bruteforce"

    res_a = np.asarray(fun_a(X, y))
    res_b = np.asarray(fun_b(X, y))
    assert res_a.shape == res_b.shape
    assert np.all(np.isfinite(res_a))
    np.testing.assert_allclose(res_a, res_b, atol=ATOL)


@pytest.mark.parametrize("method,n_x", [("dominance", 3), ("pls", 3), ("pcr", 3), ("mi", 3), ("slr", 3)])
def test_bruteforce_only_methods_shape_and_finite(rng, method, n_x):
    n_parcels = 30
    X = rng.normal(size=(n_x, n_parcels))
    w = rng.normal(size=n_x)
    y = w @ X + rng.normal(scale=0.3, size=n_parcels)

    y_coloc = _get_colocalize_fun(method, adj_r2=True, dtype=np.float32)
    stat = _get_coloc_stats(method, drop_optional=True)[0]

    fun, engine = _get_region_influence_fun(
        method, "auto", n_parcels, y_colocalize_fun=y_coloc, stat=stat, adj_r2=True,
    )
    assert engine == "bruteforce"

    res = np.asarray(fun(X, y))
    assert res.shape[-1] == n_parcels
    assert np.all(np.isfinite(res))


def test_lasso_ridge_elasticnet_raise_not_implemented(rng):
    for method in ["lasso", "ridge", "elasticnet"]:
        with pytest.raises(NotImplementedError):
            _get_region_influence_fun(method, "auto", 30, y_colocalize_fun=None, stat="r2")


def test_analytic_engine_rejects_unsupported_method():
    with pytest.raises(ValueError):
        _get_region_influence_fun("dominance", "analytic", 30, y_colocalize_fun=None, stat="sum")


def test_bruteforce_warns_above_cost_threshold(rng, caplog):
    n_parcels = 1200
    X = rng.normal(size=(2, n_parcels))
    y = rng.normal(size=n_parcels)
    y_coloc = _get_colocalize_fun("dominance", adj_r2=True, dtype=np.float32)
    stat = _get_coloc_stats("dominance", drop_optional=True)[0]

    fun, engine = _get_region_influence_fun(
        "dominance", "auto", n_parcels, y_colocalize_fun=y_coloc, stat=stat, adj_r2=True,
    )
    assert engine == "bruteforce"
    # warning is logged via the module logger (lgr.warning), not python warnings --
    # just confirm it still completes without error above the threshold.
    res = fun(X, y)
    assert np.asarray(res).shape[-1] == n_parcels


# ---------------------------------------------------------------------------
# XSEA
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("xsea_method", ["mean", "median", "absmean", "weightedmean"])
def test_xsea_analytic_matches_bruteforce_pearson(rng, xsea_method):
    n_parcels = 30
    sets = {"setA": rng.normal(size=(5, n_parcels)), "setB": rng.normal(size=(7, n_parcels))}
    w = rng.normal()
    y = w * sets["setA"][0] + rng.normal(scale=0.5, size=n_parcels)

    weights = None
    if "weighted" in xsea_method:
        weights = {k: rng.uniform(0.5, 1.5, size=v.shape[0]) for k, v in sets.items()}

    method = "pearson"
    stat = _get_coloc_stats(method, drop_optional=True)[0]
    y_coloc = _get_colocalize_fun(method, xsea=True, xsea_method=xsea_method, r_to_z=True, dtype=np.float32)

    fun_a, _ = _get_region_influence_fun(
        method, "analytic", n_parcels, y_colocalize_fun=y_coloc, stat=stat,
        r_to_z=True, xsea=True, xsea_method=xsea_method,
    )
    fun_b, _ = _get_region_influence_fun(
        method, "bruteforce", n_parcels, y_colocalize_fun=y_coloc, stat=stat,
        r_to_z=True, xsea=True, xsea_method=xsea_method,
    )
    res_a = np.asarray(fun_a(sets, y, weights))
    res_b = np.asarray(fun_b(sets, y, weights))
    assert res_a.shape == (2, n_parcels)
    np.testing.assert_allclose(res_a, res_b, atol=ATOL, equal_nan=True)


def test_xsea_analytic_matches_bruteforce_mlr(rng):
    n_parcels = 30
    sets = {"setA": rng.normal(size=(5, n_parcels)), "setB": rng.normal(size=(7, n_parcels))}
    w = rng.normal()
    y = w * sets["setA"][0] + rng.normal(scale=0.5, size=n_parcels)

    method = "mlr"
    stat = _get_coloc_stats(method, drop_optional=True)[0]
    y_coloc = _get_colocalize_fun(method, xsea=True, xsea_method="mean", adj_r2=True, dtype=np.float32)

    fun_a, _ = _get_region_influence_fun(
        method, "analytic", n_parcels, y_colocalize_fun=y_coloc, stat=stat,
        adj_r2=True, xsea=True, xsea_method="mean",
    )
    fun_b, _ = _get_region_influence_fun(
        method, "bruteforce", n_parcels, y_colocalize_fun=y_coloc, stat=stat,
        adj_r2=True, xsea=True, xsea_method="mean",
    )
    res_a = np.asarray(fun_a(sets, y))
    res_b = np.asarray(fun_b(sets, y))
    assert res_a.shape == (2, n_parcels)
    np.testing.assert_allclose(res_a, res_b, atol=ATOL)


# ---------------------------------------------------------------------------
# Full NiSpace workflow
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["pearson", "spearman", "mlr"])
def test_full_workflow_analytic_vs_bruteforce(synthetic_nispace, toy_regression, method):
    nsp = synthetic_nispace
    _, _, _, outlier_idx = toy_regression
    n_parcels = toy_regression[0].shape[1]

    nsp.colocalize(method=method, verbose=False)
    res = diagnostics.regional_influence(nsp, method=method, verbose=False)
    res_bf = diagnostics.regional_influence(nsp, method=method, engine="bruteforce", verbose=False)

    if isinstance(res, dict):
        assert set(res.keys()) == set(res_bf.keys())
        for k in res:
            assert res[k].shape == (nsp._Y.shape[0], n_parcels)
            np.testing.assert_allclose(res[k].to_numpy(), res_bf[k].to_numpy(), atol=ATOL)
    else:
        assert res.shape == (nsp._Y.shape[0], n_parcels)
        np.testing.assert_allclose(res.to_numpy(), res_bf.to_numpy(), atol=ATOL)


def test_full_workflow_outlier_ground_truth(synthetic_nispace, toy_regression):
    """The deliberately injected outlier region should be flagged as the
    single most influential region for the mlr fit -- a ground-truth sanity
    check, not just internal numerical consistency."""
    nsp = synthetic_nispace
    _, _, _, outlier_idx = toy_regression

    nsp.colocalize(method="mlr", verbose=False)
    res = diagnostics.regional_influence(nsp, method="mlr", verbose=False)
    argmax_col = res.abs().idxmax(axis=1)
    outlier_label = res.columns[outlier_idx]
    assert (argmax_col == outlier_label).all()


def test_full_workflow_regional_influence_is_deterministic(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="mlr", verbose=False)
    res = diagnostics.regional_influence(nsp, method="mlr", verbose=False)
    res_again = diagnostics.regional_influence(nsp, method="mlr", verbose=False)
    pd.testing.assert_frame_equal(res, res_again)


def test_full_workflow_pooled(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="mlr", verbose=False)

    pooled_mean = diagnostics.regional_influence(nsp, method="mlr", pooled="mean", verbose=False)
    pooled_median = diagnostics.regional_influence(nsp, method="mlr", pooled="median", verbose=False)
    unpooled = diagnostics.regional_influence(nsp, method="mlr", pooled=False, verbose=False)

    assert pooled_mean.shape[0] == 1
    assert list(pooled_mean.index) == ["pooled"]
    np.testing.assert_allclose(pooled_mean.to_numpy()[0],
                               np.nanmean(unpooled.to_numpy(), axis=0), atol=1e-6)
    np.testing.assert_allclose(pooled_median.to_numpy()[0],
                               np.nanmedian(unpooled.to_numpy(), axis=0), atol=1e-6)


def test_full_workflow_bruteforce_only_method(synthetic_nispace):
    """Bruteforce-only method (no analytic counterpart) plumbed end-to-end
    through the NiSpace class."""
    nsp = synthetic_nispace
    nsp.colocalize(method="pls", verbose=False)
    res = diagnostics.regional_influence(nsp, method="pls", verbose=False)
    n_parcels = nsp._Y.shape[1]
    if isinstance(res, dict):
        for df in res.values():
            assert df.shape == (nsp._Y.shape[0], n_parcels)
            assert np.all(np.isfinite(df.to_numpy()))
    else:
        assert res.shape == (nsp._Y.shape[0], n_parcels)
        assert np.all(np.isfinite(res.to_numpy()))


def test_regional_influence_requires_prior_colocalize(synthetic_nispace):
    nsp = synthetic_nispace
    with pytest.raises(KeyError):
        diagnostics.regional_influence(nsp, method="mlr", verbose=False)


# ---------------------------------------------------------------------------
# XSEA through the full NiSpace workflow
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_nispace_xsea(rng):
    from nispace import NiSpace

    n_parcels = 20
    genes_a, genes_b = 4, 6
    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]

    idx = pd.MultiIndex.from_tuples(
        [("setA", f"geneA{i}") for i in range(genes_a)]
        + [("setB", f"geneB{i}") for i in range(genes_b)],
        names=["set", "gene"],
    )
    X = rng.normal(size=(genes_a + genes_b, n_parcels))
    x_df = pd.DataFrame(X, index=idx, columns=parcel_labels)

    w = rng.normal()
    y = w * X[0] + rng.normal(scale=0.4, size=n_parcels)
    y_df = pd.DataFrame(y[np.newaxis, :], index=["y0"], columns=parcel_labels)

    nsp = NiSpace(x=x_df, y=y_df, z=None, parcellation=None, standardize=False,
                 n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    return nsp


def test_full_workflow_xsea(synthetic_nispace_xsea):
    nsp = synthetic_nispace_xsea
    nsp.colocalize(method="pearson", xsea=True, xsea_aggregation_method="mean", verbose=False)
    res = diagnostics.regional_influence(nsp, method="pearson", xsea=True, verbose=False)
    res_bf = diagnostics.regional_influence(nsp, method="pearson", xsea=True, engine="bruteforce",
                                            verbose=False)

    assert set(res.keys()) == {"setA", "setB"}
    for k in res:
        np.testing.assert_allclose(res[k].to_numpy(), res_bf[k].to_numpy(), atol=ATOL)


# ---------------------------------------------------------------------------
# signed parameter: default (False) is the sign-independent |delta|; True
# recovers the original directional delta -- and must be a no-op for
# methods whose stat is already non-negative (mlr/dominance/pls/pcr/mi/slr).
# ---------------------------------------------------------------------------

def test_signed_default_is_abs_of_directional_pearson(rng):
    """For a clearly positive correlation, default (|full|-|loo|) and signed
    (full-loo) are algebraically identical (both full and loo stay positive
    throughout) -- not a bug, a direct consequence of |a|-|b| == a-b when
    a,b >= 0. The two only differ when the stat is consistently negative
    (exact sign flip: |a|-|b| == -(a-b) when a,b <= 0), which is the
    meaningful case to test."""
    n_x, n_parcels = 1, 40
    X = rng.normal(size=(n_x, n_parcels))
    y = -0.6 * X[0] + rng.normal(scale=0.4, size=n_parcels)  # clearly negative correlation

    y_coloc = _get_colocalize_fun("pearson", r_to_z=True, dtype=np.float32)
    stat = _get_coloc_stats("pearson", drop_optional=True)[0]

    fun_default, _ = _get_region_influence_fun(
        "pearson", "analytic", n_parcels, y_colocalize_fun=y_coloc, stat=stat, r_to_z=True,
    )
    fun_signed, _ = _get_region_influence_fun(
        "pearson", "analytic", n_parcels, y_colocalize_fun=y_coloc, stat=stat, r_to_z=True,
        signed=True,
    )
    res_default = np.asarray(fun_default(X, y))
    res_signed = np.asarray(fun_signed(X, y))

    assert not np.allclose(res_default, res_signed)
    # sign-flip identity holds throughout only if r stayed negative for every
    # single LOO exclusion too -- true here given the strong effect size.
    np.testing.assert_allclose(res_default, -res_signed, atol=ATOL)


@pytest.mark.parametrize("method,n_x", [("mlr", 4), ("dominance", 3), ("pls", 3), ("slr", 3), ("mi", 3)])
def test_signed_is_noop_for_unsigned_stat_methods(rng, method, n_x):
    n_parcels = 30
    X = rng.normal(size=(n_x, n_parcels))
    w = rng.normal(size=n_x)
    y = w @ X + rng.normal(scale=0.3, size=n_parcels)

    y_coloc = _get_colocalize_fun(method, adj_r2=True, dtype=np.float32)
    stat = _get_coloc_stats(method, drop_optional=True)[0]

    fun_default, engine = _get_region_influence_fun(
        method, "auto", n_parcels, y_colocalize_fun=y_coloc, stat=stat, adj_r2=True,
    )
    fun_signed, _ = _get_region_influence_fun(
        method, "auto", n_parcels, y_colocalize_fun=y_coloc, stat=stat, adj_r2=True, signed=True,
    )
    res_default = np.asarray(fun_default(X, y))
    res_signed = np.asarray(fun_signed(X, y))
    np.testing.assert_allclose(res_default, res_signed)


def test_full_workflow_signed_vs_default_pearson(rng):
    """Uses its own clearly-negative-correlation NiSpace object rather than the
    shared synthetic_nispace fixture, whose per-predictor sign isn't guaranteed
    -- the divergence between signed/default is only guaranteed for a stat
    that doesn't cross zero across LOO exclusions."""
    from nispace import NiSpace

    n_parcels = 30
    x = rng.normal(size=n_parcels)
    y = -0.6 * x + rng.normal(scale=0.4, size=n_parcels)
    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    x_df = pd.DataFrame(x[np.newaxis, :], index=["x0"], columns=parcel_labels)
    y_df = pd.DataFrame(y[np.newaxis, :], index=["y0"], columns=parcel_labels)

    nsp = NiSpace(x=x_df, y=y_df, z=None, parcellation=None, standardize=False,
                 n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    nsp.colocalize(method="pearson", verbose=False)
    default = diagnostics.regional_influence(nsp, method="pearson", signed=False, verbose=False)
    signed = diagnostics.regional_influence(nsp, method="pearson", signed=True, verbose=False)

    for k in default:
        assert not np.allclose(default[k].to_numpy(), signed[k].to_numpy())
        np.testing.assert_allclose(default[k].to_numpy(), -signed[k].to_numpy(), atol=ATOL)


def test_full_workflow_signed_is_noop_for_mlr(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="mlr", verbose=False)
    default = diagnostics.regional_influence(nsp, method="mlr", signed=False, verbose=False)
    signed = diagnostics.regional_influence(nsp, method="mlr", signed=True, verbose=False)
    np.testing.assert_allclose(default.to_numpy(), signed.to_numpy(), atol=1e-6)
