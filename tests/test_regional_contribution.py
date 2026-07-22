"""Tests for regional_contribution(): core decomposition math, quadrant
labels, XSEA, and the full NiSpace workflow.

The core correctness invariant (mirrored throughout): mean(contribution) ==
rho exactly, for whatever data is actually being correlated (raw values for
pearson, ranks for spearman, Z-residualized -- and for spearman, also
Z-ranked -- for partial*). This doubles as a regression guard for the two
_rank_regress/Z-handling bugs fixed alongside this feature (see
test_rank_regress.py, test_colocalize.py) -- either bug reappearing would
break this equality for partialpearson/partialspearman specifically.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from nispace import NiSpace
import nispace.diagnostics as diagnostics
from nispace.core.region_contribution import _get_region_contribution_fun, _CONTRIBUTION_METHODS
from nispace.stats.coloc import pearson

ATOL = 1e-4


def _zscore(a):
    return (a - a.mean()) / a.std()


# ---------------------------------------------------------------------------
# Mid-level: closure called directly on plain arrays, no NiSpace involved
# ---------------------------------------------------------------------------

def test_contribution_matches_manual_zx_zy_pearson(rng):
    n_x, n_parcels = 1, 40
    X = rng.normal(size=(n_x, n_parcels))
    y = 0.6 * X[0] + rng.normal(scale=0.5, size=n_parcels)

    fun = _get_region_contribution_fun("pearson", dtype=np.float32)
    contrib, quadrant = fun(X, y)

    zx, zy = _zscore(X[0]), _zscore(y)
    manual = zx * zy
    np.testing.assert_allclose(contrib[0], manual, atol=ATOL)


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_mean_contribution_equals_rho(rng, method):
    n_x, n_parcels = 3, 50
    X = rng.normal(size=(n_x, n_parcels))
    w = rng.normal(size=n_x)
    y = w @ X + rng.normal(scale=0.4, size=n_parcels)

    if method == "spearman":
        X_in = np.array([rankdata(x) for x in X])
        y_in = rankdata(y)
    else:
        X_in, y_in = X, y

    fun = _get_region_contribution_fun(method, dtype=np.float32)
    contrib, _ = fun(X_in, y_in)

    for i_x in range(n_x):
        r = pearson(X_in[i_x], y_in)
        np.testing.assert_allclose(contrib[i_x].mean(), r, atol=ATOL)


def test_mean_contribution_equals_rho_with_mismatched_nan_patterns(rng):
    """Regression test for a real bug caught by live verification against
    real data, not synthetic tests (which had no NaNs): when x has NaNs at
    positions where y doesn't (or vice versa), zx/zy must be standardized
    using mean/std computed over the SAME shared mask, matching how
    pearson(x[mask], y[mask]) itself is computed -- zscoring x and y
    independently via nanmean/nanstd on each array separately uses a
    different effective sample whenever their NaN patterns differ, and
    mean(contribution) silently drifts away from the true rho."""
    n_parcels = 60
    x = rng.normal(size=n_parcels)
    y = 0.5 * x + rng.normal(scale=0.5, size=n_parcels)

    x_nan = x.copy()
    x_nan[[5, 40]] = np.nan  # x missing 2 parcels y has values for

    fun = _get_region_contribution_fun("pearson", dtype=np.float32)
    contrib, _ = fun(x_nan[np.newaxis, :], y)

    mask = ~np.isnan(x_nan)
    r_matched_mask = pearson(x_nan[mask], y[mask])
    np.testing.assert_allclose(np.nanmean(contrib[0]), r_matched_mask, atol=ATOL)


def test_quadrant_labels_correct(rng):
    n_x, n_parcels = 1, 60
    X = rng.normal(size=(n_x, n_parcels))
    y = 0.5 * X[0] + rng.normal(scale=0.6, size=n_parcels)

    fun = _get_region_contribution_fun("pearson", dtype=np.float32)
    contrib, quadrant = fun(X, y)

    zx, zy = _zscore(X[0]), _zscore(y)
    for i in range(n_parcels):
        if zx[i] > 0 and zy[i] > 0:
            assert quadrant[0, i] == "high_high"
        elif zx[i] < 0 and zy[i] < 0:
            assert quadrant[0, i] == "low_low"
        else:
            assert quadrant[0, i] == "discordant"


@pytest.mark.parametrize("method", ["mlr", "dominance", "pls", "mi", "slr"])
def test_unsupported_method_raises(method):
    with pytest.raises(ValueError):
        _get_region_contribution_fun(method, dtype=np.float32)


# ---------------------------------------------------------------------------
# XSEA (plain dict input)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("xsea_method", ["mean", "median", "absmean", "weightedmean"])
def test_xsea_contribution_shapes_and_quadrant(rng, xsea_method):
    n_parcels = 30
    sets = {"setA": rng.normal(size=(5, n_parcels)), "setB": rng.normal(size=(7, n_parcels))}
    y = sets["setA"][0] * 0.7 + rng.normal(scale=0.4, size=n_parcels)

    weights = None
    if "weighted" in xsea_method:
        weights = {k: rng.uniform(0.5, 1.5, size=v.shape[0]) for k, v in sets.items()}

    fun = _get_region_contribution_fun("pearson", dtype=np.float32, xsea=True, xsea_method=xsea_method)
    contrib, quadrant = fun(sets, y, weights)

    assert contrib.shape == (2, n_parcels)
    assert quadrant.shape == (2, n_parcels)
    assert set(np.unique(quadrant)) <= {"high_high", "low_low", "discordant"}


# ---------------------------------------------------------------------------
# Full NiSpace workflow
# ---------------------------------------------------------------------------

def test_full_workflow_mean_equals_rho(synthetic_nispace):
    """mean(contribution) equals the *raw* rho -- colocalize() stores the
    Fisher-z-transformed rho by default (r_to_z=True), so compare against
    tanh(stored value), not the stored value directly."""
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)
    contrib = diagnostics.regional_contribution(nsp, method="pearson", verbose=False)

    coloc = nsp.get_colocalizations(method="pearson")
    for x_lab in contrib:
        for i_y, y_lab in enumerate(contrib[x_lab].index):
            r = np.tanh(coloc.loc[y_lab, x_lab])
            np.testing.assert_allclose(contrib[x_lab].iloc[i_y].mean(), r, atol=ATOL)


def test_full_workflow_default_is_contribution_not_quadrant(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)

    default = diagnostics.regional_contribution(nsp, method="pearson", verbose=False)
    explicit_contrib = diagnostics.regional_contribution(nsp, method="pearson", quadrant=False,
                                                          verbose=False)
    for k in default:
        pd.testing.assert_frame_equal(default[k], explicit_contrib[k])
        assert default[k].to_numpy().dtype != object  # numeric, not quadrant labels


def test_full_workflow_quadrant_opt_in(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)

    quadrant = diagnostics.regional_contribution(nsp, method="pearson", quadrant=True, verbose=False)
    for k in quadrant:
        vals = set(np.unique(quadrant[k].to_numpy()))
        assert vals <= {"high_high", "low_low", "discordant"}


def test_regional_contribution_requires_prior_colocalize(synthetic_nispace):
    nsp = synthetic_nispace
    with pytest.raises(KeyError):
        diagnostics.regional_contribution(nsp, method="pearson", verbose=False)


def test_regional_contribution_rejects_unsupported_method(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="mlr", verbose=False)
    with pytest.raises(ValueError):
        diagnostics.regional_contribution(nsp, method="mlr", verbose=False)


def test_full_workflow_pooled_contribution(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)

    unpooled = diagnostics.regional_contribution(nsp, method="pearson", pooled=False, verbose=False)
    pooled_mean = diagnostics.regional_contribution(nsp, method="pearson", pooled="mean", verbose=False)
    pooled_median = diagnostics.regional_contribution(nsp, method="pearson", pooled="median",
                                                       verbose=False)

    for k in unpooled:
        assert pooled_mean[k].shape[0] == 1
        assert list(pooled_mean[k].index) == ["pooled"]
        np.testing.assert_allclose(pooled_mean[k].to_numpy()[0],
                                   np.nanmean(unpooled[k].to_numpy(), axis=0), atol=1e-6)
        np.testing.assert_allclose(pooled_median[k].to_numpy()[0],
                                   np.nanmedian(unpooled[k].to_numpy(), axis=0), atol=1e-6)


def test_full_workflow_pooled_quadrant_raises(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)

    with pytest.raises(ValueError):
        diagnostics.regional_contribution(nsp, method="pearson", quadrant=True, pooled="mean",
                                          verbose=False)


def test_full_workflow_partialspearman_with_z_matches_standard_formula(rng):
    """Regression guard: if either _rank_regress bug (rank silently discarded
    when regress also applies; Z not ranked for partial spearman) reappears,
    mean(contribution) would stop matching the standard partial-Spearman
    formula for this method specifically."""
    n_parcels = 200
    x = rng.normal(size=n_parcels)
    z = rng.exponential(size=n_parcels)
    y = 0.5 * x + 0.6 * z + rng.normal(scale=0.8, size=n_parcels)

    parcel_labels = [f"p{i}" for i in range(n_parcels)]
    x_df = pd.DataFrame(x[np.newaxis, :], index=["x0"], columns=parcel_labels)
    y_df = pd.DataFrame(y[np.newaxis, :], index=["y0"], columns=parcel_labels)
    z_df = pd.DataFrame(z[np.newaxis, :], index=["z0"], columns=parcel_labels)

    nsp = NiSpace(x=x_df, y=y_df, z=z_df, parcellation=None, standardize=False,
                 n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    nsp.colocalize(method="partialspearman", verbose=False)
    contrib = diagnostics.regional_contribution(nsp, method="partialspearman", verbose=False)
    mean_contrib = contrib["x0"].iloc[0].mean()

    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    standard_partial_spearman = (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))

    np.testing.assert_allclose(mean_contrib, standard_partial_spearman, atol=1e-3)


# ---------------------------------------------------------------------------
# XSEA through the full NiSpace workflow
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_nispace_xsea(rng):
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

    contrib = diagnostics.regional_contribution(nsp, method="pearson", xsea=True, verbose=False)
    quadrant = diagnostics.regional_contribution(nsp, method="pearson", xsea=True, quadrant=True,
                                                  verbose=False)

    assert set(contrib.keys()) == {"setA", "setB"}
    for k in contrib:
        assert contrib[k].shape == quadrant[k].shape
