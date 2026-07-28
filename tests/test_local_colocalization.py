"""Tests for local_colocalization(): the "searchlight" k-NN windowed
colocalization, its auto-detection of the colocalize()/transform_y()/
permute() pathway (no separate group_local_colocalization() variant), and
its per-region null/p-values.

Return shape: always a single dict with fixed keys {"stat_type", "mc_method",
"pooled", "stat", "p", "p_corr", "settings"} -- matches NiSpace.get_within_region_
correlations()'s return convention. "p"/"p_corr"/"mc_method" are None when
null=False; "p_corr" is also None when mc_method=None (raw p is always
computed and returned whenever a null exists). "pooled" is always False when
null=False, or when the Y axis has only one row to begin with.

Ground-truth invariant used throughout: a k=n_parcels window (the whole
brain) must reproduce the plain colocalize() result exactly, since the
"neighborhood" is then just everything -- this doubles as the main
correctness check for the windowing/aggregation machinery itself.
"""

import numpy as np
import pandas as pd
import pytest

from nispace import NiSpace
import nispace.diagnostics as diagnostics

ATOL = 1e-4


def _index_dist_mat(n_parcels):
    """Simple 1D index-distance matrix (|i-j|), good enough as a stand-in
    spatial structure for parcellation-free synthetic NiSpace objects."""
    coords = np.arange(n_parcels, dtype=float)
    return np.abs(coords[:, np.newaxis] - coords[np.newaxis, :])


# ---------------------------------------------------------------------------
# Return-shape contract: always the same dict keys, regardless of null/method
# ---------------------------------------------------------------------------

def test_return_shape_keys_always_present(synthetic_nispace):
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)
    nsp.colocalize(method="pearson", verbose=False)

    out = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, null=False, verbose=False)
    assert set(out.keys()) == {"stat_type", "mc_method", "pooled", "stat", "p", "p_corr", "settings"}
    assert out["stat_type"] == "rho"
    assert out["mc_method"] is None
    assert out["pooled"] is False
    assert out["p"] is None
    assert out["p_corr"] is None
    assert out["settings"] == {"k": 8}


def test_null_false_p_and_p_corr_are_none(synthetic_nispace):
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)
    nsp.colocalize(method="pearson", verbose=False)
    nsp.permute(what="maps", n_perm=50, maps_method="random", seed=0, verbose=False)

    out = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, null=False, verbose=False)
    assert out["mc_method"] is None and out["p"] is None and out["p_corr"] is None
    assert isinstance(out["stat"], dict)


def test_null_true_always_returns_raw_p_regardless_of_mc_method(synthetic_nispace):
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)
    nsp.colocalize(method="pearson", verbose=False)
    nsp.permute(what="maps", n_perm=50, maps_method="random", seed=0, verbose=False)

    out_default = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, verbose=False)
    assert out_default["mc_method"] == "step_maxT"
    assert out_default["p"] is not None
    assert out_default["p_corr"] is not None

    out_none = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, mc_method=None,
                                                 verbose=False)
    assert out_none["mc_method"] is None
    assert out_none["p"] is not None       # raw p always present when null exists
    assert out_none["p_corr"] is None      # only the correction is skipped

    out_maxt = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, mc_method="maxT",
                                                 verbose=False)
    assert out_maxt["mc_method"] == "maxT"
    assert out_maxt["p"] is not None and out_maxt["p_corr"] is not None
    # raw p should be identical regardless of which correction (if any) was requested
    for x_lab in out_default["p"]:
        np.testing.assert_allclose(out_default["p"][x_lab].to_numpy(),
                                   out_none["p"][x_lab].to_numpy())
        np.testing.assert_allclose(out_default["p"][x_lab].to_numpy(),
                                   out_maxt["p"][x_lab].to_numpy())


def test_settings_k_mode(synthetic_nispace):
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)
    nsp.colocalize(method="pearson", verbose=False)
    out = diagnostics.local_colocalization(nsp, k=12, dist_mat=dist_mat, null=False, verbose=False)
    assert out["settings"] == {"k": 12}


def test_settings_radius_mode_reports_effective_n(synthetic_nispace):
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)
    nsp.colocalize(method="pearson", verbose=False)
    out = diagnostics.local_colocalization(nsp, radius=5, dist_mat=dist_mat, null=False,
                                           verbose=False)
    settings = out["settings"]
    assert set(settings.keys()) == {"radius", "n_neighbors_min", "n_neighbors_median",
                                    "n_neighbors_max"}
    assert settings["radius"] == 5
    assert settings["n_neighbors_min"] <= settings["n_neighbors_median"] <= settings["n_neighbors_max"]
    assert settings["n_neighbors_min"] >= 1


def test_stat_type_matches_method_primary_stat(synthetic_nispace):
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)
    nsp.colocalize(method="mlr", verbose=False)
    out = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, null=False, verbose=False)
    assert out["stat_type"] == "r2"


# ---------------------------------------------------------------------------
# Observed statistic: correctness, method families, unsupported methods
# ---------------------------------------------------------------------------

def test_full_window_matches_plain_colocalize_pearson(synthetic_nispace):
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)

    nsp.colocalize(method="pearson", verbose=False)
    global_r = nsp.colocalize(method="pearson", store=False, verbose=False, r_to_z=False)

    res = diagnostics.local_colocalization(nsp, k=n_parcels, dist_mat=dist_mat,
                                           null=False, verbose=False)["stat"]
    for x_lab in res:
        # every window is identical (the whole brain) -> constant column equal to global rho
        np.testing.assert_allclose(res[x_lab].to_numpy(),
                                   np.full((1, n_parcels), global_r.loc["y0", x_lab]),
                                   atol=ATOL)


def test_full_window_matches_plain_colocalize_mlr(synthetic_nispace):
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)

    nsp.colocalize(method="mlr", verbose=False)
    global_r2 = nsp.colocalize(method="mlr", store=False, verbose=False, r_to_z=False,
                               force_dict=True)["r2"]

    res = diagnostics.local_colocalization(nsp, k=n_parcels, dist_mat=dist_mat,
                                           null=False, verbose=False)["stat"]
    assert isinstance(res, pd.DataFrame)  # joint method -> single DataFrame, not a dict
    np.testing.assert_allclose(res.to_numpy(),
                               np.full((1, n_parcels), global_r2.iloc[0, 0]), atol=ATOL)


def test_small_window_differs_from_global(synthetic_nispace):
    """Sanity check that windowing actually does something (not a no-op)."""
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)

    nsp.colocalize(method="pearson", verbose=False)
    res = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat,
                                           null=False, verbose=False)["stat"]
    for x_lab in res:
        assert res[x_lab].to_numpy().std() > 0  # local rho varies across regions


@pytest.mark.parametrize("method", ["lasso", "ridge", "elasticnet"])
def test_regularized_methods_rejected(synthetic_nispace, method):
    """Rejected before even checking for a prior colocalize() run -- lasso/ridge/
    elasticnet's own CV-split machinery additionally requires a parcellation (to
    build its distance matrix), which this parcellation-free fixture doesn't have,
    so colocalize(method=method) itself isn't callable here; the point of this test
    is that local_colocalization() rejects the method outright, independent of that."""
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)
    with pytest.raises(ValueError):
        diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, method=method, verbose=False)


def test_requires_prior_colocalize(synthetic_nispace):
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)
    with pytest.raises(KeyError):
        diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, method="pearson", verbose=False)


def test_requires_exactly_one_of_k_radius(synthetic_nispace):
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)
    nsp.colocalize(method="pearson", verbose=False)
    with pytest.raises(ValueError):
        diagnostics.local_colocalization(nsp, dist_mat=dist_mat, verbose=False)
    with pytest.raises(ValueError):
        diagnostics.local_colocalization(nsp, k=8, radius=10, dist_mat=dist_mat, verbose=False)


def test_dist_mat_shape_mismatch_raises(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)
    with pytest.raises(ValueError):
        diagnostics.local_colocalization(nsp, k=8, dist_mat=np.zeros((3, 3)), verbose=False)


def test_radius_mode_runs(synthetic_nispace):
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)
    nsp.colocalize(method="pearson", verbose=False)
    res = diagnostics.local_colocalization(nsp, radius=5, dist_mat=dist_mat,
                                           null=False, verbose=False)["stat"]
    for x_lab in res:
        assert res[x_lab].shape == (1, n_parcels)


def test_invalid_mc_method_raises(synthetic_nispace):
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)
    nsp.colocalize(method="pearson", verbose=False)
    nsp.permute(what="maps", n_perm=20, maps_method="random", seed=0, verbose=False)
    with pytest.raises(ValueError):
        diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, mc_method="bonferroni",
                                         verbose=False)


# ---------------------------------------------------------------------------
# Min-k guardrail (regression-family / joint methods only)
# ---------------------------------------------------------------------------

def test_min_k_guardrail_joint_method(synthetic_nispace):
    nsp = synthetic_nispace  # n_x=3 predictors -> need k > 6
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)
    nsp.colocalize(method="mlr", verbose=False)
    with pytest.raises(ValueError):
        diagnostics.local_colocalization(nsp, k=6, dist_mat=dist_mat, null=False, verbose=False)
    # just above the floor should work fine
    diagnostics.local_colocalization(nsp, k=7, dist_mat=dist_mat, null=False, verbose=False)


def test_min_k_guardrail_not_applied_to_pearson(synthetic_nispace):
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)
    nsp.colocalize(method="pearson", verbose=False)
    # k=3 is fine for a correlation method even with many predictor maps
    res = diagnostics.local_colocalization(nsp, k=3, dist_mat=dist_mat, null=False, verbose=False)
    assert isinstance(res["stat"], dict)


# ---------------------------------------------------------------------------
# Null auto-detection: no permute(), permute(what="maps"), permute(what="groups")
# ---------------------------------------------------------------------------

def test_no_permute_returns_observed_only(synthetic_nispace):
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)
    nsp.colocalize(method="pearson", verbose=False)
    res = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, verbose=False)
    assert res["p"] is None  # no permute() was run -- auto-detected null=False


def test_explicit_null_true_without_permute_raises(synthetic_nispace):
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)
    nsp.colocalize(method="pearson", verbose=False)
    with pytest.raises(ValueError):
        diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, null=True, verbose=False)


def test_null_auto_detected_from_maps_permute(synthetic_nispace):
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)
    nsp.colocalize(method="pearson", verbose=False)
    nsp.permute(what="maps", n_perm=100, maps_method="random", seed=0, verbose=False)

    out = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, verbose=False)
    assert out["p"] is not None  # null auto-picked up

    stat, p = out["stat"], out["p"]
    for x_lab in stat:
        assert stat[x_lab].shape == p[x_lab].shape == (1, n_parcels)
        assert ((p[x_lab].to_numpy() >= 0) & (p[x_lab].to_numpy() <= 1)).all()

    # null=False explicitly must override auto-detection
    out_no_null = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, null=False,
                                                    verbose=False)
    assert out_no_null["p"] is None


def test_null_auto_detected_from_groups_permute(synthetic_nispace):
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    n_x = nsp._X.shape[0]
    dist_mat = _index_dist_mat(n_parcels)
    groups = np.array(["a"] * 8 + ["b"] * 7)  # matches toy_regression's default n_y=1...

    # rebuild Y with enough subjects to form two groups (toy_regression's synthetic_nispace
    # fixture only has n_y=1 by default, so construct a small multi-subject Y here directly)
    rng = np.random.default_rng(7)
    X = nsp._X.to_numpy()
    w = rng.normal(size=n_x)
    signal = w @ X
    Y_a = rng.normal(scale=1.0, size=(8, n_parcels))
    Y_b = signal[np.newaxis, :] * 0.6 + rng.normal(scale=1.0, size=(7, n_parcels))
    Y = np.vstack([Y_a, Y_b])
    y_labels = [f"suba{i}" for i in range(8)] + [f"subb{i}" for i in range(7)]

    nsp2 = NiSpace(x=nsp._X, y=pd.DataFrame(Y, index=y_labels, columns=nsp._Y.columns),
                  z=None, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp2.fit()
    nsp2.transform_y("hedges(a,b)", groups=groups, verbose=False)
    nsp2.colocalize(method="pearson", verbose=False)
    nsp2.permute(what="groups", n_perm=100, seed=0, verbose=False)

    out = diagnostics.local_colocalization(nsp2, k=8, dist_mat=dist_mat, verbose=False)
    stat, p = out["stat"], out["p"]
    for x_lab in stat:
        assert stat[x_lab].shape == p[x_lab].shape == (1, n_parcels)
        assert ((p[x_lab].to_numpy() >= 0) & (p[x_lab].to_numpy() <= 1)).all()


def test_null_xsea_still_unsupported(rng):
    """XSEA + null is the one restriction that wasn't lifted by the null-method
    extension (mi/slr/mlr/dominance/pls/pcr all gained null support -- XSEA didn't)."""
    n_parcels = 30
    parcel_labels = [f"p{i}" for i in range(n_parcels)]
    idx = pd.MultiIndex.from_tuples([("setA", f"geneA{i}") for i in range(4)],
                                    names=["set", "gene"])
    x_xsea = pd.DataFrame(rng.normal(size=(4, n_parcels)), index=idx, columns=parcel_labels)
    y_df = pd.DataFrame(rng.normal(size=(1, n_parcels)), index=["y0"], columns=parcel_labels)

    nsp = NiSpace(x=x_xsea, y=y_df, z=None, parcellation=None, standardize=False,
                 n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    nsp.colocalize(method="pearson", xsea=True, verbose=False)
    nsp.permute(what="maps", maps_method="random", n_perm=50, seed=0, verbose=False)

    dist_mat = _index_dist_mat(n_parcels)
    with pytest.raises(NotImplementedError):
        diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, verbose=False)
    res = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, null=False, verbose=False)
    assert res["p"] is None and isinstance(res["stat"], dict)


def test_null_auto_detected_from_ymaps_permute(synthetic_nispace):
    """maps_which='Y' (nulling the target map, not the predictors) -- the mirror
    image of the default X-sided case, exercised separately since the null-side
    branch in local_colocalization() differs (X observed / Y nulled vs. the
    reverse)."""
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)
    nsp.colocalize(method="pearson", verbose=False)
    nsp.permute(what="maps", maps_which="Y", n_perm=100, maps_method="random", seed=0,
               verbose=False)
    assert nsp._last_settings.get("perm") == "Ymaps"

    out = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, verbose=False)
    stat, p = out["stat"], out["p"]
    for x_lab in stat:
        assert stat[x_lab].shape == p[x_lab].shape == (1, n_parcels)
        assert ((p[x_lab].to_numpy() >= 0) & (p[x_lab].to_numpy() <= 1)).all()


def test_null_both_sides_maps_rejected(synthetic_nispace):
    """maps_which=['X','Y'] nulls both sides, but permute() itself only retains the
    second side's null in nsp._nulls['maps_null'] (the first is overwritten) -- so
    there is no complete both-sided null to recover from a cold nsp. Must raise,
    not silently use only one side's null as if it were the full picture."""
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)
    nsp.colocalize(method="pearson", verbose=False)
    nsp.permute(what="maps", maps_which=["X", "Y"], n_perm=50, maps_method="random", seed=0,
               verbose=False)
    assert nsp._last_settings.get("perm") == "XYmaps"
    with pytest.raises(NotImplementedError):
        diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, verbose=False)


def test_null_sets_permutation_rejected(rng):
    """what='sets' (XSEA) permutes gene-to-set membership, not a per-parcel spatial
    map -- structurally not re-sliceable per region window."""
    n_parcels = 30
    parcel_labels = [f"p{i}" for i in range(n_parcels)]
    idx = pd.MultiIndex.from_tuples([("setA", f"geneA{i}") for i in range(4)]
                                    + [("setB", f"geneB{i}") for i in range(6)],
                                    names=["set", "gene"])
    x_xsea = pd.DataFrame(rng.normal(size=(10, n_parcels)), index=idx, columns=parcel_labels)
    y = rng.normal(size=(1, n_parcels))
    y_df = pd.DataFrame(y, index=["y0"], columns=parcel_labels)

    nsp = NiSpace(x=x_xsea, y=y_df, z=None, parcellation=None, standardize=False,
                 n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    nsp.colocalize(method="pearson", xsea=True, verbose=False)
    nsp.permute(what="sets", n_perm=50, seed=0, verbose=False)
    assert nsp._last_settings.get("perm") == "sets"

    dist_mat = _index_dist_mat(n_parcels)
    with pytest.raises(NotImplementedError):
        diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, verbose=False)


# ---------------------------------------------------------------------------
# Extended null support: fast path (pearson/spearman/partialpearson/
# partialspearman) and slow path (mi/slr/mlr/dominance/pls/pcr), NaN handling,
# Z-regression -- added when null support was extended beyond pearson/spearman.
# ---------------------------------------------------------------------------

def _make_nsp(rng, n_parcels=40, n_x=2, with_z=False, with_nan=False):
    labels = [f"p{i}" for i in range(n_parcels)]
    X = rng.normal(size=(n_x, n_parcels))
    w = rng.normal(size=n_x)
    y = w @ X + rng.normal(scale=0.5, size=n_parcels)
    if with_nan:
        X = X.copy()
        X[0, 3] = np.nan
        if n_x > 1:
            X[1, 7] = np.nan
    x_df = pd.DataFrame(X, index=[f"x{i}" for i in range(n_x)], columns=labels)
    y_df = pd.DataFrame(y[np.newaxis, :], index=["y0"], columns=labels)
    z_df = None
    if with_z:
        z = rng.normal(size=n_parcels)
        z_df = pd.DataFrame(z[np.newaxis, :], index=["z0"], columns=labels)
    nsp = NiSpace(x=x_df, y=y_df, z=z_df, parcellation=None, standardize=False,
                 n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    return nsp


def test_pearson_rank_true_matches_spearman_exactly(rng):
    """Ground-truth check: colocalize(method='pearson', rank=True) must give the exact
    same local_colocalization() observed statistic AND null p-values as
    colocalize(method='spearman') -- both the fast-path null computation and the
    per-window observed statistic must resolve rank the same way colocalize() itself
    would, not silently fall back to a method-name heuristic or a stale last-used value
    (regression guard: this caught a real bug where the per-window observed colocalize()
    call omitted rank=/regress_z=/zy_matched=, silently diverging from the stored
    method's resolved settings)."""
    n_parcels = 40
    dist_mat = _index_dist_mat(n_parcels)

    # build the X/Y data once and reuse for both nsp_a/nsp_b -- a valid ground-truth
    # comparison requires identical underlying data, not just the same rng seeded twice
    # (which would advance state differently and produce different draws each call).
    # k=15 (not a tiny k=8) keeps an exact-rho=1 window (colocalize()'s own
    # r_equal_one="raise" safety check) astronomically unlikely by chance.
    n_x = 2
    labels = [f"p{i}" for i in range(n_parcels)]
    X = rng.normal(size=(n_x, n_parcels))
    w = rng.normal(size=n_x)
    y = w @ X + rng.normal(scale=0.5, size=n_parcels)
    x_df = pd.DataFrame(X, index=[f"x{i}" for i in range(n_x)], columns=labels)
    y_df = pd.DataFrame(y[np.newaxis, :], index=["y0"], columns=labels)

    nsp_a = NiSpace(x=x_df, y=y_df, z=None, parcellation=None, standardize=False,
                    n_proc=1, verbose=False, return_self=False)
    nsp_a.fit()
    nsp_a.colocalize(method="pearson", rank=True, verbose=False)
    nsp_a.permute(what="maps", maps_method="random", n_perm=100, seed=42, verbose=False)
    out_a = diagnostics.local_colocalization(nsp_a, k=15, dist_mat=dist_mat, verbose=False)

    nsp_b = NiSpace(x=x_df, y=y_df, z=None, parcellation=None, standardize=False,
                    n_proc=1, verbose=False, return_self=False)
    nsp_b.fit()
    nsp_b.colocalize(method="spearman", verbose=False)
    nsp_b.permute(what="maps", maps_method="random", n_perm=100, seed=42, verbose=False)
    out_b = diagnostics.local_colocalization(nsp_b, k=15, dist_mat=dist_mat, verbose=False)

    for x_lab in out_a["stat"]:
        np.testing.assert_allclose(out_a["stat"][x_lab].to_numpy(),
                                   out_b["stat"][x_lab].to_numpy(), atol=1e-5)
        np.testing.assert_allclose(out_a["p"][x_lab].to_numpy(),
                                   out_b["p"][x_lab].to_numpy(), atol=1e-8)
        np.testing.assert_allclose(out_a["p_corr"][x_lab].to_numpy(),
                                   out_b["p_corr"][x_lab].to_numpy(), atol=1e-8)


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_fast_path_null_handles_nan(rng, method):
    n_parcels = 40
    dist_mat = _index_dist_mat(n_parcels)
    nsp = _make_nsp(rng, n_parcels=n_parcels, n_x=3, with_nan=True)
    nsp.colocalize(method=method, regress_z=False, verbose=False)
    nsp.permute(what="maps", maps_method="random", n_perm=100, seed=0, verbose=False)

    out = diagnostics.local_colocalization(nsp, k=10, dist_mat=dist_mat, verbose=False)
    for x_lab in out["stat"]:
        assert not out["stat"][x_lab].isna().any().any()
        assert ((out["p"][x_lab].to_numpy() >= 0) & (out["p"][x_lab].to_numpy() <= 1)).all()


@pytest.mark.parametrize("method", ["partialpearson", "partialspearman"])
def test_fast_path_null_partial_methods_with_z(rng, method):
    n_parcels = 40
    dist_mat = _index_dist_mat(n_parcels)
    nsp = _make_nsp(rng, n_parcels=n_parcels, n_x=2, with_z=True, with_nan=True)
    nsp.colocalize(method=method, verbose=False)
    nsp.permute(what="maps", maps_method="random", n_perm=100, seed=0, verbose=False)

    out = diagnostics.local_colocalization(nsp, k=10, dist_mat=dist_mat, verbose=False)
    for x_lab in out["stat"]:
        assert not out["stat"][x_lab].isna().any().any()
        assert ((out["p"][x_lab].to_numpy() >= 0) & (out["p"][x_lab].to_numpy() <= 1)).all()


@pytest.mark.parametrize("method", ["mi", "slr"])
def test_slow_path_null_per_predictor_methods(rng, method):
    n_parcels = 30
    dist_mat = _index_dist_mat(n_parcels)
    nsp = _make_nsp(rng, n_parcels=n_parcels, n_x=2)
    nsp.colocalize(method=method, verbose=False)
    nsp.permute(what="maps", maps_method="random", n_perm=20, seed=0, verbose=False)

    out = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, verbose=False)
    assert isinstance(out["stat"], dict) and isinstance(out["p"], dict)
    for x_lab in out["stat"]:
        assert out["stat"][x_lab].shape == out["p"][x_lab].shape == (1, n_parcels)
        assert ((out["p"][x_lab].to_numpy() >= 0) & (out["p"][x_lab].to_numpy() <= 1)).all()


@pytest.mark.parametrize("method", ["mlr", "dominance", "pls", "pcr"])
def test_slow_path_null_joint_methods(rng, method):
    n_parcels = 30
    dist_mat = _index_dist_mat(n_parcels)
    nsp = _make_nsp(rng, n_parcels=n_parcels, n_x=2)
    nsp.colocalize(method=method, verbose=False)
    nsp.permute(what="maps", maps_method="random", n_perm=15, seed=0, verbose=False)

    out = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, verbose=False)
    assert isinstance(out["stat"], pd.DataFrame) and isinstance(out["p"], pd.DataFrame)
    assert out["stat"].shape == out["p"].shape == (1, n_parcels)
    assert ((out["p"].to_numpy() >= 0) & (out["p"].to_numpy() <= 1)).all()


def test_slow_path_null_groups_pathway(rng):
    """Slow-path methods must also pick up the groups pathway, not just maps."""
    n_parcels = 30
    n_x = 2
    dist_mat = _index_dist_mat(n_parcels)
    labels = [f"p{i}" for i in range(n_parcels)]
    groups = np.array(["a"] * 8 + ["b"] * 7)

    X = rng.normal(size=(n_x, n_parcels))
    w = rng.normal(size=n_x)
    signal = w @ X
    Y_a = rng.normal(scale=1.0, size=(8, n_parcels))
    Y_b = signal[np.newaxis, :] * 0.6 + rng.normal(scale=1.0, size=(7, n_parcels))
    Y = np.vstack([Y_a, Y_b])
    y_labels = [f"suba{i}" for i in range(8)] + [f"subb{i}" for i in range(7)]

    nsp = NiSpace(x=pd.DataFrame(X, index=[f"x{i}" for i in range(n_x)], columns=labels),
                 y=pd.DataFrame(Y, index=y_labels, columns=labels),
                 z=None, parcellation=None, standardize=False,
                 n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    nsp.transform_y("hedges(a,b)", groups=groups, verbose=False)
    nsp.colocalize(method="slr", verbose=False)
    nsp.permute(what="groups", n_perm=20, seed=0, verbose=False)

    out = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, verbose=False)
    for x_lab in out["stat"]:
        assert out["stat"][x_lab].shape == out["p"][x_lab].shape == (1, n_parcels)
        assert ((out["p"][x_lab].to_numpy() >= 0) & (out["p"][x_lab].to_numpy() <= 1)).all()


def test_null_false_still_works_for_every_supported_method(rng):
    """observed-only path must remain unaffected by the null-path extension, for
    every method except lasso/ridge/elasticnet."""
    n_parcels = 30
    dist_mat = _index_dist_mat(n_parcels)
    for method in ["pearson", "spearman", "partialpearson", "partialspearman",
                  "mi", "slr", "mlr", "dominance", "pls", "pcr"]:
        nsp = _make_nsp(rng, n_parcels=n_parcels, n_x=2, with_z=True)
        nsp.colocalize(method=method, verbose=False)
        # k=15 (not 8): an 8-point window occasionally produces an exact rank tie
        # (spearman rho==1), tripping colocalize()'s r_equal_one safety check;
        # this was flaky across environments (rng draw count for _make_nsp is
        # identical, but the resulting window statistics are small-sample-sensitive)
        out = diagnostics.local_colocalization(nsp, k=15, dist_mat=dist_mat, null=False,
                                               verbose=False)
        assert out["p"] is None and out["p_corr"] is None
        assert isinstance(out["stat"], (dict, pd.DataFrame))


# ---------------------------------------------------------------------------
# Gaussian-kernel searchlight (fwhm_mm): restricted forever to pearson/
# spearman/partialpearson/partialspearman -- no window, every seed weights
# ALL parcels by exp(-d**2/(2*sigma**2)); ground-truth invariant here is a
# very wide fwhm_mm (weights -> ~uniform over the whole map) reproducing
# plain colocalize()'s whole-brain result exactly, the kernel-mode analogue
# of the k=n_parcels invariant used for the k/radius modes above.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["pearson", "spearman", "partialpearson", "partialspearman"])
def test_gaussian_wide_fwhm_matches_plain_colocalize(rng, method):
    n_parcels = 30
    dist_mat = _index_dist_mat(n_parcels)
    nsp = _make_nsp(rng, n_parcels=n_parcels, n_x=2, with_z=True)
    glob = nsp.colocalize(method=method, verbose=False, force_dict=True, r_to_z=False)
    glob_df = next(iter(glob.values()))

    out = diagnostics.local_colocalization(nsp, fwhm_mm=1e6, dist_mat=dist_mat, null=False,
                                           verbose=False)
    assert out["settings"] == {"fwhm_mm": 1e6, "sigma_mm": pytest.approx(1e6 * (1 / 2.3548200450309493))}
    for x_lab, df in out["stat"].items():
        got = df.to_numpy()
        want = glob_df[x_lab].to_numpy()[:, np.newaxis]
        assert np.allclose(got, want, atol=1e-5)


def test_gaussian_narrow_fwhm_differs_from_wide(rng):
    n_parcels = 30
    dist_mat = _index_dist_mat(n_parcels)
    nsp = _make_nsp(rng, n_parcels=n_parcels, n_x=2, with_z=False)
    nsp.colocalize(method="pearson", verbose=False)
    narrow = diagnostics.local_colocalization(nsp, fwhm_mm=2.0, dist_mat=dist_mat, null=False,
                                              verbose=False)["stat"]
    wide = diagnostics.local_colocalization(nsp, fwhm_mm=1e6, dist_mat=dist_mat, null=False,
                                            verbose=False)["stat"]
    for x_lab in narrow:
        assert not np.allclose(narrow[x_lab].to_numpy(), wide[x_lab].to_numpy())
        # a narrow kernel gives a genuinely local (varying-by-seed) statistic, unlike
        # the wide kernel's near-constant whole-brain value
        assert narrow[x_lab].to_numpy().std() > wide[x_lab].to_numpy().std()


def test_gaussian_rejects_unsupported_methods(rng):
    n_parcels = 30
    dist_mat = _index_dist_mat(n_parcels)
    for method in ["mi", "slr", "mlr", "dominance", "pls", "pcr"]:
        nsp = _make_nsp(rng, n_parcels=n_parcels, n_x=2, with_z=False)
        nsp.colocalize(method=method, verbose=False)
        with pytest.raises(ValueError):
            diagnostics.local_colocalization(nsp, fwhm_mm=10.0, dist_mat=dist_mat, null=False,
                                             verbose=False)


def test_gaussian_mutually_exclusive_with_k_and_radius(synthetic_nispace):
    nsp = synthetic_nispace
    n_parcels = nsp._Y.shape[1]
    dist_mat = _index_dist_mat(n_parcels)
    nsp.colocalize(method="pearson", verbose=False)
    with pytest.raises(ValueError):
        diagnostics.local_colocalization(nsp, k=8, fwhm_mm=10.0, dist_mat=dist_mat, verbose=False)
    with pytest.raises(ValueError):
        diagnostics.local_colocalization(nsp, radius=5, fwhm_mm=10.0, dist_mat=dist_mat, verbose=False)
    with pytest.raises(ValueError):
        diagnostics.local_colocalization(nsp, dist_mat=dist_mat, verbose=False)


def test_gaussian_rejects_xsea(rng):
    n_parcels = 30
    dist_mat = _index_dist_mat(n_parcels)
    nsp = _make_nsp(rng, n_parcels=n_parcels, n_x=2, with_z=False)
    nsp.colocalize(method="pearson", verbose=False)
    with pytest.raises(NotImplementedError):
        diagnostics.local_colocalization(nsp, fwhm_mm=10.0, dist_mat=dist_mat, xsea=True,
                                         null=False, verbose=False)


def test_gaussian_handles_nan(rng):
    n_parcels = 30
    dist_mat = _index_dist_mat(n_parcels)
    nsp = _make_nsp(rng, n_parcels=n_parcels, n_x=2, with_z=False, with_nan=True)
    nsp.colocalize(method="pearson", verbose=False)
    out = diagnostics.local_colocalization(nsp, fwhm_mm=15.0, dist_mat=dist_mat, null=False,
                                           verbose=False)
    for x_lab, df in out["stat"].items():
        arr = df.to_numpy()
        assert np.isfinite(arr).all()


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_gaussian_null_auto_detected_from_maps_permute(rng, method):
    n_parcels = 30
    dist_mat = _index_dist_mat(n_parcels)
    nsp = _make_nsp(rng, n_parcels=n_parcels, n_x=2, with_z=False)
    nsp.colocalize(method=method, verbose=False)
    nsp.permute(what="maps", n_perm=100, maps_method="random", seed=0, verbose=False)
    out = diagnostics.local_colocalization(nsp, fwhm_mm=15.0, dist_mat=dist_mat, null=True,
                                           verbose=False)
    assert out["mc_method"] == "step_maxT"
    for x_lab in out["stat"]:
        p, p_corr = out["p"][x_lab].to_numpy(), out["p_corr"][x_lab].to_numpy()
        assert p.shape == out["stat"][x_lab].shape
        assert ((p >= 0) & (p <= 1)).all()
        assert ((p_corr >= 0) & (p_corr <= 1)).all()


def test_gaussian_null_with_regress_z(rng):
    n_parcels = 30
    dist_mat = _index_dist_mat(n_parcels)
    nsp = _make_nsp(rng, n_parcels=n_parcels, n_x=2, with_z=True)
    nsp.colocalize(method="partialspearman", verbose=False)
    nsp.permute(what="maps", n_perm=80, maps_method="random", seed=0, verbose=False)
    out = diagnostics.local_colocalization(nsp, fwhm_mm=20.0, dist_mat=dist_mat, null=True,
                                           verbose=False)
    for x_lab in out["stat"]:
        p = out["p"][x_lab].to_numpy()
        assert np.isfinite(p).all()
        assert ((p >= 0) & (p <= 1)).all()


# ---------------------------------------------------------------------------
# pooled: mirrors permute()'s own pooled_p -- pools observed stat AND every null
# draw across the Y axis *before* computing p, not just an average of the final
# per-Y-row p-values. For permute(what="groups") it's not a free choice: forced to
# "mean" (per core/permute.py's _resolve_permute_mode_settings), same as permute()
# itself forces it, even when a conflicting pooled= is passed explicitly. For
# permute(what="maps") it remains a free choice, defaulting from
# nsp._last_settings["pooled_p"].
# ---------------------------------------------------------------------------

def _make_groups_nsp(rng, n_parcels=30, n_x=2, n_per_group=8):
    """Multi-row groups nsp via a *paired* transform ("elemdiff(a,b)", elementwise
    a - b) -- unlike "hedges(a,b)" (single-row group-difference map), this keeps one
    row per subject pair, the case _resolve_permute_mode_settings' docstring calls
    out as "unpaired multi-row transforms don't carry subject-specific information"
    (here: paired, but still multi-row) -- the only shape where forced pooling is
    not a no-op."""
    X = rng.normal(size=(n_x, n_parcels))
    w = rng.normal(size=n_x)
    signal = w @ X
    Y_a = rng.normal(scale=1.0, size=(n_per_group, n_parcels))
    Y_b = signal[np.newaxis, :] * 0.6 + rng.normal(scale=1.0, size=(n_per_group, n_parcels))
    Y = np.vstack([Y_a, Y_b])
    groups = np.array(["a"] * n_per_group + ["b"] * n_per_group)
    # elemdiff(a,b) is a paired formula -- subjects gives the a<->b pairing (subject i
    # in group a pairs with subject i in group b), required by permute_groups(paired=True)
    subjects = np.array(list(range(n_per_group)) * 2)
    y_labels = [f"suba{i}" for i in range(n_per_group)] + [f"subb{i}" for i in range(n_per_group)]
    x_labels = [f"x{i}" for i in range(n_x)]
    parcel_labels = [f"p{i}" for i in range(n_parcels)]

    nsp = NiSpace(x=pd.DataFrame(X, index=x_labels, columns=parcel_labels),
                 y=pd.DataFrame(Y, index=y_labels, columns=parcel_labels),
                 z=None, parcellation=None, standardize=False,
                 n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    nsp.transform_y("elemdiff(a,b)", groups=groups, subjects=subjects, verbose=False)
    return nsp


def test_pooled_forced_for_groups_null_by_default(rng):
    n_parcels, n_per_group = 30, 8
    dist_mat = _index_dist_mat(n_parcels)
    nsp = _make_groups_nsp(rng, n_parcels=n_parcels, n_per_group=n_per_group)
    assert nsp.get_y(verbose=False).shape[0] == n_per_group  # multi-row, the case that matters

    nsp.colocalize(method="pearson", verbose=False)
    nsp.permute(what="groups", n_perm=50, seed=0, verbose=False)
    assert nsp._last_settings.get("pooled_p") == "mean"  # forced by permute() itself

    out = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, verbose=False)
    assert out["pooled"] == "mean"
    for x_lab in out["stat"]:
        # "stat" (the observed statistic) is NEVER pooled -- only "p"/"p_corr" are.
        # get_colocalizations() itself stays unpooled no matter what pooled_p a later
        # permute() used (permute()'s _colocs_obs is a fresh array copy, api.py:2622,
        # never written back to the stored result) -- local_colocalization()'s "stat"
        # must match that invariant exactly.
        assert out["stat"][x_lab].shape == (n_per_group, n_parcels)
        assert out["p"][x_lab].shape == (1, n_parcels)
        assert list(out["p"][x_lab].index) == ["mean"]


def test_pooled_forced_for_groups_null_overrides_explicit_false(rng):
    """An explicit pooled=False (or any non-forced value) must NOT be honored for a
    groups null -- it's not a free choice, exactly like passing an incompatible
    pooled_p straight to permute(what="groups") isn't."""
    n_parcels, n_per_group = 30, 8
    dist_mat = _index_dist_mat(n_parcels)
    nsp = _make_groups_nsp(rng, n_parcels=n_parcels, n_per_group=n_per_group)
    nsp.colocalize(method="pearson", verbose=False)
    nsp.permute(what="groups", n_perm=50, seed=0, verbose=False)

    out = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, pooled=False,
                                           verbose=False)
    assert out["pooled"] == "mean"
    for x_lab in out["stat"]:
        assert out["stat"][x_lab].shape == (n_per_group, n_parcels)  # never pooled
        assert out["p"][x_lab].shape == (1, n_parcels)


def _make_multi_y_nsp(rng, n_parcels=30, n_x=1, n_y=10):
    """Genuine multi-row Y (unlike _make_nsp's single-row toy_regression shape) --
    needed to actually exercise pooled's reduction, since pooling n_Y==1 is always a
    no-op."""
    labels = [f"p{i}" for i in range(n_parcels)]
    X = rng.normal(size=(n_x, n_parcels))
    w = rng.normal(size=n_x)
    signal = w @ X
    Y = signal[np.newaxis, :] * 0.5 + rng.normal(scale=1.0, size=(n_y, n_parcels))
    x_df = pd.DataFrame(X, index=[f"x{i}" for i in range(n_x)], columns=labels)
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(n_y)], columns=labels)
    nsp = NiSpace(x=x_df, y=y_df, z=None, parcellation=None, standardize=False,
                 n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    return nsp


def test_pooled_free_choice_for_maps_null(rng):
    """permute(what="maps") does not force pooling -- it's a free, meaningful choice,
    defaulting from nsp._last_settings["pooled_p"] but overridable per-call. The key
    invariant checked here: "stat" NEVER changes shape (or values) with pooled -- only
    "p"/"p_corr" do -- mirroring get_colocalizations() staying unpooled regardless of
    what pooled_p a later permute() call used."""
    n_parcels, n_x, n_y = 30, 1, 10
    dist_mat = _index_dist_mat(n_parcels)
    nsp = _make_multi_y_nsp(rng, n_parcels=n_parcels, n_x=n_x, n_y=n_y)
    nsp.colocalize(method="pearson", verbose=False)
    nsp.permute(what="maps", n_perm=50, maps_method="random", seed=0, verbose=False)

    out_false = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, pooled=False,
                                                  verbose=False)
    assert out_false["pooled"] is False
    for x_lab in out_false["stat"]:
        assert out_false["stat"][x_lab].shape == (n_y, n_parcels)
        assert out_false["p"][x_lab].shape == (n_y, n_parcels)

    out_mean = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, pooled="mean",
                                                 verbose=False)
    assert out_mean["pooled"] == "mean"
    for x_lab in out_mean["stat"]:
        assert out_mean["stat"][x_lab].shape == (n_y, n_parcels)  # unchanged
        assert out_mean["p"][x_lab].shape == (1, n_parcels)
        assert list(out_mean["p"][x_lab].index) == ["mean"]

    # "stat" must be bit-for-bit identical regardless of pooled -- pooling only ever
    # touches a local copy used for the p comparison
    for x_lab in out_false["stat"]:
        pd.testing.assert_frame_equal(out_false["stat"][x_lab], out_mean["stat"][x_lab])


def test_pooled_ignored_when_null_false(rng):
    """pooled_p only ever matters for p-values -- with no null at all, it must have
    zero effect on anything, for any value passed."""
    n_parcels, n_x, n_y = 30, 1, 10
    dist_mat = _index_dist_mat(n_parcels)
    nsp = _make_multi_y_nsp(rng, n_parcels=n_parcels, n_x=n_x, n_y=n_y)
    nsp.colocalize(method="pearson", verbose=False)

    for pooled_arg in (False, True, "mean", "median"):
        out = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, null=False,
                                               pooled=pooled_arg, verbose=False)
        assert out["pooled"] is False
        assert out["p"] is None and out["p_corr"] is None
        for x_lab in out["stat"]:
            assert out["stat"][x_lab].shape == (n_y, n_parcels)


def test_pooled_noop_when_single_y_row(rng):
    """pooled=True with only one Y row is a documented no-op (same as permute()'s own
    convention) -- must not spuriously report "pooled": "mean" for a shape that never
    actually changed."""
    n_parcels = 30
    dist_mat = _index_dist_mat(n_parcels)
    nsp = _make_nsp(rng, n_parcels=n_parcels, n_x=2)
    nsp.colocalize(method="pearson", verbose=False)
    nsp.permute(what="maps", n_perm=50, maps_method="random", seed=0, verbose=False)

    out = diagnostics.local_colocalization(nsp, k=8, dist_mat=dist_mat, pooled=True,
                                           verbose=False)
    assert out["pooled"] is False
    for x_lab in out["stat"]:
        assert out["stat"][x_lab].shape[0] == 1
