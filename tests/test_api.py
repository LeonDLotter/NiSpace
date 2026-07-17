"""Tests for the NiSpace class in api.py itself -- the end-to-end user-facing
surface, not the lower-level machinery already covered elsewhere
(core/permute.py's null-map cache/dispatch -> test_permute.py, colocalize()
rank-leak regressions -> test_colocalize.py).

Covers: colocalize()/get_x()/get_y()/get_z() basics and error paths,
permute() end-to-end for what="maps" (random null, no parcellation needed)
and what="groups", get_p_values()/get_colocalizations() retrieval semantics
(force_dict, stats filtering), correct_p() routing across mc_method families
(meff, fdr_bh/statsmodels passthrough, maxT), normalize_colocalizations()/
get_normalized_colocalizations(), and copy()/to_pickle()/from_pickle()
round-trips.
"""

import numpy as np
import pandas as pd
import pytest

from nispace import NiSpace


# ── get_x / get_y / get_z ────────────────────────────────────────────────────

def test_get_x_returns_raw_data_by_default(synthetic_nispace):
    nsp = synthetic_nispace
    x = nsp.get_x(verbose=False)
    pd.testing.assert_frame_equal(x, nsp._X)


def test_get_x_copy_true_is_independent_of_internal_state(synthetic_nispace):
    nsp = synthetic_nispace
    x = nsp.get_x(verbose=False, copy=True)
    x.iloc[0, 0] = 999.0
    assert nsp._X.iloc[0, 0] != 999.0


def test_get_x_copy_false_is_live_reference(synthetic_nispace):
    nsp = synthetic_nispace
    x = nsp.get_x(verbose=False, copy=False)
    assert x is nsp._X


def test_get_x_unknown_reduction_raises_keyerror(synthetic_nispace):
    nsp = synthetic_nispace
    with pytest.raises(KeyError):
        nsp.get_x(X_reduction="not_a_real_reduction", verbose=False)


def test_get_x_maps_filter_restricts_rows(synthetic_nispace):
    nsp = synthetic_nispace
    x = nsp.get_x(maps="x0", verbose=False)
    assert list(x.index) == ["x0"]


def test_get_x_maps_filter_no_match_raises(synthetic_nispace):
    nsp = synthetic_nispace
    with pytest.raises(ValueError):
        nsp.get_x(maps="no_such_map", verbose=False)


def test_get_x_squeeze_returns_series_for_single_map(synthetic_nispace):
    nsp = synthetic_nispace
    x = nsp.get_x(maps="x0", squeeze=True, verbose=False)
    assert isinstance(x, pd.Series)


def test_get_y_unknown_transform_raises_keyerror(synthetic_nispace):
    nsp = synthetic_nispace
    with pytest.raises(KeyError):
        nsp.get_y(Y_transform="not_a_real_transform", verbose=False)


def test_get_z_raises_when_no_z_data(synthetic_nispace):
    nsp = synthetic_nispace
    with pytest.raises(ValueError):
        nsp.get_z(verbose=False)


# ── colocalize() / get_colocalizations() ─────────────────────────────────────

def test_colocalize_pearson_matches_direct_correlation(synthetic_nispace):
    nsp = synthetic_nispace
    # r_to_z=False: colocalize()'s "pearson" defaults to Fisher-z transforming
    # the correlation (r_to_z=True), so disable it here for a direct, untransformed
    # comparison against plain np.corrcoef.
    nsp.colocalize(method="pearson", r_to_z=False, verbose=False)
    coloc = nsp.get_colocalizations(verbose=False)

    X = np.array(nsp.get_x(verbose=False))
    Y = np.array(nsp.get_y(verbose=False))
    expected = np.array([
        [np.corrcoef(x_row, y_row)[0, 1] for x_row in X]
        for y_row in Y
    ])
    np.testing.assert_allclose(coloc.to_numpy(), expected, atol=1e-5)


def test_get_colocalizations_force_dict_wraps_single_stat(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)
    plain = nsp.get_colocalizations(verbose=False)
    forced = nsp.get_colocalizations(force_dict=True, verbose=False)
    assert isinstance(plain, pd.DataFrame)
    assert isinstance(forced, dict)
    pd.testing.assert_frame_equal(forced["rho"], plain)


def test_get_colocalizations_unknown_method_raises_keyerror(synthetic_nispace):
    nsp = synthetic_nispace
    with pytest.raises(KeyError):
        nsp.get_colocalizations(method="spearman", verbose=False)


# ── permute(): what="maps" (distance-matrix-free "random" null) ─────────────

@pytest.fixture
def permuted_maps_nispace(synthetic_nispace):
    """synthetic_nispace with colocalize(pearson) + permute(what="maps",
    maps_method="random") already run -- "random" is the one null method that
    needs neither a parcellation nor a distance matrix (see
    nispace.nulls._DISTMAT_FREE_METHODS), so this stays fully synthetic."""
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)
    nsp.permute(what="maps", maps_method="random", n_perm=200, seed=1, verbose=False)
    return nsp


def test_permute_maps_random_returns_p_values_in_unit_interval(permuted_maps_nispace):
    p = permuted_maps_nispace.get_p_values(verbose=False)
    assert ((p.to_numpy() >= 0) & (p.to_numpy() <= 1)).all()


def test_permute_maps_stores_retrievable_null_colocs(permuted_maps_nispace):
    nsp = permuted_maps_nispace
    assert len(nsp._nulls["_colocs"]) > 0


def test_permute_without_colocalize_runs_it_automatically(synthetic_nispace):
    """permute() should auto-run colocalize() with a warning if it wasn't
    called yet, rather than raising."""
    nsp = synthetic_nispace
    p = nsp.permute(what="maps", method="pearson", maps_method="random",
                     n_perm=200, seed=1, verbose=False)
    assert p is not None
    assert nsp._check_colocalize("pearson", raise_error=False)


def test_permute_maps_requires_parcellation_or_distmat_free_method(synthetic_nispace):
    """what="maps" with a distance-requiring null method (e.g. "moran") and no
    parcellation/dist_mat/precomputed nulls must raise, not silently proceed."""
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)
    with pytest.raises(ValueError):
        nsp.permute(what="maps", maps_method="moran", n_perm=50, verbose=False)


def test_permute_invalid_what_type_raises(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)
    with pytest.raises(ValueError):
        nsp.permute(what=123, verbose=False)


# ── permute(): what="groups" ─────────────────────────────────────────────────

@pytest.fixture
def grouped_nispace(rng):
    """Independent-groups synthetic setup (2 groups of 8 subjects each, one
    genuinely shifted) so permute(what="groups") + Y_transform="cohen(a,b)"
    has a real, non-degenerate signal to test against."""
    n_x, n_parcels, n_subj = 2, 20, 16
    X = rng.normal(size=(n_x, n_parcels))
    Y = rng.normal(size=(n_subj, n_parcels))
    groups = np.array(["a"] * 8 + ["b"] * 8)
    Y[groups == "b"] += 1.0

    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    x_df = pd.DataFrame(X, index=[f"x{i}" for i in range(n_x)], columns=parcel_labels)
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(n_subj)], columns=parcel_labels)

    nsp = NiSpace(x=x_df, y=y_df, z=None, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    nsp.colocalize(method="pearson", Y_transform="cohen(a,b)", groups=groups, verbose=False)
    return nsp, groups


def test_permute_groups_returns_p_values_distinct_from_observed_coloc(grouped_nispace):
    nsp, _ = grouped_nispace
    coloc = nsp.get_colocalizations(verbose=False)
    p = nsp.permute(what="groups", n_perm=200, seed=1, verbose=False)
    assert ((p.to_numpy() >= 0) & (p.to_numpy() <= 1)).all()
    assert not np.allclose(coloc.to_numpy(), p.to_numpy())


def test_permute_groups_rejects_binary_y(rng):
    n_x, n_parcels, n_subj = 2, 20, 10
    X = rng.normal(size=(n_x, n_parcels))
    Y = rng.normal(size=(n_subj, n_parcels))
    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    x_df = pd.DataFrame(X, index=[f"x{i}" for i in range(n_x)], columns=parcel_labels)
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(n_subj)], columns=parcel_labels)

    nsp = NiSpace(x=x_df, y=y_df, z=None, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False, binary_y=True)
    nsp.fit()
    nsp.colocalize(method="pearson", verbose=False)
    with pytest.raises(ValueError):
        nsp.permute(what="groups", n_perm=50, verbose=False)


# ── correct_p() ───────────────────────────────────────────────────────────────

def test_correct_p_fdr_bh_matches_statsmodels(permuted_maps_nispace):
    sm_multitest = pytest.importorskip("statsmodels.stats.multitest")
    nsp = permuted_maps_nispace
    p_raw = nsp.get_p_values(verbose=False)
    nsp.correct_p(mc_method="fdr_bh", verbose=False)
    p_corr = nsp.get_corrected_p_values(verbose=False)

    _, expected, _, _ = sm_multitest.multipletests(
        p_raw.to_numpy().ravel(), alpha=0.05, method="fdr_bh"
    )
    np.testing.assert_allclose(p_corr.to_numpy().ravel(), expected, atol=1e-8)


def test_correct_p_meff_is_at_least_as_conservative_as_uncorrected(permuted_maps_nispace):
    nsp = permuted_maps_nispace
    p_raw = nsp.get_p_values(verbose=False)
    nsp.correct_p(mc_method="meff", verbose=False)
    p_corr = nsp.get_corrected_p_values(verbose=False)
    assert (p_corr.to_numpy() >= p_raw.to_numpy() - 1e-8).all()


def test_correct_p_maxt_requires_stored_nulls(permuted_maps_nispace):
    """maxT/step_maxT need the full null distribution -- to_pickle(save_nulls=
    False) drops it, and correct_p("maxT") on the reloaded object must raise a
    clear KeyError rather than silently using something else."""
    nsp = permuted_maps_nispace
    nsp.correct_p(mc_method="maxT", verbose=False)  # works while nulls are present
    assert nsp._check_permute("pearson", "xmaps", mc_method="maxt", raise_error=False)

    import tempfile, os
    f = tempfile.mktemp(suffix=".pkl")
    try:
        nsp.to_pickle(f, save_nulls=False, verbose=False)
        nsp_reloaded = NiSpace.from_pickle(f, verbose=False)
        with pytest.raises(KeyError):
            nsp_reloaded.correct_p(mc_method="maxT", verbose=False)
    finally:
        if os.path.exists(f):
            os.remove(f)


def test_correct_p_store_false_does_not_persist(permuted_maps_nispace):
    nsp = permuted_maps_nispace
    before_keys = set(nsp._p_colocs.keys())
    nsp.correct_p(mc_method="fdr_bh", store=False, verbose=False)
    assert set(nsp._p_colocs.keys()) == before_keys


# ── normalize_colocalizations() ──────────────────────────────────────────────

def test_normalize_colocalizations_requires_permute_first(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)
    with pytest.raises(ValueError):
        nsp.normalize_colocalizations(verbose=False)


def test_normalize_colocalizations_matches_manual_robust_zscore(permuted_maps_nispace):
    from nispace.stats.effectsize import zscore_nan
    from nispace.stats.misc import _null_stats_to_array
    nsp = permuted_maps_nispace
    nsp.normalize_colocalizations(z_method="standard", verbose=False)
    z = nsp.get_normalized_colocalizations(verbose=False)

    obs = np.array(nsp.get_colocalizations(verbose=False), dtype=float)
    null_str = list(nsp._nulls["_colocs"].keys())[0]
    null_colocs = nsp._nulls["_colocs"][null_str]
    null_arr = _null_stats_to_array(null_colocs, "rho").astype(float)
    expected = zscore_nan(obs, null_arr)
    np.testing.assert_allclose(np.asarray(z), expected, atol=1e-5)


# ── copy() / to_pickle() / from_pickle() ─────────────────────────────────────

def test_copy_deep_is_independent(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)
    nsp_copy = nsp.copy(deep=True, verbose=False)

    nsp_copy._colocs.clear()
    assert len(nsp._colocs) > 0  # original untouched


def test_copy_shallow_shares_mutable_state(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)
    nsp_copy = nsp.copy(deep=False, verbose=False)

    nsp_copy._colocs.clear()
    assert len(nsp._colocs) == 0  # shared dict, original affected too


def test_pickle_roundtrip_preserves_colocalizations(synthetic_nispace, tmp_path):
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)
    f = tmp_path / "nsp.pkl"

    nsp.to_pickle(str(f), verbose=False)
    nsp_reloaded = NiSpace.from_pickle(str(f), verbose=False)

    pd.testing.assert_frame_equal(
        nsp_reloaded.get_colocalizations(verbose=False),
        nsp.get_colocalizations(verbose=False),
    )


def test_pickle_roundtrip_save_nulls_false_drops_nulls(permuted_maps_nispace, tmp_path):
    nsp = permuted_maps_nispace
    f = tmp_path / "nsp_no_nulls.pkl"

    nsp.to_pickle(str(f), save_nulls=False, verbose=False)
    nsp_reloaded = NiSpace.from_pickle(str(f), verbose=False)

    assert nsp_reloaded._nulls["_colocs"] == {}
    # original object (and its in-memory nulls) must be untouched by to_pickle()
    assert len(nsp._nulls["_colocs"]) > 0
    # p-values themselves still round-trip fine (they don't need the nulls)
    pd.testing.assert_frame_equal(
        nsp_reloaded.get_p_values(verbose=False),
        nsp.get_p_values(verbose=False),
    )
