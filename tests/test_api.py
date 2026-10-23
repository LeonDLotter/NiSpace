"""Tests for the NiSpace class in api.py itself -- the end-to-end user-facing
surface, not the lower-level machinery already covered elsewhere
(core/permute.py's null-map cache/dispatch -> test_permute.py, colocalize()
rank-leak regressions -> test_colocalize.py).

Covers: colocalize()/get_x()/get_y()/get_z() basics and error paths,
permute() end-to-end for what="maps" (random null, no parcellation needed)
and what="groups", get_p_values()/get_colocalizations() retrieval semantics
(force_dict, stats filtering), correct_p() routing across mc_method families
(meff, fdr_bh/statsmodels passthrough, maxT/step_maxT/minP/step_minP incl. the
pooled_p and scale-heterogeneity paths), normalize_colocalizations()/
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


# ── colocalize(): XSEA xsea_aggregation_method=None auto-selection + warning ─

@pytest.fixture
def nsp_xsea_weighted(rng):
    """X with 'set'/'gene'/'weight' MultiIndex levels, single Y map -- for
    xsea_aggregation_method=None auto-selection / weight-usage warning tests."""
    n_parcels = 20
    idx = pd.MultiIndex.from_tuples(
        [("setA", "g0", 1.0), ("setA", "g1", 3.0), ("setB", "g2", 1.0), ("setB", "g3", 2.0)],
        names=["set", "gene", "weight"],
    )
    X = rng.normal(size=(4, n_parcels))
    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    x_df = pd.DataFrame(X, index=idx, columns=parcel_labels)
    y = X[0] + rng.normal(scale=0.2, size=n_parcels)
    y_df = pd.DataFrame(y[None, :], index=["y0"], columns=parcel_labels)
    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    return nsp


@pytest.fixture
def nsp_xsea_unweighted(rng):
    """X with a 'set'/'gene' MultiIndex (no 'weight' level)."""
    n_parcels = 20
    idx = pd.MultiIndex.from_tuples(
        [("setA", "g0"), ("setA", "g1"), ("setB", "g2"), ("setB", "g3")],
        names=["set", "gene"],
    )
    X = rng.normal(size=(4, n_parcels))
    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    x_df = pd.DataFrame(X, index=idx, columns=parcel_labels)
    y = X[0] + rng.normal(scale=0.2, size=n_parcels)
    y_df = pd.DataFrame(y[None, :], index=["y0"], columns=parcel_labels)
    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    return nsp


def test_colocalize_xsea_default_mean_warns_when_weight_unused(nsp_xsea_weighted, caplog):
    nsp = nsp_xsea_weighted
    with caplog.at_level("WARNING"):
        nsp.colocalize(method="pearson", xsea=True, r_to_z=False, verbose=False)
    assert nsp._xsea_aggregation_method == "mean"
    assert any("does not use it" in r.message for r in caplog.records)


def test_colocalize_xsea_none_autoselects_weightedmean(nsp_xsea_weighted, caplog):
    nsp = nsp_xsea_weighted
    with caplog.at_level("INFO"):
        nsp.colocalize(method="pearson", xsea=True, xsea_aggregation_method=None,
                        r_to_z=False, verbose=True)
    assert nsp._xsea_aggregation_method == "weightedmean"
    assert any("auto-selecting" in r.message for r in caplog.records)
    assert not any("does not use it" in r.message for r in caplog.records)


def test_colocalize_xsea_none_autoselects_mean_when_no_weight(nsp_xsea_unweighted, caplog):
    nsp = nsp_xsea_unweighted
    with caplog.at_level("INFO"):
        nsp.colocalize(method="pearson", xsea=True, xsea_aggregation_method=None,
                        r_to_z=False, verbose=True)
    assert nsp._xsea_aggregation_method == "mean"
    assert any("auto-selecting" in r.message for r in caplog.records)
    assert not any("does not use it" in r.message for r in caplog.records)


def test_colocalize_xsea_explicit_weighted_without_weight_still_warns_and_strips(
    nsp_xsea_unweighted, caplog
):
    nsp = nsp_xsea_unweighted
    with caplog.at_level("WARNING"):
        nsp.colocalize(method="pearson", xsea=True, xsea_aggregation_method="weightedmean",
                        r_to_z=False, verbose=False)
    assert nsp._xsea_aggregation_method == "mean"
    assert any("Weighted XSEA requires" in r.message for r in caplog.records)


def test_colocalize_xsea_explicit_weighted_with_weight_present_no_warning(
    nsp_xsea_weighted, caplog
):
    nsp = nsp_xsea_weighted
    with caplog.at_level("WARNING"):
        nsp.colocalize(method="pearson", xsea=True, xsea_aggregation_method="weightedmean",
                        r_to_z=False, verbose=False)
    assert nsp._xsea_aggregation_method == "weightedmean"
    assert not any("does not use it" in r.message for r in caplog.records)
    assert not any("Weighted XSEA requires" in r.message for r in caplog.records)


# ── get_p_values()/get_normalized_colocalizations() before any permute() ────
# Regression tests for the `_last_settings["perm"]` landmine: before "perm" was
# added as a default key, calling these with no permutation ever run raised a
# bare, uninformative Exception from NiSpace._get_last() ("Last setting for
# 'perm' not found. Available: [...]") instead of a specific, catchable one.

def test_get_p_values_before_any_permute_raises_valueerror(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)
    with pytest.raises(ValueError, match="perm"):
        nsp.get_p_values(verbose=False)


def test_get_normalized_colocalizations_before_any_permute_raises_valueerror(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)
    with pytest.raises(ValueError, match="perm"):
        nsp.get_normalized_colocalizations(verbose=False)


def test_get_p_values_explicit_permute_what_before_permute_raises_keyerror(synthetic_nispace):
    """Distinct from the bare-call case above: with a concrete permute_what,
    _check_permute()'s own clean KeyError path fires (the bare-None case fails
    earlier, inside _get_df_string, with a generic ValueError instead)."""
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)
    with pytest.raises(KeyError, match="Did you run NiSpace.permute"):
        nsp.get_p_values(permute_what="xmaps", verbose=False)


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


# ── generic permute_what="maps" -> stored "xmaps"/"ymaps"/"xymaps" ──────────

def test_get_p_values_generic_maps_resolves_to_stored_side(permuted_maps_nispace):
    nsp = permuted_maps_nispace
    pd.testing.assert_frame_equal(nsp.get_p_values(permute_what="maps", verbose=False),
                                  nsp.get_p_values(permute_what="xmaps", verbose=False))
    pd.testing.assert_frame_equal(nsp.get_p_values(permute_what="MAPS", verbose=False),
                                  nsp.get_p_values(permute_what="xmaps", verbose=False))


def _custom_y_nulls(nsp, n):
    rng = np.random.default_rng(0)
    return {lbl: rng.normal(size=(n, nsp._Y.shape[1])) for lbl in nsp._Y.index}


def _n_null_colocs(nsp):
    _, nulls = nsp.get_colocalizations(get_nulls=True, nulls_permute_what="ymaps", verbose=False)
    return next(iter(nulls.values())).shape[1]   # {x_label: (stat, n_perm)}


def test_permute_n_perm_inherited_from_provided_nulls(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)
    nsp.permute(what="maps", maps_which="Y", maps_nulls=_custom_y_nulls(nsp, 37),
                seed=1, verbose=False)
    assert _n_null_colocs(nsp) == 37


def test_permute_explicit_smaller_n_perm_uses_first_provided_nulls(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)
    nsp.permute(what="maps", maps_which="Y", maps_nulls=_custom_y_nulls(nsp, 37),
                n_perm=20, seed=1, verbose=False)
    assert _n_null_colocs(nsp) == 20


def test_permute_explicit_larger_n_perm_than_provided_nulls_raises(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)
    with pytest.raises(ValueError, match="37 permutations"):
        nsp.permute(what="maps", maps_which="Y", maps_nulls=_custom_y_nulls(nsp, 37),
                    n_perm=100, seed=1, verbose=False)


def test_get_colocalizations_generic_maps_nulls(permuted_maps_nispace):
    nsp = permuted_maps_nispace
    _, nulls = nsp.get_colocalizations(get_nulls=True, nulls_permute_what="maps", verbose=False)
    _, nulls_x = nsp.get_colocalizations(get_nulls=True, nulls_permute_what="xmaps", verbose=False)
    assert nulls.keys() == nulls_x.keys()
    for stat in nulls:
        pd.testing.assert_frame_equal(nulls[stat], nulls_x[stat])


def test_get_normalized_colocalizations_generic_maps(permuted_maps_nispace):
    nsp = permuted_maps_nispace
    nsp.normalize_colocalizations(verbose=False)
    pd.testing.assert_frame_equal(nsp.get_normalized_colocalizations(perm="maps", verbose=False),
                                  nsp.get_normalized_colocalizations(perm="xmaps", verbose=False))


def test_plot_generic_maps(permuted_maps_nispace):
    import matplotlib.pyplot as plt
    permuted_maps_nispace.plot(permute_what="maps", show=False, verbose=False)
    plt.close("all")


def test_get_p_values_generic_maps_ambiguous_raises(permuted_maps_nispace):
    nsp = permuted_maps_nispace
    nsp.permute(what="maps", maps_which="Y", maps_method="random", n_perm=200, seed=1,
                verbose=False)
    with pytest.raises(ValueError, match="ambiguous"):
        nsp.get_p_values(permute_what="maps", verbose=False)
    # explicit sides still work, and the default (last run) is unaffected
    nsp.get_p_values(permute_what="xmaps", verbose=False)
    pd.testing.assert_frame_equal(nsp.get_p_values(verbose=False),
                                  nsp.get_p_values(permute_what="ymaps", verbose=False))


def test_get_p_values_generic_maps_before_permute_raises_keyerror(synthetic_nispace):
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)
    with pytest.raises(KeyError, match="Did you run NiSpace.permute"):
        nsp.get_p_values(permute_what="maps", verbose=False)


def _make_multi_y_nsp(rng, n_parcels=25, n_x=1, n_y=8):
    labels = [f"p{i}" for i in range(n_parcels)]
    X = rng.normal(size=(n_x, n_parcels))
    signal = rng.normal(size=n_x) @ X
    Y = signal[np.newaxis, :] * 0.5 + rng.normal(scale=1.0, size=(n_y, n_parcels))
    x_df = pd.DataFrame(X, index=[f"x{i}" for i in range(n_x)], columns=labels)
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(n_y)], columns=labels)
    nsp = NiSpace(x=x_df, y=y_df, z=None, parcellation=None, standardize=False,
                 n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    return nsp


def test_permute_pooled_p_never_averages_raw_rho():
    """pooled_p="mean"/"median" must pool "rho" on the Fisher-z scale regardless of
    colocalize()'s own r_to_z -- forced, not a free choice (see docstring). Ground
    truth: identical X/Y/seed with r_to_z=False vs r_to_z=True (default) must give
    IDENTICAL pooled p-values, since forcing Fisher-z before pooling makes both cases
    operate on the same numbers internally regardless of what colocalize() stored."""
    nsp_raw = _make_multi_y_nsp(np.random.default_rng(42))
    nsp_raw.colocalize(method="pearson", r_to_z=False, verbose=False)
    nsp_raw.permute(what="maps", maps_method="random", n_perm=300, seed=7,
                    pooled_p="mean", verbose=False)
    p_raw = nsp_raw.get_p_values(verbose=False)

    nsp_z = _make_multi_y_nsp(np.random.default_rng(42))
    nsp_z.colocalize(method="pearson", verbose=False)  # r_to_z=True default
    nsp_z.permute(what="maps", maps_method="random", n_perm=300, seed=7,
                  pooled_p="mean", verbose=False)
    p_z = nsp_z.get_p_values(verbose=False)

    # same underlying data/seed, only colocalize()'s r_to_z differs -- pooled p must
    # be identical, since pooling is now forced onto the Fisher-z scale either way
    np.testing.assert_allclose(p_raw.to_numpy(), p_z.to_numpy())

    # and a sanity check that this isn't a trivial pass: for this data, naive raw
    # averaging of the stored (raw, r_to_z=False) rho actually differs from the
    # correct Fisher-z average -- i.e. the fix has real, measurable effect here
    rho_raw = nsp_raw.get_colocalizations(verbose=False).to_numpy()
    naive_raw_pool = np.nanmean(rho_raw, axis=0)
    correct_z_pool = np.tanh(np.nanmean(np.arctanh(rho_raw), axis=0))
    assert not np.allclose(naive_raw_pool, correct_z_pool, atol=1e-4)


def _make_matched_pairs_nsp(rng, n_pairs=8, n_parcels=20):
    X = rng.normal(size=(n_pairs, n_parcels))
    Y = 0.6 * X + rng.normal(scale=0.5, size=(n_pairs, n_parcels))
    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    pair_labels = [f"p{i}" for i in range(n_pairs)]
    x_df = pd.DataFrame(X, index=pair_labels, columns=parcel_labels)
    y_df = pd.DataFrame(Y, index=pair_labels, columns=parcel_labels)
    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize=False,
                 n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    return nsp


def test_permute_pairs_pooling_never_averages_raw_rho():
    """what="pairs" (SPICE) pools the N×N matrix diagonal across pairs -- same
    forced-Fisher-z requirement as pooled_p above. Ground truth: r_to_z=False vs
    r_to_z=True (default), same data/seed, must give an identical p-value."""
    nsp_raw = _make_matched_pairs_nsp(np.random.default_rng(3))
    nsp_raw.colocalize(method="pearson", r_to_z=False, verbose=False)
    nsp_raw.permute(what="pairs", n_perm=2000, seed=11, verbose=False)
    p_raw = nsp_raw.get_p_values(permute_what="pairs", verbose=False)

    nsp_z = _make_matched_pairs_nsp(np.random.default_rng(3))
    nsp_z.colocalize(method="pearson", verbose=False)  # r_to_z=True default
    nsp_z.permute(what="pairs", n_perm=2000, seed=11, verbose=False)
    p_z = nsp_z.get_p_values(permute_what="pairs", verbose=False)

    np.testing.assert_allclose(p_raw.to_numpy(), p_z.to_numpy())


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


def test_permute_groups_does_not_resolve_null_method(grouped_nispace, caplog):
    # the map-null default is only resolved (and logged) when maps are permuted
    nsp, _ = grouped_nispace
    with caplog.at_level("INFO"):
        nsp.permute(what="groups", n_perm=50, seed=1, verbose=True)
    assert not any("null method" in r.message for r in caplog.records)
    # a maps permutation still resolves it (no parcellation -> 'moran' default)
    caplog.clear()
    with caplog.at_level("INFO"):
        pos = np.arange(20.0)
        nsp.permute(what="maps", dist_mat=np.abs(np.subtract.outer(pos, pos)), n_perm=20,
                    seed=1, verbose=True)
    assert any("defaulting null method to 'moran'" in r.message for r in caplog.records)


def test_get_p_values_after_groups_then_maps_without_pooled_p(grouped_nispace):
    # permute("groups") forces pooled_p="mean"; an earlier/later unpooled maps result must
    # still be retrievable without passing pooled_p
    nsp, _ = grouped_nispace
    p_maps = nsp.permute(what="maps", maps_method="random", n_perm=50, seed=1, verbose=False)
    p_groups = nsp.permute(what="groups", n_perm=50, seed=1, verbose=False)
    assert nsp._last_settings["pooled_p"] == "mean"
    pd.testing.assert_frame_equal(nsp.get_p_values(permute_what="maps", verbose=False), p_maps)
    pd.testing.assert_frame_equal(nsp.get_p_values(permute_what="xmaps", verbose=False), p_maps)
    pd.testing.assert_frame_equal(nsp.get_p_values(permute_what="groups", verbose=False),
                                  p_groups)
    _, nulls = nsp.get_colocalizations(get_nulls=True, nulls_permute_what="xmaps",
                                       verbose=False)
    assert nulls is not None
    nsp.correct_p(mc_method="fdr_bh", verbose=False)
    nsp.get_p_values(permute_what="xmaps", mc_method="fdr_bh", verbose=False)
    assert "pooled_p" not in nsp._nulls


def test_get_p_values_pooled_p_ambiguous_raises(grouped_nispace):
    nsp, _ = grouped_nispace
    nsp.permute(what="maps", maps_method="random", n_perm=50, seed=1, pooled_p="mean",
                verbose=False)
    nsp.permute(what="maps", maps_method="random", n_perm=50, seed=1, pooled_p="median",
                verbose=False)
    nsp.permute(what="groups", n_perm=50, seed=1, verbose=False)
    # last pooled_p ("mean", from groups) exists for maps too -> taken
    nsp.get_p_values(permute_what="xmaps", verbose=False)
    nsp._last_settings["pooled_p"] = False  # no unpooled maps result: mean and median both exist
    with pytest.raises(ValueError, match="ambiguous"):
        nsp.get_p_values(permute_what="xmaps", verbose=False)
    nsp.get_p_values(permute_what="xmaps", pooled_p="median", verbose=False)


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


_FAMILYWISE_MC = ["maxT", "step_maxT", "minP", "step_minP"]


@pytest.mark.parametrize("mc_method", _FAMILYWISE_MC)
def test_correct_p_familywise_requires_stored_nulls(permuted_maps_nispace, mc_method):
    """maxT/step_maxT/minP/step_minP need the full null distribution --
    to_pickle(save_nulls=False) drops it, and correct_p() on the reloaded object
    must raise a clear KeyError rather than silently using something else."""
    nsp = permuted_maps_nispace
    nsp.correct_p(mc_method=mc_method, verbose=False)  # works while nulls are present
    mc_key = mc_method.replace("_", "").lower()
    assert nsp._check_permute("pearson", "xmaps", mc_method=mc_key, raise_error=False)

    import tempfile, os
    f = tempfile.mktemp(suffix=".pkl")
    try:
        nsp.to_pickle(f, save_nulls=False, verbose=False)
        nsp_reloaded = NiSpace.from_pickle(f, verbose=False)
        with pytest.raises(KeyError):
            nsp_reloaded.correct_p(mc_method=mc_method, verbose=False)
    finally:
        if os.path.exists(f):
            os.remove(f)


@pytest.mark.parametrize("mc_method", _FAMILYWISE_MC)
def test_correct_p_familywise_at_least_as_conservative_as_uncorrected(permuted_maps_nispace,
                                                                       mc_method):
    nsp = permuted_maps_nispace
    p_raw = nsp.get_p_values(verbose=False)
    nsp.correct_p(mc_method=mc_method, verbose=False)
    p_corr = nsp.get_corrected_p_values(verbose=False)
    assert p_corr.shape == p_raw.shape
    assert (p_corr.index == p_raw.index).all() and (p_corr.columns == p_raw.columns).all()
    assert (p_corr.to_numpy() >= p_raw.to_numpy() - 1e-8).all()


@pytest.mark.parametrize("mc_method", ["minP", "step_minP"])
def test_correct_p_minp_mc_dimension_x_raises(permuted_maps_nispace, mc_method):
    with pytest.raises(ValueError, match="not supported"):
        permuted_maps_nispace.correct_p(mc_method=mc_method, mc_dimension="x", verbose=False)


@pytest.mark.parametrize("mc_method", _FAMILYWISE_MC)
def test_correct_p_familywise_pooled_p_uses_pooled_observed(mc_method):
    """With several Y rows, pooled_p="mean" tests the mean colocalization: the stored
    uncorrected p (and the null) is one "mean" row. The family-wise correction must
    compare the POOLED observed statistic to that null -- it used to compare every
    unpooled Y row to it and return an (n_Y, n_X) array under the pooled key.
    Pooling must also happen on the Fisher-z scale, so colocalize()'s r_to_z must not
    change the corrected p (same ground truth as
    test_permute_pooled_p_never_averages_raw_rho)."""
    p_corr = {}
    for r_to_z in (True, False):
        nsp = _make_multi_y_nsp(np.random.default_rng(42), n_x=4, n_y=6)
        nsp.colocalize(method="pearson", r_to_z=r_to_z, verbose=False)
        nsp.permute(what="maps", maps_method="random", n_perm=300, seed=7,
                    pooled_p="mean", verbose=False)
        p_raw = nsp.get_p_values(verbose=False)
        nsp.correct_p(mc_method=mc_method, verbose=False)
        p_corr[r_to_z] = nsp.get_p_values(mc_method=mc_method, verbose=False)
        assert p_corr[r_to_z].shape == p_raw.shape == (1, 4)
        assert list(p_corr[r_to_z].index) == ["mean"]
        assert (p_corr[r_to_z].to_numpy() >= p_raw.to_numpy() - 1e-8).all()
    np.testing.assert_allclose(p_corr[True].to_numpy(), p_corr[False].to_numpy())


@pytest.fixture
def nsp_xsea_unequal_sets(rng):
    """XSEA X with one 2-map and one 50-map set of independent random maps: the
    set-mean null of the small set is ~sqrt(25)=5x wider than the big set's (well
    above the 3x warning threshold)."""
    n_parcels = 30
    tuples = [("small", f"g{i}") for i in range(2)] + [("big", f"g{i}") for i in range(2, 52)]
    idx = pd.MultiIndex.from_tuples(tuples, names=["set", "gene"])
    X = rng.normal(size=(len(tuples), n_parcels))
    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    x_df = pd.DataFrame(X, index=idx, columns=parcel_labels)
    y_df = pd.DataFrame(rng.normal(size=(1, n_parcels)), index=["y0"], columns=parcel_labels)
    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    nsp.colocalize(method="pearson", xsea=True, xsea_aggregation_method="mean", verbose=False)
    nsp.permute(what="maps", maps_method="random", n_perm=300, seed=3, verbose=False)
    return nsp


def test_correct_p_maxt_warns_on_heterogeneous_xsea_sets(nsp_xsea_unequal_sets, caplog):
    nsp = nsp_xsea_unequal_sets
    with caplog.at_level("WARNING"):
        nsp.correct_p(mc_method="maxT", verbose=False)
    msgs = [r.message for r in caplog.records if "differ in width" in r.message]
    assert msgs and "XSEA" in msgs[0] and "minP" in msgs[0]
    assert "'small'" in msgs[0] and "'big'" in msgs[0]

    caplog.clear()
    with caplog.at_level("WARNING"):
        nsp.correct_p(mc_method="minP", verbose=False)
    assert not any("differ in width" in r.message for r in caplog.records)


def test_correct_p_maxt_no_scale_warning_for_comparable_columns(permuted_maps_nispace, caplog):
    with caplog.at_level("WARNING"):
        permuted_maps_nispace.correct_p(mc_method="maxT", verbose=False)
    assert not any("differ in width" in r.message for r in caplog.records)


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


def test_pickle_omits_coloc_closures_and_reload_rebuilds_them(synthetic_nispace, tmp_path):
    """Colocalization closures are dill-pickled by value (raw bytecode), which breaks across
    Python versions (a 3.9 pickle raised NameError: 'zeros' in permute() under 3.11). They
    must not be written; from_pickle() rebuilds them, and permute() gives the same nulls."""
    from nispace.io import from_pickle as raw_from_pickle
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)
    f = tmp_path / "nsp.pkl"
    nsp.to_pickle(str(f), verbose=False)

    assert raw_from_pickle(str(f), use_dill=True)._colocs_fun == {}
    assert "pearson" in nsp._colocs_fun   # in-memory object untouched

    nsp_reloaded = NiSpace.from_pickle(str(f), verbose=False)
    assert callable(nsp_reloaded._colocs_fun["pearson"])
    for obj in (nsp, nsp_reloaded):
        obj.permute(what="maps", maps_method="random", n_perm=50, seed=3, verbose=False)
    pd.testing.assert_frame_equal(nsp_reloaded.get_p_values(verbose=False),
                                  nsp.get_p_values(verbose=False))


def test_from_pickle_replaces_stale_pickled_closures(synthetic_nispace, tmp_path, monkeypatch):
    """Old pickles still carry closures: they are always rebuilt from the stored kwargs,
    never reused; a method without stored kwargs is dropped with a warning."""
    import nispace.api as api_mod
    from nispace.io import to_pickle as raw_to_pickle
    nsp = synthetic_nispace
    nsp.colocalize(method="pearson", verbose=False)

    def _broken(X, y, weights=None):
        raise NameError("name 'zeros' is not defined")

    nsp._colocs_fun["pearson"] = _broken
    nsp._colocs_fun["spearman"] = _broken   # no stored kwargs for this one
    del nsp._coloc_seed_by_method           # pre-seed-storage pickle
    f = tmp_path / "old.pkl"
    raw_to_pickle(nsp, str(f), use_dill=True)

    warnings = []
    monkeypatch.setattr(api_mod.lgr, "warning", lambda msg, *a, **k: warnings.append(str(msg)))
    nsp_reloaded = NiSpace.from_pickle(str(f), verbose=False)

    assert nsp_reloaded._colocs_fun["pearson"] is not _broken
    assert "spearman" not in nsp_reloaded._colocs_fun
    assert any("spearman" in w and "colocalize()" in w for w in warnings)
    nsp_reloaded.permute(what="maps", maps_method="random", n_perm=20, seed=1, verbose=False)
    assert nsp_reloaded.get_p_values(verbose=False).notna().all().all()
