"""Tests for nispace.core.permute -- the null-map cache/dispatch layer and
the table-driven `what`/pooled_p validation used by NiSpace.permute().

All tests here stay network- and parcellation-free: null generation is
exercised via method="random" (no distance matrix needed at all) and
method="moran" with an explicitly-supplied synthetic distance matrix
(parc=None), rather than a real Parcellation object.
"""

import numpy as np
import pandas as pd
import pytest

from nispace.core.nullmaps import NullMaps
from nispace.core.permute import (
    _null_method_key,
    _get_correct_mc_method,
    _resolve_permute_mode_settings,
    _resolve_permute_combo,
    _get_null_maps,
    _get_exact_p_values,
)
from nispace.stats.misc import null_to_p


# ── _null_method_key ────────────────────────────────────────────────────────

@pytest.mark.parametrize("method,expected", [
    (None, ""),
    ("moran", "moran"),
    (("spin", "moran"), "spin+moran"),
])
def test_null_method_key(method, expected):
    assert _null_method_key(method) == expected


# ── _get_correct_mc_method ───────────────────────────────────────────────────

@pytest.mark.parametrize("alias,expected", [
    ("fdr", "fdr_bh"),
    ("bonf", "bonferroni"),
    ("fdrbh", "fdr_bh"),
    ("meff", "meff_galwey"),
    ("meff_li_ji", "meff_li_ji"),
    ("maxt", "maxT"),
    ("fdr_bh", "fdr_bh"),  # already-canonical name passes through
    ("not_an_alias", "not_an_alias"),  # unknown method passes through unchanged
])
def test_get_correct_mc_method(alias, expected):
    assert _get_correct_mc_method(alias) == expected


# ── _resolve_permute_mode_settings ──────────────────────────────────────────

def test_resolve_permute_mode_settings_maps_is_free_choice():
    pooled, warning = _resolve_permute_mode_settings(["maps"], "median")
    assert pooled == "median"
    assert warning is None


@pytest.mark.parametrize("requested", ["auto", "mean"])
def test_resolve_permute_mode_settings_groups_accepts_forced_value(requested):
    pooled, warning = _resolve_permute_mode_settings(["groups"], requested)
    assert pooled == "mean"
    assert warning is None


def test_resolve_permute_mode_settings_groups_overrides_mismatch_with_warning():
    pooled, warning = _resolve_permute_mode_settings(["groups"], "median")
    assert pooled == "mean"
    assert warning is not None
    assert "median" in warning and "mean" in warning


def test_resolve_permute_mode_settings_groups_forced_in_combo():
    pooled, warning = _resolve_permute_mode_settings(["groups", "sets"], "auto")
    assert pooled == "mean"
    assert warning is None


# ── _resolve_permute_combo ──────────────────────────────────────────────────

def test_resolve_permute_combo_maps_only():
    what, perm_info, maps_which, warnings = _resolve_permute_combo(["maps"], ["X"])
    assert set(what) == {"maps"}
    assert maps_which == ["X"]
    assert perm_info == "X maps"
    assert warnings == []


def test_resolve_permute_combo_groups_only_has_fixed_perm_info():
    what, perm_info, maps_which, warnings = _resolve_permute_combo(["groups"], None)
    assert set(what) == {"groups"}
    assert perm_info == "Y groups"
    assert maps_which is None
    assert warnings == []


def test_resolve_permute_combo_forces_maps_which_with_warning():
    what, perm_info, maps_which, warnings = _resolve_permute_combo(["groups", "maps"], ["Y"])
    assert maps_which == ["X"]  # forced regardless of what was requested
    assert len(warnings) == 1
    assert "maps_which" in warnings[0]


def test_resolve_permute_combo_forced_maps_which_no_warning_when_already_correct():
    what, perm_info, maps_which, warnings = _resolve_permute_combo(["groups", "maps"], ["X"])
    assert maps_which == ["X"]
    assert warnings == []


def test_resolve_permute_combo_three_way_falls_back_with_warning():
    what, perm_info, maps_which, warnings = _resolve_permute_combo(
        ["groups", "maps", "sets"], ["X"]
    )
    assert set(what) == {"groups", "sets"}
    assert len(warnings) == 1
    assert "groups" in warnings[0] and "sets" in warnings[0]


def test_resolve_permute_combo_invalid_raises():
    with pytest.raises(ValueError, match="not defined"):
        _resolve_permute_combo(["pairs", "maps"], ["X"])


# ── _get_null_maps: fresh generation (offline) ──────────────────────────────

@pytest.fixture
def toy_data_obs(rng):
    return pd.DataFrame(rng.normal(size=(2, 10)), index=["x0", "x1"])


@pytest.fixture
def toy_dist_mat(rng):
    """Synthetic symmetric distance matrix for 10 parcels (no real geometry needed)."""
    coords = rng.normal(size=(10, 3))
    return np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))


def test_get_null_maps_random_generates_correct_shape(toy_data_obs):
    null_maps, spin = _get_null_maps(
        toy_data_obs, {}, null_method="random", n_perm=20, seed=42, n_proc=1, verbose=False,
    )
    assert isinstance(null_maps, NullMaps)
    assert null_maps.shape == (2, 20, 10)
    assert null_maps.labels == ["x0", "x1"]
    assert null_maps.null_method == "random"
    assert spin is None  # only spin methods promote a new spin_mat


def test_get_null_maps_moran_with_explicit_dist_mat(toy_data_obs, toy_dist_mat):
    null_maps, spin = _get_null_maps(
        toy_data_obs, {}, null_method="moran", n_perm=15, dist_mat=toy_dist_mat,
        seed=42, n_proc=1, verbose=False,
    )
    assert null_maps.shape == (2, 15, 10)
    assert null_maps.null_method == "moran"
    assert spin is None


def test_get_null_maps_standardize_zscores_each_perm_row(toy_data_obs):
    null_maps, _ = _get_null_maps(
        toy_data_obs, {}, null_method="random", n_perm=10, seed=42, n_proc=1, verbose=False,
        standardize=True,
    )
    # each (map, perm) slice across parcels should be ~zero-mean, unit-std
    assert np.allclose(null_maps.data.mean(axis=2), 0, atol=1e-5)
    assert np.allclose(null_maps.data.std(axis=2), 1, atol=1e-5)


# ── _get_null_maps: caching / validity checks ───────────────────────────────

@pytest.fixture
def cached_null_maps(rng):
    """A NullMaps covering 3 labels (superset of what tests need), method='random'."""
    data = rng.normal(size=(3, 20, 5)).astype(np.float32)
    return NullMaps(data, ["x0", "x1", "x2"], null_method="random")


@pytest.fixture
def small_data_obs(rng):
    return pd.DataFrame(rng.normal(size=(2, 5)), index=["x0", "x1"])


def test_get_null_maps_reuses_and_subsets_cache(small_data_obs, cached_null_maps):
    null_maps, _ = _get_null_maps(
        small_data_obs, {"maps_null": cached_null_maps}, null_method="random", n_perm=20,
        seed=42, n_proc=1, verbose=False, standardize=False,
    )
    # exact match to the cached arrays proves this was reused, not regenerated
    assert null_maps.labels == ["x0", "x1"]
    assert np.array_equal(null_maps["x0"], cached_null_maps["x0"])
    assert np.array_equal(null_maps["x1"], cached_null_maps["x1"])


def test_get_null_maps_custom_maps_skip_method_check(small_data_obs):
    """Custom null_maps bypass generation entirely -- an invalid null_method
    must not raise, because generate_null_maps() is never called."""
    custom = {lbl: np.random.default_rng(0).normal(size=(20, 5)) for lbl in ["x0", "x1"]}
    null_maps, _ = _get_null_maps(
        small_data_obs, {}, null_maps=custom, null_method="not_a_real_method",
        n_perm=20, seed=42, n_proc=1, verbose=False, standardize=False,
    )
    assert isinstance(null_maps, NullMaps)
    assert null_maps.labels == ["x0", "x1"]


def test_get_null_maps_regenerates_on_missing_label(small_data_obs, cached_null_maps):
    data_obs_new_label = pd.DataFrame(
        np.random.default_rng(1).normal(size=(2, 5)), index=["x0", "xNEW"]
    )
    null_maps, _ = _get_null_maps(
        data_obs_new_label, {"maps_null": cached_null_maps}, null_method="random", n_perm=20,
        seed=42, n_proc=1, verbose=False, standardize=False,
    )
    assert null_maps.labels == ["x0", "xNEW"]
    # regenerated, so x0 no longer matches the stale cache
    assert not np.array_equal(null_maps["x0"], cached_null_maps["x0"])


def test_get_null_maps_regenerates_on_n_perm_increase(small_data_obs, cached_null_maps):
    null_maps, _ = _get_null_maps(
        small_data_obs, {"maps_null": cached_null_maps}, null_method="random", n_perm=50,
        seed=42, n_proc=1, verbose=False, standardize=False,
    )
    assert null_maps.n_perm == 50


def test_get_null_maps_regenerates_on_method_change(small_data_obs, cached_null_maps):
    dist_mat = np.abs(np.random.default_rng(2).normal(size=(5, 5)))
    null_maps, _ = _get_null_maps(
        small_data_obs, {"maps_null": cached_null_maps}, null_method="moran", n_perm=20,
        dist_mat=dist_mat, seed=42, n_proc=1, verbose=False, standardize=False,
    )
    assert null_maps.null_method == "moran"
    assert not np.array_equal(null_maps["x0"], cached_null_maps["x0"])


# ── _get_exact_p_values ──────────────────────────────────────────────────────

def test_get_exact_p_values_matches_manual_null_to_p(rng):
    colocs_obs = {"rho": np.array([[0.5, 0.2], [0.1, -0.3]])}
    colocs_null = [
        {"rho": rng.normal(size=(2, 2))} for _ in range(99)
    ]
    p_data, p_tails = _get_exact_p_values("pearson", colocs_obs, colocs_null, verbose=False)

    assert p_tails == {"rho": "two"}
    expected = np.zeros((2, 2))
    for y in range(2):
        for x in range(2):
            obs = colocs_obs["rho"][y, x]
            null = [colocs_null[i]["rho"][y, x] for i in range(99)]
            expected[y, x] = null_to_p(obs, null, tail="two")
    assert np.allclose(p_data["rho"], expected)


def test_get_exact_p_values_drops_individual_for_mlr_without_it():
    # mlr's permuted stats are ["r2", "beta", "individual"]; if colocs_obs lacks
    # "individual" the function must not try to compute p for it.
    colocs_obs = {"r2": np.array([[0.4]]), "beta": np.array([[0.4]])}
    colocs_null = [{"r2": np.array([[0.1]]), "beta": np.array([[0.1]])} for _ in range(10)]
    p_data, _ = _get_exact_p_values("mlr", colocs_obs, colocs_null, verbose=False)
    assert set(p_data.keys()) == {"r2", "beta"}


def test_get_exact_p_values_string_p_tails_requires_single_stat():
    # pearson has one stat ("rho"), so a bare string is accepted as-is.
    colocs_obs = {"rho": np.array([[0.5]])}
    colocs_null = [{"rho": np.array([[0.1]])} for _ in range(10)]
    _, p_tails = _get_exact_p_values("pearson", colocs_obs, colocs_null, p_tails="upper", verbose=False)
    assert p_tails == {"rho": "upper"}


def test_get_exact_p_values_xsea_abs_forces_upper_tail():
    colocs_obs = {"rho": np.array([[0.5]])}
    colocs_null = [{"rho": np.array([[0.1]])} for _ in range(10)]
    _, p_tails = _get_exact_p_values(
        "pearson", colocs_obs, colocs_null, xsea_aggr="abs_mean", verbose=False
    )
    assert p_tails == {"rho": "upper"}
