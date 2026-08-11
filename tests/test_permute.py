"""Tests for nispace.core.permute -- the null-map cache/dispatch layer and
the table-driven `what`/pooled_p validation used by NiSpace.permute().

All tests here stay network- and parcellation-free: null generation is
exercised via method="random" (no distance matrix needed at all) and
method="moran" with an explicitly-supplied synthetic distance matrix
(parc=None), rather than a real Parcellation object.
"""

import logging

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
    _resolve_maps_batch_size,
    _iter_null_map_batches,
)
from nispace.stats.misc import null_to_p
from nispace import NiSpace


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


# ── _resolve_maps_batch_size ─────────────────────────────────────────────────

def test_resolve_maps_batch_size_false_disables():
    assert _resolve_maps_batch_size(12000, 1000, 1000, np.float32, False) is None
    assert _resolve_maps_batch_size(12000, 1000, 1000, np.float32, 0) is None


def test_resolve_maps_batch_size_explicit_int_clipped_to_n_data():
    assert _resolve_maps_batch_size(10, 1000, 1000, np.float32, 500) == 10
    assert _resolve_maps_batch_size(500, 1000, 1000, np.float32, 50) == 50


def test_resolve_maps_batch_size_adaptive_small_n_data_is_noop():
    # small n_data -> resolved size >= n_data -> caller's "n_data > resolved" is False
    n_data = 10
    resolved = _resolve_maps_batch_size(n_data, 1000, 1000, np.float32, None)
    assert resolved >= n_data


def test_resolve_maps_batch_size_adaptive_large_n_data_is_bounded():
    resolved = _resolve_maps_batch_size(12000, 1000, 1000, np.float32, None)
    assert resolved < 12000
    assert resolved >= 50  # _MAPS_BATCH_MIN_ROWS floor


def test_resolve_maps_batch_size_adaptive_never_below_min_rows_floor():
    # pathologically large n_perm*n_parcels -> memory target implies <1 row/batch;
    # must still floor at _MAPS_BATCH_MIN_ROWS, not collapse to 0/1
    resolved = _resolve_maps_batch_size(100000, 100000, 100000, np.float64, None)
    assert resolved >= 50


# ── _iter_null_map_batches: batched vs. unbatched equivalence ───────────────

@pytest.fixture
def many_rows_data_obs(rng):
    """137 rows (deliberately not a multiple of any nice batch size), 25 parcels."""
    data = rng.normal(size=(137, 25)).astype(np.float32)
    return pd.DataFrame(data, index=[f"gene{i}" for i in range(137)])


@pytest.fixture
def many_rows_moran_dist_mat(rng):
    coords = rng.uniform(0, 100, size=(25, 2))
    return np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))


def test_iter_null_map_batches_seed_offset_matches_unbatched(
        many_rows_data_obs, many_rows_moran_dist_mat):
    from nispace.nulls import generate_null_maps

    seed_base = 4242
    n_perm = 20
    ref, _ = generate_null_maps(
        method="moran", data=many_rows_data_obs, parcellation=None,
        dist_mat=many_rows_moran_dist_mat.copy(), parc_space="mni152",
        n_nulls=n_perm, seed=seed_base, n_proc=1, verbose=False, dtype=np.float32,
    )
    batches = list(_iter_null_map_batches(
        data_obs=many_rows_data_obs, batch_size=37, seed_base=seed_base, nispace_nulls={},
        standardize=False, n_perm=n_perm, dist_mat=many_rows_moran_dist_mat.copy(), parc=None,
        permute_which="X", dtype=np.float32, n_proc=1, verbose=False, null_method="moran",
    ))
    assert [b[1] - b[0] for b in batches] == [37, 37, 37, 26]  # boundary handling
    for row_start, row_end, batch_nm in batches:
        np.testing.assert_array_equal(batch_nm.data, ref.data[row_start:row_end])


def test_iter_null_map_batches_random_method_matches_unbatched(
        many_rows_data_obs, many_rows_moran_dist_mat):
    # method="random" needs no real geometry -- a second, cheap sanity check independent
    # of moran's fit-once machinery
    from nispace.nulls import generate_null_maps

    seed_base = 7
    n_perm = 15
    ref, _ = generate_null_maps(
        method="random", data=many_rows_data_obs, parcellation=None, dist_mat=None,
        n_nulls=n_perm, seed=seed_base, n_proc=1, verbose=False, dtype=np.float32,
    )
    batches = list(_iter_null_map_batches(
        data_obs=many_rows_data_obs, batch_size=50, seed_base=seed_base, nispace_nulls={},
        standardize=False, n_perm=n_perm, dist_mat=None, parc=None, permute_which="X",
        dtype=np.float32, n_proc=1, verbose=False, null_method="random",
    ))
    for row_start, row_end, batch_nm in batches:
        np.testing.assert_array_equal(batch_nm.data, ref.data[row_start:row_end])


# ── NiSpace.permute() row-batched fast path: end-to-end equivalence ─────────
#
# These build small-but-forced-batching NiSpace objects (parcellation=None,
# maps_method="random" -- no real geometry needed, matches this file's existing
# offline convention) and compare maps_batch_size=<small int> against
# maps_batch_size=False (explicit opt-out) for exact equivalence.

def _make_nsp(x_df, y_df, z_df=None, method="pearson", **coloc_kwargs):
    nsp = NiSpace(x=x_df, y=y_df, z=z_df, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    nsp.colocalize(method, verbose=False, **coloc_kwargs)
    return nsp


@pytest.fixture
def fastpath_x_frames(rng):
    n_parcels, n_x, n_y = 20, 23, 4  # n_x not a multiple of any nice batch size
    X = rng.normal(size=(n_x, n_parcels))
    Y = rng.normal(size=(n_y, n_parcels))
    x_df = pd.DataFrame(X, index=[f"gene{i}" for i in range(n_x)],
                        columns=[f"p{i}" for i in range(n_parcels)])
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(n_y)],
                        columns=[f"p{i}" for i in range(n_parcels)])
    return x_df, y_df


def test_permute_maps_x_univariate_batched_matches_unbatched(fastpath_x_frames):
    x_df, y_df = fastpath_x_frames
    nsp_b = _make_nsp(x_df, y_df, method="pearson")
    pb = nsp_b.permute("maps", maps_which="X", maps_method="random", n_perm=50, seed=123,
                       maps_batch_size=5, pooled_p=False, verbose=False)
    nsp_u = _make_nsp(x_df, y_df, method="pearson")
    pu = nsp_u.permute("maps", maps_which="X", maps_method="random", n_perm=50, seed=123,
                       maps_batch_size=False, pooled_p=False, verbose=False)
    np.testing.assert_allclose(pb.values, pu.values)
    # cache correctness: batched run must NOT leave a stale/full maps_null cache
    assert nsp_b._nulls.get("maps_null") is None
    assert nsp_b._nulls.get("maps_null_fastpath_info") is not None
    assert nsp_u._nulls.get("maps_null") is not None
    assert nsp_u._nulls.get("maps_null_fastpath_info") is None


def test_permute_maps_x_batched_does_not_repeat_large_nullmaps_warning(
        monkeypatch, fastpath_x_frames, caplog):
    # regression: NullMaps.standardize()/astype()/subset() each build a fresh NullMaps
    # internally but didn't forward warn_large=False -- so _get_null_maps()'s
    # post-generation .standardize() call re-triggered the generic ">1GB, consider
    # memmap_path" warning with its own default (True), once per batch, even though
    # each batch's array was deliberately kept small. Lower the threshold so a tiny
    # test array crosses it, and require standardize=True (the default) so
    # .standardize() actually runs.
    import nispace.core.nullmaps as nullmaps_module
    monkeypatch.setattr(nullmaps_module, "_NULLMAPS_WARN_BYTES", 1)

    x_df, y_df = fastpath_x_frames  # n_x=23, standardize batching threshold: batch_size=5
    nsp_b = NiSpace(x=x_df, y=y_df, z=None, parcellation=None, n_proc=1, verbose=False,
                    return_self=False)  # standardize defaults True -- exercises .standardize()
    nsp_b.fit()
    nsp_b.colocalize("pearson", verbose=False)
    with caplog.at_level(logging.WARNING):
        nsp_b.permute("maps", maps_which="X", maps_method="random", n_perm=20, seed=1,
                     maps_batch_size=5, verbose=False)
    assert "GB in memory" not in caplog.text

    caplog.clear()
    nsp_u = NiSpace(x=x_df, y=y_df, z=None, parcellation=None, n_proc=1, verbose=False,
                    return_self=False)
    nsp_u.fit()
    nsp_u.colocalize("pearson", verbose=False)
    with caplog.at_level(logging.WARNING):
        nsp_u.permute("maps", maps_which="X", maps_method="random", n_perm=20, seed=1,
                     maps_batch_size=False, verbose=False)
    # unbatched: warns exactly once (at construction), not twice (construction +
    # the post-generation .standardize() call) -- the same fix also closed that
    # pre-existing duplicate-warning bug on the unbatched path
    assert caplog.text.count("GB in memory") == 1


def test_permute_maps_x_batched_rank_true_no_z_does_not_crash(fastpath_x_frames):
    # regression test: found via a real-pickle check (intro02_nsp.pkl.blosc, spearman
    # -> rank=True, no Z ever provided). rank=True alone (no regress_z) still enters
    # the "if rank or regress_z:" branch inside the fast path, which used to
    # unconditionally read _Z_obs_arr.shape[0] to decide whether to slice Z for
    # zy_matched -- but with no Z, _Z_obs_arr is np.array(None), a 0-d array with no
    # [0] index, crashing with IndexError. method="spearman" forces rank=True with no
    # Z/regress_z involved at all, exactly reproducing the real-data crash condition.
    x_df, y_df = fastpath_x_frames
    nsp_b = _make_nsp(x_df, y_df, z_df=None, method="spearman")
    pb = nsp_b.permute("maps", maps_which="X", maps_method="random", n_perm=30, seed=9,
                       maps_batch_size=5, pooled_p=False, verbose=False)
    nsp_u = _make_nsp(x_df, y_df, z_df=None, method="spearman")
    pu = nsp_u.permute("maps", maps_which="X", maps_method="random", n_perm=30, seed=9,
                       maps_batch_size=False, pooled_p=False, verbose=False)
    np.testing.assert_allclose(pb.values, pu.values)


def test_permute_maps_fastpath_task_payload_size_independent_of_n_perm(
        monkeypatch, fastpath_x_frames):
    # regression: the fast-path closures used to receive the ENTIRE per-batch
    # `batch_perm_list` (all n_perm arrays) as a `delayed()` argument, indexing into
    # it (`batch_perm_list[i]`) *inside* the worker -- so joblib re-serialized/shipped
    # the whole batch on every one of the n_perm task dispatches instead of once.
    # Found via real usage: a job that ran in 20s unbatched never finished batched
    # (loky workers eventually got killed for excessive memory/timeouts). The fix
    # slices `batch_perm_list[i]` in the parent process before dispatch (mirroring
    # how the unbatched path always only passed `_X_null[i], _Y_null[i]`), so each
    # dispatched task's payload size must depend only on batch_size/n_parcels, never
    # on n_perm -- verified by comparing payload size at two very different n_perm.
    import pickle
    import nispace.api as api_module
    x_df, y_df = fastpath_x_frames  # n_x=23, n_parcels=20

    def max_task_payload_bytes(n_perm):
        seen = []
        real_delayed = api_module.delayed

        def spy_delayed(fn):
            wrapped = real_delayed(fn)
            def call(*args, **kwargs):
                seen.append(sum(len(pickle.dumps(a)) for a in args))
                return wrapped(*args, **kwargs)
            return call

        with monkeypatch.context() as m:
            m.setattr(api_module, "delayed", spy_delayed)
            nsp = _make_nsp(x_df, y_df, method="pearson")
            nsp.permute("maps", maps_which="X", maps_method="random", n_perm=n_perm,
                       seed=3, maps_batch_size=5, pooled_p=False, verbose=False)
        return max(seen)

    small = max_task_payload_bytes(8)
    large = max_task_payload_bytes(80)
    # payload must not scale with n_perm (10x here) -- generous headroom for pickle/
    # object overhead noise, but the bug would blow this up by ~10x
    assert large < small * 2


@pytest.fixture
def fastpath_y_frames(rng):
    n_parcels, n_x, n_y = 20, 3, 27  # n_y not a multiple of any nice batch size
    X = rng.normal(size=(n_x, n_parcels))
    Y = rng.normal(size=(n_y, n_parcels))
    x_df = pd.DataFrame(X, index=[f"x{i}" for i in range(n_x)],
                        columns=[f"p{i}" for i in range(n_parcels)])
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(n_y)],
                        columns=[f"p{i}" for i in range(n_parcels)])
    return x_df, y_df


@pytest.mark.parametrize("method,pooled_p", [
    ("pearson", False), ("pearson", "mean"),
    ("mlr", False), ("mlr", "mean"),  # multivariate -- Y-side must work regardless
])
def test_permute_maps_y_any_method_batched_matches_unbatched(fastpath_y_frames, method, pooled_p):
    x_df, y_df = fastpath_y_frames
    nsp_b = _make_nsp(x_df, y_df, method=method)
    pb = nsp_b.permute("maps", maps_which="Y", maps_method="random", n_perm=40, seed=55,
                       maps_batch_size=7, pooled_p=pooled_p, verbose=False, force_dict=True)
    nsp_u = _make_nsp(x_df, y_df, method=method)
    pu = nsp_u.permute("maps", maps_which="Y", maps_method="random", n_perm=40, seed=55,
                       maps_batch_size=False, pooled_p=pooled_p, verbose=False, force_dict=True)
    assert set(pb.keys()) == set(pu.keys())
    for stat in pb:
        np.testing.assert_allclose(pb[stat].values, pu[stat].values)


def test_permute_maps_y_batched_rank_true_no_z_does_not_crash(fastpath_y_frames):
    # Y-side counterpart of the X-side no-Z regression above -- same fix applies to
    # both branches.
    x_df, y_df = fastpath_y_frames
    nsp_b = _make_nsp(x_df, y_df, z_df=None, method="spearman")
    pb = nsp_b.permute("maps", maps_which="Y", maps_method="random", n_perm=30, seed=9,
                       maps_batch_size=7, pooled_p=False, verbose=False)
    nsp_u = _make_nsp(x_df, y_df, z_df=None, method="spearman")
    pu = nsp_u.permute("maps", maps_which="Y", maps_method="random", n_perm=30, seed=9,
                       maps_batch_size=False, pooled_p=False, verbose=False)
    np.testing.assert_allclose(pb.values, pu.values)


def test_permute_maps_y_batched_zy_matched_matches_unbatched(rng):
    # targeted regression test for the seed/z-slicing risk: _rank_regress's zy_matched
    # branch indexes z[i] by LOCAL row position -- a batched call must translate that to
    # the batch's GLOBAL row range, or this would silently apply the wrong Z rows.
    n_parcels, n_x, n_y = 18, 2, 25
    X = rng.normal(size=(n_x, n_parcels))
    Y = rng.normal(size=(n_y, n_parcels))
    Z = rng.normal(size=(n_y, n_parcels))  # one Z row per Y row (zy_matched requirement)
    x_df = pd.DataFrame(X, index=[f"x{i}" for i in range(n_x)],
                        columns=[f"p{i}" for i in range(n_parcels)])
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(n_y)],
                        columns=[f"p{i}" for i in range(n_parcels)])
    z_df = pd.DataFrame(Z, index=[f"z{i}" for i in range(n_y)],
                        columns=[f"p{i}" for i in range(n_parcels)])

    nsp_b = _make_nsp(x_df, y_df, z_df, method="pearson", regress_z=True, zy_matched=True)
    pb = nsp_b.permute("maps", maps_which="Y", maps_method="random", n_perm=30, seed=88,
                       maps_batch_size=6, pooled_p=False, verbose=False)
    nsp_u = _make_nsp(x_df, y_df, z_df, method="pearson", regress_z=True, zy_matched=True)
    pu = nsp_u.permute("maps", maps_which="Y", maps_method="random", n_perm=30, seed=88,
                       maps_batch_size=False, pooled_p=False, verbose=False)
    np.testing.assert_allclose(pb.values, pu.values)


def test_permute_maps_xsea_x_batched_matches_unbatched(rng):
    # the realistic full-transcriptome-XSEA scenario motivating this fast path: X has
    # set structure with genes repeated across sets (exercises dedupe_rows + the
    # (set,gene)->unique-row mapping under batching).
    n_parcels = 15
    parcel_labels = [f"p{i}" for i in range(n_parcels)]
    n_genes_unique = 33
    gene_names = [f"gene{i}" for i in range(n_genes_unique)]
    gene_vals = {g: rng.normal(size=n_parcels) for g in gene_names}
    tuples = ([("setA", g) for g in gene_names[:20]]
              + [("setB", g) for g in gene_names[10:30]]   # overlaps setA
              + [("setC", g) for g in gene_names[25:33]])  # overlaps setB
    idx = pd.MultiIndex.from_tuples(tuples, names=["set", "gene"])
    X = np.stack([gene_vals[g] for _, g in tuples])
    x_xsea = pd.DataFrame(X, index=idx, columns=parcel_labels)
    y_df = pd.DataFrame(rng.normal(size=(3, n_parcels)), index=["y0", "y1", "y2"],
                        columns=parcel_labels)

    nsp_b = _make_nsp(x_xsea, y_df, method="pearson", xsea=True)
    pb = nsp_b.permute(what="maps", maps_which="X", maps_method="random", n_perm=60, seed=321,
                       maps_batch_size=7, pooled_p=False, verbose=False)
    nsp_u = _make_nsp(x_xsea, y_df, method="pearson", xsea=True)
    pu = nsp_u.permute(what="maps", maps_which="X", maps_method="random", n_perm=60, seed=321,
                       maps_batch_size=False, pooled_p=False, verbose=False)
    assert list(pb.columns) == list(pu.columns)
    np.testing.assert_allclose(pb.values, pu.values)


def test_permute_maps_x_multivariate_falls_back_to_normal_path(fastpath_x_frames):
    # "guard only" scope decision: X + multivariate method must never use the
    # row-batched fast path (cannot be batched -- all X rows fit jointly per
    # permutation), regardless of maps_batch_size. This locks that decision in.
    x_df, y_df = fastpath_x_frames
    nsp = _make_nsp(x_df, y_df, method="mlr")
    nsp.permute("maps", maps_which="X", maps_method="random", n_perm=20, seed=1,
               maps_batch_size=5, verbose=False, force_dict=True)
    assert nsp._nulls.get("maps_null") is not None  # normal path cached the full cube
    assert nsp._nulls.get("maps_null_fastpath_info") is None


@pytest.mark.parametrize("method,extra_kwargs,expected_reason_snippet", [
    ("mlr", {}, "can't be batched"),
    ("pearson", {"maps_batch_size": False}, "Batching disabled"),
])
def test_permute_maps_x_preflight_warns_before_normal_generation(
        monkeypatch, fastpath_x_frames, caplog, method, extra_kwargs, expected_reason_snippet):
    # pre-flight size warning: fires BEFORE the normal (non-batched) generation call,
    # covering every reason the fast path might be skipped (not just X+multivariate) --
    # a method that can't be row-batched at all, or batching explicitly disabled via
    # maps_batch_size=False. Lower the threshold so the small test fixture crosses it.
    import nispace.api as api_module
    monkeypatch.setattr(api_module, "_NULLMAPS_WARN_BYTES", 1)

    x_df, y_df = fastpath_x_frames
    nsp = _make_nsp(x_df, y_df, method=method)
    with caplog.at_level(logging.WARNING):
        nsp.permute("maps", maps_which="X", maps_method="random", n_perm=20, seed=1,
                   verbose=False, force_dict=True, **extra_kwargs)
    assert "Generating null maps for 'X' without map-batching" in caplog.text
    assert expected_reason_snippet in caplog.text
    assert nsp._nulls.get("maps_null") is not None  # normal path still ran to completion


def test_permute_maps_preflight_warning_silent_on_cache_hit(monkeypatch, fastpath_x_frames, caplog):
    # a genuine cache hit means no new allocation happens -- must not warn
    import nispace.api as api_module
    monkeypatch.setattr(api_module, "_NULLMAPS_WARN_BYTES", 1)

    x_df, y_df = fastpath_x_frames
    nsp = _make_nsp(x_df, y_df, method="mlr")
    nsp.permute("maps", maps_which="X", maps_method="random", n_perm=20, seed=1, verbose=False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        # same call again -- second run should hit the maps_null cache
        nsp.permute("maps", maps_which="X", maps_method="random", n_perm=20, seed=1, verbose=False)
    assert "Generating null maps for 'X' without map-batching" not in caplog.text


def test_permute_maps_double_batching_falls_back_with_warning(rng, caplog):
    n_parcels, n_x, n_y = 15, 22, 22  # both axes qualify for batching
    X = rng.normal(size=(n_x, n_parcels))
    Y = rng.normal(size=(n_y, n_parcels))
    x_df = pd.DataFrame(X, index=[f"x{i}" for i in range(n_x)],
                        columns=[f"p{i}" for i in range(n_parcels)])
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(n_y)],
                        columns=[f"p{i}" for i in range(n_parcels)])
    nsp = _make_nsp(x_df, y_df, method="pearson")
    p = nsp.permute("maps", maps_which=["X", "Y"], maps_method="random", n_perm=15, seed=2,
                    maps_batch_size=5, verbose=False)
    assert p.shape == (1, n_x)
    assert nsp._nulls.get("maps_null") is not None
    assert nsp._nulls.get("maps_null_fastpath_info") is None


def test_local_colocalization_raises_specific_message_after_fastpath(fastpath_x_frames):
    from nispace import diagnostics

    x_df, y_df = fastpath_x_frames
    nsp = _make_nsp(x_df, y_df, method="pearson")
    nsp.permute("maps", maps_which="X", maps_method="random", n_perm=20, seed=1,
               maps_batch_size=5, verbose=False)
    n_parcels = x_df.shape[1]
    coords = np.arange(n_parcels, dtype=float)
    dist_mat = np.abs(coords[:, None] - coords[None, :])
    with pytest.raises(KeyError, match="map-batched fast path"):
        diagnostics.local_colocalization(nsp, k=5, dist_mat=dist_mat, verbose=False)


# ── permute() imbalanced-sides efficiency nudge ─────────────────────────────

def _imbalanced_frames(rng, n_large, n_small, n_parcels=10):
    large = pd.DataFrame(rng.normal(size=(n_large, n_parcels)),
                         index=[f"l{i}" for i in range(n_large)])
    small = pd.DataFrame(rng.normal(size=(n_small, n_parcels)),
                         index=[f"s{i}" for i in range(n_small)])
    return large, small


def test_permute_maps_warns_when_permuting_much_larger_x_side(rng, caplog):
    # regression: user pointed out that permuting the much-larger side (e.g. 12000 X
    # genes) while the other side (Y) has only a handful of maps is far more
    # expensive than permuting Y instead -- null-map cost scales with the PERMUTED
    # side's row count, not the other side's, and a spatial null only needs to
    # preserve the permuted side's own autocorrelation structure, so either side is
    # a legitimate choice for a pairwise spatial correlation test. permute() should
    # surface this as a discoverable option, not silently pay the larger cost.
    x_df, y_df = _imbalanced_frames(rng, n_large=200, n_small=2)
    nsp = _make_nsp(x_df, y_df, method="pearson")
    with caplog.at_level(logging.WARNING):
        nsp.permute("maps", maps_which="X", maps_method="random", n_perm=10, seed=0,
                   verbose=False)
    assert "Permuting 'X' (200 maps) while 'Y' has far fewer (2 maps)" in caplog.text
    assert "maps_which='Y'" in caplog.text


def test_permute_maps_warns_when_permuting_much_larger_y_side(rng, caplog):
    x_df, y_df = _imbalanced_frames(rng, n_large=2, n_small=200)
    # swap: Y is now the large side
    nsp = _make_nsp(x_df, y_df, method="pearson")
    with caplog.at_level(logging.WARNING):
        nsp.permute("maps", maps_which="Y", maps_method="random", n_perm=10, seed=0,
                   verbose=False)
    assert "Permuting 'Y' (200 maps) while 'X' has far fewer (2 maps)" in caplog.text
    assert "maps_which='X'" in caplog.text


def test_permute_maps_imbalance_warning_silent_below_ratio_threshold(rng, caplog):
    # 200 vs 50 is only 4x -- below the 5x ratio threshold, must stay silent
    x_df, y_df = _imbalanced_frames(rng, n_large=200, n_small=50)
    nsp = _make_nsp(x_df, y_df, method="pearson")
    with caplog.at_level(logging.WARNING):
        nsp.permute("maps", maps_which="X", maps_method="random", n_perm=10, seed=0,
                   verbose=False)
    assert "Permuting 'X'" not in caplog.text


def test_permute_maps_imbalance_warning_silent_below_min_rows(rng, caplog):
    # ratio is extreme (50x) but the permuted side is small in absolute terms --
    # must stay silent (not worth the memory/compute cost either way)
    x_df, y_df = _imbalanced_frames(rng, n_large=50, n_small=1)
    nsp = _make_nsp(x_df, y_df, method="pearson")
    with caplog.at_level(logging.WARNING):
        nsp.permute("maps", maps_which="X", maps_method="random", n_perm=10, seed=0,
                   verbose=False)
    assert "Permuting 'X'" not in caplog.text


def test_permute_maps_imbalance_warning_silent_when_both_sides_permuted(rng, caplog):
    # maps_which=["X","Y"] has no "other side" to switch to -- must stay silent
    # even though the imbalance is extreme
    x_df, y_df = _imbalanced_frames(rng, n_large=200, n_small=2)
    nsp = _make_nsp(x_df, y_df, method="pearson")
    with caplog.at_level(logging.WARNING):
        nsp.permute("maps", maps_which=["X", "Y"], maps_method="random", n_perm=10,
                   seed=0, verbose=False)
    assert "Permuting 'X'" not in caplog.text
    assert "Permuting 'Y'" not in caplog.text
