"""Tests for null map generation functions:
- nulls_random (moved from test_parcellation_null.py)
- nulls_moran
- nulls_variomoran
- compute_mem / moran_randomization (from _brainspace_moran.py)

All tests use synthetic data only — no data-repo access, no external BrainSpace
dependency (MoranRandomization is a bundled pure-numpy/scipy copy).
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from nispace.nulls import (
    nulls_random, nulls_moran, nulls_variomoran,
    nulls_burt2018, nulls_burt2020,
    _BRAINSMASH_AVAILABLE,
    generate_null_maps,
    _moran_fit_1_over_d,
)
import nispace.nulls as nulls_module
from nispace._brainspace_moran import compute_mem, moran_randomization

requires_brainsmash = pytest.mark.skipif(
    not _BRAINSMASH_AVAILABLE, reason="brainsmash not installed"
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def linear_dist_mat():
    """20-point 1D Euclidean distance matrix — small, well-conditioned."""
    pts = np.linspace(0, 1, 20)[:, None]
    return cdist(pts, pts).astype(np.float64)


@pytest.fixture
def smooth_map(linear_dist_mat):
    """Spatially smooth 1D map: sin wave over 20 points, low SA."""
    return np.sin(np.linspace(0, 2 * np.pi, 20)).astype(np.float32)


@pytest.fixture
def white_noise_map(rng):
    """Spatially uncorrelated 20-point map (near-zero Moran's I)."""
    return rng.standard_normal(20).astype(np.float32)


# ---------------------------------------------------------------------------
# nulls_random
# ---------------------------------------------------------------------------

def test_nulls_random_output_shape(rng):
    x = rng.standard_normal(50).astype(np.float32)
    nulls = nulls_random(x, n_nulls=30, seed=0)
    assert nulls.shape == (30, 50)


def test_nulls_random_each_row_is_permutation(rng):
    x = rng.standard_normal(40).astype(np.float32)
    nulls = nulls_random(x, n_nulls=20, seed=1)
    x_sorted = np.sort(x)
    for row in nulls:
        np.testing.assert_array_equal(np.sort(row), x_sorted)


def test_nulls_random_nan_positions_preserved():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan], dtype=np.float32)
    nulls = nulls_random(x, n_nulls=10, seed=0)
    assert nulls.shape == (10, 6)
    assert np.all(np.isnan(nulls[:, 2]))
    assert np.all(np.isnan(nulls[:, 5]))
    assert np.all(np.isfinite(nulls[:, [0, 1, 3, 4]]))


def test_nulls_random_reproducible_with_same_seed(rng):
    x = rng.standard_normal(30).astype(np.float32)
    a = nulls_random(x, n_nulls=10, seed=99)
    b = nulls_random(x, n_nulls=10, seed=99)
    np.testing.assert_array_equal(a, b)


def test_nulls_random_different_seeds_differ(rng):
    x = rng.standard_normal(30).astype(np.float32)
    a = nulls_random(x, n_nulls=10, seed=1)
    b = nulls_random(x, n_nulls=10, seed=2)
    assert not np.array_equal(a, b)


def test_nulls_random_ignores_dist_mat(rng):
    x = rng.standard_normal(20).astype(np.float32)
    dm = rng.random((20, 20)).astype(np.float32)
    a = nulls_random(x, dist_mat=None, n_nulls=5, seed=7)
    b = nulls_random(x, dist_mat=dm,   n_nulls=5, seed=7)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# compute_mem
# ---------------------------------------------------------------------------

def test_compute_mem_output_shapes(linear_dist_mat):
    n = linear_dist_mat.shape[0]
    dm = np.where(linear_dist_mat == 0, np.inf, linear_dist_mat)
    W = 1.0 / dm
    np.fill_diagonal(W, 0.0)
    mem, ev = compute_mem(W)
    assert mem.ndim == 2
    assert mem.shape[0] == n
    assert ev.ndim == 1
    assert mem.shape[1] == ev.shape[0]


def test_compute_mem_eigenvalues_descending(linear_dist_mat):
    dm = np.where(linear_dist_mat == 0, np.inf, linear_dist_mat)
    W = 1.0 / dm
    np.fill_diagonal(W, 0.0)
    _, ev = compute_mem(W)
    assert np.all(np.diff(ev) <= 0), "Eigenvalues must be in descending order"


def test_compute_mem_truncated_matches_full_top_k(linear_dist_mat):
    dm = np.where(linear_dist_mat == 0, np.inf, linear_dist_mat)
    W = 1.0 / dm
    np.fill_diagonal(W, 0.0)
    mem_full, ev_full = compute_mem(W)
    k = 5
    mem_trunc, ev_trunc = compute_mem(W, n_components=k)
    assert mem_trunc.shape[1] <= k
    # top-k eigenvalues should match (up to sign flip on eigenvectors)
    np.testing.assert_allclose(
        np.abs(ev_trunc), np.abs(ev_full[:k]), rtol=1e-3
    )


def test_compute_mem_sparse_requires_n_components(linear_dist_mat):
    import scipy.sparse as ssp
    dm = np.where(linear_dist_mat == 0, np.inf, linear_dist_mat)
    W = 1.0 / dm
    np.fill_diagonal(W, 0.0)
    W_sparse = ssp.csr_matrix(W)
    with pytest.raises(ValueError, match="n_components"):
        compute_mem(W_sparse)


def test_compute_mem_sparse_with_n_components(linear_dist_mat):
    import scipy.sparse as ssp
    dm = np.where(linear_dist_mat == 0, np.inf, linear_dist_mat)
    W = 1.0 / dm
    np.fill_diagonal(W, 0.0)
    W_sparse = ssp.csr_matrix(W)
    mem, ev = compute_mem(W_sparse, n_components=5)
    assert mem.shape[0] == linear_dist_mat.shape[0]
    assert mem.shape[1] == ev.shape[0]
    assert mem.shape[1] <= 5


# ---------------------------------------------------------------------------
# moran_randomization
# ---------------------------------------------------------------------------

def _make_W_mem(dist_mat):
    dm = np.where(dist_mat == 0, np.inf, dist_mat)
    W = 1.0 / dm
    np.fill_diagonal(W, 0.0)
    mem, mev = compute_mem(W, n_components=10)
    return mem, mev


def test_moran_randomization_output_shape(linear_dist_mat, smooth_map):
    mem, mev = _make_W_mem(linear_dist_mat)
    out = moran_randomization(smooth_map, mem, mev, n_nulls=25, seed=0)
    assert out.shape == (25, len(smooth_map))


def test_moran_randomization_reproducible(linear_dist_mat, smooth_map):
    mem, mev = _make_W_mem(linear_dist_mat)
    a = moran_randomization(smooth_map, mem, mev, n_nulls=10, seed=5)
    b = moran_randomization(smooth_map, mem, mev, n_nulls=10, seed=5)
    np.testing.assert_array_equal(a, b)


def test_moran_randomization_different_seeds_differ(linear_dist_mat, smooth_map):
    mem, mev = _make_W_mem(linear_dist_mat)
    a = moran_randomization(smooth_map, mem, mev, n_nulls=10, seed=1)
    b = moran_randomization(smooth_map, mem, mev, n_nulls=10, seed=2)
    assert not np.array_equal(a, b)


def test_moran_randomization_pair_procedure_shape(linear_dist_mat, smooth_map):
    mem, mev = _make_W_mem(linear_dist_mat)
    out = moran_randomization(smooth_map, mem, mev, n_nulls=10,
                               procedure="pair", seed=0)
    assert out.shape == (10, len(smooth_map))


def test_moran_randomization_invalid_procedure(linear_dist_mat, smooth_map):
    mem, mev = _make_W_mem(linear_dist_mat)
    with pytest.raises(ValueError, match="procedure"):
        moran_randomization(smooth_map, mem, mev, procedure="unknown")


def test_moran_randomization_singleton_mean_approximately_preserved(
        linear_dist_mat, smooth_map):
    # Singleton flips only signs of MEM coefficients — mean and variance of
    # the distribution of null means should be close to the original mean.
    mem, mev = _make_W_mem(linear_dist_mat)
    out = moran_randomization(smooth_map, mem, mev, n_nulls=200, seed=0)
    null_means = out.mean(axis=1)
    np.testing.assert_allclose(null_means.mean(), smooth_map.mean(), atol=0.3)


def test_moran_randomization_joint_output_shape(linear_dist_mat):
    rng = np.random.default_rng(0)
    x2d = rng.standard_normal((20, 3)).astype(np.float32)
    mem, mev = _make_W_mem(linear_dist_mat)
    out = moran_randomization(x2d, mem, mev, n_nulls=10, joint=True, seed=0)
    assert out.shape == (10, 20, 3)


# ---------------------------------------------------------------------------
# nulls_moran
# ---------------------------------------------------------------------------

def test_nulls_moran_output_shape(linear_dist_mat, smooth_map):
    nulls = nulls_moran(smooth_map, linear_dist_mat, n_nulls=20, seed=0)
    assert nulls.shape == (20, len(smooth_map))


def test_nulls_moran_values_finite(linear_dist_mat, smooth_map):
    nulls = nulls_moran(smooth_map, linear_dist_mat, n_nulls=10, seed=0)
    assert np.all(np.isfinite(nulls))


def test_nulls_moran_nan_positions_preserved(linear_dist_mat):
    x = np.sin(np.linspace(0, 2 * np.pi, 20)).astype(np.float32)
    x[3] = np.nan
    x[15] = np.nan
    nulls = nulls_moran(x, linear_dist_mat, n_nulls=10, seed=0)
    assert nulls.shape == (10, 20)
    assert np.all(np.isnan(nulls[:, 3]))
    assert np.all(np.isnan(nulls[:, 15]))
    assert np.all(np.isfinite(nulls[:, [0, 1, 2, 4]]))


def test_nulls_moran_reproducible(linear_dist_mat, smooth_map):
    a = nulls_moran(smooth_map, linear_dist_mat, n_nulls=10, seed=7)
    b = nulls_moran(smooth_map, linear_dist_mat, n_nulls=10, seed=7)
    np.testing.assert_array_equal(a, b)


def test_nulls_moran_different_seeds_differ(linear_dist_mat, smooth_map):
    a = nulls_moran(smooth_map, linear_dist_mat, n_nulls=10, seed=1)
    b = nulls_moran(smooth_map, linear_dist_mat, n_nulls=10, seed=2)
    assert not np.array_equal(a, b)


def test_nulls_moran_nulls_differ_from_input(linear_dist_mat, smooth_map):
    nulls = nulls_moran(smooth_map, linear_dist_mat, n_nulls=20, seed=0)
    assert not np.all(nulls == smooth_map)


# ---------------------------------------------------------------------------
# nulls_variomoran
# ---------------------------------------------------------------------------

def test_nulls_variomoran_output_shape(linear_dist_mat, smooth_map):
    nulls = nulls_variomoran(smooth_map, linear_dist_mat, n_nulls=20, seed=0)
    assert nulls.shape == (20, len(smooth_map))


def test_nulls_variomoran_values_finite(linear_dist_mat, smooth_map):
    nulls = nulls_variomoran(smooth_map, linear_dist_mat, n_nulls=10, seed=0)
    assert np.all(np.isfinite(nulls))


def test_nulls_variomoran_reproducible(linear_dist_mat, smooth_map):
    a = nulls_variomoran(smooth_map, linear_dist_mat, n_nulls=10, seed=3)
    b = nulls_variomoran(smooth_map, linear_dist_mat, n_nulls=10, seed=3)
    np.testing.assert_array_equal(a, b)


def test_nulls_variomoran_fallback_for_white_noise(linear_dist_mat, white_noise_map):
    # White noise has near-zero Moran's I → should fall back to standard 1/d W.
    # Smoke test: must complete without error and return correct shape.
    nulls = nulls_variomoran(
        white_noise_map, linear_dist_mat,
        n_nulls=10, seed=0,
        variogram_threshold=0.5,  # force fallback for near-zero SA
    )
    assert nulls.shape == (10, len(white_noise_map))
    assert np.all(np.isfinite(nulls))


def test_nulls_variomoran_and_moran_same_shape(linear_dist_mat, smooth_map):
    a = nulls_moran(smooth_map, linear_dist_mat, n_nulls=15, seed=0)
    b = nulls_variomoran(smooth_map, linear_dist_mat, n_nulls=15, seed=0)
    assert a.shape == b.shape


# ---------------------------------------------------------------------------
# nulls_burt2018 (neuromaps.nulls.batch_surrogates — always available)
# ---------------------------------------------------------------------------

def test_nulls_burt2018_output_shape(linear_dist_mat, smooth_map):
    nulls = nulls_burt2018(smooth_map, linear_dist_mat, n_nulls=10, seed=0)
    assert nulls.shape == (10, len(smooth_map))


def test_nulls_burt2018_values_finite(linear_dist_mat, smooth_map):
    nulls = nulls_burt2018(smooth_map, linear_dist_mat, n_nulls=10, seed=0)
    assert np.all(np.isfinite(nulls))


def test_nulls_burt2018_nan_positions_preserved(linear_dist_mat):
    x = np.sin(np.linspace(0, 2 * np.pi, 20)).astype(np.float32)
    x[5] = np.nan
    nulls = nulls_burt2018(x, linear_dist_mat, n_nulls=5, seed=0)
    assert nulls.shape == (5, 20)
    assert np.all(np.isnan(nulls[:, 5]))
    assert np.all(np.isfinite(nulls[:, [0, 1, 2, 3, 4]]))


def test_nulls_burt2018_reproducible(linear_dist_mat, smooth_map):
    a = nulls_burt2018(smooth_map, linear_dist_mat, n_nulls=5, seed=11)
    b = nulls_burt2018(smooth_map, linear_dist_mat, n_nulls=5, seed=11)
    np.testing.assert_array_equal(a, b)


def test_nulls_burt2018_different_seeds_differ(linear_dist_mat, smooth_map):
    a = nulls_burt2018(smooth_map, linear_dist_mat, n_nulls=5, seed=1)
    b = nulls_burt2018(smooth_map, linear_dist_mat, n_nulls=5, seed=2)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# nulls_burt2020 (brainsmash — optional, skip if absent)
# ---------------------------------------------------------------------------

@requires_brainsmash
def test_nulls_burt2020_output_shape(linear_dist_mat, smooth_map):
    nulls = nulls_burt2020(smooth_map, linear_dist_mat, n_nulls=10, seed=0)
    assert nulls.shape == (10, len(smooth_map))


@requires_brainsmash
def test_nulls_burt2020_values_finite(linear_dist_mat, smooth_map):
    nulls = nulls_burt2020(smooth_map, linear_dist_mat, n_nulls=10, seed=0)
    assert np.all(np.isfinite(nulls))


@requires_brainsmash
def test_nulls_burt2020_nan_positions_preserved(linear_dist_mat):
    x = np.sin(np.linspace(0, 2 * np.pi, 20)).astype(np.float32)
    x[7] = np.nan
    nulls = nulls_burt2020(x, linear_dist_mat, n_nulls=5, seed=0)
    assert nulls.shape == (5, 20)
    assert np.all(np.isnan(nulls[:, 7]))
    assert np.all(np.isfinite(nulls[:, [0, 1, 2, 3, 4]]))


@requires_brainsmash
def test_nulls_burt2020_reproducible(linear_dist_mat, smooth_map):
    a = nulls_burt2020(smooth_map, linear_dist_mat, n_nulls=5, seed=13)
    b = nulls_burt2020(smooth_map, linear_dist_mat, n_nulls=5, seed=13)
    np.testing.assert_array_equal(a, b)


@requires_brainsmash
def test_nulls_burt2020_different_seeds_differ(linear_dist_mat, smooth_map):
    a = nulls_burt2020(smooth_map, linear_dist_mat, n_nulls=5, seed=1)
    b = nulls_burt2020(smooth_map, linear_dist_mat, n_nulls=5, seed=2)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# generate_null_maps: requirement-check regression tests
# ---------------------------------------------------------------------------

def test_generate_null_maps_random_no_parc_no_distmat(rng):
    # Bug fix: method="random" needs no parcellation and no dist_mat
    data = rng.standard_normal(20).astype(np.float32)
    nulls, result_mat = generate_null_maps(
        method="random",
        data=data,
        parcellation=None,
        dist_mat=None,
        n_nulls=10,
        seed=0,
        verbose=False,
    )
    assert nulls.shape == (1, 10, 20)
    assert result_mat is None


def test_generate_null_maps_random_with_tuple_distmat_does_not_crash(rng):
    # Bug fix: method="random" + tuple dist_mat should not crash (dist_mat is ignored)
    pts = np.linspace(0, 1, 10)[:, None]
    dm = cdist(pts, pts).astype(np.float64)
    data = rng.standard_normal(20).astype(np.float32)
    nulls, _ = generate_null_maps(
        method="random",
        data=data,
        parcellation=None,
        dist_mat=(dm, dm),  # tuple — was crashing before the fix
        n_nulls=5,
        seed=0,
        verbose=False,
    )
    assert nulls.shape == (1, 5, 20)


def test_generate_null_maps_none_parc_none_distmat_raises_clearly(rng):
    # Bug fix: parcellation=None + dist_mat=None for distance method must raise ValueError
    # (not an opaque TypeError "data type not defined")
    data = rng.standard_normal(20).astype(np.float32)
    with pytest.raises(ValueError, match="parcellation.*dist_mat.*None"):
        generate_null_maps(
            method="moran",
            data=data,
            parcellation=None,
            dist_mat=None,
            n_nulls=5,
            verbose=False,
        )


# ---------------------------------------------------------------------------
# moran fit-once: sharing the geometry-only fit(W) across many rows
# ---------------------------------------------------------------------------

@pytest.fixture
def many_rows_dist_mat(rng):
    """40-point 2D Euclidean distance matrix, large enough to exercise multi-row sharing."""
    pts = rng.uniform(0, 100, size=(40, 2))
    return cdist(pts, pts).astype(np.float64)


@pytest.fixture
def many_rows_data_homogeneous(rng, many_rows_dist_mat):
    """30 rows sharing the exact same (no-NaN) mask."""
    n_parcels = many_rows_dist_mat.shape[0]
    return rng.standard_normal((30, n_parcels)).astype(np.float32)


@pytest.fixture
def many_rows_data_heterogeneous(many_rows_data_homogeneous):
    """Same as the homogeneous fixture, but with 3 distinct NaN-mask groups injected."""
    data = many_rows_data_homogeneous.copy()
    # group A: rows 3 and 10 share one NaN pattern
    data[3, [1, 2]] = np.nan
    data[10, [1, 2]] = np.nan
    # group B: row 15 has a different NaN pattern
    data[15, 7] = np.nan
    # remaining 27 rows: the original (no-NaN) mask
    return data


def _spy_call_count(monkeypatch, target_module, name):
    """Wrap `target_module.name` to count calls, returning a mutable {"n": int} counter."""
    orig = getattr(target_module, name)
    counter = {"n": 0}

    def wrapped(*args, **kwargs):
        counter["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(target_module, name, wrapped)
    return counter


def test_moran_fit_1_over_d_matches_nulls_moran_internal_construction(
        many_rows_dist_mat, many_rows_data_homogeneous):
    # _moran_fit_1_over_d must reproduce exactly nulls_moran's own (pre-refactor) inline
    # 1/d weight-matrix construction + compute_mem call.
    row = many_rows_data_homogeneous[0]
    dm = many_rows_dist_mat.copy()
    mem_helper, mev_helper = _moran_fit_1_over_d(dm.copy())

    dm2 = many_rows_dist_mat.copy()
    np.fill_diagonal(dm2, 1)
    dm2 **= -1
    mem_ref, mev_ref = compute_mem(dm2, spectrum="nonzero", tol=1e-6, n_components=15)

    np.testing.assert_array_equal(mem_helper, mem_ref)
    np.testing.assert_array_equal(mev_helper, mev_ref)


def test_nulls_moran_precomputed_mem_matches_internal_fit(
        many_rows_dist_mat, many_rows_data_homogeneous):
    # Passing a precomputed (mem, mev) must give bit-identical output to nulls_moran fitting
    # it internally, for the same row/seed.
    row = many_rows_data_homogeneous[0]
    # no NaNs in `row` and no all-inf rows in `many_rows_dist_mat` -> mask is all-True,
    # so fitting on the full (unmasked) matrix matches what nulls_moran fits internally
    mem, mev = _moran_fit_1_over_d(many_rows_dist_mat.copy())

    a = nulls_moran(row, many_rows_dist_mat.copy(), n_nulls=30, seed=42)
    b = nulls_moran(row, many_rows_dist_mat.copy(), n_nulls=30, seed=42,
                     _precomputed_mem=(mem, mev))
    np.testing.assert_array_equal(a, b)


def test_generate_null_maps_moran_fits_once_across_homogeneous_rows(
        monkeypatch, many_rows_dist_mat, many_rows_data_homogeneous):
    counter = _spy_call_count(monkeypatch, nulls_module, "compute_mem")
    nulls, _ = generate_null_maps(
        method="moran", data=many_rows_data_homogeneous, parcellation=None,
        dist_mat=many_rows_dist_mat.copy(), parc_space="mni152",
        n_nulls=20, seed=0, n_proc=1, verbose=False,
    )
    n_rows = many_rows_data_homogeneous.shape[0]
    assert nulls.shape == (n_rows, 20, many_rows_dist_mat.shape[0])
    # all rows share the same (no-NaN) mask -> exactly one fit, not one per row
    assert counter["n"] == 1


def test_generate_null_maps_moran_matches_old_per_row_behavior(
        many_rows_dist_mat, many_rows_data_homogeneous):
    # "old vs. new" equivalence: the fit-once path inside generate_null_maps must produce,
    # for every row, exactly what a direct per-row nulls_moran(..., seed=seed_base+i) call
    # (today's unshared path, still reachable and unmodified) produces.
    seed_base = 777
    n_nulls = 25
    nulls, _ = generate_null_maps(
        method="moran", data=many_rows_data_homogeneous, parcellation=None,
        dist_mat=many_rows_dist_mat.copy(), parc_space="mni152",
        n_nulls=n_nulls, seed=seed_base, n_proc=1, verbose=False,
    )
    for i in range(many_rows_data_homogeneous.shape[0]):
        ref = nulls_moran(many_rows_data_homogeneous[i], many_rows_dist_mat.copy(),
                           n_nulls=n_nulls, seed=seed_base + i)
        np.testing.assert_array_equal(nulls.data[i], ref)


def test_generate_null_maps_moran_heterogeneous_nan_masks_still_correct(
        monkeypatch, many_rows_dist_mat, many_rows_data_heterogeneous):
    counter = _spy_call_count(monkeypatch, nulls_module, "compute_mem")
    seed_base = 321
    n_nulls = 20
    nulls, _ = generate_null_maps(
        method="moran", data=many_rows_data_heterogeneous, parcellation=None,
        dist_mat=many_rows_dist_mat.copy(), parc_space="mni152",
        n_nulls=n_nulls, seed=seed_base, n_proc=1, verbose=False,
    )
    # 3 distinct NaN-mask groups (rows {3,10}, row {15}, the rest) -> exactly 3 fits, not
    # 1 (wrongly sharing across incompatible masks) and not 30 (no sharing at all)
    assert counter["n"] == 3
    for i in range(many_rows_data_heterogeneous.shape[0]):
        ref = nulls_moran(many_rows_data_heterogeneous[i], many_rows_dist_mat.copy(),
                           n_nulls=n_nulls, seed=seed_base + i)
        np.testing.assert_array_equal(nulls.data[i], ref, err_msg=f"row {i} mismatch")


def test_generate_null_maps_moran_fit_variogram_bypasses_sharing(
        monkeypatch, many_rows_dist_mat, many_rows_data_homogeneous):
    # fit_variogram=True (variomoran's mechanism): W depends on each row's own values and
    # must never be shared across rows -- the optimization must be a complete no-op here.
    counter = _spy_call_count(monkeypatch, nulls_module, "compute_mem")
    seed_base = 55
    n_nulls = 15
    nulls, _ = generate_null_maps(
        method="moran", data=many_rows_data_homogeneous, parcellation=None,
        dist_mat=many_rows_dist_mat.copy(), parc_space="mni152",
        n_nulls=n_nulls, seed=seed_base, n_proc=1, verbose=False,
        fit_variogram=True,
    )
    n_rows = many_rows_data_homogeneous.shape[0]
    assert counter["n"] == n_rows  # one fit per row, sharing disabled
    for i in range(n_rows):
        ref = nulls_moran(many_rows_data_homogeneous[i], many_rows_dist_mat.copy(),
                           n_nulls=n_nulls, seed=seed_base + i, fit_variogram=True)
        np.testing.assert_array_equal(nulls.data[i], ref)


def test_generate_null_maps_moran_n_proc_gt_1_matches_n_proc_1(
        many_rows_dist_mat, many_rows_data_heterogeneous):
    seed_base = 909
    n_nulls = 15
    n1, _ = generate_null_maps(
        method="moran", data=many_rows_data_heterogeneous, parcellation=None,
        dist_mat=many_rows_dist_mat.copy(), parc_space="mni152",
        n_nulls=n_nulls, seed=seed_base, n_proc=1, verbose=False,
    )
    n2, _ = generate_null_maps(
        method="moran", data=many_rows_data_heterogeneous, parcellation=None,
        dist_mat=many_rows_dist_mat.copy(), parc_space="mni152",
        n_nulls=n_nulls, seed=seed_base, n_proc=2, verbose=False,
    )
    np.testing.assert_array_equal(n1.data, n2.data)


@pytest.mark.parametrize("method", ["random", "moran"])
def test_generate_null_maps_seed_plus_i_multirow(
        method, many_rows_dist_mat, many_rows_data_homogeneous):
    # Documents/locks the seed+i invariant that row-batched generation (permute()'s
    # maps_batch_size) depends on: row i of a multi-row generate_null_maps(seed=S) call
    # must equal a standalone single-row call for that same row with seed=S+i.
    seed_base = 100
    n_nulls = 10
    data = many_rows_data_homogeneous[:5]
    nulls, _ = generate_null_maps(
        method=method, data=data, parcellation=None,
        dist_mat=many_rows_dist_mat.copy(), parc_space="mni152",
        n_nulls=n_nulls, seed=seed_base, n_proc=1, verbose=False,
    )
    for i in range(5):
        single, _ = generate_null_maps(
            method=method, data=data[i:i + 1], parcellation=None,
            dist_mat=many_rows_dist_mat.copy(), parc_space="mni152",
            n_nulls=n_nulls, seed=seed_base + i, n_proc=1, verbose=False,
        )
        np.testing.assert_array_equal(nulls.data[i], single.data[0])
