"""Tests for stats/autocorr.py (Moran's I, variogram-based SA, Mantel test).

No established ground-truth library is added here: the standard PySAL
Moran's I reference (esda/libpysal) currently requires Python>=3.12 and
scikit-bio (Mantel) requires Python>=3.10, both of which would break this
project's 3.9-3.12 CI matrix -- a strictly worse version-compatibility
problem than pingouin's (see [[project_ground_truth_testing]]), and not
worth it for a module noted elsewhere as lightly used (see
[[project_autocorr]]). Instead:
- `mantel()` IS checked against a true established-library reference:
  scipy.stats.pearsonr/spearmanr computed directly on the squareform-
  flattened distance/difference vectors -- that comparison *is* what a
  Mantel test computes, not a hand-rolled stand-in.
- `morans_i()` is checked against an independently-styled (double-loop,
  not vectorized) hand-derived reference implementation of the textbook
  global Moran's I formula, plus internal consistency against
  `morans_i_fast()`.
- `variogram_sa()` has no standard canonical library implementation to
  compare against (it's a NiSpace-specific SA scalar derived from the
  empirical semivariogram, matching brainsmash's binning convention only
  in spirit) -- checked for well-formedness and correct direction (smooth
  spatial signal -> higher SA than pure noise).
"""

import numpy as np
import pytest
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr

from nispace.stats.autocorr import morans_i, morans_i_fast, variogram_sa, mantel, mantel_fast


def _hand_morans_i(data, distmat):
    """Independent, unvectorized (double-loop) reference for global Moran's I."""
    with np.errstate(divide="ignore"):
        w = 1 / distmat
    np.fill_diagonal(w, 0)
    n = len(data)
    z = data - data.mean()
    num, W = 0.0, 0.0
    for i in range(n):
        for j in range(n):
            num += w[i, j] * z[i] * z[j]
            W += w[i, j]
    den = (z ** 2).sum()
    return (n / W) * (num / den)


@pytest.fixture
def spatial_data(rng):
    n = 20
    coords = rng.normal(size=(n, 2))
    distmat = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    data = rng.normal(size=n)
    return data, distmat


# ── morans_i ──────────────────────────────────────────────────────────────

def test_morans_i_matches_hand_derived_formula(spatial_data):
    data, distmat = spatial_data
    expected = _hand_morans_i(data.copy(), distmat.copy())
    result = morans_i(data.copy(), distmat.copy())
    assert np.isclose(result, expected)


def test_morans_i_matches_morans_i_fast_given_precomputed_weights(spatial_data):
    data, distmat = spatial_data
    result = morans_i(data.copy(), distmat.copy())

    with np.errstate(divide="ignore"):
        w = 1 / distmat.copy()
    np.fill_diagonal(w, 0)
    result_fast = morans_i_fast(w, data.copy())
    assert np.isclose(result, result_fast)


def test_morans_i_nan_drop_matches_manual_deletion(spatial_data):
    data, distmat = spatial_data
    data_nan = data.copy()
    data_nan[3] = np.nan

    result = morans_i(data_nan, distmat.copy(), nan_policy="drop")
    data_manual = np.delete(data, 3)
    distmat_manual = np.delete(np.delete(distmat, 3, axis=0), 3, axis=1)
    expected = morans_i(data_manual, distmat_manual, nan_policy="propagate")
    assert np.isclose(result, expected)


def test_morans_i_nan_raise_raises(spatial_data):
    data, distmat = spatial_data
    data_nan = data.copy()
    data_nan[0] = np.nan
    with pytest.raises(ValueError):
        morans_i(data_nan, distmat.copy(), nan_policy="raise")


def test_morans_i_invalid_nan_policy_raises(spatial_data):
    data, distmat = spatial_data
    with pytest.raises(ValueError):
        morans_i(data.copy(), distmat.copy(), nan_policy="not_a_real_policy")


def test_morans_i_smooth_signal_has_higher_autocorrelation_than_noise(rng):
    n = 60
    pos = np.arange(n).astype(float)
    distmat = np.abs(pos[:, None] - pos[None, :])
    smooth_data = np.sin(pos / 5.0)
    random_data = rng.normal(size=n)
    assert morans_i(smooth_data, distmat.copy()) > morans_i(random_data, distmat.copy())


# ── variogram_sa ──────────────────────────────────────────────────────────

def test_variogram_sa_smooth_signal_has_higher_sa_than_noise(rng):
    n = 60
    pos = np.arange(n).astype(float)
    distmat = np.abs(pos[:, None] - pos[None, :])
    smooth_data = np.sin(pos / 5.0)
    random_data = rng.normal(size=n)
    assert variogram_sa(smooth_data, distmat) > variogram_sa(random_data, distmat)


def test_variogram_sa_return_variogram_shapes(rng):
    n = 40
    coords = rng.normal(size=(n, 2))
    distmat = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    data = rng.normal(size=n)
    sa, centers, gamma_norm = variogram_sa(data, distmat, n_bins=10, return_variogram=True)
    assert isinstance(sa, float)
    assert centers.shape == gamma_norm.shape
    assert len(centers) <= 10


def test_variogram_sa_drops_nan(rng):
    n = 30
    coords = rng.normal(size=(n, 2))
    distmat = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    data = rng.normal(size=n)
    data_nan = data.copy()
    data_nan[0] = np.nan
    # must not raise / propagate NaN into the result
    sa = variogram_sa(data_nan, distmat)
    assert np.isfinite(sa)


# ── mantel ────────────────────────────────────────────────────────────────

def test_mantel_matches_scipy_pearsonr_on_flattened_vectors(spatial_data):
    data, distmat = spatial_data
    diffmat = np.abs(data[:, None] - data[None, :])
    diffmat_flat = squareform(diffmat, checks=False)
    distmat_flat = squareform(distmat)
    expected, _ = pearsonr(diffmat_flat, distmat_flat)

    result = mantel(data.copy(), distmat.copy())
    assert np.isclose(result, expected)


def test_mantel_spearman_matches_scipy_spearmanr(spatial_data):
    data, distmat = spatial_data
    diffmat = np.abs(data[:, None] - data[None, :])
    diffmat_flat = squareform(diffmat, checks=False)
    distmat_flat = squareform(distmat)
    expected, _ = spearmanr(diffmat_flat, distmat_flat)

    result = mantel(data.copy(), distmat.copy(), spearman=True)
    assert np.isclose(result, expected)


def test_mantel_matches_mantel_fast(spatial_data):
    data, distmat = spatial_data
    result = mantel(data.copy(), distmat.copy())

    distmat_flat = squareform(distmat.copy())
    result_fast = mantel_fast(data.astype(np.float64), distmat_flat.astype(np.float64))
    assert np.isclose(result, result_fast)


def test_mantel_raises_on_asymmetric_distmat(spatial_data):
    data, distmat = spatial_data
    asym = distmat.copy()
    asym[0, 1] += 1.0
    with pytest.raises(ValueError):
        mantel(data.copy(), asym)


def test_mantel_raises_on_shape_mismatch(spatial_data):
    data, distmat = spatial_data
    with pytest.raises(ValueError):
        mantel(data[:-1].copy(), distmat.copy())
