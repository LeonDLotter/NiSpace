"""Tests for nispace.stats.effectsize.

This module is public API and was fully docstring-audited previously (see
memory project_docstring_audit, batches 9+10) -- tests here also serve as an
accuracy check on those docstrings' specific claims (NaN propagate-vs-skip
per function, plain vs `_fast` equivalence, numba-jit status).

One real docstring gap found and fixed while writing these tests:
`centile_fast` (added later, not covered by the original audit batch) had
only a one-line docstring, and its `_fast` name incorrectly implied it's
numba-jitted -- it isn't (plain Python with a searchsorted loop). Docstring
now brought up to the file's own convention and corrected.

A later pass added ground-truth checks against established libraries
(scipy/pingouin) for cohen/hedges/zscore/centile_fast, rather than only
comparing against hand-written "expected" formulas in this file -- a shared
bug between the implementation and a hand-written test formula wouldn't be
caught by the earlier tests alone. Includes NaN handling where the reference
library supports it (scipy's `nan_policy="omit"`; pingouin has no NaN
handling, so `cohen_nan`/`hedges_nan` are checked against pingouin computed
on the same manually-dropna'd data).
"""

import numpy as np
import pytest

from nispace.stats.effectsize import (
    cohen, cohen_nan, cohen_nan_fast,
    cohen_paired, cohen_paired_nan, cohen_paired_nan_fast,
    hedges, hedges_nan, hedges_nan_fast,
    zscore, zscore_nan, zscore_nan_fast,
    rzscore_nan, rzscore_nan_fast,
    prc, prc_fast,
    logfc_nan, logfc_fast,
    centile_fast,
    _welford_1d,
)


# ── _welford_1d ──────────────────────────────────────────────────────────────

def test_welford_1d_matches_numpy():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    n, mean, var = _welford_1d(x)
    assert n == 5
    assert np.isclose(mean, x.mean())
    assert np.isclose(var, x.var(ddof=1))


def test_welford_1d_skips_nan():
    x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    n, mean, var = _welford_1d(x)
    assert n == 3
    assert np.isclose(mean, np.nanmean(x))
    assert np.isclose(var, np.nanvar(x, ddof=1))


def test_welford_1d_single_value_variance_is_nan():
    n, mean, var = _welford_1d(np.array([5.0]))
    assert n == 1
    assert np.isnan(var)


# ── cohen / cohen_nan / cohen_nan_fast ───────────────────────────────────────

@pytest.fixture
def two_groups(rng):
    a = rng.normal(loc=1.0, size=(10, 4))
    b = rng.normal(loc=0.0, size=(12, 4))
    return a, b


def test_cohen_matches_manual_formula(two_groups):
    a, b = two_groups
    na, nb = a.shape[0], b.shape[0]
    dof = na + nb - 2
    pooled_std = np.sqrt(((na - 1) * a.var(ddof=1, axis=0) + (nb - 1) * b.var(ddof=1, axis=0)) / dof)
    expected = (a.mean(axis=0) - b.mean(axis=0)) / pooled_std
    assert np.allclose(cohen(a, b), expected)


def test_cohen_propagates_nan_but_cohen_nan_skips():
    a = np.array([[1.0], [2.0], [np.nan]])
    b = np.array([[0.0], [1.0], [2.0]])
    assert np.isnan(cohen(a, b)[0])
    assert not np.isnan(cohen_nan(a, b)[0])


def test_cohen_nan_fast_matches_cohen_nan(two_groups):
    a, b = two_groups
    a_with_nan = a.copy()
    a_with_nan[0, 0] = np.nan
    assert np.allclose(
        cohen_nan_fast(a_with_nan, b), cohen_nan(a_with_nan, b), equal_nan=True
    )


def test_cohen_nan_fast_dof_zero_is_nan():
    a = np.array([[1.0]])  # n=1
    b = np.array([[2.0]])  # n=1 -> dof=0
    assert np.isnan(cohen_nan_fast(a, b)[0])


# ── cohen_paired / cohen_paired_nan / cohen_paired_nan_fast ─────────────────

def test_cohen_paired_matches_manual_formula(rng):
    a = rng.normal(size=(10, 3))
    b = rng.normal(size=(10, 3))
    diff = a - b
    expected = diff.mean(axis=0) / diff.std(ddof=1, axis=0)
    assert np.allclose(cohen_paired(a, b), expected)


def test_cohen_paired_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        cohen_paired(np.zeros((3, 2)), np.zeros((4, 2)))
    with pytest.raises(ValueError, match="same shape"):
        cohen_paired_nan(np.zeros((3, 2)), np.zeros((4, 2)))


def test_cohen_paired_nan_fast_matches_cohen_paired_nan(rng):
    a = rng.normal(size=(10, 3))
    b = rng.normal(size=(10, 3))
    a[0, 0] = np.nan
    assert np.allclose(
        cohen_paired_nan_fast(a, b), cohen_paired_nan(a, b), equal_nan=True
    )


# ── hedges / hedges_nan / hedges_nan_fast ────────────────────────────────────

def test_hedges_is_corrected_cohen(two_groups):
    a, b = two_groups
    na, nb = a.shape[0], b.shape[0]
    dof = na + nb - 2
    correction = 1 - (3 / (4 * dof - 1))
    assert np.allclose(hedges(a, b), cohen(a, b) * correction)


def test_hedges_nan_fast_matches_hedges_nan(two_groups):
    a, b = two_groups
    a_with_nan = a.copy()
    a_with_nan[0, 0] = np.nan
    assert np.allclose(
        hedges_nan_fast(a_with_nan, b), hedges_nan(a_with_nan, b), equal_nan=True
    )


# ── zscore / zscore_nan / zscore_nan_fast ────────────────────────────────────

def test_zscore_self_standardizes(rng):
    a = rng.normal(size=(20, 3))
    z = zscore(a)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(z.std(axis=0, ddof=1), 1, atol=1e-10)


def test_zscore_against_reference_b(rng):
    a = rng.normal(size=(5, 2))
    b = rng.normal(size=(20, 2))
    expected = (a - b.mean(axis=0)) / b.std(ddof=1, axis=0)
    assert np.allclose(zscore(a, b), expected)


def test_zscore_propagates_nan_but_zscore_nan_skips():
    a = np.array([[1.0], [2.0], [3.0]])
    b = np.array([[1.0], [np.nan], [3.0]])
    assert np.all(np.isnan(zscore(a, b)))
    assert not np.any(np.isnan(zscore_nan(a, b)))


def test_zscore_nan_fast_matches_zscore_nan(rng):
    a = rng.normal(size=(10, 3))
    b = rng.normal(size=(10, 3))
    b[0, 0] = np.nan
    assert np.allclose(zscore_nan_fast(a, b), zscore_nan(a, b), equal_nan=True)


# ── rzscore_nan / rzscore_nan_fast ───────────────────────────────────────────

def test_rzscore_nan_matches_manual_median_mad(rng):
    a = rng.normal(size=(20, 2))
    med = np.median(a, axis=0)
    mad = np.median(np.abs(a - med), axis=0)
    expected = (a - med) / (1.4826 * mad)
    assert np.allclose(rzscore_nan(a), expected)


def test_rzscore_nan_zero_mad_is_nan():
    a = np.full((4, 1), 1.0)  # constant column -> MAD = 0
    result = rzscore_nan(a)
    assert np.all(np.isnan(result))


def test_rzscore_nan_fast_matches_rzscore_nan(rng):
    a = rng.normal(size=(10, 3))
    b = rng.normal(size=(10, 3))
    b[0, 0] = np.nan
    assert np.allclose(rzscore_nan_fast(a, b), rzscore_nan(a, b), equal_nan=True)


# ── prc / prc_fast ────────────────────────────────────────────────────────────

def test_prc_matches_manual_formula(rng):
    a = rng.uniform(1, 10, size=(5, 3))
    b = rng.uniform(1, 10, size=(5, 3))
    expected = (a - b) / a * 100
    assert np.allclose(prc(a, b), expected)


def test_prc_zero_a_is_nan():
    a = np.array([[0.0, 10.0]])
    b = np.array([[5.0, 5.0]])
    result = prc(a, b)
    assert np.isnan(result[0, 0])
    assert np.isclose(result[0, 1], 50.0)


def test_prc_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        prc(np.zeros((3, 2)), np.zeros((4, 2)))


def test_prc_fast_matches_prc(rng):
    a = rng.uniform(1, 10, size=(5, 3))
    b = rng.uniform(1, 10, size=(5, 3))
    assert np.allclose(prc_fast(a, b), prc(a, b))


# ── logfc_nan / logfc_fast ────────────────────────────────────────────────────

def test_logfc_nan_zero_shift_for_nonnegative_input():
    a = np.array([[2.0, 4.0]])
    b = np.array([[1.0, 8.0]])
    expected = np.log(a / b)  # shift = 0 since all-nonnegative
    assert np.allclose(logfc_nan(a, b), expected)


def test_logfc_nan_symmetry_under_swap():
    a = np.array([[2.0, -1.0]])
    b = np.array([[5.0, 3.0]])  # negative value present -> nonzero shift
    lfc_ab = logfc_nan(a, b)
    lfc_ba = logfc_nan(b, a)
    assert np.allclose(lfc_ab, -lfc_ba)


def test_logfc_nan_masks_nan_explicitly():
    a = np.array([[1.0, np.nan]])
    b = np.array([[2.0, 3.0]])
    result = logfc_nan(a, b)
    assert np.isnan(result[0, 1])
    assert not np.isnan(result[0, 0])


def test_logfc_nan_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        logfc_nan(np.zeros((3, 2)), np.zeros((4, 2)))


def test_logfc_fast_matches_logfc_nan(rng):
    a = rng.normal(size=(5, 3))  # includes negatives -> nonzero shift path
    b = rng.normal(size=(5, 3))
    assert np.allclose(logfc_fast(a, b), logfc_nan(a, b))


# ── centile_fast ──────────────────────────────────────────────────────────────

def test_centile_fast_against_reference():
    a = np.array([[5.0], [50.0], [95.0]])
    b = np.array([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0], [70.0], [80.0], [90.0]])
    result = centile_fast(a, b).ravel()
    # manual: searchsorted(side="right") / n_valid * 100
    expected = np.array([
        np.searchsorted(np.sort(b.ravel()), v, side="right") / b.shape[0] * 100
        for v in a.ravel()
    ])
    assert np.allclose(result, expected)


def test_centile_fast_self_reference_and_nan():
    a = np.array([[1.0], [np.nan], [3.0]])
    result = centile_fast(a).ravel()
    assert np.isnan(result[1])
    assert np.allclose(result[[0, 2]], [50.0, 100.0])


def test_centile_fast_output_range():
    a = np.random.default_rng(5).normal(size=(20, 2))
    result = centile_fast(a)
    valid = result[~np.isnan(result)]
    assert np.all(valid >= 0) and np.all(valid <= 100)


# ── Ground-truth checks against established libraries ───────────────────────
# scipy/pingouin, called directly (not through another nispace wrapper).

# -- cohen / cohen_nan vs pingouin.compute_effsize(eftype="cohen") -----------

def test_cohen_matches_pingouin(rng):
    pg = pytest.importorskip("pingouin")
    a = rng.normal(1.0, size=(15, 3))
    b = rng.normal(0.0, size=(18, 3))
    result = cohen(a, b)
    for i in range(3):
        expected = pg.compute_effsize(a[:, i], b[:, i], eftype="cohen")
        assert np.isclose(result[i], expected)


def test_cohen_nan_matches_pingouin_on_dropna(rng):
    # pingouin has no NaN handling -- ground-truth against pingouin computed
    # on the same per-column dropna'd 1D data.
    pg = pytest.importorskip("pingouin")
    a = rng.normal(1.0, size=(15, 1))
    b = rng.normal(0.0, size=(18, 1))
    a[2, 0] = np.nan
    b[5, 0] = np.nan
    result = cohen_nan(a, b)[0]
    expected = pg.compute_effsize(
        a[~np.isnan(a[:, 0]), 0], b[~np.isnan(b[:, 0]), 0], eftype="cohen"
    )
    assert np.isclose(result, expected)


def test_cohen_paired_matches_pingouin_cohen_dz(rng):
    # nispace's cohen_paired uses the SD-of-difference-scores formula, i.e.
    # what pingouin calls "cohen_dz" -- NOT pingouin's default paired "cohen"
    # (which is the d_avg formula, sqrt((var1+var2)/2) in the denominator, a
    # materially different number; see test_cohen_paired_matches_manual_formula
    # above, which already checks nispace against the diff.mean()/diff.std(ddof=1)
    # formula directly). d_z is the internally-consistent choice given nispace's
    # other paired Y_transform options (prc/logfc/diff are all pure functions of
    # the paired observations alone; d_avg would break that by pulling in each
    # condition's *unpaired* variance instead).
    #
    # eftype="cohen_dz" itself needs pingouin>=0.6 (Python>=3.10), unavailable
    # on the 3.9 test env -- so derive it version-independently instead, via
    # pingouin's own documented identity d_z = t / sqrt(n) from the paired
    # t-test (pg.ttest exists in every pingouin version).
    pg = pytest.importorskip("pingouin")
    a = rng.normal(size=(20, 2))
    b = a + rng.normal(scale=0.3, size=(20, 2))
    n = a.shape[0]
    result = cohen_paired(a, b)
    for i in range(2):
        t = pg.ttest(a[:, i], b[:, i], paired=True)["T"].iloc[0]
        expected = t / np.sqrt(n)
        assert np.isclose(result[i], expected)


# -- hedges / hedges_nan vs pingouin.compute_effsize(eftype="hedges") --------

def test_hedges_matches_pingouin(rng):
    pg = pytest.importorskip("pingouin")
    a = rng.normal(1.0, size=(15, 3))
    b = rng.normal(0.0, size=(18, 3))
    result = hedges(a, b)
    for i in range(3):
        expected = pg.compute_effsize(a[:, i], b[:, i], eftype="hedges")
        assert np.isclose(result[i], expected)


def test_hedges_nan_matches_pingouin_on_dropna(rng):
    pg = pytest.importorskip("pingouin")
    a = rng.normal(1.0, size=(15, 1))
    b = rng.normal(0.0, size=(18, 1))
    a[2, 0] = np.nan
    b[5, 0] = np.nan
    result = hedges_nan(a, b)[0]
    expected = pg.compute_effsize(
        a[~np.isnan(a[:, 0]), 0], b[~np.isnan(b[:, 0]), 0], eftype="hedges"
    )
    assert np.isclose(result, expected)


# -- zscore / zscore_nan vs scipy.stats.zscore --------------------------------

def test_zscore_matches_scipy(rng):
    from scipy.stats import zscore as scipy_zscore
    a = rng.normal(size=(20, 3))
    assert np.allclose(zscore(a), scipy_zscore(a, axis=0, ddof=1))


def test_zscore_nan_matches_scipy_nan_policy_omit(rng):
    from scipy.stats import zscore as scipy_zscore
    a = rng.normal(size=(20, 3))
    a[0, 0] = np.nan
    a[5, 1] = np.nan
    result = zscore_nan(a)
    expected = scipy_zscore(a, axis=0, ddof=1, nan_policy="omit")
    assert np.allclose(result, expected, equal_nan=True)


# -- centile_fast vs scipy.stats.percentileofscore ----------------------------

def test_centile_fast_matches_scipy_percentileofscore(rng):
    from scipy.stats import percentileofscore
    b = rng.normal(size=30)
    a = rng.normal(size=5)
    result = centile_fast(a[:, np.newaxis], b[:, np.newaxis]).ravel()
    expected = np.array([percentileofscore(b, v, kind="weak") for v in a])
    assert np.allclose(result, expected)


def test_centile_fast_with_nan_matches_scipy_nan_policy_omit(rng):
    from scipy.stats import percentileofscore
    b = rng.normal(size=30)
    b[3] = np.nan
    a = rng.normal(size=5)
    result = centile_fast(a[:, np.newaxis], b[:, np.newaxis]).ravel()
    expected = np.array([
        percentileofscore(b, v, kind="weak", nan_policy="omit") for v in a
    ])
    assert np.allclose(result, expected)
