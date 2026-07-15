"""Tests for load-bearing functions in stats/misc.py:
- null_to_p
- rho_to_z / z_to_rho
"""

import numpy as np
import pytest
from nispace.stats.misc import null_to_p, rho_to_z, z_to_rho


# ---------------------------------------------------------------------------
# null_to_p
# ---------------------------------------------------------------------------

@pytest.fixture
def null():
    """Null distribution: 1000 standard-normal draws, fixed seed."""
    return np.random.default_rng(0).standard_normal(1000)


# --- tail validation ---

def test_invalid_tail_raises(null):
    with pytest.raises(ValueError, match="tail"):
        null_to_p(0.0, null, tail="both")


# --- upper tail ---

def test_upper_extreme_positive_gives_small_p(null):
    # value well above the null → very small p
    p = null_to_p(10.0, null, tail="upper")
    assert p < 0.01


def test_upper_extreme_negative_gives_large_p(null):
    # value well below the null → p near 1
    p = null_to_p(-10.0, null, tail="upper")
    assert p > 0.90


# --- lower tail ---

def test_lower_extreme_negative_gives_small_p(null):
    p = null_to_p(-10.0, null, tail="lower")
    assert p < 0.01


def test_lower_extreme_positive_gives_large_p(null):
    p = null_to_p(10.0, null, tail="lower")
    assert p > 0.90


# --- two-tailed symmetry ---

def test_two_tailed_symmetric():
    # Use a perfectly symmetric null (base + its negation) so +v and -v give identical p
    base = np.random.default_rng(0).standard_normal(500)
    null_sym = np.concatenate([base - base.mean(), -(base - base.mean())])
    p_pos = null_to_p(3.0, null_sym, tail="two")
    p_neg = null_to_p(-3.0, null_sym, tail="two")
    np.testing.assert_allclose(p_pos, p_neg, rtol=1e-6)


def test_two_tailed_extreme_gives_small_p(null):
    p = null_to_p(10.0, null, tail="two")
    assert p < 0.01


# --- return type ---

def test_scalar_input_returns_scalar(null):
    p = null_to_p(1.0, null, tail="upper")
    assert isinstance(p, (float, np.floating))


def test_array_input_returns_array(null):
    p = null_to_p(np.array([0.0, 1.0, 2.0]), null, tail="upper")
    assert isinstance(p, np.ndarray)
    assert p.shape == (3,)


# --- p clipping: never exactly 0 or 1 ---

def test_p_never_zero_or_one(null):
    for val in [-100.0, 100.0]:
        for tail in ("upper", "lower", "two"):
            p = null_to_p(val, null, tail=tail)
            assert p > 0.0
            assert p < 1.0


def test_p_clipped_to_1_over_n(null):
    # minimum p = 1 / len(null) = 0.001 for n=1000
    p = null_to_p(100.0, null, tail="upper")
    assert p >= 1.0 / len(null)


# --- null median value should give p ~0.5 for upper tail ---

def test_median_value_gives_p_near_half(null):
    med = float(np.median(null))
    p = null_to_p(med, null, tail="upper")
    assert 0.4 < p < 0.6


# --- fit_norm ---

def test_fit_norm_returns_value_in_01(null):
    p = null_to_p(2.0, null, tail="upper", fit_norm=True)
    assert 0.0 <= p <= 1.0


def test_fit_norm_extreme_positive_small_p(null):
    p = null_to_p(10.0, null, tail="upper", fit_norm=True)
    assert p < 0.01


# ---------------------------------------------------------------------------
# rho_to_z and z_to_rho
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rho", [-0.9, -0.5, 0.0, 0.3, 0.7, 0.9])
def test_round_trip_rho_to_z_to_rho(rho):
    z = rho_to_z(np.array([rho]))
    recovered = z_to_rho(z)
    np.testing.assert_allclose(recovered, [rho], atol=1e-10)


def test_rho_equals_1_does_not_produce_inf():
    z = rho_to_z(np.array([1.0]))
    assert np.isfinite(z).all()


def test_rho_equals_minus_1_does_not_produce_neginf():
    # -1 is NOT clipped by default (only +1 is), but arctanh(-1+eps) is finite
    z = rho_to_z(np.array([-1.0 + 1e-10]))
    assert np.isfinite(z).all()


def test_rho_zero_gives_z_zero():
    z = rho_to_z(np.array([0.0]))
    np.testing.assert_allclose(z, [0.0], atol=1e-12)


def test_z_to_rho_large_positive_approaches_1():
    rho = z_to_rho(np.array([100.0]))
    np.testing.assert_allclose(rho, [1.0], atol=1e-6)


def test_z_to_rho_large_negative_approaches_minus_1():
    rho = z_to_rho(np.array([-100.0]))
    np.testing.assert_allclose(rho, [-1.0], atol=1e-6)


def test_rho_to_z_array_vectorized():
    rhos = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])
    zs = rho_to_z(rhos.copy())
    assert zs.shape == rhos.shape
    recovered = z_to_rho(zs)
    np.testing.assert_allclose(recovered, rhos, atol=1e-10)


def test_rho_to_z_monotone():
    rhos = np.linspace(-0.95, 0.95, 20)
    zs = rho_to_z(rhos.copy())
    assert np.all(np.diff(zs) > 0)
