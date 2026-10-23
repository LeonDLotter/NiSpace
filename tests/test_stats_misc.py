"""Tests for load-bearing functions in stats/misc.py:
- null_to_p
- rho_to_z / z_to_rho
- residuals_nan / partial_residuals_nan (ground-truth checked against
  statsmodels.OLS -- these back the actual "partialpearson"/"partialspearman"
  colocalize() code path and clean_y()'s "protect" covariate-regression mode,
  see nispace.stats.coloc.partialpearson's docstring)
- pca (ground-truth checked against sklearn.decomposition.PCA)
- family-wise corrections: maxT_correction / step_maxT_correction (pinned against
  their pre-refactor outputs) and minP_correction / step_minP_correction
- permute_groups: every strategy, paired and unpaired
"""

import numpy as np
import pandas as pd
import pytest
from nispace.stats.misc import (
    null_to_p, rho_to_z, z_to_rho, residuals_nan, partial_residuals_nan, pca,
    maxT_correction, step_maxT_correction, minP_correction, step_minP_correction,
    kminP_correction,
    _family_scale_ratio, permute_groups,
)


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

def test_p_never_zero(null):
    for val in [-100.0, 100.0]:
        for tail in ("upper", "lower", "two"):
            p = null_to_p(val, null, tail=tail)
            assert 0.0 < p <= 1.0


def test_p_floor_is_augmented(null):
    # observed beyond the whole null: it counts as one of n + 1 draws
    n = len(null)
    assert null_to_p(100.0, null, tail="upper") == pytest.approx(1.0 / (n + 1))
    assert null_to_p(-100.0, null, tail="lower") == pytest.approx(1.0 / (n + 1))
    assert null_to_p(100.0, null, tail="two") == pytest.approx(2.0 / (n + 1))
    # observed below the whole null, upper tail: every draw is >= it
    assert null_to_p(-100.0, null, tail="upper") == 1.0


@pytest.mark.parametrize("tail", ["upper", "lower", "two"])
def test_p_valid_under_exchangeability(tail):
    # observed exchangeable with the null: P(p <= t) <= t, also at the floor (the old
    # count/n convention gave ~2x the nominal rate there for two tails)
    rng = np.random.default_rng(7)
    B, n_sim = 199, 20000
    draws = rng.standard_normal((n_sim, B + 1))
    p = np.array([null_to_p(d[0], d[1:], tail=tail) for d in draws])
    se = lambda t: 3 * np.sqrt(t * (1 - t) / n_sim)
    for t in ((2 if tail == "two" else 1) / (B + 1), 0.01, 0.05):
        assert np.mean(p <= t + 1e-12) <= t + se(t)


@pytest.mark.parametrize("tail", ["upper", "lower", "two"])
def test_p_matches_minp_column_p(tail):
    # null_to_p and the minP machinery use the same per-test p
    from nispace.stats.misc import _augmented_column_p
    rng = np.random.default_rng(3)
    null = np.round(rng.standard_normal((300, 4)), 1)   # rounded: ties included
    obs = np.array([0.0, 1.5, -2.2, 3.9])
    expected = _augmented_column_p(obs, null, tail)[0]
    got = np.array([null_to_p(obs[j], null[:, j], tail=tail) for j in range(4)])
    np.testing.assert_allclose(got, expected)


def test_p_nan_handling():
    null = np.array([0.1, np.nan, -0.3, 0.5, np.nan])
    # NaN null values are dropped from the reference (3 valid + observed = 4 draws)
    assert null_to_p(1.0, null, tail="upper") == pytest.approx(1 / 4)
    assert np.isnan(null_to_p(np.nan, null))
    np.testing.assert_array_equal(np.isnan(null_to_p(np.array([np.nan, 0.0]), null)),
                                  [True, False])


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


# ---------------------------------------------------------------------------
# residuals_nan vs statsmodels.OLS
# ---------------------------------------------------------------------------

def test_residuals_nan_matches_statsmodels_ols():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(9)
    n = 100
    X = rng.normal(size=(n, 2))
    y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(scale=0.5, size=n)

    result = residuals_nan(X, y)

    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    np.testing.assert_allclose(result, model.resid)


def test_residuals_nan_with_nan_matches_statsmodels_ols_on_complete_cases():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(9)
    n = 100
    X = rng.normal(size=(n, 2))
    y = rng.normal(size=n)
    X[3, 0] = np.nan
    y[10] = np.nan

    result = residuals_nan(X, y)

    Xc = sm.add_constant(X)
    mask = ~np.isnan(Xc).any(axis=1) & ~np.isnan(y)
    model = sm.OLS(y[mask], Xc[mask]).fit()

    assert np.isnan(result[3]) and np.isnan(result[10])
    np.testing.assert_allclose(result[mask], model.resid)


def test_residuals_nan_decenter_adds_back_mean():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(9)
    n = 60
    X = rng.normal(size=(n, 1))
    y = 0.4 * X[:, 0] + rng.normal(scale=0.3, size=n)

    plain = residuals_nan(X, y)
    decentered = residuals_nan(X, y, decenter=True)
    np.testing.assert_allclose(decentered, plain + y.mean())


# ---------------------------------------------------------------------------
# partial_residuals_nan vs statsmodels.OLS (joint model, nuisance-only removal)
# ---------------------------------------------------------------------------

def test_partial_residuals_nan_removes_only_nuisance_component():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(9)
    n = 100
    x_nuisance = rng.normal(size=(n, 1))
    x_protect = rng.normal(size=(n, 1))
    y = 0.5 * x_nuisance[:, 0] - 0.3 * x_protect[:, 0] + rng.normal(scale=0.5, size=n)

    result = partial_residuals_nan(x_nuisance, x_protect, y)

    X_full = sm.add_constant(np.column_stack([x_nuisance, x_protect]))
    model = sm.OLS(y, X_full).fit()
    nuisance_beta = model.params[1]  # column order: const, nuisance, protect
    expected = y - x_nuisance[:, 0] * nuisance_beta
    np.testing.assert_allclose(result, expected)


def test_partial_residuals_nan_with_nan_matches_statsmodels_on_complete_cases():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(9)
    n = 100
    x_nuisance = rng.normal(size=(n, 1))
    x_protect = rng.normal(size=(n, 1))
    y = rng.normal(size=n)
    x_nuisance[4, 0] = np.nan
    y[15] = np.nan

    result = partial_residuals_nan(x_nuisance, x_protect, y)

    stacked = np.column_stack([x_nuisance, x_protect, y])
    mask = ~np.isnan(stacked).any(axis=1)
    X_full = sm.add_constant(np.column_stack([x_nuisance[mask], x_protect[mask]]))
    model = sm.OLS(y[mask], X_full).fit()
    nuisance_beta = model.params[1]
    expected = y[mask] - x_nuisance[mask, 0] * nuisance_beta

    assert np.isnan(result[4]) and np.isnan(result[15])
    np.testing.assert_allclose(result[mask], expected)


# ---------------------------------------------------------------------------
# pca
# ---------------------------------------------------------------------------

@pytest.fixture
def pca_data():
    """(n_obs=50, n_features=8) random data, fixed seed."""
    return np.random.default_rng(3).normal(size=(50, 8))


def test_pca_ndarray_in_ndarray_out(pca_data):
    out = pca(pca_data, n_components=3)
    assert isinstance(out, np.ndarray)
    assert out.shape == (50, 3)


def test_pca_dataframe_in_dataframe_out(pca_data):
    df = pd.DataFrame(pca_data, index=[f"obs{i}" for i in range(50)],
                       columns=[f"feat{i}" for i in range(8)])
    out = pca(df, n_components=3)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (50, 3)
    assert list(out.columns) == ["PC1", "PC2", "PC3"]
    assert (out.index == df.index).all()


def test_pca_default_n_components_keeps_all(pca_data):
    out = pca(pca_data)
    assert out.shape == (50, 8)


def test_pca_matches_sklearn_subspace_and_explained_variance(pca_data):
    sklearn_decomposition = pytest.importorskip("sklearn.decomposition")
    sk = sklearn_decomposition.PCA(n_components=4).fit(pca_data)

    out, ev = pca(pca_data, n_components=4, return_ev=True)

    np.testing.assert_allclose(ev, sk.explained_variance_ratio_, atol=1e-8)
    # component 1 here is sign-fixed against the column mean (not sklearn's largest-loading
    # convention), so align signs before comparing subspaces directly
    for i in range(4):
        sign = 1.0 if np.corrcoef(out[:, i], sk.transform(pca_data)[:, i])[0, 1] >= 0 else -1.0
        np.testing.assert_allclose(out[:, i], sign * sk.transform(pca_data)[:, i], atol=1e-6)


def test_pca_component1_nonnegatively_correlated_with_column_mean():
    for seed in range(10):
        x = np.random.default_rng(seed).normal(size=(60, 12))
        out = pca(x, n_components=1)
        col_mean = (x - x.mean(axis=0)).mean(axis=1)
        r = np.corrcoef(out[:, 0], col_mean)[0, 1]
        assert r >= -1e-9, f"seed {seed}: PC1 negatively correlated with column mean (r={r})"


def test_pca_standardize_changes_result_when_scales_differ(pca_data):
    x = pca_data.copy()
    x[:, 0] *= 1000  # one column on a wildly different scale
    out_std = pca(x, n_components=3, standardize=True)
    out_raw = pca(x, n_components=3, standardize=False)
    assert not np.allclose(np.abs(out_std), np.abs(out_raw))


def test_pca_return_ev_sums_to_at_most_one(pca_data):
    out, ev = pca(pca_data, return_ev=True)
    assert ev.sum() <= 1.0 + 1e-8
    assert (ev >= 0).all()


def test_pca_rejects_unsupported_type():
    with pytest.raises(TypeError, match="not supported"):
        pca([[1, 2], [3, 4]])


def test_pca_rejects_non_2d():
    with pytest.raises(ValueError, match="2D"):
        pca(np.array([1.0, 2.0, 3.0]))


def test_pca_rejects_too_few_rows():
    with pytest.raises(ValueError, match="at least 2 rows"):
        pca(np.zeros((1, 5)))


def test_pca_rejects_nan():
    x = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
    with pytest.raises(ValueError, match="NaN"):
        pca(x)


def test_pca_rejects_n_components_out_of_range(pca_data):
    with pytest.raises(ValueError, match="out of range"):
        pca(pca_data, n_components=20)
    with pytest.raises(ValueError, match="out of range"):
        pca(pca_data, n_components=0)


# ---------------------------------------------------------------------------
# family-wise corrections: maxT / step_maxT / minP / step_minP
# ---------------------------------------------------------------------------

_FW_FUNS = [maxT_correction, step_maxT_correction, minP_correction, step_minP_correction]


def _as_nulls(null_arr, stat="rho"):
    """(n_perm, n_Y, n_X) array -> the list-of-dicts format the corrections consume."""
    return [{stat: null_arr[b]} for b in range(null_arr.shape[0])]


@pytest.fixture
def fw_data():
    """Seeded family: 2 Y rows x 5 X columns with heterogeneous null scales, n_perm=200.
    No observed value exceeds every family maximum, so the 1/(n_perm + 1) floor never binds."""
    rng = np.random.default_rng(20260921)
    null = rng.standard_normal((200, 2, 5)) * np.array([1.0, 0.5, 2.0, 1.0, 0.8])
    obs = np.array([[0.3, -0.9, 2.5, 1.7, -0.2],
                    [-1.1, 0.4, -3.0, 0.1, 1.9]])
    return obs, null


# pre-refactor outputs of maxT_correction/step_maxT_correction on fw_data, on the old
# count / n_perm scale -- the augmented convention maps each to (count + 1) / (n_perm + 1)
# (a monotone map, so the step-down cumulative max commutes with it)
_MAXT_PINNED = {
    ("maxT", "two", "r"): [[1., 0.91, 0.185, 0.515, 1.], [0.905, 0.995, 0.14, 1., 0.475]],
    ("step", "two", "r"): [[0.955, 0.585, 0.185, 0.215, 0.955], [0.515, 0.815, 0.14, 0.89, 0.14]],
    ("maxT", "two", "a"): [[1., 0.995, 0.39, 0.795, 1.], [0.975, 1., 0.25, 1., 0.68]],
    ("step", "two", "a"): [[0.995, 0.77, 0.25, 0.38, 0.995], [0.74, 0.995, 0.25, 0.995, 0.25]],
    ("maxT", "upper", "r"): [[0.885, 1., 0.065, 0.255, 0.975], [1., 0.89, 1., 0.965, 0.24]],
    ("step", "upper", "r"): [[0.72, 0.96, 0.065, 0.11, 0.85], [0.945, 0.835, 0.945, 0.835, 0.24]],
    ("maxT", "upper", "a"): [[0.99, 1., 0.19, 0.5, 1.], [1., 0.975, 1., 1., 0.385]],
    ("step", "upper", "a"): [[0.945, 1., 0.19, 0.4, 0.945], [1., 0.94, 1., 0.945, 0.295]],
}


@pytest.mark.parametrize("key", list(_MAXT_PINNED))
def test_maxT_step_maxT_pinned(fw_data, key):
    obs, null = fw_data
    fn = maxT_correction if key[0] == "maxT" else step_maxT_correction
    p, reject = fn(obs, _as_nulls(null), "rho", tail=key[1], how=key[2])
    n_perm = null.shape[0]
    expected = (np.array(_MAXT_PINNED[key]) * n_perm + 1) / (n_perm + 1)
    np.testing.assert_allclose(p, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(reject, expected <= 0.05)


@pytest.mark.parametrize("fn", [maxT_correction, step_maxT_correction])
def test_maxT_floor_is_augmented(fw_data, fn):
    # an observation beyond every family maximum gets 1 / (n_perm + 1), never 0
    obs, null = fw_data
    obs_ext = obs.copy()
    obs_ext[0, 2] = 100.0
    p, _ = fn(obs_ext, _as_nulls(null), "rho", how="r")
    assert p[0, 2] == pytest.approx(1 / (null.shape[0] + 1))


@pytest.mark.parametrize("fn", _FW_FUNS)
@pytest.mark.parametrize("how", ["r", "a"])
def test_family_nan_obs_column_excluded(fw_data, fn, how):
    # a test with NaN observed statistic is dropped from its family: it gets NaN, and
    # every other test is corrected exactly as if it had never been there (it used to
    # poison maxT's family max, turning every p into 0)
    obs, null = fw_data
    obs_nan = obs.copy()
    obs_nan[:, 2] = np.nan
    keep = [0, 1, 3, 4]
    p, reject = fn(obs_nan, _as_nulls(null), "rho", how=how)
    p_ref, _ = fn(obs[:, keep], _as_nulls(null[:, :, keep]), "rho", how=how)
    assert np.isnan(p[:, 2]).all() and not reject[:, 2].any()
    np.testing.assert_array_equal(p[:, keep], p_ref)


@pytest.mark.parametrize("fn", _FW_FUNS)
def test_family_sporadic_null_nan_does_not_poison(fw_data, fn):
    obs, null = fw_data
    null_nan = null.copy()
    null_nan[::7, 0, 1] = np.nan  # one test loses some permutations
    p, _ = fn(obs, _as_nulls(null_nan), "rho")
    assert np.isfinite(p).all()
    assert ((p > 0) & (p <= 1)).all()


@pytest.mark.parametrize("fn", _FW_FUNS)
def test_family_shape_mismatch_raises(fw_data, fn):
    # e.g. a pooled (1, n_X) null against unpooled (n_Y, n_X) observations must fail
    # loudly instead of being broadcast into a silently wrong (n_Y, n_X) result
    obs, null = fw_data
    with pytest.raises(ValueError, match="do not match"):
        fn(obs, _as_nulls(null[:, :1, :]), "rho")


@pytest.mark.parametrize("fn", _FW_FUNS)
def test_family_invalid_tail_or_how_raises(fw_data, fn):
    obs, null = fw_data
    with pytest.raises(ValueError, match="tail"):
        fn(obs, _as_nulls(null), "rho", tail="both")
    with pytest.raises(ValueError, match="not supported"):
        fn(obs, _as_nulls(null), "rho", how="c")


@pytest.mark.parametrize("fn", [minP_correction, step_minP_correction])
@pytest.mark.parametrize("tail", ["two", "upper", "lower"])
def test_minP_invariant_to_per_test_affine_transform(fw_data, fn, tail):
    # minP works on per-test ranks, so rescaling/shifting one test's observed value and
    # null together changes nothing -- the balance property maxT lacks
    obs, null = fw_data
    obs2, null2 = obs.copy(), null.copy()
    obs2[:, 1] = obs2[:, 1] * 37.0 + 5.0
    null2[:, :, 1] = null2[:, :, 1] * 37.0 + 5.0
    p1, _ = fn(obs, _as_nulls(null), "rho", tail=tail)
    p2, _ = fn(obs2, _as_nulls(null2), "rho", tail=tail)
    np.testing.assert_array_equal(p1, p2)


def test_maxT_not_invariant_to_per_test_affine_transform(fw_data):
    obs, null = fw_data
    obs2, null2 = obs.copy(), null.copy()
    obs2[:, 1] *= 37.0
    null2[:, :, 1] *= 37.0
    p1, _ = maxT_correction(obs, _as_nulls(null), "rho")
    p2, _ = maxT_correction(obs2, _as_nulls(null2), "rho")
    assert not np.array_equal(p1, p2)


@pytest.mark.parametrize("fn", _FW_FUNS)
@pytest.mark.parametrize("tail", ["two", "upper", "lower"])
def test_family_p_raw_floor(fw_data, fn, tail):
    # with the uncorrected p passed, a corrected p is never below it, nor below 1/n_perm
    obs, null = fw_data
    n_perm = null.shape[0]
    p_raw = np.array([[null_to_p(obs[y, x], null[:, y, x], tail=tail) for x in range(5)]
                      for y in range(2)])
    p, _ = fn(obs, _as_nulls(null), "rho", tail=tail, p_raw=p_raw)
    assert (p >= p_raw).all()
    assert (p >= 1.0 / n_perm).all()


@pytest.mark.parametrize("single,step", [(maxT_correction, step_maxT_correction),
                                         (minP_correction, step_minP_correction)])
@pytest.mark.parametrize("how", ["r", "a"])
def test_family_stepdown_never_more_conservative(fw_data, single, step, how):
    obs, null = fw_data
    p_single, _ = single(obs, _as_nulls(null), "rho", how=how)
    p_step, _ = step(obs, _as_nulls(null), "rho", how=how)
    assert (p_step <= p_single + 1e-12).all()


@pytest.mark.parametrize("fn", _FW_FUNS)
def test_family_tail_direction(fn):
    rng = np.random.default_rng(1)
    null = rng.standard_normal((500, 1, 3))
    obs = np.array([[4.0, -4.0, 0.0]])
    p_up, _ = fn(obs, _as_nulls(null), "rho", tail="upper")
    p_lo, _ = fn(obs, _as_nulls(null), "rho", tail="lower")
    assert p_up[0, 0] < 0.05 and p_up[0, 1] > 0.5
    assert p_lo[0, 1] < 0.05 and p_lo[0, 0] > 0.5


@pytest.mark.parametrize("tail", ["two", "upper", "lower"])
def test_minP_single_test_family_matches_null_to_p(tail):
    # with a family of one there is nothing to correct: minP reduces to the per-test
    # permutation p (up to the augmented n_perm+1 convention)
    rng = np.random.default_rng(2)
    null = rng.standard_normal((400, 3, 1))
    obs = np.array([[0.5], [-1.8], [2.4]])
    p, _ = minP_correction(obs, _as_nulls(null), "rho", tail=tail)
    expected = np.array([[null_to_p(obs[y, 0], null[:, y, 0], tail=tail)] for y in range(3)])
    np.testing.assert_allclose(p, expected, atol=2.5 / 400)


@pytest.mark.parametrize("fn", [step_maxT_correction, step_minP_correction])
@pytest.mark.parametrize("how", ["r", "a"])
def test_stepdown_monotone_after_conflicting_p_raw_floor(fn, how):
    # a p_raw that orders the tests against the step-down order (possible with null_to_p
    # centring or fit_norm) must not break monotonicity: the floor is applied before the
    # step-down cumulative max, not after it
    rng = np.random.default_rng(4)
    null = rng.standard_normal((500, 1, 4))
    obs = np.array([[4.0, 3.0, 2.5, 0.2]])  # step-down order = column order
    p_raw = np.array([[0.3, 0.001, 0.002, 0.9]])
    p, _ = fn(obs, _as_nulls(null), "rho", tail="upper", how=how, p_raw=p_raw)
    p = np.asarray(p)[0]
    assert (np.diff(p) >= 0).all(), p
    assert (p >= p_raw[0]).all()
    assert p[1] >= 0.3  # pulled up by the more extreme test's floor


def test_minP_balanced_where_maxT_is_not():
    # complete null, 8 exchangeable tests whose null scales span 2 orders of magnitude:
    # both control FWER, but raw maxT hands (nearly) all rejections to the widest test
    # while minP spreads them over all tests
    rng = np.random.default_rng(3)
    n_perm, m, n_sim = 999, 8, 300
    scales = np.geomspace(0.01, 1.0, m)  # widest test = last
    rej_maxT = np.zeros(m)
    rej_minP = np.zeros(m)
    fwer = {"maxT": {0.05: 0, 0.2: 0}, "minP": {0.05: 0, 0.2: 0}}
    for _ in range(n_sim):
        allv = rng.standard_normal((n_perm + 1, 1, m)) * scales
        obs, nulls = allv[0], _as_nulls(allv[1:])
        p_max, _ = maxT_correction(obs, nulls, "rho")
        p_min, _ = minP_correction(obs, nulls, "rho")
        for name, p in [("maxT", p_max), ("minP", p_min)]:
            for a in fwer[name]:
                fwer[name][a] += (p <= a).any()
        rej_maxT += (p_max[0] <= 0.2)
        rej_minP += (p_min[0] <= 0.2)
    for name in fwer:
        for a, n_fw in fwer[name].items():
            se = np.sqrt(a * (1 - a) / n_sim)
            assert n_fw / n_sim <= a + 3 * se, (name, a, n_fw / n_sim)
    assert rej_maxT[-1] / rej_maxT.sum() >= 0.9
    assert rej_minP.max() / rej_minP.sum() <= 0.4
    assert (rej_minP > 0).sum() >= m - 1


def test_minP_resolution_warning(caplog):
    # 30 two-tailed tests with only 99 permutations: no test can reach alpha
    rng = np.random.default_rng(4)
    null = rng.standard_normal((99, 1, 30))
    obs = rng.standard_normal((1, 30))
    with caplog.at_level("WARNING"):
        minP_correction(obs, _as_nulls(null), "rho")
    assert any("no test can be declared significant" in r.message for r in caplog.records)


def test_family_scale_ratio_flags_heterogeneous_nulls():
    rng = np.random.default_rng(5)
    null = rng.standard_normal((1000, 2, 3)) * np.array([1.0, 1.0, 1.0])
    null[:, 1, 2] *= 10.0  # row 1, column 2 is 10x wider
    ratio, widest, narrowest = _family_scale_ratio(null, "two", how="r")
    assert ratio > 5
    assert widest == (1, 2) and narrowest[0] == 1
    ratio_eq, _, _ = _family_scale_ratio(null[:, :1, :], "two", how="r")
    assert ratio_eq < 1.5


# --- kminP (generalized FWER) ---

@pytest.mark.parametrize("how", ["r", "a"])
@pytest.mark.parametrize("tail", ["two", "upper"])
def test_kminP_n_false_0_is_minP(fw_data, how, tail):
    obs, null = fw_data
    p_k, rej_k = kminP_correction(obs, _as_nulls(null), "rho", tail=tail, how=how, n_false=0)
    p_m, rej_m = minP_correction(obs, _as_nulls(null), "rho", tail=tail, how=how)
    np.testing.assert_array_equal(p_k, p_m)
    np.testing.assert_array_equal(rej_k, rej_m)


def test_kminP_monotone_in_n_false(fw_data):
    obs, null = fw_data
    prev = None
    for n_false in range(5):
        p, _ = kminP_correction(obs, _as_nulls(null), "rho", how="a", n_false=n_false)
        if prev is not None:
            assert (p <= prev + 1e-15).all()
        prev = p


def test_kminP_hand_computed():
    # 3 tests, 4 permutations, upper tail. Augmented column p per draw (row 0 = observed):
    #   obs [0.4, 0.4, 0.2], perms [1, .2, 1], [.8, .8, .4], [.6, .6, .8], [.2, 1, .6]
    # n_false=1: 2nd smallest per permutation = 1, .8, .6, .6 -> none <= .4 -> p = 1/5
    # n_false=0: min per permutation = .2, .4, .6, .2 -> 3 <= .4, 2 <= .2 -> p = 4/5, 3/5
    obs = np.array([[3.0, 2.0, 1.5]])
    null = np.array([[0.1, 2.5, 0.0], [0.2, 0.0, 1.0], [0.3, 1.0, 0.2],
                     [4.0, -1.0, 0.3]])[:, np.newaxis, :]
    p1, _ = kminP_correction(obs, _as_nulls(null), "rho", tail="upper", n_false=1)
    np.testing.assert_allclose(p1, [[0.2, 0.2, 0.2]])
    p0, _ = kminP_correction(obs, _as_nulls(null), "rho", tail="upper", n_false=0)
    np.testing.assert_allclose(p0, [[0.8, 0.8, 0.6]])


def test_kminP_small_family_uncorrected(fw_data):
    # a family with <= n_false tests can't make n_false + 1 errors: returned uncorrected
    obs, null = fw_data
    p, _ = kminP_correction(obs[:, :2], _as_nulls(null[:, :, :2]), "rho", n_false=2)
    p_unc, _ = kminP_correction(obs[:, :2], _as_nulls(null[:, :, :2]), "rho", n_false=0)
    assert (p <= p_unc).all()
    for y in range(2):
        for x in range(2):
            assert p[y, x] == pytest.approx(null_to_p(obs[y, x], null[:, y, x], tail="two"))


@pytest.mark.parametrize("n_false", [1, 2])
def test_kminP_controls_generalized_fwer(n_false):
    # complete null with correlated tests: each permutation in turn is the "observed" data;
    # P(>= n_false + 1 rejections) must stay <= alpha (up to Monte Carlo error)
    rng = np.random.default_rng(11)
    m, B, n_rep = 12, 999, 300  # B large enough that minP can reach alpha for m = 12
    L = rng.standard_normal((m, 3))
    cov = L @ L.T + np.eye(m)
    chol = np.linalg.cholesky(cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov))))
    n_fail = 0
    for _ in range(n_rep):
        draws = (chol @ rng.standard_normal((m, B + 1))).T  # (B + 1, m)
        p, _ = kminP_correction(draws[:1], _as_nulls(draws[1:, np.newaxis, :]), "rho",
                                n_false=n_false)
        n_fail += (p <= 0.05).sum() >= n_false + 1
    rate = n_fail / n_rep
    se = np.sqrt(0.05 * 0.95 / n_rep)
    assert 0.05 - 3 * se < rate <= 0.05 + 3 * se  # valid, and not trivially conservative


def test_kminP_resolution_warning(caplog):
    # 10 two-tailed tests, 19 permutations: the column-p floor is 2/20 = 0.1 > alpha
    rng = np.random.default_rng(1)
    obs = rng.standard_normal((1, 10))
    null = rng.standard_normal((19, 1, 10))
    with caplog.at_level("WARNING"):
        kminP_correction(obs, _as_nulls(null), "rho", n_false=1)
    assert any("kminP(1)" in r.message and "n_perm" in r.message for r in caplog.records)


@pytest.mark.parametrize("bad", [-1, 1.5, True, "1"])
def test_kminP_invalid_n_false(fw_data, bad):
    obs, null = fw_data
    with pytest.raises(ValueError):
        kminP_correction(obs, _as_nulls(null), "rho", n_false=bad)


# ---------------------------------------------------------------------------
# permute_groups
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sizes", [(7, 5), (8, 3), (10, 10), (5, 5, 3), (7, 2, 2), (3, 3, 3, 2)])
def test_permute_groups_prop_preserves_group_sizes(sizes):
    """Regression: with 3+ groups the floor-rounding leftovers were filled with one copy of
    each label, which crashed whenever the leftover count differed from the group count
    (e.g. sizes 5/5/3). Two groups always leave 0 or 2 over, so they worked by chance."""
    groups = np.repeat(np.arange(len(sizes)), sizes)
    for perm in permute_groups(groups, strategy="prop", n_perm=10, seed=0):
        np.testing.assert_array_equal(np.bincount(perm, minlength=len(sizes)), sizes)


def test_permute_groups_prop_composition_is_proportional():
    """Each new group takes floor(size * share) samples from every original group (the
    JuSpace scheme); with 10/10 that is exactly 5 from each into each."""
    groups = np.repeat([0, 1], 10)
    for perm in permute_groups(groups, strategy="prop", n_perm=10, seed=1):
        for new in (0, 1):
            np.testing.assert_array_equal(np.bincount(groups[perm == new], minlength=2), [5, 5])


@pytest.mark.parametrize("paired", [False, True])
def test_permute_groups_prop_string_labels(paired):
    """Regression: "prop" built the result in a float NaN array, so string labels
    crashed (the API dummy-codes groups first; direct calls didn't)."""
    groups = np.repeat(["hc", "pat"], 4)
    kw = dict(paired=True, subjects=np.tile(np.arange(4), 2)) if paired else {}
    perm = permute_groups(groups, strategy="prop", seed=0, **kw)
    assert perm.dtype == groups.dtype
    assert sorted(perm) == sorted(groups)


def test_permute_groups_shuffle_and_draw_unpaired():
    groups = np.repeat([0, 1, 2], [4, 3, 5])
    shuffled = permute_groups(groups, strategy="shuffle", n_perm=20, seed=0)
    assert all(np.array_equal(np.bincount(p), [4, 3, 5]) for p in shuffled)
    drawn = permute_groups(groups, strategy="draw", n_perm=20, seed=0)
    assert all(len(p) == len(groups) and set(p) <= {0, 1, 2} for p in drawn)
    # with replacement: label counts vary across draws
    assert len({tuple(np.bincount(p, minlength=3)) for p in drawn}) > 1


@pytest.mark.parametrize("strategy", ["shuffle", "prop"])
def test_permute_groups_paired_permutes_within_subject(strategy):
    n_sub, n_ses = 9, 2
    groups = np.repeat(np.arange(n_ses), n_sub)
    subjects = np.tile(np.arange(n_sub), n_ses)
    for perm in permute_groups(groups, strategy=strategy, paired=True, subjects=subjects,
                               n_perm=10, seed=0):
        for sub in range(n_sub):
            assert sorted(perm[subjects == sub]) == list(range(n_ses))
        if strategy == "prop":
            # exactly floor(n_sub / n_ses!) subjects per session ordering, rest random
            n_swapped = int((perm[:n_sub] == 1).sum())
            assert n_sub // 2 <= n_swapped <= n_sub // 2 + 1


@pytest.mark.parametrize("strategy,kw", [("shuffle", {}), ("prop", {}), ("draw", {}),
                                         ("prop", {"paired": True}),
                                         ("shuffle", {"paired": True})])
def test_permute_groups_seed_reproducible_and_n_perm_shape(strategy, kw):
    groups = np.repeat([0, 1], 6)
    if kw.get("paired"):
        kw = dict(kw, subjects=np.tile(np.arange(6), 2))
    a = permute_groups(groups, strategy=strategy, n_perm=5, seed=3, **kw)
    b = permute_groups(groups, strategy=strategy, n_perm=5, seed=3, **kw)
    assert isinstance(a, list) and len(a) == 5
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    single = permute_groups(groups, strategy=strategy, n_perm=1, seed=3, **kw)
    assert isinstance(single, np.ndarray)
    np.testing.assert_array_equal(single, a[0])


def test_permute_groups_errors():
    groups = np.repeat([0, 1], 3)
    subjects = np.tile([1, 2, 3], 2)
    with pytest.raises(ValueError, match="requires argument 'subjects'"):
        permute_groups(groups, paired=True)
    with pytest.raises(ValueError, match="Number of subjects"):
        permute_groups(groups, paired=True, subjects=subjects[:-1])
    with pytest.raises(ValueError, match="same number of samples"):
        permute_groups(np.array([0, 0, 0, 1, 1, 1, 1]), paired=True,
                       subjects=np.array([1, 2, 3, 1, 2, 3, 4]))
    with pytest.raises(ValueError, match="Unknown unpaired"):
        permute_groups(groups, strategy="bogus")
    with pytest.raises(ValueError, match="Unknown paired"):
        permute_groups(groups, strategy="draw", paired=True, subjects=subjects)

