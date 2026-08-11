"""Tests for `_check_predictor_count` -- statistical/computational guards for
multivariate colocalization methods (mlr/dominance/pcr/pls) with many X
predictors relative to usable parcels. Independent of the memory/batching
work in test_permute.py: these guard against pre-existing correctness
issues (rank-deficient OLS, combinatorial blowup) that exist regardless of
call size.
"""

import numpy as np
import pandas as pd
import pytest
import logging

from nispace import NiSpace
from nispace.core.colocalize import _check_predictor_count, _DOMINANCE_MAX_PREDICTORS_DEFAULT


# ── _check_predictor_count: pure unit tests ─────────────────────────────────

def test_mlr_warns_when_predictors_exceed_obs(caplog):
    with caplog.at_level(logging.WARNING):
        _check_predictor_count("mlr", n_predictors=50, n_obs=30)
    assert "rank-deficient" in caplog.text


def test_mlr_no_warning_when_predictors_well_below_obs(caplog):
    with caplog.at_level(logging.WARNING):
        _check_predictor_count("mlr", n_predictors=3, n_obs=100)
    assert caplog.text == ""


def test_mlr_boundary_equal_predictors_and_obs_warns(caplog):
    # n_predictors == n_obs is exactly rank-deficient too (>=), not just >
    with caplog.at_level(logging.WARNING):
        _check_predictor_count("mlr", n_predictors=20, n_obs=20)
    assert "rank-deficient" in caplog.text


def test_dominance_warns_like_mlr_below_infeasible_cutoff(caplog):
    with caplog.at_level(logging.WARNING):
        _check_predictor_count("dominance", n_predictors=15, n_obs=10)
    assert "rank-deficient" in caplog.text


def test_dominance_raises_above_max_predictors_default():
    with pytest.raises(ValueError, match="computationally infeasible"):
        _check_predictor_count("dominance", n_predictors=25, n_obs=1000)


def test_dominance_default_cutoff_matches_constant():
    # exactly at the default cutoff must NOT raise; one above must
    _check_predictor_count("dominance", n_predictors=_DOMINANCE_MAX_PREDICTORS_DEFAULT,
                           n_obs=1000)
    with pytest.raises(ValueError):
        _check_predictor_count("dominance", n_predictors=_DOMINANCE_MAX_PREDICTORS_DEFAULT + 1,
                               n_obs=1000)


def test_dominance_max_predictors_override_raises_threshold():
    # would raise at the default cutoff, but an explicit higher override permits it
    _check_predictor_count("dominance", n_predictors=23, n_obs=1000,
                           dominance_max_predictors=30)
    with pytest.raises(ValueError):
        _check_predictor_count("dominance", n_predictors=23, n_obs=1000,
                               dominance_max_predictors=20)


def test_pcr_warns_above_threshold(caplog):
    with caplog.at_level(logging.WARNING):
        _check_predictor_count("pcr", n_predictors=600, n_obs=1000)
    assert "O(p^2)" in caplog.text or "PCA" in caplog.text


def test_pcr_no_warning_below_threshold(caplog):
    with caplog.at_level(logging.WARNING):
        _check_predictor_count("pcr", n_predictors=100, n_obs=1000)
    assert caplog.text == ""


def test_pls_warns_when_predictors_approach_obs(caplog):
    with caplog.at_level(logging.WARNING):
        _check_predictor_count("pls", n_predictors=50, n_obs=30)
    assert "overfit" in caplog.text


def test_pls_no_warning_when_predictors_well_below_obs(caplog):
    with caplog.at_level(logging.WARNING):
        _check_predictor_count("pls", n_predictors=3, n_obs=100)
    assert caplog.text == ""


@pytest.mark.parametrize("method", ["pearson", "spearman", "mi", "slr", "lasso", "ridge",
                                    "elasticnet"])
def test_no_guard_for_non_gated_methods(method, caplog):
    # univariate methods and the regularized methods (designed for p>>n) are
    # entirely untouched by this guard, even at extreme predictor counts
    with caplog.at_level(logging.WARNING):
        _check_predictor_count(method, n_predictors=10000, n_obs=10)
    assert caplog.text == ""


# ── colocalize() integration ─────────────────────────────────────────────────

def _make_nsp(n_x, n_y=1, n_parcels=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_x, n_parcels))
    Y = rng.normal(size=(n_y, n_parcels))
    x_df = pd.DataFrame(X, index=[f"x{i}" for i in range(n_x)],
                        columns=[f"p{i}" for i in range(n_parcels)])
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(n_y)],
                        columns=[f"p{i}" for i in range(n_parcels)])
    nsp = NiSpace(x=x_df, y=y_df, z=None, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    return nsp


def test_colocalize_mlr_warns_on_many_predictors(caplog):
    nsp = _make_nsp(n_x=25, n_parcels=20)
    with caplog.at_level(logging.WARNING):
        nsp.colocalize("mlr", verbose=False)
    assert "rank-deficient" in caplog.text


def test_colocalize_dominance_raises_on_too_many_predictors():
    # raises before any model fitting -- n_parcels doesn't matter, this must be fast
    # regardless (2**25-1 models would otherwise be a multi-hour computation)
    nsp = _make_nsp(n_x=25, n_parcels=30)
    with pytest.raises(ValueError, match="computationally infeasible"):
        nsp.colocalize("dominance", verbose=False)


def test_colocalize_dominance_max_predictors_override():
    # small predictor count so the actual dominance fit (2**n-1 models) stays fast;
    # this only needs to confirm the override parameter threads through to the guard,
    # not exercise a large computation (that's already covered by the pure-function
    # tests above: test_dominance_max_predictors_override)
    nsp = _make_nsp(n_x=10, n_parcels=30)
    with pytest.raises(ValueError, match="computationally infeasible"):
        nsp.colocalize("dominance", verbose=False, dominance_max_predictors=5)
    nsp.colocalize("dominance", verbose=False, dominance_max_predictors=15)
    assert nsp.get_colocalizations("dominance") is not None


def test_colocalize_pls_small_predictors_no_warning(caplog):
    nsp = _make_nsp(n_x=3, n_parcels=100)
    with caplog.at_level(logging.WARNING):
        nsp.colocalize("pls", verbose=False)
    assert "overfit" not in caplog.text
