"""Tests for core.colocalize._rank_regress -- the shared rank/Z-regression
transform used by colocalize()/regional_influence()/regional_contribution().

Regression test for a real bug: when both rank=True and regress=True apply
(e.g. partialspearman with Z), the rank2d() result was silently discarded --
regress_z_fun was called on the original raw array instead of the ranked
one, making partialspearman silently compute exactly what partialpearson
computes. Fixed by chaining consistently through `arr_out`. The bug was
specific to the plain-ndarray and plain-dict branches (cases 1 and 3) --
the list branches (case 2, used for null maps) already chained correctly
by reassigning the same variable name.
"""

import numpy as np
import pytest

from nispace.core.colocalize import _rank_regress
from nispace.stats.coloc import rank2d
from nispace.stats.misc import residuals_nan


def test_no_rank_no_regress_returns_input_unchanged(rng):
    arr = rng.normal(size=(2, 30))
    out = _rank_regress(arr=arr, rank=False, regress=False, verbose=False)
    assert out is arr


def test_rank_only_ndarray(rng):
    arr = rng.normal(size=(2, 30))
    out = _rank_regress(arr=arr.copy(), rank=True, regress=False, verbose=False)
    np.testing.assert_allclose(out, rank2d(arr.T).T)


def test_regress_only_ndarray_single_z(rng):
    arr = rng.normal(size=(2, 30))
    z = rng.normal(size=(1, 30))
    out = _rank_regress(arr=arr.copy(), rank=False, regress=True, z=z.copy(),
                        zy_matched=False, verbose=False)
    expected = np.row_stack([residuals_nan(x=z[0], y=arr[i]) for i in range(arr.shape[0])])
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_regress_only_ndarray_multi_z_not_matched(rng):
    """z.shape[0] > 1, zy_matched=False -> all of z used as multi-predictor
    for every arr row (the `residuals_nan(x=z.T, ...)` branch)."""
    arr = rng.normal(size=(2, 30))
    z = rng.normal(size=(3, 30))
    out = _rank_regress(arr=arr.copy(), rank=False, regress=True, z=z.copy(),
                        zy_matched=False, verbose=False)
    expected = np.row_stack([residuals_nan(x=z.T, y=arr[i]) for i in range(arr.shape[0])])
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_regress_zy_matched(rng):
    """zy_matched=True -> z[i] matched one-to-one with arr[i] (subject-matched Z)."""
    arr = rng.normal(size=(3, 30))
    z = rng.normal(size=(3, 30))
    out = _rank_regress(arr=arr.copy(), rank=False, regress=True, z=z.copy(),
                        zy_matched=True, verbose=False)
    expected = np.row_stack([residuals_nan(x=z[i], y=arr[i]) for i in range(arr.shape[0])])
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_rank_and_regress_chains_correctly_ndarray(rng):
    """The regression test: rank must be applied BEFORE regress, not silently
    discarded (this is exactly the partialspearman-with-Z bug)."""
    arr = rng.normal(size=(2, 30))
    z = rng.normal(size=(1, 30))
    out = _rank_regress(arr=arr.copy(), rank=True, regress=True, z=z.copy(),
                        zy_matched=False, verbose=False)

    ranked = rank2d(arr.T).T
    expected_fixed = np.row_stack([residuals_nan(x=z[0], y=ranked[i]) for i in range(arr.shape[0])])
    expected_buggy = np.row_stack([residuals_nan(x=z[0], y=arr[i]) for i in range(arr.shape[0])])

    np.testing.assert_allclose(out, expected_fixed, equal_nan=True)
    assert not np.allclose(out, expected_buggy, equal_nan=True)


def test_rank_and_regress_chains_correctly_dict(rng):
    """Same bug, dict branch (XSEA case -- case 3 in _rank_regress)."""
    arr = {"setA": rng.normal(size=(3, 30)), "setB": rng.normal(size=(2, 30))}
    z = rng.normal(size=(1, 30))
    out = _rank_regress(arr={k: v.copy() for k, v in arr.items()}, rank=True, regress=True,
                        z=z.copy(), zy_matched=False, verbose=False)

    for set_name, set_arr in arr.items():
        ranked = rank2d(set_arr.T).T
        expected_fixed = np.row_stack(
            [residuals_nan(x=z[0], y=ranked[i]) for i in range(set_arr.shape[0])])
        expected_buggy = np.row_stack(
            [residuals_nan(x=z[0], y=set_arr[i]) for i in range(set_arr.shape[0])])
        np.testing.assert_allclose(out[set_name], expected_fixed, equal_nan=True)
        assert not np.allclose(out[set_name], expected_buggy, equal_nan=True)


def test_rank_and_regress_correct_list_of_arrays(rng):
    """The list branch (case 2, used for null maps) already chained correctly
    via variable reassignment -- lock this in so it can't regress either."""
    arrs = [rng.normal(size=(2, 30)) for _ in range(3)]
    z = rng.normal(size=(1, 30))
    out = _rank_regress(arr=[a.copy() for a in arrs], rank=True, regress=True,
                        z=z.copy(), zy_matched=False, n_proc=1, verbose=False)

    for arr, result in zip(arrs, out):
        ranked = rank2d(arr.T).T
        expected_fixed = np.row_stack([residuals_nan(x=z[0], y=ranked[i]) for i in range(arr.shape[0])])
        np.testing.assert_allclose(result, expected_fixed, equal_nan=True)


def test_rank_and_regress_correct_list_of_dicts(rng):
    arrs = [{"setA": rng.normal(size=(3, 30))} for _ in range(2)]
    z = rng.normal(size=(1, 30))
    out = _rank_regress(arr=[{k: v.copy() for k, v in a.items()} for a in arrs], rank=True,
                        regress=True, z=z.copy(), zy_matched=False, n_proc=1, verbose=False)

    for arr, result in zip(arrs, out):
        set_arr = arr["setA"]
        ranked = rank2d(set_arr.T).T
        expected_fixed = np.row_stack(
            [residuals_nan(x=z[0], y=ranked[i]) for i in range(set_arr.shape[0])])
        np.testing.assert_allclose(result["setA"], expected_fixed, equal_nan=True)
