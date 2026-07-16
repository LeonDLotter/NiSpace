"""Tests for nispace.core.transform_y -- the group-encoding helpers and the
formula-DSL (Y_transform="cohen(a,b)" etc.) used by group_comparison().

All pure/offline: no parcellation, no network.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from nispace.core.transform_y import (
    _dummy_code_groups,
    _num_code_subjects,
    return_arr,
    mean0,
    median0,
    std0,
    var0,
    elem_diff,
    mean0_diff,
    center0,
    _normalize_formula,
    _args_to_tuple,
    _parse_transform_formula,
    _get_transform_fun,
    _FUN_MAP,
    _PAIRED_FORMULAS,
)


# ── _dummy_code_groups ──────────────────────────────────────────────────────

def test_dummy_code_groups_two_groups_sorted_mapping():
    # smallest/alphabetically-first unique element -> 0, second -> 1
    coded = _dummy_code_groups(["b", "b", "a", "a"])
    assert list(coded) == [1, 1, 0, 0]


def test_dummy_code_groups_single_group_all_zero():
    assert _dummy_code_groups(["x", "x", "x"]) == [0, 0, 0]


def test_dummy_code_groups_more_than_two_raises():
    with pytest.raises(ValueError, match="more than two"):
        _dummy_code_groups(["a", "b", "c"])


def test_dummy_code_groups_nan_raises():
    with pytest.raises(ValueError, match="nan"):
        _dummy_code_groups(["a", None, "b"])


# ── _num_code_subjects ───────────────────────────────────────────────────────

def test_num_code_subjects_valid_pairs():
    coded = _num_code_subjects(["s1", "s1", "s2", "s2"])
    # each subject's two rows get the same code; s1 != s2 code
    assert coded[0] == coded[1]
    assert coded[2] == coded[3]
    assert coded[0] != coded[2]


def test_num_code_subjects_too_many_raises():
    with pytest.raises(ValueError, match="more than two"):
        _num_code_subjects(["s1", "s1", "s1"])


def test_num_code_subjects_too_few_raises():
    with pytest.raises(ValueError, match="less than two"):
        _num_code_subjects(["s1", "s2", "s2"])


def test_num_code_subjects_nan_raises():
    with pytest.raises(ValueError, match="nan"):
        _num_code_subjects(["s1", "s1", None, None])


# ── formula-DSL primitives ───────────────────────────────────────────────────

def test_return_arr_identity():
    x = np.array([1.0, 2.0, np.nan])
    assert np.array_equal(return_arr(x), x, equal_nan=True)


def test_mean0_median0_are_nan_aware():
    x = np.array([[1.0, np.nan], [3.0, 4.0]])
    assert np.allclose(mean0(x), [2.0, 4.0])
    assert np.allclose(median0(x), [2.0, 4.0])


def test_std0_var0_use_sample_ddof1():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert np.allclose(std0(x), np.nanstd(x, axis=0, ddof=1))
    assert np.allclose(var0(x), np.nanvar(x, axis=0, ddof=1))


def test_elem_diff():
    a, b = np.array([5.0, 3.0]), np.array([2.0, 1.0])
    assert np.allclose(elem_diff(a, b), [3.0, 2.0])


def test_mean0_diff():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[0.0, 0.0], [2.0, 2.0]])
    assert np.allclose(mean0_diff(a, b), mean0(a) - mean0(b))


def test_center0_default_centers_on_self():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(center0(a), a - mean0(a))


def test_center0_with_b_centers_on_b():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    assert np.allclose(center0(a, b), a - mean0(b))


# ── _normalize_formula ───────────────────────────────────────────────────────

@pytest.mark.parametrize("formula,expected_formula,expected_wildcard", [
    ("A - B", "elemdiff(a,b)", "elemdiff(*,*)"),
    ("mean(A)-mean(B)", "meandiff(a,b)", "meandiff(*,*)"),
    ("a - mean(b)", "center(a,b)", "center(*,*)"),
    ("mean(y)", "mean(y)", "mean(*)"),
    ("cohen(a,b)", "cohen(a,b)", "cohen(*,*)"),
])
def test_normalize_formula(formula, expected_formula, expected_wildcard):
    out_formula, out_wildcard = _normalize_formula(formula)
    assert out_formula == expected_formula
    assert out_wildcard == expected_wildcard


# ── _args_to_tuple ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("expr,expected", [
    ("cohen(a,b)", ("a", "b")),
    ("mean(a)", ("a", None)),
    ("y", ("y", None)),
    ("xyz", ("unrecognized", None)),
])
def test_args_to_tuple(expr, expected):
    assert _args_to_tuple(expr) == expected


# ── _parse_transform_formula ─────────────────────────────────────────────────

def test_parse_transform_formula_valid():
    formula, wildcard, fun, args, paired = _parse_transform_formula("a-b")
    assert wildcard == "elemdiff(*,*)"
    assert fun is elem_diff
    assert args == ["a", "b"]
    assert paired is True  # elemdiff(*,*) is in _PAIRED_FORMULAS


def test_parse_transform_formula_cohen_is_not_paired():
    # only "pairedcohen", not plain "cohen", is in _PAIRED_FORMULAS
    _, wildcard, _, _, paired = _parse_transform_formula("cohen(a,b)")
    assert wildcard == "cohen(*,*)"
    assert wildcard not in _PAIRED_FORMULAS
    assert paired is False


def test_parse_transform_formula_invalid_raises():
    with pytest.raises(ValueError, match="not allowed"):
        _parse_transform_formula("notreal(y)")


# ── _get_transform_fun (end-to-end DSL application) ──────────────────────────

@pytest.fixture
def toy_y(rng):
    return pd.DataFrame(
        rng.normal(size=(6, 4)),
        index=[f"s{i}" for i in range(6)],
        columns=[f"p{i}" for i in range(4)],
    )


@pytest.fixture
def toy_groups():
    return np.array([0, 0, 0, 1, 1, 1])


def test_get_transform_fun_mean_no_groups(toy_y):
    fun = _get_transform_fun("mean(y)")
    res = fun(y=toy_y)
    assert res.shape == (1, 4)
    assert res.index.tolist() == ["mean"]
    assert np.allclose(res.values, np.nanmean(toy_y.values, axis=0))


def test_get_transform_fun_elemdiff_with_groups(toy_y, toy_groups):
    fun = _get_transform_fun("a-b")
    res = fun(y=toy_y, groups=toy_groups)
    a = toy_y.values[toy_groups == 0]
    b = toy_y.values[toy_groups == 1]
    assert res.shape == (3, 4)
    assert res.index.tolist() == toy_y.index[toy_groups == 0].tolist()
    assert np.allclose(res.values, a - b)


def test_get_transform_fun_subjects_sorts_each_group_before_diff():
    y = pd.DataFrame(
        np.random.default_rng(1).normal(size=(4, 3)),
        index=["g1_s2", "g1_s1", "g2_s1", "g2_s2"],
        columns=["p0", "p1", "p2"],
    )
    groups = np.array([0, 0, 1, 1])
    subjects = np.array(["s2", "s1", "s1", "s2"])
    fun = _get_transform_fun("a-b")
    res = fun(y=y, groups=groups, subjects=subjects)
    assert res.index.tolist() == ["g1_s1", "g1_s2"]
    expected = y.loc[["g1_s1", "g1_s2"]].values - y.loc[["g2_s1", "g2_s2"]].values
    assert np.allclose(res.values, expected)


def test_get_transform_fun_return_df_false_gives_raw_array(toy_y):
    fun = _get_transform_fun("mean(y)", return_df=False)
    res = fun(y=toy_y)
    assert isinstance(res, np.ndarray)
    assert res.shape == (1, 4)


def test_get_transform_fun_return_paired_reports_flag(toy_y, toy_groups):
    fun, paired = _get_transform_fun("a-b", return_paired=True)
    assert paired is True
    fun2, paired2 = _get_transform_fun("cohen(a,b)", return_paired=True)
    assert paired2 is False


def test_get_transform_fun_raises_on_missing_y():
    fun = _get_transform_fun("mean(y)")
    with pytest.raises(ValueError, match="y must not be None"):
        fun(y=None)


def test_get_transform_fun_ignore_nan_warnings_suppresses_empty_slice_warning():
    # groups=[0, 0] leaves group "b" (groups==1) empty -> mean0 over axis 0
    # hits numpy's "Mean of empty slice" RuntimeWarning unless suppressed.
    y = pd.DataFrame(np.random.default_rng(0).normal(size=(2, 2)),
                      index=["s0", "s1"], columns=["p0", "p1"])
    groups = np.array([0, 0])

    fun_loud = _get_transform_fun("a-mean(b)", ignore_nan_warnings=False)
    with warnings.catch_warnings(record=True) as rec_loud:
        warnings.simplefilter("always")
        fun_loud(y=y, groups=groups)
    assert any(issubclass(w.category, RuntimeWarning) for w in rec_loud)

    fun_quiet = _get_transform_fun("a-mean(b)", ignore_nan_warnings=True)
    with warnings.catch_warnings(record=True) as rec_quiet:
        warnings.simplefilter("always")
        fun_quiet(y=y, groups=groups)
    assert not any(issubclass(w.category, RuntimeWarning) for w in rec_quiet)
