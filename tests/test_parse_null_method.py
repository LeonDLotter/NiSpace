"""Tests for _parse_null_method() and _NULL_METHOD_ALIASES in nulls.py."""

import pytest
from nispace.nulls import _parse_null_method, _NULL_METHOD_ALIASES


# ---------------------------------------------------------------------------
# Alias canonicalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alias,canonical", [
    ("spin",       "cornblath"),
    ("msr",        "moran"),
    ("brainspace", "moran"),
    ("variomsr",   "variomoran"),
    ("brainsmash", "burt2020"),
    ("variogram",  "burt2020"),
])
def test_alias_resolves_to_canonical(alias, canonical):
    cx, sc = _parse_null_method(alias)
    assert cx == canonical
    assert sc is None


def test_canonical_names_pass_through_unchanged():
    for name in ("moran", "variomoran", "burt2020", "burt2018", "cornblath", "random"):
        cx, sc = _parse_null_method(name)
        assert cx == name
        assert sc is None


def test_unknown_string_passes_through():
    # Forward compat: unrecognised names are not an error at parse time
    cx, sc = _parse_null_method("some_future_method")
    assert cx == "some_future_method"
    assert sc is None


# ---------------------------------------------------------------------------
# "+" string splitting
# ---------------------------------------------------------------------------

def test_plus_string_splits_and_canonicalizes_both():
    cx, sc = _parse_null_method("spin+moran")
    assert cx == "cornblath"
    assert sc == "moran"


def test_plus_string_no_alias_needed():
    cx, sc = _parse_null_method("moran+burt2020")
    assert cx == "moran"
    assert sc == "burt2020"


def test_plus_string_both_sides_aliased():
    cx, sc = _parse_null_method("msr+variomsr")
    assert cx == "moran"
    assert sc == "variomoran"


def test_plus_string_only_splits_on_first_plus():
    # "a+b+c" → cx="a", sc="b+c" (split at most 1)
    cx, sc = _parse_null_method("moran+burt2020+extra")
    assert cx == "moran"
    assert sc == "burt2020+extra"


# ---------------------------------------------------------------------------
# Tuple input
# ---------------------------------------------------------------------------

def test_tuple_passthrough_without_aliases():
    result = _parse_null_method(("moran", "moran"))
    assert result == ("moran", "moran")


def test_tuple_canonicalizes_both_sides():
    cx, sc = _parse_null_method(("msr", "variomsr"))
    assert cx == "moran"
    assert sc == "variomoran"


def test_tuple_with_none_sc():
    cx, sc = _parse_null_method(("moran", None))
    assert cx == "moran"
    assert sc is None


def test_tuple_with_none_cx():
    cx, sc = _parse_null_method((None, "moran"))
    assert cx is None
    assert sc == "moran"


# ---------------------------------------------------------------------------
# Return type consistency
# ---------------------------------------------------------------------------

def test_always_returns_two_tuple():
    for inp in ("moran", "spin+moran", ("moran", "cornblath")):
        result = _parse_null_method(inp)
        assert isinstance(result, tuple)
        assert len(result) == 2
