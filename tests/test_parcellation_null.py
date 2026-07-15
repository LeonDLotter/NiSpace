"""Tests for Parcellation.get_null_space() and nulls_random().

Uses object.__new__ to build minimal mock Parcellation instances that
carry only the attributes get_null_space() reads, with no data-repo access.
"""

import numpy as np
import pytest

from nispace.core.parcellation import Parcellation
from nispace.core.constants import (
    _NULL_DEFAULT_COMBINED,
    _NULL_DEFAULT_CX_SURF,
    _NULL_DEFAULT_CX_VOL,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class _MockParc(Parcellation):
    """Minimal Parcellation subclass for testing get_null_space().

    Sets only the attributes that get_null_space() reads:
    - self._images  (dict keyed by space name → spaces property reads from this)
    - self._is_combined
    - self._bilateral
    No super().__init__() call — avoids data-repo access entirely.
    """
    def __init__(self, spaces, is_combined, bilateral):
        self._images = {s: None for s in spaces}
        self._is_combined = is_combined
        self._bilateral = bilateral


def _mock_parc(spaces, is_combined, bilateral):
    return _MockParc(spaces, is_combined, bilateral)


# ---------------------------------------------------------------------------
# get_null_space: combined parcellations
# ---------------------------------------------------------------------------

def test_combined_tuple_default_returns_nested_tuple():
    # _NULL_DEFAULT_COMBINED is ("moran","moran") — should give nested tuple
    p = _mock_parc(["MNI152NLin6Asym"], is_combined=True, bilateral=False)
    result = p.get_null_space()
    if isinstance(_NULL_DEFAULT_COMBINED, tuple):
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], tuple) and isinstance(result[1], tuple)
        cx_space, cx_m = result[0]
        sc_space, sc_m = result[1]
        assert cx_space == "MNI152NLin6Asym"
        assert sc_space == "MNI152NLin6Asym"
        assert cx_m == _NULL_DEFAULT_COMBINED[0]
        assert sc_m == _NULL_DEFAULT_COMBINED[1]
    else:
        # string default — flat tuple
        space, method = result
        assert space == "MNI152NLin6Asym"
        assert method == _NULL_DEFAULT_COMBINED


def test_combined_string_default_returns_flat_tuple(monkeypatch):
    import nispace.core.parcellation as parc_mod
    monkeypatch.setattr(parc_mod, "_NULL_DEFAULT_COMBINED", "moran")
    p = _mock_parc(["MNI152NLin6Asym"], is_combined=True, bilateral=False)
    result = p.get_null_space()
    assert not isinstance(result[0], tuple)
    space, method = result
    assert space == "MNI152NLin6Asym"
    assert method == "moran"


def test_combined_mni_space_priority():
    # When multiple MNI spaces present, MNI152NLin6Asym should win
    p = _mock_parc(
        ["MNI152NLin2009cAsym", "MNI152NLin6Asym"],
        is_combined=True, bilateral=False,
    )
    result = p.get_null_space()
    chosen = result[0][0] if isinstance(result[0], tuple) else result[0]
    assert chosen == "MNI152NLin6Asym"


def test_combined_fallback_to_mni2009(monkeypatch):
    import nispace.core.parcellation as parc_mod
    monkeypatch.setattr(parc_mod, "_NULL_DEFAULT_COMBINED", "moran")
    p = _mock_parc(["MNI152NLin2009cAsym"], is_combined=True, bilateral=False)
    space, _ = p.get_null_space()
    assert space == "MNI152NLin2009cAsym"


# ---------------------------------------------------------------------------
# get_null_space: cortex-only with surface space
# ---------------------------------------------------------------------------

def test_cxonly_fslr_returned():
    p = _mock_parc(["fsLR"], is_combined=False, bilateral=False)
    space, method = p.get_null_space()
    assert space == "fsLR"
    assert method == _NULL_DEFAULT_CX_SURF


def test_cxonly_fsaverage_returned_when_no_fslr():
    p = _mock_parc(["fsaverage"], is_combined=False, bilateral=False)
    space, method = p.get_null_space()
    assert space == "fsaverage"
    assert method == _NULL_DEFAULT_CX_SURF


def test_cxonly_fslr_preferred_over_fsaverage():
    p = _mock_parc(["fsaverage", "fsLR"], is_combined=False, bilateral=False)
    space, method = p.get_null_space()
    assert space == "fsLR"


def test_cxonly_generic_surface_space_used():
    # A surface space with non-standard name that still matches "fsa" / "fslr"
    p = _mock_parc(["fsaverage5"], is_combined=False, bilateral=False)
    space, method = p.get_null_space()
    assert "fsa" in space.lower()
    assert method == _NULL_DEFAULT_CX_SURF


# ---------------------------------------------------------------------------
# get_null_space: MNI-only / bilateral
# ---------------------------------------------------------------------------

def test_cxonly_mni_only_returns_vol_method():
    p = _mock_parc(["MNI152NLin6Asym"], is_combined=False, bilateral=False)
    space, method = p.get_null_space()
    assert space == "MNI152NLin6Asym"
    assert method == _NULL_DEFAULT_CX_VOL


def test_bilateral_with_surface_takes_mni_path():
    # bilateral=True forces MNI path even if a surface space is present
    p = _mock_parc(["fsLR", "MNI152NLin6Asym"], is_combined=False, bilateral=True)
    space, method = p.get_null_space()
    assert space == "MNI152NLin6Asym"
    assert method == _NULL_DEFAULT_CX_VOL


def test_mni_space_priority_order():
    # MNI152NLin6Asym > MNI152NLin2009cAsym
    p = _mock_parc(
        ["MNI152NLin2009cAsym", "MNI152NLin6Asym"],
        is_combined=False, bilateral=False,
    )
    space, _ = p.get_null_space()
    assert space == "MNI152NLin6Asym"


def test_mni_fallback_to_first_mni():
    p = _mock_parc(["CustomMNI"], is_combined=False, bilateral=False)
    space, method = p.get_null_space()
    assert "mni" in space.lower()
    assert method == _NULL_DEFAULT_CX_VOL


# ---------------------------------------------------------------------------
# default_null_method property (delegates to get_null_space)
# ---------------------------------------------------------------------------

def test_default_null_method_matches_get_null_space():
    p = _mock_parc(["fsLR"], is_combined=False, bilateral=False)
    result = p.get_null_space()
    prop = p.default_null_method
    # property returns the method component(s) only
    if isinstance(result[0], tuple):
        assert prop == (result[0][1], result[1][1])
    else:
        assert prop == result[1]


