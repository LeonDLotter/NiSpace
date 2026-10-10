"""Tests for Parcellation.get_image_for_dataspace() -- which registered image
NiSpace.fit() activates for a requested space.

Regression: MNI152NLin6Asym and MNI152NLin2009cAsym both normalize to "mni",
and the first registered match used to win. All integrated parcellations list
MNI152NLin2009cAsym first, so 6Asym data was parcellated with the 2009c image.

Uses a minimal mock (no data-repo access), like test_parcellation_null.py.
"""

import pytest

from nispace.core.parcellation import Parcellation
from nispace.core.constants import _SPACE_DEFAULT_VOL

A6, A9 = "MNI152NLin6Asym", "MNI152NLin2009cAsym"
LIB_ORDER = [A9, A6, "fsLR", "fsaverage"]  # order used by the integrated library


class _MockParc(Parcellation):
    """Sets only what get_image_for_dataspace() reads; no super().__init__()."""
    def __init__(self, spaces, name="mock"):
        self._images = {s: None for s in spaces}
        self._name = name


def _warned(caplog):
    return [r for r in caplog.records if r.levelname == "WARNING" and "not available" in r.message]


@pytest.mark.parametrize("spaces", [LIB_ORDER, [A6, A9, "fsLR", "fsaverage"]])
@pytest.mark.parametrize("query", [A6, A9, "fsLR", "fsaverage"])
def test_exact_space_selected_regardless_of_order(spaces, query, caplog):
    with caplog.at_level("WARNING"):
        assert _MockParc(spaces).get_image_for_dataspace(query) == query
    assert not _warned(caplog)


@pytest.mark.parametrize("query,expected", [
    ("mni152nlin6asym", A6),
    ("MNI152NLin2009cAsym", A9),
    ("fsa", "fsaverage"),
    ("fs_lr", "fsLR"),
    ("fslr", "fsLR"),
])
def test_case_and_alias_match(query, expected):
    assert _MockParc(LIB_ORDER).get_image_for_dataspace(query) == expected


@pytest.mark.parametrize("query", ["MNI", "MNI152", "mni152"])
def test_generic_mni_prefers_default_space(query, caplog):
    with caplog.at_level("WARNING"):
        assert _MockParc(LIB_ORDER).get_image_for_dataspace(query) == _SPACE_DEFAULT_VOL
    assert not _warned(caplog)


def test_none_prefers_default_space():
    assert _MockParc(LIB_ORDER).get_image_for_dataspace(None) == _SPACE_DEFAULT_VOL
    assert _MockParc(["fsLR", "fsaverage"]).get_image_for_dataspace(None) == "fsLR"


def test_unavailable_specific_template_falls_back_with_warning(caplog):
    with caplog.at_level("WARNING"):
        assert _MockParc([A9]).get_image_for_dataspace(A6) == A9
    assert _warned(caplog)


def test_unregistered_template_variant_warns(caplog):
    with caplog.at_level("WARNING"):
        assert _MockParc(LIB_ORDER).get_image_for_dataspace("MNI152NLin2009aAsym") == A6
    assert _warned(caplog)


def test_generic_registered_space_accepts_specific_query_silently(caplog):
    # custom parcellation registered as plain "MNI152": nothing more specific to pick
    with caplog.at_level("WARNING"):
        assert _MockParc(["MNI152"]).get_image_for_dataspace(A6) == "MNI152"
    assert not _warned(caplog)


@pytest.mark.parametrize("spaces,query", [
    ([A9, A6], "fsLR"),
    (["fsLR"], "fsaverage"),
    (["fsaverage"], A6),
])
def test_no_compatible_space_raises(spaces, query):
    with pytest.raises(ValueError):
        _MockParc(spaces).get_image_for_dataspace(query)
