"""Tests for the NullMaps container (core/nullmaps.py)."""

import numpy as np
import pytest
from nispace.core.nullmaps import NullMaps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(n_maps=3, n_perm=20, n_parcels=10, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_maps, n_perm, n_parcels)).astype(np.float32)
    labels = [f"map{i}" for i in range(n_maps)]
    return NullMaps(data, labels, **kwargs)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

def test_non_3d_array_raises():
    with pytest.raises(ValueError, match="3-D"):
        NullMaps(np.zeros((5, 10)), ["a"])


def test_2d_array_raises():
    with pytest.raises(ValueError, match="3-D"):
        NullMaps(np.zeros((3, 10)), ["a", "b", "c"])


def test_label_length_mismatch_raises():
    data = np.zeros((3, 10, 5))
    with pytest.raises(ValueError, match="must equal"):
        NullMaps(data, ["a", "b"])  # 2 labels, shape[0]=3


def test_non_array_type_raises():
    with pytest.raises(TypeError):
        NullMaps([[1, 2], [3, 4]], ["a"])


# ---------------------------------------------------------------------------
# Shape properties
# ---------------------------------------------------------------------------

def test_shape_properties():
    nm = _make(n_maps=3, n_perm=20, n_parcels=10)
    assert nm.n_maps == 3
    assert nm.n_perm == 20
    assert nm.n_parcels == 10
    assert nm.shape == (3, 20, 10)


def test_len_equals_n_maps():
    nm = _make(n_maps=4)
    assert len(nm) == 4


def test_labels_property():
    nm = _make(n_maps=2)
    assert nm.labels == ["map0", "map1"]


# ---------------------------------------------------------------------------
# Dict-like interface
# ---------------------------------------------------------------------------

def test_getitem_known_label_returns_2d_view():
    nm = _make(n_maps=3, n_perm=20, n_parcels=10)
    view = nm["map1"]
    assert view.shape == (20, 10)
    assert np.shares_memory(view, nm.data)


def test_getitem_unknown_label_raises_keyerror():
    nm = _make(n_maps=2)
    with pytest.raises(KeyError, match="not in NullMaps"):
        _ = nm["nonexistent"]


def test_contains_present_and_absent():
    nm = _make(n_maps=2)
    assert "map0" in nm
    assert "map1" in nm
    assert "map99" not in nm


def test_keys_iteration():
    nm = _make(n_maps=3)
    assert list(nm.keys()) == ["map0", "map1", "map2"]


def test_items_gives_label_array_pairs():
    nm = _make(n_maps=2, n_perm=5, n_parcels=4)
    pairs = list(nm.items())
    assert len(pairs) == 2
    assert pairs[0][0] == "map0"
    assert pairs[0][1].shape == (5, 4)


# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------

def test_from_dict_round_trip():
    rng = np.random.default_rng(7)
    d = {
        "alpha": rng.standard_normal((15, 8)).astype(np.float32),
        "beta":  rng.standard_normal((15, 8)).astype(np.float32),
        "gamma": rng.standard_normal((15, 8)).astype(np.float32),
    }
    nm = NullMaps.from_dict(d, null_method="moran")
    assert nm.shape == (3, 15, 8)
    assert nm.labels == ["alpha", "beta", "gamma"]
    assert nm.null_method == "moran"
    np.testing.assert_array_equal(nm["alpha"], d["alpha"])
    np.testing.assert_array_equal(nm["beta"],  d["beta"])


def test_from_dict_single_entry():
    d = {"only": np.ones((10, 5), dtype=np.float32)}
    nm = NullMaps.from_dict(d)
    assert nm.shape == (1, 10, 5)
    assert nm.labels == ["only"]


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def test_merge_concatenates_along_map_axis():
    nm1 = _make(n_maps=2, n_perm=10, n_parcels=5, seed=1)
    nm2 = _make(n_maps=3, n_perm=10, n_parcels=5, seed=2)
    labels2 = [f"x{i}" for i in range(3)]
    nm2 = NullMaps(nm2.data, labels2)
    merged = NullMaps.merge(nm1, nm2)
    assert merged.shape == (5, 10, 5)
    assert merged.labels == ["map0", "map1", "x0", "x1", "x2"]


def test_merge_data_matches_inputs():
    nm1 = _make(n_maps=2, n_perm=10, n_parcels=5, seed=1)
    nm2 = NullMaps(
        np.random.default_rng(2).standard_normal((2, 10, 5)).astype(np.float32),
        ["a", "b"],
    )
    merged = NullMaps.merge(nm1, nm2)
    np.testing.assert_array_equal(merged.data[:2], nm1.data)
    np.testing.assert_array_equal(merged.data[2:], nm2.data)


def test_merge_raises_on_nperm_mismatch():
    nm1 = _make(n_maps=2, n_perm=10, n_parcels=5, seed=1)
    nm2 = NullMaps(
        np.random.default_rng(2).standard_normal((2, 20, 5)).astype(np.float32),
        ["a", "b"],
    )
    with pytest.raises(ValueError, match="n_perm"):
        NullMaps.merge(nm1, nm2)


def test_merge_raises_on_nparcels_mismatch():
    nm1 = _make(n_maps=2, n_perm=10, n_parcels=5, seed=1)
    nm2 = NullMaps(
        np.random.default_rng(2).standard_normal((2, 10, 8)).astype(np.float32),
        ["a", "b"],
    )
    with pytest.raises(ValueError):
        NullMaps.merge(nm1, nm2)


# ---------------------------------------------------------------------------
# subset
# ---------------------------------------------------------------------------

def test_subset_returns_requested_labels_in_order():
    nm = _make(n_maps=4, n_perm=10, n_parcels=5)
    sub = nm.subset(["map3", "map1"])
    assert sub.labels == ["map3", "map1"]
    assert sub.shape == (2, 10, 5)
    np.testing.assert_array_equal(sub.data[0], nm["map3"])
    np.testing.assert_array_equal(sub.data[1], nm["map1"])


def test_subset_unknown_label_raises():
    nm = _make(n_maps=2)
    with pytest.raises(KeyError):
        nm.subset(["map0", "nonexistent"])


# ---------------------------------------------------------------------------
# standardize
# ---------------------------------------------------------------------------

def test_standardize_zero_mean_per_permutation():
    nm = _make(n_maps=3, n_perm=50, n_parcels=20)
    std = nm.standardize()
    # mean across parcel axis (axis=2 of the 3D array) should be ~0 for each (map, perm)
    means = std.data.mean(axis=2)
    np.testing.assert_allclose(means, 0.0, atol=1e-5)


def test_standardize_returns_new_nullmaps():
    nm = _make()
    std = nm.standardize()
    assert std is not nm
    assert isinstance(std, NullMaps)
    # original unchanged
    assert not np.allclose(nm.data.mean(axis=2), 0.0)


def test_standardize_preserves_shape_and_labels():
    nm = _make(n_maps=3, n_perm=20, n_parcels=10)
    std = nm.standardize()
    assert std.shape == nm.shape
    assert std.labels == nm.labels


# ---------------------------------------------------------------------------
# null_method_str
# ---------------------------------------------------------------------------

def test_null_method_str_none():
    nm = _make(null_method=None)
    assert nm.null_method_str == ""


def test_null_method_str_string():
    nm = _make(null_method="moran")
    assert nm.null_method_str == "moran"


def test_null_method_str_tuple():
    nm = _make(null_method=("moran", "cornblath"))
    assert nm.null_method_str == "moran+cornblath"


# ---------------------------------------------------------------------------
# dtype casting
# ---------------------------------------------------------------------------

def test_dtype_cast_on_construction():
    data = np.zeros((2, 5, 4), dtype=np.float64)
    nm = NullMaps(data, ["a", "b"], dtype=np.float32)
    assert nm.dtype == np.float32


def test_astype_returns_new_nullmaps_with_correct_dtype():
    nm = _make()
    nm64 = nm.astype(np.float64)
    assert nm64.dtype == np.float64
    assert nm.dtype == np.float32  # original unchanged


# ---------------------------------------------------------------------------
# warn_large: derived transforms (astype/standardize/subset) must not re-warn
# ---------------------------------------------------------------------------
#
# Regression: astype()/standardize()/subset() each build a fresh NullMaps
# internally but originally didn't forward warn_large -- so a caller building
# many small NullMaps with warn_large=False (e.g. row-batched generation,
# core/permute.py's _iter_null_map_batches) would still get the >1GB warning
# re-triggered on every single .standardize() call, once per batch. Fixed by
# hardcoding warn_large=False on all three (a transform of an
# already-constructed object shouldn't re-litigate a warning that already
# fired -- or was deliberately suppressed -- at the original construction).

import logging
from nispace.core.nullmaps import _NULLMAPS_WARN_BYTES


def _make_large(warn_large=True):
    # a (n_maps, n_perm, n_parcels) float32 array just over the 1GB threshold
    n_parcels = 50
    n_perm = 50
    n_maps = _NULLMAPS_WARN_BYTES // (n_perm * n_parcels * 4) + 10
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_maps, n_perm, n_parcels)).astype(np.float32)
    labels = [f"map{i}" for i in range(n_maps)]
    return NullMaps(data, labels, warn_large=warn_large)


def test_construction_warns_when_large(caplog):
    with caplog.at_level(logging.WARNING):
        _make_large(warn_large=True)
    assert "GB in memory" in caplog.text


def test_construction_silent_when_warn_large_false(caplog):
    with caplog.at_level(logging.WARNING):
        _make_large(warn_large=False)
    assert "GB in memory" not in caplog.text


def test_astype_does_not_rewarn(caplog):
    nm = _make_large(warn_large=False)
    with caplog.at_level(logging.WARNING):
        nm.astype(np.float32)
    assert "GB in memory" not in caplog.text


def test_standardize_does_not_rewarn(caplog):
    nm = _make_large(warn_large=False)
    with caplog.at_level(logging.WARNING):
        nm.standardize()
    assert "GB in memory" not in caplog.text


def test_subset_does_not_rewarn(caplog):
    nm = _make_large(warn_large=False)
    with caplog.at_level(logging.WARNING):
        nm.subset(nm.labels[:5])
    assert "GB in memory" not in caplog.text
