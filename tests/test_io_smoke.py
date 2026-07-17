"""Smoke tests for io.py's plain-object load/save helpers -- JSON, label/
distance-matrix/spin-matrix loaders, and the raw to_pickle()/from_pickle()
(pkl/pkl.gz/pkl.blosc) round-trip. NiSpace-object pickling itself is already
covered via NiSpace.to_pickle()/from_pickle() in test_api.py; this file
exercises io.py's lower-level functions directly, with plain objects.
"""

import numpy as np
import pandas as pd
import pytest

from nispace.io import read_json, write_json, load_labels, load_distmat, to_pickle, from_pickle


# ── read_json / write_json ───────────────────────────────────────────────

def test_write_then_read_json_roundtrip(tmp_path):
    d = {"a": 1, "b": [1, 2, 3]}
    f = tmp_path / "test.json"
    write_json(d, str(f))
    assert read_json(str(f)) == d


def test_read_json_passes_through_dict_like():
    assert read_json({"x": 1}) == {"x": 1}


def test_read_json_bad_input_raises_valueerror():
    """Regression test: read_json()'s fallback branch used to catch only
    ValueError from `dict(json_path)`, but that conversion typically raises
    TypeError (e.g. for a plain int) -- so most bad inputs crashed
    uncaught. And on the rare input that *did* raise ValueError, the except
    block only printed a message and fell through to `return json_dict`,
    which was never assigned in that branch -- an UnboundLocalError on top.
    Fixed to catch (TypeError, ValueError) and raise a clear ValueError via
    lgr.critical_raise instead of printing-then-crashing."""
    with pytest.raises(ValueError):
        read_json(5)
    with pytest.raises(ValueError):
        read_json([1, 2, 3])


def test_write_json_bad_path_raises_valueerror():
    """Regression test: write_json()'s fallback branch used to just print a
    warning and then `return json_path` unchanged for a non-path-like
    `json_path` (e.g. write_json(d, 5) silently "succeeded", returning 5,
    with no file written and no exception). Fixed to raise ValueError."""
    with pytest.raises(ValueError):
        write_json({"a": 1}, 5)


# ── load_labels ───────────────────────────────────────────────────────────

def test_load_labels_from_list_passthrough():
    assert load_labels(["a", "b", "c"]) == ["a", "b", "c"]


def test_load_labels_from_csv_file(tmp_path):
    f = tmp_path / "labels.csv"
    f.write_text("region1\nregion2\nregion3\n")
    assert load_labels(str(f)) == ["region1", "region2", "region3"]


def test_load_labels_tuple_concat_vs_separate():
    lh, rh = ["a", "b"], ["c", "d"]
    concatenated = load_labels((lh, rh), concat=True)
    assert concatenated == ["a", "b", "c", "d"]
    separate = load_labels((lh, rh), concat=False)
    assert separate == (["a", "b"], ["c", "d"])


def test_load_labels_invalid_type_raises():
    with pytest.raises(ValueError):
        load_labels(5)


# ── load_distmat ──────────────────────────────────────────────────────────

def test_load_distmat_none_passes_through():
    assert load_distmat(None) is None


def test_load_distmat_from_array_passthrough():
    arr = np.array([[0.0, 1.0], [1.0, 0.0]])
    out = load_distmat(arr)
    np.testing.assert_array_equal(out, arr)


def test_load_distmat_from_headerless_csv(tmp_path):
    arr = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    f = tmp_path / "dm.csv"
    pd.DataFrame(arr).to_csv(str(f), header=False, index=False)
    out = load_distmat(str(f))
    np.testing.assert_allclose(out, arr)


def test_load_distmat_tuple_of_two():
    a = np.array([[0.0, 1.0], [1.0, 0.0]])
    b = np.array([[0.0, 2.0], [2.0, 0.0]])
    out_a, out_b = load_distmat((a, b))
    np.testing.assert_array_equal(out_a, a)
    np.testing.assert_array_equal(out_b, b)


# ── to_pickle / from_pickle (raw io.py functions, plain objects) ─────────

@pytest.mark.parametrize("suffix", [".pkl", ".pkl.gz", ".pkl.blosc"])
def test_to_pickle_from_pickle_roundtrip(tmp_path, suffix):
    obj = {"a": np.arange(10), "b": pd.DataFrame({"x": [1, 2, 3]})}
    f = tmp_path / f"obj{suffix}"
    to_pickle(obj, str(f))
    loaded = from_pickle(str(f))
    np.testing.assert_array_equal(loaded["a"], obj["a"])
    pd.testing.assert_frame_equal(loaded["b"], obj["b"])


def test_to_pickle_unsupported_extension_raises(tmp_path):
    with pytest.raises(ValueError):
        to_pickle({"a": 1}, str(tmp_path / "obj.notapkl"))


def test_to_pickle_use_dill_roundtrips_lambda(tmp_path):
    obj = {"fn": lambda x: x + 1}
    f = tmp_path / "obj_dill.pkl"
    to_pickle(obj, str(f), use_dill=True)
    loaded = from_pickle(str(f), use_dill=True)
    assert loaded["fn"](1) == 2
