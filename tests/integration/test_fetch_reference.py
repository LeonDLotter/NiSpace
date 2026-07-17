"""Sanity checks for fetch_reference()/fetch_map_info()/fetch_collection()
against the real, public NiSpace-data mirror (see [[project_data_repo_ci]]).
Uses small integrated datasets ("rsn": 14 resting-state network maps) to
keep runtime low; not an exhaustive per-dataset sweep -- Batch 6 already
covers the plain unit-tier via mocked/local-only paths, this batch's job is
just to prove the real network/data-repo path still works end-to-end.
"""

import numpy as np
import pandas as pd
import pytest

from nispace.datasets import fetch_reference, fetch_map_info, fetch_metadata, fetch_collection


def test_fetch_reference_parcellated_returns_finite_dataframe():
    df = fetch_reference("rsn", parcellation="Yan100", print_references=False, verbose=False)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0 and df.shape[1] == 100
    assert np.isfinite(df.to_numpy()).all()


def test_fetch_reference_maps_filter_narrows_result():
    df_all = fetch_reference("rsn", parcellation="Yan100", print_references=False, verbose=False)
    one_map = df_all.index[0]
    df_filtered = fetch_reference("rsn", maps=one_map, parcellation="Yan100",
                                  print_references=False, verbose=False)
    assert list(df_filtered.index) == [one_map]


def test_fetch_reference_maps_filter_no_match_raises():
    """Regression test: fetch_reference(maps=...) used to silently return an
    empty DataFrame when the substring filter matched nothing, unlike the
    sibling sets= filter which already raised a clear ValueError for the
    same no-match situation. Homogenized to raise here too."""
    with pytest.raises(ValueError):
        fetch_reference("rsn", maps="not_a_real_map_xyz", parcellation="Yan100",
                        print_references=False, verbose=False)


def test_fetch_map_info_returns_metadata_table():
    meta = fetch_map_info("pet", verbose=False)
    assert isinstance(meta, pd.DataFrame)
    assert meta.shape[0] > 0
    assert "tracer" in meta.columns


def test_fetch_metadata_deprecated_alias_forwards_verbose():
    meta = fetch_metadata("pet", verbose=False)
    assert isinstance(meta, pd.DataFrame)
    assert meta.shape[0] > 0


def test_fetch_collection_all_returns_map_table():
    coll = fetch_collection("All", dataset="mrna", verbose=False)
    assert isinstance(coll, pd.DataFrame)
    assert "map" in coll.columns
    assert coll.shape[0] > 0
