"""Unit tests for grouping reference maps by a BIDS-style map-ID entity
(fetch_reference(set_by=...) -> datasets._sets_from_entity)."""

import pytest

from nispace.datasets import _sets_from_entity, _load_collection

MAPS = [
    "target-D23_tracer-raclopride_n-156_dx-hc_pub-malen2022",
    "target-5HT1a_tracer-way100635_n-35_dx-hc_pub-savli2012",
    "target-D23_tracer-flb457_n-55_dx-hc_pub-sandiego2015",
    "target-5HT1a_tracer-cumi101_n-8_dx-hc_pub-beliveau2017",
]


def test_set_by_target_sets_and_order():
    df = _sets_from_entity(MAPS, "target")
    assert list(df.columns) == ["set", "map"]
    assert sorted(df["map"]) == sorted(MAPS)
    assert dict(zip(df["map"], df["set"]))[MAPS[0]] == "D23"
    # sets are contiguous, maps keep their input order within a set
    assert df["set"].tolist() == ["5HT1a", "5HT1a", "D23", "D23"]
    assert df["map"].tolist() == [MAPS[1], MAPS[3], MAPS[0], MAPS[2]]


def test_set_by_is_case_insensitive_and_keeps_value_case():
    df = _sets_from_entity(MAPS, "Tracer")
    assert set(df["set"]) == {"raclopride", "way100635", "flb457", "cumi101"}


def test_set_by_output_is_a_valid_collection():
    df = _sets_from_entity(MAPS, "target")
    pd_coll = _load_collection(df)
    assert list(pd_coll.columns) == ["set", "map"]
    assert pd_coll.equals(df)


def test_set_by_missing_entity_raises():
    with pytest.raises(ValueError, match="entity not found"):
        _sets_from_entity(MAPS + ["nw-Auditory_pub-dworetsky2021"], "target")
