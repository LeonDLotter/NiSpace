"""Set-indexed X: a map may belong to several sets but only once per set (overlaps warned
where they matter: colocalize without xsea, reduce_x); set enrichment
scores sets correctly with overlaps; random sets are drawn from unique maps (default and
provided background); weightedmean uses the weight level."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from nispace import NiSpace

N_DK = 68


@pytest.fixture
def maps(rng):
    return pd.DataFrame(rng.normal(size=(6, N_DK)), index=[f"m{i}" for i in range(6)])


@pytest.fixture
def y(rng):
    return pd.DataFrame(rng.normal(size=(1, N_DK)), index=["y"])


def _set_x(maps, entries, weights=None):
    if weights is None:
        idx = pd.MultiIndex.from_tuples(entries, names=["set", "map"])
    else:
        idx = pd.MultiIndex.from_tuples([(s, m, w) for (s, m), w in zip(entries, weights)],
                                        names=["set", "map", "weight"])
    return pd.DataFrame(maps.loc[[m for _, m in entries]].values, index=idx)


def _fit(x, y):
    return NiSpace(x=x, y=y, parcellation="DesikanKilliany", verbose=False).fit()


def _z(y, maps, m):
    return np.arctanh(spearmanr(y.iloc[0], maps.loc[m]).statistic)


OVERLAP = [("A", "m0"), ("A", "m1"), ("B", "m0"), ("B", "m2")]


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


def test_within_set_duplicate_dropped_with_warning(maps, y, caplog):
    x = _set_x(maps, [("A", "m0"), ("A", "m1"), ("A", "m1"), ("B", "m2"), ("B", "m3")])
    with caplog.at_level("WARNING"):
        nsp = _fit(x, y)
    assert any("within the same set" in w for w in _warnings(caplog))
    assert nsp._X.shape[0] == 4
    nsp.colocalize("spearman", xsea=True)
    a = nsp.get_colocalizations("spearman", xsea=True).iloc[0]["A"]
    assert a == pytest.approx(np.mean([_z(y, maps, "m0"), _z(y, maps, "m1")]), abs=1e-5)


def test_map_in_several_sets_kept_no_warning_at_fit(maps, y, caplog):
    # fit can't know whether the object is used for set enrichment -> info only
    with caplog.at_level("WARNING"):
        nsp = _fit(_set_x(maps, OVERLAP), y)
    assert not any("several sets" in w for w in _warnings(caplog))
    assert nsp._X.shape[0] == 4


def test_colocalize_without_xsea_warns_on_overlapping_sets(maps, y, caplog):
    nsp = _fit(_set_x(maps, OVERLAP), y)
    with caplog.at_level("WARNING"):
        nsp.colocalize("spearman", xsea=False)
    assert any("colocalize(xsea=False)" in w and "several sets" in w for w in _warnings(caplog))


def test_colocalize_with_xsea_does_not_warn(maps, y, caplog):
    nsp = _fit(_set_x(maps, OVERLAP), y)
    with caplog.at_level("WARNING"):
        nsp.colocalize("spearman", xsea=True)
    assert not any("several sets" in w for w in _warnings(caplog))


@pytest.mark.parametrize("reduction,kw,warns", [("pca", {}, True), ("mean", {}, True),
                                                ("mean", {"mean_by_set": True}, False)])
def test_reduce_x_warns_unless_set_wise(maps, y, caplog, reduction, kw, warns):
    nsp = _fit(_set_x(maps, OVERLAP), y)
    with caplog.at_level("WARNING"):
        nsp.reduce_x(reduction, **kw)
    assert any("several sets" in w for w in _warnings(caplog)) is warns


def test_plain_duplicates_still_warned(maps, y, caplog):
    with caplog.at_level("WARNING"):
        _fit(maps.loc[["m0", "m0", "m1", "m2"]], y)
    assert any("duplicate row(s) found in 'X'" in w for w in _warnings(caplog))


def test_no_warning_for_disjoint_sets(maps, y, caplog):
    with caplog.at_level("WARNING"):
        nsp = _fit(_set_x(maps, [("A", "m0"), ("A", "m1"), ("B", "m2"), ("B", "m3")]), y)
        nsp.colocalize("spearman", xsea=False)
    assert not any("several sets" in w for w in _warnings(caplog))


def test_xsea_scores_with_overlapping_sets(maps, y):
    nsp = _fit(_set_x(maps, OVERLAP), y)
    nsp.colocalize("spearman", xsea=True)
    got = nsp.get_colocalizations("spearman", xsea=True).iloc[0]
    assert got["A"] == pytest.approx(np.mean([_z(y, maps, "m0"), _z(y, maps, "m1")]), abs=1e-5)
    assert got["B"] == pytest.approx(np.mean([_z(y, maps, "m0"), _z(y, maps, "m2")]), abs=1e-5)


def test_default_background_is_unique_maps(maps, y, caplog):
    nsp = _fit(_set_x(maps, OVERLAP), y)
    nsp.colocalize("spearman", xsea=True)
    with caplog.at_level("WARNING"):
        nsp.permute("sets", method="spearman", xsea=True, n_perm=20, seed=1)
    assert any("Will use 3 unique X maps as background" in w for w in _warnings(caplog))


def test_provided_background_deduplicated(maps, y, caplog):
    nsp = _fit(_set_x(maps, OVERLAP), y)
    nsp.colocalize("spearman", xsea=True)
    bg = maps.loc[["m0", "m0", "m1", "m2", "m3", "m4", "m5"]].values
    with caplog.at_level("WARNING"):
        nsp.permute("sets", method="spearman", xsea=True, n_perm=20, seed=1, sets_X_background=bg)
    assert any("1 duplicate map(s) dropped from the provided X background" in w
               for w in _warnings(caplog))


def test_weightedmean_uses_weight_level(maps, y):
    nsp = _fit(_set_x(maps, OVERLAP, weights=[3.0, 1.0, 1.0, 1.0]), y)
    nsp.colocalize("spearman", xsea=True, xsea_aggregation_method="weightedmean")
    a = nsp.get_colocalizations("spearman", xsea=True).iloc[0]["A"]
    assert a == pytest.approx(np.average([_z(y, maps, "m0"), _z(y, maps, "m1")], weights=[3, 1]),
                              abs=1e-5)
