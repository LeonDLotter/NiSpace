"""Results of map permutations are keyed by their null method ("..._perm-xmaps_null-moran_..."),
so several null methods coexist in one NiSpace object. Network- and parcellation-free:
DataFrame input, null methods "random" and "moran" (explicit synthetic distance matrix)."""

import numpy as np
import pandas as pd
import pytest

from nispace import NiSpace
from nispace.utils.utils import (_get_df_string, _parse_df_string, _null_label,
                                 _strip_null_field, _add_null_field)


N_PARC = 30
N_PERM = 50


@pytest.fixture
def nsp():
    rng = np.random.default_rng(0)
    cols = [f"p{i}" for i in range(N_PARC)]
    x = pd.DataFrame(rng.normal(size=(4, N_PARC)), index=[f"x{i}" for i in range(4)], columns=cols)
    y = pd.DataFrame(rng.normal(size=(2, N_PARC)), index=["y0", "y1"], columns=cols)
    nsp = NiSpace(x=x, y=y, verbose=False).fit()
    nsp.colocalize("spearman")
    return nsp


@pytest.fixture
def dist_mat():
    coords = np.random.default_rng(1).uniform(-50, 50, size=(N_PARC, 3))
    return np.linalg.norm(coords[:, None] - coords[None], axis=-1)


def _permute(nsp, method, dist_mat, **kw):
    return nsp.permute("maps", maps_method=method, n_perm=N_PERM, seed=1,
                       dist_mat=dist_mat if method != "random" else None,
                       maps_use_existing=False, **kw)


# ── key helpers ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("null,label", [
    (None, None), ("moran", "moran"), (("cornblath", "moran"), "cornblath+moran"),
    ("alexander_bloch", "alexanderbloch"), ("NiMARE nulls!", "nimarenulls"), ("", "custom"),
])
def test_null_label(null, label):
    assert _null_label(null) == label


def test_key_null_field_roundtrip():
    key = _get_df_string("p", method="spearman", stat="rho", perm="xmaps",
                         null=("cornblath", "moran"), pooled_p="mean", mc="fdrbh")
    assert "_perm-xmaps_null-cornblath+moran_pooled-mean_mc-fdrbh" in key
    fields = _parse_df_string(key)
    assert fields["perm"] == "xmaps" and fields["null"] == "cornblath+moran"
    assert fields["pooled"] == "mean" and fields["mc"] == "fdrbh"
    no_null = _get_df_string("p", method="spearman", stat="rho", perm="xmaps",
                             pooled_p="mean", mc="fdrbh")
    assert _strip_null_field(key) == no_null
    assert _add_null_field(no_null, ("cornblath", "moran")) == key
    assert _add_null_field(key, "moran") == key  # already has one


def test_groups_keys_have_no_null_field():
    key = _get_df_string("p", method="spearman", stat="rho", perm="groups", mc=None)
    assert "null-" not in key and "null" not in _parse_df_string(key)


# ── permute / getters ───────────────────────────────────────────────────────

def test_two_null_methods_coexist(nsp, dist_mat):
    _permute(nsp, "random", dist_mat)
    p_random = nsp.get_p_values()
    _permute(nsp, "moran", dist_mat)
    p_moran = nsp.get_p_values()  # default: last used
    keys = [k for k in nsp._p_colocs if "mc-none" in k]
    assert {_parse_df_string(k)["null"] for k in keys} == {"random", "moran"}
    assert {_parse_df_string(k)["null"] for k in nsp._nulls["_colocs"]} == {"random", "moran"}
    pd.testing.assert_frame_equal(nsp.get_p_values(maps_method="random"), p_random)
    pd.testing.assert_frame_equal(nsp.get_p_values(maps_method="moran"), p_moran)
    assert not np.allclose(p_random.values, p_moran.values)
    # aliases resolve to the canonical name
    pd.testing.assert_frame_equal(nsp.get_p_values(maps_method="msr"), p_moran)


def test_default_falls_back_to_only_stored_method(nsp, dist_mat):
    _permute(nsp, "moran", dist_mat)
    nsp._last_settings["maps_method"] = "something-else"
    assert nsp.get_p_values().shape[1] == 4


def test_ambiguous_default_raises(nsp, dist_mat):
    _permute(nsp, "random", dist_mat)
    _permute(nsp, "moran", dist_mat)
    nsp._last_settings["maps_method"] = None
    with pytest.raises(ValueError, match="ambiguous"):
        nsp.get_p_values()


def test_unknown_method_raises(nsp, dist_mat):
    _permute(nsp, "random", dist_mat)
    with pytest.raises(KeyError, match="burt2020"):
        nsp.get_p_values(maps_method="burt2020")


def test_correct_p_per_null_method(nsp, dist_mat):
    _permute(nsp, "random", dist_mat)
    _permute(nsp, "moran", dist_mat)
    nsp.correct_p("fdr_bh")
    nsp.correct_p("maxT")  # needs the null colocalizations of the same key
    for m in ("random", "moran"):
        p = nsp.get_p_values(maps_method=m)
        p_fdr = nsp.get_p_values(maps_method=m, mc_method="fdr_bh")
        assert p_fdr.shape == p.shape and (p_fdr.values >= p.values - 1e-12).all()
        assert nsp.get_p_values(maps_method=m, mc_method="maxT").shape == p.shape


def test_nulls_and_normalized_per_null_method(nsp, dist_mat):
    _permute(nsp, "random", dist_mat)
    _permute(nsp, "moran", dist_mat)
    nsp.normalize_colocalizations()
    z_r = nsp.get_colocalizations(normalized=True, maps_method="random")
    z_m = nsp.get_colocalizations(normalized=True, maps_method="moran")
    assert z_r.shape == z_m.shape == (2, 4)
    assert not np.allclose(z_r.values, z_m.values)
    _, nulls_r = nsp.get_colocalizations(get_nulls=True, nulls_permute_what="xmaps",
                                         maps_method="random")
    _, nulls_m = nsp.get_colocalizations(get_nulls=True, nulls_permute_what="xmaps",
                                         maps_method="moran")
    assert not np.allclose(nulls_r["x0"].values, nulls_m["x0"].values)


def test_legacy_pickle_keys_migrated(nsp, dist_mat, tmp_path):
    _permute(nsp, "moran", dist_mat)
    p_moran = nsp.get_p_values()
    # simulate a pickle written before the null field existed
    for store in (nsp._p_colocs, nsp._nulls["_colocs"]):
        for k in list(store):
            store[_strip_null_field(k)] = store.pop(k)
    for k in [k for k in nsp._nulls if k.startswith("p_tails_")]:
        nsp._nulls[_strip_null_field(k)] = nsp._nulls.pop(k)
    del nsp._last_settings["maps_method"]
    path = tmp_path / "nsp.pkl"
    nsp.to_pickle(path)
    nsp2 = NiSpace.from_pickle(path, verbose=False)
    assert all("_null-moran" in k for k in nsp2._p_colocs)
    assert any(k.startswith("p_tails_") and "_null-moran" in k for k in nsp2._nulls)
    pd.testing.assert_frame_equal(nsp2.get_p_values(), p_moran)
    pd.testing.assert_frame_equal(nsp2.get_p_values(maps_method="moran"), p_moran)
