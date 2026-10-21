"""diagnostics.shared_axes(): loadings, attribution (= partial correlation) and Moran null degeneracy.
Design: _explore/shared_axes_design.md."""

import numpy as np
import pandas as pd
import pytest

from nispace import NiSpace
from nispace.datasets import fetch_axes
from nispace.diagnostics import shared_axes

N_DK = 68


@pytest.fixture(scope="module")
def dk_axes():
    return fetch_axes("DesikanKilliany", "all", verbose=False)


def _z(v):
    return (v - v.mean()) / v.std()


def _nsp(x, y, parc="DesikanKilliany", z=None, fit=True):
    nsp = NiSpace(x=x, y=y, z=z, parcellation=parc, verbose=False)
    return nsp.fit() if fit else nsp


def _attr(res, axis, x=None, y=None):
    a = res["attribution"]
    a = a[a["axis"] == axis]
    if x is not None:
        a = a[a["x"] == x]
    if y is not None:
        a = a[a["y"] == y]
    return a


# --- ground truth: attribution == colocalize(partial*) -------------------------------------------

@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_single_axis_partial_equals_colocalize(rng, method):
    x, y = rng.normal(size=(3, N_DK)), rng.normal(size=(2, N_DK))
    res = shared_axes(_nsp(x, y), axes=["gm", "radial"], method=method, null_check=False)
    for axis in ["gm", "radial"]:
        ref = _nsp(x, y, z=axis)
        ref.colocalize(f"partial{method}", r_to_z=False)
        expected = ref.get_colocalizations(f"partial{method}")
        got = _attr(res, axis).pivot(index="y", columns="x", values="rho_partial")
        np.testing.assert_allclose(got.values, expected.values, atol=1e-5)


def test_joint_partial_and_observed_equal_colocalize(rng):
    x, y = rng.normal(size=(3, N_DK)), rng.normal(size=(1, N_DK))
    axes = ["gm", "radial", "ap", "si"]
    res = shared_axes(_nsp(x, y), axes=axes, null_check=False)
    ref = _nsp(x, y, z=axes)
    ref.colocalize("partialspearman", r_to_z=False)
    ref.colocalize("spearman", regress_z=False, r_to_z=False)
    joint = _attr(res, "joint")
    np.testing.assert_allclose(joint["rho_partial"].values,
                               ref.get_colocalizations("partialspearman").values[0], atol=1e-5)
    np.testing.assert_allclose(joint["rho"].values,
                               ref.get_colocalizations("spearman").values[0], atol=1e-5)


def test_loadings_and_axis_corr(rng, dk_axes):
    x, y = rng.normal(size=(2, N_DK)), rng.normal(size=(1, N_DK))
    res = shared_axes(_nsp(x, y), axes=["gm", "ap"], null_check=False)
    load = res["loadings"]
    assert list(load.columns) == ["gm", "ap"]
    assert load.index.get_level_values("role").tolist() == ["Y", "X", "X"]
    expected = pd.Series(y[0]).corr(pd.Series(dk_axes.loc["gm"].values), method="spearman")
    assert load.loc[("Y", "0"), "gm"] == pytest.approx(expected, abs=1e-6)
    assert res["axis_corr"].loc["gm", "ap"] == pytest.approx(
        dk_axes.loc[["gm", "ap"]].T.corr("spearman").iloc[0, 1], abs=1e-6)


# --- flag logic ------------------------------------------------------------------------------------

def test_flag_only_when_both_sides_load(rng, dk_axes):
    gm = _z(dk_axes.loc["gm"].values)
    x_shared = gm + 0.4 * rng.normal(size=N_DK)
    x_indep = rng.normal(size=N_DK)
    y = gm + 0.4 * rng.normal(size=N_DK)
    res = shared_axes(_nsp(np.vstack([x_shared, x_indep]), y[None]), axes=["gm"],
                      null_check=False)
    a = _attr(res, "gm").set_index("x")
    assert a.loc["0", "flagged"] and a.loc["0", "delta"] < -0.2
    # Y loads on gm but X does not: partialling inflates |rho| (denominator), not flagged
    assert not a.loc["1", "flagged"]


def test_pooled_row_for_multiple_y(rng, dk_axes):
    gm = _z(dk_axes.loc["gm"].values)
    x = (gm + 0.5 * rng.normal(size=N_DK))[None]
    y = gm + 1.5 * rng.normal(size=(8, N_DK))  # noisy "subject" maps sharing gm
    res = shared_axes(_nsp(x, y), axes=["gm"], pooled="mean", null_check=False)
    a = _attr(res, "gm")
    assert "pooled (mean)" in a["y"].values and len(a) == 9
    pooled = a[a["y"] == "pooled (mean)"].iloc[0]
    per_row = a[a["y"] != "pooled (mean)"]
    assert pooled["rho"] == pytest.approx(np.tanh(np.arctanh(per_row["rho"]).mean()), abs=1e-6)
    assert ("Y", "pooled (mean)") in res["loadings"].index
    assert res["settings"]["pooled"] == "mean"


# --- null degeneracy --------------------------------------------------------------------------------

def test_null_degeneracy_flags_shared_dominant_gradient(rng, dk_axes):
    """Gradient X vs gradient Y: one MEM term -> flagged. Broad-spectrum (noise) X vs noise Y:
    many comparable terms -> not flagged. (A noise X vs a pure-gradient Y IS degenerate: the
    null is +- X's MEM1 coefficient times Y's MEM1 projection.)"""
    ap = _z(dk_axes.loc["ap"].values)
    x = np.vstack([ap + 0.2 * rng.normal(size=N_DK), rng.normal(size=N_DK)])
    y = np.vstack([ap + 0.2 * rng.normal(size=N_DK), rng.normal(size=N_DK)])
    res = shared_axes(_nsp(x, y), axes=["ap"], pooled=False)
    deg = res["null_degeneracy"].set_index(["y", "x"])
    assert deg.loc[("0", "0"), "top_share"] > 0.9 and deg.loc[("0", "0"), "flagged"]
    assert deg.loc[("0", "0"), "top_term"] == "mem1"
    assert not deg.loc[("1", "1"), "flagged"]
    assert deg.loc[("1", "1"), "n_eff"] > deg.loc[("0", "0"), "n_eff"]
    spec = res["null_spectrum"]
    assert set(spec["role"]) == {"X", "Y"}
    assert res["settings"]["null_source"] == "default" and res["settings"]["permuted"] == "X"


def _sarle(r):
    """Sarle's bimodality coefficient; > 0.555 (uniform) suggests bimodality."""
    from scipy.stats import kurtosis, skew
    n = len(r)
    return (skew(r) ** 2 + 1) / (kurtosis(r) + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


@pytest.mark.parametrize("parc,n", [("DesikanKilliany", N_DK), ("Schaefer100+TianS1", 116)])
def test_null_degeneracy_predicts_bimodal_null(rng, parc, n):
    """The flag must mean what it says: the actual Moran null of a flagged pair is bimodal,
    the null of an unflagged pair is not (per-hemisphere and cortex/subcortex blocks)."""
    ap = _z(fetch_axes(parc, "ap", verbose=False).values[0])
    x = np.vstack([ap + 0.2 * rng.normal(size=n), rng.normal(size=n)])
    y = np.vstack([ap + 0.3 * rng.normal(size=n), rng.normal(size=n)])
    nsp = _nsp(x, y, parc=parc)
    nsp.colocalize("pearson", r_to_z=False)
    nsp.permute("maps", maps_which="X", maps_method=nsp._parc.default_null_method, n_perm=500,
                seed=1, pooled_p=False)
    res = shared_axes(nsp, axes=["ap"])
    assert res["settings"]["null_source"] == "stored"
    deg = res["null_degeneracy"].set_index(["y", "x"])
    null = np.asarray(nsp._nulls["maps_null"].data, dtype=float)  # (n_x, n_perm, n_parcels)
    for i, j, expect_bimodal in [(0, 0, True), (1, 1, False)]:
        r = np.array([np.corrcoef(m, y[i])[0, 1] for m in null[j]])
        assert deg.loc[(str(i), str(j)), "flagged"] == expect_bimodal
        assert (_sarle(r) > 0.555) == expect_bimodal


def test_null_degeneracy_uses_permuted_maps_own_parcels(rng, dk_axes):
    """MEMs are fit on the permuted map's own finite parcels, like generate_null_maps(), not on
    the parcels finite in X and Y. Here X (fixed side) has NaN parcels and Y is permuted: the
    flag must still match the actual null's shape."""
    ap = _z(dk_axes.loc["ap"].values)
    x = np.vstack([ap + 0.2 * rng.normal(size=N_DK), rng.normal(size=N_DK)])
    x[:, [3, 40]] = np.nan
    y = np.vstack([ap + 0.3 * rng.normal(size=N_DK), rng.normal(size=N_DK)])
    nsp = _nsp(x, y)
    nsp.colocalize("pearson", r_to_z=False)
    nsp.permute("maps", maps_which="Y", maps_method=nsp._parc.default_null_method, n_perm=500,
                seed=1, pooled_p=False)
    res = shared_axes(nsp, axes=["ap"])
    assert res["settings"]["permuted"] == "Y"
    spec = res["null_spectrum"]
    assert (spec[spec["role"] == "Y"].groupby("map")["n_parcels"].sum() == N_DK).all()
    assert (spec[spec["role"] == "X"].groupby("map")["n_parcels"].sum() == N_DK - 2).all()
    deg = res["null_degeneracy"].set_index(["y", "x"])
    keep = np.isfinite(x).all(axis=0)
    null = np.asarray(nsp._nulls["maps_null"].data, dtype=float)  # (n_y, n_perm, n_parcels)
    for i, expect_bimodal in [(0, True), (1, False)]:
        r = np.array([np.corrcoef(m[keep], x[i, keep])[0, 1] for m in null[i]])
        assert deg.loc[(str(i), str(i)), "flagged"] == expect_bimodal
        assert (_sarle(r) > 0.555) == expect_bimodal


def test_drop_nan_matches_by_parcel_label(rng):
    """NiSpace(drop_nan=True) keeps a subset of parcels: axes and null blocks are selected by
    label. Attribution equals the run without dropping (NaN parcels are excluded there too)."""
    x, y = rng.normal(size=(2, N_DK)), rng.normal(size=(1, N_DK))
    x[0, [3, 40]] = np.nan
    nsp_drop = NiSpace(x=x, y=y, parcellation="DesikanKilliany", drop_nan=True,
                       verbose=False).fit()
    assert nsp_drop._X.shape[1] == N_DK - 2
    res_drop = shared_axes(nsp_drop, axes=["gm", "ap"])
    res_full = shared_axes(_nsp(x, y), axes=["gm", "ap"])
    assert list(res_drop["axes"].columns) == list(nsp_drop._X.columns)
    pd.testing.assert_frame_equal(res_drop["axes"], res_full["axes"][nsp_drop._X.columns])
    cols = ["y", "x", "axis", "n_parcels", "rho", "rho_partial"]
    pd.testing.assert_frame_equal(res_drop["attribution"][cols], res_full["attribution"][cols],
                                  atol=1e-6)
    # X map 0 has the same finite parcels in both runs -> same null weights
    d_drop = res_drop["null_degeneracy"].set_index(["y", "x"])
    d_full = res_full["null_degeneracy"].set_index(["y", "x"])
    np.testing.assert_allclose(d_drop.loc[("0", "0"), ["top_share", "n_eff"]].astype(float),
                               d_full.loc[("0", "0"), ["top_share", "n_eff"]].astype(float),
                               rtol=1e-6)


@pytest.mark.parametrize("parc,n", [("DesikanKilliany", N_DK), ("Schaefer100+TianS1", 116)])
def test_msr_signs_shared_across_blocks(rng, parc, n):
    """shared_axes() assumes generate_null_maps() draws the same MSR sign sequence in every
    null block (same seed per hemisphere / cortex / subcortex) and sums those terms. If the
    seeding changes, this fails -- update _null_degeneracy() accordingly."""
    from nispace.diagnostics import _moran_null_blocks
    from nispace.nulls import _moran_fit_1_over_d
    x = rng.normal(size=(1, n))
    nsp = _nsp(x, rng.normal(size=(1, n)), parc=parc)
    nsp.colocalize("pearson")
    nsp.permute("maps", maps_which="X", maps_method=nsp._parc.default_null_method, n_perm=100,
                seed=3)
    null = np.asarray(nsp._nulls["maps_null"].data, dtype=float)[0]
    _, _, blocks, _ = _moran_null_blocks(nsp, np.ones(n, bool))
    assert len(blocks) >= 2
    signs = []
    for name, idc, dist in blocks:
        mem = np.asarray(_moran_fit_1_over_d(dist.copy(), n_components=15)[0])
        xb = x[0, idc]
        c = (xb - xb.mean()) / xb.std(ddof=1) @ mem
        signs.append(np.sign(((null[:, idc] - null[:, idc].mean(1, keepdims=True)) @ mem) / c))
    for s in signs[1:]:
        assert (s == signs[0]).all()


def test_null_check_skipped_for_groups_permutation(rng):
    y = rng.normal(size=(20, N_DK))
    nsp = _nsp(rng.normal(size=(2, N_DK)), y)
    nsp.transform_y("hedges(a,b)", groups=[0] * 10 + [1] * 10)
    nsp.colocalize("spearman", Y_transform="hedges(a,b)")
    nsp.permute("groups", Y_transform="hedges(a,b)", n_perm=20, seed=1)
    res = shared_axes(nsp, axes=["gm"], Y_transform="hedges(a,b)")
    assert res["null_degeneracy"] is None and res["null_spectrum"] is None


# --- combined parcellation --------------------------------------------------------------------------

def test_combined_parcellation_blocks_and_partial_coverage(rng):
    n = 116  # Schaefer100 + TianS1
    nsp = _nsp(rng.normal(size=(2, n)), rng.normal(size=(1, n)), parc="Schaefer100+TianS1")
    res = shared_axes(nsp, axes=["cxsc", "saaxis"])
    assert res["settings"]["n_parcels"] == {"cxsc": 116, "saaxis": 100}
    assert set(_attr(res, "saaxis")["n_parcels"]) == {100}
    assert set(_attr(res, "joint")["n_parcels"]) == {100}
    blocks = set(res["null_spectrum"]["block"])
    assert any(b.startswith("cortex") for b in blocks) and "subcortex" in blocks


def test_default_axes_and_return_keys(rng):
    res = shared_axes(_nsp(rng.normal(size=(1, N_DK)), rng.normal(size=(1, N_DK))))
    assert res["settings"]["axes"] == ["gm", "radial", "ap", "si", "saaxis", "fcgradient1",
                                       "fcgradient2"]
    assert set(res) == {"axes", "loadings", "axis_corr", "attribution", "null_degeneracy",
                        "null_spectrum", "settings"}


def test_set_indexed_x_checked_per_map(rng):
    """Collections return X with a (set, map) index even without XSEA (e.g. fetch_reference(
    "pet", collection="UniqueTracers")). Regression: shared_axes() used to refuse such X; maps
    are now checked individually, once even when listed in several sets."""
    x, y = rng.normal(size=(3, N_DK)), rng.normal(size=(1, N_DK))
    names = ["m0", "m1", "m2"]
    plain = pd.DataFrame(x, index=names)
    sets = pd.DataFrame(np.vstack([x, x[[1]]]),
                        index=pd.MultiIndex.from_tuples(
                            [("A", "m0"), ("A", "m1"), ("B", "m2"), ("B", "m1")],
                            names=["set", "map"]))
    res_plain = shared_axes(_nsp(plain, y), axes=["gm", "ap"])
    res_sets = shared_axes(_nsp(sets, y), axes=["gm", "ap"])
    assert sorted(res_sets["attribution"]["x"].unique()) == names
    pd.testing.assert_frame_equal(res_sets["attribution"], res_plain["attribution"])
    pd.testing.assert_frame_equal(res_sets["null_degeneracy"], res_plain["null_degeneracy"])
