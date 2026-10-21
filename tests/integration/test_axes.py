"""Named spatial axes: nispace.core.axes registry, datasets.fetch_axes(), Parcellation.get_centroids()
and NiSpace(z=<axis name>). Design: _explore/shared_axes_design.md."""

import numpy as np
import pytest

from nispace import NiSpace
from nispace.core.axes import (AxisUnavailableError, _RADIAL_CENTRE, resolve_axis_names,
                               is_axis_name)
from nispace.datasets import fetch_axes, fetch_parcellation


@pytest.fixture(scope="module")
def dk():
    return fetch_parcellation("DesikanKilliany", verbose=False)


@pytest.fixture(scope="module")
def dk_axes():
    return fetch_axes("DesikanKilliany", "all", verbose=False)


# --- registry ---------------------------------------------------------------------------------

def test_resolve_names_expands_groups_and_dedupes():
    names, explicit = resolve_axis_names(["GM", "coords", "ap", "mem3"])
    assert names == ["gm", "ap", "si", "lr", "mem3"]
    assert explicit


def test_resolve_names_unknown_raises():
    with pytest.raises(ValueError, match="Unknown axis 'foo'"):
        resolve_axis_names(["gm", "foo"])


def test_is_axis_name():
    assert is_axis_name("Radial") and is_axis_name("coords") and is_axis_name("mem12")
    assert not is_axis_name("mem0") and not is_axis_name("gm.nii.gz") and not is_axis_name(3)


def test_default_set_depends_on_parcellation_type(dk):
    names, explicit = resolve_axis_names("default", dk)
    assert not explicit
    assert names == ["gm", "radial", "ap", "si", "saaxis", "fcgradient1", "fcgradient2"]
    combined = fetch_parcellation("Schaefer100+TianS1", verbose=False)
    assert resolve_axis_names("default", combined)[0] == ["gm", "radial", "ap", "si", "cxsc"]
    sc = fetch_parcellation("TianS1", verbose=False)
    assert resolve_axis_names("default", sc)[0] == ["gm", "radial", "ap", "si"]


# --- centroids / geometry ---------------------------------------------------------------------

def test_centroids_mni_vs_surface_agree(dk):
    c_mni = dk.get_centroids("MNI152NLin6Asym")
    c_surf = dk.get_centroids("fsLR")
    assert c_mni.shape == c_surf.shape == (68, 3)
    assert np.median(np.linalg.norm(c_mni - c_surf, axis=1)) < 6
    for i in range(3):
        assert np.corrcoef(c_mni[:, i], c_surf[:, i])[0, 1] > 0.99


def test_centroids_hemispheres_and_cache(dk):
    c = dk.get_centroids()
    is_lh = np.array([str(l).startswith("hemi-L") for l in dk._labels])
    assert (c[is_lh, 0] < 0).all() and (c[~is_lh, 0] > 0).all()
    c[:] = 0  # returned array is a copy, cache untouched
    assert np.abs(dk.get_centroids()).sum() > 0


def test_geometry_axes_from_centroids(dk, dk_axes):
    c = dk.get_centroids()
    np.testing.assert_allclose(dk_axes.loc["ap"].values, c[:, 1])
    np.testing.assert_allclose(dk_axes.loc["si"].values, c[:, 2])
    np.testing.assert_allclose(dk_axes.loc["lr"].values, c[:, 0])
    np.testing.assert_allclose(dk_axes.loc["medlat"].values, np.abs(c[:, 0]))
    np.testing.assert_allclose(dk_axes.loc["radial"].values,
                               np.linalg.norm(c - _RADIAL_CENTRE, axis=1))


def test_all_axes_complete_for_cortex_parcellation(dk_axes):
    assert dk_axes.shape == (15, 68)
    assert not dk_axes.isna().any().any()


def test_hemi_restricted(dk):
    full = fetch_axes("DesikanKilliany", ["gm", "radial"], verbose=False)
    rh = fetch_axes("DesikanKilliany", ["gm", "radial"], hemi="R", verbose=False)
    assert rh.shape == (2, 34)
    np.testing.assert_allclose(rh.values, full[rh.columns].values)


def test_custom_parcellation_matches_integrated(dk):
    """Custom Parcellation (from image): TPM via raw images + parcellate_data, geometry via the
    image itself -- both must match the integrated tables/computation."""
    from nispace.core.parcellation import Parcellation
    img = dk.get_image("MNI152NLin6Asym")
    custom = Parcellation.from_path(img, space="MNI152NLin6Asym", labels=dk._labels,
                                    name="custom_dk", verbose=False)
    space_before = custom._space
    a = fetch_axes(custom, ["gm", "radial", "ap"], verbose=False)
    b = fetch_axes("DesikanKilliany", ["gm", "radial", "ap"], verbose=False)
    np.testing.assert_allclose(a.values, b.values, rtol=1e-4)
    assert custom._space == space_before  # caller's active space untouched
    with pytest.raises(AxisUnavailableError, match="cortexfeatures"):
        fetch_axes(custom, "saaxis", verbose=False)


# --- compartment / Moran / coverage -----------------------------------------------------------

def test_cxsc_and_cortex_only_coverage():
    ax = fetch_axes("Schaefer100+TianS1", ["cxsc", "saaxis"], verbose=False)
    assert ax.loc["cxsc"].sum() == 16
    assert ax.loc["saaxis"].isna().sum() == 16  # cortex-only map on a combined parcellation
    assert ax.loc["saaxis"][ax.loc["cxsc"] == 0].notna().all()
    with pytest.raises(AxisUnavailableError, match="undefined for some parcels"):
        fetch_axes("Schaefer100+TianS1", "saaxis", require_complete=True, verbose=False)


def test_unavailable_axis_explicit_raises_default_skips():
    with pytest.raises(AxisUnavailableError, match="cxsc"):
        fetch_axes("DesikanKilliany", "cxsc", verbose=False)
    sc = fetch_axes("TianS1", "all", verbose=False)  # no cortexfeatures, no cxsc -> skipped
    assert "saaxis" not in sc.index and "cxsc" not in sc.index and "gm" in sc.index


def test_mems_are_smooth_orthogonal_patterns(dk):
    ax = fetch_axes(dk, ["mem1", "mem2", "mem3"], verbose=False)
    m = ax.values
    m_c = m - m.mean(axis=1, keepdims=True)
    gram = m_c @ m_c.T / np.outer(np.linalg.norm(m_c, axis=1), np.linalg.norm(m_c, axis=1))
    np.testing.assert_allclose(gram, np.eye(3), atol=0.05)
    # smooth: strongly predicted by the linear coordinate axes
    c = dk.get_centroids()
    design = np.column_stack([np.ones(68), c])
    r2 = 1 - ((m[0] - design @ np.linalg.lstsq(design, m[0], rcond=None)[0]) ** 2).sum() / \
        ((m[0] - m[0].mean()) ** 2).sum()
    assert r2 > 0.6


# --- NiSpace(z=...) ---------------------------------------------------------------------------

def test_z_axis_names_match_fetch_axes(rng):
    names = ["gm", "radial", "ap", "saaxis"]
    nsp = NiSpace(x=rng.normal(size=(1, 68)), y=rng.normal(size=(1, 68)), z=names,
                  parcellation="DesikanKilliany", standardize=False, verbose=False).fit()
    z = nsp.get_z()
    assert list(z.index) == names
    expected = fetch_axes("DesikanKilliany", names, verbose=False)
    np.testing.assert_allclose(z.values, expected.values, rtol=1e-5)


def test_z_axis_bilateral_parcellation_object(rng):
    """Regression: z= TPM names had their own code path that ignored `bilateral` (fetched the
    L+R table for a bilateral parcellation). All named z axes now go through build_axes."""
    parc = fetch_parcellation("DesikanKilliany", hemi=["L", "R"], return_dist_mat=False,
                              verbose=False)
    parc.make_bilateral()
    space_before = parc._space
    names = ["gm", "radial"]
    nsp = NiSpace(x=rng.normal(size=(1, 34)), y=rng.normal(size=(1, 34)), z=names,
                  parcellation=parc, standardize=False, verbose=False).fit()
    np.testing.assert_allclose(nsp.get_z().values, fetch_axes(parc, names, verbose=False).values)
    # NiSpace(parcellation=<Parcellation>) used to fail in fit(); it now works on a copy
    assert nsp._parc is not parc and parc._space == space_before


def test_z_custom_parcellation_forwards_parcellate_settings(rng, monkeypatch, dk):
    """TPM axes on a custom parcellation are parcellated with the NiSpace object's z settings."""
    import nispace.io as io_mod
    calls = []
    real = io_mod.parcellate_data

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(io_mod, "parcellate_data", spy)
    NiSpace(x=rng.normal(size=(1, 68)), y=rng.normal(size=(1, 68)), z="gm",
            parcellation=dk.get_image("MNI152NLin6Asym"), parcellation_space="MNI152NLin6Asym",
            resampling_target="parcellation", verbose=False).fit()
    z_calls = [c for c in calls if c.get("data_labels") == ["gm"]]
    assert len(z_calls) == 1
    assert z_calls[0]["resampling_target"] == "parcellation"


def test_to_parc_order_never_positional():
    """A table whose labels share nothing with the parcellation raises instead of being
    assigned by column position."""
    import pandas as pd
    from nispace.core.axes import _to_parc_order
    df = pd.DataFrame(np.ones((1, 3)), columns=["a", "b", "c"])
    with pytest.raises(RuntimeError, match="none of the table's parcel labels"):
        _to_parc_order(df, ["x", "y", "z"], "tpm")


def test_z_axis_combined_parcellation_coverage_raises(rng):
    n = 116
    nsp = NiSpace(x=rng.normal(size=(1, n)), y=rng.normal(size=(1, n)), z=["gm", "saaxis"],
                  parcellation="Schaefer100+TianS1", verbose=False)
    with pytest.raises(AxisUnavailableError, match="undefined for some parcels"):
        nsp.fit()


def test_z_axis_partialspearman_equals_manual_residualization(rng):
    from scipy.stats import rankdata
    x, y = rng.normal(size=(1, 68)), rng.normal(size=(1, 68))
    nsp = NiSpace(x=x, y=y, z="radial", parcellation="DesikanKilliany", standardize=False,
                  verbose=False).fit()
    nsp.colocalize("partialspearman", r_to_z=False)
    rho = nsp.get_colocalizations("partialspearman").iloc[0, 0]
    zr = rankdata(fetch_axes("DesikanKilliany", "radial", verbose=False).values[0])

    def resid(v):
        b = np.polyfit(zr, v, 1)
        return v - np.polyval(b, zr)

    expected = np.corrcoef(resid(rankdata(x[0])), resid(rankdata(y[0])))[0, 1]
    assert rho == pytest.approx(expected, abs=1e-5)
