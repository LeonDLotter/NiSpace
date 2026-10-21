"""Coverage for the ``z="gm"``/``"wm"``/``"csf"``/``"veins"``/``"arteries"``
TPM auto-fetch shortcut in ``NiSpace.fit()``. ``fit()`` builds all named z axes
through ``core.axes.build_axes`` (shared with ``fetch_axes``). Two code paths exist
for TPMs:

- fast path: when the current parcellation is a registered/integrated one,
  ``fetch_reference("tpm", ..., parcellation=...)`` returns the precomputed
  per-parcellation table directly -- no re-parcellation, and any
  ``background_value`` default/override is a no-op (the table is already
  correctly background-treated).
- fallback path: for a custom (non-integrated) parcellation, raw TPM images
  are fetched in a fixed canonical space (``MNI152NLin6Asym``) and
  parcellated the normal way, resampling into the parcellation's own space
  as needed.

Before this module existed, the mechanism had exactly one incidental usage
(``z="gm"`` with an integrated parcellation in test_workflow_e2e.py, there
only to ground-truth colocalize()'s Z-regression math). Nothing exercised
the fallback path, multi-shortcut lists, or the single-hemi-restricted edge
case below -- which broke during development of the fast path.
"""

import numpy as np
import pytest

import nispace.datasets as ds_mod
from nispace import NiSpace
from nispace.datasets import fetch_reference, fetch_parcellation


def _spy_fetch_reference(monkeypatch):
    """Wrap nispace.datasets.fetch_reference (imported at call time by core.axes) to
    record call kwargs while still delegating to the real function -- lets tests assert
    which code path (fast vs. fallback) was actually taken, not just infer it from output."""
    calls = []
    real = ds_mod.fetch_reference

    def wrapper(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(ds_mod, "fetch_reference", wrapper)
    return calls


def test_fast_path_used_and_matches_precomputed_table(rng, monkeypatch):
    calls = _spy_fetch_reference(monkeypatch)
    x, y = rng.normal(size=(1, 68)), rng.normal(size=(1, 68))
    # standardize=False: compare against fetch_reference()'s raw table directly,
    # without NiSpace's own (independent, already-tested) z-scoring in the way
    nsp = NiSpace(x=x, y=y, z="gm", parcellation="DesikanKilliany", standardize=False,
                  verbose=False)
    nsp.fit()

    assert len(calls) == 1
    assert calls[0].get("parcellation") is not None
    assert "space" not in calls[0]

    expected = fetch_reference("tpm", maps=["gm"], parcellation="DesikanKilliany",
                               standardize_parcellated=False, print_references=False,
                               verbose=False)
    np.testing.assert_allclose(nsp.get_z().values, expected.values)
    assert list(nsp.get_z().index) == ["gm"]


def test_fast_path_list_of_shortcuts(rng):
    x, y = rng.normal(size=(1, 68)), rng.normal(size=(1, 68))
    nsp = NiSpace(x=x, y=y, z=["gm", "wm"], parcellation="DesikanKilliany", verbose=False)
    nsp.fit()
    assert nsp.get_z().shape == (2, 68)
    assert list(nsp.get_z().index) == ["gm", "wm"]


def test_fast_path_is_case_insensitive(rng):
    x, y = rng.normal(size=(1, 68)), rng.normal(size=(1, 68))
    nsp_lower = NiSpace(x=x, y=y, z="gm", parcellation="DesikanKilliany", verbose=False)
    nsp_lower.fit()
    nsp_upper = NiSpace(x=x, y=y, z="GM", parcellation="DesikanKilliany", verbose=False)
    nsp_upper.fit()

    assert list(nsp_upper.get_z().index) == ["gm"]
    np.testing.assert_allclose(nsp_lower.get_z().values, nsp_upper.get_z().values)


def test_fast_path_single_hemi_restricted(rng):
    """Regression test: Parcellation.get_hemi() returns None right after
    select_hemi() runs (it only reads from _hemi_dict, which select_hemi()
    never populates) -- using it instead of _selected_hemi silently fetched
    the full bilateral TPM table against a hemi-restricted (half-sized)
    parcellation, raising a pandas shape-mismatch error."""
    x, y = rng.normal(size=(1, 34)), rng.normal(size=(1, 34))
    nsp = NiSpace(x=x, y=y, z=["gm", "wm"], parcellation="DesikanKilliany",
                  parcellation_hemi="R", verbose=False)
    nsp.fit()
    assert nsp.get_z().shape == (2, 34)


def test_fallback_path_used_for_custom_parcellation_and_matches_fast_path(rng, monkeypatch):
    calls = _spy_fetch_reference(monkeypatch)
    parc = fetch_parcellation("DesikanKilliany")
    parc.set_active_space("MNI152NLin6Asym")
    img = parc._image_obj

    x, y = rng.normal(size=(1, 68)), rng.normal(size=(1, 68))
    nsp = NiSpace(x=x, y=y, z="gm", parcellation=img, parcellation_space="MNI152NLin6Asym",
                  verbose=False)
    nsp.fit()

    assert len(calls) == 1
    assert calls[0].get("space") is not None
    assert "parcellation" not in calls[0]

    # same underlying map + same parcel geometry as the integrated-name fast path --
    # values should match exactly (both derive from the same MNI152NLin6Asym gm map,
    # processed the same way)
    nsp_fast = NiSpace(x=x, y=y, z="gm", parcellation="DesikanKilliany", verbose=False)
    nsp_fast.fit()
    np.testing.assert_allclose(nsp.get_z().values, nsp_fast.get_z().values)


def test_fallback_path_resamples_to_custom_surface_parcellation(rng):
    parc = fetch_parcellation("DesikanKilliany")
    parc.set_active_space("fsaverage")
    img = parc._image_obj

    x, y = rng.normal(size=(1, 68)), rng.normal(size=(1, 68))
    nsp = NiSpace(x=x, y=y, z="gm", parcellation=img, parcellation_space="fsaverage",
                  verbose=False)
    nsp.fit()

    Z = nsp.get_z()
    assert Z.shape == (1, 68)
    assert np.isfinite(Z.values).all()


def test_mixed_shortcut_list_raises(rng):
    """Named axes and other inputs (here: a would-be image path) can't be mixed in one
    z list -- raised explicitly instead of falling through to parcellate_data, which used
    to fail with a confusing FileNotFoundError on the axis name."""
    x, y = rng.normal(size=(1, 68)), rng.normal(size=(1, 68))
    nsp = NiSpace(x=x, y=y, z=["gm", "not_a_shortcut"], parcellation="DesikanKilliany",
                  verbose=False)
    with pytest.raises(ValueError, match="mixes named axes"):
        nsp.fit()


def test_fast_path_rows_follow_requested_order(rng):
    """Regression: fetch_reference returns TPM maps in library order (gm, wm, ...), and the
    rows used to be labelled positionally with the requested names -- z=["wm", "gm"] gave a
    'wm' row holding grey matter."""
    x, y = rng.normal(size=(1, 68)), rng.normal(size=(1, 68))
    nsp_a = NiSpace(x=x, y=y, z=["wm", "gm"], parcellation="DesikanKilliany",
                    standardize=False, verbose=False).fit()
    nsp_b = NiSpace(x=x, y=y, z=["gm", "wm"], parcellation="DesikanKilliany",
                    standardize=False, verbose=False).fit()
    za, zb = nsp_a.get_z(), nsp_b.get_z()
    assert list(za.index) == ["wm", "gm"]
    np.testing.assert_allclose(za.loc["gm"].values, zb.loc["gm"].values)
    np.testing.assert_allclose(za.loc["wm"].values, zb.loc["wm"].values)


def test_fast_path_combined_parcellation_no_deprecated_name(rng, monkeypatch):
    """Combined (cx+sc) parcellations take the fast path too. The integrated-name lookup
    must pass the (cx, sc) parts as a tuple -- concatenating them ("Yan200TianS2") still
    resolved, but through the deprecated-naming branch of _check_parcellation(), which
    logged a deprecation warning on every z="gm" fit."""
    calls = _spy_fetch_reference(monkeypatch)
    warnings = []
    real_warning = ds_mod.lgr.warning
    monkeypatch.setattr(ds_mod.lgr, "warning",
                        lambda msg, *a, **k: (warnings.append(str(msg)), real_warning(msg, *a, **k)))

    expected = fetch_reference("tpm", maps=["gm"], parcellation="Yan200+TianS2",
                               standardize_parcellated=False, print_references=False,
                               verbose=False)
    n = expected.shape[1]
    nsp = NiSpace(x=rng.normal(size=(1, n)), y=rng.normal(size=(1, n)), z="gm",
                  parcellation="Yan200+TianS2", standardize=False, verbose=False)
    nsp.fit()

    assert len(calls) == 1
    assert calls[0].get("parcellation") is not None   # fast path, not the raw-image fallback
    assert not any("deprecated naming" in w for w in warnings)
    np.testing.assert_allclose(nsp.get_z().values, expected.values)
