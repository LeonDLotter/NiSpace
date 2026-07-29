"""Coverage for the ``z="gm"``/``"wm"``/``"csf"``/``"veins"``/``"arteries"``
TPM auto-fetch shortcut in ``NiSpace.fit()`` (api.py, the "data to control
correlations for" block). Two code paths exist:

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

import nispace.api as api_mod
from nispace import NiSpace
from nispace.datasets import fetch_reference, fetch_parcellation


def _spy_fetch_reference(monkeypatch):
    """Wrap api_mod.fetch_reference to record call kwargs while still
    delegating to the real function -- lets tests assert which code path
    (fast vs. fallback) was actually taken, not just infer it from output."""
    calls = []
    real = api_mod.fetch_reference

    def wrapper(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(api_mod, "fetch_reference", wrapper)
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


def test_mixed_shortcut_list_not_intercepted(rng):
    """A list is only intercepted as TPM shortcuts if EVERY element matches --
    one non-shortcut entry falls through untouched to parcellate_data, which
    then tries (and fails) to load it as an image path."""
    x, y = rng.normal(size=(1, 68)), rng.normal(size=(1, 68))
    nsp = NiSpace(x=x, y=y, z=["gm", "not_a_shortcut"], parcellation="DesikanKilliany",
                  verbose=False)
    with pytest.raises(FileNotFoundError):
        nsp.fit()
