"""Sanity checks for a real, fully-fitted Parcellation object (built the
same way NiSpace.fit() builds one internally) -- labels, distance matrix,
and null-space resolution against real geometry. The synthetic suite
(test_parcellation_null.py) only exercises get_null_space() against mock
objects with no real image data; this batch's job is the real thing.
"""

import numpy as np
import pandas as pd

from nispace import NiSpace
from nispace.datasets import fetch_reference


def _real_parcellation():
    x_df = fetch_reference("rsn", parcellation="Yan100", print_references=False, verbose=False)
    nsp = NiSpace(x=x_df, y=None, parcellation="Yan100", verbose=False,
                  n_proc=1, return_self=False)
    nsp.fit()
    return nsp._parc


def test_real_parcellation_has_expected_label_count():
    parc = _real_parcellation()
    assert len(parc._labels) == 100


def test_real_parcellation_dist_mat_is_symmetric_and_nonnegative():
    parc = _real_parcellation()
    dm = parc.get_dist_mat(space="MNI152NLin6Asym")
    assert dm.shape == (100, 100)
    assert np.allclose(dm, dm.T)
    assert (np.asarray(dm) >= 0).all()
    assert np.allclose(np.diag(dm), 0)


def test_real_parcellation_null_space_resolves():
    parc = _real_parcellation()
    space, method = parc.get_null_space()
    assert isinstance(space, str) and len(space) > 0
    assert isinstance(method, str) and len(method) > 0


def test_fit_does_not_eagerly_load_spin_mat_by_default():
    # regression: NiSpace(load_spin_mat=True) (the old default) made
    # Parcellation.from_nispace_library() eagerly decompress spin-rotation files for
    # EVERY available space at fit() time -- for a 1000-parcel cortical atlas this is
    # ~5GB per hemisphere, ~10GB+ per space, several spaces -> tens of GB, even when
    # the null method never uses spin tests at all (e.g. moran/random). Found via a
    # real ~25GB memory report for a moran-only workflow that just happened to use an
    # integrated parcellation. Fixed by flipping NiSpace's own load_spin_mat default
    # to False; permute() already fetches spin_mat lazily, on demand, only when a spin
    # method is actually requested (Parcellation.get_spin_mat() -> lazy per-space).
    # Verify laziness directly: right after fit(), the spin-mat slot for a space with
    # real spin data available must still be an unresolved lazy spec (a tuple of
    # path/spec dicts), not the actual decompressed array data.
    x_df = fetch_reference("rsn", parcellation="Schaefer100", print_references=False,
                           verbose=False)
    nsp = NiSpace(x=x_df, y=None, parcellation="Schaefer100", verbose=False,
                  n_proc=1, return_self=False)
    nsp.fit()
    parc = nsp._parc
    space = "fsaverage" if "fsaverage" in parc._spin_mats else next(iter(parc._spin_mats))
    sm = parc._spin_mats[space]
    assert sm is not None, f"expected a registered spin-mat spec for '{space}'"
    assert isinstance(sm, tuple) and isinstance(sm[0], dict), (
        "spin_mat was eagerly resolved at fit() time instead of staying a lazy spec"
    )
    # and the lazy spec resolves correctly on first real access (permute()'s own path)
    resolved = parc.get_spin_mat(space=space)
    assert resolved is not None
    assert not (isinstance(resolved[0], dict))
