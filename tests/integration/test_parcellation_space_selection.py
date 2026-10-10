"""Regression: NiSpace.fit() must parcellate raw MNI image data with the
parcellation image of the requested template, for single and combined ("+")
integrated parcellations. Library entries list MNI152NLin2009cAsym first, so
the old first-loose-match logic activated the 2009c image for 6Asym data.
Synthetic counterpart: tests/test_parcellation_space_match.py.
"""

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from nispace import NiSpace
from nispace.datasets import fetch_parcellation

A6, A9 = "MNI152NLin6Asym", "MNI152NLin2009cAsym"
PARCS = ["DesikanKilliany", "Yan100+TianS1"]


@pytest.mark.parametrize("parc", PARCS)
def test_library_lists_2009c_first(parc):
    # precondition that made the bug visible; if this changes, the fit() tests below
    # still hold but no longer guard against order dependence
    p = fetch_parcellation(parc, verbose=False)
    mni = [s for s in p.spaces if "mni" in s.lower()]
    assert mni[:2] == [A9, A6]


@pytest.mark.parametrize("parc", PARCS)
@pytest.mark.parametrize("space", [A6, A9])
def test_get_image_for_dataspace_real(parc, space):
    p = fetch_parcellation(parc, verbose=False)
    assert p.get_image_for_dataspace(space) == space


def _raw_image(parc, space):
    """Float image on the parcellation's own grid -- raw image input for fit()."""
    p = fetch_parcellation(parc, verbose=False)
    p.set_active_space(space)
    img = p._image_obj
    return nib.Nifti1Image(np.asarray(img.get_fdata(), dtype=np.float32), img.affine)


@pytest.mark.parametrize("parc", PARCS)
@pytest.mark.parametrize("space", [A6, A9])
def test_fit_activates_requested_space(parc, space):
    y = _raw_image(parc, space)
    n = len(fetch_parcellation(parc, verbose=False)._labels)
    x = pd.DataFrame(np.random.default_rng(0).normal(size=(2, n)))
    nsp = NiSpace(x=x, y=y, parcellation=parc, data_space=space, parcellation_space=space,
                  verbose=False, n_proc=1)
    nsp.fit()
    assert nsp._parc._space == space


@pytest.mark.parametrize("parc", PARCS)
def test_fit_default_space_is_6asym(parc):
    y = _raw_image(parc, A6)
    n = len(fetch_parcellation(parc, verbose=False)._labels)
    x = pd.DataFrame(np.random.default_rng(0).normal(size=(2, n)))
    nsp = NiSpace(x=x, y=y, parcellation=parc, verbose=False, n_proc=1)
    nsp.fit()
    assert nsp._parc._space == A6
