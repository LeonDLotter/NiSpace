"""Smoke test for transforms.py::mni_to_mni() against a real MNI template
and the real EasyReg deformation fields shipped in the data repo. Needs the
optional `nitransforms` dependency (installed via `pip install
nispace[opt]`, see pyproject.toml) plus real data/network access.

All 4 supported resolutions (1/2/3/4mm) are covered. res=3/4mm previously
raised an IndexError inside nitransforms' DenseFieldTransform.map() ("index
N is out of bounds for axis 0 with size 182"), reproduced on both
nitransforms 24.1.4 and 25.1.0 -- traced to the affines.json-stored
coarse-resolution reference grids extending slightly beyond the
deformation field's native FOV. Fixed upstream in the data repo (confirmed
2026-07-17); see [[project_data_repo_ci]].
"""

import numpy as np
import nibabel as nib
import pytest

pytest.importorskip("nitransforms")

from nispace.datasets import fetch_template
from nispace.transforms import mni_to_mni


@pytest.mark.parametrize("res", [1, 2, 3, 4])
def test_mni_to_mni_real_field_smoke(res):
    img = fetch_template("MNI152NLin2009cAsym", res="2mm", verbose=False)
    out = mni_to_mni(img, mni_from="MNI152NLin2009cAsym", mni_to="MNI152NLin6Asym",
                     order=1, res=res, verbose=False)
    assert isinstance(out, nib.Nifti1Image)
    assert np.isfinite(out.get_fdata()).any()


def test_mni_to_mni_identity_shortcircuits_without_nitransforms_call():
    img = fetch_template("MNI152NLin2009cAsym", res="2mm", verbose=False)
    out = mni_to_mni(img, mni_from="MNI152NLin2009cAsym", mni_to="MNI152NLin2009cAsym",
                     verbose=False)
    assert out is img
