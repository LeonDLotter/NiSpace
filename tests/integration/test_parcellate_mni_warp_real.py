"""Parcellation across specific MNI spaces on real data: the parcellation is warped (EasyReg) into the data
space instead of being only resampled onto its grid."""

from pathlib import Path

import numpy as np
import pytest

from nispace.datasets import fetch_parcellation, fetch_reference
from nispace.io import parcellate_data


@pytest.fixture(scope="module")
def pet_2009c():
    return fetch_reference("pet", maps="5HTT", space="MNI152NLin2009cAsym", print_references=False,
                           verbose=False)[:1]


def _parc(space):
    parc = fetch_parcellation("Schaefer200Parcels7Networks", return_parcellation_only=True, verbose=False)
    parc.set_active_space(space)
    return parc


def test_warped_parcellation_matches_native_space_parcellation(pet_2009c):
    kw = dict(data_space="MNI152NLin2009cAsym", verbose=False)
    native = parcellate_data(pet_2009c, parcellation=_parc("MNI152NLin2009cAsym"), **kw).values.ravel()
    warped = parcellate_data(pet_2009c, parcellation=_parc("MNI152NLin6Asym"), **kw).values.ravel()
    # old behaviour: NLin6 parcellation only resampled onto the 2009c grid (generic data space)
    unwarped = parcellate_data(pet_2009c, parcellation=_parc("MNI152NLin6Asym"), data_space="MNI152",
                               verbose=False).values.ravel()
    ok = np.isfinite(native) & np.isfinite(warped) & np.isfinite(unwarped)
    err_warped = np.abs(warped - native)[ok].mean()
    err_unwarped = np.abs(unwarped - native)[ok].mean()
    assert np.corrcoef(warped[ok], native[ok])[0, 1] > 0.99
    assert err_warped < err_unwarped
