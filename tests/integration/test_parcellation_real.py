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
