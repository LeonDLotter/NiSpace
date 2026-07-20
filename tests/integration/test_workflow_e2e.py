"""Real end-to-end workflow smoke test: `NiSpace(x, y, z, parcellation).fit()`
-> `.colocalize()` -> `.get_colocalizations()`, using real fetched reference
network maps (not synthetic arrays, not a pre-parcellated DataFrame -- raw
NIfTI images, exercising the actual image-fetch + real-geometry-resample +
parcellate pipeline). This closes the biggest previously-flagged integration
gap: every other workflow test (tests/test_workflows.py) uses a synthetic
pre-fitted `nispace_object=`, never a real `.fit()` from raw images. See
[[project_testing_batches_plan]].

Both the `regress_z=False` (plain Pearson) and `regress_z=True` (Z regressed
out of X and Y before correlating -- the `colocalize()` default) paths are
ground-truthed against independent hand-computed values (`np.corrcoef` /
manual OLS residualization), not just checked for well-formedness.
"""

import numpy as np
import pytest

from nispace import NiSpace
from nispace.datasets import fetch_reference


@pytest.fixture(scope="module")
def real_nsp():
    x = fetch_reference("rsn", maps=["Auditory", "DefaultMode", "Visual"],
                        print_references=False, verbose=False)
    y = fetch_reference("rsn", maps=["Frontoparietal", "Salience", "DorsalAttention"],
                        print_references=False, verbose=False)
    nsp = NiSpace(x=x, y=y, z="gm", parcellation="Yan100", verbose=False,
                  n_proc=1, return_self=True, seed=42)
    nsp.fit()
    return nsp


def test_fit_parcellates_real_images_to_expected_shape(real_nsp):
    X, Y = real_nsp.get_x(), real_nsp.get_y()
    assert X.shape == (3, 100)
    assert Y.shape == (3, 100)
    assert not X.isna().any().any()
    assert not Y.isna().any().any()


def test_colocalize_pearson_no_z_matches_manual_corrcoef(real_nsp):
    X, Y = real_nsp.get_x(), real_nsp.get_y()
    real_nsp.colocalize(method="pearson", r_to_z=False, regress_z=False)
    coloc = real_nsp.get_colocalizations()

    assert coloc.shape == (3, 3)
    assert list(coloc.columns) == list(X.index)
    assert list(coloc.index) == list(Y.index)

    expected = np.array([
        [np.corrcoef(X.iloc[j].values, Y.iloc[i].values)[0, 1] for j in range(3)]
        for i in range(3)
    ])
    np.testing.assert_allclose(coloc.values, expected, atol=1e-5)


def test_colocalize_pearson_regress_z_matches_manual_residualization(real_nsp):
    """regress_z=True is colocalize()'s own default -- ground-truth it against
    an independent OLS-residualization of X and Y on Z (the real 'gm' TPM map),
    not just a well-formedness check."""
    X, Y, Z = real_nsp.get_x(), real_nsp.get_y(), real_nsp._Z
    z = Z.iloc[0].values.astype(float)

    def resid(v):
        design = np.column_stack([np.ones_like(z), z])
        beta, *_ = np.linalg.lstsq(design, v.astype(float), rcond=None)
        return v - design @ beta

    Xr = np.array([resid(X.iloc[j].values) for j in range(X.shape[0])])
    Yr = np.array([resid(Y.iloc[i].values) for i in range(Y.shape[0])])
    expected = np.array([
        [np.corrcoef(Xr[j], Yr[i])[0, 1] for j in range(3)]
        for i in range(3)
    ])

    real_nsp.colocalize(method="pearson", r_to_z=False, regress_z=True)
    coloc = real_nsp.get_colocalizations()
    np.testing.assert_allclose(coloc.values, expected, atol=1e-4)


def test_permute_and_correct_p_run_on_real_fitted_pipeline(real_nsp):
    real_nsp.colocalize(method="pearson", r_to_z=True, regress_z=False)
    real_nsp.permute(what="maps", n_perm=50, seed=42)
    p = real_nsp.get_p_values()
    assert np.isfinite(p.values).all()
    assert (p.values >= 0).all() and (p.values <= 1).all()
