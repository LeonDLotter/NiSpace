"""Full image-based pipeline on the fake parcellation (conftest.py), fully offline.

The rest of the unit tier feeds pre-parcellated DataFrames with ``parcellation=None``, so
raw-image parcellation through a ``Parcellation`` object, distance-matrix-based (Moran)
nulls, hemisphere selection/bilateralization and a workflow started from raw images were
only reachable in the integration tier. The fake is a 24-parcel, 6 mm MNI-grid volume:
every step runs in milliseconds and needs no data cache.
"""

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from nispace import NiSpace, workflows
from nispace.core.parcellation import Parcellation

N_PARCELS = 24
N_HEMI = 12


@pytest.fixture
def lab(fake_parc_data):
    img, _, _ = fake_parc_data
    return np.asarray(img.dataobj)


@pytest.fixture
def grid(fake_parc_data):
    return fake_parc_data[2]


def _parcel_means(img, lab, label_values):
    """Ground truth: plain numpy mean over each parcel's voxels (union for a list)."""
    data = img.get_fdata()
    return np.array([data[np.isin(lab, np.atleast_1d(v))].mean() for v in label_values])


def _near_copy(img, grid, scale=0.05, seed=5):
    """`img` plus a little noise inside the brain: strongly but not perfectly correlated
    (an exact copy trips the r_equal_one guard)."""
    noise = np.random.default_rng(seed).normal(scale=scale, size=img.shape) * grid["brain"]
    return nib.Nifti1Image((img.get_fdata() + noise).astype(np.float32), grid["affine"])


# --- fit() on raw images ---

def test_fit_raw_images_equals_numpy_parcel_means(fake_parcellation, fake_parc_data, lab, grid, fake_maps):
    _, labels, _ = fake_parc_data
    xs = fake_maps(3, seed=0)
    nsp = NiSpace(x=xs, y=xs[:1], parcellation=fake_parcellation, standardize=False,
                  verbose=False, n_proc=1).fit()
    X = nsp.get_x()
    assert X.shape == (3, N_PARCELS)
    assert list(X.columns) == labels
    for k, img in enumerate(xs):
        np.testing.assert_allclose(X.iloc[k].values, _parcel_means(img, lab, range(1, 25)),
                                   atol=1e-6)


def test_fit_uses_provided_dist_mat(fake_parcellation, grid, fake_maps):
    xs = fake_maps(2, seed=0)
    nsp = NiSpace(x=xs, y=xs[:1], parcellation=fake_parcellation, verbose=False).fit()
    np.testing.assert_array_equal(nsp._parc.get_dist_mat(), grid["dist_mat"])


def test_default_null_is_moran_in_volume_space(fake_parcellation):
    assert fake_parcellation.get_null_space() == ("MNI152NLin6Asym", "moran")


# --- Moran nulls on the fake distance matrix ---

def _fitted(fake_parcellation, grid, fake_maps):
    xs = fake_maps(3, seed=0)
    ys = [_near_copy(xs[0], grid)]
    nsp = NiSpace(x=xs, y=ys, parcellation=fake_parcellation, verbose=False, n_proc=1).fit()
    nsp.colocalize("spearman")
    return nsp


def test_moran_permutation_detects_near_copy(fake_parcellation, grid, fake_maps):
    nsp = _fitted(fake_parcellation, grid, fake_maps)
    n_perm = 200
    nsp.permute("maps", n_perm=n_perm, seed=3)   # default null for this parcellation: moran
    p = nsp.get_p_values("spearman").iloc[0]
    assert ((p > 0) & (p <= 1)).all()
    # Y is X0 plus noise: no null map should beat it -> smallest possible p. Bound only:
    # whether the floor is 1/n_perm or 1/(n_perm+1) is an open question (null_to_p).
    assert p.iloc[0] <= 1 / n_perm + 1e-6
    assert p.iloc[0] < p.iloc[1:].min()


def test_moran_permutation_reproducible_with_seed(fake_parc_data, grid, fake_maps):
    img, labels, _ = fake_parc_data

    def run(seed):
        parc = Parcellation.from_path(img, space="MNI152NLin6Asym", labels=labels,
                                      dist_mat=grid["dist_mat"], name="fake", verbose=False)
        nsp = _fitted(parc, grid, fake_maps)
        nsp.permute("maps", n_perm=50, seed=seed)
        return nsp.get_p_values("spearman")

    pd.testing.assert_frame_equal(run(1), run(1))
    assert not run(1).equals(run(2))


# --- hemisphere handling ---

@pytest.mark.parametrize("hemi,offset", [("L", 0), ("R", N_HEMI)])
def test_select_hemi(fake_parcellation, grid, lab, hemi, offset):
    fake_parcellation.select_hemi(hemi, verbose=False)
    assert list(fake_parcellation._labels) == [f"hemi-{hemi}_p{i}" for i in range(N_HEMI)]
    np.testing.assert_array_equal(fake_parcellation.get_dist_mat(),
                                  grid["dist_mat"][offset:offset + N_HEMI, offset:offset + N_HEMI])
    kept = np.unique(np.asarray(fake_parcellation.get_image().dataobj))
    np.testing.assert_array_equal(kept, np.r_[0, offset + 1:offset + N_HEMI + 1])


def test_make_bilateral_pools_hemispheres(fake_parcellation, grid, lab, fake_maps):
    fake_parcellation.make_bilateral()
    assert list(fake_parcellation._labels) == [f"p{i}" for i in range(N_HEMI)]
    np.testing.assert_array_equal(fake_parcellation.get_dist_mat(),
                                  grid["dist_mat"][:N_HEMI, :N_HEMI])
    # right-hemisphere voxels now carry their left homologue's value
    bil = np.asarray(fake_parcellation.get_image().dataobj)
    np.testing.assert_array_equal(bil[lab > N_HEMI], lab[lab > N_HEMI] - N_HEMI)
    np.testing.assert_array_equal(bil[(lab > 0) & (lab <= N_HEMI)], lab[(lab > 0) & (lab <= N_HEMI)])
    # and fit() averages over both hemispheres' voxels
    xs = fake_maps(2, seed=0)
    nsp = NiSpace(x=xs, y=xs[:1], parcellation=fake_parcellation, standardize=False,
                  verbose=False).fit()
    for k, img in enumerate(xs):
        expected = _parcel_means(img, lab, [[i + 1, i + 1 + N_HEMI] for i in range(N_HEMI)])
        np.testing.assert_allclose(nsp.get_x().iloc[k].values, expected, atol=1e-6)


# --- workflow from raw images (not a pre-fitted nispace_object) ---

def test_colocalization_workflow_from_raw_images(fake_parc_data, grid, fake_maps):
    img, labels, _ = fake_parc_data

    def parc():
        return Parcellation.from_path(img, space="MNI152NLin6Asym", labels=labels,
                                      dist_mat=grid["dist_mat"], name="fake", verbose=False)

    xs = fake_maps(3, seed=0)
    ys = fake_maps(2, seed=1)
    colocs, p, p_corr, nsp = workflows.colocalization(
        y=ys, x=xs, parcellation=parc(), colocalization_method="spearman", n_perm=100,
        plot=False, verbose=False, seed=0,
    )
    assert colocs.shape == p.shape == p_corr.shape == (2, 3)
    assert ((p > 0) & (p <= 1)).all().all()
    assert (p_corr.values >= p.values - 1e-12).all()
    # same colocalizations as the step-by-step API path
    manual = NiSpace(x=xs, y=ys, parcellation=parc(), verbose=False).fit()
    manual.colocalize("spearman")
    np.testing.assert_allclose(colocs.values, manual.get_colocalizations("spearman").values)
