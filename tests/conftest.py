"""Shared pytest fixtures.

These are written to be generic and reusable across the whole test suite,
not specific to any one feature -- e.g. any future test of colocalize()/
permute() can reuse ``toy_regression``/``synthetic_nispace`` as-is.

Unit vs. integration
---------------------
Plain tests (no marker) are the default tier: fast, offline, no dependency
on real parcellation/reference data -- built on ``synthetic_nispace`` or
pure-function inputs. This is what runs in CI and on a bare ``pytest``.

Tests that need real data (``~/projects/nispace-data``) and/or network
access -- e.g. exercising ``fetch_*``, null generation on real geometry,
or a full pipeline run against the ``docs/nb_introduction`` pickles --
should be marked ``@pytest.mark.integration``. These are excluded by
default (see ``addopts`` in pyproject.toml) and must be run explicitly:
``pytest -m integration``.
"""

import matplotlib
matplotlib.use("Agg")  # headless: must be set before pyplot is imported anywhere,
                        # so plot=True tests don't pop real windows (interactive
                        # backends, e.g. macosx, are the default outside CI)

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session", autouse=True)
def _skip_nilearn_forced_gc():
    """nilearn's ``safe_get_data`` forces ``gc.collect()`` on every call (to free memory
    before copying image data). In a long test session the heap is large, so each call
    costs ~0.1 s and plotting/parcellation tests make dozens of them. Tests don't need
    the memory back, so stub out that one call site."""
    import types
    try:
        from nilearn._utils import niimg
    except ImportError:
        yield
        return
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(niimg, "gc", types.SimpleNamespace(collect=lambda *a, **k: 0))
        yield


@pytest.fixture
def rng():
    """Fixed-seed numpy Generator for reproducible synthetic data across tests."""
    return np.random.default_rng(42)


def make_toy_regression(rng, n_parcels=30, n_x=3, n_y=1, noise=0.3,
                        outlier_idx=0, outlier_scale=8.0):
    """Synthetic linear regression data: y = w @ X + noise, with one
    deliberately injected outlier parcel (for ground-truth sanity checks).

    Returns
    -------
    X : ndarray (n_x, n_parcels) -- predictor maps
    Y : ndarray (n_y, n_parcels) -- target map(s)
    w : ndarray (n_x,) -- true weights used to generate Y
    outlier_idx : int -- index of the deliberately injected outlier parcel
    """
    X = rng.normal(size=(n_x, n_parcels))
    w = rng.normal(size=n_x)
    Y = np.zeros((n_y, n_parcels))
    for i in range(n_y):
        Y[i] = w @ X + rng.normal(scale=noise, size=n_parcels)
    Y[:, outlier_idx] += outlier_scale
    return X, Y, w, outlier_idx


@pytest.fixture
def toy_regression(rng):
    """Default-sized synthetic regression fixture (n_parcels=30, n_x=3, one outlier)."""
    return make_toy_regression(rng)


def make_fake_parcellation(vox=6.0, n_y=4, n_z=3):
    """Small synthetic volumetric parcellation on a coarse MNI152NLin6Asym-like grid.

    An ellipsoid "brain" (31x37x31 voxels at 6 mm) split into left/right hemispheres
    (midline column left unlabelled) and each hemisphere into ``n_y * n_z`` blocks along
    y and z: 24 parcels by default, labelled ``hemi-L_p{i}``/``hemi-R_p{i}`` so
    ``make_bilateral()``/``select_hemi()`` work. Parcel ``i`` has image value ``i + 1``.

    Fully offline: with a precomputed ``dist_mat`` (see ``fake_parcellation``) a
    ``Parcellation`` built from this image goes through ``fit()`` on raw images,
    colocalization, Moran nulls and glass-brain plotting without touching the data cache.
    Keep ~20+ parcels for anything that permutes: with fewer, null correlations hit the
    ``r_equal_one`` guard.

    Returns
    -------
    img : nib.Nifti1Image (int16 labels)
    labels : list of str
    grid : dict with ``X``, ``Y``, ``Z`` (mm coordinates per voxel), ``brain`` (bool mask),
        ``affine``, ``dist_mat`` (parcel-centroid Euclidean distances, float32)
    """
    import nibabel as nib

    x = np.arange(-90, 91, vox)
    y = np.arange(-126, 91, vox)
    z = np.arange(-72, 109, vox)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    brain = (X / 70) ** 2 + ((Y + 18) / 100) ** 2 + ((Z - 18) / 75) ** 2 <= 1
    y_block = np.digitize(Y, np.quantile(Y[brain], np.linspace(0, 1, n_y + 1)[1:-1]))
    z_block = np.digitize(Z, np.quantile(Z[brain], np.linspace(0, 1, n_z + 1)[1:-1]))
    block = y_block * n_z + z_block
    n_hemi = n_y * n_z
    lab = np.where(X < 0, 1 + block, 1 + n_hemi + block)
    lab[~brain | (np.abs(X) < vox / 2)] = 0
    affine = np.diag([vox, vox, vox, 1.0])
    affine[:3, 3] = [x[0], y[0], z[0]]
    labels = [f"hemi-L_p{i}" for i in range(n_hemi)] + [f"hemi-R_p{i}" for i in range(n_hemi)]

    xyz = np.stack([X, Y, Z], axis=-1)
    centroids = np.array([xyz[lab == i + 1].mean(0) for i in range(len(labels))])
    dist_mat = np.sqrt(((centroids[:, None] - centroids[None]) ** 2).sum(-1)).astype(np.float32)

    img = nib.Nifti1Image(lab.astype(np.int16), affine)
    grid = dict(X=X, Y=Y, Z=Z, brain=brain, affine=affine, dist_mat=dist_mat)
    return img, labels, grid


def make_fake_maps(grid, n, seed=0, noise=0.1, prefix=None):
    """``n`` spatially smooth float32 NIfTI maps on the fake parcellation's grid (one
    Gaussian blob each, plus noise; zero outside the brain)."""
    import nibabel as nib

    rng = np.random.default_rng(seed)
    X, Y, Z, brain = grid["X"], grid["Y"], grid["Z"], grid["brain"]
    maps = []
    for _ in range(n):
        c = rng.uniform(-50, 50, 3)
        d = np.exp(-((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) / (2 * 40 ** 2))
        d = d + noise * rng.normal(size=X.shape)
        d[~brain] = 0
        maps.append(nib.Nifti1Image(d.astype(np.float32), grid["affine"]))
    return maps


@pytest.fixture
def fake_parc_data():
    """``(img, labels, grid)`` from :func:`make_fake_parcellation`."""
    return make_fake_parcellation()


@pytest.fixture
def fake_maps(fake_parc_data):
    """Factory ``fake_maps(n, seed=0, **kw)``: :func:`make_fake_maps` on the fake grid."""
    grid = fake_parc_data[2]
    return lambda n, seed=0, **kw: make_fake_maps(grid, n, seed=seed, **kw)


@pytest.fixture
def fake_parcellation(fake_parc_data):
    """A fresh (per test -- NiSpace may mutate it) ``Parcellation`` built from the fake
    image, in space ``MNI152NLin6Asym``, with a precomputed centroid distance matrix
    (computing it from the image takes minutes even for 24 parcels)."""
    from nispace.core.parcellation import Parcellation

    img, labels, grid = fake_parc_data
    return Parcellation.from_path(img, space="MNI152NLin6Asym", labels=labels,
                                  dist_mat=grid["dist_mat"], symmetric=True,
                                  name="fake", verbose=False)


@pytest.fixture
def synthetic_nispace(toy_regression):
    """A fully synthetic, network-free, fitted NiSpace object.

    ``parcellation=None`` with DataFrame x/y input bypasses all parcellation
    machinery (NiSpace.fit() only touches it when self._parc is not None,
    and io.parcellate_data() passes DataFrames/ndarrays through as
    already-parcellated) -- no real Parcellation object, no network or
    data-repo access needed.
    """
    from nispace import NiSpace

    X, Y, w, outlier_idx = toy_regression
    n_x, n_parcels = X.shape
    n_y = Y.shape[0]

    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    x_df = pd.DataFrame(X, index=[f"x{i}" for i in range(n_x)], columns=parcel_labels)
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(n_y)], columns=parcel_labels)

    nsp = NiSpace(
        x=x_df, y=y_df, z=None,
        parcellation=None, standardize=False,
        n_proc=1, verbose=False,
        return_self=False,
    )
    nsp.fit()
    return nsp
