"""Smoke tests for helpers/nimare.py against a real (small, NiMARE-bundled)
coordinate dataset -- the classic 21-study NIDM-Pain dataset shipped inside
the nimare package itself (nimare.utils.get_resource_path()), so no
network/data-repo access is needed beyond what NiSpace's own Yan100
parcellation fetch already requires. Needs the optional `nimare` dependency
(installed via pip install nispace[opt], see pyproject.toml).

Well-formedness only (small n_perm/n_iters to keep runtime low, not
statistically meaningful FWE results) -- null_maps_from_nimare()'s own
`validate=True` path does real ground-truth-style checks (empirical vs.
NiMARE parametric p-values / null cluster threshold agreement) but needs
much larger n_perm/n_iters than is practical here to be non-noisy, so this
suite disables it and just proves the pipeline runs end-to-end and returns
well-shaped output.
"""

import os

import numpy as np
import pytest

nimare = pytest.importorskip("nimare")

from nimare.dataset import Dataset
from nimare.meta.cbma.ale import ALE
from nimare.correct import FWECorrector
from nimare.utils import get_resource_path

from nispace.helpers.nimare import null_maps_from_nimare, nimare_fwe_thresholds, get_binary_cluster_map


@pytest.fixture(scope="module")
def ale_result():
    dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
    dset = Dataset(dset_file)
    return ALE().fit(dset)


@pytest.fixture(scope="module")
def ale_corrected_result(ale_result):
    corr = FWECorrector(method="montecarlo", n_iters=10, n_cores=1, voxel_thresh=0.001)
    return corr.transform(ale_result)


def test_null_maps_from_nimare_continuous_mode(ale_result):
    nulls = null_maps_from_nimare(
        ale_result, parcellation="Yan100", n_perm=5, seed=0, n_proc=1,
        validate=False, verbose=False,
    )
    assert isinstance(nulls, dict)
    assert "stat" in nulls
    assert nulls["stat"].shape == (5, 100)
    assert np.isfinite(nulls["stat"]).all()


def test_nimare_fwe_thresholds_returns_voxel_and_cluster_thresholds(ale_corrected_result):
    voxel_thr, cluster_thr = nimare_fwe_thresholds(
        ale_corrected_result, alpha=0.05, cluster_stat="size", voxel_thresh=0.001
    )
    assert voxel_thr == pytest.approx(3.0902, abs=1e-3)  # z-threshold for p<0.001, deterministic
    assert isinstance(cluster_thr, (int, np.integer))


def test_get_binary_cluster_map_returns_nifti_image(ale_corrected_result):
    import nibabel as nib
    img = get_binary_cluster_map(ale_corrected_result, cluster_stat="size", alpha=0.05)
    assert isinstance(img, nib.Nifti1Image)
    vals = np.unique(np.asarray(img.dataobj))
    assert set(vals.tolist()).issubset({0.0, 1.0})


def test_null_maps_from_nimare_binary_cluster_mode(ale_result, ale_corrected_result):
    nulls = null_maps_from_nimare(
        ale_result, parcellation="Yan100", corrected_result=ale_corrected_result,
        n_perm=5, seed=0, n_proc=1, validate=False, verbose=False,
    )
    assert isinstance(nulls, dict)
    stat_key = next(iter(nulls))
    arr = nulls[stat_key]
    assert arr.shape == (5, 100)
    # binary cluster-coverage nulls must stay within [0, 1]
    assert (arr >= 0).all() and (arr <= 1).all()
