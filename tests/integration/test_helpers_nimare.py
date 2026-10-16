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


def _assert_nulls_random(arr):
    # random foci -> no two nulls identical and near-zero mean inter-null correlation
    # (identity to the observed map is checked inside null_maps_from_nimare itself)
    assert not any(np.allclose(arr[i], arr[j]) for i in range(len(arr)) for j in range(i))
    r = np.corrcoef(arr)[np.triu_indices(len(arr), 1)]
    assert np.nanmean(r) < 0.3


def test_null_maps_from_nimare_nulls_are_random(ale_result):
    # Regression: NiMARE >= 0.20 reuses stale i/j/k columns from inputs_["coordinates"]
    # unless they are overwritten, which made every null equal to the observed map.
    nulls = null_maps_from_nimare(
        ale_result, parcellation="Yan100", n_perm=5, seed=0, n_proc=1,
        validate=False, verbose=False,
    )["stat"]
    _assert_nulls_random(nulls)


def test_null_maps_from_nimare_nulls_are_random_studyset():
    # Studyset input (NIMADS) -- the path the original bug report used
    try:
        from nimare.nimads import Studyset
        rng = np.random.default_rng(0)
        studies = [{
            "id": f"s{i}", "name": f"s{i}",
            "analyses": [{
                "id": "a", "name": "a",
                "points": [{"coordinates": rng.uniform([-60, -90, -30], [60, 60, 70]).tolist(),
                            "space": "MNI"} for _ in range(5)],
                "metadata": {"sample_size": 30},
            }],
        } for i in range(10)]
        result = ALE().fit(Studyset({"id": "x", "name": "x", "studies": studies}))
    except (ImportError, KeyError):
        pytest.skip("NiMARE version cannot fit an estimator on a Studyset")
    nulls = null_maps_from_nimare(
        result, parcellation="Yan100", n_perm=5, seed=1, n_proc=1,
        validate=False, verbose=False,
    )["stat"]
    _assert_nulls_random(nulls)


def test_nimare_colocalization_inherits_n_perm_from_nimare_nulls(ale_result):
    import pandas as pd
    from nispace.workflows import nimare_colocalization
    nulls = null_maps_from_nimare(
        ale_result, parcellation="Yan100", n_perm=7, seed=0, n_proc=1,
        map_label="pain", validate=False, verbose=False,
    )
    x = pd.DataFrame(np.random.default_rng(0).normal(size=(3, 100)),
                     index=["x0", "x1", "x2"])   # pre-parcellated toy X (Yan100)
    nsp = nimare_colocalization(
        y=ale_result.get_map("stat"), x=x, parcellation="Yan100",
        nimare_nulls=nulls, seed=0, plot=False, verbose=False,
        return_nispace_only=True,   # single Y image -> label 0, auto-matched to "pain"
    )
    _, null_colocs = nsp.get_colocalizations(
        get_nulls=True, nulls_permute_what="ymaps", verbose=False
    )
    assert next(iter(null_colocs.values())).shape[1] == 7


def test_check_null_maps_degenerate():
    from nispace.helpers.nimare import _check_null_maps_degenerate
    rng = np.random.default_rng(0)
    good = rng.random((5, 20))
    _check_null_maps_degenerate(good, obs_parc=rng.random(20))
    # all-empty rows (binary mode, no surviving cluster) are legitimately identical
    _check_null_maps_degenerate(np.zeros((5, 20)))
    with pytest.raises(RuntimeError, match="non-empty null maps are identical"):
        _check_null_maps_degenerate(np.tile(good[0], (5, 1)))
    with pytest.raises(RuntimeError, match="identical to the observed"):
        _check_null_maps_degenerate(good, obs_parc=good[2])


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


# ---------------------------------------------------------------------------
# Reporting-baseline null (fetches the 'mabaseline' Neurosynth map over the network)
# ---------------------------------------------------------------------------

def test_null_maps_from_nimare_reporting_baseline_continuous(ale_result):
    nulls = null_maps_from_nimare(
        ale_result, parcellation="Yan100", reporting_baseline="neurosynth",
        n_perm=5, seed=0, n_proc=1, validate=False, verbose=False,
    )
    assert nulls["stat"].shape == (5, 100)
    assert np.isfinite(nulls["stat"]).all()


def test_null_maps_from_nimare_reporting_baseline_binary(ale_result, ale_corrected_result):
    nulls = null_maps_from_nimare(
        ale_result, parcellation="Yan100", corrected_result=ale_corrected_result,
        reporting_baseline="neurosynth",
        n_perm=5, seed=0, n_proc=1, validate=False, verbose=False,
    )
    arr = next(iter(nulls.values()))
    assert arr.shape == (5, 100)
    assert (arr >= 0).all() and (arr <= 1).all()


def test_null_maps_from_nimare_coordinate_pool_baseline(ale_result):
    # deterministic SALE: null foci resampled from a pool of reported coordinates
    pool = ale_result.estimator.inputs_["coordinates"][["x", "y", "z"]].values
    nulls = null_maps_from_nimare(
        ale_result, parcellation="Yan100", reporting_baseline=pool,
        n_perm=5, seed=0, n_proc=1, validate=False, verbose=False,
    )
    arr = nulls["stat"]
    assert arr.shape == (5, 100)
    assert np.isfinite(arr).all()
    # no two nulls identical, but unlike the uniform null they are strongly
    # correlated by design: every null focus lands on one of the pool's voxels
    assert not any(np.allclose(arr[i], arr[j]) for i in range(5) for j in range(i))


@pytest.mark.parametrize("as_pool", [False, True], ids=["image", "coord_pool"])
def test_reporting_baseline_concentrated_in_one_parcel_drives_nulls(ale_result, as_pool):
    # all baseline mass in one parcel -> that parcel must carry the highest mean null ALE
    import nibabel as nib
    from nilearn.image import resample_img
    from nimare.utils import vox2mm
    from nispace.datasets import fetch_parcellation

    mask_img = ale_result.estimator.masker.mask_img
    parc = fetch_parcellation("Yan100", return_parcellation_only=True, verbose=False)
    mni_spaces = [s for s in parc.spaces if "mni" in s.lower()]
    space = next((s for s in mni_spaces if "lin6" in s.lower()), mni_spaces[0])
    parc_arr = resample_img(
        parc.get_image(space), target_affine=mask_img.affine, target_shape=mask_img.shape,
        interpolation="nearest", force_resample=True, copy_header=True,
    ).get_fdata()
    labels = np.trim_zeros(np.unique(parc_arr))  # same order as the null-map columns
    in_mask = mask_img.get_fdata() != 0
    target_idx = int(np.argmax([((parc_arr == lab) & in_mask).sum() for lab in labels]))
    baseline = ((parc_arr == labels[target_idx]) & in_mask).astype(np.float32)
    if as_pool:
        # the same prior expressed as coordinates instead of a density map
        baseline = vox2mm(np.vstack(np.where(baseline)).T.astype(float), mask_img.affine)
    else:
        baseline = nib.Nifti1Image(baseline, mask_img.affine)

    nulls = null_maps_from_nimare(
        ale_result, parcellation=parc, reporting_baseline=baseline,
        n_perm=5, seed=0, n_proc=1, validate=False, verbose=False,
    )
    assert int(np.argmax(nulls["stat"].mean(axis=0))) == target_idx


@pytest.fixture(scope="module")
def scale_result():
    # small mask keeps SCALE's voxel-wise null computation fast
    import nibabel as nib
    from nimare.meta.cbma.ale import SCALE
    from nimare.utils import get_template

    from nimare.utils import vox2mm

    tpl = get_template("mni152_2mm", mask="brain")
    # small mask keeps the null fast, but must overlap the parcellation used
    # below (left frontal/precentral: 6 Yan100 parcels) or every parcellated
    # null map is trivially all-zero
    small = np.zeros(tpl.shape, dtype=np.uint8)
    small[20:30, 60:70, 45:55] = 1
    small[tpl.get_fdata() == 0] = 0
    dset = Dataset(
        os.path.join(get_resource_path(), "nidm_pain_dset.json"),
        mask=nib.Nifti1Image(small, tpl.affine),
    )
    # SCALE's own null pool: a deterministic, clearly non-uniform subset of the
    # mask (every third voxel), so reusing it must differ from uniform sampling
    xyz = vox2mm(np.vstack(np.where(small)).T[::3].astype(float), tpl.affine)
    return SCALE(xyz=xyz, n_iters=5, n_cores=1).fit(dset)


def test_scale_result_rejects_non_stat_outcome_map(scale_result):
    with pytest.raises(ValueError, match="SCALE"):
        null_maps_from_nimare(
            scale_result, parcellation="Yan100", outcome_map="z",
            n_perm=2, seed=0, n_proc=1, validate=False, verbose=False,
        )


def test_scale_result_rejects_binary_mode(scale_result, ale_corrected_result):
    with pytest.raises(NotImplementedError, match="SCALE"):
        null_maps_from_nimare(
            scale_result, parcellation="Yan100", corrected_result=ale_corrected_result,
            n_perm=2, seed=0, n_proc=1, validate=False, verbose=False,
        )


def test_scale_result_adopts_own_null(scale_result):
    # by default a SCALE result's own null pool is reused, not uniform sampling
    kwargs = dict(
        parcellation="Yan100", n_perm=4, seed=0, n_proc=1, validate=False, verbose=False,
    )
    own = null_maps_from_nimare(scale_result, **kwargs)["stat"]
    uniform = null_maps_from_nimare(scale_result, reporting_baseline="uniform", **kwargs)["stat"]
    assert own.shape == uniform.shape == (4, 100)
    assert np.isfinite(own).all()
    assert not np.allclose(own, uniform)


def test_scale_result_warns_when_own_null_unusable(scale_result, caplog):
    est = scale_result.estimator
    original_xyz = est.xyz
    est.xyz = np.zeros((1, 3))   # nimare-gpu's placeholder for "no pool given"
    try:
        with caplog.at_level("WARNING"):
            nulls = null_maps_from_nimare(
                scale_result, parcellation="Yan100",
                n_perm=2, seed=0, n_proc=1, validate=False, verbose=False,
            )
    finally:
        est.xyz = original_xyz
    assert any("uniform sampling is used" in r.message for r in caplog.records)
    assert nulls["stat"].shape == (2, 100)
