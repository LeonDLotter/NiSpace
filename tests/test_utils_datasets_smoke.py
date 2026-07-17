"""Smoke tests for the pure, local-file (no network) helpers in
utils/utils_datasets.py -- hash calculation, file-extension parsing, and
the nifti/gifti dtype-compression helpers. The network/data-repo-dependent
functions in this module (download*, get_file, sync_osf, _check_hash
against the real hash_lib) are out of scope for the unit tier; see
tests/integration/ for real fetch_* coverage instead.
"""

import hashlib

import numpy as np
import nibabel as nib
import pytest

from nispace.utils.utils_datasets import (
    calculate_md5_hash, calculate_sha256_hash, _get_file_ext,
    _compress_nifti, _compress_gifti,
)


# ── hash functions ────────────────────────────────────────────────────────

def test_calculate_md5_hash_matches_hashlib(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello world")
    assert calculate_md5_hash(str(f)) == hashlib.md5(b"hello world").hexdigest()


def test_calculate_sha256_hash_matches_hashlib(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello world")
    assert calculate_sha256_hash(str(f)) == hashlib.sha256(b"hello world").hexdigest()


# ── _get_file_ext ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("remote,expected", [
    ("foo.nii.gz", "nii.gz"),
    ("foo.nii", "nii"),
    ("foo.shape.gii", "shape.gii"),
    ("foo.func.gii.gz", "func.gii.gz"),
    ("foo.csv", "csv"),
])
def test_get_file_ext(remote, expected):
    assert _get_file_ext(remote) == expected


# ── _compress_nifti / _compress_gifti ────────────────────────────────────

def test_compress_gifti_roundtrips_data(tmp_path, rng):
    darray = nib.gifti.GiftiDataArray(rng.normal(size=10).astype(np.float32))
    gii = nib.GiftiImage(darrays=[darray])
    src = tmp_path / "src.gii"
    dst = tmp_path / "dst.gii"
    gii.to_filename(str(src))

    _compress_gifti(str(src), str(dst))
    loaded = nib.load(str(dst))
    np.testing.assert_allclose(loaded.agg_data(), gii.agg_data())


def test_compress_nifti_dtype_param_shrinks_stored_dtype(tmp_path, rng):
    """Regression test: `_compress_nifti(dtype=np.float32)` used to cast the
    in-memory array to float32 (`img.get_fdata().astype(dtype)`), but then
    call `image.new_img_like(img, img_dat, copy_header=True)`, which copies
    the *original* header -- including its float64 on-disk storage dtype --
    straight through, silently overriding the cast. Fixed by explicitly
    calling `img.header.set_data_dtype(dtype)` before saving."""
    arr = rng.normal(size=(5, 5, 5)).astype(np.float64)
    img = nib.Nifti1Image(arr, affine=np.eye(4))
    src = tmp_path / "src.nii.gz"
    dst = tmp_path / "dst.nii.gz"
    img.to_filename(str(src))

    _compress_nifti(str(src), str(dst), dtype=np.float32)
    loaded = nib.load(str(dst))
    np.testing.assert_allclose(loaded.get_fdata(), arr, atol=1e-5)
    assert loaded.get_data_dtype() == np.float32
