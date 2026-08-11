"""Real-data/real-geometry check of the row-batched null-map fast path
(NiSpace.permute()'s maps_batch_size) against the unbatched path.

Loads a real, previously-computed NiSpace pickle (docs/nb_introduction) and
runs the real, auto-selected null method (Moran on real Yan200/fsLR
geometry) both batched and unbatched, forcing maps_use_existing=False so
both runs actually generate fresh null maps rather than reusing the
pickle's cached ones. See [[feedback_verify_against_real_data]] -- this is
the check that originally caught a real bug (IndexError from an unguarded
0-d `_Z_obs_arr.shape[0]` when rank=True but no Z was ever provided) that
the synthetic suite, built without that specific rank-without-Z real-data
shape, did not surface.

No cross-row reduction occurs anywhere in this pipeline (each row's null
draw and colocalization statistic is computed independently of every other
row, batched or not), so batched vs. unbatched is expected to be
bit-identical, not merely close.
"""

from pathlib import Path

import numpy as np
import pytest

from nispace import NiSpace

_DOCS_DIR = Path(__file__).parent.parent.parent / "docs" / "nb_introduction"


def _pickle_path(name):
    p = _DOCS_DIR / name
    if not p.exists():
        pytest.skip(f"{p} not present")
    return str(p)


def test_permute_maps_x_batched_matches_unbatched_real_geometry():
    nsp_b = NiSpace.from_pickle(_pickle_path("intro02_nsp.pkl.blosc"), verbose=False)
    nsp_u = NiSpace.from_pickle(_pickle_path("intro02_nsp.pkl.blosc"), verbose=False)
    assert nsp_b._X.shape[0] > 5  # sanity: enough rows for a small maps_batch_size to matter

    nsp_b.permute(what="maps", maps_which="X", n_perm=100, seed=777,
                 maps_use_existing=False, maps_batch_size=5, pooled_p=False, verbose=False)
    nsp_u.permute(what="maps", maps_which="X", n_perm=100, seed=777,
                 maps_use_existing=False, maps_batch_size=False, pooled_p=False, verbose=False)

    pb = nsp_b.get_p_values(pooled_p=False)
    pu = nsp_u.get_p_values(pooled_p=False)
    assert list(pb.columns) == list(pu.columns)
    np.testing.assert_array_equal(pb.values, pu.values)

    # fast path actually engaged (not silently skipped, e.g. via a cache hit)
    assert nsp_b._nulls.get("maps_null_fastpath_info") is not None
    assert nsp_b._nulls.get("maps_null") is None
    assert nsp_u._nulls.get("maps_null") is not None


def test_permute_maps_x_batched_rank_true_no_z_real_data_does_not_crash():
    # the exact real-data condition that originally surfaced the 0-d _Z_obs_arr bug:
    # this pickle's stored method is "spearman" (rank=True), and no Z was ever provided.
    nsp = NiSpace.from_pickle(_pickle_path("intro02_nsp.pkl.blosc"), verbose=False)
    assert nsp._Z is None
    p = nsp.permute(what="maps", maps_which="X", n_perm=50, seed=1,
                    maps_use_existing=False, maps_batch_size=5, pooled_p=False, verbose=False)
    p = nsp.get_p_values(pooled_p=False)
    assert ((p.to_numpy() >= 0) & (p.to_numpy() <= 1)).all()
