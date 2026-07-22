"""Sanity checks against real, previously-computed NiSpace pickles from the
intro notebooks (docs/nb_introduction/*.pkl.blosc). These exercise code
paths the synthetic suite structurally cannot: from_pickle()'s legacy-format
migration logic (NullMaps backfill, storage-dict backfill for attrs added
after these pickles were created), and real numerical output on real data/
parcellations. See [[feedback_verify_against_real_data]].
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


def test_load_intro02_with_nulls():
    nsp = NiSpace.from_pickle(_pickle_path("intro02_nsp.pkl.blosc"), verbose=False)
    assert isinstance(nsp, NiSpace)
    coloc = nsp.get_colocalizations(verbose=False)
    assert coloc.shape[0] >= 1
    assert np.isfinite(coloc.to_numpy()).any()

    # backfill attrs added after this pickle was created must be present
    for attr in ["_coloc_kwargs_by_method"]:
        assert hasattr(nsp, attr)

    # this pickle was saved with nulls -- p-values and maxT-style correction should work
    p = nsp.get_p_values(verbose=False)
    assert ((p.to_numpy() >= 0) & (p.to_numpy() <= 1)).all()
    assert len(nsp._nulls["_colocs"]) > 0


def test_load_intro02_no_nulls():
    nsp = NiSpace.from_pickle(_pickle_path("intro02_nsp_no_nulls.pkl.blosc"), verbose=False)
    assert isinstance(nsp, NiSpace)
    # p-values (not requiring the null distribution itself) still round-trip
    p = nsp.get_p_values(verbose=False)
    assert ((p.to_numpy() >= 0) & (p.to_numpy() <= 1)).all()
    # but the null distribution itself was dropped before saving
    assert nsp._nulls["_colocs"] == {}
    # so maxT correction (needs the full null) must fail clearly, not silently
    with pytest.raises(KeyError):
        nsp.correct_p(mc_method="maxT", verbose=False)


def test_load_intro10():
    nsp = NiSpace.from_pickle(_pickle_path("intro10_nsp.pkl.blosc"), verbose=False)
    assert isinstance(nsp, NiSpace)
    coloc = nsp.get_colocalizations(verbose=False)
    assert coloc.shape[0] >= 1


def test_reloaded_pickle_copy_and_pickle_again_roundtrips(tmp_path):
    """A pickle that's already been through from_pickle()'s migration path
    should still copy()/to_pickle()/from_pickle() cleanly afterwards -- i.e.
    the migration doesn't leave the object in some half-upgraded state."""
    nsp = NiSpace.from_pickle(_pickle_path("intro02_nsp.pkl.blosc"), verbose=False)
    nsp_copy = nsp.copy(verbose=False)

    f = tmp_path / "reloaded.pkl"
    nsp_copy.to_pickle(str(f), save_nulls=False, verbose=False)
    nsp_reloaded = NiSpace.from_pickle(str(f), verbose=False)

    import pandas as pd
    pd.testing.assert_frame_equal(
        nsp_reloaded.get_colocalizations(verbose=False),
        nsp.get_colocalizations(verbose=False),
    )
