"""permute() resolves the default map-null method (and loads/fits the parcellation's null
space) only when maps are permuted; group/set permutations leave the parcellation alone."""

import numpy as np
import pandas as pd

from nispace import NiSpace

N_DK = 68


def _grouped(rng):
    x = pd.DataFrame(rng.normal(size=(3, N_DK)), index=[f"x{i}" for i in range(3)])
    y = pd.DataFrame(rng.normal(size=(16, N_DK)), index=[f"s{i}" for i in range(16)])
    groups = np.array(["a"] * 8 + ["b"] * 8)
    nsp = NiSpace(x=x, y=y, parcellation="DesikanKilliany", verbose=False).fit()
    nsp.colocalize(method="spearman", Y_transform="cohen(a,b)", groups=groups, verbose=False)
    return nsp


def test_permute_groups_leaves_parcellation_space(rng, caplog):
    nsp = _grouped(rng)
    space_before = nsp._parc._space
    hemi_before = set(nsp._parc._hemi_dict)
    with caplog.at_level("INFO"):
        nsp.permute(what="groups", n_perm=20, seed=1, verbose=True)
    assert not any("null method" in r.message for r in caplog.records)
    assert nsp._parc._space == space_before
    assert set(nsp._parc._hemi_dict) == hemi_before


def test_permute_maps_still_resolves_default_null_method(rng, caplog):
    nsp = _grouped(rng)
    with caplog.at_level("INFO"):
        nsp.permute(what="maps", n_perm=20, seed=1, verbose=True)
    assert any("Using default null method" in r.message for r in caplog.records)
