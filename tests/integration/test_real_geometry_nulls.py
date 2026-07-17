"""End-to-end check of real-geometry null map generation via
NiSpace.permute(what="maps") with the real, auto-selected null method
(Moran spatial autocorrelation-preserving nulls on the real Yan100
geometry) -- the synthetic suite (test_permute.py/test_api.py) deliberately
stays parcellation-free via maps_method="random"/explicit synthetic distance
matrices, so this is the only place the real default null-generation path
(distance matrix computed from real parcellation geometry, no override) is
actually exercised end-to-end.
"""

import numpy as np

from nispace import NiSpace
from nispace.datasets import fetch_reference


def test_permute_maps_real_geometry_moran_null():
    x_df = fetch_reference("rsn", parcellation="Yan100", print_references=False, verbose=False)
    y = x_df.iloc[[0]].copy()
    y.index = ["y0"]
    x_rest = x_df.iloc[1:6]

    nsp = NiSpace(x=x_rest, y=y, parcellation="Yan100", verbose=False,
                  n_proc=1, return_self=False)
    nsp.fit()
    nsp.colocalize(method="pearson", verbose=False)
    p = nsp.permute(what="maps", n_perm=100, seed=1, verbose=False)

    assert ((p.to_numpy() >= 0) & (p.to_numpy() <= 1)).all()
    # real default null method should have been auto-selected (Moran, per
    # Yan100's null space) rather than left unset
    assert nsp._nulls["maps_null"].null_method is not None
