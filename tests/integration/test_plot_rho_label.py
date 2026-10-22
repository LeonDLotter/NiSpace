"""Categorical plot labels Fisher-z "rho" as z(Rho): colocalize(r_to_z=True) (the default) and XSEA
store arctanh(rho); only r_to_z=False stores plain rho."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nispace import NiSpace

N_DK = 68


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _label(out):
    fig, ax, _ = out if isinstance(out, tuple) else next(iter(out.values()))
    return ax.get_xlabel() + ax.get_ylabel()


def _nsp(rng, x_index=None):
    x = pd.DataFrame(rng.normal(size=(4, N_DK)), index=x_index)
    return NiSpace(x=x, y=rng.normal(size=(1, N_DK)), parcellation="DesikanKilliany",
                   verbose=False).fit()


def test_default_rho_is_labelled_fisher_z(rng):
    nsp = _nsp(rng)
    nsp.colocalize("spearman")                      # r_to_z=True by default
    assert "$z$(" in _label(nsp.plot(method="spearman", show=False))


def test_raw_rho_keeps_plain_label(rng):
    nsp = _nsp(rng)
    nsp.colocalize("spearman", r_to_z=False)
    lab = _label(nsp.plot(method="spearman", show=False))
    assert "Rho" in lab and "$z$(" not in lab


def test_xsea_is_labelled_fisher_z(rng):
    idx = pd.MultiIndex.from_tuples([("A", "m0"), ("A", "m1"), ("B", "m2"), ("B", "m3")],
                                    names=["set", "map"])
    nsp = _nsp(rng, x_index=idx)
    nsp.colocalize("spearman", xsea=True, r_to_z=False)   # forced to the z scale for XSEA
    assert "$z$(" in _label(nsp.plot(method="spearman", xsea=True, show=False))


def test_passed_in_result_unknown_scale_plain_label(rng):
    nsp = _nsp(rng)
    nsp.colocalize("spearman")
    colocs = nsp.get_colocalizations("spearman", force_dict=True)
    lab = _label(nsp.plot(colocalizations_dict=colocs, show=False))
    assert "$z$(" not in lab
