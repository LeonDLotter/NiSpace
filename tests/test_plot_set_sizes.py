"""XSEA categorical plots show set sizes after the set names: "$(n=k)$" (k maps), or
"$(n=k/t)$" for PET maps (t unique targets).

Runs on the synthetic fake parcellation (conftest.py); the parcellation is only a vehicle here."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nispace import NiSpace
from nispace.core.plot import _set_size_labels

N_PARCELS = 24  # fake_parcellation


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _pet(target, tracer, pub):
    return f"target-{target}_tracer-{tracer}_n-10_dx-hc_pub-{pub}"


def test_set_size_labels():
    # overlapping PET sets with a weight level: maps counted per set, targets unique per set
    idx = pd.MultiIndex.from_tuples([
        ("Dopamine", _pet("D23", "flb457", "a2020"), 0.5),
        ("Dopamine", _pet("D23", "raclopride", "b2021"), 0.5),
        ("Dopamine", _pet("DAT", "fpcit", "c2022"), 1.0),
        ("Monoamines", _pet("DAT", "fpcit", "c2022"), 1.0),
        ("Monoamines", _pet("SERT", "dasb", "d2023"), 1.0),
    ], names=["set", "map", "weight"])
    assert _set_size_labels(pd.DataFrame(index=idx)) == {
        "Dopamine": "$(n=3/2)$", "Monoamines": "$(n=2/2)$"}
    # other maps: map count only
    idx = pd.MultiIndex.from_tuples([("A", "g1"), ("A", "g2"), ("B", "g3")], names=["set", "map"])
    assert _set_size_labels(pd.DataFrame(index=idx)) == {"A": "$(n=2)$", "B": "$(n=1)$"}
    # no sets
    assert _set_size_labels(pd.DataFrame(index=["m0", "m1"])) is None


def test_xsea_plot_labels_show_set_sizes(rng, fake_parcellation):
    idx = pd.MultiIndex.from_tuples([("A", "m0"), ("A", "m1"), ("A", "m2"), ("B", "m3")],
                                    names=["set", "map"])
    x = pd.DataFrame(rng.normal(size=(4, N_PARCELS)), index=idx)
    nsp = NiSpace(x=x, y=rng.normal(size=(1, N_PARCELS)), parcellation=fake_parcellation,
                  verbose=False).fit()
    nsp.colocalize("spearman", xsea=True)
    fig, ax, _ = nsp.plot(method="spearman", xsea=True, show=False)
    labels = {t.get_text() for t in ax.get_yticklabels()}
    assert labels == {"A $(n=3)$", "B $(n=1)$"}


def test_significant_labels_get_bold_math(rng, fake_parcellation):
    # math text ignores the font weight: significant labels get \mathbf, the others keep \mathrm
    y = rng.normal(size=(1, N_PARCELS))
    names = [_pet("5HT2a", "alt", "a2020"), _pet("D1", "sch", "b2021"), _pet("GABAa", "flu", "c2022")]
    x = pd.DataFrame([y[0] + rng.normal(scale=s, size=N_PARCELS) for s in (0.1, 5, 5)], index=names)
    nsp = NiSpace(x=x, y=y, parcellation=fake_parcellation, verbose=False).fit()
    nsp.colocalize("spearman")
    nsp.permute("maps", n_perm=100, seed=1)
    nsp.correct_p(mc_method="fdr_bh")
    p = nsp.get_p_values("spearman", permute_what="maps", mc_method="fdr_bh").iloc[0]
    fig, ax, _ = nsp.plot(method="spearman", mc_method="fdr_bh", show=False)
    for lab in ax.get_yticklabels():
        txt = lab.get_text()
        if lab.get_weight() == "bold":
            assert r"_\mathbf{" in txt and r"_\mathrm{" not in txt
        else:
            assert r"_\mathbf{" not in txt
    assert any(lab.get_weight() == "bold" for lab in ax.get_yticklabels()) == bool((p < 0.05).any())
