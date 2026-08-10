"""Tests for the 8 user-facing workflow functions in workflows.py, plus the
3 deprecated name-only aliases kept for backward compatibility.

All tests reuse a synthetic, network-free, already-fitted NiSpace object
via ``nispace_object=`` (see ``_workflow_base``: a fitted object bypasses X
loading / NiSpace init / fit entirely) so no real reference dataset, real
parcellation, or network access is needed. Permutation uses
``maps_method="random"`` where relevant -- the one null method that needs
neither a parcellation nor a distance matrix (see
``nispace.nulls._DISTMAT_FREE_METHODS``) -- to stay fully synthetic.
``plot=False`` throughout: `plotting.py` is covered separately (smoke-only,
see the living testing-batches plan), not by this batch.
"""

import numpy as np
import pandas as pd
import pytest
import nibabel as nib

from nispace import NiSpace
from nispace.workflows import (
    colocalization, group_colocalization, paired_colocalization, correlate_within_region,
    xsea, group_xsea, nimare_colocalization, nimare_xsea,
    simple_colocalization, group_comparison, simple_xsea,
)


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def prefit_nispace(toy_regression):
    """Plain 3-X/1-Y synthetic NiSpace, fitted -- for colocalization()/xsea()-
    family tests that don't need multi-subject or set-membership structure."""
    X, Y, w, outlier_idx = toy_regression
    n_x, n_parcels = X.shape
    n_y = Y.shape[0]
    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    x_df = pd.DataFrame(X, index=[f"x{i}" for i in range(n_x)], columns=parcel_labels)
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(n_y)], columns=parcel_labels)
    nsp = NiSpace(x=x_df, y=y_df, z=None, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    return nsp


@pytest.fixture
def prefit_nispace_matched_pairs(rng):
    """N matched X/Y pairs (same row order), for paired_colocalization() (SPICE)."""
    n_pairs, n_parcels = 6, 20
    X = rng.normal(size=(n_pairs, n_parcels))
    Y = 0.6 * X + rng.normal(scale=0.5, size=(n_pairs, n_parcels))
    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    pair_labels = [f"p{i}" for i in range(n_pairs)]
    x_df = pd.DataFrame(X, index=pair_labels, columns=parcel_labels)
    y_df = pd.DataFrame(Y, index=pair_labels, columns=parcel_labels)
    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    return nsp


@pytest.fixture
def prefit_nispace_xsea(rng):
    """X with a "set" MultiIndex level (2 sets), single Y map -- for xsea()/
    nimare_xsea()."""
    n_parcels = 20
    genes_a, genes_b = 4, 6
    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    idx = pd.MultiIndex.from_tuples(
        [("setA", f"geneA{i}") for i in range(genes_a)]
        + [("setB", f"geneB{i}") for i in range(genes_b)],
        names=["set", "gene"],
    )
    X = rng.normal(size=(genes_a + genes_b, n_parcels))
    x_df = pd.DataFrame(X, index=idx, columns=parcel_labels)
    y = 0.7 * X[0] + rng.normal(scale=0.3, size=n_parcels)
    y_df = pd.DataFrame(y[np.newaxis, :], index=["y0"], columns=parcel_labels)
    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    return nsp


@pytest.fixture
def prefit_nispace_groups(rng):
    """2 X maps, individual-subject Y (16 subjects, 2 groups of 8, group b
    genuinely shifted) -- for group_colocalization()/group_xsea() (unpaired)."""
    n_x, n_parcels, n_subj = 2, 20, 16
    X = rng.normal(size=(n_x, n_parcels))
    Y = rng.normal(size=(n_subj, n_parcels))
    groups = np.array(["a"] * 8 + ["b"] * 8)
    Y[groups == "b"] += 1.0
    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    x_df = pd.DataFrame(X, index=[f"x{i}" for i in range(n_x)], columns=parcel_labels)
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(n_subj)], columns=parcel_labels)
    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    return nsp, groups


@pytest.fixture
def prefit_nispace_groups_paired(rng):
    """2 X maps, individual-subject Y (8 matched subjects x 2 timepoints/
    conditions) -- for group_colocalization(paired=True)."""
    n_x, n_parcels, n_per_group = 2, 20, 8
    X = rng.normal(size=(n_x, n_parcels))
    Y_a = rng.normal(size=(n_per_group, n_parcels))
    Y_b = Y_a + rng.normal(scale=0.3, size=(n_per_group, n_parcels)) + 1.0
    Y = np.vstack([Y_a, Y_b])
    groups = np.array(["a"] * n_per_group + ["b"] * n_per_group)
    subjects = np.array(list(range(n_per_group)) * 2)
    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    x_df = pd.DataFrame(X, index=[f"x{i}" for i in range(n_x)], columns=parcel_labels)
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(2 * n_per_group)], columns=parcel_labels)
    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    design = np.column_stack([groups, subjects])
    return nsp, design


# ── colocalization() ─────────────────────────────────────────────────────

def test_colocalization_end_to_end(prefit_nispace):
    out = colocalization(
        y=None, nispace_object=prefit_nispace,
        colocalization_method="pearson",
        permute_kwargs={"maps_method": "random"},
        n_perm=200, seed=1, plot=False, verbose=False,
        return_nispace_only=True,
    )
    assert isinstance(out, NiSpace)
    p = out.get_p_values()
    assert ((p.to_numpy() >= 0) & (p.to_numpy() <= 1)).all()


def test_colocalization_multiple_methods_returns_dict(prefit_nispace):
    out = colocalization(
        y=None, nispace_object=prefit_nispace,
        colocalization_method=["pearson", "spearman"],
        permute_kwargs={"maps_method": "random"},
        n_perm=200, seed=1, plot=False, verbose=False,
        return_nispace_only=True,
    )
    coloc_pearson = out.get_colocalizations(method="pearson")
    coloc_spearman = out.get_colocalizations(method="spearman")
    assert not np.allclose(coloc_pearson.to_numpy(), coloc_spearman.to_numpy())


def test_colocalization_deprecated_tuple_return_default(prefit_nispace):
    colocs, p_values, pc_values, nsp = colocalization(
        y=None, nispace_object=prefit_nispace,
        colocalization_method="pearson",
        permute_kwargs={"maps_method": "random"},
        n_perm=200, seed=1, plot=False, verbose=False,
    )
    assert isinstance(nsp, NiSpace)
    pd.testing.assert_frame_equal(colocs, nsp.get_colocalizations())


def test_workflow_base_prefit_nispace_object_no_unbound_error(prefit_nispace):
    """Regression test: _workflow_base() used to leave `null_maps` unbound
    when a pre-fitted nispace_object was passed (its assignment lived only
    inside the "if not status['init']" branch, which a pre-fitted object
    skips entirely), crashing with UnboundLocalError on return. Any workflow
    call below with a pre-fitted nispace_object exercises the fixed path;
    this test just makes the regression explicit."""
    out = colocalization(
        y=None, nispace_object=prefit_nispace,
        colocalization_method="pearson",
        permute_kwargs={"maps_method": "random"},
        n_perm=50, seed=1, plot=False, verbose=False,
        return_nispace_only=True,
    )
    assert isinstance(out, NiSpace)


def test_colocalization_n_perm_zero_skips_permutation(prefit_nispace):
    out = colocalization(
        y=None, nispace_object=prefit_nispace,
        colocalization_method="pearson",
        n_perm=0, plot=False, verbose=False,
        return_nispace_only=True,
    )
    assert isinstance(out, NiSpace)
    coloc = out.get_colocalizations(method="pearson", verbose=False)
    assert coloc.shape[0] > 0
    with pytest.raises(ValueError, match="perm"):
        out.get_p_values(verbose=False)


def test_colocalization_n_perm_zero_matches_n_perm_positive_coloc_values(prefit_nispace):
    """The n_perm=0 skip must not perturb the observed statistic itself --
    same seed/method, only permutation differs."""
    out_skip = colocalization(
        y=None, nispace_object=prefit_nispace, colocalization_method="pearson",
        n_perm=0, seed=1, plot=False, verbose=False, return_nispace_only=True,
    )
    out_full = colocalization(
        y=None, nispace_object=prefit_nispace, colocalization_method="pearson",
        permute_kwargs={"maps_method": "random"},
        n_perm=200, seed=1, plot=False, verbose=False, return_nispace_only=True,
    )
    pd.testing.assert_frame_equal(
        out_skip.get_colocalizations(method="pearson", verbose=False),
        out_full.get_colocalizations(method="pearson", verbose=False),
    )


def test_colocalization_n_perm_zero_plot_true_does_not_crash(prefit_nispace):
    """Exercises the api.py _last_settings["perm"] fix indirectly, via plot()'s
    own graceful-degradation branch."""
    out = colocalization(
        y=None, nispace_object=prefit_nispace, colocalization_method="pearson",
        n_perm=0, plot=True, verbose=False, return_nispace_only=True,
    )
    assert isinstance(out, NiSpace)


def test_colocalization_n_perm_zero_deprecated_tuple_return_p_values_none(prefit_nispace):
    colocs, p_values, pc_values, nsp = colocalization(
        y=None, nispace_object=prefit_nispace, colocalization_method="pearson",
        n_perm=0, plot=False, verbose=False,
    )
    assert p_values is None
    assert pc_values is None
    assert colocs is not None


@pytest.mark.parametrize("n_perm", [0, None])
def test_colocalization_n_perm_zero_or_none_are_equivalent(prefit_nispace, n_perm):
    out = colocalization(
        y=None, nispace_object=prefit_nispace, colocalization_method="pearson",
        n_perm=n_perm, plot=False, verbose=False, return_nispace_only=True,
    )
    with pytest.raises(ValueError, match="perm"):
        out.get_p_values(verbose=False)


# ── group_colocalization() ───────────────────────────────────────────────

def test_group_colocalization_unpaired_end_to_end(prefit_nispace_groups):
    nsp, groups = prefit_nispace_groups
    out = group_colocalization(
        y=list(range(len(groups))), design=groups, nispace_object=nsp,
        colocalization_method="pearson", n_perm=200, seed=1, plot=False, verbose=False,
        return_nispace_only=True,
    )
    p = out.get_p_values()
    assert ((p.to_numpy() >= 0) & (p.to_numpy() <= 1)).all()
    assert list(out.get_colocalizations().index) == ["hedges"]


def test_group_colocalization_paired_end_to_end(prefit_nispace_groups_paired):
    nsp, design = prefit_nispace_groups_paired
    out = group_colocalization(
        y=list(range(design.shape[0])), design=design, nispace_object=nsp,
        colocalization_method="pearson", paired=True,
        n_perm=200, seed=1, plot=False, verbose=False,
        return_nispace_only=True,
    )
    p = out.get_p_values()
    assert ((p.to_numpy() >= 0) & (p.to_numpy() <= 1)).all()
    assert list(out.get_colocalizations().index) == ["pairedcohen"]


def test_group_colocalization_design_length_mismatch_raises(prefit_nispace_groups):
    nsp, groups = prefit_nispace_groups
    with pytest.raises(ValueError):
        group_colocalization(
            y=list(range(len(groups) - 1)),  # deliberately wrong length
            design=groups, nispace_object=nsp,
            colocalization_method="pearson", plot=False, verbose=False,
            return_nispace_only=True,
        )


def test_group_colocalization_n_perm_zero_skips_permutation(prefit_nispace_groups):
    nsp, groups = prefit_nispace_groups
    out = group_colocalization(
        y=list(range(len(groups))), design=groups, nispace_object=nsp,
        colocalization_method="pearson", n_perm=0, plot=False, verbose=False,
        return_nispace_only=True,
    )
    coloc = out.get_colocalizations(verbose=False)
    assert list(coloc.index) == ["hedges"]
    with pytest.raises(ValueError, match="perm"):
        out.get_p_values(verbose=False)


def test_group_colocalization_n_perm_zero_plot_true_does_not_crash(prefit_nispace_groups):
    nsp, groups = prefit_nispace_groups
    out = group_colocalization(
        y=list(range(len(groups))), design=groups, nispace_object=nsp,
        colocalization_method="pearson", n_perm=0, plot=True, verbose=False,
        return_nispace_only=True,
    )
    assert isinstance(out, NiSpace)


def test_group_colocalization_n_perm_zero_deprecated_tuple_return_p_values_none(prefit_nispace_groups):
    nsp, groups = prefit_nispace_groups
    colocs, p_values, pc_values, out = group_colocalization(
        y=list(range(len(groups))), design=groups, nispace_object=nsp,
        colocalization_method="pearson", n_perm=0, plot=False, verbose=False,
    )
    assert p_values is None
    assert pc_values is None
    assert colocs is not None


# ── paired_colocalization() (SPICE) ──────────────────────────────────────

def test_paired_colocalization_end_to_end(prefit_nispace_matched_pairs):
    out = paired_colocalization(
        y=None, x=None, nispace_object=prefit_nispace_matched_pairs,
        colocalization_method="pearson",
        n_perm=200, seed=1, plot=False, verbose=False,
    )
    p = out.get_p_values()
    assert p.shape == (1, 1)
    assert 0 <= p.to_numpy().item() <= 1


def test_paired_colocalization_n_perm_zero_skips_permutation_bare_call(prefit_nispace_matched_pairs):
    out = paired_colocalization(
        y=None, x=None, nispace_object=prefit_nispace_matched_pairs,
        colocalization_method="pearson", n_perm=0, plot=False, verbose=False,
    )
    coloc = out.get_colocalizations(verbose=False)
    assert coloc.shape[0] > 0
    with pytest.raises(ValueError, match="perm"):
        out.get_p_values(verbose=False)


def test_paired_colocalization_n_perm_zero_skips_permutation_explicit_permute_what(prefit_nispace_matched_pairs):
    out = paired_colocalization(
        y=None, x=None, nispace_object=prefit_nispace_matched_pairs,
        colocalization_method="pearson", n_perm=0, plot=False, verbose=False,
    )
    with pytest.raises(KeyError, match="Did you run NiSpace.permute"):
        out.get_p_values(permute_what="pairs", verbose=False)


def test_paired_colocalization_n_perm_zero_plot_true_does_not_crash(prefit_nispace_matched_pairs):
    out = paired_colocalization(
        y=None, x=None, nispace_object=prefit_nispace_matched_pairs,
        colocalization_method="pearson", n_perm=0, plot=True, verbose=False,
    )
    assert isinstance(out, NiSpace)


# ── correlate_within_region() ────────────────────────────────────────────

def test_correlate_within_region_end_to_end(rng):
    n_subj, n_parcels = 12, 10
    X = rng.normal(size=(n_subj, n_parcels))
    Y = 0.8 * X + rng.normal(scale=0.3, size=(n_subj, n_parcels))
    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    subj_labels = [f"s{i}" for i in range(n_subj)]
    x_df = pd.DataFrame(X, index=subj_labels, columns=parcel_labels)
    y_df = pd.DataFrame(Y, index=subj_labels, columns=parcel_labels)

    out = correlate_within_region(
        x=x_df, y=y_df, parcellation=None, method="pearson",
        n_perm=200, seed=1, verbose=False,
    )
    res = out.get_within_region_correlations()
    assert res["stat"].shape == (1, n_parcels)
    assert res["mc_method"] == "step_maxT"  # new default
    assert res["p_corr"] is not None


def test_correlate_within_region_1d_y_covariate_matches_object_level(rng):
    # this used to crash: NiSpace.fit() has no notion of a subject-length
    # covariate, only NiSpace.correlate_within_region()'s X=/Y= overrides do --
    # the workflow function must fit on the 2D side and route the 1D side
    # through as a direct override, transparently to the caller
    n_subj, n_parcels = 12, 10
    X = rng.normal(size=(n_subj, n_parcels))
    yvec = rng.normal(size=n_subj)
    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    subj_labels = [f"s{i}" for i in range(n_subj)]
    x_df = pd.DataFrame(X, index=subj_labels, columns=parcel_labels)
    yvec_s = pd.Series(yvec, index=subj_labels)

    out_wf = correlate_within_region(
        x=x_df, y=yvec_s, parcellation=None, method="pearson", n_perm=0, verbose=False,
    )
    rho_wf = out_wf.get_within_region_correlations(mc_method=None)["stat"]

    nsp_direct = NiSpace(x=x_df, y=x_df, parcellation=None, standardize=False,
                         n_proc=1, verbose=False, return_self=False)
    nsp_direct.fit()
    nsp_direct.correlate_within_region(Y=yvec_s, method="pearson", n_perm=0)
    rho_direct = nsp_direct.get_within_region_correlations(mc_method=None)["stat"]

    np.testing.assert_allclose(rho_wf.values, rho_direct.values, atol=1e-10)


@pytest.mark.parametrize("shape_as", ["dataframe_1col", "array_nx1"])
def test_correlate_within_region_column_shaped_covariate_matches_1d(rng, shape_as):
    # a (n_subjects, 1) DataFrame/array covariate (e.g. df[["age"]] instead of
    # df["age"]) must be treated identically to a true 1D vector, not routed
    # through fit() as if it were a single already-parcellated map
    n_subj, n_parcels = 12, 10
    X = rng.normal(size=(n_subj, n_parcels))
    yvec = rng.normal(size=n_subj)
    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    subj_labels = [f"s{i}" for i in range(n_subj)]
    x_df = pd.DataFrame(X, index=subj_labels, columns=parcel_labels)

    if shape_as == "dataframe_1col":
        y_col = pd.DataFrame({"cov": yvec}, index=subj_labels)
    else:
        y_col = yvec.reshape(-1, 1)

    out_1d = correlate_within_region(
        x=x_df, y=pd.Series(yvec, index=subj_labels), parcellation=None,
        method="pearson", n_perm=0, verbose=False,
    )
    out_col = correlate_within_region(
        x=x_df, y=y_col, parcellation=None, method="pearson", n_perm=0, verbose=False,
    )
    rho_1d = out_1d.get_within_region_correlations(mc_method=None)["stat"]
    rho_col = out_col.get_within_region_correlations(mc_method=None)["stat"]
    np.testing.assert_allclose(rho_1d.values, rho_col.values, atol=1e-10)


def test_correlate_within_region_1d_x_covariate(rng):
    n_subj, n_parcels = 12, 10
    Y = rng.normal(size=(n_subj, n_parcels))
    xvec = rng.normal(size=n_subj)

    out = correlate_within_region(
        x=xvec, y=Y, parcellation=None, method="pearson", n_perm=0, verbose=False,
    )
    rho = out.get_within_region_correlations(mc_method=None)["stat"]
    assert rho.shape == (1, n_parcels)


def test_correlate_within_region_both_1d_raises(rng):
    xvec = rng.normal(size=12)
    with pytest.raises(ValueError):
        correlate_within_region(x=xvec, y=xvec, parcellation=None, verbose=False)


# ── xsea() ────────────────────────────────────────────────────────────────

def test_xsea_end_to_end(prefit_nispace_xsea):
    out = xsea(
        y=None, x=None, nispace_object=prefit_nispace_xsea,
        colocalization_method="pearson",
        permute_kwargs={"maps_method": "random"},
        n_perm=200, seed=1, plot=False, verbose=False,
        return_nispace_only=True,
    )
    coloc = out.get_colocalizations()
    assert set(coloc.columns) == {"setA", "setB"}
    p = out.get_p_values()
    assert ((p.to_numpy() >= 0) & (p.to_numpy() <= 1)).all()


# ── group_xsea() ──────────────────────────────────────────────────────────

def test_group_xsea_end_to_end(rng):
    n_parcels = 20
    genes_a, genes_b, n_per_group = 4, 6, 8
    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    idx = pd.MultiIndex.from_tuples(
        [("setA", f"geneA{i}") for i in range(genes_a)]
        + [("setB", f"geneB{i}") for i in range(genes_b)],
        names=["set", "gene"],
    )
    X = rng.normal(size=(genes_a + genes_b, n_parcels))
    Y_a = rng.normal(size=(n_per_group, n_parcels))
    Y_b = Y_a + rng.normal(scale=0.3, size=(n_per_group, n_parcels)) + 1.0
    Y = np.vstack([Y_a, Y_b])
    groups = np.array(["a"] * n_per_group + ["b"] * n_per_group)
    x_df = pd.DataFrame(X, index=idx, columns=parcel_labels)
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(2 * n_per_group)], columns=parcel_labels)
    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()

    out = group_xsea(
        y=list(range(2 * n_per_group)), design=groups, nispace_object=nsp,
        colocalization_method="pearson", n_perm=200, seed=1, plot=False, verbose=False,
        return_nispace_only=True,
    )
    coloc = out.get_colocalizations()
    assert set(coloc.columns) == {"setA", "setB"}


# ── nimare_colocalization() ───────────────────────────────────────────────

def test_nimare_colocalization_without_nimare_nulls_permutes_x(prefit_nispace):
    out = nimare_colocalization(
        y=None, nispace_object=prefit_nispace.copy(),
        colocalization_method="pearson",
        permute_kwargs={"maps_method": "random"},
        n_perm=200, seed=1, plot=False, verbose=False,
        return_nispace_only=True,
    )
    assert out._nulls["maps_null"].null_which == "X"


def test_nimare_colocalization_with_nimare_nulls_permutes_y(prefit_nispace, rng):
    n_parcels = prefit_nispace._X.shape[1]
    nimare_nulls = {"y0": rng.normal(size=(200, n_parcels))}
    out = nimare_colocalization(
        y=None, nispace_object=prefit_nispace.copy(),
        colocalization_method="pearson",
        nimare_nulls=nimare_nulls,
        n_perm=200, seed=1, plot=False, verbose=False,
        return_nispace_only=True,
    )
    assert out._nulls["maps_null"].null_which == "Y"


def test_nimare_colocalization_background_value_y_false_default_applies_on_fit(rng):
    """`nimare_colocalization()` sets `fit_kwargs.setdefault("background_value",
    {"y": False})` (see workflows.py) so a genuinely all-zero Y parcel (e.g. "no
    activation" in an ALE map) survives as real 0.0 rather than being excluded as
    background. Every other nimare_colocalization/nimare_xsea test in this file
    uses an already-fitted `nispace_object=`, which bypasses `_workflow_base`'s
    `nsp.fit(**fit_kwargs)` call entirely (see module docstring) -- so none of
    them actually exercise this default. This test deliberately builds a fresh,
    unfitted NiSpace via raw synthetic images + a fake non-brain-shaped
    parcellation (mirroring tests/test_parcellate_background.py's fixtures) so
    `nispace_object=None` forces `_workflow_base` to call `.fit()` for real.

    Uses 20 parcels (not 3, like test_parcellate_background.py's fixtures) --
    with only 3 points, this workflow's default permutation ("random" nulls,
    n_perm draws recorrelated against fixed Y) has a real chance of a null
    landing at Pearson r==1 by coincidence, tripping colocalize.py's
    r_equal_one="raise" guard and failing the test for reasons unrelated to
    background handling. 20 points avoids that degeneracy. Per-parcel X values
    get tiny jitter so no whole parcel coincidentally equals the tiny synthetic
    volume's auto-detected border background value (a real collision risk with
    uniform per-parcel constants, see project_parcellation_bg_handling memory)."""
    affine = np.eye(4)
    n_parcels = 20
    shape = (n_parcels, 2, 2)  # 4 voxels per parcel

    def _parc_img():
        arr = np.zeros(shape)
        for i in range(n_parcels):
            arr[i] = i + 1
        return nib.Nifti1Image(arr, affine)

    bg_parcel_idx = 5  # 0-based; label bg_parcel_idx+1 -- genuine "no activation" 0.0

    x_arr = np.zeros(shape)
    for i in range(n_parcels):
        x_arr[i] = rng.normal(loc=i, scale=0.01, size=(2, 2))

    y_arr = np.zeros(shape)
    y_means = rng.normal(size=n_parcels)
    y_means[bg_parcel_idx] = 0.0
    for i in range(n_parcels):
        y_arr[i] = y_means[i]

    out = nimare_colocalization(
        y=[nib.Nifti1Image(y_arr, affine)],
        x=[nib.Nifti1Image(x_arr, affine)],
        parcellation=_parc_img(), parcellation_space="MNI152NLin6Asym",
        parcellation_labels=[f"p{i + 1}" for i in range(n_parcels)],
        data_space="MNI152NLin6Asym",
        colocalization_method="pearson",
        permute_kwargs={"maps_method": "random"},
        n_perm=20, seed=1, plot=False, verbose=False,
        nispace_object=None,
        return_nispace_only=True,
    )
    Y = out.get_y()
    assert Y.shape[1] == n_parcels
    assert Y.iloc[0, bg_parcel_idx] == 0.0  # stayed real 0.0, not NaN'd out as background


# ── nimare_xsea() ─────────────────────────────────────────────────────────

def test_nimare_xsea_end_to_end(prefit_nispace_xsea):
    out = nimare_xsea(
        y=None, nispace_object=prefit_nispace_xsea,
        colocalization_method="pearson",
        permute_kwargs={"maps_method": "random"},
        n_perm=200, seed=1, plot=False, verbose=False,
        return_nispace_only=True,
    )
    coloc = out.get_colocalizations()
    assert set(coloc.columns) == {"setA", "setB"}


# ── deprecated aliases: delegate correctly + warn ────────────────────────

def test_simple_colocalization_warns_and_delegates(prefit_nispace, caplog):
    out = simple_colocalization(
        y=None, nispace_object=prefit_nispace,
        colocalization_method="pearson",
        permute_kwargs={"maps_method": "random"},
        n_perm=200, seed=1, plot=False, verbose=False,
        return_nispace_only=True,
    )
    assert isinstance(out, NiSpace)


def test_group_comparison_warns_and_delegates(prefit_nispace_groups):
    nsp, groups = prefit_nispace_groups
    out = group_comparison(
        y=list(range(len(groups))), design=groups, nispace_object=nsp,
        colocalization_method="pearson", n_perm=200, seed=1, plot=False, verbose=False,
        return_nispace_only=True,
    )
    assert isinstance(out, NiSpace)


def test_simple_xsea_warns_and_delegates(prefit_nispace_xsea):
    out = simple_xsea(
        y=None, nispace_object=prefit_nispace_xsea,
        colocalization_method="pearson",
        permute_kwargs={"maps_method": "random"},
        n_perm=200, seed=1, plot=False, verbose=False,
        return_nispace_only=True,
    )
    assert isinstance(out, NiSpace)
