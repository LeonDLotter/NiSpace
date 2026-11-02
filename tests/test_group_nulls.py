"""Tests for nispace.core.group_nulls: label-shuffle and Freedman-Lane group-permutation nulls,
and their use through NiSpace.permute(what="groups") / clean_y().

Ground truth for Freedman-Lane is a direct numpy implementation (lstsq) of the procedure
(Freedman & Lane 1983; Winkler et al. 2014): permute the residuals of the covariates-only model,
add the fitted values back, repeat the observed covariate regression, recompute the transform.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from nispace import NiSpace
from nispace.core.group_nulls import (_FreedmanLane, _row_permutations, generate_group_nulls,
                                      _hash_arrays)
from nispace.core.transform_y import _get_transform_fun
from nispace.stats.misc import permute_groups


def _transform(formula):
    return _get_transform_fun(formula, return_df=False, dtype=np.float32,
                              ignore_nan_warnings=True)


def _design(rng, n_per_group=20, r=0.6, n_parcels=30, paired=False):
    """Two groups (or two sessions), a covariate associated with group, Y driven by the
    covariate only (no group effect)."""
    n = 2 * n_per_group
    g = np.r_[np.zeros(n_per_group), np.ones(n_per_group)].astype(int)
    cov = r * (g - 0.5) / 0.5 + np.sqrt(1 - r ** 2) * rng.normal(size=n)
    Y = cov[:, None] + rng.normal(size=(n, n_parcels))
    subjects = np.r_[np.arange(n_per_group), np.arange(n_per_group)] if paired else None
    if paired:
        Y += rng.normal(size=(n_per_group, n_parcels))[subjects]     # subject effects
    return g, cov, Y, subjects


def _nsp_cleaned(Y, g, cov_df, subjects=None, protect_subjects=False, **nsp_kwargs):
    labels = np.where(g == 0, "HC", "PAT")
    cols = [f"p{i}" for i in range(Y.shape[1])]
    y_df = pd.DataFrame(Y, index=[f"s{i}" for i in range(len(Y))], columns=cols)
    nsp = NiSpace(x=y_df.iloc[:2], y=y_df, parcellation=None, standardize=False, n_proc=1,
                  verbose=False, return_self=False, **nsp_kwargs)
    nsp.fit()
    protect = pd.DataFrame({"groups": labels})
    if protect_subjects:
        protect["subjects"] = np.asarray(subjects).astype(str)
    nsp.clean_y("between", covariates_between=cov_df, protect=protect)
    return nsp, labels


def _reference_fl(Y, nuisance, protect, g, pis, formula, subjects=None):
    """Plain numpy Freedman-Lane (no NaN)."""
    n = len(Y)
    R = [np.ones((n, 1)), nuisance]
    if subjects is not None:
        uniq = np.unique(subjects)
        R.append((subjects[:, None] == uniq[None, 1:]).astype(float))
    R = np.column_stack(R)
    fitted = R @ np.linalg.lstsq(R, Y, rcond=None)[0]
    res = Y - fitted
    X_full = np.column_stack([nuisance, protect, np.ones(n)])
    k = nuisance.shape[1]
    at = _transform(formula)
    out = []
    for pi in pis:
        Ys = fitted + res[pi]
        beta = np.linalg.lstsq(X_full, Ys, rcond=None)[0]
        out.append(at(y=Ys - nuisance @ beta[:k], groups=g, subjects=subjects))
    return np.stack(out, axis=1)


# ── row permutations ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("paired", [False, True])
def test_row_permutations_reproduce_label_shuffle(rng, paired):
    g = np.r_[np.zeros(6), np.ones(6)].astype(int)
    subjects = np.r_[np.arange(6), np.arange(6)] if paired else None
    labels = permute_groups(g, subjects=subjects, paired=paired, n_perm=20, seed=5)
    rows = _row_permutations(g, subjects, paired=paired, n_perm=20, seed=5)
    for lab, pi in zip(labels, rows):
        # label shuffling gives row j the label lab[j]; the row permutation puts row pi[k] where
        # the original label g[k] is: the same assignment
        assert (lab[pi] == g).all()
        if paired:
            assert (subjects[pi] == subjects).all()


@pytest.mark.parametrize("paired", [False, True])
def test_intercept_only_freedman_lane_equals_label_shuffle(rng, paired):
    g, _, Y, subjects = _design(rng, paired=paired)
    formula = "pairedcohen(b,a)" if paired else "hedges(b,a)"
    at = _transform(formula)
    fl = _FreedmanLane(Y, steps=[], keep=np.ones(len(g), bool),
                       subjects=subjects if paired else None)
    c_lab = generate_group_nulls(Y.astype(np.float32), g, subjects, at, 40, seed=3,
                                 paired=paired, verbose=False)
    c_fl = generate_group_nulls(Y.astype(np.float32), g, subjects, at, 40, seed=3,
                                paired=paired, fl=fl, verbose=False)
    np.testing.assert_allclose(c_fl, c_lab, atol=1e-5)


# ── replay of clean_y ─────────────────────────────────────────────────────────

def test_replay_reproduces_clean_y_with_missing_values(rng):
    g, cov, Y, _ = _design(rng)
    Y[3, 5] = np.nan                       # parcel with a missing value
    cov_df = pd.DataFrame({"age": cov, "sex": rng.integers(0, 2, len(g)).astype(float)})
    cov_df.loc[7, "age"] = np.nan          # subject with a missing covariate
    nsp, _ = _nsp_cleaned(Y, g, cov_df)
    # second between step
    nsp.clean_y("between", covariates_between=pd.DataFrame({"motion": rng.normal(size=len(g))}),
                protect=pd.DataFrame({"groups": g.astype(str)}))
    steps = nsp._clean_y_between_steps
    assert len(steps) == 2 and "Y_pre" in steps[0] and "Y_pre" not in steps[1]
    fl = _FreedmanLane(steps[0]["Y_pre"], steps, np.ones(len(g), bool))
    rep = fl.clean(steps[0]["Y_pre"])
    obs = nsp._Y.to_numpy()
    assert (np.isnan(rep) == np.isnan(obs)).all()
    np.testing.assert_allclose(rep[~np.isnan(obs)], obs[~np.isnan(obs)], atol=1e-4)
    # the subject with a missing covariate is left out of the permutation
    assert 7 not in fl.perm_pos


# ── Freedman-Lane vs reference ────────────────────────────────────────────────

@pytest.mark.parametrize("paired", [False, True])
def test_freedman_lane_matches_reference(rng, paired):
    g, cov, Y, subjects = _design(rng, paired=paired, n_parcels=12)
    if paired:
        cov = rng.normal(size=len(g)) + 0.8 * g      # scan-level covariate (e.g. motion)
    cov_df = pd.DataFrame({"cov": cov})
    nsp, labels = _nsp_cleaned(Y, g, cov_df, subjects=subjects, protect_subjects=paired)
    step = nsp._clean_y_between_steps[0]
    formula = "pairedcohen(b,a)" if paired else "hedges(b,a)"
    fl = _FreedmanLane(step["Y_pre"], [step], np.ones(len(g), bool),
                       subjects=subjects if paired else None)
    cube = generate_group_nulls(nsp._Y.to_numpy(np.float32), g, subjects, _transform(formula),
                                30, seed=1, paired=paired, fl=fl, verbose=False)
    pis = _row_permutations(g, subjects, paired=paired, n_perm=30, seed=1)
    ref = _reference_fl(step["Y_pre"].astype(float), step["nuisance"].astype(float),
                        step["protect"].astype(float), g, pis, formula,
                        subjects=subjects if paired else None)
    np.testing.assert_allclose(cube, ref, atol=1e-4)


def test_freedman_lane_nan_parcel_matches_per_parcel_reference(rng):
    g, cov, Y, _ = _design(rng, n_parcels=4)
    Y[2, 1] = np.nan
    nsp, _ = _nsp_cleaned(Y, g, pd.DataFrame({"cov": cov}))
    step = nsp._clean_y_between_steps[0]
    fl = _FreedmanLane(step["Y_pre"], [step], np.ones(len(g), bool))
    pi = _row_permutations(g, n_perm=1, seed=4)[0]
    out = fl.permuted(pi)[:, 1].astype(float)
    # reference for the NaN parcel: fit on observed rows, fitted values for all rows
    y = step["Y_pre"][:, 1].astype(float)
    ok = ~np.isnan(y)
    R = np.column_stack([np.ones(len(g)), step["nuisance"]]).astype(float)
    b = np.linalg.lstsq(R[ok], y[ok], rcond=None)[0]
    fitted, res = R @ b, np.where(ok, y - R @ b, np.nan)
    ys = fitted + res[pi]
    ok2 = ~np.isnan(ys)
    X = np.column_stack([step["nuisance"], step["protect"], np.ones(len(g))]).astype(float)
    beta = np.linalg.lstsq(X[ok2], ys[ok2], rcond=None)[0]
    ref = np.where(ok2, ys - step["nuisance"][:, 0] * beta[0], np.nan)
    assert (np.isnan(out) == np.isnan(ref)).all()
    np.testing.assert_allclose(out[ok2], ref[ok2], atol=1e-4)


# ── calibration ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("paired", [False, True])
def test_freedman_lane_per_parcel_fpr_is_nominal(paired):
    """No group effect, covariate associated with group: Freedman-Lane keeps the per-parcel
    false positive rate at alpha; shuffling labels of the cleaned data does not."""
    rng = np.random.default_rng(11)
    fpr_fl, fpr_labels = [], []
    for _ in range(4):
        g, cov, Y, subjects = _design(rng, n_parcels=300, paired=paired)
        if paired:
            # scan-level covariate (e.g. head motion), higher in the second session
            cov = rng.normal(size=len(g)) + 1.0 * g
            Y = cov[:, None] + rng.normal(size=Y.shape) + rng.normal(size=(20, 300))[subjects]
        nsp, labels = _nsp_cleaned(Y, g, pd.DataFrame({"cov": cov}), subjects=subjects,
                                   protect_subjects=paired)
        formula = "pairedcohen(PAT,HC)" if paired else "hedges(PAT,HC)"
        nsp.transform_y(formula, groups=labels,
                        subjects=subjects.astype(str) if paired else None)
        nsp.compare_within_region(n_perm=199, seed=1, verbose=False)
        fpr_fl.append((nsp.get_within_region_comparisons(mc_method=None,
                                                         verbose=False)["p"].values < 0.05).mean())
        # label shuffling on the cleaned data (the scheme used before Freedman-Lane)
        obs = nsp.get_y(verbose=False).to_numpy()[0]
        cube = generate_group_nulls(nsp._Y.to_numpy(np.float32), g, subjects,
                                    _transform("pairedcohen(b,a)" if paired else "hedges(b,a)"),
                                    199, seed=1, paired=paired, verbose=False)[0]
        p_lab = (1 + (np.abs(cube) >= np.abs(obs)).sum(0)) / 200
        fpr_labels.append((p_lab < 0.05).mean())
    assert 0.03 <= np.mean(fpr_fl) <= 0.07
    assert np.mean(fpr_labels) > np.mean(fpr_fl) + 0.02


# ── settings and warnings ─────────────────────────────────────────────────────

def test_freedman_lane_rejects_non_shuffle_strategy(rng):
    g, cov, Y, _ = _design(rng)
    nsp, labels = _nsp_cleaned(Y, g, pd.DataFrame({"cov": cov}))
    nsp.transform_y("hedges(PAT,HC)", groups=labels)
    with pytest.raises(ValueError, match="strategy"):
        nsp.compare_within_region(n_perm=20, groups_strategy="proportional", verbose=False)


def test_within_after_between_falls_back_in_permute_and_raises_in_compare(rng, caplog):
    g, cov, Y, _ = _design(rng)
    nsp, labels = _nsp_cleaned(Y, g, pd.DataFrame({"cov": cov}))
    with caplog.at_level(logging.WARNING, logger="nispace"):
        nsp.clean_y("within", covariates_within=rng.normal(size=Y.shape[1]))
    assert "can't account for the between-subject covariates" in caplog.text
    assert nsp._clean_y_fl_valid is False
    nsp.transform_y("hedges(PAT,HC)", groups=labels)
    with pytest.raises(ValueError, match="within"):
        nsp.compare_within_region(n_perm=20, verbose=False)
    nsp.colocalize("pearson", verbose=False)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="nispace"):
        nsp.permute("groups", n_perm=20, verbose=False)
    assert "shuffles labels of the covariate-cleaned Y" in caplog.text
    assert nsp._nulls["groups_null_info"]["scheme"] == "labels"


def test_permute_groups_uses_freedman_lane_with_covariates(rng, caplog):
    g, cov, Y, _ = _design(rng)
    nsp, labels = _nsp_cleaned(Y, g, pd.DataFrame({"cov": cov}))
    nsp.transform_y("hedges(PAT,HC)", groups=labels)
    nsp.colocalize("pearson", verbose=False)
    with caplog.at_level(logging.INFO, logger="nispace"):
        nsp.permute("groups", n_perm=20, seed=1, verbose=True)
    assert "Freedman-Lane group permutation" in caplog.text
    assert nsp._nulls["groups_null_info"]["scheme"] == "freedman-lane"


def test_paired_constant_covariates_are_reported(rng, caplog):
    g, _, Y, subjects = _design(rng, paired=True)
    age = np.r_[rng.normal(size=20), np.zeros(20)]
    age[20:] = age[:20]                                      # constant within subjects
    cov_df = pd.DataFrame({"age": age, "motion": rng.normal(size=40)})
    nsp, labels = _nsp_cleaned(Y, g, cov_df, subjects=subjects, protect_subjects=True)
    nsp.transform_y("pairedcohen(PAT,HC)", groups=labels, subjects=subjects.astype(str))
    with caplog.at_level(logging.INFO, logger="nispace"):
        nsp.compare_within_region(n_perm=20, verbose=True)
    assert "Covariates constant within subjects drop out of the paired comparison: age" \
        in caplog.text


def test_combat_is_treated_as_fixed_with_warning(rng, caplog):
    pytest.importorskip("neuroHarmonize")
    n = 40
    g = np.r_[np.zeros(20), np.ones(20)].astype(int)
    Y = rng.normal(size=(n, 20)) + 5
    cov_df = pd.DataFrame({"site": np.tile(["s1", "s2"], 20), "age": rng.normal(size=n)})
    labels = np.where(g == 0, "HC", "PAT")
    y_df = pd.DataFrame(Y, index=[f"s{i}" for i in range(n)],
                        columns=[f"p{i}" for i in range(20)])
    nsp = NiSpace(x=y_df.iloc[:2], y=y_df, parcellation=None, standardize=False, n_proc=1,
                  verbose=False, return_self=False)
    nsp.fit()
    nsp.clean_y("between", covariates_between=cov_df, combat=True,
                protect=pd.DataFrame({"groups": labels}),
                combat_protect=pd.DataFrame({"groups": labels}))
    assert nsp._clean_y_between_steps[0]["combat"] is True
    nsp.transform_y("hedges(PAT,HC)", groups=labels)
    with caplog.at_level(logging.WARNING, logger="nispace"):
        nsp.compare_within_region(n_perm=20, verbose=False)
    assert "ComBat harmonization is treated as fixed" in caplog.text


def test_hash_arrays_distinguishes_values_shapes_and_none():
    a = np.arange(6.0)
    assert _hash_arrays(a) == _hash_arrays(a.copy())
    assert _hash_arrays(a) != _hash_arrays(a.reshape(2, 3))
    assert _hash_arrays(a) != _hash_arrays(a + 1)
    assert _hash_arrays(None) != _hash_arrays(a)
