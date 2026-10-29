"""NiMARE integration helpers."""

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import label as nd_label
from ..utils.utils import tqdm

from ..utils.utils import vol_to_vect_arr

lgr = logging.getLogger(__name__)

# NiMARE's default brain mask is FSL MNI152 (= MNI152NLin6Asym) at 2 mm.
_NIMARE_SPACE = "MNI152NLin6Asym"


# Module-level so joblib can pickle it.
def _run_one_perm(
    iter_ijk, iter_xyz, iter_df, est, parc_arr, parc_idc, bg_values,
    voxel_thresh_z, min_cluster_extent, cluster_stat,
    return_cluster_stat=False,
):
    """Run one coordinate-sampling permutation and return a parcellated map.

    The parcellation array (parc_arr/parc_idc/bg_values) is pre-resampled to
    ALE space once before the loop, so this function only calls vol_to_vect_arr
    on the raw 3-D array — no neuromaps resampling per iteration.

    When return_cluster_stat=True and voxel_thresh_z is not None, returns
    (parc_map, max_cluster_stat) where max_cluster_stat is the size or mass of
    the largest cluster BEFORE FWE filtering — matching what NiMARE stores in
    null_distributions_ for FWE threshold derivation.
    """
    iter_df = iter_df.copy()
    # Both voxel indices AND mm coordinates must be overwritten: newer NiMARE
    # (confirmed for 0.20.0 and 0.22.1) only recomputes i/j/k from x/y/z for
    # DataFrame input if the i/j/k columns are missing, and the
    # estimator's inputs_["coordinates"] already carries them from the original
    # fit -- writing only x/y/z made every permutation reproduce the observed map.
    # Older NiMARE (e.g., 0.6.x) recomputes i/j/k from x/y/z unconditionally,
    # which gives the same indices here (integer ijk -> mm -> ijk round trip).
    iter_df[["i", "j", "k"]] = np.squeeze(iter_ijk, axis=1)   # (n_coords, 1, 3) → (n_coords, 3)
    iter_df[["x", "y", "z"]] = np.squeeze(iter_xyz, axis=1)
    iter_ma = est.kernel_transformer.transform(
        iter_df, masker=est.masker, return_type="sparse"
    )
    iter_ss = est._compute_summarystat(iter_ma)  # ALE values in [0, 1], NOT z-scores

    if voxel_thresh_z is not None:
        # Convert ALE stat → z before comparing to voxel_thresh_z,
        # mirroring NiMARE's own cluster correction pipeline.
        if hasattr(est, "_summarystat_to_p"):
            _result = est._summarystat_to_p(iter_ss, null_method="approximate")
            if isinstance(_result, tuple):
                # NiMARE >= 0.6: returns (p_values, z_values) — use z directly.
                iter_z = _result[1]
            else:
                # older NiMARE: returns p_values only.
                from nimare.transforms import p_to_z
                iter_z = p_to_z(_result, tail="one")
        elif hasattr(est, "_ale_to_p"):
            from nimare.transforms import p_to_z
            iter_z = p_to_z(est._ale_to_p(iter_ss), tail="one")
        else:
            raise AttributeError(
                "Estimator has neither _summarystat_to_p nor _ale_to_p. "
                "Cluster-map null generation requires NiMARE >= 0.2.0."
            )
        arr3d = est.masker.inverse_transform(
            (iter_z >= voxel_thresh_z).astype(np.float32)
        ).get_fdata()

        # Label connected components once; compute per-cluster stats for both
        # filtering and validation (max stat before filtering = NiMARE semantics).
        max_stat = 0
        need_labels = (
            (min_cluster_extent is not None and min_cluster_extent > 0)
            or return_cluster_stat
        )
        if need_labels:
            labeled, n_comps = nd_label(arr3d > 0)
            if cluster_stat == "mass":
                z_arr3d = est.masker.inverse_transform(iter_z).get_fdata()
            ci_stats = []
            for ci in range(1, n_comps + 1):
                s = (
                    float(z_arr3d[labeled == ci].sum())
                    if cluster_stat == "mass"
                    else int((labeled == ci).sum())
                )
                ci_stats.append(s)
            max_stat = max(ci_stats) if ci_stats else 0
            if min_cluster_extent is not None and min_cluster_extent > 0:
                for ci, s in enumerate(ci_stats, 1):
                    if s < min_cluster_extent:
                        arr3d[labeled == ci] = 0

        arr3d = arr3d.astype(np.float32)
    else:
        arr3d = est.masker.inverse_transform(iter_ss).get_fdata().astype(np.float32)
        max_stat = None

    parc_map = vol_to_vect_arr(arr3d, parc_arr, parc_idc, bg_values)
    if return_cluster_stat:
        return parc_map, max_stat
    return parc_map


def _sample_null_idx(n_vox, n_coords, n_perm, rng, weights=None):
    """Draw null-focus voxel indices of shape (n_coords, n_perm).

    weights=None is the uniform draw (the exact RNG call used before
    reporting baselines existed, so seeded results are unchanged). Otherwise
    each focus lands on voxel v with probability weights[v], as in the
    probabilistic SCALE/SALE null.
    """
    if weights is None:
        return rng.integers(n_vox, size=(n_coords, n_perm))
    if len(weights) != n_vox:
        raise ValueError(
            f"Got {len(weights)} reporting-baseline weights for {n_vox} mask voxels."
        )
    return rng.choice(n_vox, size=(n_coords, n_perm), p=weights)


def _coord_pool_to_weights(xyz, mask_img, source="coordinate pool"):
    """Turn an (n, 3) pool of mm coordinates into null sampling weights.

    Drawing foci with replacement from a pool of reported coordinates -- the
    deterministic SCALE/SALE null -- is equivalent to drawing mask voxels with
    probability proportional to how often the pool hits each voxel, because
    NiMARE snaps every focus to a voxel before convolving it with its kernel.
    Coordinates outside the mask cannot be sampled and are dropped.
    """
    from nimare.utils import mm2vox

    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"{source} must be an (n, 3) array of x/y/z coordinates in mm, "
            f"got shape {xyz.shape}."
        )
    mask_arr = np.asarray(mask_img.get_fdata()) != 0
    n_vox = int(mask_arr.sum())
    # position of each mask voxel in the np.vstack(np.where(mask)).T order the
    # sampler indexes into
    lookup = np.full(mask_arr.shape, -1, dtype=np.int64)
    lookup[mask_arr] = np.arange(n_vox)

    ijk = np.asarray(mm2vox(xyz, mask_img.affine), dtype=np.int64)
    in_bounds = np.all((ijk >= 0) & (ijk < np.array(mask_arr.shape)), axis=1)
    flat_idx = np.full(ijk.shape[0], -1, dtype=np.int64)
    flat_idx[in_bounds] = lookup[tuple(ijk[in_bounds].T)]
    keep = flat_idx >= 0
    n_dropped = int((~keep).sum())
    if n_dropped:
        lgr.warning(
            f"{n_dropped} of {xyz.shape[0]} coordinates of the {source} fall outside "
            "the estimator's mask and are ignored for null sampling."
        )
    counts = np.bincount(flat_idx[keep], minlength=n_vox).astype(np.float64)
    n_hit = int((counts > 0).sum())
    if n_hit < 2:
        raise ValueError(
            f"The {source} covers {n_hit} mask voxel(s); it cannot define a null "
            "distribution of coordinates."
        )
    lgr.info(
        f"Reporting baseline from {source}: {int(counts.sum())} coordinates on "
        f"{n_hit} of {n_vox} mask voxels."
    )
    return counts / counts.sum()


def _scale_estimator_weights(est):
    """Null sampling weights taken from a fitted SCALE estimator, or None.

    A SCALE result was itself tested against a reporting-baseline null, so
    reusing that null makes the null maps generated here match it. GPU
    implementations in probabilistic mode carry it as a normalized voxel-wise
    probability (``sampling_prob``, e.g. nimare-gpu's ``DeviceSCALE``), the
    deterministic version samples from a pool of reported foci (``xyz``).
    Returns None (with a warning) if neither is usable.
    """
    prob = getattr(est, "sampling_prob", None)
    if prob is not None:
        w = np.asarray(prob, dtype=np.float64).ravel()
        n_vox = int(np.count_nonzero(est.masker.mask_img.get_fdata()))
        if w.size != n_vox:
            lgr.warning(
                f"SCALE estimator's sampling_prob has {w.size} values for {n_vox} mask "
                "voxels; ignoring it."
            )
        elif not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            lgr.warning(
                "SCALE estimator's sampling_prob is not a usable non-negative "
                "distribution; ignoring it."
            )
        else:
            lgr.info(
                "Null coordinates are sampled from the SCALE estimator's own "
                "probability map (estimator.sampling_prob)."
            )
            return w / w.sum()

    xyz = getattr(est, "xyz", None)
    if xyz is None:
        return None
    try:
        return _coord_pool_to_weights(
            xyz, est.masker.mask_img,
            source="SCALE estimator's own coordinate pool (estimator.xyz)",
        )
    except ValueError as e:
        lgr.warning(f"Cannot reuse the SCALE estimator's own null: {e}")
        return None


def _fetch_reporting_baseline(query, verbose=True):
    """Fetch a map of the 'mabaseline' reference dataset by (sub)string query.

    The query must match exactly one map ID; this is checked against the
    datalib before any download. The map is returned in 'MNIOriginal' space,
    i.e., the unmodified source image.
    """
    from ..datasets import fetch_reference, reference_lib

    available = list(reference_lib.get("mabaseline", {}).get("map", {}).keys())
    matches = [m for m in available if query.lower() in m.lower()]
    if len(matches) != 1:
        raise ValueError(
            f"reporting_baseline='{query}' matched {len(matches)} maps of the "
            f"'mabaseline' reference dataset {matches}, but must match exactly one. "
            f"Available: {available}. Alternatively, pass a NIfTI image or file path."
        )
    map_id = matches[0]
    lgr.info(
        f"Using reporting baseline '{map_id}' ('MNIOriginal' space) from the "
        "'mabaseline' reference dataset."
    )
    fps = fetch_reference(
        "mabaseline", maps=map_id, space="MNIOriginal",
        print_references=verbose, verbose=verbose,
    )
    return nib.load(fps[0])


def _resolve_reporting_baseline(reporting_baseline, mask_img, verbose=True):
    """Turn a reporting-baseline specification into null sampling weights.

    Returns None for reporting_baseline=None (uniform sampling). Otherwise
    returns a float64 vector with one weight per mask voxel, summing to 1 and
    ordered like np.vstack(np.where(mask_arr)).T -- the candidate null
    coordinates in null_maps_from_nimare(). Normalization within the
    estimator's mask is the only transformation applied to the values.
    """
    if reporting_baseline is None:
        return None

    mask_arr = np.asarray(mask_img.get_fdata())

    if isinstance(reporting_baseline, (str, Path)):
        if str(reporting_baseline).lower() == "uniform":
            lgr.info("reporting_baseline='uniform': sampling null coordinates uniformly.")
            return None
        if Path(reporting_baseline).is_file():
            lgr.info(f"Loading reporting baseline from file: {reporting_baseline}")
            baseline = nib.load(reporting_baseline)
        else:
            baseline = _fetch_reporting_baseline(str(reporting_baseline), verbose=verbose)
    elif isinstance(reporting_baseline, nib.spatialimages.SpatialImage):
        baseline = reporting_baseline
    elif hasattr(reporting_baseline, "coordinates") or hasattr(reporting_baseline, "columns"):
        # NiMARE Dataset, or a coordinate table with x/y/z columns
        is_dset = hasattr(reporting_baseline, "coordinates")
        source = "NiMARE dataset" if is_dset else "coordinate table"
        table = reporting_baseline.coordinates if is_dset else reporting_baseline
        missing = [c for c in ("x", "y", "z") if c not in getattr(table, "columns", [])]
        if missing:
            raise ValueError(
                f"reporting_baseline ({source}) is missing the {missing} column(s) "
                "needed to read coordinates."
            )
        return _coord_pool_to_weights(
            table[["x", "y", "z"]].values, mask_img, source=source
        )
    elif isinstance(reporting_baseline, np.ndarray):
        if reporting_baseline.ndim == 2 and reporting_baseline.shape[1] == 3:
            # (n, 3) pool of mm coordinates -> deterministic SCALE/SALE null
            return _coord_pool_to_weights(reporting_baseline, mask_img)
        if reporting_baseline.shape != mask_arr.shape:
            raise ValueError(
                f"reporting_baseline array has shape {reporting_baseline.shape}, but the "
                f"estimator's mask has shape {mask_arr.shape}. Arrays must be on the mask "
                "grid (pass a NIfTI image to resample automatically) or be an (n, 3) "
                "array of x/y/z coordinates in mm."
            )
        baseline = None
        arr = np.asarray(reporting_baseline, dtype=np.float64)
    else:
        raise TypeError(
            "reporting_baseline must be None, 'uniform', a 'mabaseline' map query (str), "
            "a file path, a NIfTI image, a numpy array, or a coordinate pool "
            f"((n, 3) array, table, or NiMARE dataset). Got: {type(reporting_baseline)}"
        )

    if baseline is not None:
        if len(baseline.shape) == 4 and baseline.shape[3] == 1:
            from nilearn.image import index_img
            baseline = index_img(baseline, 0)
        elif len(baseline.shape) != 3:
            raise ValueError(
                f"reporting_baseline image must be 3D, got shape {baseline.shape}."
            )
        if (baseline.shape != mask_img.shape
                or not np.allclose(baseline.affine, mask_img.affine)):
            # linear, not "continuous" (cubic spline): a convex combination of
            # non-negative values cannot become negative
            from nilearn.image import resample_img
            lgr.info("Resampling reporting baseline to the estimator's mask grid (linear).")
            baseline = resample_img(
                baseline,
                target_affine=mask_img.affine,
                target_shape=mask_img.shape,
                interpolation="linear",
                force_resample=True,
                copy_header=True,
            )
        arr = np.asarray(baseline.get_fdata(), dtype=np.float64)

    # same mask definition and C-order as np.where(mask_arr) in the caller
    w = arr[mask_arr != 0]
    n_nan = int(np.isnan(w).sum())
    if n_nan:
        lgr.warning(f"{n_nan} mask voxels have NaN reporting-baseline values; treated as 0.")
        w = np.nan_to_num(w, nan=0.0)
    if (w < 0).any():
        raise ValueError(
            f"reporting_baseline has negative values within the mask (min={w.min():.4g}); "
            "it must be a non-negative density."
        )
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            f"reporting_baseline sums to {total} within the estimator's mask; it must "
            "have a positive, finite sum."
        )
    n_zero = int((w == 0).sum())
    if n_zero:
        lgr.info(
            f"{n_zero} of {w.size} mask voxels have zero reporting-baseline weight; "
            "no null foci will be placed there."
        )
    return w / total


def _check_null_maps_degenerate(null_arr, obs_parc=None):
    """Raise if coordinate-sampling null maps are clearly not random.

    Independently drawn foci never give identical (non-empty) parcellated maps,
    and never reproduce the observed map. Either pattern means the random
    coordinates did not reach NiMARE's kernel (e.g., stale i/j/k columns), so
    every downstream p-value would be invalid -- fail loudly instead of logging.
    All-zero rows (binary mode, no supra-threshold cluster) are legitimately
    identical and handled by the separate empty-map warning.
    """
    def _same(a, b):
        # relative tolerance only: ALE values can be tiny, an absolute one would
        # declare any two near-zero maps "identical"
        return np.allclose(a, b, rtol=1e-5, atol=0, equal_nan=True)

    nonempty = null_arr[~np.all((null_arr == 0) | np.isnan(null_arr), axis=1)]
    if nonempty.shape[0] >= 2 and all(_same(row, nonempty[0]) for row in nonempty[1:]):
        raise RuntimeError(
            f"All {nonempty.shape[0]} non-empty null maps are identical -- the random "
            "coordinates were not applied. This is a bug; please report it together "
            "with your NiMARE version."
        )
    if obs_parc is not None:
        n_obs = sum(_same(row, obs_parc) for row in nonempty)
        if n_obs > 0:
            raise RuntimeError(
                f"{n_obs}/{null_arr.shape[0]} null map(s) are identical to the observed "
                "map -- the random coordinates were not applied. This is a bug; please "
                "report it together with your NiMARE version."
            )


def _validate_null_maps(
    null_arr, max_stats, result, outcome_map,
    parc_arr, parc_idc, voxel_thresh_z, nimare_cluster_extent,
    cluster_stat, alpha, observed_map, baseline_used=False,
):
    """Log validation metrics comparing our null distribution to NiMARE's own outputs.

    Cluster mode: compares the (1-alpha) percentile of our max-cluster-stat null
    distribution against the threshold stored in corrected_result.estimator.null_distributions_.
    Optionally re-derives the cluster map with our threshold and compares to
    observed_map (Pearson r).

    Continuous mode: computes empirical parcel-level p-values from null_arr and
    correlates (Spearman) against NiMARE's parametric voxelwise p-map parcellated
    with the same scheme.

    With baseline_used=True, NiMARE's outputs still assume a uniform null, so the
    metrics are reported as the effect of the reporting baseline rather than as
    agreement checks.
    """
    _bg = np.array([], dtype=np.float64)

    if max_stats is not None:
        # ---- cluster mode ----
        max_stats_arr = np.asarray(max_stats, dtype=float)
        our_thresh = np.percentile(max_stats_arr, 100.0 * (1.0 - alpha))
        lgr.info("=== Validation (cluster mode) ===")
        lgr.info(
            f"  NiMARE FWE {cluster_stat} threshold (from corrector): "
            f"{nimare_cluster_extent}"
        )
        lgr.info(
            f"  Our null-derived {cluster_stat} threshold "
            f"({100*(1-alpha):.0f}th pctile of {len(max_stats)} perms): "
            f"{our_thresh:.1f}"
        )
        if nimare_cluster_extent is not None and nimare_cluster_extent > 0:
            ratio = our_thresh / nimare_cluster_extent
            if baseline_used:
                lgr.info(
                    f"  Base-rate inflation factor (ours / NiMARE's uniform-null threshold): "
                    f"{ratio:.3f}  (> 1: chance convergence is higher under the reporting "
                    "baseline; ~1: the baseline barely matters for this dataset)"
                )
            else:
                lgr.info(
                    f"  Ratio (ours / NiMARE): {ratio:.3f}  "
                    "(should be ~1.0 for large n_perm; deviations are sampling noise)"
                )

        # Re-derive cluster map with our threshold and compare to observed_map.
        if observed_map is not None:
            obs_z_img = result.get_map("z")
            obs_z_arr = obs_z_img.get_fdata()
            binary_arr = (obs_z_arr >= voxel_thresh_z).astype(np.float32)
            labeled, n_comps = nd_label(binary_arr > 0)
            for ci in range(1, n_comps + 1):
                s = (
                    float(obs_z_arr[labeled == ci].sum())
                    if cluster_stat == "mass"
                    else int((labeled == ci).sum())
                )
                if s < our_thresh:
                    binary_arr[labeled == ci] = 0
            our_cluster_parc = vol_to_vect_arr(
                binary_arr, parc_arr, parc_idc, _bg
            ).astype(float)
            if isinstance(observed_map, nib.spatialimages.SpatialImage):
                obs_user_parc = vol_to_vect_arr(
                    observed_map.get_fdata().astype(np.float32),
                    parc_arr, parc_idc, _bg,
                ).astype(float)
            else:
                obs_user_parc = np.asarray(observed_map, dtype=float)
            mask = np.isfinite(our_cluster_parc) & np.isfinite(obs_user_parc)
            if mask.sum() > 1:
                from scipy.stats import pearsonr
                r, _ = pearsonr(our_cluster_parc[mask], obs_user_parc[mask])
                if baseline_used:
                    lgr.info(
                        f"  Re-derived cluster map vs observed_map: r={r:.4f}  "
                        "(expected < 1 under a reporting baseline: the lower r, the more of "
                        "the observed cluster map would not survive a baseline-referenced "
                        "extent threshold)"
                    )
                else:
                    lgr.info(
                        f"  Re-derived cluster map vs observed_map: r={r:.4f}  "
                        "(should be ~1.0; low r → threshold mismatch or wrong cluster_stat)"
                    )

    else:
        # ---- continuous mode ----
        lgr.info("=== Validation (continuous mode) ===")
        obs_parc = vol_to_vect_arr(
            result.get_map(outcome_map).get_fdata().astype(np.float32),
            parc_arr, parc_idc, _bg,
        ).astype(float)
        valid = np.isfinite(obs_parc)
        if valid.sum() < 2:
            lgr.warning("  Too few valid parcels for continuous-mode validation.")
            return
        # Empirical p: proportion of null values >= observed at each parcel.
        emp_p = (null_arr[:, valid] >= obs_parc[valid]).mean(axis=0)
        # Some estimators (e.g. SCALE) store only -log10(p), not p.
        if result.maps.get("p") is not None:
            nimare_p_arr = result.get_map("p").get_fdata().astype(np.float32)
        elif result.maps.get("logp") is not None:
            nimare_p_arr = 10.0 ** -result.get_map("logp").get_fdata().astype(np.float32)
        else:
            lgr.warning(
                "  Result has neither a 'p' nor a 'logp' map; skipping the comparison "
                "against NiMARE's own p-values."
            )
            return
        nimare_p_parc = vol_to_vect_arr(
            nimare_p_arr, parc_arr, parc_idc, _bg,
        ).astype(float)
        nimare_p_valid = nimare_p_parc[valid]
        finite = np.isfinite(emp_p) & np.isfinite(nimare_p_valid)
        if finite.sum() > 1:
            from scipy.stats import spearmanr
            r, pval = spearmanr(emp_p[finite], nimare_p_valid[finite])
            lgr.info(
                f"  Empirical p (our null) vs NiMARE parametric p-map: "
                f"Spearman r={r:.3f}  (p={pval:.4f})"
            )
            if baseline_used:
                lgr.info(
                    "  Note: ours uses a reporting baseline, NiMARE's p-map does not "
                    "(ALE) or references every voxel to its own null (SCALE) — lower "
                    "agreement than under a uniform null is expected. Measured on the "
                    "NIDM-Pain dataset: r~0.96 under a uniform null vs. r~-0.18 for a "
                    "SCALE result compared against its own voxel-specific p-map."
                )
            else:
                lgr.info(
                    "  Note: NiMARE p is voxelwise-approximate; ours is parcel-level "
                    "Monte Carlo — perfect agreement is not expected, but r should be high."
                )


def null_maps_from_nimare(
    result,
    parcellation,
    corrected_result=None,
    reporting_baseline=None,
    voxel_thresh=0.001,
    outcome_map="stat",
    cluster_stat="size",
    alpha=0.05,
    n_perm=1000,
    map_label=None,
    observed_map=None,
    validate=True,
    seed=None,
    n_proc=1,
    dtype=np.float32,
    verbose=True,
):
    """Generate parcellated null maps from a fitted NiMARE MetaResult.

    Re-implements NiMARE's MonteCarlo FWE coordinate-sampling loop: for each
    permutation, study coordinates are drawn at random from the brain mask, a
    full ALE map is computed, and the result is parcellated. By default,
    coordinates are drawn uniformly, which produces null maps under the same
    generative model NiMARE uses for its own FWE correction. With
    *reporting_baseline*, coordinates are instead drawn in proportion to how
    often each location is reported across the literature (the SCALE/SALE null
    :cite:`langner2014,magielse2025`), so that colocalization is tested beyond
    what any meta-analysis would show.

    Two modes are supported:

    * **Continuous** (default, ``corrected_result=None``): each null map
      contains parcellated ALE stat values, matching an unthresholded
      ``result.get_map("stat")`` as Y.
    * **Binary cluster** (``corrected_result`` provided): each null ALE map is
      converted to z-scores via the estimator's null distribution, thresholded
      at the voxel-level CDT, and parcellated to a [0, 1] cluster-coverage map.
      The FWE cluster extent filter is **not** applied to null maps — applying it
      would make ~95% of them empty by construction (that is what FWE alpha=0.05
      means). Null maps represent CDT-thresholded random ALE; the FWE correction
      is already reflected in the observed Y image from :func:`get_binary_cluster_map`.

    Parameters
    ----------
    result : nimare.results.MetaResult
        Fitted MetaResult (e.g. from ``ALE().fit(dataset)``). Must embed a
        CBMA estimator with ``inputs_["coordinates"]`` and
        ``kernel_transformer``. IBMA estimators are not supported.
    parcellation : str or :class:`~nispace.core.parcellation.Parcellation`
        Parcellation to use. Either a NiSpace parcellation name (e.g.
        ``"Schaefer200"``) or a
        :class:`~nispace.core.parcellation.Parcellation` object (e.g.
        ``nsp._parc`` from a fitted :class:`~nispace.NiSpace` instance). Must
        have at least one volumetric/MNI space — surface-only parcellations are
        not supported.
    corrected_result : nimare.results.MetaResult or None, optional
        The corrected MetaResult returned by ``FWECorrector.transform(result)``.
        If provided, switches to binary cluster mode: the FWE cluster extent
        threshold is extracted from
        ``corrected_result.estimator.null_distributions_`` via
        :func:`nimare_fwe_thresholds` and applied to each null ALE map. Only
        ``method="montecarlo"`` is supported. Default: ``None`` (continuous
        mode).
    reporting_baseline : None, str, path, NIfTI image, ndarray, or coordinates, optional
        Spatial prior for the null coordinates. Some regions are reported far
        more often than others across neuroimaging studies, regardless of the
        topic, and ALE maps inherit this "reporting baseline". A uniform null
        ignores it, so colocalization with anything resembling the baseline can
        appear significant for every meta-analysis. With a reporting baseline,
        each null coordinate is drawn with probability proportional to the
        baseline value at that voxel, i.e., the null hypothesis becomes "the
        foci are a random draw from the literature at large" and the test asks
        whether colocalization is *specific* to the analyzed topic
        :cite:`langner2014,magielse2025`. Applies to both modes.

        * ``None`` (default): uniform sampling within the estimator's mask --
          except for SCALE results, which reuse the estimator's own null (see
          Notes).
        * ``str``: a map of NiSpace's ``"mabaseline"`` reference dataset,
          matched by substring (must match exactly one map), fetched in its
          original space. ``"neurosynth"``: density of all foci in the
          Neurosynth database :cite:`yarkoni2011`, as computed for C-SALE
          :cite:`magielse2025`. A string that is an existing file path is
          loaded as a NIfTI image instead. ``"uniform"`` forces uniform
          sampling (the only way to override the SCALE default).
        * NIfTI image or file path: any non-negative voxel-wise map, e.g., a
          baseline computed from another database. Resampled (linearly) to the
          estimator's mask grid if needed.
        * ndarray: a non-negative array on the estimator's mask grid
          (``result.estimator.masker.mask_img``).
        * Coordinate pool: an ``(n, 3)`` array of x/y/z coordinates in mm, a
          table with ``x``/``y``/``z`` columns, or a NiMARE dataset. Null foci
          are then drawn from the reported coordinates themselves, which is
          the deterministic SCALE/SALE null :cite:`langner2014` — e.g., all
          foci of the database the analyzed studies were drawn from. Note
          that, unlike a smooth density map, such a pool gives zero weight to
          every voxel it never hits.

        Values are only normalized to sum to 1 within the estimator's mask;
        NaN is treated as 0 and voxels with zero weight never receive null foci.
        Consider reporting results under both nulls.
    voxel_thresh : float, optional
        Uncorrected voxel-level p-value for the cluster-defining threshold
        (CDT) used when *corrected_result* is provided. Must match the
        threshold actually applied during FWE correction. NiMARE's default
        (and the default here) is ``0.001`` (z ≈ 3.09). NiMARE's
        ``FWECorrector`` silently ignores ``voxel_thresh`` passed to its
        constructor, so always specify it here explicitly if you used a
        non-default CDT. Ignored when *corrected_result* is ``None``.
        Default: ``0.001``.
    outcome_map : str, optional
        Key in ``result.maps`` used as the default *map_label* and for the
        continuous-mode sanity check. For ALE, available keys are ``"stat"``
        (ALE values), ``"z"``, and ``"p"``. The permutation loop always starts
        from ALE stat values (``_compute_summarystat``) regardless of this
        setting. Default: ``"stat"``.
    cluster_stat : {"size", "mass"}, optional
        Cluster-level statistic used for FWE correction. Only relevant when
        *corrected_result* is provided.

        * ``"size"`` (default): cluster extent in voxels. Each null cluster is
          removed if its voxel count is below the FWE threshold.
        * ``"mass"`` : cluster mass (sum of z-scores within the cluster). Each
          null cluster is removed if its mass is below the FWE threshold.

        Must match the statistic used to produce the observed Y image (i.e. the
        ``map_key`` passed to :func:`get_binary_cluster_map`).
    alpha : float, optional
        FWE significance level passed to :func:`nimare_fwe_thresholds` when
        *corrected_result* is provided, to derive the cluster extent threshold.
        Default: 0.05.
    n_perm : int, optional
        Number of null permutations. Default: 1000.
    map_label : str or None, optional
        Key in the returned dict. Defaults to *outcome_map*. Must match the
        Y-map label in your NiSpace instance so that
        ``permute(maps_nulls=...)`` routes to the right null distribution.
    observed_map : NIfTI image or array-like of shape (n_parcels,), optional
        Y map for a sanity check (Pearson r against the expected observed map,
        warned if < 0.99). Accepts either a NIfTI image in NiMARE's native
        space (parcellated internally) or a pre-parcellated array.
        In **continuous mode**: compared against the parcellated
        ``result.get_map(outcome_map)``.
        In **binary cluster mode**: compared against the observed ALE z-map
        thresholded and cluster-filtered with the same parameters — i.e. the
        exact procedure applied to each null map. Pass the output of
        :func:`get_binary_cluster_map` here (image or parcellated array).
    validate : bool, optional
        If True (default), log validation metrics after the permutation loop.

        * **Cluster mode**: compares the ``(1-alpha)`` percentile of the
          max-cluster-stat null distribution collected across permutations
          against the threshold from
          ``corrected_result.estimator.null_distributions_`` (these should
          agree within sampling noise). If *observed_map* is provided, also
          re-derives the cluster map using our threshold and reports Pearson r
          against *observed_map*.
        * **Continuous mode**: computes empirical parcel-level p-values from
          the null array and correlates (Spearman) against NiMARE's parametric
          voxelwise p-map parcellated with the same scheme.

        With *reporting_baseline*, NiMARE's outputs still assume a uniform
        null, so agreement is not expected: in cluster mode, the threshold
        ratio is reported as a *base-rate inflation factor* (how much larger
        clusters become by chance under the baseline; ~1 means the baseline
        barely matters for this dataset).
    seed : int or None, optional
        Random seed for reproducibility. Default: None.
    n_proc : int, optional
        Number of parallel processes (``joblib.Parallel`` n_jobs). Default: 1.
    dtype : numpy dtype, optional
        Output array dtype. Default: ``np.float32``.
    verbose : bool, optional
        Show progress bar and log messages. Default: True.

    Returns
    -------
    dict
        ``{map_label: ndarray(n_perm, n_parcels)}``, directly passable as
        ``maps_nulls=`` to :meth:`~nispace.NiSpace.permute`.

    Notes
    -----
    **Binary cluster mode — value distributions and correlation method.**
    After parcellation, each parcel in the binary cluster map receives the
    fraction of its voxels that fall inside the significant cluster (values in
    [0, 1], typically 0–0.4 for the observed map). Use ``method="pearson"``
    in :meth:`~nispace.NiSpace.colocalize` — Pearson on a fractional
    cluster-coverage map is equivalent to point-biserial correlation and is
    more appropriate than Spearman, which degrades due to heavy ties at zero.

    Binary null maps are thresholded at the voxel-level CDT only; the FWE
    cluster extent filter is **not** applied to null maps (applying it would
    make ~95% of them empty by construction — that is what FWE alpha=0.05
    means). Null maps therefore have a lower value range (typically 0–0.05–0.1)
    than the observed map: random coordinates produce many small scattered
    clusters while real coordinates converge into one focused cluster. This
    contrast is the signal the permutation test exploits. With a
    *reporting_baseline*, null coordinates concentrate in frequently reported
    regions, so null clusters become larger and more frequent there. The
    observed Y, however, still carries NiMARE's uniform-null FWE extent
    threshold, which is more lenient than a baseline-referenced one would be.
    The marginal clusters it retains tend to lie in frequently reported
    regions, so binary-mode results can remain slightly anticonservative for
    reference maps resembling the baseline.

    **Reporting baseline and SCALE.** NiMARE's :class:`~nimare.meta.cbma.ale.SCALE`
    (and GPU implementations deriving from it) already tests against a
    reporting-baseline null, and its ``"z"``/``"logp"`` maps are baseline-corrected.
    For SCALE results, only ``outcome_map="stat"`` (plain ALE values) is
    accepted, because the null maps generated here are ALE values too. Binary
    cluster mode is not supported for SCALE results. Alternatively, a SCALE
    z-map can be used as Y with NiSpace's standard spatial nulls.

    With ``reporting_baseline=None``, a SCALE result's own null is reused, so
    the null maps are drawn from the same distribution the SCALE map itself
    was tested against: the estimator's voxel-wise probability map
    (``sampling_prob``, set by GPU implementations such as nimare-gpu's
    ``DeviceSCALE`` in probabilistic mode, https://github.com/amnsbr/nimare-gpu)
    if present, otherwise its coordinate pool (``xyz``). Any explicit
    *reporting_baseline* overrides this, and ``"uniform"`` restores uniform
    sampling.

    Relies on the private NiMARE methods ``_compute_summarystat`` and
    ``_summarystat_to_p`` (or ``_ale_to_p`` for older versions). Requires
    NiMARE >= 0.2.0; informative errors are raised for missing attributes.

    NiMARE's default brain mask is MNI152NLin6Asym at 2 mm. The
    MNI152NLin6Asym image for the parcellation is preferred to minimise
    resampling error. The parcellation is resampled to ALE space once before
    the permutation loop; each null map is parcellated with a direct array
    operation rather than a per-iteration neuromaps call.

    **Continuous workflow** (unthresholded ALE stat map as Y)::

        import nimare
        from nimare.meta.cbma import ALE
        from nimare.tests.utils import get_test_data_path
        from nispace import NiSpace, null_maps_from_nimare
        from nispace.datasets import fetch_reference

        dset = nimare.dataset.Dataset(get_test_data_path() + "/test_pain_dataset.json")
        result = ALE().fit(dset)
        ale_img = result.get_map("stat")

        ref = fetch_reference("pet", maps="VAChT", parcellation="Schaefer200")
        nsp = NiSpace(x=ref, y=ale_img, y_labels=["pain_ale"],
                      parcellation="Schaefer200")
        nsp.fit()
        nsp.colocalize()

        null_maps = null_maps_from_nimare(
            result, nsp._parc,
            map_label="pain_ale",
            observed_map=ale_img,
            n_perm=1000,
        )
        nsp.permute("maps", maps_which="Y", maps_nulls=null_maps, n_perm=1000)
        p = nsp.get_p_values()

    **Reporting-baseline null** (same as above, testing topic-specificity)::

        null_maps_baseline = null_maps_from_nimare(
            result, nsp._parc,
            reporting_baseline="neurosynth",
            map_label="pain_ale",
            n_perm=1000,
        )

    **Deterministic SALE null** (foci resampled from a coordinate database)::

        null_maps_pool = null_maps_from_nimare(
            result, nsp._parc,
            reporting_baseline=full_dset,   # NiMARE dataset, table, or (n, 3) array
            map_label="pain_ale",
            n_perm=1000,
        )

    **Binary cluster workflow** (FWE cluster map binarized as Y)::

        from nimare.correct import FWECorrector
        from nispace import get_binary_cluster_map

        fwe = FWECorrector(method="montecarlo", n_iters=1000)
        corrected = fwe.transform(result)    # null_distributions_ live here
        binary_img = get_binary_cluster_map(corrected)

        nsp = NiSpace(x=ref, y=binary_img, y_labels=["pain_cluster"],
                      parcellation="Schaefer200")
        nsp.fit()
        nsp.colocalize()

        null_maps = null_maps_from_nimare(
            result, nsp._parc,
            corrected_result=corrected,      # pass the corrected MetaResult
            voxel_thresh=0.001,
            map_label="pain_cluster",
            observed_map=binary_img,
            n_perm=1000,
        )
        nsp.permute("maps", maps_which="Y", maps_nulls=null_maps, n_perm=1000)
        p = nsp.get_p_values()
    """
    lgr.warning(
        "null_maps_from_nimare() is experimental. The API and behaviour may "
        "change in future releases without notice."
    )

    # --- lazy import ---
    try:
        from nimare.utils import vox2mm
    except ImportError as exc:
        raise ImportError(
            "null_maps_from_nimare requires 'nimare'. "
            "Install with: pip install nimare"
        ) from exc

    # --- validate result ---
    if not (hasattr(result, "estimator") and hasattr(result, "maps")):
        raise TypeError(
            "result must be a NiMARE MetaResult (returned by estimator.fit(dataset)). "
            f"Got: {type(result)}"
        )
    est = result.estimator

    # --- validate estimator ---
    if not (
        hasattr(est, "inputs_")
        and "coordinates" in est.inputs_
        and hasattr(est, "kernel_transformer")
    ):
        raise TypeError(
            "The estimator in result does not appear to be a CBMA estimator "
            "(missing inputs_['coordinates'] or kernel_transformer). "
            "IBMA and non-standard estimators are not supported."
        )
    if not hasattr(est, "_compute_summarystat"):
        raise AttributeError(
            "estimator._compute_summarystat not found. "
            "This NiMARE version may be incompatible (requires >= 0.2.0)."
        )

    # --- validate outcome_map ---
    available = [k for k, v in result.maps.items() if v is not None]
    if outcome_map not in result.maps or result.maps[outcome_map] is None:
        raise ValueError(
            f"outcome_map='{outcome_map}' not found in result.maps. "
            f"Available: {available}"
        )

    # --- SCALE estimators (incl. subclasses, e.g., GPU implementations) ---
    scale_weights = None
    try:
        from nimare.meta.cbma.ale import SCALE as _SCALE
    except ImportError:
        _SCALE = None
    if _SCALE is not None and isinstance(est, _SCALE):
        if outcome_map != "stat":
            raise ValueError(
                f"outcome_map='{outcome_map}' is not supported for SCALE results: SCALE's "
                "z/logp maps are already reporting-baseline corrected, while the null maps "
                "generated here are ALE values. Use outcome_map='stat' (plain ALE values) "
                "as Y, or use the SCALE z-map as Y with NiSpace's standard spatial nulls."
            )
        if corrected_result is not None:
            raise NotImplementedError(
                "Binary cluster mode is not supported for SCALE results."
            )
        if reporting_baseline is None:
            # a SCALE result was tested against its own reporting-baseline null;
            # reuse it so the null maps match the map they will be compared to
            scale_weights = _scale_estimator_weights(est)
            if scale_weights is None:
                lgr.warning(
                    "result comes from a SCALE estimator, whose null samples foci from "
                    "the estimator's own coordinates, but that null could not be read "
                    "from the estimator, so uniform sampling is used. Pass "
                    "reporting_baseline=... (e.g., 'neurosynth') for a "
                    "reporting-baseline null."
                )
        else:
            lgr.info(
                "result comes from a SCALE estimator; null maps use the given "
                "reporting_baseline, not the estimator's own null."
            )

    # --- reporting baseline -> null sampling weights over mask voxels (None = uniform) ---
    if scale_weights is not None:
        baseline_weights = scale_weights
    else:
        baseline_weights = _resolve_reporting_baseline(
            reporting_baseline, est.masker.mask_img, verbose=verbose
        )

    # --- derive cluster thresholds from corrected result (binary cluster mode) ---
    if corrected_result is not None:
        if cluster_stat not in ("size", "mass"):
            raise ValueError(f"cluster_stat must be 'size' or 'mass', got {cluster_stat!r}.")
        _voxel_thresh_z, _min_cluster_extent = nimare_fwe_thresholds(
            corrected_result, alpha=alpha, cluster_stat=cluster_stat,
            voxel_thresh=voxel_thresh,
        )
    else:
        _voxel_thresh_z, _min_cluster_extent = None, None

    # --- resolve Parcellation object ---
    from ..core.parcellation import Parcellation as _Parcellation
    if isinstance(parcellation, str):
        from ..datasets import fetch_parcellation
        lgr.info(f"Fetching parcellation '{parcellation}' for null map generation.")
        parc_obj = fetch_parcellation(
            parcellation, return_parcellation_only=True, verbose=verbose
        )
    elif isinstance(parcellation, _Parcellation):
        parc_obj = parcellation
    else:
        raise TypeError(
            f"parcellation must be a NiSpace parcellation name (str) or Parcellation "
            f"object, got {type(parcellation)}."
        )

    mni_spaces = [s for s in parc_obj.spaces if "mni" in s.lower()]
    if not mni_spaces:
        raise ValueError(
            "Parcellation has no volumetric/MNI space. "
            "Surface-only parcellations are not supported for NiMARE null maps."
        )
    # Prefer NiMARE's native space (MNI152NLin6Asym) to minimise resampling error.
    nimare_mni = [s for s in mni_spaces if "lin6" in s.lower()]
    parc_space = nimare_mni[0] if nimare_mni else mni_spaces[0]
    parc_img = parc_obj.get_image(parc_space)

    # --- pre-resample parcellation to ALE space (once, avoids per-iteration neuromaps call) ---
    from nilearn.image import resample_img as _nilearn_resample_img
    lgr.info("Pre-resampling parcellation to ALE native space.")
    _parc_resampled = _nilearn_resample_img(
        parc_img,
        target_affine=est.masker.mask_img.affine,
        target_shape=est.masker.mask_img.shape,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    parc_arr_precomp = _parc_resampled.get_fdata().astype(np.float32)
    parc_idc_precomp = np.trim_zeros(np.unique(parc_arr_precomp)).astype(np.float32)
    # bg_values for vol_to_vect_arr: NaN is always excluded automatically.
    # Both modes pass an empty array so that 0-valued voxels are included in
    # each parcel's mean:
    #   binary mode  → parcel value = fraction of voxels in cluster ∈ [0, 1]
    #   continuous   → parcel value = mean ALE stat (ALE=0 inside brain is valid)
    # Using [0.0] in binary mode would exclude non-cluster voxels from the mean,
    # making every parcel 1.0 or NaN (fully binary) instead of fractional.
    bg_values_precomp = np.array([], dtype=np.float64)

    # --- optional observed-map comparison ---
    if observed_map is not None:
        if _voxel_thresh_z is not None:
            # Binary mode: apply the same voxel threshold + cluster extent to the
            # observed z-map, parcellate, then compare to the user's binary Y.
            obs_z_img = result.get_map("z")
            obs_z_arr = obs_z_img.get_fdata()
            binary_arr = (obs_z_arr >= _voxel_thresh_z).astype(np.float32)
            if _min_cluster_extent is not None and _min_cluster_extent > 0:
                labeled, n_comps = nd_label(binary_arr > 0)
                for ci in range(1, n_comps + 1):
                    if (labeled == ci).sum() < _min_cluster_extent:
                        binary_arr[labeled == ci] = 0
            obs_img = nib.Nifti1Image(binary_arr, obs_z_img.affine, obs_z_img.header)
            check_desc = (
                f"binary cluster map (z≥{_voxel_thresh_z:.3f}, "
                f"cluster_stat={cluster_stat}, min_extent={_min_cluster_extent})"
            )
        else:
            obs_img = result.get_map(outcome_map)
            check_desc = f"'{outcome_map}' map"

        # obs_img and observed_map (if NIfTI) are both in NiMARE's native ALE space,
        # so we can call vol_to_vect_arr directly with the pre-resampled parcellation.
        _bg_none = np.array([], dtype=np.float64)
        obs_parc = vol_to_vect_arr(
            obs_img.get_fdata().astype(np.float32),
            parc_arr_precomp, parc_idc_precomp, _bg_none,
        ).astype(float)
        if isinstance(observed_map, nib.spatialimages.SpatialImage):
            obs_user = vol_to_vect_arr(
                observed_map.get_fdata().astype(np.float32),
                parc_arr_precomp, parc_idc_precomp, _bg_none,
            ).astype(float)
        else:
            obs_user = np.asarray(observed_map, dtype=float)
        mask = np.isfinite(obs_parc) & np.isfinite(obs_user)
        if mask.sum() > 1:
            from scipy.stats import pearsonr
            r, _ = pearsonr(obs_parc[mask], obs_user[mask])
            if r < 0.99:
                lgr.warning(
                    f"Parcellated {check_desc} and observed_map have r={r:.3f} (<0.99). "
                    "Verify that the Y map passed to NiSpace was produced with the "
                    "same corrected_result (use get_binary_cluster_map)."
                )
            else:
                lgr.info(f"Observed-map sanity check passed (r={r:.3f}).")

    # --- coordinate sampling ---
    mask_arr = est.masker.mask_img.get_fdata()
    null_ijk = np.vstack(np.where(mask_arr)).T              # (n_mask_voxels, 3)
    n_coords = est.inputs_["coordinates"].shape[0]
    rng = np.random.default_rng(seed)
    rand_idx = _sample_null_idx(
        null_ijk.shape[0], n_coords, n_perm, rng, weights=baseline_weights
    )
    rand_ijk = null_ijk[rand_idx, :]                        # (n_coords, n_perm, 3)
    rand_xyz = vox2mm(rand_ijk, est.masker.mask_img.affine)  # (n_coords, n_perm, 3)
    iter_ijks = np.split(rand_ijk, rand_ijk.shape[1], axis=1)  # list of (n_coords, 1, 3)
    iter_xyzs = np.split(rand_xyz, rand_xyz.shape[1], axis=1)  # list of (n_coords, 1, 3)

    iter_df = est.inputs_["coordinates"].copy()

    mode = "binary cluster" if _voxel_thresh_z is not None else "continuous"
    sampling = "uniform" if baseline_weights is None else "reporting-baseline-weighted"
    lgr.info(
        f"Generating {n_perm} {mode} null maps from NiMARE {type(est).__name__} "
        f"estimator ({sampling} coordinate sampling, n_proc={n_proc})."
    )
    if _voxel_thresh_z is not None:
        lgr.info(
            f"  voxel_thresh_z={_voxel_thresh_z:.3f}, "
            f"min_cluster_voxels={_min_cluster_extent}"
        )

    # In cluster mode with validate=True, _run_one_perm returns (parc_map, max_stat)
    # so we can compare our null-derived FWE threshold to NiMARE's stored threshold.
    # Note: min_cluster_extent is NOT passed to _run_one_perm — the FWE cluster
    # extent filter is applied only to the observed map (by get_binary_cluster_map),
    # not to null maps. Applying it to nulls makes ~95% of them empty by design
    # (that is what FWE alpha=0.05 means). return_cluster_stat still collects the
    # pre-filter max stat per perm so the validation comparison is meaningful.
    _return_cluster_stat = validate and _voxel_thresh_z is not None

    # --- permutation loop (NiMARE-style: generator + tqdm for progress) ---
    raw_results = [
        r for r in tqdm(
            Parallel(return_as="generator", n_jobs=n_proc)(
                delayed(_run_one_perm)(
                    iter_ijks[i], iter_xyzs[i], iter_df, est,
                    parc_arr_precomp, parc_idc_precomp, bg_values_precomp,
                    _voxel_thresh_z, None, cluster_stat,   # None: no FWE extent filter on nulls
                    _return_cluster_stat,
                )
                for i in range(n_perm)
            ),
            total=n_perm,
            disable=not verbose,
            desc="NiMARE null maps",
        )
    ]

    if _return_cluster_stat:
        null_maps_list = [r[0] for r in raw_results]
        max_stats = [r[1] for r in raw_results]
    else:
        null_maps_list = raw_results
        max_stats = None

    # --- stack ---
    null_arr = np.stack(null_maps_list, axis=0).astype(dtype)  # (n_perm, n_parcels)
    label = map_label or outcome_map
    lgr.info(f"Done. null_maps['{label}'] shape: {null_arr.shape}")

    # --- degeneracy check (always on, independent of validate) ---
    _check_null_maps_degenerate(
        null_arr,
        obs_parc=(
            None if _voxel_thresh_z is not None else
            vol_to_vect_arr(
                result.get_map(outcome_map).get_fdata().astype(np.float32),
                parc_arr_precomp, parc_idc_precomp, bg_values_precomp,
            ).astype(dtype)
        ),
    )

    # --- empty-map check ---
    empty_frac = float((null_arr == 0).all(axis=1).mean())
    if empty_frac > 0.5:
        lgr.warning(
            f"{100 * empty_frac:.0f}% of null maps are entirely zero. "
            "Colocalization against all-zero maps is undefined. "
            "Consider switching to continuous mode (corrected_result=None)."
        )
    elif empty_frac > 0.1:
        lgr.warning(
            f"{100 * empty_frac:.0f}% of null maps are entirely zero "
            "(no CDT-surviving cluster in the permutation). "
            "This is expected for datasets with few coordinates."
        )

    # --- validation ---
    if validate:
        _validate_null_maps(
            null_arr, max_stats, result, outcome_map,
            parc_arr_precomp, parc_idc_precomp,
            _voxel_thresh_z, _min_cluster_extent, cluster_stat, alpha,
            observed_map, baseline_used=baseline_weights is not None,
        )

    return {label: null_arr}


def nimare_fwe_thresholds(corrected_result, alpha=0.05, cluster_stat="size", voxel_thresh=0.001):
    """Extract voxel z-threshold and cluster extent threshold from a NiMARE FWE-corrected result.

    Reads ``corrected_result.estimator.null_distributions_`` (populated during
    ``FWECorrector.transform()``) to derive the FWE cluster extent threshold.
    The voxel-level CDT z-score is derived from *voxel_thresh* directly —
    NiMARE's ``FWECorrector`` silently ignores any ``voxel_thresh`` passed to
    its constructor, so it must be provided explicitly here.

    Parameters
    ----------
    corrected_result : nimare.results.MetaResult
        The corrected MetaResult returned by ``FWECorrector.transform(result)``.
        The cluster-size null distributions are stored on its estimator
        (``corrected_result.estimator.null_distributions_``).
        Only ``method="montecarlo"`` is supported.
    alpha : float, optional
        FWE significance level used to derive the cluster extent threshold from
        the null distribution. Default: 0.05.
    cluster_stat : {"size", "mass"}, optional
        Which cluster-level null distribution to read. ``"size"`` (default)
        uses the max-cluster-voxel-count distribution; ``"mass"`` uses the
        max-cluster-z-sum distribution. Must match the statistic used to
        produce the observed cluster image. Default: ``"size"``.
    voxel_thresh : float, optional
        Uncorrected voxel-level p-value for the cluster-defining threshold
        (CDT). Must match the threshold actually used during FWE correction.
        NiMARE's default (and the default here) is ``0.001`` (z ≈ 3.09).

    Returns
    -------
    voxel_thresh_z : float
        Z-score corresponding to *voxel_thresh*
        (``scipy.stats.norm.isf(voxel_thresh)``).
        E.g. ``voxel_thresh=0.001`` → z ≈ 3.09.
    min_cluster_extent : int or float or None
        FWE cluster extent threshold: the ``(1 - alpha)`` percentile of the
        max-cluster-stat null distribution. Voxel count (``int``) for
        ``cluster_stat="size"``; z-score sum (``float``) for
        ``cluster_stat="mass"``. ``None`` if the relevant distribution is not
        found in ``corrected_result.estimator.null_distributions_``.

    Examples
    --------
    >>> fwe = FWECorrector(method="montecarlo", n_iters=1000)
    >>> corrected = fwe.transform(result)
    >>> z_thresh, min_ext = nimare_fwe_thresholds(corrected, cluster_stat="size", voxel_thresh=0.001)
    >>> binary_img = get_binary_cluster_map(corrected, cluster_stat="size")
    >>> null_maps = null_maps_from_nimare(result, nsp._parc,
    ...                                   corrected_result=corrected, voxel_thresh=0.001)
    """
    from scipy.stats import norm

    voxel_thresh_z = float(norm.isf(voxel_thresh))
    lgr.info(f"Voxel z-threshold (p<{voxel_thresh}): {voxel_thresh_z:.4f}")

    null_dists = getattr(
        getattr(corrected_result, "estimator", None), "null_distributions_", {}
    )
    cluster_keys = [
        k for k in null_dists
        if f"desc-{cluster_stat}" in k and "values" in k
    ]
    min_cluster_extent = None
    if cluster_keys:
        null_vals = np.asarray(null_dists[cluster_keys[0]])
        raw = np.percentile(null_vals, 100 * (1 - alpha))
        min_cluster_extent = int(np.ceil(raw)) if cluster_stat == "size" else float(raw)
        lgr.info(
            f"Cluster {cluster_stat} threshold (FWE alpha={alpha}): {min_cluster_extent}."
        )
    else:
        lgr.warning(
            f"No cluster-{cluster_stat} null distribution found in "
            "corrected_result.estimator.null_distributions_. "
            "min_cluster_extent will be None."
        )

    return voxel_thresh_z, min_cluster_extent


def get_binary_cluster_map(corrected_result, cluster_stat="size", alpha=0.05, map_key=None):
    """Extract a binary (0/1) cluster mask from a NiMARE FWE-corrected MetaResult.

    NiMARE's cluster-corrected logp maps (e.g.
    ``logp_desc-size_level-cluster_corr-FWE_method-montecarlo``) store a
    cluster-level ``-log10(p_FWE)`` value at every voxel that survived the
    voxel-level CDT, **including non-significant clusters**. Voxels in
    non-significant clusters have ``0 < logp < -log10(alpha)``. This function
    binarizes correctly by thresholding at ``logp >= -log10(alpha)``.

    Pass the returned image as ``y=`` to :class:`~nispace.NiSpace` and use
    :func:`null_maps_from_nimare` with the same *corrected_result*, *cluster_stat*,
    and *alpha* to generate matching binary null maps for permutation testing.

    Parameters
    ----------
    corrected_result : nimare.results.MetaResult
        Result of ``FWECorrector.transform(result)``.
    cluster_stat : {"size", "mass"}, optional
        Which cluster-corrected map to binarize. ``"size"`` (default) uses
        ``logp_desc-size_level-cluster_corr-*``; ``"mass"`` uses
        ``logp_desc-mass_level-cluster_corr-*``. Must match the *cluster_stat*
        passed to :func:`null_maps_from_nimare`.
    alpha : float, optional
        FWE significance threshold. Voxels with ``logp >= -log10(alpha)`` are
        kept. Default: 0.05. Must match the *alpha* passed to
        :func:`null_maps_from_nimare`.
    map_key : str or None, optional
        Override: explicit key in ``corrected_result.maps`` to binarize,
        bypassing the *cluster_stat* search. Pass when NiMARE uses a
        non-standard naming scheme.

    Returns
    -------
    nibabel.Nifti1Image
        Binary NIfTI image (float32, values 0 or 1) in NiMARE's native space
        (MNI152NLin6Asym at 2 mm).

    Examples
    --------
    >>> fwe = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=1000)
    >>> corrected = fwe.transform(result)
    >>> binary_img = get_binary_cluster_map(corrected, alpha=0.05)
    >>> nsp = NiSpace(x=ref_maps, y=binary_img, y_labels=["pain_cluster"],
    ...               parcellation="Schaefer200")
    >>> nsp.fit(); nsp.colocalize()
    >>> null_maps = null_maps_from_nimare(
    ...     result, "Schaefer200", corrector=fwe, alpha=0.05,
    ...     map_label="pain_cluster", observed_map=binary_img,
    ... )
    >>> nsp.permute("maps", maps_which="Y", maps_nulls=null_maps)
    """
    if map_key is None:
        available = [k for k, v in corrected_result.maps.items() if v is not None]
        candidates = [
            k for k in available
            if f"logp_desc-{cluster_stat}" in k and "cluster_corr" in k
        ]
        if not candidates and cluster_stat == "size":
            candidates = [k for k in available if "logp_desc-mass" in k and "cluster_corr" in k]
        if not candidates:
            raise ValueError(
                "Could not find a cluster-corrected logp map in corrected_result.maps. "
                f"Available keys: {available}. Pass map_key= explicitly."
            )
        if len(candidates) > 1:
            lgr.warning(
                f"Multiple cluster logp maps found: {candidates}. Using '{candidates[0]}'. "
                "Pass map_key= to select explicitly."
            )
        map_key = candidates[0]

    img = corrected_result.get_map(map_key)
    arr = img.get_fdata()
    logp_thresh = -np.log10(alpha)
    binary = (arr >= logp_thresh).astype(np.float32)
    lgr.info(
        f"Binary cluster map from '{map_key}' "
        f"(logp >= {logp_thresh:.4f}, alpha={alpha}): "
        f"{int(binary.sum())} significant voxels."
    )
    return nib.Nifti1Image(binary, img.affine, img.header)
