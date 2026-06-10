
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd

from neuromaps.images import load_data

import logging
lgr = logging.getLogger(__name__)
from ..nulls import (
    _img_space_for_neuromaps, _img_density_for_neuromaps,
    find_parcel_hemispheres, get_distance_matrix,
)
from ..io import load_distmat, load_spinmat, load_img, load_labels, load_l2rmap
from ..utils.utils import set_log, relabel_nifti_parc, relabel_gifti_parc


# ---------------------------------------------------------------------------
# Space-name normalization helpers
# ---------------------------------------------------------------------------

_SPACE_NORM = {
    "mni": "mni", "mni152": "mni", "mni152nlin2009casym": "mni",
    "mni152nlin6asym": "mni", "mni152lin": "mni",
    "fsa": "fsaverage", "fsaverage": "fsaverage",
    "fslr": "fslr", "fs_lr": "fslr",
}

def _norm_space(space):
    if space is None:
        return None
    return _SPACE_NORM.get(space.lower().replace(" ", "").replace("-", ""), space.lower())


def _spaces_match(query, candidate):
    """Return True when query space is compatible with candidate space name."""
    q = _norm_space(query)
    c = _norm_space(candidate)
    if q == c:
        return True
    # "mni" matches any MNI variant
    if q == "mni" and "mni" in candidate.lower():
        return True
    if c == "mni" and "mni" in query.lower():
        return True
    return False


# ---------------------------------------------------------------------------
# Bilateral label symmetry helper
# ---------------------------------------------------------------------------

def _bilateral_labels_match(labels, lh_prefix="hemi-L_", rh_prefix="hemi-R_"):
    """Match LH and RH labels by stripping hemisphere prefixes.

    Finds labels that start with *lh_prefix* or *rh_prefix*, strips those
    prefixes, and pairs them by name.  Does not assume any particular ordering
    or that the count of LH and RH labels is equal a priori.

    Parameters
    ----------
    labels : sequence of str
    lh_prefix : str
        Prefix that identifies left-hemisphere labels.
    rh_prefix : str
        Prefix that identifies right-hemisphere labels.

    Returns
    -------
    ok : bool
        True when every LH label has exactly one matching RH label and
        vice versa (and at least one pair was found).
    lh_idc : np.ndarray
        0-based indices into *labels* for the matched LH parcels (in LH order).
    rh_idc : np.ndarray
        0-based indices into *labels* for the matched RH parcels,
        in the same order as *lh_idc* (i.e. ``lh_idc[i]`` ↔ ``rh_idc[i]``).
    bilateral_labels : list[str]
        Stripped label names in matched order.
    unmatched : list[str]
        Stripped names that could not be paired.
    """
    labels = [str(l) for l in labels]

    lh_pairs = [(i, l[len(lh_prefix):]) for i, l in enumerate(labels) if l.startswith(lh_prefix)]
    rh_pairs = [(i, l[len(rh_prefix):]) for i, l in enumerate(labels) if l.startswith(rh_prefix)]

    rh_lookup = {name: idx for idx, name in rh_pairs}
    lh_names  = {name for _, name in lh_pairs}

    lh_idc, rh_idc, bilateral_labels, unmatched = [], [], [], []
    for lh_idx, lh_name in lh_pairs:
        if lh_name in rh_lookup:
            lh_idc.append(lh_idx)
            rh_idc.append(rh_lookup[lh_name])
            bilateral_labels.append(lh_name)
        else:
            unmatched.append(lh_name)

    for _, rh_name in rh_pairs:
        if rh_name not in lh_names:
            unmatched.append(rh_name)

    ok = len(lh_pairs) > 0 and len(unmatched) == 0 and len(lh_idc) == len(rh_idc) == len(lh_pairs) == len(rh_pairs)
    return ok, np.array(lh_idc), np.array(rh_idc), bilateral_labels, unmatched


# ---------------------------------------------------------------------------
# Parcellation class
# ---------------------------------------------------------------------------

class Parcellation:
    """Multi-space parcellation container.

    Stores parcellation images, distance matrices, and spin matrices for
    one or more brain spaces (e.g., MNI152, fsaverage, fsLR).  A single
    *active space* (`_space`) is set by `set_active_space()` and is used by
    all backward-compatible single-value properties (`_image_obj`, `_dist_mat`,
    `_idc_byhemi`, etc.).

    Typical construction:
    - Custom parcellations  → `Parcellation.from_path(...)`
    - Integrated library    → `Parcellation.from_nispace_library(...)`
    - Legacy (deprecated)   → `Parcellation(...).fit()`
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        # legacy single-space init (kept for backward compat)
        parcellation=None, space=None, labels=None, resolution=None,
        hemi=None, symmetric=False, left2right_mapping=None, lrcorr=None,
        labels_lh=None, labels_rh=None, labels_img_lh=None, labels_img_rh=None,
        idc_lh=None, idc_rh=None, dist_mat=None, spin_mat=None, name=None,
        # multi-space / combined
        level=None, is_combined=False, cx_name=None, sc_name=None,
        # hemisphere prefix strings used by make_bilateral / _bilateral_labels_match
        lh_prefix="hemi-L_", rh_prefix="hemi-R_",
    ):
        # --- shared (space-independent) ---
        self._name = name
        self._level = level          # "cortex" | "subcortex" | "combined"
        self._is_combined = is_combined
        self._cx_name = cx_name      # name of cortex component (combined only)
        self._sc_name = sc_name      # name of subcortex component (combined only)
        self._n_cx_labels = None     # number of cx labels in combined parcellation
        self._sc_dist_mat = None     # loaded sc dist_mat (combined only)
        self._sc_dist_mat_spec = None  # lazy-load spec for sc dist_mat (combined only)
        self._cx_dist_mat = None     # loaded cx dist_mat (combined only)
        self._cx_dist_mat_spec = None  # lazy-load spec for cx dist_mat (combined only)
        self._labels = np.array(labels) if labels is not None else None
        self._symmetric = symmetric
        self._l2rmap = left2right_mapping
        self._lrcorr = lrcorr

        # --- per-space storage (keyed by space name string) ---
        # Values in _images may be: loaded nib objects OR path strings/tuples
        # (lazy-loaded on first access via set_active_space / get_surface_for_spins)
        self._images   = {}   # {space: nib.Nifti1Image | (lh_gii, rh_gii) | path | (path, path)}
        self._dist_mats = {}  # {space: ndarray | (ndarray, ndarray) | None}
        self._spin_mats = {}  # {space: (spins_lh, spins_rh) | None}

        # derived per-space (populated by _fit_space)
        self._idc_byhemi_dict        = {}  # {space: {"L": arr, "R": arr}}
        self._labels_byhemi_dict     = {}  # {space: {"L": arr, "R": arr}}
        self._labels_img_dict        = {}  # {space: arr of integer parcel labels}
        self._labels_img_byhemi_dict = {}  # {space: {"L": arr, "R": arr}}
        self._hemi_dict              = {}  # {space: ("L", "R") | None}
        self._resolution_dict        = {}  # {space: str}
        self._is_surface_dict        = {}  # {space: bool}

        # --- combined-specific ---
        self._cx_idc_lh   = None    # cx-LH parcel indices within the combined data vector
        self._cx_idc_rh   = None    # cx-RH parcel indices within the combined data vector
        self._cx_symmetric = None   # symmetry of the cx component (combined only)
        self._sc_symmetric = None   # symmetry of the sc component (combined only)
        # {space: {"img_paths": ..., "image": None|loaded, "spin_mat": ...}}
        self._cx_surface = {}

        # --- bilateral ---
        self._bilateral  = False    # True after make_bilateral()
        self._lh_prefix  = lh_prefix
        self._rh_prefix  = rh_prefix

        # --- hemisphere selection (set by select_hemi) ---
        self._selected_hemi = None   # "L" or "R" after select_hemi()

        # --- active context (set by NiSpace.fit via set_active_space) ---
        self._space = space  # may be None until activated

        # ---- legacy single-space init support ----
        self._legacy_source  = parcellation
        self._legacy_hemi    = hemi
        self._legacy_dist_mat = dist_mat
        self._legacy_spin_mat = spin_mat
        self._legacy_res     = resolution
        # partial idc / labels provided at __init__ time (legacy API)
        self._legacy_idc_lh  = idc_lh
        self._legacy_idc_rh  = idc_rh
        self._legacy_labels_lh  = labels_lh
        self._legacy_labels_rh  = labels_rh
        self._legacy_labels_img_lh = labels_img_lh
        self._legacy_labels_img_rh = labels_img_rh

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_path(
        cls, source, space=None, labels=None, dist_mat=None, spin_mat=None,
        symmetric=False, l2rmap=None, lrcorr=None, hemi=None, name=None,
        level=None, verbose=True,
    ):
        """Build a single-space Parcellation from a user-provided image.

        Parameters
        ----------
        source : str, Path, nib.Nifti1Image, nib.GiftiImage, or tuple thereof
        space  : str, optional – inferred from image type if not given
        """
        set_log(lgr, verbose)
        lgr.info(f"Building Parcellation from path / image{f' ({name})' if name else ''}.")
        p = cls(name=name, level=level, symmetric=symmetric,
                left2right_mapping=l2rmap, lrcorr=lrcorr)

        # load image
        image = load_img(source)
        if space is None:
            try:
                space = _img_space_for_neuromaps(image)
            except Exception:
                space = "custom"
                lgr.warning("Could not infer parcellation space from image. Setting space='custom'.")
        lgr.info(f"Parcellation space: '{space}'.")

        # warn on obvious space/image-type mismatch
        _is_surf_img = isinstance(image, tuple) or (
            hasattr(image, "darrays")  # nib.GiftiImage
        )
        _space_looks_mni = "mni" in space.lower()
        _space_looks_surf = any(k in space.lower() for k in ("fslr", "fsaverage", "fsa"))
        if _is_surf_img and _space_looks_mni:
            lgr.warning(
                f"Parcellation.from_path: image appears to be a surface (GIfTI) "
                f"but space='{space}' looks like an MNI/volumetric space. "
                "This may cause errors downstream."
            )
        elif not _is_surf_img and _space_looks_surf:
            lgr.warning(
                f"Parcellation.from_path: image appears to be volumetric (NIfTI) "
                f"but space='{space}' looks like a surface space. "
                "This may cause errors downstream."
            )

        # labels
        if labels is not None:
            p._labels = np.array(load_labels(labels))

        # add space and fit
        p.add_space(space, image=image, dist_mat=dist_mat, spin_mat=spin_mat)
        p._fit_space(space)

        # set active space
        p._space = space

        # infer level if not provided
        if p._level is None:
            p._level = "cortex" if p._is_surface else "unknown"

        # hemi override
        if hemi is not None:
            h = (hemi,) if isinstance(hemi, str) else tuple(hemi)
            p._hemi_dict[space] = h

        p.validate()
        return p

    @classmethod
    def from_nispace_library(
        cls, name_or_list, lib_entry_or_list, data_dir,
        load_dist_mat=True, load_spin_mat=True, lrcorr_threshold=0.0,
        overwrite=False, check_file_hash=True, verbose=True,
    ):
        """Build a multi-space Parcellation from the NiSpace parcellation library.

        Parameters
        ----------
        name_or_list      : str or [cx_name, sc_name]
        lib_entry_or_list : dict or [cx_lib_entry, sc_lib_entry]
        data_dir          : str or Path  – root NiSpace data directory
        """
        set_log(lgr, verbose)
        from ..utils.utils_datasets import get_file
        from ..utils.utils import merge_parcellations

        data_dir =Path(data_dir)
        gf_kw = dict(overwrite=overwrite, hash_check=check_file_hash)
        is_combined = isinstance(name_or_list, list)

        if is_combined:
            return cls._from_nispace_library_combined(
                name_or_list, lib_entry_or_list, data_dir,
                load_dist_mat, load_spin_mat, lrcorr_threshold,
                gf_kw, verbose,
            )
        else:
            return cls._from_nispace_library_single(
                name_or_list, lib_entry_or_list, data_dir,
                load_dist_mat, load_spin_mat, lrcorr_threshold,
                gf_kw, verbose,
            )

    @classmethod
    def _from_nispace_library_single(
        cls, name, lib_entry, data_dir, load_dist_mat, load_spin_mat,
        lrcorr_threshold, gf_kw, verbose,
    ):
        """Build a single (non-combined) multi-space Parcellation from the library."""
        from ..utils.utils_datasets import get_file

        # Filter to actual space entries (dicts with a "map" key); skip top-level metadata fields
        # (label, level, symmetric, license, citation) added in the datalib refactor
        space_entries = {k: v for k, v in lib_entry.items() if isinstance(v, dict) and "map" in v}
        # Top-level metadata (new in refactor; None for old JSON without these fields)
        entry_level = lib_entry.get("level")
        entry_symmetric = lib_entry.get("symmetric")
        entry_doi = lib_entry.get("citation", {}).get("doi", "")
        spaces = list(space_entries.keys())
        lgr.info(
            f"Building {entry_level or ''} Parcellation for '{name}' from library."
            + (f" DOI: {entry_doi}" if entry_doi else "")
        )
        lgr.info(f"Available spaces: {', '.join(spaces)}")
        p = cls(name=name)

        shared_loaded = False  # load labels/l2rmap/lrcorr only once

        for space, space_lib in space_entries.items():
            is_vol = "mni" in space.lower()
            base   = data_dir / "parcellation" / name / space

            # ---- image ----
            if is_vol:
                img_path = get_file(
                    base / f"parc-{name}_space-{space}.%s",
                    **space_lib["map"], **gf_kw,
                )
            else:
                img_path = tuple(
                    get_file(
                        base / f"parc-{name}_space-{space}_hemi-{h}.%s",
                        **space_lib["map"][h], **gf_kw,
                    )
                    for h in ["L", "R"]
                )

            # ---- shared data (first space only) ----
            if not shared_loaded:
                if "label" in space_lib:
                    if is_vol:
                        label_path = get_file(
                            base / f"parc-{name}_space-{space}.label.txt",
                            **space_lib["label"], **gf_kw,
                        )
                        p._labels = np.array(load_labels(label_path))
                    else:
                        label_paths = tuple(
                            get_file(
                                base / f"parc-{name}_space-{space}_hemi-{h}.label.txt",
                                **space_lib["label"][h], **gf_kw,
                            )
                            for h in ["L", "R"]
                        )
                        p._labels = np.array(load_labels(label_paths))

                sym = (
                    entry_symmetric
                    if entry_symmetric is not None
                    else space_lib.get(
                        "symmetric",
                        "l2rmap" not in space_lib and "lrcorr" not in space_lib,
                    )
                )
                p._symmetric = sym

                if not sym and "l2rmap" in space_lib:
                    l2r_path = get_file(
                        base / f"parc-{name}_space-{space}.l2rmap.csv.gz",
                        **space_lib["l2rmap"], **gf_kw,
                    )
                    p._l2rmap = load_l2rmap(l2r_path, threshold=lrcorr_threshold)

                if not sym and "lrcorr" in space_lib:
                    lrc_path = get_file(
                        base / f"parc-{name}_space-{space}.lrcorr.csv.gz",
                        **space_lib["lrcorr"], **gf_kw,
                    )
                    p._lrcorr = load_l2rmap(lrc_path, threshold=lrcorr_threshold)

                p._level = entry_level or space_lib.get("level", None)
                shared_loaded = True

            # ---- distance matrix ----
            dm = None
            if "distmat" in space_lib:
                if is_vol:
                    dm_path_template = base / f"parc-{name}_space-{space}.dist.csv.gz"
                    if load_dist_mat:
                        dm_local = get_file(dm_path_template, **space_lib["distmat"], **gf_kw)
                        dm = load_distmat(dm_local)
                    else:
                        dm = {"path_template": dm_path_template,
                              "spec": space_lib["distmat"], "gf_kw": gf_kw}
                else:
                    dm_specs = []
                    has_all = True
                    for h in ["L", "R"]:
                        if space_lib["distmat"].get(h) is not None:
                            dm_specs.append({
                                "path_template": base / f"parc-{name}_space-{space}_hemi-{h}.dist.csv.gz",
                                "spec": space_lib["distmat"][h], "gf_kw": gf_kw,
                            })
                        else:
                            lgr.info(f"  Distance matrix for '{name}' hemi-{h} in '{space}' not available.")
                            has_all = False
                            dm_specs.append(None)
                    if has_all:
                        if load_dist_mat:
                            dm_paths = tuple(
                                get_file(s["path_template"], **s["spec"], **s["gf_kw"])
                                for s in dm_specs
                            )
                            dm = load_distmat(dm_paths)
                        else:
                            dm = tuple(dm_specs)

            # ---- spin matrix (surface only) ----
            sm = None
            if not is_vol and "spinmat" in space_lib:
                # derive local extension from remote path (.npy legacy or .npz new)
                _sm_ext = Path(space_lib["spinmat"]["L"]["remote"]).suffix
                if load_spin_mat:
                    sm_paths = tuple(
                        get_file(
                            base / f"parc-{name}_space-{space}_hemi-{h}.spin{_sm_ext}",
                            **space_lib["spinmat"][h], **gf_kw,
                        )
                        for h in ["L", "R"]
                    )
                    sm = load_spinmat(sm_paths)
                else:
                    sm = tuple(
                        {"path_template": base / f"parc-{name}_space-{space}_hemi-{h}.spin{_sm_ext}",
                         "spec": space_lib["spinmat"][h], "gf_kw": gf_kw}
                        for h in ["L", "R"]
                    )
            elif not is_vol:
                lgr.debug(f"  No pre-computed spin matrix for '{name}' in '{space}'.")

            # ---- store image path (lazy load) ----
            p.add_space(space, image=img_path, dist_mat=dm, spin_mat=sm)

        p.validate(pre_activation=True)
        return p

    @classmethod
    def _from_nispace_library_combined(
        cls, names, lib_entries, data_dir, load_dist_mat, load_spin_mat,
        lrcorr_threshold, gf_kw, verbose,
    ):
        """Build a combined (cx+sc) multi-space Parcellation from the library."""
        set_log(lgr, verbose)
        from ..utils.utils_datasets import get_file
        from ..utils.utils import merge_parcellations

        cx_name, sc_name = names[0], names[1]
        cx_lib, sc_lib   = lib_entries[0], lib_entries[1]

        lgr.info(f"Building combined Parcellation '{cx_name}+{sc_name}' from library.")

        # find common MNI spaces — filter to actual space entries (dicts with "map" key);
        # top-level metadata keys (label, level, symmetric, license, citation) are skipped
        cx_spaces = [k for k, v in cx_lib.items() if isinstance(v, dict) and "map" in v]
        sc_spaces = [k for k, v in sc_lib.items() if isinstance(v, dict) and "map" in v]
        common_mni = [s for s in cx_spaces if "mni" in s.lower() and s in sc_spaces]
        if not common_mni:
            lgr.critical_raise(
                f"No common MNI space found for cx='{cx_name}' and sc='{sc_name}'. "
                f"cx spaces: {cx_spaces}, sc spaces: {sc_spaces}",
                ValueError,
            )
        lgr.info(f"  Common MNI space(s) for combined: {common_mni}.")

        # TODO (combined parc naming): migrate to "+" separator, e.g. f"{cx_name}+{sc_name}",
        # and support tuple input ("Schaefer100", "TianS1") in fetch_parcellation/_check_parcellation.
        combined_name = f"{cx_name}{sc_name}"
        p = cls(name=combined_name, level="combined", is_combined=True,
                cx_name=cx_name, sc_name=sc_name)

        for mni_space in common_mni:
            base_cx = data_dir / "parcellation" / cx_name / mni_space
            base_sc = data_dir / "parcellation" / sc_name / mni_space

            # load cx image
            cx_img_path = get_file(
                base_cx / f"parc-{cx_name}_space-{mni_space}.%s",
                **cx_lib[mni_space]["map"], **gf_kw,
            )
            cx_img = load_img(cx_img_path)

            # load sc image
            sc_img_path = get_file(
                base_sc / f"parc-{sc_name}_space-{mni_space}.%s",
                **sc_lib[mni_space]["map"], **gf_kw,
            )
            sc_img = load_img(sc_img_path)

            # merge
            lgr.info(f"  Merging '{cx_name}' and '{sc_name}' for space '{mni_space}'.")
            merged_img = merge_parcellations([cx_img, sc_img], quick=True)

            # shared metadata (first MNI space only)
            if p._labels is None:
                cx_label_path = get_file(
                    base_cx / f"parc-{cx_name}_space-{mni_space}.label.txt",
                    **cx_lib[mni_space]["label"], **gf_kw,
                )
                sc_label_path = get_file(
                    base_sc / f"parc-{sc_name}_space-{mni_space}.label.txt",
                    **sc_lib[mni_space]["label"], **gf_kw,
                )
                cx_labels = load_labels(cx_label_path)
                sc_labels = load_labels(sc_label_path)
                p._labels = np.array(cx_labels + sc_labels)
                p._n_cx_labels = len(cx_labels)

                # symmetry: read from top-level JSON field (post-refactor) or fall back to
                # space-level key / l2rmap/lrcorr presence check (pre-refactor compat)
                cx_sym = (
                    cx_lib["symmetric"]
                    if "symmetric" in cx_lib
                    else cx_lib[mni_space].get(
                        "symmetric",
                        "l2rmap" not in cx_lib[mni_space] and "lrcorr" not in cx_lib[mni_space],
                    )
                )
                sc_sym = (
                    sc_lib["symmetric"]
                    if "symmetric" in sc_lib
                    else sc_lib[mni_space].get(
                        "symmetric",
                        "l2rmap" not in sc_lib[mni_space] and "lrcorr" not in sc_lib[mni_space],
                    )
                )
                p._cx_symmetric = cx_sym
                p._sc_symmetric = sc_sym
                p._symmetric = cx_sym and sc_sym

                if not cx_sym and "l2rmap" in cx_lib[mni_space]:
                    l2r_path = get_file(
                        base_cx / f"parc-{cx_name}_space-{mni_space}.l2rmap.csv.gz",
                        **cx_lib[mni_space]["l2rmap"], **gf_kw,
                    )
                    p._l2rmap = load_l2rmap(l2r_path, threshold=lrcorr_threshold)

                if not cx_sym and "lrcorr" in cx_lib[mni_space]:
                    lrc_path = get_file(
                        base_cx / f"parc-{cx_name}_space-{mni_space}.lrcorr.csv.gz",
                        **cx_lib[mni_space]["lrcorr"], **gf_kw,
                    )
                    p._lrcorr = load_l2rmap(lrc_path, threshold=lrcorr_threshold)

            p.add_space(mni_space, image=merged_img, dist_mat=None, spin_mat=None)

        # ---- component dist_mat lazy specs (split-null avoids recomputing full combined dist_mat) ----
        for pref_sc_space in ["MNI152NLin6Asym", "MNI152NLin2009cAsym"]:
            if pref_sc_space in common_mni and "distmat" in sc_lib.get(pref_sc_space, {}):
                p._sc_dist_mat_spec = {
                    "space": pref_sc_space,
                    "path_template": (
                        data_dir / "parcellation" / sc_name / pref_sc_space
                        / f"parc-{sc_name}_space-{pref_sc_space}.dist.csv.gz"
                    ),
                    "spec": sc_lib[pref_sc_space]["distmat"],
                    "gf_kw": gf_kw,
                }
                break
        # Prefer surface geodesic dist_mat (fsLR/fsaverage) for cx; fall back to MNI Euclidean.
        # Surface spaces store per-hemi files (L+R) → use "hemi_specs" key instead of "path_template".
        for pref_cx_space in ["fsLR", "fsaverage", "MNI152NLin6Asym", "MNI152NLin2009cAsym"]:
            cx_space_lib = cx_lib.get(pref_cx_space, {})
            if "distmat" not in cx_space_lib:
                continue
            dm_entry = cx_space_lib["distmat"]
            _is_hemi = isinstance(dm_entry, dict) and any(h in dm_entry for h in ("L", "R"))
            base_cx = data_dir / "parcellation" / cx_name / pref_cx_space
            if _is_hemi:
                p._cx_dist_mat_spec = {
                    "space": pref_cx_space,
                    "hemi_specs": tuple(
                        {
                            "path_template": base_cx / f"parc-{cx_name}_space-{pref_cx_space}_hemi-{h}.dist.csv.gz",
                            "spec": dm_entry[h],
                            "gf_kw": gf_kw,
                        }
                        for h in ("L", "R")
                        if dm_entry.get(h) is not None
                    ),
                }
            else:
                p._cx_dist_mat_spec = {
                    "space": pref_cx_space,
                    "path_template": base_cx / f"parc-{cx_name}_space-{pref_cx_space}.dist.csv.gz",
                    "spec": dm_entry,
                    "gf_kw": gf_kw,
                }
            break

        # ---- cx surface data for future split-null support ----
        cx_surface_spaces = [s for s in cx_spaces if "mni" not in s.lower()]
        for surf_space in cx_surface_spaces:
            lgr.info(f"  Fetching cx surface data for '{cx_name}' in '{surf_space}' (for spin tests).")
            base_cx_surf = data_dir / "parcellation" / cx_name / surf_space
            cx_img_paths = tuple(
                get_file(
                    base_cx_surf / f"parc-{cx_name}_space-{surf_space}_hemi-{h}.%s",
                    **cx_lib[surf_space]["map"][h], **gf_kw,
                )
                for h in ["L", "R"]
            )
            cx_sm = None
            if "spinmat" in cx_lib[surf_space]:
                _cx_sm_ext = Path(cx_lib[surf_space]["spinmat"]["L"]["remote"]).suffix
                if load_spin_mat:
                    cx_sm_paths = tuple(
                        get_file(
                            base_cx_surf / f"parc-{cx_name}_space-{surf_space}_hemi-{h}.spin{_cx_sm_ext}",
                            **cx_lib[surf_space]["spinmat"][h], **gf_kw,
                        )
                        for h in ["L", "R"]
                    )
                    cx_sm = load_spinmat(cx_sm_paths)
                else:
                    cx_sm = tuple(
                        {"path_template": base_cx_surf / f"parc-{cx_name}_space-{surf_space}_hemi-{h}.spin{_cx_sm_ext}",
                         "spec": cx_lib[surf_space]["spinmat"][h], "gf_kw": gf_kw}
                        for h in ["L", "R"]
                    )
            p._cx_surface[surf_space] = {
                "img_paths": cx_img_paths,
                "image": None,   # lazy-loaded on first access
                "spin_mat": cx_sm,
            }

        lgr.info(
            f"Combined parcellation '{combined_name}' ready. "
            f"MNI space(s): {common_mni}. "
            f"Cx surface space(s) for spins: {cx_surface_spaces}."
        )
        p.validate(pre_activation=True)
        return p

    # ------------------------------------------------------------------
    # Legacy fit() — keeps backward compat for Parcellation(...).fit()
    # ------------------------------------------------------------------

    def fit(self):
        """Single-space fit from legacy __init__ arguments (backward compat)."""
        if self._legacy_source is None:
            lgr.warning("Parcellation.fit() called but no source image was provided.")
            return self

        img  = load_img(self._legacy_source)
        space = self._space if self._space is not None else _img_space_for_neuromaps(img)

        # load legacy dist_mat / spin_mat
        dm = load_distmat(self._legacy_dist_mat) if self._legacy_dist_mat is not None else None
        sm = self._legacy_spin_mat  # already loaded or None

        # populate
        self.add_space(space, image=img, dist_mat=dm, spin_mat=sm)

        # apply legacy idc / labels if provided externally at init time
        if self._legacy_idc_lh is not None or self._legacy_idc_rh is not None:
            self._idc_byhemi_dict[space] = {
                "L": self._legacy_idc_lh,
                "R": self._legacy_idc_rh,
            }
        if self._legacy_labels_lh is not None or self._legacy_labels_rh is not None:
            self._labels_byhemi_dict[space] = {
                "L": self._legacy_labels_lh,
                "R": self._legacy_labels_rh,
            }
        if self._legacy_labels_img_lh is not None or self._legacy_labels_img_rh is not None:
            self._labels_img_byhemi_dict[space] = {
                "L": self._legacy_labels_img_lh,
                "R": self._legacy_labels_img_rh,
            }

        # infer level
        if self._level is None:
            is_surf = isinstance(img, (nib.GiftiImage, tuple))
            self._level = "cortex" if is_surf else "unknown"

        self._fit_space(space)
        self._space = space
        return self

    # ------------------------------------------------------------------
    # Space management
    # ------------------------------------------------------------------

    def add_space(self, space, image=None, dist_mat=None, spin_mat=None):
        """Add data for one parcellation space.

        `image` may be a loaded nib object, a path string, or a tuple thereof.
        Image loading is deferred until the space is activated or accessed.
        """
        if image is not None:
            self._images[space] = image
        if dist_mat is not None:
            self._dist_mats[space] = dist_mat
        if spin_mat is not None:
            self._spin_mats[space] = spin_mat

    def _ensure_image_loaded(self, space):
        """Load image for *space* from stored path if not yet loaded."""
        img = self._images.get(space)
        if img is None:
            lgr.critical_raise(
                f"No image available for parcellation space '{space}' "
                f"(available: {self.spaces}).",
                KeyError,
            )
        is_path = isinstance(img, (str,Path)) or (
            isinstance(img, tuple) and isinstance(img[0], (str,Path))
        )
        if is_path:
            lgr.info(f"Lazy-loading parcellation image for space '{space}'.")
            self._images[space] = load_img(img)

    def _ensure_dist_mat_loaded(self, space):
        """Load dist mat for *space* from stored path/spec if not yet loaded."""
        dm = self._dist_mats.get(space)
        if dm is None:
            return
        if isinstance(dm, (np.ndarray, tuple)) and not (
            isinstance(dm, tuple) and dm and isinstance(dm[0], (str,Path, dict))
        ):
            return  # already loaded
        # lazy-load: dm is a path, tuple of paths, or a lazy-spec dict / tuple of dicts
        lgr.info(f"Lazy-loading dist mat for '{self._name}' in space '{space}'.")
        if isinstance(dm, dict):
            from ..utils.utils_datasets import get_file
            keep_idc = dm.get("keep_idc")
            local_path = get_file(dm["path_template"], **dm["spec"], **dm["gf_kw"])
            loaded = load_distmat(local_path)
            if keep_idc is not None and isinstance(loaded, np.ndarray):
                loaded = loaded[np.ix_(keep_idc, keep_idc)]
            self._dist_mats[space] = loaded
        elif isinstance(dm, tuple) and dm and isinstance(dm[0], dict):
            from ..utils.utils_datasets import get_file
            paths = tuple(
                get_file(d["path_template"], **d["spec"], **d["gf_kw"])
                if d is not None else None
                for d in dm
            )
            self._dist_mats[space] = load_distmat(paths)
        elif isinstance(dm, (str,Path)):
            self._dist_mats[space] = load_distmat(dm)
        elif isinstance(dm, tuple) and dm and isinstance(dm[0], (str,Path)):
            self._dist_mats[space] = load_distmat(dm)

    def _ensure_spin_mat_loaded(self, space):
        """Load spin mat for *space* from lazy spec if not yet loaded."""
        sm = self._spin_mats.get(space)
        if sm is None or not (isinstance(sm, tuple) and sm and isinstance(sm[0], dict)):
            return  # None or already loaded
        from ..utils.utils_datasets import get_file
        lgr.info(f"Lazy-loading spin mat for '{self._name}' in space '{space}'.")
        paths = tuple(get_file(d["path_template"], **d["spec"], **d["gf_kw"]) for d in sm)
        self._spin_mats[space] = load_spinmat(paths)

    def _fit_space(self, space):
        """Compute per-space derived attributes (idc_byhemi, labels_img, etc.)."""
        if space in self._idc_byhemi_dict:
            return  # already fitted

        self._ensure_image_loaded(space)
        img = self._images[space]

        # surface or volume?
        is_surf = isinstance(img, (nib.GiftiImage, tuple))
        is_uni  = isinstance(img, nib.GiftiImage)
        self._is_surface_dict[space] = is_surf

        # raw integer labels in image
        data = load_data(img).astype(int)
        labels_img = np.trim_zeros(np.unique(data))
        self._labels_img_dict[space] = labels_img

        # resolution
        res = self._legacy_res if self._legacy_res is not None \
            else _img_density_for_neuromaps(img)
        self._resolution_dict[space] = res

        # hemi
        if is_surf and not is_uni:
            hemi = ("L", "R")
        elif is_uni:
            h = self._selected_hemi or self._legacy_hemi or "L"
            hemi = (h,) if isinstance(h, str) else tuple(h)
        else:
            hemi = None
        self._hemi_dict[space] = hemi

        # if _labels not yet set, fall back to integer labels
        if self._labels is None:
            self._labels = labels_img.astype(str)

        # idc_byhemi
        # use pre-populated values from legacy init if available, else compute
        if space in self._idc_byhemi_dict and self._idc_byhemi_dict[space]["L"] is not None:
            pass  # already set by add_space from legacy path
        elif self._bilateral and not is_uni:
            # bilateral: both hemispheres share the same parcel indices 0..N_lh-1
            idc_both = np.arange(len(self._labels))
            self._idc_byhemi_dict[space]        = {"L": idc_both, "R": idc_both}
            self._labels_img_byhemi_dict[space] = {"L": labels_img, "R": labels_img}
            self._labels_byhemi_dict[space]     = {"L": self._labels, "R": self._labels}
        elif not is_uni:
            (idc_lh, idc_rh), (li_lh, li_rh) = find_parcel_hemispheres(img)
            self._idc_byhemi_dict[space]        = {"L": idc_lh,  "R": idc_rh}
            self._labels_img_byhemi_dict[space] = {"L": li_lh,   "R": li_rh}
            # label names by hemi
            lbl = self._labels
            self._labels_byhemi_dict[space] = {
                "L": lbl[idc_lh] if idc_lh is not None else None,
                "R": lbl[idc_rh] if idc_rh is not None else None,
            }
        else:
            # unilateral surface
            idc_all = np.arange(len(labels_img))
            h_key = hemi[0] if hemi else "L"
            other  = "R" if h_key == "L" else "L"
            self._idc_byhemi_dict[space]        = {h_key: idc_all,            other: np.array([], dtype=int)}
            self._labels_img_byhemi_dict[space] = {h_key: labels_img, other: np.array([], dtype=int)}
            self._labels_byhemi_dict[space] = {
                h_key: self._labels[idc_all],
                other: np.array([]),
            }

    def set_active_space(self, space):
        """Activate *space* for this Parcellation instance.

        Loads the image if necessary, computes derived attributes, and sets
        `_space`.  Must be called by NiSpace.fit() before using the Parcellation.
        """
        if space not in self._images:
            lgr.critical_raise(
                f"Cannot activate space '{space}' for parcellation '{self._name}': "
                f"not available. Available spaces: {self.spaces}.",
                ValueError,
            )
        self._ensure_image_loaded(space)
        self._fit_space(space)
        old = self._space
        self._space = space
        if old != space:
            lgr.info(
                f"Parcellation '{self._name}': active space set to '{space}'"
                + (f" (was '{old}')." if old else ".")
            )

        # compute combined-specific cx indices now that primary space is fitted
        if self._is_combined and self._cx_idc_lh is None:
            self._compute_cx_idc(space)

    def _compute_cx_idc(self, space):
        """For combined parcellations: derive cx-specific LH/RH indices."""
        idc = self._idc_byhemi_dict.get(space, {})
        idc_lh = idc.get("L")
        idc_rh = idc.get("R")
        if idc_lh is None or idc_rh is None:
            return
        # labels for cx are the first n_cx entries
        # (cx parcels come first in the merged image)
        n_cx = len(self._labels) - self._get_sc_n_parcels()
        if n_cx <= 0:
            return
        self._cx_idc_lh = idc_lh[idc_lh < n_cx]
        self._cx_idc_rh = idc_rh[idc_rh < n_cx]
        lgr.info(
            f"Combined parcellation: cx-LH parcels = {len(self._cx_idc_lh)}, "
            f"cx-RH parcels = {len(self._cx_idc_rh)}."
        )

    def _get_sc_n_parcels(self):
        """Number of subcortex parcels (combined only)."""
        if not self._is_combined or self._sc_name is None:
            return 0
        if self._n_cx_labels is not None:
            return len(self._labels) - self._n_cx_labels
        return 0

    # ------------------------------------------------------------------
    # Bilateral transformation
    # ------------------------------------------------------------------

    def make_bilateral(self):
        """Relabel parcels so both hemispheres share indices 1…N_bil.

        After calling:
        - ``_labels`` contains N_bil hemisphere-prefix-stripped labels.
        - All images are relabeled (RH values → matching LH values).
        - Distance matrices are averaged across hemispheres.
        - Spin matrices are cleared (null maps fall back to 'moran').
        - ``_bilateral`` is True.

        Requires a symmetric parcellation (``_symmetric=True``).
        Label matching is done by stripping ``_lh_prefix`` / ``_rh_prefix``.
        """
        if self._bilateral:
            return self
        if not self._symmetric:
            raise ValueError(
                "make_bilateral() requires a symmetric parcellation (_symmetric=True)."
            )
        if self._labels is None or len(self._labels) == 0:
            raise ValueError("make_bilateral() requires non-empty labels.")

        ok, lh_idc, rh_idc, bilateral_labels, unmatched = _bilateral_labels_match(
            self._labels, self._lh_prefix, self._rh_prefix
        )
        if not ok:
            raise ValueError(
                f"make_bilateral() '{self._name}': label matching failed — "
                f"{len(unmatched)} unmatched label(s): {unmatched[:5]}. "
                f"Check that all labels carry '{self._lh_prefix}' / '{self._rh_prefix}' prefixes "
                f"and that every LH label has a matching RH label."
            )

        N_bil     = len(lh_idc)
        new_vals  = np.arange(N_bil, dtype=np.int32) + 1  # 1-based bilateral parcel values
        # old 1-based parcel values (lh_idc/rh_idc are 0-based into _labels)
        old_lh_vals = (lh_idc + 1).astype(np.int32)
        old_rh_vals = (rh_idc + 1).astype(np.int32)

        self._labels = np.array(bilateral_labels)

        # clear l2rmap / lrcorr (irrelevant after bilateral)
        self._l2rmap = None
        self._lrcorr = None

        # --- relabel images ---
        for space in list(self._images.keys()):
            is_path = isinstance(self._images[space], (str,Path)) or (
                isinstance(self._images[space], tuple)
                and self._images[space]
                and isinstance(self._images[space][0], (str,Path))
            )
            if is_path:
                self._ensure_image_loaded(space)
            img = self._images[space]

            if isinstance(img, nib.Nifti1Image):
                # map all old parcel values → bilateral values
                all_old = np.concatenate([old_lh_vals, old_rh_vals])
                all_new = np.concatenate([new_vals,    new_vals])
                self._images[space] = relabel_nifti_parc(img, new_order=all_old, new_labels=all_new)

            elif isinstance(img, tuple) and len(img) == 2:
                lh_img, rh_img = img
                # LH GIFTI: sorted(old_lh_vals) → new_vals reordered by argsort
                lh_sort = np.argsort(old_lh_vals)
                lh_img_new = relabel_gifti_parc(lh_img, new_labels=new_vals[lh_sort])
                # RH GIFTI: sorted(old_rh_vals) → new_vals reordered by argsort
                rh_sort = np.argsort(old_rh_vals)
                rh_img_new = relabel_gifti_parc(rh_img, new_labels=new_vals[rh_sort])
                self._images[space] = (lh_img_new, rh_img_new)
            # unilateral GiftiImage: skip

        # --- average dist_mats ---
        for space in list(self._dist_mats.keys()):
            self._ensure_dist_mat_loaded(space)
            dm = self._dist_mats[space]
            if dm is None:
                continue
            if isinstance(dm, tuple) and len(dm) == 2:
                dm_l, dm_r = dm
                if dm_l is not None and dm_r is not None:
                    self._dist_mats[space] = (dm_l.astype(float) + dm_r.astype(float)) / 2
                else:
                    self._dist_mats[space] = dm_l if dm_l is not None else dm_r
            elif isinstance(dm, np.ndarray) and dm.ndim == 2:
                # volumetric N×N matrix: extract matched LH/RH blocks by index
                dm_l_block = dm[np.ix_(lh_idc, lh_idc)].astype(float)
                dm_r_block = dm[np.ix_(rh_idc, rh_idc)].astype(float)
                self._dist_mats[space] = (dm_l_block + dm_r_block) / 2

        # --- nullify spin_mats ---
        if any(v is not None for v in self._spin_mats.values()):
            lgr.warning(
                "make_bilateral(): spin matrices are incompatible with bilateral relabeling "
                "and have been cleared. Null maps will fall back to 'moran'."
            )
        for space in self._spin_mats:
            self._spin_mats[space] = None

        # --- reset combined-specific cx indices (recomputed by set_active_space) ---
        self._cx_idc_lh = None
        self._cx_idc_rh = None

        # --- clear per-space derived caches (recomputed on next access) ---
        self._idc_byhemi_dict.clear()
        self._labels_byhemi_dict.clear()
        self._labels_img_dict.clear()
        self._labels_img_byhemi_dict.clear()
        self._hemi_dict.clear()
        self._resolution_dict.clear()
        self._is_surface_dict.clear()

        self._bilateral = True
        lgr.info(
            f"make_bilateral() '{self._name}': {len(lh_idc) * 2} → {N_bil} parcels."
        )
        self.validate(pre_activation=True)
        return self

    # ------------------------------------------------------------------
    # Hemisphere selection
    # ------------------------------------------------------------------

    def select_hemi(self, hemi, verbose=True):
        """Keep only parcels from one hemisphere.

        Filters ``_labels``, parcellation images, distance matrices, and
        clears spin matrices.  Safe to call on a pre-activation Parcellation
        (images may still be lazy paths).

        Parameters
        ----------
        hemi : str or list/tuple of str
            ``"L"``, ``"R"``, ``["L"]``, ``["R"]`` — or ``["L", "R"]`` / ``None``
            (no-op, returns self unchanged).

        Returns
        -------
        self
        """
        # --- normalise hemi arg ---
        set_log(lgr, verbose)
        if hemi is None:
            return self
        if isinstance(hemi, (list, tuple)):
            hemi_list = [h for h in hemi if h in ("L", "R")]
        else:
            hemi_list = [hemi] if hemi in ("L", "R") else []

        if len(hemi_list) == 0:
            lgr.warning(f"select_hemi: unrecognised hemi value {hemi!r}. Skipping.")
            return self
        if {"L", "R"}.issubset(set(hemi_list)):
            return self  # both hemispheres requested — no-op

        keep_hemi = hemi_list[0]

        # --- determine keep_idc (0-based indices into current _labels) ---
        keep_idc = None

        # prefer already-computed idc_byhemi from any fitted space
        for idc_dict in self._idc_byhemi_dict.values():
            idc_h = idc_dict.get(keep_hemi)
            if idc_h is not None and len(idc_h) > 0:
                keep_idc = np.sort(idc_h).astype(int)
                break

        # fallback: label-prefix matching
        if keep_idc is None and self._labels is not None:
            prefix = self._lh_prefix if keep_hemi == "L" else self._rh_prefix
            keep_idc = np.array(
                [i for i, lbl in enumerate(self._labels) if str(lbl).startswith(prefix)],
                dtype=int,
            )

        if keep_idc is None or len(keep_idc) == 0:
            lgr.warning(
                f"select_hemi('{keep_hemi}'): no parcels found for that hemisphere. Skipping."
            )
            return self

        # 1-based parcel values that correspond to keep_idc in NIfTI images
        keep_vals_1based = (keep_idc + 1).astype(np.int32)

        # --- filter labels ---
        if self._labels is not None:
            self._labels = self._labels[keep_idc]

        # --- filter images ---
        for space in list(self._images.keys()):
            img = self._images[space]

            is_surf_path = (
                isinstance(img, tuple) and img
                and isinstance(img[0], (str,Path))
            )
            is_surf_loaded = (
                isinstance(img, tuple) and img
                and isinstance(img[0], nib.GiftiImage)
            )

            if is_surf_path or is_surf_loaded:
                # (lh, rh) tuple — keep the requested hemisphere
                idx = 0 if keep_hemi == "L" else 1
                self._images[space] = img[idx] if len(img) > idx else img[0]

            elif isinstance(img, nib.Nifti1Image):
                # loaded volume — zero out unwanted parcel voxels
                data = img.get_fdata(dtype=np.float32).copy()
                mask = np.isin(data, keep_vals_1based)
                data[~mask] = 0
                self._images[space] = nib.Nifti1Image(data, img.affine, img.header)

            elif isinstance(img, (str,Path)):
                # lazy volume path — load now and mask immediately
                loaded = load_img(img)
                if isinstance(loaded, nib.Nifti1Image):
                    data = loaded.get_fdata(dtype=np.float32).copy()
                    mask = np.isin(data, keep_vals_1based)
                    data[~mask] = 0
                    self._images[space] = nib.Nifti1Image(data, loaded.affine, loaded.header)
                else:
                    self._images[space] = loaded  # shouldn't happen for vol path

        # --- filter dist_mats ---
        for space in list(self._dist_mats.keys()):
            dm = self._dist_mats[space]
            if dm is None:
                continue

            is_surf_dm = isinstance(dm, tuple)
            is_vol_loaded = isinstance(dm, np.ndarray) and dm.ndim == 2
            is_vol_lazy = isinstance(dm, dict)

            if is_surf_dm:
                # (dm_l, dm_r) or (spec_l, spec_r) — keep one
                idx = 0 if keep_hemi == "L" else 1
                self._dist_mats[space] = dm[idx] if len(dm) > idx else None

            elif is_vol_loaded:
                # sub-block for selected hemisphere
                self._dist_mats[space] = dm[np.ix_(keep_idc, keep_idc)]

            elif is_vol_lazy:
                # attach keep_idc so _ensure_dist_mat_loaded can apply the mask
                self._dist_mats[space] = dict(dm, keep_idc=keep_idc)

        # --- trim spin_mats: keep the relevant half, zero-out the other ---
        for space in list(self._spin_mats):
            self._ensure_spin_mat_loaded(space)
        for space, sm in self._spin_mats.items():
            if sm is None or not (isinstance(sm, tuple) and len(sm) == 2):
                continue
            spins_lh, spins_rh = sm
            is_t_mat = spins_lh is not None and hasattr(spins_lh, "ndim") and spins_lh.ndim == 3
            if is_t_mat:
                # Cornblath T-matrix: shape (n_perm, n_hemi, n_hemi)
                n_perm = spins_lh.shape[0]
                empty = np.zeros((n_perm, 0, 0), dtype=np.float32)
                if keep_hemi == "L":
                    self._spin_mats[space] = (spins_lh, empty)
                else:
                    self._spin_mats[space] = (empty, spins_rh)
            else:
                # parcel-index spin_mat: shape (n_hemi, n_perm)
                if keep_hemi == "L":
                    n_perm = spins_lh.shape[1] if spins_lh is not None and spins_lh.ndim == 2 else 0
                    self._spin_mats[space] = (
                        spins_lh,
                        np.zeros((0, n_perm), dtype=spins_rh.dtype if spins_rh is not None else np.int32),
                    )
                else:
                    n_perm = spins_rh.shape[1] if spins_rh is not None and spins_rh.ndim == 2 else 0
                    self._spin_mats[space] = (
                        np.zeros((0, n_perm), dtype=spins_lh.dtype if spins_lh is not None else np.int32),
                        spins_rh,
                    )

        # --- l2rmap / lrcorr are irrelevant for a single hemisphere ---
        self._l2rmap = None
        self._lrcorr = None

        # --- clear per-space derived caches (recomputed on next set_active_space) ---
        self._idc_byhemi_dict.clear()
        self._labels_byhemi_dict.clear()
        self._labels_img_dict.clear()
        self._labels_img_byhemi_dict.clear()
        self._hemi_dict.clear()
        self._resolution_dict.clear()
        self._is_surface_dict.clear()

        self._selected_hemi = keep_hemi
        lgr.info(f"select_hemi('{keep_hemi}'): {len(keep_idc)} parcels selected.")
        return self

    # ------------------------------------------------------------------
    # Space query helpers
    # ------------------------------------------------------------------

    def get_image_for_dataspace(self, data_space):
        """Return the space name that best matches *data_space*.

        Raises ValueError if no compatible space is found.
        """
        if data_space is None:
            return self.spaces[0]
        for s in self.spaces:
            if _spaces_match(data_space, s):
                return s
        lgr.critical_raise(
            f"No parcellation space matches data space '{data_space}'. "
            f"Available: {self.spaces}.",
            ValueError,
        )

    def get_dist_mat(self, space=None, compute_if_missing=True,
                     resample=2, centroids=False, n_proc=1):
        """Return distance matrix for *space* (defaults to active space).

        If not preloaded and *compute_if_missing* is True, computes it on the
        fly and caches the result.
        """
        space = space or self._space
        self._ensure_dist_mat_loaded(space)
        dm = self._dist_mats.get(space)
        if dm is not None:
            return dm
        if not compute_if_missing:
            return None
        # compute
        lgr.info(f"Computing distance matrix for '{self._name}' in space '{space}'.")
        self._ensure_image_loaded(space)
        img = self._images[space]
        hemi = self._hemi_dict.get(space)
        dm = get_distance_matrix(
            parc=img,
            parc_space=space,
            parc_hemi=hemi,
            parc_resample=resample,
            centroids=centroids,
            n_proc=n_proc,
        )
        self._dist_mats[space] = dm
        return dm

    def get_spin_mat(self, space=None):
        """Return spin matrix for *space* (defaults to active space)."""
        space = space or self._space
        self._ensure_spin_mat_loaded(space)
        return self._spin_mats.get(space)

    def get_image(self, space=None):
        """Return the parcellation image for *space* (default: active space).

        Returns a ``nib.Nifti1Image`` for MNI spaces and a
        ``(lh_GiftiImage, rh_GiftiImage)`` tuple for surface spaces.
        """
        space = space or self._space
        if space is None:
            lgr.critical_raise(
                "No active space set. Call set_active_space() first or pass space=.",
                ValueError,
            )
        if space not in self._images:
            lgr.critical_raise(
                f"Space '{space}' not available. Available: {self.spaces}.",
                ValueError,
            )
        self._ensure_image_loaded(space)
        return self._images[space]

    def get_labels(self):
        """Return parcel labels (space-independent)."""
        return self._labels

    def get_hemi(self, space=None):
        """Return hemisphere tuple, e.g. ``('L', 'R')``, for *space* (default: active space)."""
        space = space or self._space
        return self._hemi_dict.get(space)

    def get_idc_byhemi(self, space=None):
        """Return ``{'L': array, 'R': array}`` of parcel indices for *space* (default: active space)."""
        space = space or self._space
        return self._idc_byhemi_dict.get(space, {"L": None, "R": None})

    def get_labels_byhemi(self, space=None):
        """Return ``{'L': array, 'R': array}`` of parcel labels for *space* (default: active space)."""
        space = space or self._space
        return self._labels_byhemi_dict.get(space, {"L": None, "R": None})

    def get_resolution(self, space=None):
        """Return resolution / density string for *space* (default: active space)."""
        space = space or self._space
        return self._resolution_dict.get(space)

    def is_surface_space(self, space=None):
        """Return ``True`` if *space* (default: active) is a surface space."""
        space = space or self._space
        return self._is_surface_dict.get(space, False)

    def get_surface_for_spins(self, preferred="fsLR"):
        """Return *(surface_image, spin_mat, space_name)* for spin tests.

        For surface-primary parcellations returns the primary image.
        For MNI-primary cortex parcellations with available surface spaces,
        lazy-loads the surface image and returns it.
        For combined cx+sc parcellations, returns the cortex-only surface
        stored in `_cx_surface`.
        Returns *(None, None, None)* when no surface data is available.
        """
        # 1. already surface-primary
        if self._is_surface:
            return self._image_obj, self._spin_mat, self._space

        # 2. combined: cx surface lives in _cx_surface
        for sname in [preferred, "fsaverage", "fsLR"]:
            if sname in self._cx_surface:
                s = self._cx_surface[sname]
                if s.get("image") is None and s.get("img_paths") is not None:
                    lgr.info(f"Lazy-loading cx surface image for '{self._cx_name or self._name}' ('{sname}').")
                    s["image"] = load_img(s["img_paths"])
                cx_sm = s.get("spin_mat")
                if isinstance(cx_sm, tuple) and cx_sm and isinstance(cx_sm[0], dict):
                    from ..utils.utils_datasets import get_file
                    lgr.info(f"Lazy-loading cx surface spin mat for '{self._cx_name or self._name}' ('{sname}').")
                    paths = tuple(get_file(d["path_template"], **d["spec"], **d["gf_kw"]) for d in cx_sm)
                    cx_sm = load_spinmat(paths)
                    s["spin_mat"] = cx_sm
                return s.get("image"), cx_sm, sname

        # 3. non-combined MNI-primary but has surface in _images
        for sname in [preferred, "fsaverage", "fsLR"]:
            if sname in self._images:
                self._ensure_image_loaded(sname)
                img = self._images[sname]
                if isinstance(img, (nib.GiftiImage, tuple)):
                    sm = self.get_spin_mat(sname)
                    return img, sm, sname

        lgr.warning(
            f"No surface parcellation data found for '{self._name}'. "
            f"Spin tests require a surface space."
        )
        return None, None, None

    # ------------------------------------------------------------------
    # Backward-compat properties (delegate to active space)
    # ------------------------------------------------------------------

    @property
    def _image_obj(self):
        self._ensure_image_loaded(self._space)
        return self._images.get(self._space)

    @property
    def _dist_mat(self):
        self._ensure_dist_mat_loaded(self._space)
        return self._dist_mats.get(self._space)

    @_dist_mat.setter
    def _dist_mat(self, value):
        if self._space is not None:
            self._dist_mats[self._space] = value

    @property
    def _spin_mat(self):
        return self._spin_mats.get(self._space)

    @_spin_mat.setter
    def _spin_mat(self, value):
        if self._space is not None:
            self._spin_mats[self._space] = value

    @property
    def _idc_byhemi(self):
        return self._idc_byhemi_dict.get(self._space, {"L": None, "R": None})

    @property
    def _labels_byhemi(self):
        return self._labels_byhemi_dict.get(self._space, {"L": None, "R": None})

    @property
    def _labels_img(self):
        return self._labels_img_dict.get(self._space)

    @property
    def _labels_img_byhemi(self):
        return self._labels_img_byhemi_dict.get(self._space, {"L": None, "R": None})

    @property
    def _hemi(self):
        return self._hemi_dict.get(self._space)

    @property
    def _resolution(self):
        return self._resolution_dict.get(self._space)

    @property
    def _is_surface(self):
        return self._is_surface_dict.get(self._space, False)

    @property
    def _is_unilateral_surface(self):
        img = self._images.get(self._space)
        return isinstance(img, nib.GiftiImage)

    # ---- read-only info properties ----

    @property
    def spaces(self):
        """List of available space names."""
        return list(self._images.keys())

    @property
    def has_surface(self):
        """True if any space (primary or secondary) is a surface."""
        for space, img in self._images.items():
            # check is_surface_dict first (already computed)
            if self._is_surface_dict.get(space):
                return True
            # check image type if available
            if isinstance(img, (nib.GiftiImage, tuple)) and not isinstance(img, (str,Path)):
                return True
        # also check cx_surface
        return bool(self._cx_surface)

    def get_null_space(self):
        """Return optimal null strategy for null map generation.

        For non-combined parcellations returns ``(space_name, null_method)``.

        For combined (cx+sc) parcellations returns a nested tuple
        ``((cx_space, cx_method), (sc_space, sc_method))`` when cx and sc
        strategies differ, or a plain ``(space, "moran")`` when no surface
        space is available for the cx component.

        Priority
        --------
        Spin (alexander_bloch) — cortex-only or cx component with surface space:
            fsLR  >  fsaverage  >  any surface space name
        Moran — combined (cx+sc) sc component, or no surface available:
            MNI152NLin2009cAsym  >  MNI152NLin6Asym  >  any MNI  >  first available
        """
        def _is_surf(s):
            return any(k in s.lower() for k in ("fsa", "fsaverage", "fslr", "fs_lr"))

        def _best_mni():
            for preferred in ["MNI152NLin6Asym", "MNI152NLin2009cAsym", "MNI152", "MNIOriginal", "MNI"]:
                if preferred in self.spaces:
                    return preferred
            for s in self.spaces:
                if "mni" in s.lower():
                    return s
            return self.spaces[0]

        if self._is_combined:
            # cx: prefer surface space for spin — combined parcs store surface in _cx_surface
            cx_surface_spaces = list(self._cx_surface.keys()) if self._cx_surface else []
            cx_space = None
            for preferred in ["fsLR", "fsaverage"]:
                if preferred in cx_surface_spaces:
                    cx_space = preferred
                    break
            if cx_space is None:
                for s in cx_surface_spaces:
                    if _is_surf(s):
                        cx_space = s
                        break
            sc_space = _best_mni()
            cx_method = "cornblath" if cx_space is not None else "moran"
            if cx_space is None:
                cx_space = sc_space
            # if both strategies are identical, return single pair
            if cx_method == "moran" and cx_space == sc_space:
                return sc_space, "moran"
            return ((cx_space, cx_method), (sc_space, "moran"))

        if not self._bilateral:
            for preferred in ["fsLR", "fsaverage"]:
                if preferred in self.spaces:
                    return preferred, "cornblath"
            for s in self.spaces:
                if _is_surf(s):
                    return s, "cornblath"

        return _best_mni(), "moran"

    def get_sc_idc(self):
        """Subcortex parcel indices in the combined data vector; ``None`` for non-combined.

        Computes cx/sc split on demand if ``set_active_space()`` has not yet been called.
        """
        if not self._is_combined:
            return None
        if self._cx_idc_lh is None:
            # use already-fitted space, or fit a MNI space on demand
            space = self._space or next(iter(self._idc_byhemi_dict), None)
            if space is None:
                for preferred in ["MNI152NLin6Asym", "MNI152NLin2009cAsym", "MNI152", "MNIOriginal"]:
                    if preferred in self.spaces:
                        space = preferred
                        break
                if space is None:
                    space = self.spaces[0]
            self._fit_space(space)
            self._compute_cx_idc(space)
        if self._cx_idc_lh is None:
            lgr.warning("get_sc_idc: cx_idc not yet computed.")
            return None
        cx_all = np.concatenate([self._cx_idc_lh, self._cx_idc_rh])
        return np.setdiff1d(np.arange(len(self._labels)), cx_all)

    def get_sc_dist_mat(self):
        """Lazy-load and return ``(dist_mat, space)`` for sc parcels; ``(None, None)`` otherwise.

        Used by the split-null path to avoid computing a full combined dist_mat.
        """
        if not self._is_combined or self._sc_dist_mat_spec is None:
            return None, None
        if self._sc_dist_mat is not None:
            return self._sc_dist_mat, self._sc_dist_mat_spec["space"]
        spec = self._sc_dist_mat_spec
        from ..utils.utils_datasets import get_file
        lgr.info(f"Lazy-loading sc dist mat for '{self._sc_name or self._name}' (space '{spec['space']}').")
        local_path = get_file(spec["path_template"], **spec["spec"], **spec["gf_kw"])
        self._sc_dist_mat = load_distmat(local_path)
        return self._sc_dist_mat, spec["space"]

    def get_cx_dist_mat(self):
        """Lazy-load and return ``(dist_mat, space)`` for cx parcels; ``(None, None)`` otherwise.

        Used by the split-null path to avoid computing a full combined dist_mat.
        """
        if not self._is_combined or self._cx_dist_mat_spec is None:
            return None, None
        if self._cx_dist_mat is not None:
            return self._cx_dist_mat, self._cx_dist_mat_spec["space"]
        spec = self._cx_dist_mat_spec
        from ..utils.utils_datasets import get_file
        lgr.info(f"Lazy-loading cx dist mat for '{self._cx_name or self._name}' (space '{spec['space']}').")
        if "hemi_specs" in spec:
            # per-hemisphere surface geodesic dist_mat
            paths = tuple(
                get_file(hs["path_template"], **hs["spec"], **hs["gf_kw"])
                for hs in spec["hemi_specs"]
            )
            self._cx_dist_mat = load_distmat(paths)
        else:
            # single-file MNI Euclidean
            local_path = get_file(spec["path_template"], **spec["spec"], **spec["gf_kw"])
            self._cx_dist_mat = load_distmat(local_path)
        return self._cx_dist_mat, spec["space"]

    @property
    def default_null_method(self):
        """Recommended null method based on available spaces (see ``get_null_space``)."""
        result = self.get_null_space()
        if isinstance(result[0], tuple):
            # combined: return (cx_method, sc_method)
            return (result[0][1], result[1][1])
        return result[1]

    # ------------------------------------------------------------------
    # Distance-matrix helper (kept for api._get_dist_mat compat)
    # ------------------------------------------------------------------

    def get_dist_mat_legacy(self, resample=2, centroids=False, n_proc=1, recalculate=False):
        """Backward-compat wrapper used by api._get_dist_mat."""
        if self._dist_mat is not None and not recalculate:
            return self._dist_mat
        return self.get_dist_mat(
            space=self._space, compute_if_missing=True,
            resample=resample, centroids=centroids, n_proc=n_proc,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, pre_activation=False):
        """Run consistency checks and log results.

        Parameters
        ----------
        pre_activation : bool
            If True, skip checks that require an active space (e.g., idc_byhemi).
            Use this immediately after construction before set_active_space().
        """
        ok = True
        prefix = f"Parcellation '{self._name}'"

        # 1. at least one space
        if not self._images:
            lgr.error(f"{prefix}: no images loaded.")
            ok = False

        # 2. labels length vs parcel count (for fitted spaces)
        for space in self._labels_img_dict:
            n_img = len(self._labels_img_dict[space])
            if self._labels is not None and len(self._labels) != n_img:
                lgr.warning(
                    f"{prefix} space '{space}': label count ({len(self._labels)}) "
                    f"does not match image parcel count ({n_img})."
                )

        # 3. idc_byhemi checks (only for fitted spaces)
        for space, idc in self._idc_byhemi_dict.items():
            idc_l = idc.get("L")
            idc_r = idc.get("R")
            if idc_l is None or idc_r is None:
                continue
            # no overlap (bilateral: shared indices are expected)
            if not self._bilateral:
                overlap = np.intersect1d(idc_l, idc_r)
                if len(overlap) > 0:
                    lgr.error(
                        f"{prefix} space '{space}': LH and RH indices overlap "
                        f"({len(overlap)} shared parcels)."
                    )
                    ok = False
            # coverage
            n_parcels = len(self._labels_img_dict.get(space, []))
            if n_parcels:
                covered = len(idc_l) + len(idc_r)
                if covered < n_parcels and not self._is_combined:
                    lgr.warning(
                        f"{prefix} space '{space}': {n_parcels - covered} parcel(s) "
                        f"not assigned to either hemisphere."
                    )
            # index bounds
            for side, idc_h in [("L", idc_l), ("R", idc_r)]:
                if n_parcels and idc_h is not None and len(idc_h):
                    if idc_h.max() >= n_parcels or idc_h.min() < 0:
                        lgr.error(
                            f"{prefix} space '{space}' hemi-{side}: "
                            f"indices out of bounds [0, {n_parcels})."
                        )
                        ok = False

        # 4. dist_mat shape (skip lazy-spec entries — not yet loaded)
        for space, dm in self._dist_mats.items():
            if dm is None or isinstance(dm, dict):
                continue
            mats = dm if isinstance(dm, tuple) else (dm,)
            for m in mats:
                if m is None or isinstance(m, dict):
                    continue
                if hasattr(m, "ndim") and m.ndim == 2 and m.shape[0] != m.shape[1]:
                    lgr.error(f"{prefix} space '{space}': distance matrix is not square.")
                    ok = False

        # 5. spin_mat shape (only if idc is available)
        for space, sm in self._spin_mats.items():
            if sm is None:
                continue
            if not (isinstance(sm, tuple) and len(sm) == 2):
                lgr.error(f"{prefix} space '{space}': spin_mat must be a 2-tuple (spins_lh, spins_rh).")
                ok = False
                continue
            spins_lh, spins_rh = sm
            idc = self._idc_byhemi_dict.get(space)
            if idc and idc.get("L") is not None:
                n_lh = len(idc["L"])
                n_rh = len(idc["R"])
                # Cornblath T-matrix: (n_perm, n_hemi, n_hemi); parcel-index: (n_hemi, n_perm)
                is_t_mat = hasattr(spins_lh, "ndim") and spins_lh.ndim == 3
                lh_parc_dim = spins_lh.shape[1] if is_t_mat else spins_lh.shape[0]
                rh_parc_dim = spins_rh.shape[1] if is_t_mat else spins_rh.shape[0]
                if lh_parc_dim != n_lh:
                    lgr.warning(
                        f"{prefix} space '{space}': spins_lh parcel dim ({lh_parc_dim}) "
                        f"!= n_lh ({n_lh})."
                    )
                if rh_parc_dim != n_rh:
                    lgr.warning(
                        f"{prefix} space '{space}': spins_rh parcel dim ({rh_parc_dim}) "
                        f"!= n_rh ({n_rh})."
                    )

        # 6. l2rmap shape
        if self._l2rmap is not None and isinstance(self._l2rmap, pd.DataFrame):
            for space, idc in self._idc_byhemi_dict.items():
                idc_l = idc.get("L")
                idc_r = idc.get("R")
                if idc_l is None or idc_r is None:
                    continue
                exp_shape = (len(idc_l), len(idc_r))
                if self._l2rmap.shape != exp_shape:
                    lgr.warning(
                        f"{prefix}: l2rmap shape {self._l2rmap.shape} does not match "
                        f"expected ({exp_shape}) for space '{space}'."
                    )

        # 7. combined: cx indices
        if self._is_combined and not pre_activation:
            if self._cx_idc_lh is None or self._cx_idc_rh is None:
                lgr.warning(f"{prefix}: cx_idc not yet computed (call set_active_space first).")

        if ok:
            lgr.info(f"{prefix}: validation passed.")
        return ok

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(self, space=None, cmap="gist_rainbow", colorbar=False, **kwargs):
        """Visualize the parcellation with each parcel in a distinct color.

        Wraps :func:`nispace.plotting.brainplot` with ROI-style defaults
        (analogous to ``nilearn.plotting.plot_roi``).  The parcellation image
        for the active space is passed directly — no tabular data conversion.
        Parameter kind can be "slice", "surface", or "glass" and determines
        the type of plot produced.

        Parameters
        ----------
        space : str, optional
            Parcellation space to render.  Defaults to the active space.
        cmap : str
            Colormap for distinct parcel colors.
            Default ``"gist_rainbow"`` matches ``nilearn.plotting.plot_roi``.
        colorbar : bool
            Show colorbar.  Default ``False`` — parcel indices carry no
            meaningful scale.
        **kwargs
            Forwarded to :func:`nispace.plotting.brainplot`.

        Returns
        -------
        fig : matplotlib.Figure
        axes_out : list of matplotlib.Axes
        """
        from ..plotting import brainplot

        space = space or self._space
        if space is None:
            raise ValueError(
                "No active space set. Call set_active_space() first or pass space=."
            )
        self._ensure_image_loaded(space)
        img = self._images[space]

        kwargs.setdefault("symmetric_cmap", False)
        kwargs.setdefault("title", self._name or False)

        return brainplot(img, space=space, cmap=cmap, colorbar=colorbar, **kwargs)
