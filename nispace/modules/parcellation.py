
import pathlib
import nibabel as nib
import numpy as np
import pandas as pd

from neuromaps.images import load_data

from .. import lgr
from ..nulls import (
    _img_space_for_neuromaps, _img_density_for_neuromaps,
    find_parcel_hemispheres, get_distance_matrix,
)
from ..io import load_distmat, load_spinmat, load_img, load_labels, load_l2rmap
from ..utils.utils import set_log


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
    ):
        # --- shared (space-independent) ---
        self._name = name
        self._level = level          # "cortex" | "subcortex" | "combined"
        self._is_combined = is_combined
        self._cx_name = cx_name      # name of cortex component (combined only)
        self._sc_name = sc_name      # name of subcortex component (combined only)
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
        self._cx_idc_lh = None  # cx-LH parcel indices within the combined data vector
        self._cx_idc_rh = None  # cx-RH parcel indices within the combined data vector
        # {space: {"img_paths": ..., "image": None|loaded, "spin_mat": ...}}
        self._cx_surface = {}

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

        data_dir = pathlib.Path(data_dir)
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

        spaces = list(lib_entry.keys())
        lgr.info(f"Building multi-space Parcellation for '{name}' from library.")
        lgr.info(f"Available spaces: {', '.join(spaces)}")
        p = cls(name=name)

        shared_loaded = False  # load labels/l2rmap/lrcorr only once

        for space, space_lib in lib_entry.items():
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

                sym = "l2rmap" not in space_lib and "lrcorr" not in space_lib
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

                p._level = space_lib.get("level", None)
                shared_loaded = True

            # ---- distance matrix ----
            dm = None
            if load_dist_mat and "distmat" in space_lib:
                if is_vol:
                    dm_path = get_file(
                        base / f"parc-{name}_space-{space}.dist.csv.gz",
                        **space_lib["distmat"], **gf_kw,
                    )
                    dm = load_distmat(dm_path)
                else:
                    dm_paths = []
                    has_all = True
                    for h in ["L", "R"]:
                        if space_lib["distmat"].get(h) is not None:
                            dm_paths.append(
                                get_file(
                                    base / f"parc-{name}_space-{space}_hemi-{h}.dist.csv.gz",
                                    **space_lib["distmat"][h], **gf_kw,
                                )
                            )
                        else:
                            lgr.info(f"  Distance matrix for '{name}' hemi-{h} in '{space}' not available.")
                            has_all = False
                            dm_paths.append(None)
                    dm = load_distmat(tuple(dm_paths)) if has_all else None

            # ---- spin matrix (surface only) ----
            sm = None
            if load_spin_mat and not is_vol and "spinmat" in space_lib:
                sm_paths = tuple(
                    get_file(
                        base / f"parc-{name}_space-{space}_hemi-{h}.spin.npy",
                        **space_lib["spinmat"][h], **gf_kw,
                    )
                    for h in ["L", "R"]
                )
                sm = load_spinmat(sm_paths)
            elif load_spin_mat and not is_vol:
                lgr.info(f"  No pre-computed spin matrix for '{name}' in '{space}'.")

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

        # find common MNI spaces
        cx_spaces = list(cx_lib.keys())
        sc_spaces = list(sc_lib.keys())
        common_mni = [s for s in cx_spaces if "mni" in s.lower() and s in sc_spaces]
        if not common_mni:
            lgr.critical_raise(
                f"No common MNI space found for cx='{cx_name}' and sc='{sc_name}'. "
                f"cx spaces: {cx_spaces}, sc spaces: {sc_spaces}",
                ValueError,
            )
        lgr.info(f"  Common MNI space(s) for combined: {common_mni}.")

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

                # symmetry / l2rmap / lrcorr from cx (sc is always symmetric)
                cx_sym = "l2rmap" not in cx_lib[mni_space] and "lrcorr" not in cx_lib[mni_space]
                p._symmetric = cx_sym  # combined is asymmetric if cx is

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
            if load_spin_mat and "spinmat" in cx_lib[surf_space]:
                cx_sm_paths = tuple(
                    get_file(
                        base_cx_surf / f"parc-{cx_name}_space-{surf_space}_hemi-{h}.spin.npy",
                        **cx_lib[surf_space]["spinmat"][h], **gf_kw,
                    )
                    for h in ["L", "R"]
                )
                cx_sm = load_spinmat(cx_sm_paths)
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
        is_path = isinstance(img, (str, pathlib.Path)) or (
            isinstance(img, tuple) and isinstance(img[0], (str, pathlib.Path))
        )
        if is_path:
            lgr.info(f"Lazy-loading parcellation image for space '{space}'.")
            self._images[space] = load_img(img)

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
            h = self._legacy_hemi if self._legacy_hemi is not None else "L"
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
            self._idc_byhemi_dict[space]        = {h_key: idc_all,   other: np.array([])}
            self._labels_img_byhemi_dict[space] = {h_key: labels_img, other: np.array([])}
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
        # sc parcels are at the end of _labels; we can detect them by hemi-tag absence
        # simple heuristic: count labels without hemi- prefix that are not in cx
        # fall back to 0 if uncertain
        try:
            sc_count = sum(1 for l in self._labels if "hemi-" not in str(l)
                           and not any(cx in str(l) for cx in [self._cx_name or ""]))
            return sc_count if sc_count > 0 else 0
        except Exception:
            return 0

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

    def get_surface_for_spins(self, preferred="fsaverage"):
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
                    lgr.info(f"Lazy-loading cx surface image for '{self._name}' ('{sname}').")
                    s["image"] = load_img(s["img_paths"])
                return s.get("image"), s.get("spin_mat"), sname

        # 3. non-combined MNI-primary but has surface in _images
        for sname in [preferred, "fsaverage", "fsLR"]:
            if sname in self._images:
                self._ensure_image_loaded(sname)
                img = self._images[sname]
                if isinstance(img, (nib.GiftiImage, tuple)):
                    sm = self._spin_mats.get(sname)
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
            if isinstance(img, (nib.GiftiImage, tuple)) and not isinstance(img, (str, pathlib.Path)):
                return True
        # also check cx_surface
        return bool(self._cx_surface)

    def get_null_space(self):
        """Return ``(space_name, null_method)`` for optimal null map generation.

        Priority
        --------
        Spin (alexander_bloch) — cortex-only parcellation with any surface space:
            fsLR  >  fsaverage  >  any surface space name
        Moran — combined (cx+sc) parcellation, or no surface available:
            MNI152NLin2009cAsym  >  MNI152NLin6Asym  >  any MNI  >  first available
        """
        def _is_surf(s):
            return any(k in s.lower() for k in ("fsa", "fsaverage", "fslr", "fs_lr"))

        if not self._is_combined:
            for preferred in ["fsLR", "fsaverage"]:
                if preferred in self.spaces:
                    return preferred, "alexander_bloch"
            for s in self.spaces:
                if _is_surf(s):
                    return s, "alexander_bloch"

        # TODO: combined parcellations always fall through to moran.
        # For proper combined null maps, return a split strategy, e.g.:
        #   ((cx_surf_space, "alexander_bloch"), (mni_space, "moran"))
        # and update _get_null_maps + generate_null_maps to run both pipelines
        # and merge results into a single null array.

        # moran fallback
        for preferred in ["MNI152NLin2009cAsym", "MNI152NLin6Asym", "MNI152"]:
            if preferred in self.spaces:
                return preferred, "moran"
        for s in self.spaces:
            if "mni" in s.lower():
                return s, "moran"
        return self.spaces[0], "moran"

    @property
    def default_null_method(self):
        """Recommended null method based on available spaces (see ``get_null_space``)."""
        _, method = self.get_null_space()
        return method

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
            # no overlap
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

        # 4. dist_mat shape
        for space, dm in self._dist_mats.items():
            if dm is None:
                continue
            mats = dm if isinstance(dm, tuple) else (dm,)
            for m in mats:
                if m is not None and m.ndim == 2 and m.shape[0] != m.shape[1]:
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
                if spins_lh.shape[0] != n_lh:
                    lgr.warning(
                        f"{prefix} space '{space}': spins_lh rows ({spins_lh.shape[0]}) "
                        f"!= n_lh ({n_lh})."
                    )
                if spins_rh.shape[0] != n_rh:
                    lgr.warning(
                        f"{prefix} space '{space}': spins_rh rows ({spins_rh.shape[0]}) "
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
