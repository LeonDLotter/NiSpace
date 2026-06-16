from typing import Union, List, Dict, Tuple
from pathlib import Path
import textwrap
import pandas as pd
import numpy as np
import os

from requests import Session

import logging
lgr = logging.getLogger(__name__)
from . import __version__, __commit__
from .core.constants import _PARC_DEFAULT, _SPACE_DEFAULT_VOL, _SPACE_DEFAULT_SURF
from .stats.misc import zscore_df
from .utils.utils import _rm_ext, set_log, merge_parcellations
from .utils.utils_datasets import get_file
from .io import read_json, write_json, load_img, load_distmat, load_spinmat, load_labels
from .nulls import _img_density_for_neuromaps

# Set the default nispace data directory environment variable
os.environ['NISPACE_DATA_DIR'] = str(Path.home() / "nispace-data")

datalib_dir =Path(__file__).parent / "datalib"
reference_lib = read_json(datalib_dir / "reference.json")
template_lib = read_json(datalib_dir / "template.json")
parcellation_lib = read_json(datalib_dir / "parcellation.json")
example_lib = read_json(datalib_dir / "example.json")

# ==================================================================================================
# DEPRECATION MESSAGE STRINGS
# ==================================================================================================

_DEPR_NISPACE_DATA_DIR = (
    "The 'nispace_data_dir' parameter is deprecated and will be removed in the first non-dev release. "
    "Please use the NISPACE_DATA_DIR environment variable instead."
)
_DEPR_FETCH_PARC_LEGACY = (
    "Passing 'space=' to fetch_parcellation() and receiving individual arrays is deprecated "
    "and will be removed in the first non-dev release. "
    "Use return_parcellation_only=True (or omit space=) to get a Parcellation object, "
    "then call .get_image(), .get_dist_mat(), etc. as needed."
)
_DEPR_COMBINED_PARC_NAME = (
    "Combined parcellation '{old}' uses a deprecated naming format "
    "(concatenated or space-separated). "
    "Use the '+' separator or a tuple instead: '{new}'. "
    "Support for the old format will be removed in the first non-dev release."
)

def keys2list(dct):
    return list(dct.keys())

def keys2str(dct, sep=", "):
    return sep.join(list(dct.keys()))

# TODO (first non-dev release): remove nispace_data_dir parameter from all fetch_* functions and delete _resolve_nispace_data_dir()
def _resolve_nispace_data_dir(nispace_data_dir):
    if nispace_data_dir is not None:
        lgr.warning(_DEPR_NISPACE_DATA_DIR)
        os.environ["NISPACE_DATA_DIR"] = str(nispace_data_dir)
    return os.getenv('NISPACE_DATA_DIR')

# EMPTY NISPACE DATA DIR ===========================================================================

# _EMPTY_DATA_CONFIRMED = False
# def empty_nispace_data_dir(nispace_data_dir: Union[str,Path] = None):
#     global _EMPTY_DATA_CONFIRMED
#     if nispace_data_dir is None:
#         nispace_data_dir =Path.home() / "nispace-data"
#     if not _EMPTY_DATA_CONFIRMED:
#         lgr.warning("If you call this function again, it will remove all contents of your NiSpace "
#                     f"data directory at {nispace_data_dir}.")
#         lgr.warning("Call it again to proceed.")
#         _EMPTY_DATA_CONFIRMED = True
#     else:
#         lgr.warning(f"Emptying nispace data dir at {nispace_data_dir}.")
#         shutil.rmtree(nispace_data_dir)
#         nispace_data_dir.mkdir(parents=True, exist_ok=True)


# FILE HANDLING ====================================================================================

def _file_desc(fname, feature_position):
    if isinstance(fname,Path):
        fname = fname.name
    fname = fname.split(".")[0]
    if isinstance(feature_position, int):
        return fname.split("_")[feature_position].split("-")[1]
    elif isinstance(feature_position, str):
        return fname.split(f"{feature_position}-")[1].split("_")[0]
    
# BRAIN TEMPLATES ==================================================================================

def fetch_template(template: str = "MNI152NLin2009cAsym",
                   res: str = None,
                   desc: str = None,
                   #parcellation: str = None,
                   hemi: Union[List[str], str] = ["L", "R"],
                   nispace_data_dir: Union[str,Path] = None,
                   overwrite: bool = False,
                   check_file_hash: bool = True,
                   verbose: bool = True):
    """
    Fetch a brain template.
    
    Parameters
    ----------
    template : str, optional
        The template to fetch. Default is "MNI152NLin2009cAsym".
        
    res : str, optional
        The resolution of the template to fetch. Defaults: "1mm" for MNI152, "41k" for
        fsaverage (available: 3k/10k/41k/164k), "32k" for fsLR (available: 4k/8k/32k/164k).

    desc : str, optional
        The image type. Defaults: "T1w" for MNI152 (also: brain/mask/gmprob); "pial" for
        fsaverage (also: white/inflated/sphere/sulc/medial/vaavg); "midthickness" for fsLR
        (also: inflated/veryinflated[32k+]/sphere/sulc/medial/vaavg).
        
    hemi : list of str, optional
        The hemispheres to fetch. Default is ["L", "R"].
        
    nispace_data_dir : str orPath, optional
        The directory containing the NiSpace data. Default is None.
        
    Returns
    -------
    The template.
    """
    verbose = set_log(lgr, verbose)
    
    # check if template exists
    if template not in template_lib:
        raise ValueError(f"Template '{template}' not found. Available: {keys2str(template_lib)}")
    
    nispace_data_dir = _resolve_nispace_data_dir(nispace_data_dir)

    # paths
    base_dir =Path(nispace_data_dir) / "template" / template
    map_dir = base_dir / "map"
    
    # set defaults:
    if "mni" in template.lower():
        res = "1mm" if res is None else res
        desc = "T1w" if desc is None else desc
        hemi = None
    elif "fsa" in template.lower():
        res = "41k" if res is None else res
        desc = "pial" if desc is None else desc
        if hemi is None:
            hemi = ["L", "R"]
    elif "fslr" in template.lower():
        res = "32k" if res is None else res
        desc = "midthickness" if desc is None else desc
        if hemi is None:
            hemi = ["L", "R"]
    
    # check settings
    if res not in template_lib[template]:
        raise ValueError(f"res = '{res}' not defined. Choose one of {keys2str(template_lib[template])}!")
    if desc not in template_lib[template][res]:
        raise ValueError(f"desc = '{desc}' not defined. Choose one of {keys2str(template_lib[template][res])}!")
    if hemi is not None:
        if isinstance(hemi, str):
            hemi = [hemi]
        if hemi not in [["L"], ["R"], ["L", "R"]]:
            raise ValueError(f"hemi = '{hemi}' not defined. Choose one of 'L', 'R', or ['L', 'R']!")
    
    # get kwargs
    get_file_kwargs = dict(overwrite=overwrite, hash_check=check_file_hash)
    
    # get file
    lgr.info(f"Loading {template} '{desc}' template in '{res}' resolution.")
    if "mni" in template.lower():
        tpl_file = get_file(
            map_dir / desc / f"tpl-{template}_desc-{desc}_res-{res}.%s", 
            **template_lib[template][res][desc], 
            **get_file_kwargs, 
        )
    else:
        tpl_file = ()
        for h in hemi:
            tpl_file += get_file(
                map_dir / desc / f"tpl-{template}_desc-{desc}_res-{res}_hemi-{h}.%s", 
                **template_lib[template][res][desc][h], 
                **get_file_kwargs, 
            ),
        if len(tpl_file) == 1: 
            tpl_file = tpl_file[0]
    
    # f
    return tpl_file

# PARCELLATIONS ===================================================================================

def _parc_alias(parcellation: str):
    if "alias" in parcellation_lib[parcellation]:
        parc = parcellation_lib[parcellation]["alias"]
    else:
        parc = parcellation
    return parc

def _parc_symmetric(parc_labels):
    labels_lh = [l.split("hemi-L")[1] for l in parc_labels if "hemi-L" in l]
    labels_rh = [l.split("hemi-R")[1] for l in parc_labels if "hemi-R" in l]
    if not labels_lh or not labels_rh:
        return False
    if labels_lh == labels_rh:
        return True
    return False

def _print_parcellations():
    return ", ".join([p for p in parcellation_lib.keys() if "alias" not in parcellation_lib[p]])

def _check_parcellation(parcellation, force_list: bool = False, force_str: bool = False,
                        raise_not_found=True):
    """
    Check if a parcellation name is valid and return the correct parcellation name as a string or
    a list of strings containing a cortex-subcortex combination.

    Accepts:
    - Single name string: ``"Schaefer100"`` or ``"Schaefer100Parcels7Networks"``
    - "+" combined string: ``"Schaefer100+TianS1"`` — preferred new format
    - Tuple of two names: ``("Schaefer100", "TianS1")`` — converted to "+" form
    - Concatenated string: ``"Schaefer100TianS1"`` — deprecated, use "+" form instead
    """

    # 0a. Accept tuple/list → convert to "+" string
    if isinstance(parcellation, (tuple, list)):
        parcellation = "+".join(str(p) for p in parcellation)

    # Path strings are not library names — catch before substring matching eats the path
    if isinstance(parcellation, os.PathLike) or (
        isinstance(parcellation, str) and (
            os.sep in parcellation
            or "/" in parcellation
            or any(parcellation.endswith(ext) for ext in (".nii", ".nii.gz", ".gii", ".gii.gz"))
        )
    ):
        if raise_not_found:
            lgr.critical_raise(
                f"'{parcellation}' looks like a file path, not an integrated parcellation name. "
                "Pass a pathlib.Path object or use a registered library name.",
                ValueError,
            )
        return None

    # helper function to check if an iterable of n=2 parcellations are cortex-subcortex combinations
    def _check_cortex_subcortex(parc):
        levels = []
        for p in parc:
            levels.append(parcellation_lib[p]["level"])
        if set(levels) != {"cortex", "subcortex"}:
            lgr.critical_raise(f"Only cortex-subcortex combinations are allowed, not: {', '.join(levels)} ",
                                ValueError)
        else:
            # if we got to here, we have a cortex-subcortex combination; now ensure correct order
            return [parc[levels.index("cortex")], parc[levels.index("subcortex")]]

    assert isinstance(parcellation, str), \
        f"Parcellation must be a string or tuple/list of strings, not {type(parcellation)}!"

    # (2) Check if it is in parcellation_lib as is, or resolve via alias
    parc = None
    if parcellation in parcellation_lib:
        parc = _parc_alias(parcellation)
        # alias may point to a "+" combined name (e.g. "BrainnetomeCortical+BrainnetomeSubcortical")
        # that is not itself a library key — fall through to the "+" split path below
        if isinstance(parc, str) and parc not in parcellation_lib:
            parcellation = parc
            parc = None

    # (2b) "+" combined form: split, resolve each part, validate cx+sc.
    # Handles both direct user input ("Schaefer100+TianS1") and alias-resolved "+" targets.
    if parc is None and "+" in parcellation:
        parts = [p.strip() for p in parcellation.split("+")]
        if len(parts) != 2:
            lgr.critical_raise(
                f"Combined parcellation '{parcellation}' must contain exactly 2 '+'-separated names.",
                ValueError,
            )
            return None
        resolved = []
        for p in parts:
            r = _check_parcellation(p, raise_not_found=raise_not_found)
            if r is None:
                return None
            if isinstance(r, list):
                lgr.critical_raise(
                    f"Each part of a '+'-combined name must resolve to a single parcellation, not '{p}'.",
                    ValueError,
                )
                return None
            resolved.append(r)
        parc = _check_cortex_subcortex(resolved)
        if force_list and not force_str:
            return parc
        elif force_str and not force_list:
            return "+".join(parc)
        return parc

    # (3) Partial match — old concatenated or space-separated combined form (deprecated)
    if parc is None:
        # get a list of potential partial matches; skip combined-alias keys (their resolved alias
        # is not itself a library entry, so they would pollute de-nesting in step 3b)
        parc_matches = list(set([
            _parc_alias(p) for p in parcellation_lib
            if p in parcellation and _parc_alias(p) in parcellation_lib
        ]))
        # (3a) No match found: raise error
        if len(parc_matches) == 0:
            if raise_not_found:
                lgr.critical_raise(f"Parcellation '{parcellation}' not found.\nAvailable "
                                   f"(cortex-subcortex-combinations allowed): {_print_parcellations()}",
                                   FileNotFoundError)
            else:
                return
        # (3b) > 2 matches found: check if matches are contained in each other or raise error
        elif len(parc_matches) > 2:
            # (3b1) check if matches are contained in each other and remove the contained ones
            parc = parc_matches.copy()
            for p in parc_matches:
                if any([p in p_other for p_other in set(parc_matches) - {p}]):
                    parc.remove(p)
            # (3b2) if still not 2, raise error
            if len(parc) != 2:
                lgr.critical_raise(f"Parcellation '{parcellation}' matches more than 2 parcellations: "
                                   "{', '.join(parc_matches)}.",
                                   ValueError)
        # (3c) 1 match found: use it
        elif len(parc_matches) == 1:
            parc = parc_matches[0]
        # (3d) 2 matches found: check if they are cortex-subcortex combinations
        else:
            parc = _check_cortex_subcortex(parc_matches)
        # Deprecation: combined name resolved via old concatenated/space-separated form
        # TODO (first non-dev release): remove partial-match resolution for combined names
        if isinstance(parc, list):
            lgr.warning(_DEPR_COMBINED_PARC_NAME.format(old=parcellation, new="+".join(parc)))

    # output format
    if force_list and not force_str and isinstance(parc, str):
        parc = [parc]
    elif force_str and not force_list and isinstance(parc, list):
        parc = "+".join(parc)
    return parc
                

def fetch_parcellation(parcellation: str = _PARC_DEFAULT,
                       space: str = None,
                       hemi: Union[List[str], str] = ["L", "R"],
                       bilateral: bool = False,
                       return_parcellation_only: bool = False,
                       # TODO (first non-dev release): remove all return_* kwargs and legacy space= array-return path; keep return_parcellation_only only
                       return_labels: bool = True,
                       return_space: bool = False,
                       return_resolution: bool = False,
                       return_symmetric: bool = False,
                       return_dist_mat: bool = False,
                       return_spin_mat: bool = False,
                       return_loaded: bool = True,
                       nispace_data_dir: Union[str,Path] = None,
                       overwrite: bool = False,
                       check_file_hash: bool = True,
                       verbose: bool = True):
    """
    Fetch a parcellation.
    """
    verbose = set_log(lgr, verbose)
    
    # check parcellation and return correct name or list of two names
    parc = _check_parcellation(parcellation)
    # if list, we need to merge parcellation and associated data , so we need to load stuff
    return_loaded = True if isinstance(parc, str) else return_loaded
    
    nispace_data_dir = _resolve_nispace_data_dir(nispace_data_dir)

    # function to load individual parcellation and associated data
    def load_parc(p, space=space, hemi=hemi, return_labels=return_labels, return_space=return_space,
                  return_resolution=return_resolution, return_symmetric=return_symmetric,
                  return_dist_mat=return_dist_mat, return_spin_mat=return_spin_mat, return_loaded=return_loaded,
                  nispace_data_dir=nispace_data_dir, overwrite=overwrite, check_file_hash=check_file_hash):
        
        # Check space — filter to actual space entries (dicts with "map" key)
        _avail_spaces = {k: v for k, v in parcellation_lib[p].items() if isinstance(v, dict) and "map" in v}
        if space is None:
            space = next(iter(_avail_spaces))
        else:
            if space not in _avail_spaces:
                lgr.critical_raise(f"Space '{space}' not found for parcellation '{p}'.\n"
                                   f"Available: {', '.join(_avail_spaces)}",
                                   ValueError)
        
        # data directory
        base_dir =Path(nispace_data_dir) / "parcellation" / p / space
        
        # Symmetry
        symmetric = parcellation_lib[p].get("symmetric", parcellation_lib[p][space].get("symmetric", True))
        
        # LOAD
        _level = parcellation_lib[p].get("level") or parcellation_lib[p][space].get("level", "")
        _doi = parcellation_lib[p].get("citation", {}).get("doi", "")
        lgr.info(
            f"Loading {_level} parcellation '{p}' in '{space}' space."
            + (f" DOI: {_doi}" if _doi else "")
        )
        
        # get kwargs
        get_file_kwargs = dict(overwrite=overwrite, hash_check=check_file_hash)
        
        # volume
        if "mni" in space.lower():

            # get files
            parcellation_file = get_file(
                base_dir / f"parc-{p}_space-{space}.%s", **parcellation_lib[p][space]["map"],
                **get_file_kwargs,
            )
            if return_labels:
                label_file = get_file(
                    base_dir / f"parc-{p}_space-{space}.label.txt", **parcellation_lib[p][space]["label"],
                    **get_file_kwargs,
                )
            if return_dist_mat:
                distmat_file = get_file(
                    base_dir / f"parc-{p}_space-{space}.dist.csv.gz", **parcellation_lib[p][space]["distmat"],
                    **get_file_kwargs,
                )
            if return_spin_mat:
                spinmat_file = None  # spin tests not available for volumetric parcellations
        
        # surface
        else:
            
            # check hemis
            if isinstance(hemi, str):
                hemi = [hemi]
            if hemi not in [["L"], ["R"], ["L", "R"]]:
                raise ValueError(f"hemi = '{hemi}' not defined. Choose one of 'L', 'R', or ['L', 'R']!")

            # get files
            parcellation_file, label_file, distmat_file, spinmat_file = (), (), (), ()
            for h in hemi:
                parcellation_file += get_file(
                    base_dir / f"parc-{p}_space-{space}_hemi-{h}.%s", **parcellation_lib[p][space]["map"][h],
                    **get_file_kwargs,
                ),
                if return_labels:
                    label_file += get_file(
                        base_dir / f"parc-{p}_space-{space}_hemi-{h}.label.txt", **parcellation_lib[p][space]["label"][h],
                        **get_file_kwargs,
                    ),
                if return_dist_mat:
                    if "fslr" in space.lower():
                        lgr.warning("Distance matrices for fslr spaces are currently not available. Returning None.")
                        distmat_file += None,
                    else:
                        distmat_file += get_file(
                            base_dir / f"parc-{p}_space-{space}_hemi-{h}.dist.csv.gz", **parcellation_lib[p][space]["distmat"][h],
                            **get_file_kwargs,
                        ),
                if return_spin_mat:
                    if "spinmat" in parcellation_lib[p][space]:
                        spinmat_file += get_file(
                            base_dir / f"parc-{p}_space-{space}_hemi-{h}.spin.npy", **parcellation_lib[p][space]["spinmat"][h],
                            **get_file_kwargs,
                        ),
                    else:
                        spinmat_file += None,
            if return_spin_mat and "spinmat" not in parcellation_lib[p][space]:
                lgr.info(f"No pre-computed spin matrix available for '{p}' in '{space}' space.")
            if len(parcellation_file) == 1:
                parcellation_file = parcellation_file[0]
                if return_labels:
                    label_file = label_file[0]
                if return_dist_mat:
                    distmat_file = distmat_file[0]
                if return_spin_mat:
                    spinmat_file = spinmat_file[0]
            
        # return      
        
        # build output
        out = {}
        # parc
        out["parc"] = load_img(parcellation_file) if return_loaded else parcellation_file
        # label
        if return_labels:
            out["label"] = load_labels(label_file) if return_loaded else label_file
        # space
        if return_space:
            out["space"] = space
        # res
        if return_resolution:
            out["res"] = _img_density_for_neuromaps(load_img(parcellation_file))
        # symmetric
        if return_symmetric:
            out["sym"] = symmetric
        # distmat
        if return_dist_mat:
            out["distmat"] = load_distmat(distmat_file) if return_loaded else distmat_file
        # spinmat
        if return_spin_mat:
            out["spinmat"] = load_spinmat(spinmat_file) if return_loaded else spinmat_file

        return out
    
    # ---- NEW PATH: space=None OR return_parcellation_only=True → Parcellation object ----
    if space is None or return_parcellation_only:
        from .core.parcellation import Parcellation
        if isinstance(parc, list):
            parc_obj = Parcellation.from_nispace_library(
                parc,
                [parcellation_lib[parc[0]], parcellation_lib[parc[1]]],
                nispace_data_dir,
                load_dist_mat=return_dist_mat,
                load_spin_mat=return_spin_mat,
                overwrite=overwrite,
                check_file_hash=check_file_hash,
                verbose=verbose,
            )
        else:
            parc_obj = Parcellation.from_nispace_library(
                parc,
                parcellation_lib[parc],
                nispace_data_dir,
                load_dist_mat=return_dist_mat,
                load_spin_mat=return_spin_mat,
                overwrite=overwrite,
                check_file_hash=check_file_hash,
                verbose=verbose,
            )
        if bilateral:
            parc_obj.make_bilateral()
        # hemisphere selection: normalise hemi arg and call select_hemi if single hemi
        _hemi_set = set([hemi] if isinstance(hemi, str) else list(hemi))
        if not _hemi_set >= {"L", "R"}:
            h = next(iter(_hemi_set), None)
            if h in ("L", "R"):
                parc_obj.select_hemi(h, verbose=verbose)
        # activate the requested space if one was given
        if space is not None:
            parc_obj.set_active_space(space)
        return parc_obj

    # ---- LEGACY PATH: space explicitly given and return_parcellation_only=False ----
    lgr.warning(_DEPR_FETCH_PARC_LEGACY)
    # run load_parc for a single parcellation
    if isinstance(parc, str):
        out = load_parc(parc)
        if len(out) == 1:
            return list(out.values())[0]
        else:
            return tuple(out.values())

    # run load_parc for 2 parcellations
    out_cortex = load_parc(parc[0])
    out_subcortex = load_parc(parc[1])
    lgr.info(f"Merging to cortex-subcortex parcellation '{parc[0]}+{parc[1]}'.")
        
    # now, we will have to combine the data
    out = {}
    # combine parcellations
    out["parc"] = merge_parcellations([out_cortex["parc"], out_subcortex["parc"]], quick=True)#[0]
    # label
    if return_labels:
        out["label"] = out_cortex["label"] + out_subcortex["label"]
    # space
    if return_space:
        out["space"] = out_cortex["space"]
    # res
    if return_resolution:
        out["res"] = out_cortex["res"]
    # symmetric
    if return_symmetric:
        out["sym"] = True if out_cortex["sym"] and out_subcortex["sym"] else False
    # distmat
    if return_dist_mat:
        lgr.info("Distance matrices for merged parcellations are currently not available. Returning None.")
        out["distmat"] = None
    # spinmat
    if return_spin_mat:
        out["spinmat"] = None

    # return
    if len(out) == 1:
        return list(out.values())[0]
    else:
        return tuple(out.values())

def fetch_collection(collection: Union[str,Path, np.ndarray, pd.DataFrame, pd.Series, list],
                     dataset: str = None,
                     maps: list = None,
                     set_size_range: Union[None, Tuple[int, int]] = None,
                     weight_range: Union[None, Tuple[float, float]] = None,
                     weight_quantile: float = None,
                     set_specificity: float = None,
                     return_maps: bool = False,
                     nispace_data_dir: Union[str,Path] = None,
                     overwrite: bool = False,
                     check_file_hash: bool = True,
                     verbose: bool = True):
    """
    Fetch a collection that defines a subset (and optional grouping) of maps.

    A collection is a mapping from map IDs to optional set labels and weights. The result
    is a DataFrame with columns ``["map"]``, ``["set", "map"]``, or ``["set", "map", "weight"]``
    depending on the collection content.

    Three `.collect` file formats are supported:

    1. **Simple list** — plain text, one map ID per line, ``map`` header.
    2. **JSON set dict** — ``{"set_name": ["map1", "map2", ...], ...}``.
    3. **CSV set table** — columns: ``map``, ``set, map``, or ``set, map, weight``.

    Parameters
    ----------
    collection : str, Path, ndarray, DataFrame, Series, or list
        When ``dataset`` is given: the name of an integrated collection (e.g. ``"All"``,
        ``"BrainSpanWeights"``).
        When ``dataset`` is None: a path to a ``.collect`` file, or an in-memory
        array-like / DataFrame that is used directly.
    dataset : str, optional
        Name of an integrated NiSpace reference dataset (e.g. ``"mrna"``, ``"pet"``).
        If provided, ``collection`` must be the name of one of that dataset's registered
        collections.
    maps : list, optional
        Restrict to this subset of map IDs after loading.
    set_size_range : tuple (int, int), optional
        Keep only sets whose membership count falls within ``[min, max]`` (inclusive).
    weight_range : tuple (float, float), optional
        Keep only entries whose weight is within ``[min, max]`` (inclusive).
        Ignored when the collection has no weights.
    weight_quantile : float, optional
        Within each set, keep only entries with weight ≥ this quantile. Ignored when
        the collection has no weights.
    set_specificity : float in (0, 1], optional
        Keep only maps that appear in ≤ ``set_specificity`` fraction of all sets,
        i.e. discard ubiquitous maps.
    return_maps : bool, default False
        If True, return a tuple ``(collection_df, maps_avail)`` where ``maps_avail`` is
        the deduplicated list of map IDs after all filters.
    nispace_data_dir : str or Path, optional
        Override the NiSpace data directory (default: ``$NISPACE_DATA_DIR``).
    overwrite : bool, default False
        Re-download the collection file even if it is already cached.
    check_file_hash : bool, default True
        Verify the SHA-256 hash of the downloaded file.
    verbose : bool, default True
        Print progress messages.

    Returns
    -------
    collection_df : DataFrame
        Columns: ``["map"]`` for unstructured collections; ``["set", "map"]`` for grouped
        collections; ``["set", "map", "weight"]`` for weighted collections.
    maps_avail : list of str
        Only returned when ``return_maps=True``. Deduplicated map IDs present in
        ``collection_df`` after filtering.
    """
    verbose = set_log(lgr, verbose)
    
    # If dataset is provided, assume to load integrated collection
    # check if dataset is valid
    if dataset is not None:
        if dataset not in reference_lib:
            lgr.critical_raise(f"Dataset '{dataset}' not found. Available: {keys2str(reference_lib)}",
                               ValueError)
        else:
            
            nispace_data_dir = _resolve_nispace_data_dir(nispace_data_dir)
            
            # base dir
            base_dir =Path(nispace_data_dir) / "reference" / dataset
            
            # get integrated collection
            if collection in reference_lib[dataset]["collection"]:
                lgr.info(f"Loading integrated collection '{collection}' for dataset '{dataset}'.")
                collection_path = base_dir / f"collection-{collection}.collect"
                collection_file = get_file(
                    collection_path, **reference_lib[dataset]["collection"][collection],
                    overwrite=overwrite, hash_check=check_file_hash,
                )
            else:
                lgr.critical_raise(f"Collection '{collection}' not found for dataset '{dataset}'. "
                                   f"Available: {keys2str(reference_lib[dataset]['collection'])}",
                                   ValueError)

    # dataset is not provided, assume to load custom collection
    else:
        
        # check if collection is a file
        if isinstance(collection, (str,Path)):
            collection_file =Path(collection)
            if collection_file.exists():
                lgr.info(f"Loading custom collection from file: {collection_file}")
            else:
                lgr.critical_raise(f"Assuming collection '{collection_file}' to be a file, but it does not exist! "
                                   "Ensure that the file exists and try again. If you want to load an integrated collection, "
                                   "use the 'dataset' argument.",
                                   ValueError)
        
        # else, assume to load array-like object
        else:
            lgr.info(f"Loading custom collection of type {type(collection)}.")
            collection_file = collection

    # Load collection file; 1-column df (= maps) or 2-column df (= set and maps)
    collection_df = _load_collection(collection_file)
    
    # apply filters
    collection_df, maps_avail = _apply_collection_filter(
        collection_df=collection_df,
        maps=maps,
        set_size_range=set_size_range,
        weight_range=weight_range,
        weight_quantile=weight_quantile,
        set_specificity=set_specificity
    )
    
    # return
    return collection_df if not return_maps else (collection_df, maps_avail)
        

def apply_collection(data: pd.DataFrame, collection: pd.DataFrame):
    if not np.isin(["map"], collection.columns).all():
        lgr.critical_raise("collection must have at least a 'map' column.")
    
    maps_intersection = data.index.intersection(collection["map"].unique())
    collection_df_intersection = collection[collection["map"].isin(maps_intersection)]
    data_out = data.copy()
    data_out = data_out.loc[collection_df_intersection["map"]]     
    data_out.index = pd.MultiIndex.from_frame(collection_df_intersection)
    return data_out


# REFERENCE DATA - PRIVATE =========================================================================

def _filter_maps(maps_avail: List[str], 
                 maps: Union[str, List[str], Dict[str, Union[str, list]]]) -> List[Path]:
    
    def matches_filters(map_name: str, filters: Dict[str, Union[str, List[str]]]) -> bool:
        for filter_name, filter_content in filters.items():
            if filter_content not in [None, False, "", []]:
                if isinstance(filter_content, (str, int)):
                    filter_content = [filter_content]
                filter_content = list(map(str, filter_content))
                if filter_name == "n" and filter_content[0].startswith(">"):
                    try:
                        filter_n = int(filter_content[0].replace(">", ""))
                        n_value = int(_file_desc(map_name, 2))
                        if n_value <= filter_n:
                            return False
                    except (ValueError, IndexError):
                        continue  # Skip this filter if parsing fails
                else:
                    if not any(f"{filter_name}-{content}".lower() in map_name.lower() 
                               for content in filter_content):
                        return False
        return True

    if isinstance(maps, str):
        maps = [maps]
    if isinstance(maps, list):
        maps = list(set(maps))
        filtered_maps = [f for f in maps_avail if any(map_str in f for map_str in maps)]
    elif isinstance(maps, dict):
        filtered_maps = [f for f in maps_avail if matches_filters(f, maps)]
    else:
        filtered_maps = maps_avail
        
    return filtered_maps


def _load_collection(collection_path):
    
    # if path, read file
    if isinstance(collection_path, (str,Path)):
        collection_path =Path(collection_path)
        ext = collection_path.suffix
        
        # if "collect" file, detect if dict or table
        if ext == ".collect":
            with open(collection_path) as f:
                header = f.readline()
                if header.startswith("{"):
                    ext = ".json"
                else:
                    ext = ".csv"
        
        # if json, load into dict
        if ext == ".json":
            collection = read_json(collection_path)
                
        # else, try to directly load as table file
        else:
            with open(collection_path) as f:
                header = f.readline().strip("\n")
                if any([h in header for h in ["set", "map", "weight"]]):
                    header = 0
                else: 
                    header = None
            collection = pd.read_csv(collection_path, header=header, sep=",")
    else:
        collection = collection_path
        
    # if array, convert all do df
    if isinstance(collection, (np.ndarray, pd.DataFrame, pd.Series, list)):   
        collection = pd.DataFrame(collection)
        
    # if dict, convert to df as well
    elif isinstance(collection, dict):   
        collection = pd.concat([pd.DataFrame({0:k, 1:v}) for k, v in collection.items()])
        
    # else
    else:
        raise TypeError(f"Datatype {type(collection_path)} not accepted for argument 'collection'.")
        
    # process depending on number of columns
    n_cols = collection.shape[1]
    if n_cols == 0:
        raise ValueError("No columns detected in collection file?!")
    elif n_cols == 1:
        collection.columns = ["map"]
    elif n_cols == 2:
        collection.columns = ["set", "map"]
    elif n_cols == 3:
        collection.columns = ["set", "map", "weight"]
    else:
        raise ValueError(f"Collection file with > 3 columns not supported ({n_cols} columns)!")
    
    # return
    return collection.reset_index(drop=True)


def _apply_collection_filter(#dataset: str,
                             collection_df: pd.DataFrame,
                             maps: List[Union[str,Path]] = None, 
                             #collection: str,
                             #nispace_data_dir: Union[str,Path],
                             set_size_range: Union[None, Tuple[int, int]] = None,
                             weight_range: Union[None, Tuple[float, float]] = None,
                             weight_quantile: Union[None, float] = None,
                             set_specificity: Union[None, float] = None,
                             #overwrite: bool = False,
                             #check_file_hash: bool = True
                             ) -> List[Path]:
    # Apply maps filter
    lgr.info(f"Filtering maps by collection.")
    if maps is None or len(maps) == 0:
        filtered_map_files = collection_df["map"].unique()
    elif isinstance(maps[0],Path):
        map_names = [_rm_ext(f.name) for f in maps]
        filtered_map_files = [
            maps[map_names.index(f_name)] for f_name in collection_df["map"].unique()
            if f_name in map_names
        ]
        collection_df = collection_df[collection_df["map"].isin(map_names)]
    else:
        filtered_map_files = [
            f_name for f_name in collection_df["map"].unique() 
            if any(m == f_name for m in maps)
        ]
        collection_df = collection_df[collection_df["map"].isin(filtered_map_files)]
    
    # Apply
    if set_specificity is not None:
        n_sets = len(collection_df["set"].unique())
        collection_df = (
            collection_df
            .groupby("map")
            .filter(lambda x: x.shape[0] <= n_sets * set_specificity)
            .reset_index(drop=True)
        )
        lgr.info(f"Keeping maps occuring in <= {set_specificity:.02%} of "
                 f"{collection_df['set'].nunique()} retained sets.")
    
    # Apply weight quantile filter
    if weight_quantile is not None:
        if "weight" not in collection_df.columns:
            lgr.warning("Collection does not seem to contain weights, will not apply weight filter.")
        else:
            collection_df = (
                collection_df
                .groupby("set", sort=False)
                .apply(lambda x: x[x.weight >= x.weight.quantile(0.9)])   
                .reset_index(drop=True)
            )
            lgr.info(f"Filtered to maps with weights >= quantile {weight_quantile} within each set.")
            
    # Apply absolute weight filter
    if weight_range is not None:
        if "weight" not in collection_df.columns:
            lgr.warning("Collection does not seem to contain weights, will not apply weight filter.")
        else:
            weight_range = [
                x if x is not None else x_ 
                for x, x_ 
                in zip(weight_range, (-np.inf, np.inf))
            ]
            collection_df = collection_df[collection_df["weight"].between(*weight_range, inclusive="both")]
            lgr.info(f"Filtered to {len(collection_df['set'].unique())} collection sets with weights between "
                    f"{weight_range[0]} and {weight_range[1]}.")
        
    # Apply size filter
    if set_size_range is not None:
        if "set" in collection_df.columns and isinstance(set_size_range, (tuple, list)):
            set_size_range = [
                x if x is not None else x_ 
                for x, x_ 
                in zip(set_size_range, (1, np.inf))
            ]
            collection_df = (
                collection_df
                .groupby("set", sort=False)
                .filter(lambda x: set_size_range[0] <= x.shape[0] <= set_size_range[1]) 
                .reset_index(drop=True)  
            )
            n_sets = len(collection_df["set"].unique())
            if n_sets == 0:
                lgr.critical_raise(f"No collection sets found with between {set_size_range[0]} and "
                                   f"{set_size_range[1]} maps. Adjust the 'set_size_range' parameter.",
                                   ValueError)
            filtered_map_files = list( set(filtered_map_files).intersection(set(collection_df["map"])) )
            lgr.info(f"Filtered to {n_sets} collection sets with between "
                     f"{set_size_range[0]} and {set_size_range[1]} maps.")

    return collection_df, filtered_map_files


def _load_parcellated_data(dataset: str,
                           nispace_data_dir: Union[str,Path],
                           parc: Union[str, List[str]],
                           map_files: List[str],
                           collection_df: pd.DataFrame,
                           standardize: bool,
                           set_size_range: Union[None, Tuple[int, int]] = None,
                           merge_how: str = "inner",
                           overwrite: bool = False,
                           check_file_hash: bool = True,
                           verbose: bool = True) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    verbose = set_log(lgr, verbose)
    
    # tab dir
    tab_dir =Path(nispace_data_dir) / "reference" / dataset / "tab"
    
    # parcellation can be string with one parcellation name or list of two parcellation names
    if isinstance(parc, str):
        lgr.info(f"Loading data parcellated with '{parc}'")
        # all to list
        parc = [parc]
    elif isinstance(parc, list):
        lgr.info(f"Loading and {merge_how}-merging data parcellated with '{parc[0]}' and '{parc[1]}'")
    else:
        lgr.critical_raise(f"Invalid parcellation type: {type(parc)}", ValueError)
    
    # loop through parcellations
    data = []
    for p in parc:
        # check if parcellation available for this data
        if p not in reference_lib[dataset]["tab"]:
            lgr.critical_raise(f"Dataset '{dataset}' is not available for parcellation '{p}'!\n"
                               f"Available: {keys2str(reference_lib[dataset]['tab'])}",
                               FileNotFoundError)
        # file
        parcellation_file = tab_dir / f"dset-{dataset}_parc-{p}.csv.gz"
        lgr.debug(f"Loading {parcellation_file}")
            
        # load data
        data.append(pd.read_csv(
            get_file(
                parcellation_file, **reference_lib[dataset]["tab"][p], 
                overwrite=overwrite, hash_check=check_file_hash,
            ), 
            index_col=0
        ))
        lgr.debug(f"Loaded parcellated data of shape {data[-1].shape}")
        lgr.debug(f"First 5 map names: {data[-1].index.to_list()[:5]}")
        
    # merge if necessary: all maps are kept even if they are not present in both parcellations
    if len(parc) > 1:
        data = data[0].merge(data[1], how=merge_how, left_index=True, right_index=True)
    else:
        data = data[0]

    # Apply filter to the dataframe index
    lgr.debug(f"Applying filtering based on maps, first 5: {map_files[:5]}")
    if map_files and isinstance(map_files[0],Path):
        map_files = [_rm_ext(f.name) for f in map_files]
    data = data.loc[data.index.intersection(map_files)]
    lgr.debug(f"Shape after filtering based on map_names: {data.shape}")
    
    # Apply collection index (-> handles maps that are present multiple times in different sets)
    if collection_df is not None:
        data = apply_collection(data, collection_df)
        # Re-apply set size filter: maps missing from parcellated CSV can silently reduce set sizes
        if set_size_range is not None and "set" in data.index.names:
            size_range = [
                x if x is not None else x_
                for x, x_ in zip(set_size_range, (1, np.inf))
            ]
            set_counts = data.index.get_level_values("set").value_counts()
            valid_sets = set_counts[(set_counts >= size_range[0]) & (set_counts <= size_range[1])].index
            data = data[data.index.get_level_values("set").isin(valid_sets)]
            lgr.debug(f"After parcellated-data set size filter: {len(valid_sets)} sets remaining.")

    # Standardize
    if standardize:
        lgr.info("Standardizing parcellated data.")
        data = zscore_df(data, along="rows")

    return data
    

def _print_map_citation_table(meta: pd.DataFrame, map_info_cfg: dict):
    if meta is None or len(meta) == 0:
        return
    cite_col = map_info_cfg.get("cite_column", "doi")
    display_cols = [c for c in map_info_cfg.get("display_columns", []) if c in meta.columns]
    note_col = map_info_cfg.get("note_column")

    col_widths = {
        col: max((len(str(v)) for v in meta[col] if not pd.isna(v)), default=0)
        for col in display_cols
    }

    lines = []
    for _, row in meta.iterrows():
        parts = [
            ("" if pd.isna(row.get(col, float("nan"))) else str(row[col])).ljust(col_widths[col])
            for col in display_cols
        ]
        doi_str = ""
        if cite_col in row.index and not pd.isna(row[cite_col]):
            dois = [d.strip() for d in str(row[cite_col]).split(";") if d.strip()]
            doi_str = " " + "; ".join(
                f"https://doi.org/{d}" if not d.startswith("http") else d for d in dois
            )
        lines.append(("  " + "  ".join(parts) + doi_str).rstrip())
        if note_col and note_col in row.index and not pd.isna(row.get(note_col, float("nan"))):
            lines.append(f"   CAVE: {row[note_col]}")
    print("\n".join(lines))


def _print_references(dataset: str, meta: pd.DataFrame = None, collection_name: str = None):
    cfg = reference_lib[dataset]

    # 1. Description (replaces reference.txt)
    if "description" in cfg:
        paragraphs = cfg["description"].split("\n\n")
        print("\n\n".join(
            textwrap.fill(p.replace("\n", " "), width=100, break_long_words=False)
            for p in paragraphs
        ))

    # 2. Dataset-level citations
    for c in cfg.get("citations", []):
        print(f"  - {c['ref']}  https://doi.org/{c['doi']}")

    print(f"To ensure reproducibility, note the NiSpace version: {__version__} (commit: {__commit__}).\n")

    # 3. Per-map citation table (PET / enigma / cortexfeatures / tpm / bigbrain)
    if meta is not None and "map_info" in cfg:
        _print_map_citation_table(meta, cfg["map_info"])

    # 4. Collection-level citations (mrna / magicc gene sets)
    if collection_name and collection_name in cfg.get("collection", {}):
        for c in cfg["collection"][collection_name].get("citations", []):
            print(f"  [{collection_name}] {c['ref']}  https://doi.org/{c['doi']}")
    
    
# REFERENCE DATA - PUBLIC ==========================================================================

# TODO: fetch_reference is ~280 lines and handles two distinct code paths (image-path fetching vs.
# parcellated data loading). Consider splitting into fetch_reference_maps() and fetch_reference_tab()
# with a shared dispatcher, or at minimum extracting the two branches into private helpers.
def fetch_reference(dataset: str,
                    maps: Union[None, str, List[str], Dict[str, Union[str, list]]] = None,
                    space: str = None,
                    collection: str = None,
                    sets: Union[None, str, List[str]] = None,
                    set_size_range: Union[None, Tuple[int, int]] = None,
                    weight_range: Union[None, Tuple[float, float]] = None,
                    weight_quantile: Union[None, float] = None,
                    set_specificity: Union[None, float] = None,
                    parcellation: str = None,
                    bilateral: bool = False,
                    hemi: Union[str, List[str]] = None,
                    standardize_parcellated: bool = False,
                    return_metadata: bool = False,
                    print_references: bool = True,
                    osf_config_file: str = None,
                    github_config_file: str = None,
                    nispace_data_dir: Union[str,Path] = None,
                    overwrite: bool = False,
                    check_file_hash: bool = True,
                    verbose: bool = True):
    verbose = set_log(lgr, verbose)

    # --- handle Parcellation object passed as parcellation= ---
    from .core.parcellation import Parcellation as _Parcellation
    if isinstance(parcellation, _Parcellation):
        if parcellation._is_combined:
            parcellation = (parcellation._cx_name or "") + (parcellation._sc_name or "")
        else:
            parcellation = parcellation._name
    elif bilateral and parcellation is None:
        lgr.warning("bilateral=True has no effect without a parcellation.")

    # Check dataset availability
    if isinstance(dataset, str):
        dataset = dataset.lower()
        if dataset not in reference_lib:
            lgr.critical_raise(f"Dataset '{dataset}' not found! Available datasets: {keys2str(reference_lib)}",
                               ValueError)
        elif parcellation is None and "map" not in reference_lib[dataset]:
            lgr.critical_raise(f"Dataset '{dataset}' is only available as parcellated data, choose a parcellation!",
                               ValueError)
    else:
        lgr.critical_raise(f"Invalid dataset type; expecting string or pandas DataFrame/Series, got {type(dataset)}",
                           TypeError)
    lgr.info(f"Loading {dataset} maps.")

    nispace_data_dir = _resolve_nispace_data_dir(nispace_data_dir)

    # base dir
    base_dir =Path(nispace_data_dir) / "reference" / dataset
    map_dir = base_dir / "map"
    tab_dir = base_dir / "tab"
    
    # Check if parcellation is defined correctly and load map lists
    if parcellation is not None:
        # check parcellation and return correct name or list of two names
        parc = _check_parcellation(parcellation)

        # bilateral symmetry guard: only valid for symmetric library parcellations
        if bilateral:
            names_to_check = parc if isinstance(parc, list) else [parc]
            for _pname in names_to_check:
                if not isinstance(_pname, str) or _pname not in parcellation_lib:
                    continue
                is_sym = all(
                    space_data.get("symmetric", True)
                    for space_data in parcellation_lib[_pname].values()
                    if isinstance(space_data, dict)
                )
                if not is_sym:
                    raise ValueError(
                        f"bilateral=True requires a symmetric parcellation. "
                        f"'{_pname}' is not symmetric."
                    )

        # load maps from collection "All", which should be available for all datasets
        maps_avail = _load_collection(get_file(
            base_dir / f"collection-All.collect", **reference_lib[dataset]["collection"]["All"],
            overwrite=overwrite, hash_check=check_file_hash,
        ))["map"].to_list()
    
    # Check space availability and load map lists
    else:

        # auto-select space when none given: prefer vol default, then surf default, then first available
        if space is None:
            _avail_spaces = list({s for m in reference_lib[dataset]["map"].values() for s in m})
            if _SPACE_DEFAULT_VOL in _avail_spaces:
                space = _SPACE_DEFAULT_VOL
            elif _SPACE_DEFAULT_SURF in _avail_spaces:
                space = _SPACE_DEFAULT_SURF
            elif _avail_spaces:
                space = _avail_spaces[0]
            else:
                lgr.critical_raise(f"No image maps found for dataset '{dataset}'.", ValueError)
            lgr.info(f"Auto-selected space '{space}' for dataset '{dataset}'.")

        # get list of map image files
        maps_avail = [m for m, v in reference_lib[dataset]["map"].items() if space in v]
        if len(maps_avail) == 0:
            lgr.critical_raise(f"Found no maps for space '{space}' in dataset '{dataset}'.",
                               ValueError)
            
        # Remove private maps
        if not osf_config_file and not github_config_file:
            if "mni" in space.lower():
                maps_avail = [
                    m for m in maps_avail
                    if reference_lib[dataset]["map"][m][space]["host"] not in ["osfprivate", "github-nispace-private"]
                ]
            else:
                maps_avail = [
                    m for m in maps_avail
                    if (reference_lib[dataset]["map"][m][space]["L"]["host"]
                        if "L" in reference_lib[dataset]["map"][m][space]
                        else reference_lib[dataset]["map"][m][space]["R"]["host"])
                    not in ["osfprivate", "github-nispace-private"]
                ]
        
    lgr.debug(f"Loaded {len(maps_avail)} unfiltered map(s). "
              f"First 5: {maps_avail[:5] if len(maps_avail) >= 5 else maps_avail[:len(maps_avail)]}")

    # Filter by 'maps'
    if maps:
        n_tmp = len(maps_avail)
        lgr.info(f"Applying filter: {maps}")
        maps_avail = _filter_maps(maps_avail, maps)
        lgr.info(f"Filtered from {n_tmp} to {len(maps_avail)} maps.")
    
    # Filter by 'collection'
    if collection == "All":
        collection = None
    if collection:
        collection_df, maps_avail = fetch_collection(
            collection,
            dataset, 
            maps_avail, 
            return_maps=True,
            set_size_range=set_size_range,
            weight_range=weight_range,
            weight_quantile=weight_quantile,
            set_specificity=set_specificity,
            overwrite=overwrite, 
            check_file_hash=check_file_hash,
            verbose=verbose            
        )
    else:
        collection_df = None

    # Filter by 'sets'
    if sets is not None:
        if collection_df is None or "set" not in collection_df.columns:
            lgr.warning(f"'sets={sets}' filter ignored: no collection with sets is loaded.")
        else:
            if isinstance(sets, str):
                sets = [sets]
            avail_sets = collection_df["set"].unique().tolist()
            missing = [s for s in sets if s not in avail_sets]
            if missing:
                show = avail_sets[:20]
                lgr.critical_raise(
                    f"Set(s) {missing} not found in collection. Available ({len(avail_sets)}): {show}"
                    + (" ..." if len(avail_sets) > 20 else ""),
                    ValueError
                )
            collection_df = collection_df[collection_df["set"].isin(sets)]
            maps_avail = collection_df["map"].tolist()

    # Load tabulated data if 'parcellation' is specified
    if parcellation:
        data = _load_parcellated_data(
            dataset=dataset,
            parc=parc,
            map_files=maps_avail,
            collection_df=collection_df,
            standardize=standardize_parcellated,
            set_size_range=set_size_range,
            nispace_data_dir=nispace_data_dir,
            overwrite=overwrite,
            check_file_hash=check_file_hash,
            verbose=verbose,
        )
        # bilateral: average matched LH/RH columns by label prefix
        if bilateral:
            from .core.parcellation import _bilateral_labels_match
            ok, lh_idc, rh_idc, bilateral_cols, unmatched = _bilateral_labels_match(
                data.columns.tolist()
            )
            if not ok:
                lgr.warning(
                    f"Bilateral averaging: label matching failed "
                    f"({len(unmatched)} unmatched label(s): {unmatched[:5]}). "
                    "Returning non-bilateral data."
                )
            else:
                N_before = data.shape[1]
                lh_vals = data.iloc[:, lh_idc].values
                rh_vals = data.iloc[:, rh_idc].values
                data = pd.DataFrame(
                    (lh_vals + rh_vals) / 2,
                    index=data.index,
                    columns=bilateral_cols,
                )
                lgr.info(f"Bilateral averaging applied: {N_before} → {len(bilateral_cols)} parcels.")
        
    # Fetch paths to maps if no 'parcellation' is specified
    else:
        # get kwargs
        get_file_kwargs = dict(
            overwrite=overwrite, hash_check=check_file_hash,
            osf_config_file=osf_config_file, github_config_file=github_config_file)
        
        # MNI: one file per map
        if "mni" in space.lower():
            data = [
                get_file(
                    map_dir / m / f"{m}_space-{space}.%s", **reference_lib[dataset]["map"][m][space], 
                    **get_file_kwargs,
                ) 
                for m in maps_avail
            ]
        # surface: two files per map
        else:
            data = []
            for m in maps_avail:
                data.append(tuple([
                    get_file(
                        map_dir / m / f"{m}_space-{space}_hemi-{_h}.%s", **reference_lib[dataset]["map"][m][space][_h],
                        **get_file_kwargs,
                    )
                    for _h in reference_lib[dataset]["map"][m][space].keys()
                ]))
        
    # Hemisphere filter
    if hemi is not None:
        _hemi_set = set([hemi] if isinstance(hemi, str) else list(hemi))
        if not _hemi_set >= {"L", "R"}:
            _h_keep = next(iter(_hemi_set), None)
            if _h_keep in ("L", "R"):
                if isinstance(data, pd.DataFrame):
                    _prefix = "hemi-L_" if _h_keep == "L" else "hemi-R_"
                    _keep_cols = [c for c in data.columns if str(c).startswith(_prefix)]
                    if _keep_cols:
                        data = data[_keep_cols]
                        lgr.info(f"hemi='{_h_keep}': kept {len(_keep_cols)} parcels.")
                    else:
                        lgr.warning(
                            f"hemi='{_h_keep}': no columns with prefix '{_prefix}' found. "
                            "Returning all columns."
                        )
                else:
                    lgr.warning(
                        f"hemi='{_h_keep}' filtering is only supported for parcellated data "
                        "(parcellation= must be set). Full images returned."
                    )

    # Print references / return per-map info
    if return_metadata or print_references:
        if "map_info" in reference_lib[dataset]:
            meta = fetch_map_info(dataset, maps_avail, overwrite=overwrite, check_file_hash=check_file_hash)
        else:
            meta = None

        if return_metadata:
            data = (data + (meta,)) if isinstance(data, tuple) else (data, meta)
        if print_references & verbose:
            _print_references(dataset, meta, collection_name=collection)

    return data


def fetch_map_info(dataset: str,
                   maps: Union[str, list] = None,
                   overwrite: bool = False,
                   check_file_hash: bool = True,
                   nispace_data_dir: Union[str,Path] = None):
    if not isinstance(dataset, str):
        return None
    dataset = dataset.lower()
    if dataset not in reference_lib:
        return None
    if "metadata" not in reference_lib[dataset]:
        return None

    nispace_data_dir = _resolve_nispace_data_dir(nispace_data_dir)
    base_dir =Path(nispace_data_dir) / "reference" / dataset

    meta = pd.read_csv(
        get_file(
            base_dir / "map_info.csv", **reference_lib[dataset]["metadata"],
            overwrite=overwrite, hash_check=check_file_hash,
        ),
        index_col=0,
    )

    if maps is not None:
        if isinstance(maps, str):
            maps = [maps]
        meta = meta[meta.index.isin(maps)]

    return meta


def fetch_metadata(dataset: str,
                   maps: Union[str, list] = None,
                   collection: str = None,
                   overwrite: bool = False,
                   check_file_hash: bool = True,
                   nispace_data_dir: Union[str,Path] = None):
    """Deprecated alias for fetch_map_info()."""
    return fetch_map_info(dataset, maps=maps, overwrite=overwrite,
                          check_file_hash=check_file_hash, nispace_data_dir=nispace_data_dir)


# EXAMPLE DATA =====================================================================================
 
def fetch_example(example: str,
                  parcellation: str = None,
                  return_associated_data: bool = True,
                  nispace_data_dir: Union[str,Path] = None,
                  overwrite: bool = False,
                  check_file_hash: bool = True,
                  verbose: bool = True):
    """
    Fetch an example dataset.
    """
    verbose = set_log(lgr, verbose)
    
    nispace_data_dir = _resolve_nispace_data_dir(nispace_data_dir)

    # base dir
    base_dir =Path(nispace_data_dir) / "example"

    # check available
    example = example.lower()
    if example not in example_lib:
        lgr.critical_raise(f"Example '{example}' not found. Available: {list(example_lib.keys())}",
                           ValueError)
    
    # check parcellation
    if parcellation is not None:
        parc = _check_parcellation(parcellation, force_list=True)
    else:
        lgr.critical_raise("Currently, only parcellated example datasets are available. Please specify a parcellation.",
                           ValueError)
        
    # get kwargs
    get_file_kwargs = dict(overwrite=overwrite, hash_check=check_file_hash)
        
    # load tabulated data
    if all(p in example_lib[example]["tab"] for p in parc):
        lgr.info(f"Loading example dataset: '{example}', parcellated with: {'+'.join(parc)}.")
        example_data = pd.concat([
            pd.read_csv(
                get_file(
                    base_dir / f"example-{example}_parc-{p}.csv.gz", **example_lib[example]["tab"][p],
                    **get_file_kwargs,
                ), 
                index_col=0
            )
            for p in parc
        ], axis=1)
    else:
        lgr.critical_raise(f"Parcellation '{parcellation}' not found for example '{example}'.\n"
                           f"Available parcellations: {list(example_lib[example]['tab'].keys())}",
                           ValueError)
    
    # Check for info data
    if return_associated_data and "info" in example_lib[example]:
        lgr.info("Returning parcellated and associated subject data.")
        example_info = pd.read_csv(
            get_file(
                base_dir / f"example-{example}_info.csv", **example_lib[example]["info"],
                **get_file_kwargs,
            ),
            index_col=0
        )
        return example_data, example_info
    else:
        return example_data


def fetch_plot(
    name: str,
    kind: str = None,
    display: bool = False,
    nispace_data_dir: Union[str, Path] = None,
    overwrite: bool = False,
    check_file_hash: bool = True,
    verbose: bool = True,
) -> Path:
    """Fetch the overview plot PNG for a parcellation or reference dataset.

    Parameters
    ----------
    name : str
        Parcellation name (e.g. ``"Yan200"``) or reference dataset name
        (e.g. ``"pet"``).
    kind : {"parcellation", "reference"}, optional
        Whether to look up *name* in the parcellation or reference library.
        If ``None`` (default), both libraries are searched automatically.
    display : bool
        If ``True``, display the image inline (Jupyter / IPython).
    nispace_data_dir : str or Path, optional
        Override the NiSpace data directory (deprecated; use NISPACE_DATA_DIR).
    overwrite : bool
        Re-download even if the local file already exists.
    check_file_hash : bool
        Verify SHA-256 against the known hash after download.
    verbose : bool
        Enable verbose logging.

    Returns
    -------
    Path
        Local path to the downloaded PNG file.
    """
    verbose = set_log(lgr, verbose)
    nispace_data_dir = _resolve_nispace_data_dir(nispace_data_dir)

    def _resolve(lib, name, is_parc):
        entry = lib.get(name, {})
        if is_parc and "alias" in entry:
            entry = lib.get(entry["alias"], {})
        return entry

    if kind is not None:
        kind = kind.lower()
        if kind not in ("parcellation", "reference"):
            raise ValueError(f"'kind' must be 'parcellation' or 'reference', not '{kind}'.")
        libs = [(kind, parcellation_lib if kind == "parcellation" else reference_lib)]
    else:
        libs = [("parcellation", parcellation_lib), ("reference", reference_lib)]

    entry = {}
    resolved_kind = None
    for k, lib in libs:
        e = _resolve(lib, name, k == "parcellation")
        if e:
            entry = e
            resolved_kind = k
            break

    if not entry:
        parc_with_plots = sorted(
            k for k, v in parcellation_lib.items()
            if isinstance(v, dict) and "plot" in v
        )
        ref_with_plots = sorted(
            k for k, v in reference_lib.items()
            if isinstance(v, dict) and "plot" in v
        )
        raise ValueError(
            f"No plot found for '{name}'. "
            f"Parcellations with plots: {parc_with_plots}. "
            f"Reference datasets with plots: {ref_with_plots}."
        )

    plot_info = entry.get("plot")
    if plot_info is None:
        raise ValueError(f"No plot registered for {resolved_kind} '{name}'.")

    local_path = Path(nispace_data_dir) / plot_info["remote"]
    get_file_kwargs = dict(overwrite=overwrite, hash_check=check_file_hash)
    path = Path(get_file(local_path, **plot_info, **get_file_kwargs))

    if display:
        try:
            from IPython.display import display as _ipy_display, Image
            _ipy_display(Image(str(path)))
        except ImportError:
            lgr.warning("IPython not available; cannot display image inline.")

    return path
