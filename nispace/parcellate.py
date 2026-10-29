# -*- coding: utf-8 -*-
"""
Functionality for parcellating data, copied from neuromaps 0.0.4 and
adapted for convenient use in NiSpace
"""

import warnings
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import new_img_like, math_img
import numpy as np
import pandas as pd

from nilearn import image as nimage
from neuromaps.datasets import DENSITIES
from neuromaps.images import construct_shape_gii, load_gifti, load_nifti, load_data
from neuromaps.transforms import _check_hemi, _estimate_density
from neuromaps.nulls.spins import parcels_to_vertices

import logging
lgr = logging.getLogger(__name__)

# ==================================================================================================
# DEPRECATION MESSAGE STRINGS
# ==================================================================================================

_DEPR_IGNORE_BACKGROUND_DATA = (
    "'ignore_background_data' is deprecated and will be removed in the first non-dev "
    "release. Use 'background_value' instead: pass background_value=False to disable "
    "background exclusion (equivalent to ignore_background_data=False), or a scalar/"
    "list/'auto' to enable it (equivalent to ignore_background_data=True)."
)

# monkey patch to neuromaps ALIAS
# TODO: ALIAS is only still needed for DENSITIES validation (line ~91) and resample_images calls.
# _volumetric detection and transform branching already use 'mni' in space.lower() instead.
# Consider replacing the remaining ALIAS uses with explicit 'mni'/'fsaverage'/'fslr' checks
# and removing this dict entirely.
ALIAS = dict(
    fslr='fsLR', fsavg='fsaverage', 
    mni152='MNI152', mni='MNI152', 
    mni152nlin6asym='MNI152', mni152nlin2009asym='MNI152', mni152nlin2009casym='MNI152',
    MNI152NLin6Asym='MNI152', MNI152NLin2009Asym='MNI152', MNI152NLin2009cAsym='MNI152',
    FSLR='fsLR', CIVET='civet'
)

from nispace.utils.utils import (get_background_value, vol_to_vect_arr, vol_to_vect_arr_stats, _resolve_bg_array,
                                 _quiet)
from nispace.core.constants import _SPACE_DEFAULT_VOL

_INFO_SEEN = set()


def _info_once(message, level="info"):
    """Log `message` once per Python session (e.g. per-map notes in parcellation loops)."""
    if message not in _INFO_SEEN:
        _INFO_SEEN.add(message)
        getattr(lgr, level)(message)


def _mni_space_or_none(space):
    """Canonical MNI space name for `space`, or None if it does not identify one (e.g. "MNI152").

    Names other than the generic "MNI152" / "MNI" that do not resolve (typos, ambiguous names) are
    warned about once, since they silently disable MNI-to-MNI warps.
    """
    from .transform import _resolve_space, _GENERIC_MNI_NAMES
    mni = _resolve_space(space, strict=False)
    if mni is None and space.lower() not in _GENERIC_MNI_NAMES:
        _info_once(f"'{space}' does not identify a single supported MNI space and is treated as unspecified "
                   f"MNI space: no MNI-to-MNI warp; {_SPACE_DEFAULT_VOL} is assumed for projections to the "
                   "surface. Use a canonical name (see nispace.transform.mni_to_mni) to avoid this.",
                   level="warning")
    return mni


def _check_not_civet(space):
    if isinstance(space, str) and space.lower() == "civet":
        from .transform import _CIVET_MSG
        raise ValueError(_CIVET_MSG)


def _is_mni(space):
    """True for MNI volume space names (containing "mni", or a canonical MNI space such as CAT12IXI555)."""
    from .transform import _is_mni_space_name
    return _is_mni_space_name(space)


def _resample_to_img(img, target, method):
    """nilearn resample_to_img with a fresh header (dtype follows the array, no on-disk scaling).

    nilearn 0.12 warns about the copy_header default changing in 0.13 on every call with
    copy_header=False, also when nothing is resampled; we keep False on purpose.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="From release 0.13.0 onwards", category=FutureWarning)
        return nimage.resample_to_img(img, target, interpolation=method, force_resample=False,
                                      copy_header=False)


def _resample_images(src, trg, src_space, trg_space, hemi=None, resampling="transform_to_src",
                     method="linear", src_mni=None, trg_mni=None):
    """Bring data (`src`) and parcellation (`trg`) into a common space.

    Same strategies as neuromaps' ``resample_images`` (``"transform_to_src"``: `trg` is transformed to
    the space / density of `src`; ``"transform_to_trg"``: the reverse), but surface transforms run on
    NiSpace's hash-checked templates via :mod:`nispace.transform` (no Workbench, no neuromaps cache).
    MNI ↔ MNI is plain grid resampling (nilearn); warps between two different specific MNI spaces are
    done beforehand by :meth:`Parcellater.transform`. Volume ↔ surface always projects the volume onto the
    surface from its actual MNI space (`src_mni` / `trg_mni`, canonical names; registration fusion composed
    with the EasyReg field). If that space is unknown (generic "MNI152"), MNI152NLin6Asym is assumed, as
    neuromaps does.

    Returns
    -------
    (src, trg)
        Nifti1Image for volumes, tuple of GiftiImage for surfaces.
    """
    from .transform import mni_to_surf, surf_to_surf

    def _surf(imgs):
        return tuple(load_gifti(img) for img, _ in _check_hemi(imgs, hemi))

    def _den(imgs):
        return _estimate_density((imgs,), hemi)[0]

    src_vol, trg_vol = src_space == "MNI152", trg_space == "MNI152"
    if src_vol and trg_vol:
        src, trg = load_nifti(src), load_nifti(trg)
        if resampling == "transform_to_src":
            trg = _resample_to_img(trg, src, method)
        else:
            src = _resample_to_img(src, trg, method)
        return src, trg
    if src_vol or trg_vol:
        if (resampling == "transform_to_src" and src_vol) or (resampling == "transform_to_trg" and trg_vol):
            raise ValueError("Cannot transform surface data into MNI152 space; surface-to-volume "
                             "transforms are not implemented yet.")
        vol, surf, surf_space = (src, trg, trg_space) if src_vol else (trg, src, src_space)
        mni_from = src_mni if src_vol else trg_mni
        if mni_from is None:
            _info_once(f"MNI space of the volumetric {'data' if src_vol else 'parcellation'} not specified; "
                       f"assuming {_SPACE_DEFAULT_VOL} for the projection to {surf_space}.")
            mni_from = _SPACE_DEFAULT_VOL
        surf = _surf(surf)
        hemis = None if len(surf) == 2 else hemi
        with _quiet():  # internal call: don't change the caller's log level
            vol = mni_to_surf(load_nifti(vol), mni_from, surf_space, den_to=_den(surf), hemi=hemis,
                              method=method, verbose=False)
        vol = vol if isinstance(vol, tuple) else (vol,)
        return (vol, surf) if src_vol else (surf, vol)
    hemis = None if hemi is None else hemi
    with _quiet():
        if resampling == "transform_to_src":
            trg = surf_to_surf(_surf(trg), trg_space, src_space, den_to=_den(src), hemi=hemis, method=method,
                               verbose=False)
        else:
            src = surf_to_surf(_surf(src), src_space, trg_space, den_to=_den(trg), hemi=hemis, method=method,
                               verbose=False)
    src = src if isinstance(src, tuple) else (src,)
    trg = trg if isinstance(trg, tuple) else (trg,)
    return _surf(src), _surf(trg)

def _gifti_to_array(gifti):
    """ Converts tuple of `gifti` to numpy array
    """
    return np.hstack([load_gifti(img).agg_data() for img in gifti])

def _array_to_gifti(data):
    """ Converts numpy `array` to tuple of gifti images
    """
    return tuple(construct_shape_gii(arr) for arr in np.split(data, 2))


class Parcellater():
    """
    Class for parcellating arbitrary volumetric / surface data.
    Copied from neuromaps :cite:`markello2022` 0.0.4 and adapted for convenient use in NiSpace.

    Parameters
    ----------
    parcellation : str or os.PathLike or Nifti1Image or GiftiImage or tuple
        Parcellation image or surfaces, where each region is identified by a
        unique integer ID. All regions with an ID of 0 are ignored.
    space : str
        The space in which `parcellation` is defined
    resampling_target : {'data', 'parcellation', None}, optional
        Gives which image gives the final shape/size. For example, if
        `resampling_target` is 'data', the `parcellation` is resampled to the
        space + resolution of the data, if needed. If it is 'parcellation' then
        any data provided to `.fit()` are transformed to the space + resolution
        of `parcellation`. Providing None means no resampling; if spaces +
        resolutions of the `parcellation` and data provided to `.fit()` do not
        match a ValueError is raised. Default: 'data'
    hemi : {'L', 'R'}, optional
        If provided `parcellation` represents only one hemisphere of a surface
        atlas then this specifies which hemisphere. If not specified it is
        assumed that `parcellation` is (L, R) hemisphere. Ignored if `space` is
        'MNI152'. Default: None

    References
    ----------
    :cite:`markello2022`.
    """

    def __init__(self, parcellation, space, resampling_target='data', hemi=None):
        """
        Construct a Parcellater from a parcellation image/surfaces.

        See the class docstring for parameter details.

        Raises
        ------
        ValueError
            If `resampling_target` is not one of {'data', 'parcellation', None},
            or if `space` is not a space known to neuromaps (see
            ``neuromaps.datasets.DENSITIES``).
        """
        self.parcellation = parcellation
        self._volumetric = _is_mni(space)
        # "MNI152" = any MNI volume space for the resampling logic; the specific space (canonical name, or
        # None if not identifiable, e.g. generic "MNI152") is kept for MNI-to-MNI warps and
        # volume-to-surface projections
        self._mni_space = _mni_space_or_none(space) if self._volumetric else None
        _check_not_civet(space)
        self.space = 'MNI152' if self._volumetric else ALIAS.get(space, space)
        self.resampling_target = resampling_target
        self.hemi = hemi
        self._mni_transformers = {}   # fitted MNI-to-MNI SpaceTransformers, reused across maps
        self._warped_parc = {}        # parcellation warped into a data space, reused across maps

        if self.resampling_target == 'parcellation':
            self._resampling = 'transform_to_trg'
        else:
            self._resampling = 'transform_to_src'

        if not self._volumetric:
            self.parcellation, self.hemi = zip(
                *_check_hemi(self.parcellation, self.hemi)
            )

        if self.resampling_target not in ('parcellation', 'data', None):
            raise ValueError('Invalid value for `resampling_target`: '
                             f'{resampling_target}')

        if self.space not in DENSITIES:
            raise ValueError(f'Invalid value for `space`: {space}')

    def fit(self):
        """
        Load and validate the parcellation, preparing it for data extraction.

        Loads `self.parcellation` into memory (as a `Nifti1Image` or tuple of
        `GiftiImage`, depending on `space`) and populates
        `self.parcellation_idc` with the sorted, non-zero parcel IDs found in
        it. Must be called before `.transform()`; `.fit_transform()` calls it
        automatically.

        Returns
        -------
        self : Parcellater
            The fitted instance, to allow chaining (e.g. `self.fit().transform(...)`).
        """

        # load parcellation
        if not self._volumetric:
            self.parcellation = tuple(
                load_gifti(img) for img in self.parcellation
            )
        else:
            self.parcellation = load_nifti(self.parcellation)
            
        # get parcel idc
        self.parcellation_idc = np.trim_zeros(np.unique(load_data(self.parcellation)))
            
        self._fit = True
        return self

    def transform(self, data, space, background_value="auto", hemi=None,
                  fill_dropped=True, report_background_parcels=False,
                  min_num_valid_datapoints=None, min_fraction_valid_datapoints=None,
                  ignore_background_data=None):
        """
        Applies parcellation to `data` in `space`

        Parameters
        ----------
        data : str or os.PathLike or Nifti1Image or GiftiImage or tuple
            Data to parcellate
        space : str
            The space in which `data` is defined
        background_value : float, list, set, array, 'auto', or False
            Value(s) to treat as background, or ``False`` to disable
            background exclusion entirely. When disabled, background/zero is
            treated as real data -- never masked, never triggers the
            empty-mean-to-NaN path (NaN itself is still always excluded,
            regardless of this parameter). Accepts:

            - ``'auto'`` (default): auto-detect from border voxels
              (volumetric) or medial wall median (surface), combined with
              exact ``0.0`` -- equivalent to ``['auto', 0.0]``.
            - float (e.g. ``0.0``): exclude that specific value only.
            - list/set/array: any combination of floats and the
              ``'auto'``/``None`` sentinel.
            - ``False``: disable background exclusion entirely.
        hemi : {'L', 'R'}, optional
            If provided `data` represents only one hemisphere of a surface
            dataset then this specifies which hemisphere. If not specified it
            is assumed that `data` is (L, R) hemisphere. Ignored if `space` is
            'MNI152'. Default: None
        fill_dropped : bool
            Whether to expand the returned array to the full original parcel
            set (`self.parcellation_idc`, from `.fit()`), NaN-filling any
            parcel that vanished entirely during resampling to `data`'s grid
            (`self._parc_idc_dropped`). If False, the returned array only
            covers parcels present in the resampled parcellation, which may
            be shorter than `self.parcellation_idc`. Default: True
        report_background_parcels : bool
            Whether to explicitly record parcels whose raw (pre-exclusion)
            data was entirely background -- every non-NaN raw voxel/vertex in
            the parcel matches `background_value`. Such parcels are already
            NaN via empty-mean aggregation regardless of this flag, so
            enabling it does not change the returned values -- it only
            additionally records the affected parcels (`self._parc_idc_bg`,
            surfaced in `parcellate_data()`'s logging), separately from
            parcels dropped during resampling or excluded via the
            `min_*_valid_datapoints` options. Always a no-op when
            `background_value=False`: with background exclusion disabled,
            `background_value` may label real, meaningful data (e.g.
            binary/cluster-coverage maps, where an all-zero parcel is a
            genuine 0%-overlap result, not missing background) and must never
            be flagged here. Default: False
        min_num_valid_datapoints : int, optional
            Minimum number of valid (non-background, non-NaN) datapoints
            required per parcel; parcels below this are set to NaN and
            recorded in `self._parc_idc_excl`. Default: None
        min_fraction_valid_datapoints : float, optional
            Minimum fraction of valid (non-background, non-NaN) datapoints,
            relative to the parcel's total voxel/vertex count in the
            resampled parcellation, required per parcel; parcels below this
            are set to NaN and recorded in `self._parc_idc_excl`.
            Default: None
        ignore_background_data : bool, optional
            Deprecated. Use `background_value` instead -- pass
            ``background_value=False`` for what used to be
            ``ignore_background_data=False``. If explicitly passed, takes
            precedence over `background_value` and replicates the old
            two-independent-parameter behavior for the deprecation transition
            period. Default: None (not set)

        Returns
        -------
        parcellated : np.ndarray
            Parcellated `data`
        """

        self._check_fitted()

        data_mni = _mni_space_or_none(space) if _is_mni(space) else None
        _check_not_civet(space)
        space = 'MNI152' if _is_mni(space) else ALIAS.get(space, space)
        if (self.resampling_target == 'data' and space == 'MNI152'
                and not self._volumetric):
            raise ValueError('Cannot use resampling_target="data" when '
                             'provided parcellation is in surface space and '
                             'provided data are in MNI152 space.')
        elif (self.resampling_target == 'parcellation' and self._volumetric
                and space != 'MNI152'):
            raise ValueError('Cannot use resampling_target="parcellation" '
                             'when provided parcellation is in MNI152 space '
                             'and provided data are in surface space.')

        if hemi in [("L", "R"), ["L", "R"]]:
            hemi = None
        if hemi is not None and hemi not in self.hemi:
            raise ValueError(f'Cannot parcellate data from {hemi} hemisphere '
                             f'when parcellation was provided for incompatible '
                             f'hemisphere: {self.hemi}')

        if isinstance(data, np.ndarray):
            data = _array_to_gifti(data)
        if self.resampling_target in ('data', None):
            resampling_method = 'nearest'
        else:
            resampling_method = 'linear'
        
        # data and parcellation in two different specific MNI spaces: warp the side that moves
        # (resampling_target 'data': the parcellation, nearest; 'parcellation': the data, linear).
        # The generic "MNI152" on either side never triggers a warp.
        parcellation, parc_mni = self.parcellation, self._mni_space
        if (self._volumetric and space == 'MNI152' and data_mni is not None
                and parc_mni is not None and data_mni != parc_mni):
            data = load_nifti(data)
            if self.resampling_target == 'parcellation':
                data = self._mni_warp(data, data_mni, parc_mni, self.parcellation, resampling_method)
                data_mni = parc_mni
            else:
                parcellation = self._mni_warp(self.parcellation, parc_mni, data_mni, data,
                                              resampling_method, cache_result=True)
                parc_mni = data_mni

        data, parc = _resample_images(data, parcellation,
                                      space, self.space, hemi=hemi,
                                      resampling=self._resampling,
                                      method=resampling_method,
                                      src_mni=data_mni, trg_mni=parc_mni)
        self._parc = parc
        self._parc_arr = load_data(parc)
        self._parc_idc = np.trim_zeros(np.unique(self._parc_arr))
        self._parc_idc_dropped = []
        self._parc_idc_bg = []
        self._parc_idc_excl = []
        
        # TODO (first non-dev release): remove the legacy ignore_background_data path
        # resolve background_value (+ deprecated ignore_background_data) into
        # (is_disabled, bg_spec) -- see _DEPR_IGNORE_BACKGROUND_DATA for the
        # legacy path, which replicates the old two-independent-parameter
        # behavior exactly for the deprecation transition period.
        if ignore_background_data is not None:
            lgr.warning(_DEPR_IGNORE_BACKGROUND_DATA)
            bg_spec = list(background_value) if isinstance(background_value, (list, tuple, np.ndarray, set)) \
                      else [background_value]
            is_disabled = not ignore_background_data
        elif background_value is False:
            is_disabled = True
            bg_spec = []
        elif isinstance(background_value, (list, tuple, np.ndarray, set)):
            is_disabled = False
            bg_spec = list(background_value)
        elif background_value is None or background_value == "auto":
            is_disabled = False
            bg_spec = ["auto", 0.0]
        else:
            is_disabled = False
            bg_spec = [background_value]
        needs_auto = (not is_disabled) and any(v in (None, "auto") for v in bg_spec)

        if ((self.resampling_target == 'data'
             and 'mni' in space.lower())
                or (self.resampling_target == 'parcellation'
                    and self._volumetric)):
            data = nib.concat_images([nib.squeeze_image(data)])
            darr = data.get_fdata()
            auto_value = get_background_value(data) if needs_auto else np.nan
            bg_arr = _resolve_bg_array(bg_spec, auto_value) if not is_disabled \
                     else np.array([], dtype=np.float64)
            means, n_valid, n_total, all_background = vol_to_vect_arr_stats(
                darr, self._parc_arr, self.parcellation_idc, bg_arr)

        else:
            if not self._volumetric:
                for n, _ in enumerate(parc):
                    parc[n].labeltable.labels = \
                        self.parcellation[n].labeltable.labels
            darr = _gifti_to_array(data)
            if needs_auto:
                density, = _estimate_density((data,), hemi=hemi)
                mask_space = space if self.resampling_target in ('data', None) else self.space
                from .datasets import fetch_template
                _mw_L, _mw_R = fetch_template(mask_space, desc="medial", res=density, verbose=False)
                atlas_medialwall = _mw_L if hemi == 'L' else _mw_R if hemi == 'R' else (_mw_L, _mw_R)
                nomedialwall = load_data(atlas_medialwall)
                auto_value = np.median(darr[nomedialwall == 0])
            else:
                auto_value = np.nan
            parc_arr = _gifti_to_array(parc)
            bg_arr = _resolve_bg_array(bg_spec, auto_value) if not is_disabled \
                     else np.array([], dtype=np.float64)
            means, n_valid, n_total, all_background = vol_to_vect_arr_stats(
                darr, parc_arr, self.parcellation_idc, bg_arr)

        parcellated = means

        # detect parcels that vanished after resampling (n_total==0 -- zero
        # voxels/vertices carry this label in the resampled parcellation at
        # all) -- means already contains NaN there "for free", no separate
        # fill step needed. domain_idc tracks which label set parcellated/
        # n_valid/n_total are currently indexed by, for the blocks below.
        dropped_mask = (n_total == 0)
        self._parc_idc_dropped = list(self.parcellation_idc[dropped_mask])
        if not fill_dropped:
            keep = ~dropped_mask
            parcellated = parcellated[keep]
            n_valid = n_valid[keep]
            n_total = n_total[keep]
            domain_idc = self.parcellation_idc[keep]
        else:
            domain_idc = self.parcellation_idc

        # record parcels whose raw (pre-exclusion) data was entirely background --
        # i.e. every non-NaN raw voxel/vertex in the parcel is a background value.
        # These parcels are already NaN via the empty-valid-mean path regardless
        # of this flag; enabling it only additionally records them in
        # self._parc_idc_bg (surfaced in parcellate_data()'s logging),
        # distinguishing "NaN because background" from "NaN because all raw data
        # was itself NaN" (missing data, not reported here) or "NaN because the
        # parcel vanished during resampling" (self._parc_idc_dropped, above --
        # a dropped parcel always has all_background=False by construction, so
        # this never double-reports a dropped parcel as background).
        # Deliberately gated to background exclusion being enabled: when
        # background_value=False, background_value may label real, meaningful
        # data (e.g. binary_y cluster-coverage maps, where an all-zero parcel
        # is a genuine 0%-overlap result, not missing background) and must
        # never be flagged here.
        if report_background_parcels and not is_disabled and len(bg_arr) > 0:
            self._parc_idc_bg = list(self.parcellation_idc[all_background])

        # drop parcels for which there are too few non-background voxels/vertices (= datapoints)
        # given as a minimum number of datapoints and/or a minimum fraction of datapoints
        if min_num_valid_datapoints or min_fraction_valid_datapoints:
            excl_filter = np.zeros(len(domain_idc), dtype=bool)
            # criterion: minimum number of valid datapoints per parcel
            if min_num_valid_datapoints:
                excl_filter = excl_filter | (n_valid < min_num_valid_datapoints)
            # criterion: minimum fraction of non-bg datapoints in data relative to parc per parcel
            if min_fraction_valid_datapoints:
                frac_valid = np.divide(
                    n_valid, n_total,
                    out=np.zeros(len(domain_idc), dtype=np.float64),
                    where=n_total != 0
                )
                excl_filter = excl_filter | (frac_valid < min_fraction_valid_datapoints)

            # apply
            parcellated[excl_filter] = np.nan
            self._parc_idc_excl = list(domain_idc[excl_filter])

        return parcellated

    def inverse_transform(self, data):
        """
        Project `data` to space + density of parcellation

        Parameters
        ----------
        data : array_like
            Parcellated data to be projected to the space of parcellation

        Returns
        -------
        data : Nifti1Image or tuple-of-nib.GiftiImage
            Provided `data` in space + resolution of parcellation
        """

        if not self._volumetric:
            verts = parcels_to_vertices(data, self.parcellation)
            img = _array_to_gifti(verts)
        else:
            data = np.atleast_2d(data)
            img = NiftiLabelsMasker(self.parcellation).fit() \
                                                      .inverse_transform(data)
        return img

    def fit_transform(self, data, space, background_value="auto", hemi=None,
                      fill_dropped=True, report_background_parcels=False,
                      min_num_valid_datapoints=None, min_fraction_valid_datapoints=None,
                      ignore_background_data=None):

        """
        Call `.fit()` followed by `.transform(data, space, ...)` in one step.

        Convenience wrapper; see `.fit()` and `.transform()` for details on
        what each step does. All parameters are forwarded to `.transform()`.

        Parameters
        ----------
        data : str or os.PathLike or Nifti1Image or GiftiImage or tuple
            Data to parcellate. See `.transform()`.
        space : str
            The space in which `data` is defined. See `.transform()`.
        background_value : float, list, set, array, 'auto', or False
            See `.transform()`. Default: ``'auto'``
        hemi : {'L', 'R'}, optional
            See `.transform()`. Default: None
        fill_dropped : bool
            See `.transform()`. Default: True
        report_background_parcels : bool
            See `.transform()`. Default: False
        min_num_valid_datapoints : int, optional
            See `.transform()`. Default: None
        min_fraction_valid_datapoints : float, optional
            See `.transform()`. Default: None
        ignore_background_data : bool, optional
            Deprecated. See `.transform()`. Default: None (not set)

        Returns
        -------
        parcellated : np.ndarray
            Parcellated `data`. See `.transform()`.
        """
        return self.fit().transform(data, space,
                                    background_value=background_value, hemi=hemi,
                                    fill_dropped=fill_dropped,
                                    report_background_parcels=report_background_parcels,
                                    min_num_valid_datapoints=min_num_valid_datapoints,
                                    min_fraction_valid_datapoints=min_fraction_valid_datapoints,
                                    ignore_background_data=ignore_background_data)

    def _mni_warp(self, img, mni_from, mni_to, reference, method, cache_result=False):
        """Warp volume `img` from `mni_from` to `mni_to` (EasyReg, :mod:`nispace.transform`) at the template
        resolution closest to `reference`'s voxel size. Fitted transforms are reused across maps; with
        `cache_result` (the parcellation), the warped image is too."""
        from .transform import SpaceTransformer, _AVAILABLE_RES
        vox = float(np.min(np.abs(load_nifti(reference).header.get_zooms()[:3])))
        res = min(_AVAILABLE_RES, key=lambda r: abs(r - vox))
        order = 0 if method == "nearest" else 1
        key = (mni_from, mni_to, res, order)
        if cache_result and key in self._warped_parc:
            return self._warped_parc[key]
        if key not in self._mni_transformers:
            lgr.info(f"{'Parcellation' if cache_result else 'Data'} in {mni_from}, "
                     f"{'data' if cache_result else 'parcellation'} in {mni_to}: warping the "
                     f"{'parcellation (nearest)' if cache_result else 'data (linear)'} at {res}mm with EasyReg.")
            with _quiet():  # internal call: don't change the caller's log level
                self._mni_transformers[key] = SpaceTransformer(mni_from, mni_to, res=res, method=order,
                                                               verbose=False).fit()
        with _quiet():
            out = self._mni_transformers[key].transform(load_nifti(img))
        if cache_result:
            self._warped_parc[key] = out
        return out

    def _check_fitted(self):
        if not hasattr(self, '_fit'):
            raise ValueError(f'It seems that {self.__class__.__name__} has '
                             'not been fit. You must call `.fit()` before '
                             'calling `.transform()`')
