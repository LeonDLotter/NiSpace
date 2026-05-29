# -*- coding: utf-8 -*-
"""
Functionality for parcellating data, copied from neuromaps 0.0.4 and 
adapted for convenient use in NiSpace
"""

import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import new_img_like, math_img
import numpy as np
import pandas as pd

from neuromaps.datasets import DENSITIES
from neuromaps.images import construct_shape_gii, load_gifti, load_nifti, load_data
from neuromaps.resampling import resample_images
from neuromaps.transforms import _check_hemi, _estimate_density
from neuromaps.nulls.spins import parcels_to_vertices

# monkey patch to neuromaps ALIAS
ALIAS = dict(
    fslr='fsLR', fsavg='fsaverage', 
    mni152='MNI152', mni='MNI152', 
    mni152nlin6asym='MNI152', mni152nlin2009asym='MNI152',
    MNI152NLin6Asym='MNI152', MNI152NLin2009Asym='MNI152',
    FSLR='fsLR', CIVET='civet'
)

from nispace.utils.utils import get_background_value, vol_to_vect_arr, _resolve_bg_array

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
    Copied from neuromaps 0.0.4 and adapted for convenient use in NiSpace.

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
    labels : list, optional
        List of labels corresponding to indices in parcellation file
    """

    def __init__(self, parcellation, space, resampling_target='data', hemi=None):
        self.parcellation = parcellation
        self.space = ALIAS.get(space, space)
        self.resampling_target = resampling_target
        self.hemi = hemi
        self._volumetric = self.space == 'MNI152'

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
        """ Prepare parcellation for data extraction
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

    def transform(self, data, space, ignore_background_data=True,
                  background_value=["auto", 0.0], hemi=None,
                  fill_dropped=True, background_parcels_to_nan=False,
                  min_num_valid_datapoints=None, min_fraction_valid_datapoints=None):
        """
        Applies parcellation to `data` in `space`

        Parameters
        ----------
        data : str or os.PathLike or Nifti1Image or GiftiImage or tuple
            Data to parcellate
        space : str
            The space in which `data` is defined
        hemi : {'L', 'R'}, optional
            If provided `data` represents only one hemisphere of a surface
            dataset then this specifies which hemisphere. If not specified it
            is assumed that `data` is (L, R) hemisphere. Ignored if `space` is
            'MNI152'. Default: None
        ignore_background_data : bool
            Whether to exclude background voxels/vertices from parcel-mean
            computation. When True, values specified by `background_value` are
            masked before averaging. Default: True
        background_value : float, list, set, array, or 'auto'
            Value(s) to treat as background when `ignore_background_data=True`.
            Accepts a scalar, or any collection of scalars and/or the sentinel
            string ``'auto'``/``None``:

            - float (e.g. ``0.0``): exclude that specific value
            - ``'auto'`` or ``None``: auto-detect from border voxels (volumetric)
              or medial wall median (surface)
            - list/set/array: any combination of the above

            Default: ``['auto', 0.0]`` (excludes detected background and zeros)
        background_parcels_to_nan : bool
            Whether to set parcels whose mean equals the single resolved
            background value to NaN after aggregation. Only meaningful when
            `ignore_background_data=False` and `background_value` resolves to
            exactly one scalar; otherwise redundant (all-background parcels
            already return NaN from empty-mean aggregation). Default: False

        Returns
        -------
        parcellated : np.ndarray
            Parcellated `data`
        """

        self._check_fitted()

        space = ALIAS.get(space, space)
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
        
        data, parc = resample_images(data, self.parcellation,
                                     space, self.space, hemi=hemi,
                                     resampling=self._resampling,
                                     method=resampling_method)
        self._parc = parc
        self._parc_arr = load_data(parc)
        self._parc_idc = np.trim_zeros(np.unique(self._parc_arr))
        self._parc_idc_dropped = []
        self._parc_idc_bg = []
        self._parc_idc_excl = []
        
        # normalise background_value spec to a plain list once, used by both branches
        bg_spec = list(background_value) if isinstance(background_value, (list, tuple, np.ndarray, set)) \
                  else [background_value]
        needs_auto = ignore_background_data and any(v in (None, "auto") for v in bg_spec)

        if ((self.resampling_target == 'data'
             and space.lower() == 'mni152')
                or (self.resampling_target == 'parcellation'
                    and self._volumetric)):
            data = nib.concat_images([nib.squeeze_image(data)])
            darr = data.get_fdata()
            auto_value = get_background_value(data) if needs_auto else np.nan
            bg_arr = _resolve_bg_array(bg_spec, auto_value) if ignore_background_data \
                     else np.array([], dtype=np.float64)
            parcellated = vol_to_vect_arr(darr, self._parc_arr, self._parc_idc, bg_arr)

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
                _mw_L, _mw_R = fetch_template(mask_space, desc="medial", res=density, check_file_hash=False, verbose=False)
                atlas_medialwall = _mw_L if hemi == 'L' else _mw_R if hemi == 'R' else (_mw_L, _mw_R)
                nomedialwall = load_data(atlas_medialwall)
                auto_value = np.median(darr[nomedialwall == 0])
            else:
                auto_value = np.nan
            parc_arr = _gifti_to_array(parc)
            bg_arr = _resolve_bg_array(bg_spec, auto_value) if ignore_background_data \
                     else np.array([], dtype=np.float64)
            parcellated = vol_to_vect_arr(darr, parc_arr, self._parc_idc, bg_arr)

        # drop parcels whose mean equals the background value — only for the scalar case
        if background_parcels_to_nan and len(bg_arr) == 1:
            bg_idc = parcellated == bg_arr[0]
            parcellated[bg_idc] = np.nan
            self._parc_idc_bg = list(self.parcellation_idc[bg_idc])
            
        # drop parcels for which there are too few non-background voxels/vertices (= datapoints)
        # given as a minimum number of datapoints and/or a minimum fraction of datapoints
        # this option is computationally expensive!
        # TODO: improve efficiency, get rid of loop?
        if ((min_num_valid_datapoints or min_fraction_valid_datapoints) 
                and background_value is not None):
            
            # load data
            parc_array = load_data(parc)
            data_arr = load_data(data)
            not_bg = ~np.isnan(data_arr.astype(float))
            for _v in bg_arr:
                not_bg &= (data_arr != _v)
            parc_array_nogb = parc_array[not_bg]
            
            parc_n_datapoints = np.zeros(len(self.parcellation_idc), dtype=int)
            data_n_nobg = parc_n_datapoints.copy()
            for i, idx in enumerate(self.parcellation_idc):
                # number of datapoints per original parcel in resampled parcellation 
                parc_n_datapoints[i] = (parc_array==idx).sum()
                # number of non-bg datapoints in data per parcel
                data_n_nobg[i] = (parc_array_nogb==idx).sum()

            # exclude parcels based on criteria
            excl_filter = np.full_like(parc_n_datapoints, False, dtype=bool)
            # criterion: minimum number of valid datapoints per parcel
            if min_num_valid_datapoints:
                excl_filter = excl_filter | (data_n_nobg < min_num_valid_datapoints)
            # criterion: minimum fraction of non-bg datapoints in data relative to parc per parcel
            if min_fraction_valid_datapoints:
                data_frac_nobg = np.divide(
                    data_n_nobg, parc_n_datapoints, 
                    out=np.zeros_like(data_n_nobg, dtype=np.float64), 
                    where=parc_n_datapoints!=0
                )
                excl_filter = excl_filter | (data_frac_nobg < min_fraction_valid_datapoints)
            
            # apply
            parcellated[excl_filter] = np.nan
            self._parc_idc_excl = list(self.parcellation_idc[excl_filter])
        
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

    def fit_transform(self, data, space, ignore_background_data=True,
                      background_value=["auto", 0.0], hemi=None,
                      fill_dropped=True, background_parcels_to_nan=False,
                      min_num_valid_datapoints=None, min_fraction_valid_datapoints=None):
                      
        """ Prepare and perform parcellation of `data`
        """
        return self.fit().transform(data, space, ignore_background_data,
                                    background_value, hemi, fill_dropped,
                                    background_parcels_to_nan,
                                    min_num_valid_datapoints,
                                    min_fraction_valid_datapoints)

    def _check_fitted(self):
        if not hasattr(self, '_fit'):
            raise ValueError(f'It seems that {self.__class__.__name__} has '
                             'not been fit. You must call `.fit()` before '
                             'calling `.transform()`')
