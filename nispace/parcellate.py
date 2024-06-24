# -*- coding: utf-8 -*-
"""
Functionality for parcellating data, copied from neuromaps 0.0.4 and 
adapted for convenient use in NiSpace
"""

import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import new_img_like, math_img
#from nilearn.masking import compute_background_mask
import numpy as np
import pandas as pd

from neuromaps.datasets import ALIAS, DENSITIES, fetch_atlas
from neuromaps.images import construct_shape_gii, load_gifti, load_nifti, load_data
from neuromaps.resampling import resample_images
from neuromaps.transforms import _check_hemi, _estimate_density
from neuromaps.nulls.spins import vertices_to_parcels, parcels_to_vertices

from nispace.utils import get_background_value


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
    Class for parcellating arbitrary volumetric / surface data
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
                  background_value=None, hemi=None, 
                  fill_dropped=True, background_parcels_to_nan=True,
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
        ignore_background_data: bool
            Specifies whether the background data values should be ignored
            when computing the average `data` within each parcel. If set to
            True and `background_value` is set to None, the background_value is
            estimated from the data: if there are NaNs in the data, the
            background value is set to NaN. Otherwise, it is estimated as
            the median of the values on the border of the images for
            volumetric images or as the median of the values within the medial
            wall for surface images. The background value can also be set
            manually using the `background_value` parameter. Default: False
        background_value: float
            Specifies the background value to ignore when computing the
            averages and when `ignore_background_data` is True.
            Default: None

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

        if hemi is not None and hemi not in self.hemi:
            raise ValueError('Cannot parcellate data from {hemi} hemisphere '
                             'when parcellation was provided for incompatible '
                             'hemisphere: {self.hemi}')

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
        self._parc_idc = np.trim_zeros(np.unique(load_data(self.parcellation)))
        self._parc_idc_dropped = []
        self._parc_idc_bg = []
        self._parc_idc_excl = []
        
        if ((self.resampling_target == 'data'
             and space.lower() == 'mni152')
                or (self.resampling_target == 'parcellation'
                    and self._volumetric)):
            data = nib.concat_images([nib.squeeze_image(data)])
            if ignore_background_data:
                if background_value is None:
                    background_value = get_background_value(data)
                    mask_img = math_img(f"data != {background_value}", data=data)
                else:
                    mask_img = new_img_like(data, data.get_fdata() != background_value)
            else:
                mask_img = None
                
            # parcellate
            masker = NiftiLabelsMasker(
                parc, mask_img=mask_img, resampling_target=None
            )
            parcellated = masker.fit_transform(data).squeeze()
            
            # take care of parcels dropped by nilearn
            # we use an intermediate pandas array because indexing is simple here
            if fill_dropped:
                # indices 
                idc_orig = self.parcellation_idc      
                idc_resampled = np.array(masker.labels_)
                # new array with original indices
                parcellated_series = pd.Series(index=idc_orig)
                # write data into original positions, leaving dropped parcels with nan
                parcellated_series.loc[idc_resampled] = parcellated
                # replace np array
                parcellated = np.array(parcellated_series)
                # save stuff
                self._parc_idc = idc_resampled
                self._parc_idc_dropped = list( set(idc_orig) ^ set(idc_resampled) )  
            
        else:
            if not self._volumetric:
                for n, _ in enumerate(parc):
                    parc[n].labeltable.labels = \
                        self.parcellation[n].labeltable.labels
            darr = _gifti_to_array(data)
            if ignore_background_data and background_value is None:
                density, = _estimate_density((data,), hemi=hemi)
                if self.resampling_target in ('data', None):
                    mask_space = space
                elif self.resampling_target == 'parcellation':
                    mask_space = self.space
                nomedialwall = load_data(fetch_atlas(mask_space, density)['medial'])
                background_value = np.median(darr[nomedialwall == 0])
            parcellated = vertices_to_parcels(darr, parc, background=background_value)

        # fill parcels with background intensity with nan, works only if background_value exists
        if background_parcels_to_nan and background_value is not None:
            bg_idc = parcellated == background_value
            parcellated[bg_idc] = np.nan
            self._parc_idc_bg = list(self.parcellation_idc[bg_idc])
            
        # drop parcels for which there are too few non-background voxels/vertices (= datapoints)
        # given as a minimum number of datapoints and/or a minimum fraction of datapoints
        # this option is computationally expensive!
        # TODO: improve efficiency, get rid of list comprehension?
        if ((min_num_valid_datapoints or min_fraction_valid_datapoints) 
                and background_value is not None):
            
            # load data
            parc_array = load_data(parc)
            parc_array_nogb = parc_array[load_data(data) != background_value]
            
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

    def fit_transform(self, data, space, ignore_background_data=False,
                      background_value=None, hemi=None, fill_dropped=True):
        """ Prepare and perform parcellation of `data`
        """
        return self.fit().transform(data, space, ignore_background_data,
                                    background_value, hemi, fill_dropped)

    def _check_fitted(self):
        if not hasattr(self, '_fit'):
            raise ValueError(f'It seems that {self.__class__.__name__} has '
                             'not been fit. You must call `.fit()` before '
                             'calling `.transform()`')
