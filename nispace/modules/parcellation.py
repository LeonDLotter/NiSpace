
import nibabel as nib
import numpy as np
import pandas as pd

from neuromaps.images import load_data

from .. import lgr
from ..nulls import _img_space_for_neuromaps, _img_density_for_neuromaps, find_parcel_hemispheres, get_distance_matrix
from ..io import load_distmat, load_img, load_labels

class Parcellation():
    """
    """
    
    def __init__(self, parcellation, space=None, labels=None, resolution=None, hemi=None, 
                 symmetric=False, left2right_mapping=None,
                 labels_lh=None, labels_rh=None, labels_img_lh=None, labels_img_rh=None,
                 idc_lh=None, idc_rh=None, dist_mat=None, 
                 ):
        
        self._source = parcellation
        self._space = space
        self._hemi = hemi
        self._resolution = resolution
        self._labels = labels
        self._labels_byhemi = {"L": labels_lh, "R": labels_rh}
        self._labels_img_byhemi = {"L": labels_img_lh, "R": labels_img_rh}
        self._symmetric = symmetric
        self._idc_byhemi = {"L": idc_lh, "R": idc_rh}
        self._dist_mat = dist_mat
        self._l2rmap = left2right_mapping
        
    def fit(self):
        
        self._image_obj = load_img(self._source)
        self._data = load_data(self._image_obj).astype(int)
        self._labels_img = np.trim_zeros(np.unique(self._data))
        self._space = self._space if self._space is not None else _img_space_for_neuromaps(self._image_obj)
        self._is_surface = isinstance(self._image_obj, (nib.GiftiImage, tuple)) 
        self._is_unilateral_surface = isinstance(self._image_obj, nib.GiftiImage) 
        self._labels = np.array(load_labels(self._labels)) if self._labels is not None else self._labels_img
        self._resolution = self._resolution if self._resolution is not None else _img_density_for_neuromaps(self._image_obj) 
        self._dist_mat = load_distmat(self._dist_mat) if self._dist_mat is not None else None
        self._hemi = ("L", "R") if self._is_surface and not self._is_unilateral_surface else self._hemi
        
        # labels/indices by hemi
        if not self._is_unilateral_surface:
            (idc_lh, idc_rh), (labels_img_lh, labels_img_rh) = find_parcel_hemispheres(self._image_obj)
            self._idc_byhemi["L"] = self._idc_byhemi["L"] if self._idc_byhemi["L"] is not None else idc_lh
            self._idc_byhemi["R"] = self._idc_byhemi["R"] if self._idc_byhemi["R"] is not None else idc_rh
            self._labels_img_byhemi["L"] = self._labels_img_byhemi["L"] if self._labels_img_byhemi["L"] is not None else labels_img_lh
            self._labels_img_byhemi["R"] = self._labels_img_byhemi["R"] if self._labels_img_byhemi["R"] is not None else labels_img_rh
            self._labels_byhemi["L"] = self._labels_byhemi["L"] if self._labels_byhemi["L"] is not None else self._labels[idc_lh]
            self._labels_byhemi["R"] = self._labels_byhemi["R"] if self._labels_byhemi["R"] is not None else self._labels[idc_rh]
        
        # check: if one gifti, hemi must be provided
        if self._is_unilateral_surface and self._hemi not in ["L", "R"]:
            lgr.warning("Got single surface hemisphere but hemi not in ['L', 'R']. Assuming 'L'!")
            self._hemi = "L"
        # TODO: add checks
        
        return self
            
    def get_dist_mat(self, resample=2, centroids=False, n_proc=1, recalculate=False):
        if self._dist_mat is not None and not recalculate:
            return self._dist_mat
        else:
            self._dist_mat = get_distance_matrix(
                parc=self._image_obj,
                parc_space=self._space,
                parc_hemi=self._hemi,
                parc_resample=resample,
                centroids=centroids,
                n_proc=n_proc
            )
            return self._dist_mat
            