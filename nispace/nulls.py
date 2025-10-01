import nibabel as nib
import numpy as np
import pandas as pd
import warnings
from joblib import Parallel, delayed
from nilearn.image import resample_img, coord_transform
from neuromaps.images import load_gifti, load_nifti, load_data
from neuromaps.nulls.nulls import batch_surrogates
from neuromaps.nulls.nulls import _get_distmat
from neuromaps.datasets import fetch_fsaverage, fetch_fslr
from scipy.spatial.distance import cdist
from sklearn.preprocessing import minmax_scale
from tqdm.auto import tqdm
from numba import njit

# import MoranRandomization function, copied from brainspace, as our default null model
# brainspace was removed as an dependency because it installs vtk, which is a large 3d rendering
# library that NiSpace does not use. 
from .modules.brainspace_moran import MoranRandomization
    
# brainsmash is optional dependency. Moran
try:
    from brainsmash.mapgen import Base
    _BRAINSMASH_AVAILABLE = True
except ImportError:
    _BRAINSMASH_AVAILABLE = False

from . import lgr
from .stats.coloc import corr
from .utils.utils import (set_log, mirror_nifti, mirror_gifti, vect_to_vol_arr, vol_to_vect_arr,
                          _corr_vector)


def _dist_mat_from_coords(coords, dtype=np.float32):
    dist_mat = np.zeros((coords.shape[0], coords.shape[0]), dtype=dtype)
    for i, row in enumerate(coords):
        dist_mat[i] = cdist(row[None], coords).astype(dtype)   
    return dist_mat

def _img_density_for_neuromaps(img):
    if isinstance(img, nib.GiftiImage):
        img = (img,)
    if isinstance(img, nib.Nifti1Image):
        return f"{np.round((img.affine[0,0])):.0f}mm"
    elif isinstance(img, tuple):
        return f"{np.round((img[0].agg_data().shape[0]/1000)):.0f}k"
    else:
        raise ValueError(f"Provide input of type nib.Nifti1Image or (tuple of) nib.GiftiImage(s)!")
    
def _img_space_for_neuromaps(img):
    if isinstance(img, nib.GiftiImage):
        img = (img,)
    if isinstance(img, nib.Nifti1Image):
        return "mni152"
    elif isinstance(img, tuple):
        density = _img_density_for_neuromaps(img[0])
        if density in ['3k', '10k', '41k']:
            return "fsaverage"
        elif density in ['4k', '8k', '32k']:
            return "fslr"
        elif density == '164k':
            lgr.warning("Identified surface image with 164k density, assuming fsLR but could be fsaverage!")
            return "fslr"
        else:
            lgr.critical_raise(f"Identified surface image with unknown density {density}!")
    else:
        raise ValueError(f"Provide input of type nib.Nifti1Image or (tuple of) nib.GiftiImage(s)!")

def _get_null_data_mask(data_1d, dist_mat):
    med = np.isinf(dist_mat + np.diag([np.inf] * len(dist_mat))).all(axis=1)
    mask = np.logical_not(np.logical_or(np.isnan(data_1d), med))
    return mask

def _symmetrize_nans(data_1d, idc):
    # check dtype
    if not isinstance(data_1d, (np.ndarray, pd.Series)) or not isinstance(idc, (list, tuple)):
        raise ValueError("'data_1d' must be a numpy array or pandas Series and 'idc' must be a list or tuple!")
    # check length
    if len(data_1d) != len(np.concatenate(idc)):
        raise ValueError("Length of 'data_1d' must match sum of number of elements in 'idc'!")
    # check if all idc have the same length
    if not all(len(i_idc) == len(idc[0]) for i_idc in idc):
        raise ValueError("All elements in 'idc' must have the same length!")
    # symmetrize
    data_1d = np.array(data_1d)
    isnan = np.full(len(idc[0]), False)
    for i_idc in idc:
        isnan = np.logical_or(isnan, np.isnan(data_1d[i_idc]))
    for i_idc in idc:
        data_1d[i_idc] = np.where(isnan, np.nan, data_1d[i_idc])
    # return
    return data_1d

def correlate_hemis_parc(data, parc_idc_lh=None, parc_idc_rh=None, l2rmap=None, rank=False):
    data = np.atleast_2d(np.array(data))
    parc_idc_lh = np.array(parc_idc_lh)
    parc_idc_rh = np.array(parc_idc_rh)
    n = data.shape[1]
    n_hemi = n // 2
    if n / 2 != n_hemi:
        raise ValueError("Data must have an even number of parcels")
    if parc_idc_lh is None:
        parc_idc_lh = np.arange(n_hemi)
    if parc_idc_rh is None:
        parc_idc_rh = np.arange(n_hemi, n)
    data_lh = data[:, parc_idc_lh]
    data_rh = data[:, parc_idc_rh]
    if l2rmap is not None:
        l2rmap = np.nan_to_num(np.array(l2rmap)).astype(data.dtype)
        for i in range(data.shape[0]):
            data_lh[i, :] = _apply_l2rmap(data[i, :], l2rmap, parc_idc_lh, parc_idc_rh)[parc_idc_rh]
    r = []
    for i in range(data.shape[0]):
        lh, rh = data_lh[i,:], data_rh[i,:]
        notnan = ~(np.isnan(lh) | np.isnan(rh))
        r.append(corr(lh[notnan], rh[notnan], rank=rank))
    return np.array(r)


def find_parcel_hemispheres(parcellation):
    
    # easy: surface
    if isinstance(parcellation, tuple):
        
        # load data
        data_lh, data_rh =  load_data(parcellation[0]), load_data(parcellation[1])
        
        # get labels and indices
        labels_lh, labels_rh = np.trim_zeros(np.unique(data_lh)), np.trim_zeros(np.unique(data_rh))
        labels_all = np.concatenate([labels_lh, labels_rh])
        
        # get indices
        idc_all = np.arange(len(labels_all))
        idc_lh = idc_all[np.isin(labels_all, labels_lh)]
        idc_rh = idc_all[np.isin(labels_all, labels_rh)]
    
    # complicated: volume
    elif isinstance(parcellation, nib.Nifti1Image):
        
        # load data
        data = load_data(parcellation)
        data_flat = data[data != 0].flatten()
        
        # get labels 
        labels_all = np.trim_zeros(np.unique(data))

        # get MNI coordinates
        ijk = np.argwhere(data != 0)
        x, y, z = coord_transform(ijk[:,0], ijk[:,1], ijk[:,2], parcellation.affine)
        
        # check for every label if the majority of voxels is in the left or right hemisphere
        labels_lh, labels_rh = [], []
        idc_lh, idc_rh = [], []
        for idx, lab in enumerate(labels_all):
            lr = (x[data_flat == lab] > 0).mean()
            if lr < 0.5:
                labels_lh.append(lab)
                idc_lh.append(idx)
            else:
                labels_rh.append(lab)
                idc_rh.append(idx)
                
        # return indices and labels
        idc_lh, idc_rh = np.array(idc_lh, dtype=int), np.array(idc_rh, dtype=int)
        labels_lh, labels_rh = np.array(labels_lh, dtype=int), np.array(labels_rh, dtype=int)
    
    # one gifti
    elif isinstance(parcellation, nib.GiftiImage):
        idc_lh, idc_rh, labels_lh, labels_rh = [None] * 4
        
    else:
        raise ValueError(f"Parcellation type {type(parcellation)} not supported.")
    
    return (idc_lh, idc_rh), (labels_lh, labels_rh)


# @njit(cache=True)
# def _apply_l2rmap(data_1d, l2rmap, parc_idc_lh, parc_idc_rh):
#     data_lh_1d = data_1d[parc_idc_lh]
#     data_mirrored_1d = np.full_like(data_1d, np.nan)
#     data_mirrored_1d[parc_idc_lh] = data_lh_1d
#     for i_parcel, idx_parcel in enumerate(parc_idc_rh):
#         data_mirrored_1d[idx_parcel] = np.nansum( data_lh_1d * l2rmap[:, i_parcel] )
#     return data_mirrored_1d

@njit(cache=True)
def _apply_l2rmap(data_1d, l2rmap, parc_idc_lh, parc_idc_rh):
    """Note: l2rmap must be without nans, but replacing them with 0s should not be a problem"""
    data_mirrored_1d = data_1d.copy()
    data_lh_1d = data_1d[parc_idc_lh]
    lh_notnan = ~np.isnan(data_lh_1d)
    data_mirrored_1d[parc_idc_rh] = np.dot(data_lh_1d[lh_notnan], l2rmap[lh_notnan, :])
    return data_mirrored_1d

def _mirror_parc_maps(data, parc_idc_lh, parc_idc_rh, 
                      l2rmap=None, interhemi_correlation=1, seed=None,
                      parc=None, project_to_volume=False, resample_vol=2, n_proc=1):
    
    data = np.array(data)
    if data.ndim == 1:
        data = data[None, :]
    data_lh = data[:, parc_idc_lh]
    data_mirrored = np.full_like(data, np.nan)
    data_mirrored[:, parc_idc_lh] = data_lh
    
    # simple copy of left to right indices
    if not project_to_volume and l2rmap is None:
        data_mirrored[:, parc_idc_rh] = data_lh
        
        # introduce interhemispheric correlation
        if interhemi_correlation != 1:
            
            # apply correlated vector function
            for i in range(data.shape[0]):
                data_mirrored[i, parc_idc_rh] = _corr_vector(
                    data_mirrored[i, parc_idc_rh], interhemi_correlation, seed)
            
            
    # use left-to-right mapping
    elif not project_to_volume and l2rmap is not None:
        
        l2rmap = np.nan_to_num(np.array(l2rmap)).astype(data.dtype)
        parc_idc_lh = np.array(parc_idc_lh)
        parc_idc_rh = np.array(parc_idc_rh)
        
        if interhemi_correlation == 1:
            mirror_fun = lambda x, seed=None: _apply_l2rmap(x, l2rmap, parc_idc_lh, parc_idc_rh)
        else:
            def mirror_fun(data_1d, seed=None):
                data_1d = _apply_l2rmap(data_1d, l2rmap, parc_idc_lh, parc_idc_rh)
                data_1d[parc_idc_rh] = _corr_vector(
                    data_1d[parc_idc_rh], interhemi_correlation, seed)
                return data_1d
            
        for i in range(data.shape[0]):
            data_mirrored[i, :] = mirror_fun(data[i, :], seed=i**2+i if seed is not None else None) 
        
    # else: complicated approach for cases in which the parcellation is not bilaterally symmetric
    # 1. take left-hemisphere parcels and mirror them across the x-axis
    # 2. project the right hemisphere data into volume space using the original parcellation
    # 3. re-parcellate the right-hemisphere data using the original parcellation
    elif project_to_volume:
        # TODO: test and debug
        raise NotImplementedError("Projecting to volume not at all tested!")
        if parc is None:
            raise ValueError("'parc' must be provided if 'project_to_volume' is True!")
        
        # parcellation data
        if resample_vol is not None:
            parc = resample_img(parc, target_affine=np.eye(3) * resample_vol, interpolation="nearest",
                                force_resample=True, copy_header=True)
        parc_data = parc.get_fdata()
        idc_all = np.trim_zeros(np.unique(parc_data))
        idc_left = idc_all[parc_idc_lh]
        idc_right = idc_all[parc_idc_rh]
        
        # 1.
        parc_left = mirror_nifti(parc_data, affine=parc.affine, direction="drop_right")
        parc_right = mirror_nifti(parc_data, affine=parc.affine, direction="drop_left")
        parc_right_mirrored = mirror_nifti(parc_left, affine=parc.affine, direction="switch")
                
        # 2. and 3. 
                
        def par_fun(vect_left):
            vol_right_mirrored = vect_to_vol_arr(vect_left, parc_right_mirrored, idc_left)
            vect_right_mirrored = vol_to_vect_arr(vol_right_mirrored, parc_right, idc_right)
            return vect_right_mirrored
        
        data_mirrored_rh = Parallel(n_jobs=n_proc)(
            delayed(par_fun)(data[i, :]) 
            for i in range(data.shape[0])
        )
        data_mirrored[:, parc_idc_rh] = np.stack(data_mirrored_rh, axis=0)
            
        # for i in range(data.shape[0]):
        #     vect_left = data[i, parc_idc_lh]
        #     vol_right_mirrored = vect_to_vol_arr(vect_left, parc_right_mirrored, idc_left)
        #     data_mirrored[i, parc_idc_rh] = vol_to_vect_arr(vol_right_mirrored, parc_right, idc_right)
    
    else:
        raise ValueError("Invalid input arguments, no mirroring method selected!")
    
    return data_mirrored
        

def nulls_burt2020(data_1d, dist_mat, n_nulls=1000, seed=None, **kwargs):
    data_1d = np.array(data_1d).flatten()
    # results array with shape (n_nulls, n_parcels)
    null_data = np.full((n_nulls, len(data_1d)), np.nan)
    # mask
    mask = _get_null_data_mask(data_1d, dist_mat)
    data_1d = data_1d[mask]
    dist_mat = dist_mat[np.ix_(mask, mask)]
    # null maps
    null_data[:, mask] = Base(
        x=data_1d, 
        D=dist_mat, 
        seed=seed,
        **kwargs
    )(n_nulls, 50)
    # return
    return null_data.astype(data_1d.dtype)

def nulls_burt2018(data_1d, dist_mat, n_nulls=1000, seed=None, **kwargs):
    data_1d = np.array(data_1d).flatten()
    # results array with shape (n_nulls, n_parcels)
    null_data = np.full((n_nulls, len(data_1d)), np.nan)
    # mask
    mask = _get_null_data_mask(data_1d, dist_mat)
    data_1d = data_1d[mask]
    dist_mat = dist_mat[np.ix_(mask, mask)]
    # data adjustment
    data_1d += np.abs(np.nanmin(data_1d)) + 0.1
    # null maps
    null_data[:, mask] = batch_surrogates(dist_mat, data_1d, n_surr=n_nulls, seed=seed, **kwargs).T
    # return
    return null_data.astype(data_1d.dtype)

def nulls_moran(data_1d, dist_mat, n_nulls=1000, seed=None, **kwargs):
    data_1d = np.array(data_1d).flatten()
    # results array with shape (n_nulls, n_parcels)
    null_data = np.full((n_nulls, len(data_1d)), np.nan)
    # mask
    mask = _get_null_data_mask(data_1d, dist_mat)
    data_1d = data_1d[mask]
    dist_mat = dist_mat[np.ix_(mask, mask)]
    # distance matrix adjustment
    np.fill_diagonal(dist_mat, 1)
    dist_mat **= -1
    # null maps
    null_data[:, mask] = MoranRandomization(
        procedure=kwargs.pop("procedure", "singleton"),
        joint=kwargs.pop("joint", True),
        seed=seed, 
        n_nulls=n_nulls,
        **kwargs
    ).fit(dist_mat).randomize(data_1d)
    # return
    return null_data.astype(data_1d.dtype)

def nulls_random(data_1d, dist_mat=None, n_nulls=1000, seed=None):
    # results array with shape (n_nulls, n_parcels)
    null_data = np.full((n_nulls, len(data_1d)), np.nan)
    # mask
    mask = ~np.isnan(data_1d)
    data_1d = data_1d[mask]
    # null maps
    rng = np.random.default_rng(seed)
    null_data[:, mask] = np.stack([rng.permutation(data_1d) for _ in range(n_nulls)], axis=0)
    # return
    return null_data.astype(data_1d.dtype)


_NULL_METHODS = {
    # Random
    "random": nulls_random,
    # Moran's I implemented via BrainSpace -> volumetric and surface
    "moran": nulls_moran,
    "brainspace": nulls_moran,
    # Variogram-method implemented via Brainsmash -> volumetric and surface
    "burt2020": nulls_burt2020,
    "brainsmash": nulls_burt2020,
    "variogram": nulls_burt2020,
    # Smoothing-method from Burt2018 -> volumetric and surface
    "burt2018": nulls_burt2018,
    # TODO: add spin methods
}    


def get_distance_matrix(parc, parc_space, parc_hemi=["L", "R"], 
                        parc_resample=2, centroids=False, surf_euclidean=False,
                        n_proc=1, verbose=True, dtype=np.float32):
    verbose = set_log(lgr, verbose)
    # TODO: ADD SUPPORT FOR PARCELLATION OBJECTS TO DISTANCE MATRIX GENERATION
    
    ## generate distance matrix
    # case volumetric 
    if "mni" in parc_space.lower():
        # get parcellation data
        parc = load_nifti(parc)
        if parc_resample:
            if parc_resample is True:
                parc_resample = 3
            lgr.info(f"Downsampling volumetric parcellation to voxelsize of {parc_resample} "
                      "for distance matrix generation.")
            parc = resample_img(
                parc, 
                target_affine=np.diag([parc_resample] * 3), 
                interpolation="nearest",
                force_resample=True, copy_header=True
            )
        parc_data = parc.get_fdata()
        parc_affine = parc.affine
        parcels = np.trim_zeros(np.unique(parc_data))
        n_parcels = len(parcels)
        mask = np.logical_not(np.logical_or(np.isclose(parc_data, 0), np.isnan(parc_data)))
        parc_data_m = parc_data * mask

        # case distances between volumetric parcel centroids
        if centroids:  
            # get centroids
            ijk = find_vol_parc_centroids(parc_data_m, affine=parc_affine, parcel_idc=parcels)
            # get distances
            dist = _dist_mat_from_coords(ijk, dtype)
            
        # case mean distances between parcel-wise voxels 
        else:
            # get parcel-wise coordinates in world space
            ijk_parcels = dict()
            for i_parcel in parcels:
                xyz_parcel = np.column_stack(np.where(parc_data_m==i_parcel))
                ijk_parcels[i_parcel] = nib.affines.apply_affine(parc_affine, xyz_parcel)
                
            def mni_dist(i, i_parcel):
                dist_i = np.zeros(n_parcels, dtype=dtype)
                j = i
                for _ in range(n_parcels - j):
                    dist_i[j] = cdist(ijk_parcels[i_parcel], ijk_parcels[parcels[j]]) \
                        .mean().astype(dtype)
                    j += 1
                return dist_i
            
            lgr.info(f"Estimating euclidean distance matrix between {n_parcels} volumetric parcels.")
            dist_list = Parallel(n_jobs=n_proc)(
                delayed(mni_dist)(i, i_parcel) for i, i_parcel in enumerate(tqdm(
                    parcels, 
                    desc=f"Running ({n_proc} proc)", disable=not verbose
                ))
            )
            dist = np.r_[dist_list]
            # mirror to lower triangle
            dist = dist + dist.T
            # zero diagonal
            np.fill_diagonal(dist, 0)
    
    # case surface
    elif parc_space in ["fsaverage", "fsLR", "fsa", "fslr"]:
        
        if surf_euclidean & ("fsa" in parc_space):
            lgr.info(f"Estimating euclidean distance matrix between surface parcels.")
            _parc_centroids = find_surf_parc_centroids(
                parc=parc,
                parc_space=parc_space,
                parc_hemi=parc_hemi, 
                parc_density=_img_density_for_neuromaps(parc), 
            )
            dist = _dist_mat_from_coords(_parc_centroids, dtype=dtype)
            
        elif surf_euclidean & ("fsa" not in parc_space):
            lgr.warning("Distance matrix generation currently not implemented for surface " 
                        "spaces other than fsaverage. Will use random splits!")
        
        # TODO: implement fsLR distance matrix
        elif "fsLR" in parc_space.lower():
            lgr.critical_raise(
                "Distance matrix generation currently not implemented for fsLR space. "
                "Use 'fsaverage' instead!",
                ValueError
            )
        else:
            lgr.info(f"Estimating geodesic distance matrix between surface parcels.")
            def surf_dist(i_hemi, hemi):
                dist = _get_distmat(
                    hemi, 
                    atlas=parc_space, 
                    density=_img_density_for_neuromaps(parc[i_hemi]), 
                    parcellation=parc[i_hemi] if len(parc_hemi) > 1 else parc,
                    n_proc=n_proc
                )
                return(dist)
            
            dist = Parallel(n_jobs=n_proc)(
                delayed(surf_dist)(i, h) for i, h in enumerate(tqdm(
                    parc_hemi, 
                    desc=f"Calculating distance matrix ({n_proc} proc)", 
                    disable=not verbose
                ))
            )
            
            if isinstance(parc, tuple):
                dist = tuple(dist)
            else:
                dist = dist[0]

    # case other
    else:
        lgr.error(f"Distance matrix generation not supported for space {parc_space}!")
        
    ## return
    return dist

def find_vol_parc_centroids(parc, affine=None, parcel_idc=None, return_data_space=False):
    # get parcellation data
    if isinstance(parc, np.ndarray):
        parc_data = parc
    else:
        parc = load_nifti(parc)
        parc_data = parc.get_fdata()
    
    # get affine matrix
    if affine is None:
        if not isinstance(parc, nib.Nifti1Image):
            lgr.critical_raise("If 'affine' is not provided, 'parc' must be a Nifti image!",
                               TypeError)
        affine = parc.affine
    
    # get parcel indices
    if parcel_idc is None:
        parcel_idc = np.trim_zeros(np.unique(parc_data))
    
    # get centroid coordinates in world space
    xyz = np.zeros((len(parcel_idc), 3), float)
    for i, i_parcel in enumerate(parcel_idc):
        xyz[i, :] = np.column_stack(np.where(parc_data==i_parcel)).mean(axis=0)
    mni = nib.affines.apply_affine(affine, xyz)
    
    return mni if not return_data_space else (mni, xyz)


def find_surf_parc_centroids(parc, parc_space="fsaverage", parc_hemi=None, parc_density=None):
    # TODO: switch template fetching to nispace after we added fsaverage and fsLR templates in all 
    # resolutions

    # check parc space
    if "fsa" in parc_space.lower():
        fetch_func = fetch_fsaverage
        surf_name = "pial"
    elif "fslr" in parc_space.lower():
        fetch_func = fetch_fslr
        surf_name = "midthickness"
    else:
        lgr.critical_raise(f"Parcellation space '{parc_space}' not supported. "
                           f"Choose one of 'fsaverage' or 'fsLR'.", ValueError)
    
    # get parcellation
    if isinstance(parc_hemi, str):
        parc_hemi = [parc_hemi]
        
    if isinstance(parc, (tuple, list)) & (len(parc)==2):
        parc = (load_gifti(parc[0]), load_gifti(parc[1]))
        parc_hemi = ["L", "R"]
        lgr.info("Two-hemispheric parcellation provided, assuming order ['L', 'R'].")
    elif isinstance(parc, (str, nib.GiftiImage)):
        parc = (load_gifti(parc),)
        if len(parc_hemi)>1:
            lgr.warning("Provided parcellation is one hemisphere but parc_label indicated both hemispheres. "
                        "Setting parc_hemi to ['L']!")
            parc_hemi = ["L"]
        else:
            lgr.info(f"One-hemispheric parcellation provided (hemisphere: {parc_hemi})")
    else:
        lgr.critical_raise(f"Parcellation must be provided as (tuple/list of) path(s) or Gifti image(s), "
                           f"not {type(parc)}!",
                           TypeError)
    
    # guess parc_density if None
    if parc_density is None:
        parc_density = _img_density_for_neuromaps(parc)

    # get standard surface
    surfaces = fetch_func(parc_density)[surf_name]
    if (len(parc_hemi)==1) & (parc_hemi[0]=="L"):
        surfaces = load_gifti(surfaces[0]),
    elif (len(parc_hemi)==1) & (parc_hemi[0]=="R"):
        surfaces = load_gifti(surfaces[1]),
    elif len(parc_hemi)==2:
        surfaces = (load_gifti(surfaces[0]), load_gifti(surfaces[1]))
    else:
        lgr.critical_raise("Problem with 'parc_hemi'. Provide ['L'], ['R'], or ['L', 'R']",
                           ValueError)
    
    centroids = []
    # iterate hemispheres
    for parc_h, surf_h in zip(parc, surfaces):
        labels = parc_h.darrays[0].data
        coords = surf_h.darrays[0].data
        
        # iterate parcels ("labels") and collect mean coordinates
        for idx in np.trim_zeros(np.unique(labels)):
            parcel = np.atleast_2d(coords[labels == idx].mean(axis=0))
            parcel = coords[np.argmin(cdist(coords, parcel), axis=0)[0]]
            centroids.append(parcel)
            
    return np.row_stack(centroids)       


def generate_null_maps(method, data, parcellation, dist_mat=None, 
                       parc_space=None, parc_hemi=None, parc_symmetric=False,
                       n_nulls=1000, parc_resample=2, centroids=False, 
                       parc_idc_lh=None, parc_idc_rh=None, parc_idc_sc=None, 
                       lr_mirror_dist_mat=False, lr_mirror_null_maps=False, l2rmap=None, 
                       match_interhemi_correlation=True, report_interhemi_correlation=False,
                       cx_sc_minmax_scale=False,
                       dtype=float,
                       n_proc=1, seed=None, verbose=True,
                       **kwargs):
    if verbose is False:
        set_log(lgr, verbose)
    
    ## Checks
    # null method
    if method not in _NULL_METHODS:
        lgr.critical_raise(f"Null method {method} not implemented!",
                           ValueError)
    null_fun = _NULL_METHODS[method]
    random_nulls = False
    if null_fun.__name__ == "nulls_burt2020" and not _BRAINSMASH_AVAILABLE:
        lgr.critical_raise("Null method 'burt2020' requires brainsmash! Run 'pip install brainsmash'!",
                           ImportError)
    elif null_fun.__name__ == "nulls_random":
        random_nulls = True
        
    # input data
    if not isinstance(data, (pd.DataFrame, pd.Series, np.ndarray)):
        lgr.critical_raise(f"Input data not array-like! Type: {type(data)}",
                           ValueError)
    if isinstance(data, pd.DataFrame):
        data_labs = list(data.index)
    elif isinstance(data, pd.Series):
        data_labs = [data.name]
    data = np.array(data)
    if len(data.shape) == 1:
        data = data[np.newaxis, :]
    n_data = data.shape[0]
    if "data_labs" not in locals():
        data_labs = list(range(n_data))
        
    # print
    lgr.info(f"Null map generation: Assuming n = {n_data} data vector(s) for "
             f"n = {data.shape[1]} parcels.")
    
    ## random nulls -> no distmat
    if random_nulls:
        dist_mat = (None, None) if isinstance(dist_mat, tuple) else None
        
    ## distance matrix provided -> we dont need parcellation
    if dist_mat is not None and not random_nulls:        
        lgr.info(f"Using provided distance matrix/matrices.")
        if isinstance(dist_mat, (np.ndarray, pd.DataFrame)):
            n_parcels = dist_mat.shape[0]
            if parc_space is None:
                lgr.warning("Distance matrix provided as array but 'parc_space' is None: "
                            "Assuming 'mni152'! Define 'parc_space' if one surface hemisphere!")
                parc_space = "mni152"
        elif isinstance(dist_mat, tuple):
            n_parcels = (dist_mat[0].shape[0],
                         dist_mat[1].shape[0])     
            if parc_space is None:
                lgr.warning("Distance matrix provided as tuple but 'parc_space' is None: "
                            "Assuming 'fsaverage'!")
                parc_space = "fsaverage"
        else:
            lgr.critical("Distance matrix is wrong data type, should be array or tuple of arrays, "
                         f"is: {type(dist_mat)}! Setting 'dist_mat' to None!")
            dist_mat = None
      
    ## get dist mat -> we need parcellation
    if dist_mat is None and not random_nulls:
        # load function
        def load_parc(parc, parc_type, parc_space):
            if parc_type=="nifti":
                parc = load_nifti(parc)
                parc_space = "MNI152" if parc_space is None else parc_space
                n_parcels = len(np.trim_zeros(np.unique(parc.get_fdata())))
            elif parc_type=="gifti":
                parc = load_gifti(parc)
                parc_space = "fsaverage" if parc_space is None else parc_space
                n_parcels = len(np.trim_zeros(np.unique(parc.darrays[0].data)))
            elif parc_type=="giftituple":
                parc = (load_gifti(parc[0]), load_gifti(parc[1]))
                parc_space = "fsaverage" if parc_space is None else parc_space
                n_parcels = (len(np.trim_zeros(np.unique(parc[0].darrays[0].data))),
                             len(np.trim_zeros(np.unique(parc[1].darrays[0].data))))
            return parc, parc_space, n_parcels
        
        # recognize parcellation type
        if isinstance(parcellation, nib.Nifti1Image):
            parc_type = "nifti"
        elif isinstance(parcellation, nib.GiftiImage):
            parc_type = "gifti"
        elif isinstance(parcellation, tuple):
            parc_type = "giftituple"
        elif isinstance(parcellation, str):
            if parcellation.endswith(".nii") | parcellation.endswith(".nii.gz"):
                parc_type = "nifti"
            elif parcellation.endswith(".gii") | parcellation.endswith(".gii.gz"):
                parc_type = "gifti"
            else:
                lgr.critical_raise(f"'parcellation' is string ({parcellation}) "
                                "but ending was not recognized!",
                                ValueError)
        else:
            lgr.critical_raise(f"'parcellation' data type ({type(parcellation)}) not defined!",
                            TypeError)    
            
        # load parcellation
        parc, parc_space, n_parcels = load_parc(parcellation, parc_type, parc_space)

        # check for problems
        if isinstance(parc, nib.GiftiImage):
            if parc_hemi is None:
                lgr.warning("If only one gifti parcellation image is supplied, 'parc_hemi' must "
                            "be one of: ['L'], ['R']! Assuming left hemisphere!" )
                parc_hemi = ["L"]
            elif len(parc_hemi) > 1:
                lgr.warning("If only one gifti parcellation image is supplied, 'parc_hemi' can "
                            "only be one of: ['L'], ['R']! Assuming left hemisphere!" )
                parc_hemi = ["L"]
        if isinstance(parc, tuple):
            if parc_hemi is None:
                parc_hemi = ["L", "R"]
            elif len(parc_hemi) == 1:
                lgr.warning("If 'parc_hemi' is ['L'] or ['R'], only one gifti parcellation image "
                            "should be supplied as string or gifti! Assuming both hemispheres!")
                parc_hemi = ["L", "R"]   
        if np.sum(n_parcels) != data.shape[1]:
            lgr.error(f"Number of parcels in data (1. dimension, {data.shape[1]}) "
                      f"does not match number of parcels in parcellation ({n_parcels})!")
        
        # print
        temp = f", parc_hemi = {parc_hemi}"
        lgr.info(f"Loaded parcellation (parc_space = '{parc_space}'"
                f"{temp if parc_space in ['fsaverage', 'fsLR', 'fsa', 'fslr'] else ''}).")
     
        ## calculate distance matrix
        # lgr.info("Calculating distance matrix/matrices ({d}).".format(
        #     d='euclidean' if parc_space in ['mni','MNI','mni152','MNI152'] else 'geodesic'))
        dist_mat = get_distance_matrix(
            parc=parc, 
            parc_space=parc_space,
            parc_hemi=parc_hemi,
            parc_resample=parc_resample,
            centroids=centroids,
            n_proc=n_proc,
            verbose=verbose
        )
    
    ## generate null data     
    
    # check symmetry settings
    if lr_mirror_dist_mat and not parc_symmetric:
        lgr.warning("Left-right mirroring of distance matrix (lr_mirror_dist_mat) requested, but "
                    "parcellation may not be symmetric.\nCheck if your parcellation is symmetric and "
                    "set 'parc_symmetric' to True.\nBe careful, this might lead to unexpected results!")
        lr_mirror_dist_mat = False
    if lr_mirror_null_maps and not parc_symmetric and l2rmap is None:
        lgr.warning("Left-right mirroring of null maps ('lr_mirror_null_maps') requested, but "
                    "'parc_symmetric' is False\nand no left-to-right mapping df ('l2rmap') provided. "
                    "Check if your parcellation is symmetric and set 'parc_symmetric' to True.\n"
                    "Be careful, this might lead to unexpected results!")
        lr_mirror_null_maps = False
    
    # check if separate indices for hemispheres are provided as tuple of arrays
    if parc_idc_lh is not None and parc_idc_rh is not None:
        if not isinstance(parc_idc_lh, (list, np.ndarray)) or not isinstance(parc_idc_rh, (list, np.ndarray)):
            lgr.warning("'parc_idc_lh' and 'parc_idc_rh' must be lists or arrays! Setting both to None!")
            parc_idc_lh, parc_idc_rh = None, None
    elif parc_idc_lh is None and parc_idc_rh is None:
        pass
    else:
        for var, idc in [("parc_idc_lh", parc_idc_lh), ("parc_idc_rh", parc_idc_rh)]:
            if idc is not None:
                if not isinstance(idc, (list, np.ndarray)):
                    lgr.warning(f"'{var}' must be a list or array! Setting '{var}' to None!")
                    locals()[var] = None
                else:
                    lgr.warning(f"Only indices of {var} provided, inferring indices of other hemisphere!")
                    if var == "parc_idc_lh":
                        parc_idc_rh = np.setdiff1d(np.arange(data.shape[1]), idc)
                    else:
                        parc_idc_lh = np.setdiff1d(np.arange(data.shape[1]), idc)
            
    # check if separate indices for subcortex are provided as array
    if parc_idc_sc is not None:
        if not isinstance(parc_idc_sc, (list, np.ndarray)):
            lgr.warning("'parc_idc_sc' must be a list or array! Setting 'parc_idc_sc' to None!")
            parc_idc_sc = None
    
    # create indices for cortex
    if parc_idc_sc is not None:
        parc_idc_cx = np.setdiff1d(np.arange(data.shape[1]), parc_idc_sc)
        if len(parc_idc_cx) == 0:
            parc_idc_sc, parc_idc_cx = None, None
            
    # get all index lists according to which we want to split the data and distance matrix
    if isinstance(dist_mat, tuple): # surface input
        split_by_idc = (
            np.arange(dist_mat[0].shape[0]), # left hemisphere
            np.arange(dist_mat[1].shape[0]) + dist_mat[0].shape[0], # right hemisphere
        )
    elif parc_idc_lh is None and parc_idc_sc is None: # none given
        split_by_idc = (
            np.arange(data.shape[1]), # whole dataset
        )
        lr_mirror_dist_mat = False
    elif parc_idc_lh is not None and parc_idc_sc is not None: # hemis + sc given
        lgr.info("Generating null data separately for left and right cortex and subcortex.")
        split_by_idc = (
            np.intersect1d(parc_idc_lh, parc_idc_cx),  # left cortex
            np.intersect1d(parc_idc_rh, parc_idc_cx),  # right cortex
            np.intersect1d(parc_idc_lh, parc_idc_sc),  # left subcortex
            np.intersect1d(parc_idc_rh, parc_idc_sc),  # right subcortex
        )
    elif parc_idc_lh is not None: # hemi but not sc given
        lgr.info("Generating null data separately for left and right hemisphere.")
        split_by_idc = (
            parc_idc_lh,  # whole left hemisphere
            parc_idc_rh,  # whole right hemisphere
        )
    elif parc_idc_sc is not None: # sc but not hemi given
        lgr.info("Generating null data separately for cortex and subcortex.")
        split_by_idc = (
            parc_idc_cx, # whole cortex
            parc_idc_sc, # whole subcortex
        )
        # set lr_mirror_dist_mat to False as we apparently dont have hemisphere-indices available
        lr_mirror_dist_mat = False
    else:
        lgr.critical_raise("Problem with 'split_by_idc': No indices generated/provided!", 
                           ValueError)
        
    # check if indices are missing
    missing_idc = np.setdiff1d(np.arange(data.shape[1]), np.concatenate(split_by_idc))
    if len(missing_idc) == data.shape[1]:
        lgr.critical_raise("No parcel indices are present in the processed data! Check the provided "
                           "'parc_idc_lh', 'parc_idc_rh', and 'parc_idc_sc' variables.",
                           ValueError)
    elif len(missing_idc) > 0:
        lgr.warning(f"Some parcel indices are missing in the processed data! You might want to check "
                    f"the provided 'parc_idc_lh', 'parc_idc_rh', and 'parc_idc_sc' variables. "
                    f"Missing indices: {missing_idc}")
        
    # check duplicate indices
    if np.unique(np.concatenate(split_by_idc)).size != np.concatenate(split_by_idc).size:
        lgr.critical_raise("Duplicate indices found in 'parc_idc_lh' and 'parc_idc_rh'! "
                           "Check if 'parc_idc_lh' and 'parc_idc_rh' are correctly defined.",
                           ValueError)
    
    # check if any index set is empty
    if any(len(idc) == 0 for idc in split_by_idc):
        lgr.critical_raise(f"Empty index set found! {[len(idc) for idc in split_by_idc]}",
                           ValueError)
        
    # split distance matrix to align with surface hemisphere distance matrices 
    if not isinstance(dist_mat, tuple):
        if dist_mat is not None:
            dist_mat_split = tuple([dist_mat[np.ix_(i, i)] for i in split_by_idc])
        else:
            dist_mat_split = tuple([None] * len(split_by_idc))
    else:
        dist_mat_split = dist_mat
    
    # check distance matrices
    if not random_nulls:
        if any(dist is None or dist.shape[0] != dist.shape[1] for dist in dist_mat_split):
            lgr.critical_raise("Distance matrix is not square or None! Check the provided distance matrix.",
                            ValueError)
        
    # mirror distance matrix if requested
    if lr_mirror_dist_mat and dist_mat is not None:
        lgr.info("Left-right averaging distance matrices to generate symmetrized null maps.")
        if len(dist_mat_split) == 2:
            dist_mat_split = tuple([np.mean(dist_mat_split, axis=0)] * 2)
            if not np.allclose(dist_mat_split[0], dist_mat_split[1]):
                lgr.critical_raise("Left-right averaged whole-hemisphere distance matrices are not equal! "
                                   "Check if 'parc_idc_lh' and 'parc_idc_rh' are correctly defined.",
                                   ValueError)
        elif len(dist_mat_split) == 4:
            dist_mat_split = tuple([np.mean(dist_mat_split[:2], axis=0)] * 2 + 
                                   [np.mean(dist_mat_split[2:], axis=0)] * 2)
            if not (np.allclose(dist_mat_split[0], dist_mat_split[1]) and np.allclose(dist_mat_split[2], dist_mat_split[3])):
                lgr.critical_raise("Left-right averaged cortical and subcortical distance matrices are not equal! "
                                   "Check if 'parc_idc_lh' and 'parc_idc_rh' are correctly defined.",
                                   ValueError)
        if np.isnan(data).any():
            lgr.info("Symmetrizing NaNs in data.")
            for i in range(data.shape[0]):
                data[i, :] = _symmetrize_nans(data[i, :], [parc_idc_lh, parc_idc_rh])
                
    # define function to generate null data
    def par_fun(data_1d, seed):
        null_data = np.full((n_nulls, len(data_1d)), np.nan)
        for idc, dist in zip(split_by_idc, dist_mat_split):
            data_1d_sel = data_1d[idc]
            if np.isnan(data_1d_sel).all():
                null_data[:, idc] = np.nan
            else:
                null_data[:, idc] = null_fun(data_1d=data_1d_sel, dist_mat=dist, n_nulls=n_nulls, seed=seed, **kwargs)
        return null_data
    
    # run null data generation
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
    nulls = Parallel(n_jobs=n_proc)(
        delayed(par_fun)(data[i, :], seed + i) 
        for i in tqdm(
            range(n_data), 
            desc=f"{null_fun.__name__.split('_')[1].capitalize()} null maps ({n_proc} proc)", 
            disable=not verbose
        )
    )
    nulls = {l: n.astype(dtype) for l, n in zip(data_labs, nulls)}
    
    # mirror null maps if requested
    if lr_mirror_null_maps and parc_idc_lh is not None and parc_idc_rh is not None:
        # checks
        if (parc_idc_lh is None or parc_idc_rh is None):
            lgr.warning("Left-to-right mirroring of null maps requested but 'parc_idc_lh' and/or "
                        "'parc_idc_rh' are not defined! Skipping mirroring.")
        elif len(parc_idc_lh) != len(parc_idc_rh):
            lgr.warning("Left-to-right mirroring of null maps requested but left/right hemisphere "
                        "parcel indices have different lengths! Skipping mirroring.")
        else:
            lgr.info("Left-to-right mirroring null maps "
                     f"({'simple mirroring' if parc_symmetric else 'using left-to-right mapping'}).")
            # get interhemispheric correlation if necessary
            if match_interhemi_correlation:
                lgr_msg = "Matching interhemispheric correlation. "
            if match_interhemi_correlation or report_interhemi_correlation:
                interhemi_corr_obs = correlate_hemis_parc(data, parc_idc_lh, parc_idc_rh, l2rmap)
                lgr_msg += f"Observed correlation: {interhemi_corr_obs.mean():.2f}"
                if len(nulls) > 1:
                    lgr_msg += f" (mean), {interhemi_corr_obs.min():.2f} - {interhemi_corr_obs.max():.2f} (range)"
                lgr.info(lgr_msg)
            else:
                interhemi_corr_obs = np.ones(len(nulls))
            
            # run for each input map / set of null maps
            def par_fun(n, r, seed):
                return _mirror_parc_maps(
                    n, parc_idc_lh, parc_idc_rh, 
                    l2rmap=None if parc_symmetric else l2rmap,
                    interhemi_correlation=r, 
                    seed=seed
                )
            nulls_list = Parallel(n_jobs=n_proc)(
                delayed(par_fun)(n, interhemi_corr_obs[i], seed + i ** 2) 
                for i, n in enumerate(tqdm(nulls.values(), desc="Mirroring null maps", disable=not verbose))
            )
            nulls = {l: n for l, n in zip(nulls.keys(), nulls_list)}
            
            # get interhemispheric correlation of null maps
            if report_interhemi_correlation:
                interhemi_corr_nulls = np.zeros(len(nulls))
                for i, (l, n) in enumerate(nulls.items()):
                    interhemi_corr_nulls[i] = correlate_hemis_parc(n, parc_idc_lh, parc_idc_rh, l2rmap).mean()
                lgr.info(f"Interhemispheric correlation of null maps:\n" + \
                         "\n".join([f"{l}: obs: {interhemi_corr_obs[i]:.3f} -> null (mean): {interhemi_corr_nulls[i]:.3f}" 
                                    for i, l in enumerate(nulls)]))
            
    # adjust scaling
    if cx_sc_minmax_scale:
        if parc_idc_sc is None:
            lgr.warning("To perform subcortical and cortical scaling adjustment, provide 'parc_idc_sc'!")
        else:
            lgr.info("Matching min-max range of subcortical and cortical null data to observed data.")
            for i, (l, n) in enumerate(nulls.items()):
                # get min, max, and mean of original subcortical data
                scale_sc = (np.nanmin(data[i, parc_idc_sc]), np.nanmax(data[i, parc_idc_sc]))
                # get min, max, and mean of original cortical data
                scale_cx = (np.nanmin(data[i, parc_idc_cx]), np.nanmax(data[i, parc_idc_cx]))
                # scale subcortical null data
                nulls[l][:, parc_idc_sc] = minmax_scale(n[:, parc_idc_sc], feature_range=scale_sc, axis=1)
                # scale cortical null data
                nulls[l][:, parc_idc_cx] = minmax_scale(n[:, parc_idc_cx], feature_range=scale_cx, axis=1)
                
    ## return
    lgr.info("Null data generation finished.")
    return nulls, dist_mat
        
    