import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.image import resample_img
from neuromaps.images import load_gifti, load_nifti
from neuromaps.nulls.nulls import batch_surrogates
from neuromaps.nulls.nulls import _get_distmat
from neuromaps.datasets import fetch_fsaverage
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm
try:
    from brainspace.null_models.moran import MoranRandomization
    _BRAINSPACE_AVAILABLE = True
except ImportError:
    _BRAINSPACE_AVAILABLE = False
try:
    from brainsmash.mapgen import Base
    _BRAINSMASH_AVAILABLE = True
except ImportError:
    _BRAINSMASH_AVAILABLE = False

from . import lgr
from .utils import set_log


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
        return "fsaverage"
    else:
        raise ValueError(f"Provide input of type nib.Nifti1Image or (tuple of) nib.GiftiImage(s)!")

def _get_null_data_mask(data_1d, dist_mat):
    med = np.isinf(dist_mat + np.diag([np.inf] * len(dist_mat))).all(axis=1)
    mask = np.logical_not(np.logical_or(np.isnan(data_1d), med))
    return mask

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

def nulls_moran(data_1d, dist_mat, n_nulls=1000, seed=None,**kwargs):
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
        joint=True, 
        tol=1e-6, 
        n_rep=n_nulls, 
        random_state=seed,
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
                        downsample_vol=None, centroids=False, surf_euclidean=False,
                        n_proc=1, verbose=True, dtype=np.float32):
    verbose = set_log(lgr, verbose)
    
    ## generate distance matrix
    # case volumetric 
    if parc_space in ["MNI152", "mni152", "MNI", "mni"]:
        # get parcellation data
        parc = load_nifti(parc)
        if downsample_vol:
            if downsample_vol is True:
                downsample_vol = 3
            lgr.info(f"Downsampling volumetric parcellation to voxelsize of {downsample_vol} "
                      "for distance matrix generation.")
            parc = resample_img(
                parc, 
                target_affine=np.diag([downsample_vol, downsample_vol, downsample_vol]), 
                interpolation="nearest"
            )
        parc_data = parc.get_fdata()
        parc_affine = parc.affine
        parcels = np.trim_zeros(np.unique(parc_data))
        n_parcels = len(parcels)
        mask = np.logical_not(np.logical_or(np.isclose(parc_data, 0), np.isnan(parc_data)))
        parc_data_m = parc_data * mask

        # case distances between parcel centroids
        if centroids:  
            # get centroid coordinates in world space
            xyz = np.zeros((n_parcels, 3), float)
            for i, i_parcel in enumerate(parcels):
                xyz[i,:] = np.column_stack(np.where(parc_data_m==i_parcel)).mean(axis=0)
            ijk = nib.affines.apply_affine(parc_affine, xyz)
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
                parc_hemi=parc_hemi, 
                parc_density=_img_density_for_neuromaps(parc), 
            )
            dist = _dist_mat_from_coords(_parc_centroids, dtype=dtype)
            
        elif surf_euclidean & ("fsa" not in parc_space):
            lgr.warning("Distance matrix generation currently not implemented for surface " 
                        "spaces other than fsaverage. Will use random splits!")
            
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
            
            n_jobs = 2 if (n_proc > 1) & len(parc_hemi) > 1 else 1
            dist = Parallel(n_jobs=n_jobs)(
                delayed(surf_dist)(i, h) for i, h in enumerate(tqdm(
                    parc_hemi, 
                    desc=f"Calculating distance matrix ({n_jobs} proc)", 
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


def find_surf_parc_centroids(parc, parc_hemi, parc_density="10k"):

    # get parcellation
    if isinstance(parc_hemi, str):
        parc_hemi = [parc_hemi]
        
    if isinstance(parc, (tuple, list)) & (len(parc)==2):
        parc = (load_gifti(parc[0]), load_gifti(parc[1]))
        parc_hemi = ["L", "R"]
        print("Two-hemispheric parcellation provided, assuming order ['L', 'R'].")
    elif isinstance(parc, (str, nib.GiftiImage)):
        parc = (load_gifti(parc),)
        if len(parc_hemi)>1:
            print("Provided parcellation is one hemisphere but parc_label indicated both hemispheres. "
                  "Setting parc_hemi to ['L']!")
            parc_hemi = ["L"]
        else:
            print(f"One-hemispheric parcellation provided (hemisphere: {parc_hemi})")
    else:
        print(f"Parcellation must be provided as (tuple/list of) path(s) or Gifti image(s), "
            f"not {type(parc)}!")

    # get standard surface
    surfaces = fetch_fsaverage(parc_density)["pial"]
    if (len(parc_hemi)==1) & (parc_hemi[0]=="L"):
        surfaces = load_gifti(surfaces[0]),
    elif (len(parc_hemi)==1) & (parc_hemi[0]=="R"):
        surfaces = load_gifti(surfaces[1]),
    elif len(parc_hemi)==2:
        surfaces = (load_gifti(surfaces[0]), load_gifti(surfaces[1]))
    else:
        print("Problem with 'parc_hemi'. Provide ['L'], ['R'], or ['L', 'R']")
    
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
                       parc_space=None, parc_hemi=None, 
                       n_nulls=1000, centroids=False,
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
    if null_fun.__name__ == "nulls_moran" and not _BRAINSPACE_AVAILABLE:
        lgr.critical_raise("Null method 'moran' requires brainspace! Run 'pip install brainspace'!",
                           ImportError)
    elif null_fun.__name__ == "nulls_burt2020" and not _BRAINSMASH_AVAILABLE:
        lgr.critical_raise("Null method 'burt2020' requires brainsmash! Run 'pip install brainsmash'!",
                           ImportError)
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
    
    ## distance matrix provided -> we dont need parcellation
    if dist_mat is not None:
        lgr.info(f"Using provided distance matrix/matrices.")
        if isinstance(dist_mat, np.ndarray):
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
    if dist_mat is None:   
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
            if (parc_hemi is None) | (len(parc_hemi)>1):
                lgr.warning("If only one gifti parcellation image is supplied, 'parc_hemi' must "
                            "be one of: ['L'], ['R']! Assuming left hemisphere!" )
                parc_hemi = ["L"]
        if isinstance(parc, tuple):
            if (parc_hemi is None) | (len(parc_hemi)==1):
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
        lgr.info("Calculating distance matrix/matrices ({d}).".format(
            d='euclidean' if parc_space in ['mni','MNI','mni152','MNI152'] else 'geodesic'))
        dist_mat = get_distance_matrix(
            parc=parc, 
            parc_space=parc_space,
            parc_hemi=parc_hemi,
            centroids=centroids,
            n_proc=n_proc,
            verbose=False
        )
    
    ## generate null data    
    # case two surface files
    if isinstance(dist_mat, tuple):
        def par_fun(data_1d):
            data_1d_split = (data_1d[:len(dist_mat[0])], 
                             data_1d[len(dist_mat[0]):])
            null_data = []
            for data, dist in zip(data_1d_split, dist_mat):
                null_data.append(
                    null_fun(data_1d=data, dist_mat=dist, n_nulls=n_nulls, seed=seed, **kwargs)
                )
            return np.concatenate(null_data, 1)
    
    # case one surface/volume file
    else:
        def par_fun(data_1d):
            return null_fun(data_1d=data_1d, dist_mat=dist_mat, n_nulls=n_nulls, seed=seed, **kwargs)
        
    nulls = Parallel(n_jobs=n_proc)(
        delayed(par_fun)(data[i, :]) 
        for i in tqdm(
            range(n_data), 
            desc=f"{null_fun.__name__.split('_')[1].capitalize()} null maps ({n_proc} proc)", 
            disable=not verbose
        )
    )
    nulls = {l: n.astype(dtype) for l, n in zip(data_labs, nulls)}

    ## return
    lgr.info("Null data generation finished.")
    return nulls, dist_mat
        
    