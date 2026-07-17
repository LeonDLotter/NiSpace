import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.image import resample_img, coord_transform
from neuromaps.images import load_gifti, load_nifti, load_data, PARCIGNORE
from neuromaps.nulls.nulls import batch_surrogates
from neuromaps.nulls.spins import gen_spinsamples, get_parcel_centroids, spin_parcels
from collections import namedtuple
from neuromaps.points import make_surf_graph
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

_SurfPair = namedtuple("_SurfPair", ["L", "R"])

# import MoranRandomization function, copied from brainspace, as our default null model
# brainspace was removed as an dependency because it installs vtk, which is a large 3d rendering
# library that NiSpace does not use. 
from ._brainspace_moran import MoranRandomization
    
# brainsmash is optional dependency. Moran
try:
    from brainsmash.mapgen import Base
    _BRAINSMASH_AVAILABLE = True
except ImportError:
    _BRAINSMASH_AVAILABLE = False

import logging
lgr = logging.getLogger(__name__)
from .stats.coloc import corr
from .utils.utils import set_log
from .core.nullmaps import NullMaps

# ==================================================================================================
# DEPRECATION MESSAGE STRINGS
# ==================================================================================================

_DEPR_RETURN_DICT = (
    "return_dict=True is deprecated and will be removed in a future release. "
    "The returned NullMaps supports dict-like access."
)


def _dist_mat_from_coords(coords, dtype=np.float32):
    dist_mat = np.zeros((coords.shape[0], coords.shape[0]), dtype=dtype)
    for i, row in enumerate(coords):
        dist_mat[i] = cdist(row[None], coords).astype(dtype)
    return dist_mat


def _surf_geodesic_row(i, parcel_verts, graph, n_parcels, centroids, dtype):
    """Compute one row of the geodesic parcel-parcel distance matrix (upper triangle).

    Default (centroids=False): multi-source Dijkstra from all vertices of parcel i,
    then average distances to all vertices of each parcel j >= i.
    Centroids (centroids=True): single-source Dijkstra from the centroid vertex of
    parcel i, then read off distances to centroid vertices of parcels j >= i.
    """
    src = parcel_verts[i] if not centroids else parcel_verts[i][:1]
    dists = dijkstra(graph, directed=False, indices=src)  # (n_src, n_verts)
    if dists.ndim == 1:
        dists = dists[np.newaxis, :]
    row = np.zeros(n_parcels, dtype=dtype)
    for j in range(i, n_parcels):
        tgt = parcel_verts[j] if not centroids else parcel_verts[j][:1]
        row[j] = dists[:, tgt].mean()
    return row


def _surf_dist_hemi(gifti_surf, gifti_parc, medial_gifti, centroids, n_proc, dtype, verbose, hemi=""):
    """Geodesic parcel-parcel distance matrix for one hemisphere.

    Mirrors the volumetric voxel-to-voxel path:
    - build graph once
    - one Parallel job per parcel (upper triangle only)
    - mirror to fill lower triangle
    """
    vert, faces = load_gifti(gifti_surf).agg_data()
    labels = load_gifti(gifti_parc).agg_data()
    if labels.ndim > 1:
        labels = labels.squeeze()
    labels = labels.astype(int)

    # medial-wall mask: True = exclude vertex
    medial_mask = np.zeros(len(vert), dtype=bool)
    if medial_gifti is not None:
        mw = load_gifti(medial_gifti).agg_data()
        if mw.ndim > 1:
            mw = mw.squeeze()
        medial_mask = ~mw.astype(bool)

    # build graph with medial wall excluded
    graph = make_surf_graph(vert, faces, mask=medial_mask)

    # collect vertex indices per parcel (all parcels; medial-wall parcels fall back to all vertices)
    parc_ids = list(np.trim_zeros(np.unique(labels)))
    n_parcels = len(parc_ids)

    def _parcel_verts(pid, require_non_medial=True):
        verts = np.where((labels == pid) & ~medial_mask)[0] if require_non_medial \
                else np.where(labels == pid)[0]
        if len(verts) == 0:
            verts = np.where(labels == pid)[0]
            lgr.warning(f"Parcel {pid} has no non-medial-wall vertices; "
                        "it will be kept but distances may be unreliable.")
        return verts

    if centroids:
        # snap centroid: vertex within parcel closest to coordinate mean
        parcel_verts = []
        for pid in parc_ids:
            verts_idx = _parcel_verts(pid)
            mean_coord = vert[verts_idx].mean(axis=0)
            snap = verts_idx[np.argmin(np.linalg.norm(vert[verts_idx] - mean_coord, axis=1))]
            parcel_verts.append(np.array([snap]))
    else:
        parcel_verts = [_parcel_verts(pid) for pid in parc_ids]

    hemi_tag = f" {hemi}" if hemi else ""
    mode_tag = "centroid" if centroids else "vertex-to-vertex"
    lgr.info(f"Estimating geodesic distance matrix: {hemi_tag + ', ' if hemi_tag else ''}{n_parcels} surface parcels, "
             f"{mode_tag} mode, {n_proc} proc.")

    dist_rows = Parallel(n_jobs=n_proc)(
        delayed(_surf_geodesic_row)(i, parcel_verts, graph, n_parcels, centroids, dtype)
        for i in tqdm(range(n_parcels), desc=f"Distance matrix{hemi_tag} ({n_proc} proc)", disable=not verbose)
    )

    dist = np.array(dist_rows, dtype=dtype)
    dist = dist + dist.T
    np.fill_diagonal(dist, 0)
    return dist

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

def correlate_hemis_parc(data, parc_idc_lh=None, parc_idc_rh=None, rank=False):
    """Per-row left/right-hemisphere correlation of parcellated data.

    Plain numpy, using :func:`nispace.stats.coloc.corr` (not numba itself,
    but the underlying `corr` call is numba-jitted). Not currently called
    anywhere in NiSpace (dead code) — kept as a standalone diagnostic
    utility, e.g. for sanity-checking left-right symmetry of a map.

    Parameters
    ----------
    data : array-like
        1D (single map) or 2D (``n_maps, n_parcels``) parcellated data;
        1D input is treated as a single row.
    parc_idc_lh, parc_idc_rh : array-like of int, optional
        Column positions belonging to the left/right hemisphere. Default:
        split the columns into two equal halves.
    rank : bool, default=False
        Rank-transform before correlating (Spearman instead of Pearson).

    Returns
    -------
    np.ndarray
        Shape ``(n_maps,)``; per-row LH-RH correlation. NaN pairs are
        excluded per row; rows with fewer than 2 valid pairs or zero
        variance in either hemisphere give NaN.
    """
    data = np.atleast_2d(np.array(data))
    n = data.shape[1]
    n_hemi = n // 2
    if parc_idc_lh is None:
        parc_idc_lh = np.arange(n_hemi)
    if parc_idc_rh is None:
        parc_idc_rh = np.arange(n_hemi, n)
    parc_idc_lh = np.array(parc_idc_lh)
    parc_idc_rh = np.array(parc_idc_rh)
    data_lh = data[:, parc_idc_lh]
    data_rh = data[:, parc_idc_rh]
    r = []
    for i in range(data.shape[0]):
        lh, rh = data_lh[i,:], data_rh[i,:]
        notnan = ~(np.isnan(lh) | np.isnan(rh))
        lh_sel, rh_sel = lh[notnan], rh[notnan]
        # ddof=0: zero-variance guard only, choice doesn't affect the degenerate-
        # case detection, pinned for consistency
        if len(lh_sel) < 2 or np.std(lh_sel, ddof=0) == 0 or np.std(rh_sel, ddof=0) == 0:
            r.append(np.nan)
        else:
            r.append(corr(lh_sel, rh_sel, rank=rank))
    return np.array(r)


def find_parcel_hemispheres(parcellation):
    """Auto-detect per-parcel hemisphere membership for a parcellation.

    Plain numpy/nibabel (no numba). Used in ``core/parcellation.py`` for
    parcel-hemisphere bookkeeping (e.g. building `parc_idc_lh`/`parc_idc_rh`
    for downstream spin/distance-based null generation).

    Parameters
    ----------
    parcellation : tuple of two GiftiImage, nib.Nifti1Image, or single GiftiImage
        - Bilateral surface (``(lh_img, rh_img)``): trivial split by hemisphere.
        - Volumetric: each parcel's hemisphere is decided by the majority of
          its voxels' x-coordinate sign (world space, via the image affine).
        - Single (unilateral) GiftiImage: hemisphere membership cannot be
          determined without a second hemisphere to compare against.

    Returns
    -------
    (idc_lh, idc_rh), (labels_lh, labels_rh)
        Index arrays (positions into the concatenated label list) and label
        arrays for each hemisphere. All four are ``None`` for single-Gifti
        input (undeterminable).

    Raises
    ------
    ValueError
        If `parcellation` is not one of the supported types.
    """
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


def _avg_dist_mats(D1, D2):
    return (D1 + D2) / 2


def nulls_burt2020(data_1d, dist_mat, n_nulls=1000, seed=None, **kwargs):
    """Variogram-matched surrogate maps (Burt et al. 2020, brainsmash).

    Generates surrogates by permuting ``data_1d`` and smoothing to match its empirical
    variogram, via the ``brainsmash`` package's ``Base`` class. ``dist_mat`` is the
    parcel-by-parcel distance matrix; ``**kwargs`` (e.g. ``resample``, ``batch_size``) are
    forwarded to ``Base``. :cite:`burt2020`.

    Intended to be called through :func:`generate_null_maps`, which handles NaN masking,
    hemisphere splitting, and parallelization across maps — not meant to be called directly.
    """
    data_1d = np.array(data_1d).flatten()
    # results array with shape (n_nulls, n_parcels)
    null_data = np.full((n_nulls, len(data_1d)), np.nan)
    # mask
    mask = _get_null_data_mask(data_1d, dist_mat)
    data_1d = data_1d[mask]
    dist_mat = dist_mat[np.ix_(mask, mask)]
    # default settings
    n = len(data_1d)                                       # true parcel count after masking
    kwargs.setdefault("resample", True)                    # preserves full distribution, False de-means nulls
    batch_size = kwargs.pop("batch_size", 100)             # expose
    # null maps
    null_data[:, mask] = Base(
        x=data_1d,
        D=dist_mat,
        seed=seed,
        **kwargs
    )(n_nulls, batch_size)
    # return
    return null_data.astype(data_1d.dtype)

def nulls_burt2018(data_1d, dist_mat, n_nulls=1000, seed=None, **kwargs):
    """Spatial autoregressive surrogate maps (Burt et al. 2018).

    Generates surrogates via ``brainsmash.utils.batch_surrogates``, which fits a spatial
    autoregressive model relating ``data_1d`` to ``dist_mat`` and samples from it (values are
    Box-Cox-shifted internally to satisfy positivity, then shifted back). :cite:`burt2018`.

    Intended to be called through :func:`generate_null_maps`, which handles NaN masking,
    hemisphere splitting, and parallelization across maps — not meant to be called directly.
    """
    data_1d = np.array(data_1d).flatten()
    # results array with shape (n_nulls, n_parcels)
    null_data = np.full((n_nulls, len(data_1d)), np.nan)
    # mask
    mask = _get_null_data_mask(data_1d, dist_mat)
    data_1d = data_1d[mask]
    dist_mat = dist_mat[np.ix_(mask, mask)]
    # batch_surrogates requires positive values (Box-Cox transform); shift and undo on output
    # so that the returned nulls have the same value distribution as the original input
    shift = np.abs(np.nanmin(data_1d)) + 0.1
    null_data[:, mask] = batch_surrogates(
        dist_mat, data_1d + shift, n_surr=n_nulls, seed=seed, **kwargs).T - shift
    # return
    return null_data.astype(data_1d.dtype)

def _build_variogram_w(data_1d, dist_mat, n_bins=20, kernel="exponential", nugget=False):
    """Fit parametric variogram to data_1d; return covariance kernel matrix W.

    Fits γ(h) to the empirical semi-variogram, converts to C(h), and returns
    the full covariance matrix W[i,j] = C(d[i,j]). Positive-definite by
    Bochner's theorem for the exponential and Gaussian kernels; the spherical
    kernel has compact support (W is naturally sparse beyond its range).

    Parameters
    ----------
    n_bins : int
        Number of quantile-based variogram bins. Default 20.
    kernel : {'exponential', 'gaussian', 'spherical'}
        Variogram model. 'exponential' matches BrainSMASH default and is the
        most robust for parcellated brain maps. Short aliases 'exp', 'gau',
        'sph' are accepted.
    nugget : bool
        If True, fit a nugget term (variance at h→0⁺), accounting for
        measurement noise or fine-scale variance not captured by the model.
        Adds one free parameter to the fit.
    """
    from scipy.optimize import curve_fit

    _ALIASES = {"exp": "exponential", "gau": "gaussian", "sph": "spherical"}
    kernel = _ALIASES.get(kernel.lower(), kernel.lower())
    if kernel not in ("exponential", "gaussian", "spherical"):
        raise ValueError(
            f"kernel must be 'exponential', 'gaussian', or 'spherical'; got '{kernel}'"
        )

    x = data_1d - data_1d.mean()
    triu_i, triu_j = np.triu_indices(len(x), k=1)
    h = dist_mat[triu_i, triu_j]
    sv = (x[triu_i] - x[triu_j]) ** 2 / 2.0

    # quantile-based bins → approximately equal pair count per bin
    edges = np.percentile(h, np.linspace(0, 100, n_bins + 1))
    bh, bg = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (h >= lo) & (h < hi)
        if mask.sum() >= 3:
            bh.append(h[mask].mean())
            bg.append(sv[mask].mean())
    bh = np.asarray(bh, dtype=float)
    bg = np.asarray(bg, dtype=float)

    sill0 = float(np.var(x))
    gt63 = bg > 0.63 * sill0
    rng0 = float(bh[gt63][0]) if gt63.any() else float(bh.mean())

    def _norm_cov(h, rng):
        if kernel == "exponential":
            return np.exp(-h / rng)
        elif kernel == "gaussian":
            return np.exp(-(h / rng) ** 2)
        else:  # spherical — compact support at rng
            r = np.minimum(h / rng, 1.0)
            return np.where(h <= rng, 1.0 - 1.5 * r + 0.5 * r ** 3, 0.0)

    if nugget:
        def _var_model(h, nug, sill, rng):
            return nug + (sill - nug) * (1.0 - _norm_cov(h, rng))
        p0 = [0.0, sill0, rng0]
        bounds = ([0.0, 0.0, 1e-3], [sill0, 10.0 * sill0 + 1e-9, 1e9])
    else:
        def _var_model(h, sill, rng):
            return sill * (1.0 - _norm_cov(h, rng))
        p0 = [sill0, rng0]
        bounds = ([0.0, 1e-3], [10.0 * sill0 + 1e-9, 1e9])

    try:
        popt, _ = curve_fit(_var_model, bh, bg, p0=p0, bounds=bounds, maxfev=2000)
    except Exception:
        popt = p0

    if nugget:
        nug_fit, sill_fit, rng_fit = popt
    else:
        nug_fit, sill_fit, rng_fit = 0.0, popt[0], popt[1]

    W = (sill_fit - nug_fit) * _norm_cov(dist_mat, rng_fit)
    np.fill_diagonal(W, sill_fit)
    return np.maximum(W, 0.0)


def nulls_moran(data_1d, dist_mat, n_nulls=1000, seed=None, **kwargs):
    """Moran Spectral Randomization (MSR) surrogate maps (Wagner & Dray 2015).

    Generates surrogates via BrainSpace's ``MoranRandomization``, using a spatial weight
    matrix ``W`` built from ``dist_mat`` (standard 1/d weights by default, or a
    variogram-fitted covariance kernel if ``fit_variogram=True``, falling back to 1/d if the
    map's own Moran's I is below ``variogram_threshold``). Notable ``**kwargs``:
    ``procedure`` (default ``"singleton"``), ``joint``, ``n_components`` (default 15),
    ``fit_variogram``, ``variogram_n_bins``/``variogram_kernel``/``variogram_nugget``/``variogram_threshold``.
    :cite:`wagner2015` (original MSR method); :cite:`vos_de_wael2020` (BrainSpace implementation).

    Intended to be called through :func:`generate_null_maps`, which handles NaN masking,
    hemisphere splitting, and parallelization across maps — not meant to be called directly.
    """
    data_1d = np.array(data_1d).flatten()
    # results array with shape (n_nulls, n_parcels)
    null_data = np.full((n_nulls, len(data_1d)), np.nan)
    # mask
    mask = _get_null_data_mask(data_1d, dist_mat)
    data_1d = data_1d[mask]
    dist_mat = dist_mat[np.ix_(mask, mask)]
    # weight matrix W: either standard 1/d or variogram-fitted covariance kernel
    fit_variogram = kwargs.pop("fit_variogram", False)
    variogram_n_bins     = kwargs.pop("variogram_n_bins",     20)
    variogram_kernel     = kwargs.pop("variogram_kernel",     "exponential")
    variogram_nugget     = kwargs.pop("variogram_nugget",     False)
    variogram_threshold  = kwargs.pop("variogram_threshold",  0)
    if fit_variogram and variogram_threshold is not None:
        # compute Moran's I with standard 1/d weights to decide whether the map
        # has enough SA to make variogram fitting meaningful
        dm_w = np.where(dist_mat == 0, np.inf, dist_mat)
        dm_w = (1.0 / dm_w)
        np.fill_diagonal(dm_w, 0.0)
        xc = data_1d - data_1d.mean()
        S0 = dm_w.sum()
        morans_i = float(len(data_1d) / S0) * float(xc @ dm_w @ xc) / float(xc @ xc)
        if morans_i < variogram_threshold:
            lgr.debug("Moran's I=%.3f < threshold=%.3f; falling back to 1/d W",
                      morans_i, variogram_threshold)
            fit_variogram = False
    if fit_variogram:
        # W[i,j] = C(d[i,j]) fitted to the map's own SA scale.
        # MEMs are the KL eigenbasis of the map → larger effective K for smooth maps
        # → better-calibrated FPR at high alpha relative to fixed 1/d.
        W = _build_variogram_w(data_1d, dist_mat,
                               n_bins=variogram_n_bins,
                               kernel=variogram_kernel,
                               nugget=variogram_nugget)
    else:
        np.fill_diagonal(dist_mat, 1)
        dist_mat **= -1
        W = dist_mat
    # null maps
    # procedure='singleton', n_components=15: GRF benchmark shows no single K is optimal
    # across all SA levels. K=15 is a reasonable default for singleton (calibrated at alpha=1–2,
    # marginally anti-conservative at alpha=3). Override via maps_procedure / maps_n_components kwargs.
    null_data[:, mask] = MoranRandomization(
        procedure=kwargs.pop("procedure", "singleton"),
        joint=kwargs.pop("joint", True),
        n_components=kwargs.pop("n_components", 15),
        seed=seed,
        n_nulls=n_nulls,
        **kwargs
    ).fit(W).randomize(data_1d)
    # return
    return null_data.astype(data_1d.dtype)

def nulls_variomoran(data_1d, dist_mat, n_nulls=1000, seed=None, **kwargs):
    """Variogram-adapted Moran Spectral Randomization (varioMSR).

    Wrapper around :func:`nulls_moran` with ``fit_variogram=True`` as default.
    Fits an empirical variogram to ``data_1d``, builds a covariance kernel W tuned
    to the map's own SA scale, and uses the resulting MEMs as the randomization basis.
    Falls back to standard 1/d W when Moran's I ≤ ``variogram_threshold`` (default 0).

    All :func:`nulls_moran` kwargs are forwarded; ``fit_variogram`` and
    ``variogram_*`` kwargs can be overridden in the usual way via ``maps_*`` prefixes.
    """
    kwargs.setdefault("fit_variogram", True)
    return nulls_moran(data_1d, dist_mat, n_nulls=n_nulls, seed=seed, **kwargs)


def nulls_random(data_1d, dist_mat=None, n_nulls=1000, seed=None):
    """Fully random (spatially unconstrained) null maps.

    Generates each null as an independent full permutation of the non-NaN values of
    ``data_1d`` (no resampling with replacement). ``dist_mat`` is accepted but never used —
    it exists only so this function shares a call signature with the other ``nulls_*``
    functions for uniform dispatch inside :func:`generate_null_maps`.

    Intended to be called through :func:`generate_null_maps` (``method="random"``), which
    handles parallelization across maps — not meant to be called directly.
    """
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

_DISTMAT_METHODS = {"moran", "msr", "variomoran", "variomsr", "burt2018", "burt2020"}
_SPIN_METHODS = {"alexander_bloch", "spin", "vasa", "hungarian", "cornblath", "baum"}
_DISTMAT_FREE_METHODS = {"random"}  # methods that never need a distance matrix

_SPIN_METHOD_MAP = {
    "alexander_bloch": "original",
    "spin": "cornblath",
    "vasa": "vasa",
    "hungarian": "hungarian",
    "baum": "baum",
    "cornblath": "cornblath",
}

_NULL_METHODS = {
    # Random
    "random": nulls_random,
    # Moran Spectral Randomization (MSR) via BrainSpace → volumetric and surface
    "moran": nulls_moran,
    "msr": nulls_moran,           # alias: Moran Spectral Randomization
    "brainspace": nulls_moran,    # legacy alias
    # Variogram-adapted MSR (varioMSR): fits W to the map's own SA scale
    "variomoran": nulls_variomoran,
    "variomsr": nulls_variomoran,  # alias: variogram-adapted MSR
    # Variogram-method implemented via Brainsmash -> volumetric and surface
    "burt2020": nulls_burt2020,
    "brainsmash": nulls_burt2020,
    "variogram": nulls_burt2020,
    # Smoothing-method from Burt2018 -> volumetric and surface
    "burt2018": nulls_burt2018,
    # Spin tests -> surface only (handled via spin code path)
    "alexander_bloch": None,
    "spin": None,
    "vasa": None,
    "hungarian": None,
    "baum": None,
    "cornblath": None,
}

# Canonical names for aliases — normalised at parse time so cache keys are stable
_NULL_METHOD_ALIASES = {
    "spin": "cornblath",
    "msr": "moran",
    "brainspace": "moran",
    "variomsr": "variomoran",
    "brainsmash": "burt2020",
    "variogram": "burt2020",
}


def _parse_null_method(method):
    """Parse and canonicalise null method to ``(cx_method, sc_method)`` or ``(method, None)``.

    Accepts:
    - ``str``: single method → ``(method, None)``
    - ``"cx+sc"`` shorthand, e.g. ``"spin+moran"`` → ``("spin", "moran")``
    - ``tuple[str, str]``: ``(cx_method, sc_method)`` → returned as-is

    All components are normalised through ``_NULL_METHOD_ALIASES`` so that aliases
    (e.g. ``"spin"`` / ``"cornblath"``) map to the same canonical name and
    do not cause spurious cache invalidation.
    """
    def _canon(m):
        return _NULL_METHOD_ALIASES.get(m, m) if m is not None else None

    if isinstance(method, tuple) and len(method) == 2:
        return (_canon(method[0]), _canon(method[1]))
    if isinstance(method, str) and "+" in method:
        parts = method.split("+", 1)
        return (_canon(parts[0]), _canon(parts[1]))
    return (_canon(method), None)


def _get_surface_atlas(parc_space, density):
    """Return (atlas_dict, surf_key) for a supported surface space."""
    from .datasets import fetch_template
    if "fsa" in parc_space.lower():
        space, surf_key = "fsaverage", "pial"
    elif "fslr" in parc_space.lower():
        space, surf_key = "fsLR", "midthickness"
    else:
        lgr.critical_raise(
            f"Surface space '{parc_space}' not supported. Use 'fsaverage' or 'fsLR'.",
            ValueError,
        )
    atlas = {}
    for desc in [surf_key, "sphere", "medial"]:
        L, R = fetch_template(space, desc=desc, res=density, check_file_hash=False, verbose=False)
        atlas[desc] = _SurfPair(L, R)
    return atlas, surf_key


def _gen_spinsamples_parallel(coords, hemiid, n_perm, seed=None, n_proc=1, out=None,
                               batch_size=None):
    """Wrapper for ``gen_spinsamples`` with optional process-based parallelism.

    Splits ``n_perm`` across ``n_proc`` processes, each with an independent
    child seed from ``seed`` via ``np.random.SeedSequence``.  Results are
    statistically equivalent (i.i.d. rotations) but not byte-identical to a
    single seeded call when n_proc > 1.

    *batch_size* controls how many permutations each batch contains.
    Smaller batches → more batches → finer-grained tqdm progress.  Defaults to
    ``ceil(n_perm / n_workers)`` (one batch per worker) when not set.

    If *out* is provided (pre-allocated ``(n_vert, n_perm)`` F-contiguous
    array or memmap), workers write directly to non-overlapping column slices.
    """
    from joblib import effective_n_jobs
    n_workers = effective_n_jobs(n_proc)  # resolves -1/-2/etc. to actual core count
    if n_workers == 1:
        result = gen_spinsamples(
            coords, hemiid, n_rotate=n_perm, method="original",
            check_duplicates=False, seed=seed, verbose=True,
        )
        if out is not None:
            out[:] = result
            return out
        return result

    n_batches = (-(-n_perm // batch_size) if batch_size  # ceil division
                 else n_workers)
    seq = np.random.SeedSequence(seed)
    child_seeds = [int(s.generate_state(1)[0]) for s in seq.spawn(n_batches)]
    batches = [b for b in np.array_split(np.arange(n_perm), n_batches) if len(b) > 0]

    def _run_batch(b, s):
        res = gen_spinsamples(
            coords, hemiid, n_rotate=len(b), method="original",
            check_duplicates=False, seed=s,
        )
        if out is not None:
            out[:, b[0]: b[-1] + 1] = res  # write to non-overlapping column slice
            return len(b)
        return res

    collected = []
    with tqdm(total=n_perm, desc="Generating spins") as pbar:
        for r in Parallel(n_jobs=n_workers, prefer="processes",
                          return_as="generator_unordered")(
            delayed(_run_batch)(b, s) for b, s in zip(batches, child_seeds)
        ):
            if isinstance(r, int):   # out is not None: worker returned batch size
                pbar.update(r)
            else:                    # out is None: worker returned result array
                collected.append(r)
                pbar.update(r.shape[1])

    if out is not None:
        return out
    return np.concatenate(collected, axis=1)


def _spin_parcels_parallel(surfaces, parcellation, n_perm, seed=None, n_proc=1):
    """Wrapper for ``spin_parcels`` (Baum) with optional thread-based parallelism."""
    from joblib import effective_n_jobs
    n_workers = effective_n_jobs(n_proc)
    if n_workers == 1:
        return spin_parcels(
            surfaces, parcellation, n_rotate=n_perm, seed=seed, check_duplicates=False)
    seq = np.random.SeedSequence(seed)
    child_seeds = [int(s.generate_state(1)[0]) for s in seq.spawn(n_workers)]
    batches = np.array_split(np.arange(n_perm), n_workers)
    results = Parallel(n_jobs=n_workers, prefer="threads")(
        delayed(spin_parcels)(
            surfaces, parcellation, n_rotate=len(b), seed=s, check_duplicates=False,
        )
        for b, s in zip(batches, child_seeds) if len(b) > 0
    )
    return np.concatenate(results, axis=1)


def generate_spins(parc, parc_space, n_perm=1000, method="original", seed=None,
                   parc_hemi=None):
    """Generate spin resampling indices for a surface parcellation.

    Supports bilateral (tuple of two GiftiImages) and single-hemisphere
    (single GiftiImage) parcellations.

    Parameters
    ----------
    parc : tuple of two GiftiImage, or a single GiftiImage
        Bilateral (lh_img, rh_img) or single-hemisphere surface parcellation.
    parc_space : str
        Surface space of ``parc`` (e.g. ``"fsaverage"``, ``"fsLR"``).
    n_perm : int, default=1000
        Number of spin permutations to generate.
    method : str, default="original"
        Rotation-generation method forwarded to ``neuromaps``' ``gen_spinsamples``. One of
        ``"original"`` (Alexander-Bloch method), ``"vasa"``, ``"hungarian"``, or
        ``"cornblath"`` (:cite:`alexander_bloch2018`, :cite:`vasa2018`, :cite:`kuhn1955`,
        :cite:`cornblath2020`).
    seed : int, optional
        Random seed for reproducibility.
    parc_hemi : list of str, optional
        Which hemisphere a single (unilateral) ``parc`` belongs to (``["L"]`` or ``["R"]``);
        defaults to ``"L"`` with a warning if not given. Not needed for a bilateral ``parc``.

    Returns
    -------
    spins_lh, spins_rh : ndarray of int32
        Spin indices. For bilateral parcellations both have shape ``(n_parcels_hemi, n_perm)``.
        For a single-hemisphere parcellation the unused hemisphere gets shape ``(0, n_perm)``.
        RH indices are local to ``[0, n_rh)``.

    Notes
    -----
    Typically invoked through :func:`generate_null_maps` for spin-based null methods, but also
    called directly (e.g. by ``NiSpace``/``Parcellation``) when precomputing or caching a spin
    matrix ahead of repeated use.
    """
    is_bilateral = isinstance(parc, tuple)
    is_unilateral = isinstance(parc, nib.GiftiImage)

    if not (is_bilateral or is_unilateral):
        lgr.critical_raise(
            "Spin tests require a surface parcellation: either a bilateral "
            "(lh_img, rh_img) tuple or a single GiftiImage.",
            ValueError,
        )

    density = _img_density_for_neuromaps(parc)
    atlas, _ = _get_surface_atlas(parc_space, density)
    spheres = atlas["sphere"]

    if is_bilateral:
        centroids, hemiid = get_parcel_centroids(
            surfaces=(spheres[0], spheres[1]),
            parcellation=(parc[0], parc[1]),
            method="surface",
        )
    else:
        # single hemisphere — determine which side
        keep_hemi = None
        if parc_hemi is not None:
            h = parc_hemi[0] if isinstance(parc_hemi, (list, tuple)) else parc_hemi
            keep_hemi = h if h in ("L", "R") else None
        if keep_hemi is None:
            keep_hemi = "L"
            lgr.warning("generate_spins: parc_hemi not specified for unilateral image; assuming 'L'.")

        centroids = find_surf_parc_centroids(
            parc, parc_space=parc_space, parc_hemi=[keep_hemi], parc_density=density,
        )
        # hemiid: 0 = LH, 1 = RH — tells gen_spinsamples the geometry of the rotation
        hemiid = np.zeros(len(centroids), dtype=int) if keep_hemi == "L" \
            else np.ones(len(centroids), dtype=int)

    spins = gen_spinsamples(
        coords=centroids,
        hemiid=hemiid,
        n_rotate=n_perm,
        method=method,
        seed=seed,
        verbose=False,
    )

    n_lh = int((hemiid == 0).sum())
    spins_lh = spins[:n_lh, :].astype(np.int32)
    spins_rh = (spins[n_lh:, :] - n_lh).astype(np.int32)
    return spins_lh, spins_rh


def generate_baum_spins(parc, parc_space, n_perm=1000, seed=None, n_proc=1):
    """Generate Baum-method spin matrix via vertex-level rotation + modal parcel assignment.

    Returns ``(spins_lh, spins_rh)`` int32 arrays of shape ``(n_lh_parcels, n_perm)`` /
    ``(n_rh_parcels, n_perm)``.  Values are parcel indices (0-based, local to each hemisphere);
    -1 indicates a parcel fully absorbed by the medial wall after rotation.
    """
    if not isinstance(parc, tuple):
        lgr.critical_raise(
            "generate_baum_spins requires a bilateral surface parcellation tuple (lh_img, rh_img).",
            ValueError)
    parc_lh, parc_rh = parc
    density = _img_density_for_neuromaps(parc_lh)
    atlas, _ = _get_surface_atlas(parc_space, density)
    spheres = atlas["sphere"]

    # spin_parcels generates vertex-level spins and assigns each parcel the modal label
    # check_duplicates=False: vertex-level coords make duplicates physically impossible
    regions = _spin_parcels_parallel(
        surfaces=(spheres.L, spheres.R),
        parcellation=(parc_lh, parc_rh),
        n_perm=n_perm,
        seed=seed,
        n_proc=n_proc,
    )  # (n_parcels_total, n_perm), global 0-based indices, -1 = dropped

    n_lh = len(np.unique(parc_lh.agg_data())) - 1  # subtract background label 0
    spins_lh = regions[:n_lh, :].astype(np.int32)  # already LH-local (0..n_lh-1)
    raw_rh = regions[n_lh:, :]
    spins_rh = np.where(raw_rh >= 0, raw_rh - n_lh, -1).astype(np.int32)
    return spins_lh, spins_rh



def _build_cornblath_T_batch(spins_path, spins_shape, n_vert_lh,
                             labels_lh, labels_rh, src_counts_lh, src_counts_rh,
                             n_lh, n_rh, T_lh_path, T_rh_path, dtype,
                             k_start, k_end, normalize=False):
    """Worker: build T matrices for permutations [k_start, k_end) from memmaps."""
    n_perm = spins_shape[1]
    spins = np.memmap(spins_path, dtype=np.int32, mode="r", shape=spins_shape, order="F")
    T_lh = np.memmap(T_lh_path, dtype=dtype, mode="r+", shape=(n_perm, n_lh, n_lh))
    T_rh = np.memmap(T_rh_path, dtype=dtype, mode="r+", shape=(n_perm, n_rh, n_rh))
    for k in range(k_start, k_end):
        spun_lh = labels_lh[spins[:n_vert_lh, k]]
        valid = (spun_lh > 0) & (labels_lh > 0)
        s, d = spun_lh[valid] - 1, labels_lh[valid] - 1
        np.add.at(T_lh[k], (d, s), 1.0 / src_counts_lh[s])
        if normalize:
            t = T_lh[k].astype(np.float32)
            cs = t.sum(axis=0, keepdims=True)
            T_lh[k] = np.where(cs > 0, t / cs, 0.0).astype(dtype)
        spun_rh = labels_rh[spins[n_vert_lh:, k] - n_vert_lh]
        valid = (spun_rh > 0) & (labels_rh > 0)
        s, d = spun_rh[valid] - 1, labels_rh[valid] - 1
        np.add.at(T_rh[k], (d, s), 1.0 / src_counts_rh[s])
        if normalize:
            t = T_rh[k].astype(np.float32)
            cs = t.sum(axis=0, keepdims=True)
            T_rh[k] = np.where(cs > 0, t / cs, 0.0).astype(dtype)
    T_lh.flush()
    T_rh.flush()
    return k_end - k_start


def generate_cornblath_mat(parc, parc_space, n_perm=1000, seed=None, n_proc=1,
                           dtype=np.float32, memmap_dir=None, batch_size=100,
                           normalize=False):
    """Generate Cornblath fractional transition matrices.

    For each rotation k, ``T[k, j, i]`` = fraction of parcel i's vertices that land in
    parcel j.  When all of parcel i's vertices rotate into the medial wall,
    ``T[k, :, i].sum() == 0`` and the corresponding null value is set to NaN at application.

    Returns ``(T_lh, T_rh)`` arrays of shape
    ``(n_perm, n_lh_parcels, n_lh_parcels)`` / ``(n_perm, n_rh_parcels, n_rh_parcels)``.
    With *memmap_dir* these are ``np.memmap``-backed arrays (file-backed, low RAM footprint).

    Parameters
    ----------
    memmap_dir : path-like or None
        Directory for temporary memmap files.  When set, vertex spin indices and both T
        matrices are kept on disk rather than in RAM.  The vertex spin file is deleted
        after the T-matrix loop; the T-matrix files persist until the caller deletes them
        (or the directory is cleaned up, e.g. via ``tempfile.TemporaryDirectory``).
        Recommended for large parcellations (n_lh > 200) or high n_perm (> 5000).
        Required for multi-process parallelism (``n_proc > 1``).
    """
    import os
    import tempfile

    if not isinstance(parc, tuple):
        lgr.critical_raise(
            "generate_cornblath_mat requires a bilateral surface parcellation tuple (lh_img, rh_img).",
            ValueError)
    parc_lh, parc_rh = parc

    density = _img_density_for_neuromaps(parc_lh)
    atlas, _ = _get_surface_atlas(parc_space, density)
    spheres = atlas["sphere"]

    # vertex-level coordinates
    coords, hemiid = get_parcel_centroids(
        surfaces=(spheres.L, spheres.R), method="surface")
    n_vert = len(coords)
    n_vert_lh = int((hemiid == 0).sum())

    # allocate vertex spin index array — F-contiguous so column reads ([:, k]) are fast
    if memmap_dir is not None:
        _spins_fd, _spins_path = tempfile.mkstemp(suffix=".spins.dat", dir=memmap_dir)
        os.close(_spins_fd)
        all_spins = np.memmap(_spins_path, dtype=np.int32, mode="w+",
                              shape=(n_vert, n_perm), order="F")
    else:
        all_spins = None  # returned by _gen_spinsamples_parallel

    all_spins = _gen_spinsamples_parallel(
        coords=coords, hemiid=hemiid, n_perm=n_perm, seed=seed, n_proc=n_proc,
        out=all_spins, batch_size=batch_size,
    )  # (n_vert_total, n_perm)

    vert_spins_lh = all_spins[:n_vert_lh, :]  # view — no copy

    # vertex → parcel label arrays (0 = medial wall, 1..n_parc = parcel, 1-based global)
    labels_lh = parc_lh.agg_data().astype(int)
    labels_rh_global = parc_rh.agg_data().astype(int)
    n_lh = len(np.unique(labels_lh)) - 1
    n_rh = len(np.unique(labels_rh_global)) - 1
    labels_rh = np.where(labels_rh_global > 0, labels_rh_global - n_lh, 0)

    # source parcel vertex counts (denominator)
    src_counts_lh = np.bincount(labels_lh[labels_lh > 0], minlength=n_lh + 1)[1:].astype(dtype)
    src_counts_rh = np.bincount(labels_rh[labels_rh > 0], minlength=n_rh + 1)[1:].astype(dtype)
    src_counts_lh[src_counts_lh == 0] = 1.0
    src_counts_rh[src_counts_rh == 0] = 1.0

    # allocate T matrices
    if memmap_dir is not None:
        _T_lh_fd, _T_lh_path = tempfile.mkstemp(suffix=".T_lh.dat", dir=memmap_dir)
        _T_rh_fd, _T_rh_path = tempfile.mkstemp(suffix=".T_rh.dat", dir=memmap_dir)
        os.close(_T_lh_fd); os.close(_T_rh_fd)
        T_lh = np.memmap(_T_lh_path, dtype=dtype, mode="w+", shape=(n_perm, n_lh, n_lh))
        T_rh = np.memmap(_T_rh_path, dtype=dtype, mode="w+", shape=(n_perm, n_rh, n_rh))
    else:
        T_lh = np.zeros((n_perm, n_lh, n_lh), dtype=dtype)
        T_rh = np.zeros((n_perm, n_rh, n_rh), dtype=dtype)

    from joblib import effective_n_jobs
    n_workers = effective_n_jobs(n_proc)

    if n_workers > 1 and memmap_dir is not None:
        n_batches = (-(-n_perm // batch_size) if batch_size else n_workers)
        batches = [b for b in np.array_split(np.arange(n_perm), n_batches) if len(b) > 0]
        with tqdm(total=n_perm, desc="T-matrix") as pbar:
            for r in Parallel(n_jobs=n_workers, prefer="processes",
                              return_as="generator_unordered")(
                delayed(_build_cornblath_T_batch)(
                    _spins_path, (n_vert, n_perm), n_vert_lh,
                    labels_lh, labels_rh, src_counts_lh, src_counts_rh,
                    n_lh, n_rh, _T_lh_path, _T_rh_path, dtype,
                    int(b[0]), int(b[-1]) + 1, normalize,
                )
                for b in batches
            ):
                pbar.update(r)
    else:
        for k in tqdm(range(n_perm), desc="T-matrix"):
            src_lh = labels_lh[vert_spins_lh[:, k]]
            valid = (src_lh > 0) & (labels_lh > 0)
            s, d = src_lh[valid] - 1, labels_lh[valid] - 1
            np.add.at(T_lh[k], (d, s), 1.0 / src_counts_lh[s])
            if normalize:
                t = T_lh[k].astype(np.float32)
                cs = t.sum(axis=0, keepdims=True)
                T_lh[k] = np.where(cs > 0, t / cs, 0.0).astype(dtype)
            src_rh = labels_rh[all_spins[n_vert_lh:, k] - n_vert_lh]
            valid = (src_rh > 0) & (labels_rh > 0)
            s, d = src_rh[valid] - 1, labels_rh[valid] - 1
            np.add.at(T_rh[k], (d, s), 1.0 / src_counts_rh[s])
            if normalize:
                t = T_rh[k].astype(np.float32)
                cs = t.sum(axis=0, keepdims=True)
                T_rh[k] = np.where(cs > 0, t / cs, 0.0).astype(dtype)

    # release vertex spin array and delete its backing file (no longer needed)
    if memmap_dir is not None:
        del vert_spins_lh, all_spins
        os.unlink(_spins_path)
        T_lh.flush()
        T_rh.flush()

    return T_lh, T_rh


def apply_cornblath_mat(data_1d, T_lh, T_rh, idc_lh, idc_rh, n_perm=None):
    """Apply precomputed Cornblath transition matrices to a 1D data array.

    ``T_lh`` / ``T_rh``: float32 ``(n_perm, n_parc_hemi, n_parc_hemi)`` as returned by
    :func:`generate_cornblath_mat`.  If ``n_perm`` is less than ``T_lh.shape[0]``, only
    the first ``n_perm`` rotations are used.

    Returns ``null_data`` of shape ``(n_perm, n_parcels)``.
    """
    if n_perm is None:
        n_perm = T_lh.shape[0]
    T_lh = T_lh[:n_perm]
    T_rh = T_rh[:n_perm]
    idc_lh = np.asarray(idc_lh, dtype=int)
    idc_rh = np.asarray(idc_rh, dtype=int)
    data_lh = data_1d[idc_lh].astype(np.float32)
    data_rh = data_1d[idc_rh].astype(np.float32)

    def _apply_hemi(T, d):
        """Apply T to d, masking out NaN input parcels by re-normalising each row."""
        nan_mask = np.isnan(d)
        if nan_mask.any():
            T = T.copy()
            T[:, :, nan_mask] = 0.0          # zero weight for NaN-input parcels
            d = np.where(nan_mask, 0.0, d)
            row_sums = T.sum(axis=2, keepdims=True)   # (n_perm, n_parc, 1)
            np.divide(T, row_sums, out=T, where=row_sums != 0)
        null = np.einsum("kij,j->ki", T, d)  # (n_perm, n_parc)
        if nan_mask.any():
            # parcels whose full weight came from NaN inputs → all-zero row after zero-fill → NaN
            null[T.sum(axis=2) == 0] = np.nan
        return T, null

    T_lh, null_lh = _apply_hemi(T_lh, data_lh)
    T_rh, null_rh = _apply_hemi(T_rh, data_rh)

    # parcels where all source vertices rotated to medial wall → column sum == 0 → NaN
    null_lh[T_lh.sum(axis=2) == 0] = np.nan
    null_rh[T_rh.sum(axis=2) == 0] = np.nan

    null_data = np.full((n_perm, len(data_1d)), np.nan, dtype=data_1d.dtype)
    null_data[:, idc_lh] = null_lh
    null_data[:, idc_rh] = null_rh
    return null_data


def apply_spins(data_1d, spins_lh, spins_rh, idc_lh, idc_rh, n_perm=None):
    """Apply precomputed spin indices to a 1D data array.

    Parameters
    ----------
    data_1d : array-like
        1D array of parcel values, length ``n_parcels``.
    spins_lh, spins_rh : ndarray of int
        Spin indices as returned by :func:`generate_spins`/``generate_baum_spins``, shape
        ``(n_lh_parcels, n_perm)`` / ``(n_rh_parcels, n_perm)``. ``-1`` entries (Baum-method
        parcels absorbed by the medial wall after rotation) are handled by setting those
        output positions to NaN.
    idc_lh, idc_rh : array-like of int
        Positions in ``data_1d`` corresponding to each hemisphere's parcels, in the same order
        ``spins_lh``/``spins_rh`` index into.
    n_perm : int, optional
        Number of permutations to apply; defaults to ``spins_lh.shape[1]`` (use all).

    Returns
    -------
    null_data : ndarray
        Shape ``(n_perm, n_parcels)``. Positions not covered by ``idc_lh``/``idc_rh`` (e.g.
        subcortex) remain NaN.

    Notes
    -----
    Intended to be called through :func:`generate_null_maps`, which calls this once per data
    row as part of its spin-test code path — not meant to be called directly in most cases.
    """
    if n_perm is None:
        n_perm = spins_lh.shape[1]
    n_parcels = len(data_1d)
    null_data = np.full((n_perm, n_parcels), np.nan, dtype=data_1d.dtype)
    idc_lh = np.asarray(idc_lh, dtype=int)
    idc_rh = np.asarray(idc_rh, dtype=int)
    data_lh = data_1d[idc_lh]
    data_rh = data_1d[idc_rh]
    has_neg = spins_lh.min() < 0 or spins_rh.min() < 0
    for k in range(n_perm):
        if has_neg:
            m = spins_lh[:, k] >= 0
            v = data_lh[np.where(m, spins_lh[:, k], 0)]
            v[~m] = np.nan
            null_data[k, idc_lh] = v
            m = spins_rh[:, k] >= 0
            v = data_rh[np.where(m, spins_rh[:, k], 0)]
            v[~m] = np.nan
            null_data[k, idc_rh] = v
        else:
            null_data[k, idc_lh] = data_lh[spins_lh[:, k]]
            null_data[k, idc_rh] = data_rh[spins_rh[:, k]]
    return null_data


def get_distance_matrix(parc, parc_space, parc_hemi=["L", "R"],
                        parc_resample=2, centroids=False, surf_euclidean=False,
                        n_proc=1, verbose=True, dtype=np.float32):
    """Compute a parcel-by-parcel distance matrix for a volumetric or surface parcellation.

    Dispatches on `parc_space`: MNI/volumetric parcellations get a euclidean
    distance matrix (voxel-to-voxel mean, or centroid-to-centroid if
    `centroids=True`, via :func:`find_vol_parc_centroids`); fsaverage/fsLR
    surface parcellations get a geodesic (mesh-surface) distance matrix by
    default, or a euclidean centroid-to-centroid one if `surf_euclidean=True`
    (via :func:`find_surf_parc_centroids`). `parc_resample` triggers a
    parcel-loss-guarded resampling pass first (skipped with a warning if it
    would drop any parcel). Used by ``api.py`` and
    ``core/parcellation.py``'s ``Parcellation.get_dist_mat`` path (see
    [[project_distance_matrices]]) whenever no precomputed distance matrix is
    available.

    Parameters
    ----------
    parc : image-like or tuple
        Volumetric parcellation image, or ``(lh_img, rh_img)`` tuple of
        surface GiftiImages.
    parc_space : str
        Reference space; must contain ``"mni"`` for the volumetric path, or
        be one of ``"fsaverage"``/``"fsLR"``/``"fsa"``/``"fslr"`` for the
        surface path.
    parc_hemi : list of str, default=["L", "R"]
        Hemispheres present (surface path only).
    parc_resample : int, str, or bool, default=2
        Volumetric: target voxel size in mm (``True`` -> 3mm). Surface:
        target density string (e.g. ``"32k"``). Falsy disables resampling.
    centroids : bool, default=False
        Volumetric path: use centroid-to-centroid distances instead of mean
        voxel-to-voxel distances.
    surf_euclidean : bool, default=False
        Surface path: use euclidean centroid-to-centroid distances instead
        of the default geodesic mesh distance.
    n_proc : int, default=1
        Number of parallel jobs for the (per-parcel or per-hemisphere) loop.
    verbose : bool, default=True
        Show progress bars / info logging.
    dtype : default=np.float32
        Output distance matrix dtype.

    Returns
    -------
    np.ndarray or tuple of np.ndarray
        A single 2D distance matrix for a volumetric parcellation; a tuple
        of one 2D matrix per hemisphere for a surface parcellation.
    """
    verbose = set_log(lgr, verbose)

    ## generate distance matrix
    # case volumetric
    if "mni" in parc_space.lower():
        # get parcellation data
        parc = load_nifti(parc)
        if parc_resample and not isinstance(parc_resample, str):
            if parc_resample is True:
                parc_resample = 3
            current_voxsize = abs(round(parc.affine[0, 0]))
            lgr.info(f"Resampling volumetric parcellation from {current_voxsize}mm to {parc_resample}mm "
                      "for distance matrix generation.")
            ids_before = set(np.trim_zeros(np.unique(parc.get_fdata())))
            parc_resampled = resample_img(
                parc,
                target_affine=np.diag([parc_resample] * 3),
                interpolation="nearest",
                force_resample=True, copy_header=True
            )
            lost = ids_before - set(np.trim_zeros(np.unique(parc_resampled.get_fdata())))
            if lost:
                lgr.warning(
                    f"Resampling to {parc_resample}mm voxels would drop {len(lost)} parcel(s) "
                    f"(IDs: {sorted(int(i) for i in lost)}). Skipping downsampling and using "
                    f"original {current_voxsize}mm resolution."
                )
            else:
                parc = parc_resampled
        parc_data = parc.get_fdata()
        parc_affine = parc.affine
        parcels = np.trim_zeros(np.unique(parc_data))
        n_parcels = len(parcels)
        mask = np.logical_not(np.logical_or(np.isclose(parc_data, 0), np.isnan(parc_data)))
        parc_data_m = parc_data * mask

        # case distances between volumetric parcel centroids
        if centroids:
            lgr.info(f"Estimating euclidean distance matrix: {n_parcels} volumetric parcels, centroid mode.")
            # get centroids
            ijk = find_vol_parc_centroids(parc_data_m, affine=parc_affine, parcel_idc=parcels)
            # get distances
            dist = _dist_mat_from_coords(ijk, dtype)

        # case mean distances between parcel-to-parcel voxels
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
                    dist_i[j] = \
                        cdist(ijk_parcels[i_parcel], ijk_parcels[parcels[j]]).mean().astype(dtype)
                    j += 1
                return dist_i

            lgr.info(f"Estimating euclidean distance matrix: {n_parcels} volumetric parcels, "
                     f"voxel-to-voxel mode, {n_proc} proc.")
            dist_list = Parallel(n_jobs=n_proc)(
                delayed(mni_dist)(i, i_parcel) for i, i_parcel in enumerate(tqdm(
                    parcels,
                    desc=f"Distance matrix ({n_proc} proc)", disable=not verbose
                ))
            )
            dist = np.r_[dist_list]
            # mirror to lower triangle
            dist = dist + dist.T
            # zero diagonal
            np.fill_diagonal(dist, 0)
    
    # case surface
    elif parc_space in ["fsaverage", "fsLR", "fsa", "fslr"]:

        if parc_resample and isinstance(parc_resample, str):
            from neuromaps.transforms import fsaverage_to_fsaverage, fslr_to_fslr
            current_density = _img_density_for_neuromaps(parc[0] if isinstance(parc, tuple) else parc)
            if current_density != parc_resample:
                lgr.info(f"Resampling surface parcellation from {current_density} to {parc_resample} density "
                         "for distance matrix generation.")
                resample_fn = fsaverage_to_fsaverage if "fsa" in parc_space.lower() else fslr_to_fslr

                def _parc_ids(p):
                    imgs = p if isinstance(p, tuple) else (p,)
                    return set(np.trim_zeros(np.unique(
                        np.concatenate([load_data(img).astype(int).ravel() for img in imgs])
                    )))

                ids_before = _parc_ids(parc)
                parc_resampled = resample_fn(parc, parc_resample, method="nearest")
                lost = ids_before - _parc_ids(parc_resampled)
                if lost:
                    lgr.warning(
                        f"Resampling to {parc_resample} would drop {len(lost)} parcel(s) "
                        f"(IDs: {sorted(lost)}). Skipping downsampling and using original "
                        f"{current_density} density."
                    )
                else:
                    parc = parc_resampled
                    

        if surf_euclidean:
            density = _img_density_for_neuromaps(parc[0] if isinstance(parc, tuple) else parc)
            lgr.info(f"Estimating euclidean distance matrix: {parc_space} {density} surface parcels, "
                     f"centroid mode.")
            _parc_centroids = find_surf_parc_centroids(
                parc=parc,
                parc_space=parc_space,
                parc_hemi=parc_hemi,
                parc_density=density,
            )
            dist = _dist_mat_from_coords(_parc_centroids, dtype=dtype)

        else:
            density = _img_density_for_neuromaps(parc[0] if isinstance(parc, tuple) else parc)
            atlas, surf_key = _get_surface_atlas(parc_space, density)

            hemis = parc_hemi if isinstance(parc_hemi, (list, tuple)) else [parc_hemi]
            dist_hemis = []
            for i_hemi, hemi in enumerate(hemis):
                surf_path   = getattr(atlas[surf_key], hemi)
                medial_path = getattr(atlas["medial"], hemi)
                parc_h = parc[i_hemi] if isinstance(parc, tuple) else parc
                dist_hemis.append(
                    _surf_dist_hemi(surf_path, parc_h, medial_path,
                                    centroids, n_proc, dtype, verbose, hemi=hemi)
                )

            dist = tuple(dist_hemis) if isinstance(parc, tuple) else dist_hemis[0]

    # case other
    else:
        lgr.error(f"Distance matrix generation not supported for space {parc_space}!")
        
    ## return
    return dist

def find_vol_parc_centroids(parc, affine=None, parcel_idc=None, return_data_space=False, snap=True):
    """Compute per-parcel mean voxel coordinates in world (MNI) space.

    Plain numpy/nibabel (no numba). Used internally by :func:`get_distance_matrix`
    (`centroids=True` path).

    Parameters
    ----------
    parc : np.ndarray or image-like
        Parcellation label array, or an image to load one from.
    affine : np.ndarray, optional
        4x4 affine mapping voxel to world coordinates. Required if `parc` is
        a plain array; otherwise taken from `parc` itself.
    parcel_idc : array-like, optional
        Parcel label values to compute centroids for. Defaults to all
        nonzero labels present in `parc`.
    return_data_space : bool, default=False
        Also return centroid coordinates in voxel (data) space.
    snap : bool, default=True
        Snap the mean coordinate to the nearest voxel actually inside the
        parcel (guards against the raw mean landing outside a non-convex
        parcel). If False, the raw (possibly off-parcel) mean is returned.

    Returns
    -------
    np.ndarray, or (np.ndarray, np.ndarray) if `return_data_space=True`
        Centroid coordinates in world space, shape ``(n_parcels, 3)`` (and,
        if requested, the same in voxel space).

    Raises
    ------
    TypeError
        If `affine` is not given and `parc` is not a Nifti1Image.
    """
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
        voxel_ijk = np.column_stack(np.where(parc_data == i_parcel)).astype(float)
        mean_ijk = voxel_ijk.mean(axis=0)
        if snap:
            # snap to nearest voxel actually within the parcel
            mean_ijk = voxel_ijk[cdist(mean_ijk[None], voxel_ijk)[0].argmin()]
        xyz[i, :] = mean_ijk
    mni = nib.affines.apply_affine(affine, xyz)

    return mni if not return_data_space else (mni, xyz)


def find_surf_parc_centroids(parc, parc_space="fsaverage", parc_hemi=None, parc_density=None, snap=True):
    """Compute per-parcel mean vertex coordinates on a standard cortical surface.

    Plain numpy/nibabel/neuromaps (no numba). Surface counterpart of
    :func:`find_vol_parc_centroids`. Used internally by
    :func:`get_distance_matrix` (`surf_euclidean=True` path) and directly by
    :func:`generate_spins` for the single-hemisphere spin-index code path.

    Parameters
    ----------
    parc : str, nib.GiftiImage, or tuple/list of two
        Surface parcellation: a single GiftiImage/path (one hemisphere) or a
        2-tuple/list ``(lh, rh)``.
    parc_space : str, default="fsaverage"
        Standard surface space to fetch coordinates from (``"fsaverage"`` or
        ``"fsLR"``).
    parc_hemi : str or list of str, optional
        Which hemisphere(s) `parc` represents. Required (as a 1-element
        list) for single-hemisphere input; forced to ``["L", "R"]`` (with an
        info message) for 2-tuple input.
    parc_density : str, optional
        Surface density (e.g. ``"32k"``). Guessed from `parc`'s vertex count
        if not given.
    snap : bool, default=True
        Snap the mean coordinate to the nearest vertex actually inside the
        parcel (the raw mean of surface coordinates is generally not itself
        a vertex on the mesh). If False, the raw mean is returned.

    Returns
    -------
    np.ndarray
        Centroid coordinates, shape ``(n_parcels_total, 3)``, concatenated
        across hemispheres in the order given by `parc`/`parc_hemi`.

    Raises
    ------
    TypeError
        If `parc` is not a supported type.
    """
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
    atlas, surf_key = _get_surface_atlas(parc_space, parc_density)
    surfaces = atlas[surf_key]
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
            parcel_coords = coords[labels == idx]
            mean_coord = parcel_coords.mean(axis=0)
            if snap:
                # snap to nearest vertex within the parcel (guaranteed to be on the surface)
                parcel = parcel_coords[cdist(mean_coord[None], parcel_coords)[0].argmin()]
            else:
                parcel = mean_coord
            centroids.append(parcel)

    return np.row_stack(centroids)




def generate_null_maps(method, data, parcellation, dist_mat=None, spin_mat=None,
                       parc_space=None, parc_hemi=None, parc_symmetric=False,
                       n_nulls=1000, parc_resample=2, centroids=False,
                       parc_idc_lh=None, parc_idc_rh=None, parc_idc_sc=None,
                       lr_mirror_dist_mat=False, split_hemi=None,
                       parc_name=None,
                       dist_mat_sc=None, parc_space_sc=None,
                       dist_mat_cx=None, parc_space_cx=None,
                       dtype=float,
                       n_proc=1, seed=None, verbose=True,
                       return_dict=False,
                       **kwargs):
    """Generate spatially-constrained (or fully random) null maps for one or more parcellated inputs.

    This is the low-level engine behind :meth:`~nispace.api.NiSpace.permute`'s spatial null
    models (called internally via ``core.permute._get_null_maps``) — most users will not call
    it directly. It dispatches to a distance-based surrogate method (:func:`nulls_moran`,
    :func:`nulls_burt2018`, :func:`nulls_burt2020`, or plain permutation via :func:`nulls_random`)
    or, for cortex-only surface methods, a spin-test method (via :func:`generate_spins`,
    ``generate_baum_spins``, or ``generate_cornblath_mat`` + :func:`apply_spins`/``apply_cornblath_mat``),
    based on ``method``.

    Parameters
    ----------
    method : str or tuple of str
        Null method to use. One of the distance-based methods ``"random"``, ``"moran"``
        (aliases ``"msr"``, ``"brainspace"``), ``"variomoran"`` (alias ``"variomsr"``),
        ``"burt2020"`` (aliases ``"brainsmash"``, ``"variogram"``), ``"burt2018"``; or one of
        the spin-test methods (cortex/surface only) ``"alexander_bloch"``, ``"spin"``
        (alias of ``"cornblath"``), ``"vasa"``, ``"hungarian"``, ``"baum"``, ``"cornblath"``.
        See :doc:`/citation` for the citation of each method. Aliases are canonicalized
        internally for stable cache keys. For combined cortex+subcortex parcellations, pass a
        ``(cx_method, sc_method)`` tuple or a ``"cx_method+sc_method"`` shorthand string (e.g.
        ``"spin+moran"``) to use a spin test for cortex and a distance-based method for
        subcortex (subcortex cannot itself be spun — ``sc_method`` must not be a spin method).
        There is no implicit default here: this function always requires ``method`` to be
        given explicitly; the "moran" default seen elsewhere in NiSpace is resolved one level
        up, in ``Parcellation.get_null_space()`` / ``core.permute._get_null_maps``.
    data : array-like, pandas Series, or pandas DataFrame
        One or more parcellated maps to generate nulls for, shape ``(n_parcels,)`` or
        ``(n_maps, n_parcels)``. A DataFrame's index (or a Series' name) becomes the label(s)
        attached to the returned :class:`~nispace.core.nullmaps.NullMaps`.
    parcellation : Parcellation, NIfTI image, GIfTI image, tuple of two GIfTI images, str, or None
        The parcellation the data is defined on. Passing a :class:`~nispace.core.parcellation.Parcellation`
        object is preferred — it lets this function reuse already-cached distance/spin matrices
        and metadata instead of recomputing them. Can be ``None`` if a usable ``dist_mat`` is
        already supplied, or for ``method="random"``, which needs no spatial information at all.
    dist_mat : array-like or tuple of two array-likes, optional
        Precomputed parcel-by-parcel distance matrix — a single 2D array (e.g. volumetric/MNI)
        or a ``(dist_lh, dist_rh)`` tuple (surface, one matrix per hemisphere). If given,
        distance computation is skipped. Ignored for spin methods and for ``"random"``. If
        omitted for a distance-based method, it is computed from ``parcellation`` via
        :func:`get_distance_matrix`.
    spin_mat : tuple, optional
        Precomputed spin/rotation data for a spin-test ``method``. Expected shape depends on
        the method: a ``(spins_lh, spins_rh)`` pair of 2D int arrays (as returned by
        :func:`generate_spins`/``generate_baum_spins``) for ``"alexander_bloch"``/``"baum"``,
        or a ``(T_lh, T_rh)`` pair of 3D arrays (as returned by ``generate_cornblath_mat``) for
        ``"cornblath"``/``"spin"``. Regenerated (with a warning) if shape/type don't match; for
        ``"vasa"``/``"hungarian"`` any provided ``spin_mat`` is always discarded and regenerated,
        since these methods can't reuse a precomputed rotation set.
    parc_space : str, optional
        Reference space of the parcellation (e.g. ``"mni152"``, ``"fsaverage"``, ``"fsLR"``).
        Required unless it can be inferred from ``dist_mat``'s type (array → assumed
        ``"mni152"``, tuple → assumed ``"fsaverage"``, both with a warning) or a
        ``Parcellation`` object.
    parc_hemi : list of str, optional
        Hemispheres present, e.g. ``["L", "R"]`` or ``["L"]``. Required for spin methods.
    parc_symmetric : bool, default=False
        Whether the parcellation is left-right symmetric. Only relevant to
        ``lr_mirror_dist_mat`` (forced off with a warning if this is ``False``).
    n_nulls : int, default=1000
        Number of null maps/permutations to generate per input map.
    parc_resample : int or str, default=2
        Forwarded to :func:`get_distance_matrix` when a distance matrix must be computed:
        target voxel size (mm) for volumetric resampling, or a target density string for
        surface resampling.
    centroids : bool, default=False
        Forwarded to :func:`get_distance_matrix`: use parcel centroid-to-centroid distances
        (faster) instead of mean voxel/vertex-to-voxel/vertex distances (more precise).
    parc_idc_lh, parc_idc_rh : array-like of int, optional
        Column positions in ``data`` belonging to the left/right hemisphere. Required for spin
        methods. For distance-based methods, used to determine per-hemisphere null generation
        and ``lr_mirror_dist_mat``. If only one is given, the other is inferred as its
        complement (with a warning).
    parc_idc_sc : array-like of int, optional
        Column positions in ``data`` belonging to subcortex. Required when ``method`` is a
        ``(cx_method, sc_method)`` tuple (raises ``ValueError`` if missing); the complement
        becomes the cortex index set.
    lr_mirror_dist_mat : bool, default=False
        If True, average the LH and RH (or cortex/subcortex) distance-matrix blocks together
        before generating nulls, so the same spatial null structure is imposed symmetrically on
        both, and symmetrize NaNs in ``data`` across hemispheres accordingly. Requires
        ``parc_symmetric=True`` (else disabled with a warning); raises ``ValueError`` if the
        averaged blocks aren't numerically close afterward (a sanity check on
        ``parc_idc_lh``/``parc_idc_rh``).
    split_hemi : bool, optional
        Whether to generate nulls separately per hemisphere block. Defaults to whether
        ``dist_mat`` is a tuple (surface) or not (volumetric).
    parc_name : str, optional
        Informational label, forwarded through recursive split-path calls; not otherwise used.
    dist_mat_sc, parc_space_sc : optional
        Precomputed subcortex-only distance matrix and its space, used in the split
        (``sc_method is not None``) path to avoid ever computing a full combined distance
        matrix. Preferred over slicing a full ``dist_mat``.
    dist_mat_cx, parc_space_cx : optional
        Same as above, for the cortex-only sub-call in the split path (only relevant when
        ``cx_method`` is not itself a spin method).
    dtype : data-type, default=float
        Output dtype for the returned null maps (and for internal distance-matrix arrays).
    n_proc : int, default=1
        Parallel workers: used both inside :func:`get_distance_matrix` and to parallelize null
        generation across the rows (maps) of ``data`` via ``joblib.Parallel``.
    seed : int, optional
        Base random seed. If given, row ``i`` of ``data`` is seeded with ``seed + i`` — this
        keeps results reproducible under parallelism, but means rows are not independently
        drawn in the strict i.i.d. sense. If omitted, a random base seed is drawn once.
    verbose : bool, default=True
        Whether to print progress messages and progress bars.
    return_dict : bool, default=False
        Deprecated. If True, returns a plain ``{label: null_array}`` dict instead of a
        :class:`~nispace.core.nullmaps.NullMaps` object. The returned ``NullMaps`` already
        supports dict-like access, so there is no remaining reason to use this.
    **kwargs
        Forwarded to the underlying null-generating function for distance-based methods —
        e.g. ``fit_variogram``, ``procedure``, ``joint``, ``n_components`` for
        :func:`nulls_moran`; ``resample``, ``batch_size`` for :func:`nulls_burt2020`. NiSpace's
        higher-level API (:meth:`~nispace.api.NiSpace.permute`) exposes these via ``maps_*``-prefixed
        keyword arguments that get stripped and forwarded here.

    Returns
    -------
    null_maps : NullMaps
        The generated null maps, shape ``(n_maps, n_nulls, n_parcels)``, labeled by
        ``data``'s index/name (or a positional range if unlabeled). A plain dict instead if
        ``return_dict=True`` (deprecated).
    result_mat : array-like or tuple
        The distance matrix or spin matrix actually used — identical to what was passed in via
        ``dist_mat``/``spin_mat`` if provided, or the freshly computed/generated one otherwise.
        Returned so callers can cache and reuse it across repeated calls.

    Raises
    ------
    ValueError
        For an unrecognized ``method``; a missing ``parc_idc_sc`` when a split method is
        requested; a spin method requested for ``sc_method``; a non-surface or incomplete
        parcellation for spin methods; inconsistent/empty/duplicate hemisphere or subcortex
        index sets; a non-square or missing distance-matrix block; a failed
        ``lr_mirror_dist_mat`` symmetry check; or non-array-like ``data``.
    TypeError
        For an unrecognized ``parcellation`` object type.
    ImportError
        If ``method`` resolves to :func:`nulls_burt2020` and the optional ``brainsmash``
        package is not installed.

    Notes
    -----
    See :doc:`/citation` for the citation of each null method.
    """
    verbose = set_log(lgr, verbose)

    # parse method: supports tuple (cx_method, sc_method) and "cx+sc" shorthand
    cx_method, sc_method = _parse_null_method(method)

    # input data (needed before split path so data_labs is available)
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

    # accept Parcellation object: unpack to flat image/space/index parameters
    if parcellation is not None:
        from .core.parcellation import Parcellation as _Parcellation
        if isinstance(parcellation, _Parcellation):
            _parc = parcellation
            if parc_name is None:
                parc_name = _parc._name
            parc_symmetric = _parc._symmetric
            # ensure an active space is set (needed for _image_obj, _idc_byhemi, _cx_idc_lh/rh)
            if _parc._space is None:
                _ns = _parc.get_null_space()
                if isinstance(_ns[0], tuple):
                    # combined: use sc/MNI space (holds the merged volumetric image)
                    _auto_space = _ns[1][0]
                elif cx_method in _SPIN_METHODS:
                    # spin: use the surface space from get_null_space()
                    _auto_space = _ns[0]
                else:
                    # non-spin: prefer MNI volume (stored Euclidean dist_mat, faster)
                    _auto_space = next(
                        (s for s in ["MNI152NLin6Asym", "MNI152NLin2009cAsym", "MNI152",
                                     "MNIOriginal", "MNI"]
                         if s in _parc.spaces),
                        _ns[0],  # fallback to get_null_space() suggestion
                    )
                lgr.info(f"Parcellation '{_parc._name}' has no active space; "
                         f"auto-selecting '{_auto_space}'.")
                _parc.set_active_space(_auto_space)
            # sc indices for combined parcellations (needed for split path below)
            if parc_idc_sc is None and _parc._is_combined:
                parc_idc_sc = _parc.get_sc_idc()
            # component dist_mats for combined + split method (lazy, avoids computing full combined)
            if sc_method is not None and _parc._is_combined:
                if dist_mat_sc is None:
                    dist_mat_sc, _sc_spc = _parc.get_sc_dist_mat()
                    if parc_space_sc is None:
                        parc_space_sc = _sc_spc
                if dist_mat_cx is None:
                    dist_mat_cx, _cx_spc = _parc.get_cx_dist_mat()
                    if parc_space_cx is None:
                        parc_space_cx = _cx_spc
            # resolve image and spatial metadata
            if cx_method in _SPIN_METHODS:
                surf_img, surf_spin_mat, surf_space = _parc.get_surface_for_spins()
                if surf_img is not None:
                    parcellation = surf_img
                    if parc_space is None:
                        parc_space = surf_space
                    if parc_hemi is None:
                        parc_hemi = ("L", "R")
                    if spin_mat is None and surf_spin_mat is not None:
                        spin_mat = surf_spin_mat
                    if parc_idc_lh is None:
                        parc_idc_lh = (_parc._cx_idc_lh
                                       if _parc._is_combined and _parc._cx_idc_lh is not None
                                       else _parc._idc_byhemi.get("L"))
                    if parc_idc_rh is None:
                        parc_idc_rh = (_parc._cx_idc_rh
                                       if _parc._is_combined and _parc._cx_idc_rh is not None
                                       else _parc._idc_byhemi.get("R"))
                else:
                    parcellation = _parc._image_obj
                    if parc_space is None:
                        parc_space = _parc._space
                    if parc_hemi is None:
                        parc_hemi = _parc._hemi
            else:
                parcellation = _parc._image_obj
                if parc_space is None:
                    parc_space = _parc._space
                if parc_hemi is None:
                    parc_hemi = _parc._hemi
                if parc_idc_lh is None:
                    parc_idc_lh = _parc._idc_byhemi.get("L")
                if parc_idc_rh is None:
                    parc_idc_rh = _parc._idc_byhemi.get("R")
                # lazy-load stored dist_mat — avoids recomputing from the image
                if dist_mat is None:
                    dist_mat = _parc._dist_mat

    ## SPLIT PATH: cx_method + sc_method differ (or same — handles (m,m) as well)
    if sc_method is not None:
        if parc_idc_sc is None:
            lgr.critical_raise(
                "Split null method requires 'parc_idc_sc' to identify subcortex parcels.",
                ValueError)
        if sc_method in _SPIN_METHODS:
            lgr.critical_raise(
                f"Spin methods are cortex-only; sc_method='{sc_method}' is not valid. "
                f"Use one of {_DISTMAT_METHODS} for subcortex.",
                ValueError)
        parc_idc_sc = np.asarray(parc_idc_sc)
        parc_idc_cx = np.setdiff1d(np.arange(data.shape[1]), parc_idc_sc)

        lgr.info(f"Split null method: cx='{cx_method}' ({len(parc_idc_cx)} parcels), "
                 f"sc='{sc_method}' ({len(parc_idc_sc)} parcels).")

        # CX PATH
        if cx_method in _SPIN_METHODS:
            # spin: full data + surface parcellation; sc positions → NaN in output
            cx_nulls, result_mat = generate_null_maps(
                method=cx_method, data=data, parcellation=parcellation,
                dist_mat=None, spin_mat=spin_mat,
                parc_space=parc_space, parc_hemi=parc_hemi, parc_symmetric=parc_symmetric,
                parc_resample=parc_resample, n_nulls=n_nulls, centroids=centroids,
                parc_idc_lh=parc_idc_lh, parc_idc_rh=parc_idc_rh, parc_idc_sc=None,
                lr_mirror_dist_mat=lr_mirror_dist_mat, split_hemi=split_hemi,
                parc_name=parc_name, dtype=dtype, n_proc=n_proc, seed=seed,
                verbose=verbose, **kwargs)
            # cx_nulls: (n_maps, n_perm, n_parcels) — sc positions are NaN
            merged = cx_nulls.data.copy()
        else:
            # non-spin: subset cx data + cx dist_mat (prefer pre-loaded, else slice from full)
            data_cx = data[:, parc_idc_cx]
            if dist_mat_cx is not None:
                _dist_mat_cx = dist_mat_cx
            elif dist_mat is not None:
                _dist_mat_cx = dist_mat[np.ix_(parc_idc_cx, parc_idc_cx)]
            else:
                _dist_mat_cx = None
            cx_nulls, result_mat = generate_null_maps(
                method=cx_method, data=data_cx, parcellation=None,
                parc_space=parc_space_cx or parc_space,
                dist_mat=_dist_mat_cx, n_nulls=n_nulls, centroids=centroids,
                split_hemi=None, parc_idc_lh=None, parc_idc_rh=None,
                parc_name=parc_name, dtype=dtype, n_proc=n_proc, seed=seed,
                verbose=verbose, **kwargs)
            # cx_nulls: (n_maps, n_perm, n_cx)
            merged = np.full((n_data, n_nulls, data.shape[1]), np.nan, dtype=dtype)
            merged[:, :, parc_idc_cx] = cx_nulls.data

        # SC PATH (always non-spin)
        data_sc = data[:, parc_idc_sc]
        # use pre-provided sc dist_mat (preferred), else slice from full dist_mat
        if dist_mat_sc is not None:
            _dist_mat_sc = dist_mat_sc
        elif dist_mat is not None:
            _dist_mat_sc = dist_mat[np.ix_(parc_idc_sc, parc_idc_sc)]
        else:
            _dist_mat_sc = None
        sc_nulls, _ = generate_null_maps(
            method=sc_method, data=data_sc, parcellation=None,
            parc_space=parc_space_sc,
            dist_mat=_dist_mat_sc, n_nulls=n_nulls, centroids=centroids,
            parc_name=parc_name, dtype=dtype, n_proc=n_proc, seed=seed,
            verbose=verbose, **kwargs)
        # sc_nulls: (n_maps, n_perm, n_sc)
        merged[:, :, parc_idc_sc] = sc_nulls.data

        return NullMaps(merged, data_labs, dtype=dtype,
                        null_method=(cx_method, sc_method), null_type="spatial"), result_mat

    ## SINGLE METHOD PATH
    method = cx_method  # unwrap from parse result

    ## Checks
    # null method
    if method not in _NULL_METHODS:
        lgr.critical_raise(f"Null method {method} not implemented!",
                           ValueError)
    null_fun = _NULL_METHODS[method]
    random_nulls = False
    if null_fun is not None and null_fun.__name__ == "nulls_burt2020" and not _BRAINSMASH_AVAILABLE:
        lgr.critical_raise("Null method 'burt2020' requires brainsmash! Run 'pip install brainsmash'!",
                           ImportError)
    elif null_fun is not None and null_fun.__name__ == "nulls_random":
        random_nulls = True

    # print
    lgr.info(f"Null map generation: Assuming n = {n_data} data vector(s) for "
             f"n = {data.shape[1]} parcels.")

    ## spin nulls -> separate code path, bypass dist_mat entirely
    if method in _SPIN_METHODS:
        # validate: surface parcellation required (bilateral tuple or unilateral GiftiImage)
        if not isinstance(parcellation, (tuple, nib.GiftiImage)):
            lgr.critical_raise(
                f"Null method '{method}' requires a surface parcellation. "
                f"Volumetric parcellations are not supported (use a distance-based null instead).",
                ValueError,
            )

        # validate: hemisphere indices required
        if parc_idc_lh is None or parc_idc_rh is None:
            lgr.critical_raise(
                f"Null method '{method}' requires 'parc_idc_lh' and 'parc_idc_rh'.",
                ValueError
            )
        idc_lh = np.array(parc_idc_lh)
        idc_rh = np.array(parc_idc_rh)

        ## --- Cornblath: fractional transition matrix path ---
        if method == "cornblath":
            if not isinstance(parcellation, tuple):
                lgr.critical_raise(
                    "Null method 'cornblath' requires a bilateral surface parcellation tuple.",
                    ValueError)
            if spin_mat is not None:
                if (isinstance(spin_mat, tuple) and len(spin_mat) == 2
                        and isinstance(spin_mat[0], np.ndarray)
                        and spin_mat[0].ndim == 3 and spin_mat[0].shape[0] >= n_nulls):
                    T_lh, T_rh = spin_mat[0], spin_mat[1]
                    lgr.info("Using provided precomputed Cornblath transition matrix.")
                else:
                    lgr.warning("Provided 'spin_mat' is not a valid Cornblath T-matrix "
                                "(expected 3-D float32 tuple with n_perm >= n_nulls). Regenerating.")
                    spin_mat = None
            if spin_mat is None:
                lgr.info(f"Generating Cornblath transition matrices (n={n_nulls}).")
                T_lh, T_rh = generate_cornblath_mat(
                    parc=parcellation, parc_space=parc_space, n_perm=n_nulls, seed=seed,
                    n_proc=n_proc)
                spin_mat = (T_lh, T_rh)
            _null_list = []
            for i, lab in enumerate(tqdm(data_labs, desc="Cornblath null maps", disable=not verbose)):
                _null_list.append(apply_cornblath_mat(
                    data_1d=data[i, :].astype(dtype),
                    T_lh=T_lh, T_rh=T_rh,
                    idc_lh=idc_lh, idc_rh=idc_rh,
                    n_perm=n_nulls,
                ))

        ## --- Baum / Alexander-Bloch / Vasa / Hungarian: parcel-index path ---
        else:
            spin_method = _SPIN_METHOD_MAP[method]

            # precomputed spin_mat accepted for alexander_bloch/spin and baum; always regen for vasa/hungarian
            if spin_mat is not None and spin_method in ("original", "baum"):
                if (isinstance(spin_mat, tuple) and len(spin_mat) == 2
                        and isinstance(spin_mat[0], np.ndarray)
                        and spin_mat[0].ndim == 2 and spin_mat[0].shape[1] >= n_nulls):
                    spins_lh, spins_rh = spin_mat[0], spin_mat[1]
                    lgr.info("Using provided precomputed spin matrix.")
                else:
                    lgr.warning("Provided 'spin_mat' must be a 2-D int tuple with n_perm >= n_nulls. Regenerating.")
                    spin_mat = None
            elif spin_mat is not None:
                spin_mat = None  # vasa/hungarian always regenerate

            if spin_mat is None:
                if method == "baum":
                    if not isinstance(parcellation, tuple):
                        lgr.critical_raise(
                            "Null method 'baum' requires a bilateral surface parcellation tuple.",
                            ValueError)
                    lgr.info(f"Generating Baum spin samples (vertex-modal, n={n_nulls}).")
                    spins_lh, spins_rh = generate_baum_spins(
                        parc=parcellation, parc_space=parc_space, n_perm=n_nulls, seed=seed,
                        n_proc=n_proc)
                else:
                    lgr.info(f"Generating spin samples (method='{spin_method}', n={n_nulls}).")
                    spins_lh, spins_rh = generate_spins(
                        parc=parcellation, parc_space=parc_space, n_perm=n_nulls,
                        method=spin_method, seed=seed, parc_hemi=parc_hemi,
                    )
                spin_mat = (spins_lh, spins_rh)

            _null_list = []
            for i, lab in enumerate(tqdm(data_labs, desc="Spin null maps", disable=not verbose)):
                _null_list.append(apply_spins(
                    data_1d=data[i, :].astype(dtype),
                    spins_lh=spins_lh, spins_rh=spins_rh,
                    idc_lh=idc_lh, idc_rh=idc_rh,
                    n_perm=n_nulls,
                ))

        # stack: (n_maps, n_perm, n_parcels) — always 3-D even for n_maps=1
        nulls = NullMaps(np.stack(_null_list), data_labs, dtype=dtype,
                         null_method=method, null_type="spatial")

        lgr.info("Null data generation finished.")
        # TODO (first non-dev release): remove return_dict parameter
        if return_dict:
            lgr.warning(_DEPR_RETURN_DICT)
            return {lbl: nulls[lbl] for lbl in nulls.keys()}, spin_mat
        return nulls, spin_mat

    ## random nulls -> no distmat
    if random_nulls:
        dist_mat = None  # random nulls never use dist_mat; drop any provided value (including tuples)

    ## distance matrix provided -> we dont need parcellation
    if dist_mat is not None and not random_nulls:        
        lgr.info(f"Using provided distance matrix/matrices.")
        if isinstance(dist_mat, (np.ndarray, pd.DataFrame)):
            n_parcels = dist_mat.shape[0]
            dist_mat = np.array(dist_mat, dtype=dtype)
            if parc_space is None:
                lgr.warning("Distance matrix provided as array but 'parc_space' is None: "
                            "Assuming 'mni152'! Define 'parc_space' if one surface hemisphere!")
                parc_space = "mni152"
        elif isinstance(dist_mat, tuple):
            n_parcels = (dist_mat[0].shape[0],
                         dist_mat[1].shape[0])     
            dist_mat = tuple(np.array(dm, dtype=dtype) for dm in dist_mat)
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
        if parcellation is None:
            lgr.critical_raise(
                f"Null method '{method}' requires a parcellation or a pre-computed distance "
                "matrix, but both 'parcellation' and 'dist_mat' are None.",
                ValueError,
            )
        elif isinstance(parcellation, nib.Nifti1Image):
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
                    "parcellation may not be symmetric. Set 'parc_symmetric=True' to enable this. "
                    "Disabling lr_mirror_dist_mat.")
        lr_mirror_dist_mat = False

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
            
    # auto-detect split_hemi: True for surface (tuple dist_mat), False for volumetric
    if split_hemi is None:
        split_hemi = isinstance(dist_mat, tuple)

    # if split_hemi=False and dist_mat is a surface tuple, flatten to block-diagonal single matrix
    if not split_hemi and isinstance(dist_mat, tuple):
        n_blocks = [d.shape[0] for d in dist_mat]
        n_total = sum(n_blocks)
        D_full = np.zeros((n_total, n_total), dtype=dtype)
        offset = 0
        for d in dist_mat:
            n = d.shape[0]
            D_full[offset:offset+n, offset:offset+n] = d
            offset += n
        dist_mat = D_full

    # get all index lists according to which we want to split the data and distance matrix
    if isinstance(dist_mat, tuple): # surface input with split_hemi=True
        split_by_idc = (
            np.arange(dist_mat[0].shape[0]), # left hemisphere
            np.arange(dist_mat[1].shape[0]) + dist_mat[0].shape[0], # right hemisphere
        )
    elif split_hemi and parc_idc_lh is not None:
        lgr.info("Generating null data separately for left and right hemisphere.")
        split_by_idc = (
            parc_idc_lh,  # whole left hemisphere
            parc_idc_rh,  # whole right hemisphere
        )
    else:
        split_by_idc = (np.arange(data.shape[1]),) # whole-brain (default)
        
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
        if len(dist_mat_split) == 1 and parc_idc_lh is not None and parc_idc_rh is not None:
            # split_hemi=False: patch the LH and RH diagonal blocks of the full dist_mat
            lh, rh = np.array(parc_idc_lh), np.array(parc_idc_rh)
            d = dist_mat_split[0].copy()
            avg_cx = _avg_dist_mats(d[np.ix_(lh, lh)], d[np.ix_(rh, rh)])
            d[np.ix_(lh, lh)] = avg_cx
            d[np.ix_(rh, rh)] = avg_cx
            dist_mat_split = (d,)
        elif len(dist_mat_split) == 2:
            avg = _avg_dist_mats(dist_mat_split[0], dist_mat_split[1])
            dist_mat_split = (avg, avg)
            if not np.allclose(dist_mat_split[0], dist_mat_split[1]):
                lgr.critical_raise("Left-right averaged whole-hemisphere distance matrices are not equal! "
                                   "Check if 'parc_idc_lh' and 'parc_idc_rh' are correctly defined.",
                                   ValueError)
        elif len(dist_mat_split) == 4:
            avg_cx = _avg_dist_mats(dist_mat_split[0], dist_mat_split[1])
            avg_sc = _avg_dist_mats(dist_mat_split[2], dist_mat_split[3])
            dist_mat_split = (avg_cx, avg_cx, avg_sc, avg_sc)
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
                null_data[:, idc] = null_fun(data_1d=data_1d_sel, dist_mat=dist,
                                             n_nulls=n_nulls, seed=seed, **kwargs)
        return null_data

    # run null data generation
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
    null_list = Parallel(n_jobs=n_proc)(
        delayed(par_fun)(data[i, :], seed + i)
        for i in tqdm(
            range(n_data),
            desc=f"{null_fun.__name__.split('_')[1].capitalize()} null maps ({n_proc} proc)",
            disable=not verbose
        )
    )
    # stack: (n_maps, n_perm, n_parcels) — always 3-D even for n_maps=1
    nulls = NullMaps(np.stack(null_list).astype(dtype), data_labs, dtype=dtype,
                     null_method=method, null_type="spatial")

    ## return
    lgr.info("Null data generation finished.")
    # TODO (first non-dev release): remove return_dict parameter
    if return_dict:
        lgr.warning(_DEPR_RETURN_DICT)
        return {lbl: nulls[lbl] for lbl in nulls.keys()}, dist_mat
    return nulls, dist_mat
        
    