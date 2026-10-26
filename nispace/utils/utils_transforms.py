"""Numerical kernels for space transforms (pure numpy/scipy, no Connectome Workbench).

Surface resampling re-implements Workbench's ``ADAP_BARY_AREA`` method
(``SurfaceResamplingHelper.cxx``): barycentric weights of the closest point on the
source / target sphere meshes, adaptive choice between gathering and scattering
weights, vertex-area correction, and an optional source ROI (medial wall). Weights are
returned as sparse matrices so that a transform is a single matrix product for any
number of maps.

Volume sampling builds sparse trilinear (or nearest-neighbour) matrices for a fixed
input grid, so the same matrix is reused for every map on that grid.
"""

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree


# SURFACE ==========================================================================================

def _sphere_coords(coords, radius=100.0):
    """Project vertex coordinates onto a sphere of `radius` (as Workbench does before resampling)."""
    coords = np.asarray(coords, dtype=np.float64)
    return coords * (radius / np.linalg.norm(coords, axis=1, keepdims=True))


def _vertex_triangles(faces, n_vert):
    """(n_vert, max_degree) array of the triangles incident to each vertex, padded with -1."""
    faces = np.asarray(faces, dtype=np.int64)
    tri = np.repeat(np.arange(len(faces)), 3)
    vert = faces.ravel()
    order = np.argsort(vert, kind="stable")
    vert, tri = vert[order], tri[order]
    counts = np.bincount(vert, minlength=n_vert)
    out = -np.ones((n_vert, counts.max()), dtype=np.int64)
    pos = np.arange(len(vert)) - np.repeat(np.r_[0, np.cumsum(counts)[:-1]], counts)
    out[vert, pos] = tri
    return out


def _closest_point_barycentric(p, a, b, c):
    """Barycentric weights (for a, b, c) of the closest point on triangles (a, b, c) to points p.

    Vectorised version of Ericson, Real-Time Collision Detection (2005), 5.1.5. Inputs broadcast
    to (..., 3). Returns weights (..., 3) and squared distances (...).
    """
    p, a, b, c = np.broadcast_arrays(p, a, b, c)
    dot = lambda x, y: np.einsum("...i,...i", x, y)
    ab, ac = b - a, c - a
    d1, d2 = dot(ab, p - a), dot(ac, p - a)
    d3, d4 = dot(ab, p - b), dot(ac, p - b)
    d5, d6 = dot(ab, p - c), dot(ac, p - c)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    w = np.zeros(p.shape[:-1] + (3,))
    done = np.zeros(p.shape[:-1], dtype=bool)

    def _set(mask, wa, wb, wc):
        m = mask & ~done
        for k, val in enumerate((wa, wb, wc)):
            w[m, k] = val[m] if np.ndim(val) else val
        done[m] = True

    with np.errstate(divide="ignore", invalid="ignore"):
        _set((d1 <= 0) & (d2 <= 0), 1.0, 0.0, 0.0)                                   # vertex a
        _set((d3 >= 0) & (d4 <= d3), 0.0, 1.0, 0.0)                                  # vertex b
        v = d1 / (d1 - d3)
        _set((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1 - v, v, 0.0)                       # edge ab
        _set((d6 >= 0) & (d5 <= d6), 0.0, 0.0, 1.0)                                  # vertex c
        t = d2 / (d2 - d6)
        _set((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1 - t, 0.0, t)                       # edge ac
        u = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        _set((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), 0.0, 1 - u, u)         # edge bc
        denom = 1.0 / (va + vb + vc)
        _set(np.ones_like(done), 1 - (vb + vc) * denom, vb * denom, vc * denom)      # interior
    q = w[..., :1] * a + w[..., 1:2] * b + w[..., 2:] * c
    return w, dot(p - q, p - q)


def barycentric_weights(from_coords, from_faces, to_coords, k=8, chunk=20000):
    """Closest-point barycentric weights of each `to` vertex on the `from` sphere mesh.

    Both meshes are first projected onto a sphere of radius 100 (as in Workbench). Candidate
    triangles are those incident to the `k` nearest `from` vertices.

    Returns
    -------
    scipy.sparse.csr_matrix
        (n_to, n_from) matrix with up to 3 weights per row, each row summing to 1.
    """
    from_coords, to_coords = _sphere_coords(from_coords), _sphere_coords(to_coords)
    from_faces = np.asarray(from_faces, dtype=np.int64)
    vtri = _vertex_triangles(from_faces, len(from_coords))
    tree = cKDTree(from_coords)
    n_to = len(to_coords)
    idx, wgt = np.empty((n_to, 3), dtype=np.int64), np.empty((n_to, 3))
    for s in range(0, n_to, chunk):
        p = to_coords[s:s + chunk]
        _, nn = tree.query(p, k=min(k, len(from_coords)))
        cand = vtri[nn].reshape(len(p), -1)
        valid = cand >= 0
        tri = from_faces[np.where(valid, cand, 0)]                                  # (m, n_cand, 3)
        w, d2 = _closest_point_barycentric(p[:, None, :], *(from_coords[tri[..., i]] for i in range(3)))
        best = np.argmin(np.where(valid, d2, np.inf), axis=1)
        rows = np.arange(len(p))
        idx[s:s + chunk], wgt[s:s + chunk] = tri[rows, best], w[rows, best]
    mat = sp.csr_matrix((wgt.ravel(), (np.repeat(np.arange(n_to), 3), idx.ravel())),
                        shape=(n_to, len(from_coords)))
    mat.eliminate_zeros()
    return mat


def adap_bary_area_weights(cur_sphere, new_sphere, cur_area, new_area, cur_roi=None):
    """Sparse resampling weights equal to Workbench ``-metric-resample ... ADAP_BARY_AREA``.

    Parameters
    ----------
    cur_sphere, new_sphere : tuple of (coords, faces)
        Source and target sphere meshes (registered to each other).
    cur_area, new_area : array_like
        Vertex areas (e.g. the ``vaavg`` midthickness area metrics) of source and target.
    cur_roi : array_like, optional
        Source ROI (> 0 = valid, e.g. the non-medial-wall mask). Excluded source vertices get zero
        weight after area correction and each row is renormalised (``-current-roi``).

    Returns
    -------
    scipy.sparse.csr_matrix
        (n_new, n_cur) weights; rows sum to 1, or are empty where no valid source vertex contributes.
    """
    (cur_coords, cur_faces), (new_coords, new_faces) = cur_sphere, new_sphere
    forward = barycentric_weights(cur_coords, cur_faces, new_coords)                 # new gathers from cur
    reverse = barycentric_weights(new_coords, new_faces, cur_coords).T.tocsr()      # cur scatters onto new
    # per new vertex: use the scattering weights if they contain a source vertex the gathering ones don't
    fwd_pat, rev_pat = forward.astype(bool).astype(np.int8), reverse.astype(bool).astype(np.int8)
    extra = rev_pat - rev_pat.multiply(fwd_pat)
    use_rev = np.asarray(extra.sum(axis=1)).ravel() > 0
    adap = (sp.diags((~use_rev).astype(float)) @ forward + sp.diags(use_rev.astype(float)) @ reverse)
    # area correction: * new area, / scatter sum per source vertex, * source area
    adap = sp.diags(np.asarray(new_area, dtype=np.float64)) @ adap
    scatter = np.asarray(adap.sum(axis=0)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(scatter > 0, np.asarray(cur_area, dtype=np.float64) / scatter, 0.0)
    if cur_roi is not None:
        scale = scale * (np.asarray(cur_roi) > 0)
    adap = (adap @ sp.diags(scale)).tocsr()
    adap.eliminate_zeros()
    rowsum = np.asarray(adap.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        inv = np.where(rowsum != 0, 1.0 / rowsum, 0.0)
    return (sp.diags(inv) @ adap).tocsr()


def resample_popular(weights, labels, fill=0):
    """Label resampling (Workbench "popular"): per target vertex, the label with the largest summed
    weight; ties go to the label that reached the maximum first in source-vertex order, as in
    Workbench. Rows without weights get `fill`. `labels` may be (n_cur,) or (n_maps, n_cur)."""
    labels = np.asarray(labels)
    if labels.ndim == 2:
        return np.stack([resample_popular(weights, lab, fill) for lab in labels])
    weights = weights.tocsr()
    w = weights.tocoo()
    uniq, inv = np.unique(labels[w.col], return_inverse=True)
    sums = sp.csr_matrix((w.data, (w.row, inv)), shape=(w.shape[0], len(uniq)))
    out = np.full(w.shape[0], fill, dtype=labels.dtype)
    has = np.diff(sums.indptr) > 0
    out[has] = uniq[np.asarray(sums.argmax(axis=1)).ravel()[has]]
    # resolve exact ties like Workbench (rare): replay the running sums in source-vertex order
    s_coo = sums.tocoo()
    row_max = np.zeros(w.shape[0])
    np.maximum.at(row_max, s_coo.row, s_coo.data)
    n_at_max = np.bincount(s_coo.row[s_coo.data == row_max[s_coo.row]], minlength=w.shape[0])
    for i in np.flatnonzero(n_at_max > 1):
        s, e = weights.indptr[i], weights.indptr[i + 1]
        acc, best, best_w = {}, fill, -1.0
        for j, wt in zip(weights.indices[s:e], weights.data[s:e]):
            acc[labels[j]] = acc.get(labels[j], 0.0) + wt
            if acc[labels[j]] > best_w:
                best, best_w = labels[j], acc[labels[j]]
        out[i] = best
    return out


# VOLUME ===========================================================================================

def world_to_voxel(coords, affine):
    """Map world (RAS mm) coordinates (n, 3) to voxel coordinates with the inverse of `affine`."""
    inv = np.linalg.inv(np.asarray(affine, dtype=np.float64))
    return np.asarray(coords, dtype=np.float64) @ inv[:3, :3].T + inv[:3, 3]


def sample_field(field_img, coords):
    """Trilinearly sample a coordinate-map field (X, Y, Z, 3; absolute RAS mm) at world `coords`.

    Used to compose a mapping a priori: e.g. MNI152NLin6Asym vertex positions -> the field from
    space X to MNI152NLin6Asym -> the corresponding positions in space X.
    """
    data = np.asarray(field_img.dataobj, dtype=np.float32)
    vox = world_to_voxel(coords, field_img.affine).T
    return np.stack([map_coordinates(data[..., i], vox, order=1, mode="nearest") for i in range(3)], axis=1)


def volume_sampling_matrix(coords, affine, shape, method="linear"):
    """Sparse (n_points, prod(shape)) matrix sampling a volume on (`affine`, `shape`) at world `coords`.

    ``"linear"`` = trilinear (as ``scipy.interpolate.interpn``), ``"nearest"`` = nearest voxel.
    Voxel order is C-order (``data.reshape(-1)``). Points outside the grid get an empty row.

    Returns
    -------
    (scipy.sparse.csr_matrix, numpy.ndarray of bool)
        The matrix and a mask of points inside the grid.
    """
    shape = np.asarray(shape[:3])
    vox = world_to_voxel(coords, affine)
    n = len(vox)
    inside = np.all((vox >= -1e-6) & (vox <= shape - 1 + 1e-6), axis=1)
    vox = np.clip(vox, 0, shape - 1)
    if method == "nearest":
        ijk = np.rint(vox).astype(np.int64)
        cols = np.ravel_multi_index(ijk.T, shape)
        rows, data = np.arange(n), np.ones(n)
    elif method == "linear":
        i0 = np.minimum(np.floor(vox).astype(np.int64), shape - 2).clip(min=0)
        f = vox - i0
        rows, cols, data = [], [], []
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ijk = i0 + np.array([dx, dy, dz])
                    w = (np.where(dx, f[:, 0], 1 - f[:, 0]) * np.where(dy, f[:, 1], 1 - f[:, 1])
                         * np.where(dz, f[:, 2], 1 - f[:, 2]))
                    rows.append(np.arange(n)); cols.append(np.ravel_multi_index(ijk.T, shape)); data.append(w)
        rows, cols, data = np.concatenate(rows), np.concatenate(cols), np.concatenate(data)
    else:
        raise ValueError(f"method={method!r} not supported for volume sampling; use 'linear' or 'nearest'.")
    keep = inside[rows] & (data != 0)
    mat = sp.csr_matrix((data[keep], (rows[keep], cols[keep])), shape=(n, int(np.prod(shape))))
    return mat, inside
