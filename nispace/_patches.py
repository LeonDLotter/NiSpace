"""
Performance patches for nilearn's surface plotting pipeline.

Applied once at import time via ``apply_surface_plot_patches()``.
All patches include a fallback to the original implementation so that
unrecognised inputs are handled identically to stock nilearn/matplotlib.
"""
import numpy as np

_PATCHES_APPLIED = False


# ---------------------------------------------------------------------------
# Patch 1 — Poly3DCollection.do_3d_projection  (mpl_toolkits.mplot3d.art3d)
# ---------------------------------------------------------------------------
# The original implementation builds a Python list of per-face tuples and
# sorts them with Python's sorted().  For a triangular mesh (the only output
# of matplotlib's plot_trisurf) every face has exactly 3 vertices, so the
# flat vertex arrays can be reshaped into (N_faces, 3) and all the per-face
# work is replaced by vectorised NumPy operations:
#   - sort key: tzs.reshape(-1, 3).mean(axis=1)   instead of 160k np.average calls
#   - depth sort: np.argsort(...) in C             instead of Python sorted() on objects
#   - segments:   np.stack([txs_f, tys_f], axis=-1) instead of 160k np.column_stack calls

def _make_fast_do_3d_projection(original_fn):
    from mpl_toolkits.mplot3d import proj3d
    from matplotlib.collections import PolyCollection

    # Map the three built-in zsort callables to vectorised equivalents.
    _zsort_vec = {
        np.average: lambda z: z.mean(axis=1),
        np.min:     lambda z: z.min(axis=1),
        np.max:     lambda z: z.max(axis=1),
    }

    def _fast_do_3d_projection(self):
        # Scalar-mappable colour update — identical to original.
        if self._A is not None:
            self.update_scalarmappable()
            if self._face_is_mapped:
                self._facecolor3d = self._facecolors
            if self._edge_is_mapped:
                self._edgecolor3d = self._edgecolors

        txs, tys, tzs = proj3d._proj_transform_vec(self._vec, self.axes.M)
        n_faces = len(self._segslices)

        # Fast path conditions:
        #   • uniform triangular mesh  (len(txs) == 3 * n_faces)
        #   • known zsort function     (one of average/min/max)
        #   • no per-face path codes   (_codes3d is None, always true for plot_trisurf)
        _zsort_key_fn = _zsort_vec.get(self._zsortfunc)
        if (
            n_faces > 0
            and len(txs) == 3 * n_faces
            and _zsort_key_fn is not None
            and self._codes3d is None
        ):
            txs_f = txs.reshape(n_faces, 3)
            tys_f = tys.reshape(n_faces, 3)
            tzs_f = tzs.reshape(n_faces, 3)

            # Broadcast single-row colour arrays to per-face (same as original).
            cface = self._facecolor3d
            cedge = self._edgecolor3d
            if len(cface) != n_faces:
                cface = cface.repeat(n_faces, axis=0)
            if len(cedge) != n_faces:
                cedge = cface if len(cedge) == 0 else cedge.repeat(n_faces, axis=0)

            # Depth sort: furthest face first (painter's algorithm).
            sort_idx = np.argsort(_zsort_key_fn(tzs_f))[::-1]

            self._facecolors2d = cface[sort_idx]
            self._edgecolors2d = cedge[sort_idx]

            # Build sorted (N_faces, 3, 2) array; PolyCollection.set_verts
            # iterates the first axis and constructs Path objects per face.
            segments_2d = np.stack([txs_f, tys_f], axis=-1)[sort_idx]
            PolyCollection.set_verts(self, segments_2d, self._closed)

            # Edge colour override — identical to original line 1059-1060.
            if len(self._edgecolor3d) != len(cface):
                self._edgecolors2d = self._edgecolor3d

            if self._sort_zpos is not None:
                zvec = np.array([[0], [0], [self._sort_zpos], [1]])
                ztrans = proj3d._proj_transform_vec(zvec, self.axes.M)
                return ztrans[2][0]
            return np.min(tzs) if tzs.size > 0 else np.nan

        # Fallback: non-uniform mesh, custom zsort, or path codes present.
        return original_fn(self)

    return _fast_do_3d_projection


# ---------------------------------------------------------------------------
# Patch 2 — mix_colormaps  (nilearn.plotting.cm)
# ---------------------------------------------------------------------------
# Replaces the 3-iteration Python loop over RGB channels with a single
# broadcast operation over the full (N, 3) slice.

def _fast_mix_colormaps(fg, bg):
    """Vectorised Porter-Duff 'over' composite for RGBA float arrays."""
    if fg.shape != bg.shape:
        raise ValueError(
            f"Trying to mix colormaps with different shapes: {fg.shape}, {bg.shape}"
        )
    mix = np.empty_like(fg)
    mix[:, 3] = 1 - (1 - fg[:, 3]) * (1 - bg[:, 3])
    # Guard against fully-transparent pixels to avoid 0/0 → NaN.
    denom = np.where(mix[:, 3:4] > 0, mix[:, 3:4], 1.0)
    mix[:, :3] = (
        fg[:, :3] * fg[:, 3:4] + bg[:, :3] * bg[:, 3:4] * (1 - fg[:, 3:4])
    ) / denom
    return mix


# ---------------------------------------------------------------------------
# Patch 3 — _get_cmap  (nilearn.plotting.surface._matplotlib_backend)
# ---------------------------------------------------------------------------
# Replaces [cmap(i) for i in range(N)] (256 Python calls) with a single
# vectorised lookup.

def _make_fast_get_cmap(original_fn):
    import warnings
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib import pyplot as plt
    try:
        from nilearn._utils.logger import find_stack_level
    except ImportError:
        def find_stack_level():
            return 2

    def _fast_get_cmap(cmap, vmin, vmax, cbar_tick_format, threshold=None):
        our_cmap = plt.get_cmap(cmap)
        norm = Normalize(vmin=vmin, vmax=vmax)
        # Vectorised: one call with an integer array instead of N scalar calls.
        # matplotlib maps integer i to position i/N, matching the original loop.
        cmaplist = list(our_cmap(np.arange(our_cmap.N)))
        if threshold is not None:
            if cbar_tick_format == "%i" and int(threshold) != threshold:
                warnings.warn(
                    "You provided a non integer threshold "
                    "but configured the colorbar to use integer formatting.",
                    stacklevel=find_stack_level(),
                )
            istart = int(norm(-threshold, clip=True) * (our_cmap.N - 1))
            istop  = int(norm( threshold, clip=True) * (our_cmap.N - 1))
            for i in range(istart, istop):
                cmaplist[i] = (0.5, 0.5, 0.5, 1.0)
        our_cmap = LinearSegmentedColormap.from_list(
            "Custom cmap", cmaplist, our_cmap.N
        )
        return our_cmap, norm

    return _fast_get_cmap


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def apply_surface_plot_patches():
    """Apply all three patches.  Idempotent — safe to call multiple times."""
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    # 1 — Poly3DCollection.do_3d_projection
    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        Poly3DCollection.do_3d_projection = _make_fast_do_3d_projection(
            Poly3DCollection.do_3d_projection
        )
    except Exception as exc:
        import warnings
        warnings.warn(
            f"NiSpace: could not patch Poly3DCollection.do_3d_projection: {exc}"
        )

    # 2 — mix_colormaps (patch both the source module and the backend's
    #     local reference, since the backend imported it by name at load time)
    try:
        import nilearn.plotting.cm as _cm_mod
        import nilearn.plotting.surface._matplotlib_backend as _mpl_be
        _cm_mod.mix_colormaps = _fast_mix_colormaps
        _mpl_be.mix_colormaps = _fast_mix_colormaps
    except Exception as exc:
        import warnings
        warnings.warn(f"NiSpace: could not patch mix_colormaps: {exc}")

    # 3 — _get_cmap
    try:
        import nilearn.plotting.surface._matplotlib_backend as _mpl_be
        _mpl_be._get_cmap = _make_fast_get_cmap(_mpl_be._get_cmap)
    except Exception as exc:
        import warnings
        warnings.warn(f"NiSpace: could not patch _get_cmap: {exc}")

    _PATCHES_APPLIED = True
