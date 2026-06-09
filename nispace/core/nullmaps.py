import copy
import logging
import numpy as np
from scipy.stats import zscore as scipy_zscore

lgr = logging.getLogger(__name__)

_NULLMAPS_WARN_BYTES = 1_000_000_000  # 1 GB


class NullMaps:
    """
    Container for null map arrays.

    Stores null data as a contiguous (n_maps, n_perm, n_parcels) 3-D array.
    Supports optional memory-mapping for large permutation counts.

    Covers two null types:

    - ``null_type="spatial"``: spatially-constrained permuted maps (moran, spin, burt, random).
      Stored in ``NiSpace._nulls["maps_null"]``.
    - ``null_type="group"``: transform-derived contrast maps from group label permutation
      (cohend, mean, centile, ...).
      Stored in ``NiSpace._nulls["groups_null"]``.

    Parameters
    ----------
    data : np.ndarray, shape (n_maps, n_perm, n_parcels)
        Must be exactly 3-D.  Use ``NullMaps.from_dict()`` to convert a legacy dict.
    labels : list of str
        Map labels; ``len(labels)`` must equal ``data.shape[0]``.
    dtype : dtype-like, optional
        If given, data is cast to this dtype on construction.
    null_method : str, optional
        The method that generated these nulls.
        Spatial: ``"moran"``, ``"random"``, ``"burt2018"``, ``"burt2020"``,
        ``"alexander_bloch"``, ``"spin"``, ``"vasa"``, ``"hungarian"``
        (and their aliases ``"brainspace"``, ``"brainsmash"``, ``"variogram"``).
        Group: the transform name, e.g. ``"cohend"``, ``"mean"``, ``"centile"``.
        Future spatial: ``"spin+moran"`` (Stage 2 / issue #44).
    null_type : {"spatial", "group"}, default "spatial"
        ``"spatial"`` for autocorrelation-preserving null maps;
        ``"group"`` for group-permutation contrast maps.
    memmap_path : str, Path, or True, optional
        Memory-map the backing array.  ``True`` creates a temp file automatically.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: list,
        dtype=None,
        null_method=None,
        null_type: str = "spatial",
        null_which=None,
        memmap_path=None,
    ):
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be np.ndarray, got {type(data)}")
        if data.ndim != 3:
            raise ValueError(
                f"NullMaps data must be 3-D (n_maps, n_perm, n_parcels), "
                f"got shape {data.shape}.  Use NullMaps.from_dict() for dict input."
            )
        if len(labels) != data.shape[0]:
            raise ValueError(
                f"len(labels) ({len(labels)}) must equal data.shape[0] ({data.shape[0]})"
            )

        if dtype is not None:
            data = data.astype(dtype, copy=False)

        self._data = data
        self._labels = list(labels)
        self._label_to_idx = {lbl: i for i, lbl in enumerate(self._labels)}
        self.null_method = null_method
        self.null_type = null_type
        self.null_which = null_which  # "X" | "Y" | None — identifies which maps these nulls are for

        if not isinstance(data, np.memmap) and data.nbytes > _NULLMAPS_WARN_BYTES:
            lgr.warning(
                f"NullMaps is {data.nbytes / 1e9:.1f} GB in memory. "
                f"Consider passing memmap_path=True or a file path to reduce RAM usage."
            )

        if memmap_path is not None:
            if memmap_path is True:
                import os, tempfile
                memmap_path = os.path.join(tempfile.gettempdir(), "nispace_nullmaps.npy")
            self.to_memmap(memmap_path)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        d: dict,
        dtype=None,
        memmap_path=None,
        null_method=None,
        null_type: str = "spatial",
        null_which=None,
    ) -> "NullMaps":
        """
        Build a NullMaps from a legacy ``{label: (n_perm, n_parcels) array}`` dict.

        Parameters
        ----------
        d : dict
        dtype : dtype-like, optional
        memmap_path : str, Path, or True, optional
        null_method : str, optional
        null_type : str, optional
        null_which : str, optional
        """
        labels = list(d.keys())
        # np.stack produces (n_maps, n_perm, n_parcels); works for len(d) == 1
        data = np.stack([np.asarray(d[lbl]) for lbl in labels])
        return cls(
            data,
            labels,
            dtype=dtype,
            memmap_path=memmap_path,
            null_method=null_method,
            null_type=null_type,
            null_which=null_which,
        )

    # ------------------------------------------------------------------
    # Shape / identity properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        """The raw (n_maps, n_perm, n_parcels) array (may be a memmap)."""
        return self._data

    @property
    def labels(self) -> list:
        return self._labels

    @property
    def n_maps(self) -> int:
        return self._data.shape[0]

    @property
    def n_perm(self) -> int:
        return self._data.shape[1]

    @property
    def n_parcels(self) -> int:
        return self._data.shape[2]

    @property
    def shape(self) -> tuple:
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    @property
    def is_memmap(self) -> bool:
        return isinstance(self._data, np.memmap)

    # ------------------------------------------------------------------
    # Dict-like interface (backward compat for external callers)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_maps

    def __contains__(self, label) -> bool:
        return label in self._label_to_idx

    def __getitem__(self, label) -> np.ndarray:
        """Return 2-D view (n_perm, n_parcels) for a single label."""
        try:
            return self._data[self._label_to_idx[label]]
        except KeyError:
            raise KeyError(
                f"Label '{label}' not in NullMaps. "
                f"Available: {self._labels[:5]}{'...' if len(self._labels) > 5 else ''}"
            )

    def keys(self):
        return iter(self._labels)

    def values(self):
        """Iterate 2-D (n_perm, n_parcels) views, one per map."""
        return iter(self._data)

    def items(self):
        return zip(self._labels, self._data)

    @property
    def null_method_str(self) -> str:
        """String representation of null_method; tuple displayed as 'cx+sc'."""
        if self.null_method is None:
            return ""
        if isinstance(self.null_method, tuple):
            return "+".join(str(m) for m in self.null_method)
        return str(self.null_method)

    def __repr__(self) -> str:
        kind = "memmap" if self.is_memmap else "array"
        return (
            f"NullMaps({kind}, null_type={self.null_type!r}, "
            f"null_method={self.null_method_str!r}, null_which={self.null_which!r}, "
            f"n_maps={self.n_maps}, n_perm={self.n_perm}, "
            f"n_parcels={self.n_parcels}, dtype={self.dtype})"
        )

    # ------------------------------------------------------------------
    # Array operations (return new NullMaps, never mutate in-place)
    # ------------------------------------------------------------------

    def astype(self, dtype) -> "NullMaps":
        """Return a new NullMaps with data cast to *dtype*."""
        return NullMaps(
            self._data.astype(dtype),
            self._labels,
            null_method=self.null_method,
            null_type=self.null_type,
            null_which=self.null_which,
        )

    def standardize(self) -> "NullMaps":
        """
        Z-score each (perm, parcel) slice along the parcel axis (axis=2).

        Equivalent to the old per-map loop::

            {k: zscore_df(null_maps[k], along="rows") for k in null_maps}

        which z-scored each permutation's parcel values across parcels.
        Only called for spatial null maps (not group null maps).

        Returns a new NullMaps backed by a plain (non-memmap) array.
        """
        std_data = scipy_zscore(
            self._data, axis=2, nan_policy="omit"
        ).astype(self._data.dtype)
        return NullMaps(
            std_data,
            self._labels,
            null_method=self.null_method,
            null_type=self.null_type,
            null_which=self.null_which,
        )

    def subset(self, labels) -> "NullMaps":
        """Return a new NullMaps containing only the requested labels (order preserved)."""
        idc = [self._label_to_idx[lbl] for lbl in labels]
        return NullMaps(
            self._data[idc],
            list(labels),
            null_method=self.null_method,
            null_type=self.null_type,
            null_which=self.null_which,
        )

    @classmethod
    def merge(cls, *null_maps_list, order: list = None) -> "NullMaps":
        """
        Concatenate multiple NullMaps along the map axis (axis 0).

        All inputs must share the same (n_perm, n_parcels).  If *order* is given,
        the merged result is re-ordered to match via ``subset()``.

        Note: this classmethod is available but is NOT wired into ``_get_null_maps``.
        Null maps are always regenerated on a partial cache miss.
        """
        shapes = [(nm.n_perm, nm.n_parcels) for nm in null_maps_list]
        if len(set(shapes)) != 1:
            raise ValueError(
                f"Cannot merge NullMaps with different (n_perm, n_parcels): {shapes}"
            )
        all_labels = [lbl for nm in null_maps_list for lbl in nm.labels]
        if len(all_labels) != len(set(all_labels)):
            raise ValueError("Duplicate labels across NullMaps — cannot merge.")
        all_data = np.concatenate([nm.data for nm in null_maps_list], axis=0)
        methods = {nm.null_method for nm in null_maps_list}
        null_method = methods.pop() if len(methods) == 1 else None
        types = {nm.null_type for nm in null_maps_list}
        null_type = types.pop() if len(types) == 1 else "spatial"
        which_set = {nm.null_which for nm in null_maps_list}
        null_which = which_set.pop() if len(which_set) == 1 else None
        merged = cls(all_data, all_labels, null_method=null_method, null_type=null_type,
                     null_which=null_which)
        if order is not None:
            merged = merged.subset(order)
        return merged

    # ------------------------------------------------------------------
    # Bulk numpy access (hot path in api.py)
    # ------------------------------------------------------------------

    def iter_perms(self):
        """
        Iterate over permutations.

        Yields 2-D arrays of shape (n_maps, n_parcels), one per permutation.
        """
        return iter(self._data.transpose(1, 0, 2))

    def perm_list(self, dtype=None) -> list:
        """
        Return a list of *n_perm* arrays, each of shape (n_maps, n_parcels).

        Direct replacement for the hot-path list comprehension in api.py::

            [np.c_[[maps[i,:] for maps in d.values()]] for i in range(n_perm)]

        Parameters
        ----------
        dtype : dtype-like, optional
            If given and different from the stored dtype, each array is cast before return.
            If the stored dtype already matches, views are returned without copying.
        """
        arr = self._data.transpose(1, 0, 2)
        if dtype is not None and np.dtype(dtype) != self._data.dtype:
            return [a.astype(dtype) for a in arr]
        return list(arr)

    # ------------------------------------------------------------------
    # Memmap support
    # ------------------------------------------------------------------

    def to_memmap(self, path) -> None:
        """
        Write the current array to a .npy memmap file at *path* and replace
        the internal backing array with the memmap view (in-place).

        Pass ``True`` to use an automatically-created temp file instead.
        """
        if path is True:
            import os, tempfile
            path = os.path.join(tempfile.gettempdir(), "nispace_nullmaps.npy")
        path = str(path)
        mm = np.lib.format.open_memmap(
            path, mode="w+", dtype=self._data.dtype, shape=self._data.shape
        )
        mm[:] = self._data
        mm.flush()
        self._data = mm
        lgr.info(f"NullMaps memory-mapped to {path} ({self.nbytes / 1e9:.2f} GB).")

    def flush(self) -> None:
        """Flush pending memmap writes (no-op if not a memmap)."""
        if self.is_memmap:
            self._data.flush()

    def to_array(self) -> np.ndarray:
        """Return a plain (non-memmap) ndarray copy of the data."""
        return np.array(self._data)

    # ------------------------------------------------------------------
    # Pickle / deepcopy support
    # ------------------------------------------------------------------

    def __deepcopy__(self, memo):
        """deepcopy always converts memmap → plain array so the copy is self-contained."""
        new = NullMaps.__new__(NullMaps)
        new._data = self.to_array()
        new._labels = copy.deepcopy(self._labels, memo)
        new._label_to_idx = {lbl: i for i, lbl in enumerate(new._labels)}
        new.null_method = self.null_method
        new.null_type = self.null_type
        new.null_which = self.null_which
        return new

    def __getstate__(self):
        """Pickle: always serialize as plain array, never a dangling memmap path."""
        return {
            "data": self.to_array(),
            "labels": self._labels,
            "null_method": self.null_method,
            "null_type": self.null_type,
            "null_which": self.null_which,
        }

    def __setstate__(self, state):
        self._data = state["data"]
        self._labels = state["labels"]
        self._label_to_idx = {lbl: i for i, lbl in enumerate(self._labels)}
        self.null_method = state.get("null_method")
        self.null_type = state.get("null_type", "spatial")  # default for old pickles
        self.null_which = state.get("null_which")           # default for old pickles
