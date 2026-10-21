"""Named spatial axes: tissue, geometry, compartment and cortical-hierarchy maps per parcel.

One registry shared by :func:`nispace.datasets.fetch_axes`, ``NiSpace(z=...)`` and
:func:`nispace.diagnostics.shared_axes`, so all three build identical maps. Design notes:
``_explore/shared_axes_design.md``.

Geometry/compartment/MEM axes are computed from the parcellation itself; tissue maps come from
the ``tpm`` dataset and cortical-hierarchy maps from the ``cortexfeatures`` dataset.
"""

import copy
import re

import numpy as np
import pandas as pd

import logging
lgr = logging.getLogger(__name__)

from .constants import _SPACE_DEFAULT_VOL


# MNI152 whole-brain-mask centre of mass, rounded (6Asym: 0.6/-21.5/9.8; 2009cAsym: -0.5/-22.1/9.5).
# One fixed centre for every space: per-space centres differ by only 1-3 mm, while cortex-based
# centres (surfaces) sit ~7 mm higher, which would make `radial` depend on the parcellation type.
_RADIAL_CENTRE = np.array([0.0, -20.0, 10.0])

_AXES = {
    # tissue / vessel probability ("tpm" dataset)
    "gm":       dict(kind="tpm", desc="Grey-matter probability (SPM)."),
    "wm":       dict(kind="tpm", desc="White-matter probability (SPM)."),
    "csf":      dict(kind="tpm", desc="CSF probability (SPM)."),
    "veins":    dict(kind="tpm", desc="Venous probability (Huck 2019)."),
    "arteries": dict(kind="tpm", desc="Arterial probability (Mouches 2019)."),
    # geometry (parcel centroids)
    "radial":   dict(kind="geometry",
                     desc="Distance (mm) of the parcel centroid from the brain centre "
                          "(MNI 0, -20, 10); + = peripheral."),
    "ap":       dict(kind="geometry", desc="Centroid MNI y; + = anterior."),
    "si":       dict(kind="geometry", desc="Centroid MNI z; + = superior."),
    "lr":       dict(kind="geometry", desc="Centroid MNI x; + = right."),
    "medlat":   dict(kind="geometry", desc="Distance from the midline, |x|; + = lateral."),
    # compartment
    "cxsc":     dict(kind="compartment",
                     desc="Compartment indicator, 1 = subcortex, 0 = cortex (combined parcellations)."),
    # cortical hierarchy ("cortexfeatures" dataset, cortex only)
    "saaxis":      dict(kind="cortexfeatures", map="feature-saaxis_pub-sydnor2021",
                        desc="Sensorimotor-association axis (Sydnor 2021)."),
    "fcgradient1": dict(kind="cortexfeatures", map="feature-fcgradient1_pub-margulies2016",
                        desc="Principal FC gradient, unimodal -> transmodal (Margulies 2016)."),
    "fcgradient2": dict(kind="cortexfeatures", map="feature-fcgradient2_pub-margulies2016",
                        desc="Second FC gradient, visual -> somatomotor (Margulies 2016)."),
    "t1t2":        dict(kind="cortexfeatures", map="feature-t1t2_pub-hcps1200",
                        desc="T1w/T2w ratio, myelin proxy (HCP S1200)."),
    "thickness":   dict(kind="cortexfeatures", map="feature-thickness_pub-hcps1200",
                        desc="Cortical thickness (HCP S1200)."),
}
_AXIS_GROUPS = {"coords": ["ap", "si", "lr"]}
_MEM_PATTERN = re.compile(r"^mem([1-9]\d*)$")
_TPM_AXES = frozenset(k for k, v in _AXES.items() if v["kind"] == "tpm")
_DEFAULT_AXES = ["gm", "radial", "ap", "si"]
_DEFAULT_AXES_COMBINED = ["cxsc"]
_DEFAULT_AXES_CORTEX = ["saaxis", "fcgradient1", "fcgradient2"]


class AxisUnavailableError(ValueError):
    """An axis exists in the registry but cannot be built for this parcellation."""


def _axis_kind(name):
    if name in _AXES:
        return _AXES[name]["kind"]
    if _MEM_PATTERN.match(name):
        return "mem"
    return None


def is_axis_name(name):
    """True if `name` is a registry axis or axis group (case-insensitive)."""
    if not isinstance(name, str):
        return False
    name = name.lower()
    return name in _AXIS_GROUPS or _axis_kind(name) is not None


def _valid_names_msg():
    return (f"Valid axes: {list(_AXES)}, groups: {list(_AXIS_GROUPS)}, "
            "Moran eigenvectors: 'mem1', 'mem2', ...; or 'default' / 'all'.")


def _library_parc_name(parc):
    """Integrated-library name of `parc` (str, or (cx, sc) tuple for combined), else None."""
    from ..datasets import _check_parcellation
    if parc is None:
        return None
    name = (parc._cx_name, parc._sc_name) if parc._is_combined else parc._name
    if name is None:
        return None
    return _check_parcellation(name, raise_not_found=False)


def _cortexfeatures_table(parc):
    """Name of the cortexfeatures table for `parc` (cortex part for combined), else None."""
    from ..datasets import reference_lib
    lib_name = _library_parc_name(parc)
    if lib_name is None:
        return None
    cx_name = lib_name[0] if isinstance(lib_name, (list, tuple)) else lib_name
    return cx_name if cx_name in reference_lib.get("cortexfeatures", {}).get("tab", {}) else None


def resolve_axis_names(axes, parc=None):
    """Expand `axes` into a list of registry names.

    Parameters
    ----------
    axes : str or list of str
        ``"default"``, ``"all"``, a registry name, a group name (``"coords"``), ``"memK"``, or a
        list of these. Case-insensitive.
    parc : Parcellation, optional
        Needed to resolve ``"default"`` (depends on the parcellation type).

    Returns
    -------
    names : list of str
        Unique names in the given order.
    explicit : bool
        False for ``"default"``/``"all"`` (unavailable axes may be skipped), True otherwise.
    """
    if isinstance(axes, str):
        axes = [axes]
    axes = [a.lower() if isinstance(a, str) else a for a in axes]
    if axes == ["default"]:
        names = list(_DEFAULT_AXES)
        if parc is not None and parc._is_combined:
            names += _DEFAULT_AXES_COMBINED
        elif parc is not None and _cortexfeatures_table(parc) is not None:
            names += _DEFAULT_AXES_CORTEX
        return names, False
    if axes == ["all"]:
        return list(_AXES), False

    names = []
    for a in axes:
        if not isinstance(a, str):
            lgr.critical_raise(f"Axis names must be strings, got {type(a)}. {_valid_names_msg()}",
                               ValueError)
        expanded = _AXIS_GROUPS.get(a, [a])
        for n in expanded:
            if _axis_kind(n) is None:
                lgr.critical_raise(f"Unknown axis '{n}'. {_valid_names_msg()}", ValueError)
            if n not in names:
                names.append(n)
    return names, True


def _match_tpm(items, names, key=str):
    """Order fetched TPM items (paths or index labels) by requested `names`.

    `fetch_reference` returns maps in library order (gm, wm, csf, arteries, veins), not in the
    requested order, so rows must be matched by name, never assigned positionally.
    """
    out = []
    for n in names:
        hits = [it for it in items if f"tissue-{n}_" in key(it)]
        if len(hits) != 1:
            lgr.critical_raise(f"Could not identify the TPM map for '{n}' (matches: {hits}).",
                               RuntimeError)
        out.append(hits[0])
    return out


def _to_parc_order(df, labels, what):
    """Align a fetched (maps x parcels) table to the parcellation's labels."""
    labels = [str(l) for l in labels]
    cols = [str(c) for c in df.columns]
    if set(labels) <= set(cols):
        df = df.copy()
        df.columns = cols
        return df[labels].to_numpy(dtype=float)
    if set(cols) & set(labels):  # partial overlap: e.g. cortex-only table for a combined parc
        df = df.copy()
        df.columns = cols
        return df.reindex(columns=labels).to_numpy(dtype=float)
    # no shared label: never fall back to column position, it can silently misassign values
    lgr.critical_raise(f"{what}: none of the table's parcel labels match the parcellation "
                       f"({len(cols)} columns, {len(labels)} parcels; e.g. '{cols[0]}' vs "
                       f"'{labels[0]}').", RuntimeError)


def _build_tpm(parc, names, verbose, parcellate_kwargs=None):
    from ..datasets import fetch_reference
    from ..io import parcellate_data
    if _library_parc_name(parc) is not None:
        tab = fetch_reference("tpm", maps=names, parcellation=parc, hemi=parc._selected_hemi,
                              bilateral=parc._bilateral, standardize_parcellated=False,
                              print_references=False, verbose=verbose)
        tab = tab.loc[_match_tpm(list(tab.index), names)]
        return _to_parc_order(tab, parc._labels, "tpm")
    # custom parcellation: parcellate the raw TPM images (fixed canonical volume space)
    paths = fetch_reference("tpm", maps=names, space=_SPACE_DEFAULT_VOL,
                            print_references=False, verbose=verbose)
    paths = _match_tpm(paths, names, key=lambda p: str(p))
    parc_act = copy.copy(parc)  # don't change the caller's active space
    if parc_act._space is None:
        parc_act.set_active_space(parc_act.get_image_for_dataspace(_SPACE_DEFAULT_VOL))
    tab = parcellate_data(paths, data_labels=names, data_space=_SPACE_DEFAULT_VOL,
                          parcellation=parc_act, verbose=verbose, **(parcellate_kwargs or {}))
    return tab.to_numpy(dtype=float)


def _build_geometry(centroids, name):
    x, y, z = centroids.T
    if name == "radial":
        return np.sqrt(((centroids - _RADIAL_CENTRE) ** 2).sum(axis=1))
    return {"ap": y, "si": z, "lr": x, "medlat": np.abs(x)}[name]


def _build_cxsc(parc):
    if not parc._is_combined:
        raise AxisUnavailableError("'cxsc' needs a combined cortex+subcortex parcellation.")
    out = np.zeros(len(parc._labels))
    out[np.asarray(parc.get_sc_idc(), dtype=int)] = 1.0
    return out


def _build_mems(centroids, ks):
    """Moran eigenvectors of the 1/d weight matrix between parcel centroids.

    Centroid distances (not the null's voxel-averaged/geodesic matrices) give one whole-map basis
    for every parcellation type. Signs are arbitrary; fixed so the largest |value| is positive.
    """
    from ..nulls import _moran_fit_1_over_d
    n = len(centroids)
    k_max = max(ks)
    if k_max > n - 2:
        raise AxisUnavailableError(f"'mem{k_max}' requested but only {n} parcels.")
    dist = np.sqrt(((centroids[:, None, :] - centroids[None, :, :]) ** 2).sum(-1))
    if (dist[~np.eye(n, dtype=bool)] == 0).any():
        raise AxisUnavailableError("Two parcels share a centroid; Moran eigenvectors undefined.")
    mem, _ = _moran_fit_1_over_d(dist, n_components=k_max)
    mem = np.asarray(mem, dtype=float)
    out = {}
    for k in ks:
        v = mem[:, k - 1]
        out[k] = v * np.sign(v[np.argmax(np.abs(v))])
    return out


def _build_cortexfeatures(parc, names, verbose):
    from ..datasets import fetch_reference
    table = _cortexfeatures_table(parc)
    if table is None:
        raise AxisUnavailableError(
            f"{names} need an integrated cortex parcellation with a 'cortexfeatures' table "
            f"(parcellation: '{parc._name}').")
    tab = fetch_reference("cortexfeatures", maps=[_AXES[n]["map"] for n in names],
                          parcellation=table, bilateral=parc._bilateral,
                          print_references=False, verbose=verbose)
    tab = tab.loc[[_AXES[n]["map"] for n in names]]
    return _to_parc_order(tab, parc._labels, "cortexfeatures")


def build_axes(parc, axes="default", columns=None, require_complete=False, verbose=False,
               parcellate_kwargs=None):
    """Build axis maps for a parcellation.

    Parameters
    ----------
    parc : Parcellation
        Integrated or custom parcellation object.
    axes : str or list of str
        See :func:`resolve_axis_names`.
    columns : sequence, optional
        Column labels of the output (must match the number of parcels). Default: parcel labels.
    require_complete : bool, default False
        Raise if an axis is undefined (NaN) for any parcel, e.g. a cortex-only map on a combined
        parcellation. ``NiSpace(z=...)`` sets this: NaN in Z would silently drop those parcels
        from every analysis.
    verbose : bool, default False
    parcellate_kwargs : dict, optional
        Extra :func:`~nispace.io.parcellate_data` arguments for the one case that parcellates
        images: TPM axes on a custom parcellation. ``NiSpace(z=...)`` passes its own settings
        (e.g. ``resampling_target``, the z ``background_value``) so Z is built like z data.

    Returns
    -------
    pd.DataFrame (n_axes, n_parcels)
        Index = axis names. Axes unavailable for this parcellation are skipped (with an info
        message) for ``"default"``/``"all"`` and raise :class:`AxisUnavailableError` otherwise.
    """
    if parc is None:
        lgr.critical_raise("Axis maps require a parcellation.", ValueError)
    names, explicit = resolve_axis_names(axes, parc)
    n = len(parc._labels)
    built, centroids = {}, None

    def _unavailable(err):
        if explicit:
            lgr.critical_raise(str(err), AxisUnavailableError)
        lgr.info(f"Skipping axis: {err}")

    tpm = [a for a in names if a in _TPM_AXES]
    if tpm:
        built.update(zip(tpm, _build_tpm(parc, tpm, verbose, parcellate_kwargs)))
    cf = [a for a in names if _axis_kind(a) == "cortexfeatures"]
    if cf:
        try:
            built.update(zip(cf, _build_cortexfeatures(parc, cf, verbose)))
        except AxisUnavailableError as err:
            _unavailable(err)
    geo = [a for a in names if _axis_kind(a) in ("geometry", "mem")]
    if geo:
        centroids = parc.get_centroids()
    for a in [a for a in names if _axis_kind(a) == "geometry"]:
        built[a] = _build_geometry(centroids, a)
    if "cxsc" in names:
        try:
            built["cxsc"] = _build_cxsc(parc)
        except AxisUnavailableError as err:
            _unavailable(err)
    mems = {a: int(_MEM_PATTERN.match(a).group(1)) for a in names if _axis_kind(a) == "mem"}
    if mems:
        try:
            vecs = _build_mems(centroids, sorted(set(mems.values())))
            built.update({a: vecs[k] for a, k in mems.items()})
        except AxisUnavailableError as err:
            _unavailable(err)

    names = [a for a in names if a in built]
    out = pd.DataFrame(np.vstack([built[a] for a in names]) if names else np.empty((0, n)),
                       index=pd.Index(names, name="axis"),
                       columns=parc._labels if columns is None else columns)
    if out.shape[1] != n:
        lgr.critical_raise(f"Got {out.shape[1]} columns for {n} parcels.", ValueError)

    n_nan = out.isna().sum(axis=1)
    if require_complete and (n_nan > 0).any():
        bad = {a: int(k) for a, k in n_nan.items() if k > 0}
        lgr.critical_raise(
            f"Axis map(s) undefined for some parcels: {bad} (of {n}). As covariates these would "
            "drop those parcels from every analysis; use a parcellation the axis covers fully, "
            "or pass the map with explicit handling as an array.", AxisUnavailableError)
    return out
