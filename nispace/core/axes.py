"""Named spatial axes: tissue, geometry, parcel shape, compartment and cortical-hierarchy maps
per parcel.

One registry shared by :func:`nispace.datasets.fetch_axes`, ``NiSpace(z=...)`` and
:func:`nispace.diagnostics.shared_axes`, so all three build identical maps. Design notes:
``_explore/shared_axes_design.md``, parcel shape: ``_explore/parcel_geometry_axes/NOTES.md``.

Geometry/parcel-shape/compartment/MEM axes are computed from the parcellation itself; tissue maps
come from the ``tpm`` dataset and cortical-hierarchy maps from the ``cortexfeatures`` dataset.
"""

import copy
import re

import nibabel as nib
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
    # parcel shape (parcellation image in the active space)
    "size":       dict(kind="parcelgeom",
                       desc="log number of voxels (volume) or vertices (surface): parcel size "
                            "and amount of data per parcel."),
    "area":       dict(kind="parcelgeom",
                       desc="log surface area (mm², population-average vertex areas); "
                            "surface only."),
    "dispersion": dict(kind="parcelgeom",
                       desc="log mean squared distance to the parcel centroid relative to an "
                            "equal-size ball (volume) or disc (surface sphere); 0 = round."),
    "elongation": dict(kind="parcelgeom",
                       desc="log sqrt(lambda1 / lambda2) of the coordinate covariance; "
                            "0 = no main direction."),
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
    "sulc":        dict(kind="cortexfeatures", map="feature-sulc_pub-hcps1200",
                        desc="Sulcal depth (HCP S1200); HCP sign, - = sulcal, + = gyral."),
    "curv":        dict(kind="cortexfeatures", map="feature-curv_pub-hcps1200",
                        desc="Mean curvature (HCP S1200); HCP sign, - = sulcal, + = gyral."),
}
_AXIS_GROUPS = {"coords": ["ap", "si", "lr"],
                "parcelgeom": ["size", "area", "dispersion", "elongation"]}
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


def resolve_axis_names(axes, parc=None, space=None):
    """Expand `axes` into a list of registry names.

    Parameters
    ----------
    axes : str or list of str
        ``"default"``, ``"all"``, a registry name, a group name (``"coords"``,
        ``"parcelgeom"``; the latter includes ``"area"`` only for surface parcellations),
        ``"memK"``, or a list of these. Case-insensitive.
    parc : Parcellation, optional
        Needed to resolve ``"default"`` (depends on the parcellation type) and
        ``"parcelgeom"``.
    space : str, optional
        Space of the parcel-shape axes (see :func:`build_axes`).

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
        if a == "parcelgeom" and parc is not None and _geom_space(parc, space)[1] != "surface":
            # group: drop what can't be built for every parcel
            expanded = [n for n in expanded if n != "area"]
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


def _geom_space(parc, space=None):
    """Space for parcel-shape axes: `space`, else the active (data) space, else the centroid
    default. Returns (space name, kind) with kind "volume", "surface", or "combined_surface"
    (combined parcellation: cortex from its stored surface, subcortex from the MNI volume)."""
    from .parcellation import _spaces_match
    if space is None:
        space = parc._space or parc._centroid_space()
    elif parc._is_combined and space not in parc.spaces:
        cx_space = next((s for s in parc._cx_surface if _spaces_match(space, s)), None)
        if cx_space is None:
            raise AxisUnavailableError(
                f"Parcel shape: no '{space}' image for combined parcellation '{parc._name}' "
                f"(MNI: {parc.spaces}, cortex surfaces: {list(parc._cx_surface)}).")
        return cx_space, "combined_surface"
    else:
        space = parc.get_image_for_dataspace(space)
    surface = isinstance(parc.get_image(space), (tuple, nib.GiftiImage))
    return space, "surface" if surface else "volume"


def _vertex_area(coords, faces):
    """Area per vertex: one third of each adjacent triangle."""
    a, b, c = coords[faces[:, 0]], coords[faces[:, 1]], coords[faces[:, 2]]
    tri = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return np.bincount(faces.ravel(), weights=np.repeat(tri / 3, 3), minlength=len(coords))


def _shape_stats(vals, xyz, w):
    """Per parcel value: weight sum, trace and two largest eigenvalues of the weighted
    coordinate covariance (bincount moments, one pass over all points)."""
    uniq, pos = np.unique(vals, return_inverse=True)
    k = len(uniq)
    wsum = np.bincount(pos, weights=w, minlength=k)
    mean = np.column_stack([np.bincount(pos, weights=w * xyz[:, d], minlength=k)
                            for d in range(3)]) / wsum[:, None]
    cov = np.empty((k, 3, 3))
    for d in range(3):
        for e in range(d, 3):
            m2 = np.bincount(pos, weights=w * xyz[:, d] * xyz[:, e], minlength=k) / wsum
            cov[:, d, e] = cov[:, e, d] = m2 - mean[:, d] * mean[:, e]
    lam = np.linalg.eigvalsh(cov)[:, ::-1]
    return uniq, wsum, np.trace(cov, axis1=1, axis2=2), lam[:, 0], lam[:, 1]


def _surface_blocks(imgs, hemis, space):
    """(values, count, area, dispersion, elongation) per hemisphere of a surface parcellation."""
    from ..nulls import _img_density_for_neuromaps
    from ..datasets import fetch_template
    from neuromaps.images import load_data
    density = _img_density_for_neuromaps(imgs[0])
    sph = fetch_template(space, desc="sphere", res=density, verbose=False)
    vaavg = fetch_template(space, desc="vaavg", res=density,
                           verbose=False)
    blocks = []
    for img_h, hemi in zip(imgs, hemis):
        i_h = ("L", "R").index(hemi)
        v = np.asarray(load_data(img_h)).astype(int).ravel()
        g = nib.load(sph[i_h])
        xyz, faces = np.asarray(g.darrays[0].data, dtype=float), g.darrays[1].data
        w = _vertex_area(xyz, faces)
        area = np.asarray(nib.load(vaavg[i_h]).darrays[0].data, dtype=float)
        keep = v != 0
        uniq, a_sph, rg2, l1, l2 = _shape_stats(v[keep], xyz[keep], w[keep])
        pos = np.searchsorted(uniq, v[keep])
        count = np.bincount(pos, minlength=len(uniq))
        area_p = np.bincount(pos, weights=area[keep], minlength=len(uniq))
        blocks.append((uniq, count, area_p, rg2 / (a_sph / (2 * np.pi)), np.sqrt(l1 / l2)))
    return blocks


def _volume_blocks(img, bilateral):
    """(values, count, area [NaN], dispersion, elongation) of a volumetric parcellation; one
    block, or two for bilateral images (values shared across hemispheres, split at x = 0)."""
    data = np.rint(np.asarray(img.dataobj)).astype(int)
    ijk = np.nonzero(data)
    vals = data[ijk]
    xyz = nib.affines.apply_affine(img.affine, np.column_stack(ijk))
    vox = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
    sides = [xyz[:, 0] < 0, xyz[:, 0] >= 0] if bilateral else [np.ones(len(vals), bool)]
    blocks = []
    for m in sides:
        uniq, count, rg2, l1, l2 = _shape_stats(vals[m], xyz[m], np.ones(m.sum()))
        r2 = (3 * count * np.prod(vox) / (4 * np.pi)) ** (2 / 3)
        # + each voxel's own inertia (a²/12 per axis), relevant for tiny parcels only
        disp = (rg2 + (vox ** 2).sum() / 12) / (0.6 * r2)
        blocks.append((uniq, count, np.full(len(uniq), np.nan), disp, np.sqrt(l1 / l2)))
    return blocks


def _blocks_to_stats(blocks, pool):
    """Rows in the parcellater's order (as get_centroids): blocks concatenated (surface: L then
    R), or pooled by value (volumes, bilateral): counts/areas summed, shape count-weighted."""
    if not pool:
        return [np.concatenate([b[i] for b in blocks]) for i in range(1, 5)]
    uniq = np.unique(np.concatenate([b[0] for b in blocks]))
    acc = np.zeros((len(uniq), 4))
    area_nan = all(np.isnan(b[2]).all() for b in blocks)
    for b_vals, count, area_p, disp, elong in blocks:
        i = np.searchsorted(uniq, b_vals)
        acc[i, 0] += count
        acc[i, 1] += np.nan_to_num(area_p)
        acc[i, 2] += count * disp
        acc[i, 3] += count * elong
    area_p = np.full(len(uniq), np.nan) if area_nan else acc[:, 1]
    return acc[:, 0], area_p, acc[:, 2] / acc[:, 0], acc[:, 3] / acc[:, 0]


def _build_parcelgeom(parc, names, space=None):
    """Parcel size and shape from the parcellation image in `space` (default: active space).

    Volumes: voxel coordinates (mm). Surfaces: sphere-mesh coordinates weighted by vertex area,
    so cortical folding does not enter the shape; ``area`` sums the population-average vertex
    areas (``vaavg``). ``dispersion`` = R_g² / R_g²(ideal) with R_g² the mean squared distance to
    the centroid and the ideal an equal-size ball (3/5 r²) or disc (r²/2). Bilateral
    parcellations: computed per hemisphere, then counts/areas summed and shape averaged
    (weighted by count). All four are log-transformed (right-skewed, ratios >= 1).

    Combined parcellation with a surface `space`: cortex from the stored cortex surface,
    subcortex from the MNI volume. Vertex and voxel values are not comparable, so the log values
    are z-scored within each compartment (``area``: cortex only, subcortex NaN).
    """
    from ..io import load_img
    space, kind = _geom_space(parc, space)
    n = len(parc._labels)
    if kind == "combined_surface":
        if parc._bilateral:
            raise AxisUnavailableError("Parcel shape from the cortex surface is not available for "
                                       "bilateral combined parcellations; use an MNI space.")
        entry = parc._cx_surface[space]
        if entry.get("image") is None:
            entry["image"] = load_img(entry["img_paths"])
        cx = _blocks_to_stats(_surface_blocks(entry["image"], ("L", "R"), space), pool=False)
        vol_space = parc.get_image_for_dataspace(None)
        vol = _blocks_to_stats(_volume_blocks(parc.get_image(vol_space), False), pool=True)
        sc_idc = np.asarray(parc.get_sc_idc(), dtype=int)
        cx_idc = np.setdiff1d(np.arange(n), sc_idc)
        if len(cx[0]) != len(cx_idc) or len(vol[0]) != n:
            lgr.critical_raise(f"Parcellation '{parc._name}': {len(cx[0])} cortex surface and "
                               f"{len(vol[0])} volume parcels for {len(cx_idc)} cortex / {n} "
                               "labels.", ValueError)
        lgr.info(f"Parcel shape: cortex from '{space}' surface, subcortex from '{vol_space}' "
                 "volume; log values z-scored within each compartment.")
        out = []
        for i in range(4):
            vals = np.full(n, np.nan)
            vals[cx_idc] = np.log(cx[i])
            vals[sc_idc] = np.log(vol[i][sc_idc])
            for idc in (cx_idc, sc_idc):
                v = vals[idc]
                vals[idc] = (v - np.nanmean(v)) / np.nanstd(v) if np.isfinite(v).any() else v
            out.append(vals)
        stats = dict(zip(["size", "area", "dispersion", "elongation"], out))
        return [stats[n_] for n_ in names]

    img = parc.get_image(space)
    parc._fit_space(space)
    if kind == "surface":
        imgs = img if isinstance(img, tuple) else (img,)
        hemis = parc._hemi_dict[space] or ("L", "R")[:len(imgs)]
        stats = _blocks_to_stats(_surface_blocks(imgs, hemis, space), pool=parc._bilateral)
    else:
        stats = _blocks_to_stats(_volume_blocks(img, parc._bilateral), pool=True)
    if len(stats[0]) != n:
        lgr.critical_raise(f"Parcellation '{parc._name}' ({space}): {len(stats[0])} parcel "
                           f"values in the image but {n} labels.", ValueError)
    stats = dict(zip(["size", "area", "dispersion", "elongation"], stats))
    return [np.log(stats[n_]) for n_ in names]


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
               parcellate_kwargs=None, space=None):
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
    space : str, optional
        Space of the parcel-shape axes (``size``, ``area``, ``dispersion``, ``elongation``).
        Default: the parcellation's active space, else its centroid default (MNI first). Use it
        when the data were parcellated in another space than the active one, e.g. tables of
        surface datasets. For a combined parcellation, a surface space takes the cortex from
        the stored cortex surface and the subcortex from the MNI volume (log values z-scored
        within each compartment).

    Returns
    -------
    pd.DataFrame (n_axes, n_parcels)
        Index = axis names. Axes unavailable for this parcellation are skipped (with an info
        message) for ``"default"``/``"all"`` and raise :class:`AxisUnavailableError` otherwise.
    """
    if parc is None:
        lgr.critical_raise("Axis maps require a parcellation.", ValueError)
    names, explicit = resolve_axis_names(axes, parc, space=space)
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
    pg = [a for a in names if _axis_kind(a) == "parcelgeom"]
    if pg:
        try:
            g_space, g_kind = _geom_space(parc, space)
            if "area" in pg and g_kind == "volume":
                _unavailable(AxisUnavailableError(f"'area' needs a surface parcellation (space: "
                                                  f"'{g_space}')."))
                pg.remove("area")
            if pg:
                built.update(zip(pg, _build_parcelgeom(parc, pg, space)))
        except AxisUnavailableError as err:
            _unavailable(err)
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
