#!/usr/bin/env python3
"""Generate RST reference pages for NiSpace datasets and parcellations.

Called automatically from docs/conf.py during the Sphinx build.
Can also be run standalone for quick testing:
    python docs/gen_dataset_pages.py
"""
import json
from pathlib import Path

DOCS_DIR = Path(__file__).parent
REPO_DIR = DOCS_DIR.parent
DATALIB_DIR = REPO_DIR / "nispace" / "datalib"
OUT_DIR = DOCS_DIR / "_auto"

GITHUB_BASE = "https://github.com/leondlotter/nispace-data"

PARC_N_PARCELS = {
    "Schaefer100": 100, "Schaefer200": 200, "Schaefer400": 400,
    "TianS1": 16, "TianS2": 32, "TianS3": 50, "Glasser": 360,
    "DesikanKilliany": 68, "DesikanKillianyTourville": 84, "Destrieux": 148,
    "Aseg": 51, "HarvardOxfordCortical": 48, "HarvardOxfordSubcortical": 21,
}

DATASET_TITLES = {}
# DATASET_TITLES = {
#     "pet":           "PET – Neurotransmitter receptor/transporter maps",
#     "mrna":          "mRNA – Allen Human Brain Atlas gene expression",
#     "magicc":        "MAGICC – Continuous cortical gene expression (Wagstyl et al.)",
#     "rsn":           "RSN – Resting-state network probability maps",
#     "grf":           "GRF – Gaussian Random Field maps",
#     "neurosynth":    "Neurosynth – Meta-analytic cognitive function maps",
#     "cortexfeatures":"CortexFeatures – Cortical topology and physiology features",
#     "bigbrain":      "BigBrain – Histological features (BigBrainWarp)",
#     "tpm":           "TPM – Tissue probability maps",
#     "enigmathick":   "ENIGMAthick – ENIGMA cortical thickness effect size maps",
#     "enigmaarea":    "ENIGMAarea – ENIGMA surface area effect size maps"
# }

DATASET_ONE_LINERS = {
    "pet":           "PET receptor/transporter density maps; 52 tracers across major neurotransmitter systems",
    "mrna":          "Allen Human Brain Atlas mRNA expression; 15,000+ genes with gene-set collections",
    "magicc":        "AHBA gene expression mapped to continuous cortical space (Wagstyl et al., 2024)",
    "rsn":           "Resting-state network probability maps; 14 canonical networks (Dworetsky et al., 2021)",
    "grf":           "Gaussian Random Field maps with controlled spatial autocorrelation; for null model validation",
    "neurosynth":    "Meta-analytic z-maps for ~1000 cognitive terms from the Neurosynth database",
    "cortexfeatures":"Cortical topology features: thickness, T1w/T2w, SA axis, MEG, metabolism, FC gradients",
    "bigbrain":      "Histological depth features and layer thickness from the BigBrain atlas (Paquola et al., 2021)",
    "tpm":           "Tissue probability maps: grey/white matter, CSF, arteries, veins",
    "enigmathick":   "Cohen's d maps comparing cortical thickness between neuro-psychiatric disorders and controls",
    "enigmaarea":    "Cohen's d maps comparing surface area between neuro-psychiatric disorders and controls"
}


# ---------------------------------------------------------------------------
# GitHub URL helpers
# ---------------------------------------------------------------------------

def _gh_blob(remote):
    return f"{GITHUB_BASE}/blob/main/{remote}"

def _gh_tree(remote):
    return f"{GITHUB_BASE}/tree/main/{remote}"


# ---------------------------------------------------------------------------
# HTML building helpers
# ---------------------------------------------------------------------------

def _c(text):
    """Plain inline code."""
    return f"<code>{text}</code>"

def _cl(text, url):
    """Inline code with hyperlink."""
    return f'<code><a href="{url}">{text}</a></code>'

def _raw_html(html_lines):
    """Wrap HTML lines in a RST ``.. raw:: html`` block (returns list of RST lines)."""
    out = [".. raw:: html", ""]
    for element in html_lines:
        for line in str(element).split("\n"):
            out.append(f"   {line}" if line.strip() else "")
    out.append("")
    return out

def _html_table(headers, rows, css_class="nispace-table"):
    """Build an HTML table string."""
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = "\n    ".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        f'<table class="{css_class}">\n'
        f"  <thead><tr>{th}</tr></thead>\n"
        f"  <tbody>\n    {body}\n  </tbody>\n"
        f"</table>"
    )

def _desc_to_html(text):
    """Convert reference.txt description text to HTML, handling bullet lists."""
    html = []
    para = []
    in_list = False

    def flush_para():
        if para:
            html.append("<p>" + " ".join(para) + "</p>")
            para.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_para()
            if in_list:
                html.append("</ul>")
                in_list = False
        elif line.startswith("- "):
            flush_para()
            if not in_list:
                html.append("<ul>")
                in_list = True
            html.append(f"<li>{line[2:]}</li>")
        else:
            if in_list:
                html.append("</ul>")
                in_list = False
            para.append(line)

    flush_para()
    if in_list:
        html.append("</ul>")
    return "\n".join(html)


# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------

def _map_folder_url(map_data):
    """GitHub tree URL for map folder from first public github-nispace entry."""
    for space, val in map_data.items():
        if not isinstance(val, dict):
            continue
        if val.get("host") == "github-nispace":
            folder = "/".join(val["remote"].split("/")[:-1])
            return _gh_tree(folder)
        for hval in val.values():
            if isinstance(hval, dict) and hval.get("host") == "github-nispace":
                folder = "/".join(hval["remote"].split("/")[:-1])
                return _gh_tree(folder)
    return None

def _map_is_private(map_data):
    """True if map has no public github-nispace entry."""
    for space, val in map_data.items():
        if not isinstance(val, dict):
            continue
        if val.get("host") == "github-nispace":
            return False
        for hval in val.values():
            if isinstance(hval, dict) and hval.get("host") == "github-nispace":
                return False
    return True

def _map_spaces(map_data):
    """NiSpace-processed spaces (github-nispace or private, non-Original)."""
    spaces = []
    nispace_hosts = {"github-nispace", "github-nispace-private"}
    for space, val in map_data.items():
        if "Original" in space or not isinstance(val, dict):
            continue
        is_nispace = (
            val.get("host") in nispace_hosts
            or any(isinstance(hv, dict) and hv.get("host") in nispace_hosts
                   for hv in val.values() if isinstance(hv, dict))
        )
        if is_nispace:
            spaces.append(space)
    return spaces


# ---------------------------------------------------------------------------
# RST helpers
# ---------------------------------------------------------------------------

def _parse_reference_txt(path):
    """Return {dataset: full_description} from reference.txt."""
    sections = {}
    current, lines = None, []
    for line in path.read_text().splitlines():
        if line.startswith("# "):
            if current:
                sections[current] = "\n".join(lines).strip()
            current, lines = line[2:].strip(), []
        else:
            lines.append(line)
    if current:
        sections[current] = "\n".join(lines).strip()
    return sections

def _list_table(headers, rows, widths=None):
    """Render an RST list-table."""
    lines = [".. list-table::", "   :header-rows: 1"]
    if widths:
        lines.append(f"   :widths: {' '.join(str(w) for w in widths)}")
    lines.append("")
    lines.append(f"   * - {headers[0]}")
    for h in headers[1:]:
        lines.append(f"     - {h}")
    for row in rows:
        lines.append(f"   * - {row[0]}")
        for cell in row[1:]:
            lines.append(f"     - {cell}")
    return "\n".join(lines)

def _section(title, char="-"):
    return f"{title}\n{char * len(title)}"


# ---------------------------------------------------------------------------
# Parcellations page
# ---------------------------------------------------------------------------

def _gen_parcellations(parc_lib):
    real = {k: v for k, v in parc_lib.items() if "alias" not in v}
    aliases_rev = {}
    for alias, data in parc_lib.items():
        if "alias" in data:
            aliases_rev.setdefault(data["alias"], []).append(alias)

    lines = []
    lines.append(
        "``NiSpace`` ships with the following built-in parcellations. "
        "Fetch them via :func:`nispace.datasets.fetch_parcellation`:"
    )
    lines.append("")
    lines.append(".. code-block:: python")
    lines.append("")
    lines.append("   from nispace.datasets import fetch_parcellation")
    lines.append("   parc = fetch_parcellation(\"Schaefer200\")")
    lines.append("")
    lines.append(
        "Cortical and subcortical parcellations can be **combined** by concatenating "
        "their names (e.g. ``\"Schaefer200TianS1\"``). "
        "NiSpace resolves the combination automatically."
    )
    lines.append("")
    lines.append(".. note::")
    lines.append("")
    lines.append(
        f"   Linked items point to files or folders in the "
        f"`NiSpace data repository <{GITHUB_BASE}>`__."
    )
    lines.append("")

    rows = []
    for name, data in real.items():
        spaces = list(data.keys())
        level = data[spaces[0]].get("level", "?").capitalize()
        n = PARC_N_PARCELS.get(name, "—")
        space_str = ", ".join(f"<code>{s}</code>" for s in spaces)
        alias_str = ", ".join(f"<code>{a}</code>" for a in aliases_rev.get(name, []))
        url = _gh_tree(f"parcellation/{name}")
        rows.append([_cl(name, url), level, str(n), space_str, alias_str or "—"])

    table_html = _html_table(
        ["Name", "Level", "Parcels", "Available spaces", "Aliases"], rows
    )
    lines.extend(_raw_html(table_html.splitlines()))

    lines.append(".. note::")
    lines.append("")
    lines.append(
        "   Not all parcellations have precomputed distance matrices or spin matrices. "
        "   Run :func:`nispace.nulls.get_distance_matrix` to compute a distance matrix "
        "   for any parcellation."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Datasets page
# ---------------------------------------------------------------------------

def _map_dropdown(maps):
    """RST lines for a collapsible HTML table of maps with space links."""
    public_maps = {n: d for n, d in maps.items() if not _map_is_private(d)}
    if not public_maps:
        return []

    rows_html = []
    for mname in sorted(public_maps.keys()):
        mdata = public_maps[mname]
        url = _map_folder_url(mdata)
        name_cell = _cl(mname, url) if url else _c(mname)
        spaces = _map_spaces(mdata)
        spaces_cell = ", ".join(_c(s) for s in spaces) if spaces else "—"
        rows_html.append(f"<tr><td>{name_cell}</td><td>{spaces_cell}</td></tr>")

    table = (
        f'<table class="nispace-table">'
        f'<thead><tr><th>Map</th><th>Available spaces</th></tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table>'
    )
    details = (
        f'<details class="nispace-details">'
        f'<summary><strong>Show all {len(public_maps)} maps</strong></summary>'
        f'{table}'
        f'</details>'
    )
    return _raw_html([details])


def _info_block(n_maps, tabs, colls, meta):
    """RST lines for per-dataset info (HTML, with code+link for repo items)."""
    items = []
    if n_maps:
        items.append(f"<strong>Individual maps:</strong> {n_maps}")
    if tabs:
        parts = [
            _cl(p, _gh_blob(e["remote"])) if e.get("host") == "github-nispace" else _c(p)
            for p, e in tabs.items()
        ]
        items.append("<strong>Precomputed for:</strong> " + ", ".join(parts))
    if colls:
        parts = [
            _cl(n, _gh_blob(e["remote"])) if e.get("host") == "github-nispace" else _c(n)
            for n, e in colls.items()
        ]
        items.append("<strong>Collections:</strong> " + ", ".join(parts))
    if meta and meta.get("host") == "github-nispace":
        items.append("<strong>Metadata:</strong> " + _cl("metadata.csv", _gh_blob(meta["remote"])))

    html = "<p>" + "<br>".join(items) + "</p>"
    return _raw_html([html])


def _gen_datasets(ref_lib, descriptions):
    lines = []

    # Overview table
    lines.append(
        "``NiSpace`` provides the following built-in reference datasets. "
        "Fetch them via :func:`nispace.datasets.fetch_reference`:"
    )
    lines.append("")
    lines.append(".. code-block:: python")
    lines.append("")
    lines.append("   from nispace.datasets import fetch_reference")
    lines.append("   pet = fetch_reference(\"pet\", collection=\"UniqueTracers\",")
    lines.append("                         parcellation=\"Schaefer200\")")
    lines.append("")

    lines.append(".. note::")
    lines.append("")
    lines.append(
        f"   Linked items point to files or folders in the "
        f"`NiSpace data repository <{GITHUB_BASE}>`__."
    )
    lines.append("")

    headers = ["Key", "Description", "Maps", "Precomputed parcellations", "Collections"]
    ov_rows = []
    for ds, data in ref_lib.items():
        _maps = data.get("map", {})
        n_maps = sum(1 for d in _maps.values() if not _map_is_private(d))
        n_tabs = len(data.get("tab", {}))
        n_colls = len(data.get("collection", {}))
        maps_str = str(n_maps) if n_maps else f"tabular ({n_tabs} parcellations)"
        ov_rows.append([
            _cl(ds, f"#ds-{ds}"),
            DATASET_ONE_LINERS.get(ds, ""),
            maps_str,
            str(n_tabs),
            str(n_colls),
        ])
    lines.extend(_raw_html(_html_table(headers, ov_rows).splitlines()))
    lines.append("")
    lines.append(
        "See :func:`nispace.datasets.fetch_metadata` to retrieve detailed per-map "
        "metadata (tracers, publications, licenses) for datasets that carry it (e.g. ``pet``)."
    )
    lines.append("")

    # Collections tip
    lines.append(".. tip::")
    lines.append("")
    lines.append(
        "   Most datasets ship with **collections** — curated subsets of maps. "
        "Pass the ``collection`` argument to :func:`~nispace.datasets.fetch_reference` "
        "to load only the maps in a given collection:"
    )
    lines.append("")
    lines.append("   .. code-block:: python")
    lines.append("")
    lines.append("      from nispace.datasets import fetch_reference, fetch_collection")
    lines.append("      pet = fetch_reference(\"pet\", collection=\"UniqueTracers\",")
    lines.append("                            parcellation=\"Schaefer200\")")
    lines.append("      colls = fetch_collection(\"UniqueTracers\", dataset=\"pet\")")
    lines.append("")
    lines.append("----")
    lines.append("")

    # Per-dataset sections
    for ds, data in ref_lib.items():
        title = DATASET_TITLES.get(ds, ds)
        lines.append(f".. _ds-{ds}:")
        lines.append("")
        lines.append(_section(title, "-"))
        lines.append("")

        # Description (HTML to handle bullet lists properly)
        desc = descriptions.get(ds, "")
        if desc:
            lines.extend(_raw_html([_desc_to_html(desc)]))

        maps = data.get("map", {})
        tabs = data.get("tab", {})
        colls = data.get("collection", {})
        meta = data.get("metadata", {})
        public_maps = {n: d for n, d in maps.items() if not _map_is_private(d)}
        n_maps = len(public_maps)

        # Info block
        lines.extend(_info_block(n_maps, tabs, colls, meta))

        # Map dropdown
        if n_maps:
            lines.extend(_map_dropdown(maps))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Templates page
# ---------------------------------------------------------------------------

def _tmpl_images_html(imgs_data):
    """Render template images as HTML code cells, linked for github-nispace entries."""
    parts = []
    for img_name, img_entry in imgs_data.items():
        if not isinstance(img_entry, dict):
            parts.append(_c(img_name))
            continue
        # Volumetric: direct host/remote
        if "host" in img_entry:
            if img_entry["host"] == "github-nispace":
                parts.append(_cl(img_name, _gh_blob(img_entry["remote"])))
            else:
                parts.append(_c(img_name))
        # Surface: nested L/R hemispheres
        else:
            first_hemi = next((v for v in img_entry.values() if isinstance(v, dict)), None)
            if first_hemi and first_hemi.get("host") == "github-nispace":
                parts.append(_cl(img_name, _gh_blob(first_hemi["remote"])))
            else:
                parts.append(_c(img_name))
    return ", ".join(parts)


def _gen_templates(tmpl_lib):
    lines = []
    lines.append(
        "``NiSpace`` provides built-in brain templates for use with parcellation and "
        "visualization. Fetch them via :func:`nispace.datasets.fetch_template`:"
    )
    lines.append("")
    lines.append(".. code-block:: python")
    lines.append("")
    lines.append("   from nispace.datasets import fetch_template")
    lines.append("   t1w = fetch_template(\"MNI152NLin2009cAsym\", res=\"2mm\", desc=\"T1w\")")
    lines.append("")
    lines.append(
        "All other resolutions are sourced from "
        "`TemplateFlow <https://www.templateflow.org>`__."
    )
    lines.append("")
    lines.append(".. note::")
    lines.append("")
    lines.append(
        f"   Linked items point to files or folders in the "
        f"`NiSpace data repository <{GITHUB_BASE}>`__ "
        "(3mm resolution files only)."
    )
    lines.append("")

    rows = []
    for space, resolutions in tmpl_lib.items():
        for res, imgs in resolutions.items():
            images_html = _tmpl_images_html(imgs)
            rows.append([_c(space), _c(res), images_html])

    table_html = _html_table(["Space", "Resolution", "Available images"], rows)
    lines.extend(_raw_html(table_html.splitlines()))

    lines.append(".. note::")
    lines.append("")
    lines.append(
        "   Volumetric templates (``MNI152NLin2009cAsym``, ``MNI152NLin6Asym``) are "
        "NIfTI files. Surface templates (``fsaverage``) are GIFTI files with "
        "separate left (``L``) and right (``R``) hemisphere files."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate(app=None):
    """Generate RST content into docs/_auto/. Called by conf.py setup()."""
    OUT_DIR.mkdir(exist_ok=True)

    ref_lib = json.loads((DATALIB_DIR / "reference.json").read_text())
    parc_lib = json.loads((DATALIB_DIR / "parcellation.json").read_text())
    tmpl_lib = json.loads((DATALIB_DIR / "template.json").read_text())
    descriptions = _parse_reference_txt(DATALIB_DIR / "reference.txt")

    (OUT_DIR / "parcellations_content.rst").write_text(_gen_parcellations(parc_lib))
    (OUT_DIR / "datasets_content.rst").write_text(_gen_datasets(ref_lib, descriptions))
    (OUT_DIR / "templates_content.rst").write_text(_gen_templates(tmpl_lib))

    if app is None:
        print(f"Generated {OUT_DIR / 'parcellations_content.rst'}")
        print(f"Generated {OUT_DIR / 'datasets_content.rst'}")
        print(f"Generated {OUT_DIR / 'templates_content.rst'}")


if __name__ == "__main__":
    generate()
