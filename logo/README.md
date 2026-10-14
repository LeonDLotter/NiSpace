# NiSpace logo

`./build_logo.sh [python]` regenerates everything in `final/` in about 20 s.

All inputs are downloaded from pinned remote sources and checked by MD5 (`remote.py`). They are cached in `work/data/`, which is not tracked by git:

- the FreeSurfer recon of the MNI152NLin6Asym template: `rh.pial`, `rh.thickness`, `rh.aparc.annot` from [GIN `llotter/mni_freesurfer`](https://gin.g-node.org/llotter/mni_freesurfer), commit `4e7d399`;
- JetBrains Mono Bold ([SIL OFL 1.1](https://github.com/JetBrains/JetBrainsMono/blob/master/OFL.txt)), commit `02bb50b`.

Requirements: Python with numpy, scipy, nibabel, scikit-image, matplotlib and fonttools. PNG export needs `rsvg-convert` (librsvg).

## Files in `final/`

| File | Use |
|---|---|
| `nispace_logo.svg` | main logo: `ni` + brain → hexbin, `space` below |
| `nispace_logo_fullname.svg` | the same, with "neuroimaging spatial colocalization environment" in a navy box |
| `nispace_icon.svg` | the logo graphic without text, square: avatar, social |
| `nispace_favicon.svg` | the partly dissolved brain only (12 columns, no flying cells or plot), square: 16–48 px and the 180 px touch icon |
| `*_<size>.png` | PNG exports of the above |

There is one version for all backgrounds. Light rims around the letters, cells and box keep it readable on dark backgrounds.

## Pipeline

1. **`brain_scene.py`**: renders the right lateral view of the native pial surface (0.5 mm z-buffer). It derives:
   - the outline;
   - the Sylvian line, from the border of the Desikan-Killiany temporal regions with the opercular regions and insula, extended through the front edge;
   - the colour, from cortical thickness (5 mm smoothed).
2. **`hex_logo.py`**: puts a hexagon grid on the brain (18 columns for the logo, 12 for the icons). Edge cells are scaled by how much of them lies inside the brain, with a minimum coverage of 0.12. Colour is the thickness rank, from thin = red to thick = navy.
3. **`hexbin_logo.py`**: the frontal cells blow away.
   - 55% of the leaving cells land in a hexbin band along a regression line; the rest stay in flight.
   - Each landed cell's colour shifts to its rank in the plot.
   - The lighter base hexagons stay behind in the brain as an imprint.
   - Layout `lockup` works in font units (F = 100). With `NSP_BRAIN_ONLY=1` only the partly dissolved brain is drawn (favicon).
4. **`compose_logo.py`**: sets the text as outlines and aligns everything on the letters' ink edges:
   - "ni", "space" and the box share one left edge;
   - the full name spans exactly to the end of the line;
   - the gaps n→s and s→box are equal.
5. **`square_svg.py`**: crops the icon and favicon to the graphic's own bounding box (without the text) and pads it to a square.

## Fixed design decisions

- **Colours:** red `#B2201F` for "ni", navy `#023D6B` for "space" and the box, grey `#8C95A1` for the line. Rims and bases are the cell colour plus 35% white.
- **Brain size:** 1.25 F tall; its frontal pole ends above the "e" of "space".
- **Line spacing:** 0.9 F.
- **Regression line:** starts one natural letter gap (the c→e ink gap, 0.13 F) after "space", on its baseline. The plot's width/height is 0.75.
- **Full name:** n·i·s·p·a·c·e in white; the rest in light red ("neuroimaging") and light blue (the other words), both the colour plus 55% white.
