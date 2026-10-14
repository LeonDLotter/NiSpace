# NiSpace logo

`./build_logo.sh [python]` regenerates everything in `final/` in about 20 s.

All inputs are downloaded from pinned remote sources and checked by MD5 (`remote.py`). They are cached in `work/data/`, which is not tracked by git:

- the FreeSurfer recon of the MNI152NLin6Asym template: `rh.pial`, `rh.thickness`, `rh.aparc.annot` from [GIN `llotter/mni_freesurfer`](https://gin.g-node.org/llotter/mni_freesurfer), commit `4e7d399`;
- JetBrains Mono Bold (wordmark) and ExtraBold (full name) ([SIL OFL 1.1](https://github.com/JetBrains/JetBrainsMono/blob/master/OFL.txt)), commit `02bb50b`.

Requirements: Python with numpy, scipy, nibabel, scikit-image, matplotlib and fonttools. PNG export needs `rsvg-convert` (librsvg).

## Files in `final/`

| File | Use |
|---|---|
| `nispace_logo.svg` | main logo: `ni` + brain → hexbin, `space` below |
| `nispace_logo_fullname.svg` | the same, with "neuroimaging" under "ni" and "spatial colocalization environment" under "space" |
| `nispace_icon.svg` | the logo graphic without text, square: avatar, social |
| `nispace_favicon.svg` | the partly dissolved brain only (12 columns, no flying cells or plot), square: 16–48 px and the 180 px touch icon |
| `nispace_social_preview.svg` | the full-name logo centred on black, 1280 × 640 (`card_svg.py`): GitHub social preview |
| `*_<size>.png` | PNG exports of the above (`nispace_social_preview_1280x640.png` for upload) |

There is one version for all backgrounds. Light rims around the letters and cells keep it readable on dark backgrounds; the full name uses mid tones that read on both (about 3.5:1 on white and on dark grey).

## License

The logo files in `final/` are © Leon D. Lotter and licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), unlike the rest of this repository (CC BY-NC-SA 4.0). You may use and adapt them for any purpose, including commercial publications, if you give attribution:

> NiSpace logo by Leon D. Lotter (https://github.com/LeonDLotter/NiSpace), CC BY 4.0

When you use the logo in connection with research, please also cite NiSpace (see [Citation](https://nispace.readthedocs.io/en/latest/citation.html)), currently: Lotter & Dukart, Zenodo 2024, https://doi.org/10.5281/zenodo.12514622.

The license covers copyright only. It grants no trademark rights and does not permit implying that NiSpace or its authors endorse you or your use. The build scripts in this folder are part of the software and remain under the repository license.

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
   - Rim widths can be set separately for moved cells and for the cells that stay in the brain.
4. **`compose_logo.py`**: sets the text as outlines and aligns everything on the letters' ink edges:
   - "ni" and "space" share one left edge; both lines of the full name start at the outer edge of their rims;
   - "neuroimaging" is exactly as wide as "ni" with its rim, and "spatial colocalization environment" has the same size;
   - each line of the full name sits the same ink gap below the big letters above it.
5. **`square_svg.py`**: crops the icon and favicon to the graphic's own bounding box (without the text) and pads it to a square.

## Fixed design decisions

- **Colours:** red `#B2201F` for "ni", navy `#023D6B` for "space", grey `#8C95A1` for the line. Rims and bases are the cell colour plus 35% white.
- **Cell rims:** 2.2 on the moved (flying and landed) cells, so they stay visible on dark backgrounds; 1.0 in the brain.
- **Brain size:** 1.25 F tall; its frontal pole ends above the "e" of "space".
- **Line spacing:** 0.9 F in the graphic's layout. In the short logo "ni" sits 0.06 F lower (closer to "space"), in the full-name version 0.03 F higher. Both shorten the descender of the "p" to 60%.
- **Regression line:** starts one natural letter gap (the c→e ink gap) plus one hex-lattice step after "space", on its baseline; a whole lattice step moves the plot without re-drawing its pattern. The plot's width/height is 0.75.
- **Full name:** split, each part under its own syllable, no box, ExtraBold.
  - "ni" moves 0.03 F up to make room; the graphic stays where it is.
  - The lower line moves up with the shortened "p".
  - Gaps: 0.07 F from the bottom of "ni" to the top of the "i" in "neuroimaging", and from the end of the "p" to the top of an "i" in "colocalization".
  - Colours: n·i in light red (red + 35% white), s·p·a·c·e in light navy (navy + 40% white), the rest in grey `#808893`.
  - The highlighted letters are 1.26×, as tall as the "t". Their cells grow by the extra ink, so every ink gap equals the normal one.
