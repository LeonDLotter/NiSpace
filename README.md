# <a name="top"></a>`NiSpace`: `N`euro`i`maging `Spa`tial `C`olocalization `E`nvironment
<!--
[![DOI](https://zenodo.org/badge/506986337.svg)](https://zenodo.org/badge/latestdoi/506986337)  
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey)](http://creativecommons.org/licenses/by-nc-sa/4.0/)  
---
-->

*Spatial (alteration) patterns observed in MRI images may often reflect function and dysfunction of underlying biological systems. This applies alike to function and structure, on the surface or in the volumetric space, to typical activation patterns, or to maps of disordered brain function, for example obtained from patients compared to a reference cohort.*

In recent years, several methods have been developed to compare spatial patterns between brain maps. In the simplest case, two brain maps are correlated with each other at the voxel- or parcel-level. The resulting correlation coefficient reflects the degree to which the two maps share a spatial pattern. We refer to this spatial correlation as "*colocalization*". The `NiSpace` toolbox aims to provide the most comprehensive, yet easy-to-use and flexible framework for colocalization estimation, significance testing, and visualization to date.

There are of course many other tools available, of which a few are listed below:

| Name | Target Problem | Significance Testing | Volume/Surface | Interface | 
|------|----------------|----------------------|----------------|-----------|
| [JuSpace](https://github.com/juryxy/JuSpace) | Colocalization between nuclear imaging and case-control-difference maps | group permutation, null maps | volume | MATLAB-GUI |
| [neuromaps](https://netneurolab.github.io/neuromaps/) | Providing various reference brain maps, as well as advanced imaging space transformation and null map estimation functions | null maps | surface, volume | Python-API |
| [ENIGMA Toolbox](https://enigma-toolbox.readthedocs.io/) | Relationships between effect-size maps and various reference datasets | null maps | surface | Python/MATLAB-API |
| [BrainSpace](https://brainspace.readthedocs.io/) | Focus on gradient mapping but includes null map generation functions | null maps | surface, volume | Python/MATLAB-API |
| [Imaging Transcriptomics Toolbox](https://imaging-transcriptomics.readthedocs.io) | Gene-Set-Enrichment-Analysis on neuroimaging data using Allen Brain Atlas |  | surface (volume) | Python-API |
| [GAMBA](https://github.com/dutchconnectomelab/GAMBA-MATLAB) | Gene-Set-Enrichment-Analysis on neuroimaging data using Allen Brain Atlas | gene-set permutation, null maps | volume (surface) | Web-GUI/MATLAB-API |
| [GCEA](https://github.com/benfulcher/GeneCategoryEnrichmentAnalysis) | Gene-Set-Enrichment-Analysis on neuroimaging data using Allen Brain Atlas | gene-set permutation, null maps | volume (surface) | MATLAB-API |

`NiSpace` tries to incorporate most of the functionality of these tools in a unified framework. It took many ideas and implementations from the toolboxes listed above. Two prior tools developed by me (Leon Lotter) –  [JuSpyce](https://github.com/leondlotter/JuSpyce) and [ABAnnotate](https://github.com/leondlotter/ABAnnotate) – were discontinued in favor of `NiSpace`.

 Name | Target Problem | Significance Testing | Volume/Surface | Interface | 
|------|----------------|----------------------|----------------|-----------|
| `NiSpace` | Colocalization between one or multiple brain maps in single-map, case-control, and set-enrichment settings. Incorporates advanced imaging space transformation through neuromaps. Includes a large range of reference datasets | null maps, group permutation, set permutation | volume, surface | Python-API (GUI planned) |

## Citation

There is no paper for `NiSpace` yet. Please cite at least the following when you the tool in your work:
<!-- - [Lotter & Dukart, 2024](https://doi.org/10.5281/zenodo.6884932) -->
- [Dukart et al., 2021](https://doi.org/10.1002/hbm.25244)
- [Markello, Hansen, et al., 2022](https://doi.org/10.1038/s41592-022-01625-w)
- If you use the implemented "moran" null map function: [Vos de Wael et al., 2020](https://doi.org/10.1038/s42003-020-0794-7)
- If you use the implemented "burt2020" null map function: [Burt et al., 2020](https://doi.org/10.1016/j.neuroimage.2020.117038)
- If you use the implemented "burt2018" null map function: [Burt et al., 2018](https://doi.org/10.1038/s41593-018-0195-0)
- When fetching included reference datasets, you will automatically get lists with references which we recommend you to cite

## Contact

Do you have questions, comments or suggestions, or would like to contribute to the toolbox? Feel free to open an issue here on GitHub or [contact me](mailto:leondlotter@gmail.com)! 

---
[Back to the top](#top)


