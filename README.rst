
``NiSpace``: ``N``euro``i``maging ``S``patial ``C``olocalization ``E``nvironment
================================================================================

.. image:: https://zenodo.org/badge/XXXXXXX.svg
   :target: https://zenodo.org/badge/latestdoi/XXXXXXX
   :alt: Zenodo record
.. image:: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey
   :target: http://creativecommons.org/licenses/by-nc-sa/4.0/
   :alt: License: CC BY-NC-SA 4.0

*Spatial (alteration) patterns observed in MRI images may often reflect function and dysfunction of underlying biological systems. This applies alike to function and structure, on the surface or in the volumetric space, to typical activation patterns, or to maps of disordered brain function, for example obtained from patients compared to a reference cohort.*

In recent years, several methods have been developed to compare spatial patterns between brain maps. In the simplest case, two brain maps are correlated with each other at the voxel- or parcel-level. The resulting correlation coefficient reflects the degree to which the two maps share a spatial pattern. We refer to this spatial correlation as "*colocalization*". The `NiSpace` toolbox aims to provide the most comprehensive, yet easy-to-use and flexible framework for colocalization estimation, significance testing, and visualization to date.

There are of course many other tools available, of which a few are listed below:

.. list-table::
   :widths: 15 40 15 15 15
   :header-rows: 1

   * - Name
     - Target Problem
     - Significance Testing
     - Volume/Surface
     - Interface
   * - `JuSpace <https://github.com/juryxy/JuSpace>`_
     - Colocalization between nuclear imaging and case-control-difference maps
     - group permutation, null maps
     - volume
     - MATLAB-GUI
   * - `neuromaps <https://netneurolab.github.io/neuromaps/>`_
     - Providing various reference brain maps, as well as advanced imaging space transformation and null map estimation functions
     - null maps
     - surface, volume
     - Python-API
   * - `ENIGMA Toolbox <https://enigma-toolbox.readthedocs.io/>`_
     - Relationships between effect-size maps and various reference datasets
     - null maps
     - surface
     - Python/MATLAB-API
   * - `BrainSpace <https://brainspace.readthedocs.io/>`_
     - Focus on gradient mapping but includes null map generation functions
     - null maps
     - surface, volume
     - Python/MATLAB-API
   * - `Imaging Transcriptomics Toolbox <https://imaging-transcriptomics.readthedocs.io>`_
     - Gene-Set-Enrichment-Analysis on neuroimaging data using Allen Brain Atlas
     - 
     - surface (volume)
     - Python-API
   * - `GAMBA <https://github.com/dutchconnectomelab/GAMBA-MATLAB>`_
     - Gene-Set-Enrichment-Analysis on neuroimaging data using Allen Brain Atlas
     - gene-set permutation, null maps
     - volume (surface)
     - Web-GUI/MATLAB-API
   * - `GCEA <https://github.com/benfulcher/GeneCategoryEnrichmentAnalysis>`_
     - Gene-Set-Enrichment-Analysis on neuroimaging data using Allen Brain Atlas
     - gene-set permutation, null maps
     - volume (surface)
     - MATLAB-API

``NiSpace`` tries to incorporate most of the functionality of these tools in a unified framework. It took many ideas and implementations from the toolboxes listed above. Two prior tools developed by me (Leon Lotter) –  [JuSpyce](https://github.com/leondlotter/JuSpyce) (basis for ``NiSpace``, Python) and [ABAnnotate](https://github.com/leondlotter/ABAnnotate) (easy-to-use neuroimaging gene-set enrichment based on [GCEA](https://github.com/benfulcher/GeneCategoryEnrichmentAnalysis), MATLAB) – were discontinued in favor of ``NiSpace``.

.. list-table::
   :widths: 15 40 15 15 15
   :header-rows: 1

   * - Name
     - Target Problem
     - Significance Testing
     - Volume/Surface
     - Interface
   * - `NiSpace <https://github.com/LeonDLotter/NiSpace>`_
     - Colocalization between two or multiple brain maps in single-map, case-control, and set-enrichment settings. Generalizes set-enrichment approach to all kinds of reference maps. Incorporates advanced imaging space transformation through neuromaps. Includes a large range of reference datasets
     - null maps, group permutation, set permutation
     - volume, surface
     - Python-API (GUI planned)


Installation
------------

``NiSpace`` is not yet available on PyPI. Currently, the easiest way is to install the development version directly from the repository using pip:

.. code-block:: bash

   pip install git+https://github.com/LeonDLotter/NiSpace.git@dev

We recommend using Python 3.9+ in a dedicated environment (e.g., via [Anaconda](https://www.anaconda.com/)).  
There are some optional dependencies. When calling the respective `NiSpace` functions, you are prompted to install them. However, to avoid this, you can install them directly with the other dependencies using:

.. code-block:: bash

   pip install "git+https://github.com/LeonDLotter/NiSpace.git@dev#egg=nispace[opt]"


Citation
--------

There is no paper for `NiSpace` yet. Please cite at least the following when you use out tools in your work:
- `Dukart et al., HBM 2021 <https://doi.org/10.1002/hbm.25244>`_
- `Markello, Hansen, et al., Nat. Methods 2022 <https://doi.org/10.1038/s41592-022-01625-w>`_
- If you use the implemented "moran" null map function: `Vos de Wael et al., Comm. Biol. 2020 <https://doi.org/10.1038/s42003-020-0794-7>`_
- If you use the implemented "burt2020" null map function: `Burt et al., NeuroImage 2020 <https://doi.org/10.1016/j.neuroimage.2020.117038>`_
- If you use the implemented "burt2018" null map function: `Burt et al., Nat. Neurosci. 2018 <https://doi.org/10.1038/s41593-018-0195-0>`_
- When fetching included reference datasets, `NiSpace` will automatically print information on references we recommend you to cite

Furthermore, the preceding tools, JuSpyce and ABAnnotate, were used in the following works:
- JuSpyce: `Lotter et al., bioRxiv 2024 <https://doi.org/10.1101/2023.05.05.539537>`_
- JuSpyce: `Lotter et al., Neuropsychopharm. 2024 <https://doi.org/10.1038/s41386-024-01880-9>`_
- JuSpyce/ABAnnotate: `Lotter et al., Neurosci. & Biobehav. Rev. 2023 <https://doi.org/10.1016/j.neubiorev.2023.105042>`_
- ABAnnotate: `Feng et al., Comm. Biol. 2023 <https://doi.org/10.1038/s42003-023-05647-8>`_


Contact
-------

Do you have questions, comments or suggestions, or would like to contribute to the toolbox? Feel free to open an issue here on GitHub or [contact me](mailto:leondlotter@gmail.com)! 



