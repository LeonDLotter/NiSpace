
``NiSpace``: NeuroImaging Spatial Colocalization Environment
================================================================================

.. image:: https://img.shields.io/badge/DOI-10.5281/zenodo.12514622-#1082C3
  :target: https://zenodo.org/doi/10.5281/zenodo.12514622
  :alt: DOI: 10.5281/zenodo.12514622
.. image:: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey
  :target: http://creativecommons.org/licenses/by-nc-sa/4.0/
  :alt: License: CC BY-NC-SA 4.0


*Spatial (alteration) patterns observed in MRI images may often reflect function and dysfunction of underlying biological systems. This applies alike to function and structure, on the surface or in the volumetric space, and to typical as well as disordered brain-anatomical and functional patterns.*

In recent years, several methods have been developed to compare spatial patterns between brain maps. In the simplest case, two brain maps are correlated with each other at the voxel- or parcel-level. The resulting correlation coefficient reflects the degree to which the two maps share a spatial pattern. We refer to this spatial correlation as "*colocalization*". The `NiSpace` toolbox aims to provide the most comprehensive, yet easy-to-use and flexible framework for colocalization estimation, significance testing, and visualization to date.

``NiSpace`` is under development and its `documentation <https://nispace.readthedocs.io/>`_ currently is (very) incomplete. We welcome anyone who would like to give it a try – your feedback is highly appreciated! 


Installation
------------

You can install the development version of ``NiSpace`` in a Python 3.9+ environment via command line using pip:

.. code-block:: bash

   pip install git+https://github.com/LeonDLotter/NiSpace.git@dev

For reproducibility, consider installing a specific commit:

.. code-block:: bash

   pip install git+https://github.com/LeonDLotter/NiSpace.git@{commit_hash}

NiSpace by default generates null maps using the Moran randomization method. All other null models are optional dependencies.  
Install optional dependencies directly with:

.. code-block:: bash

   pip install "git+https://github.com/LeonDLotter/NiSpace.git@dev#egg=nispace[opt]"


Support & Updates
-----------------

For questions about the toolbox, analysis choices, or if you need help with its application, we recommend to open a new topic on `NeuroStars <https://neurostars.org/>`_ using the tag `"nispace" <https://neurostars.org/tag/nispace>`_.
If you encounter bugs, or would like to request a new feature, dataset, or parcellation, feel free to open a `GitHub issue <https://github.com/LeonDLotter/NiSpace/issues>`_.  

We will post regular updates on new features and releases on `NeuroStars <https://neurostars.org/tags/c/announcements/6/nispace>`_.  
To automatically get notified about new posts, you can create a NeuroStars account and enable notifications for `"the nispace tag" <https://neurostars.org/tag/nispace>`_ (bell icon on the right side).


Other available tools
---------------------

There are of course many other related tools available, of which a few are listed below:

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

``NiSpace`` tries to incorporate most of the functionality of these tools in a unified framework. It took many ideas and implementations from the toolboxes listed above. Two prior tools developed by me (Leon Lotter) – `JuSpyce <https://github.com/leondlotter/JuSpyce>`_ (basis for ``NiSpace``, Python) and `ABAnnotate <https://github.com/leondlotter/ABAnnotate>`_ (easy-to-use neuroimaging gene-set enrichment based on `GCEA <https://github.com/benfulcher/GeneCategoryEnrichmentAnalysis>`_, MATLAB) – were discontinued in favor of ``NiSpace``.

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


Citation
--------

There is no paper for ``NiSpace`` yet. Please cite the following when you use our tools in your work:

* ``NiSpace`` Zenodo DOI: `Lotter & Dukart, Zenodo 2024 <https://doi.org/10.5281/zenodo.12514623>`_
* JuSpace toolbox paper: `Dukart et al., HBM 2021 <https://doi.org/10.1002/hbm.25244>`_
* neuromaps toolbox paper: `Markello, Hansen, et al., Nat. Methods 2022 <https://doi.org/10.1038/s41592-022-01625-w>`_
* Two publications with which NiSpace's core was developed: `Lotter et al., Neurosci. & Biobehav. Rev. 2023 <https://doi.org/10.1016/j.neubiorev.2023.105042>`_; `Lotter et al., Nat. Commun. 2024 <https://doi.org/10.1038/s41467-024-52366-7>`_

See the documentation's `citation <https://nispace.readthedocs.io/en/latest/citation.html>`_ section for more information.


Contact
-------

You can contact me via `email <mailto:leondlotter@gmail.com>`_. For usage questions, please consider opening a topic on NeuroStars (see `Support & Updates`), so that others can benefit from our exchange.



