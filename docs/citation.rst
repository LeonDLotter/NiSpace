.. _citation:

Citation
========

There is currently no dedicated toolbox paper for ``NiSpace``. Please cite at least the following when you use our tools in your work:

* NiSpace Zenodo DOI: :cite:`lotter_zenodo2024`
* JuSpace toolbox paper: :cite:`dukart2021`
* neuromaps toolbox paper: :cite:`markello2022`
* Application papers in the context of which ``NiSpace``'s core methods were developed:
  :cite:`lotter2023`; :cite:`lotter2024`; :cite:`lotter2026`
* If you use the implemented null map functions:

  * ``"moran"``: :cite:`wagner2015` (original method); :cite:`vos_de_wael2020` (BrainSpace implementation)
  * ``"burt2020"``: :cite:`burt2020`; implemented via `brainsmash <https://github.com/murraylab/brainsmash>`_
  * ``"burt2018"``: :cite:`burt2018`
  * ``"spin"`` / ``"cornblath"``: :cite:`cornblath2020`
  * ``"baum"``: :cite:`baum2020`
  * ``"alexander_bloch"``: :cite:`alexander_bloch2018`
  * ``"vasa"``: :cite:`vasa2018`
  * ``"hungarian"``: :cite:`kuhn1955` (assignment algorithm); popularized for spin tests by :cite:`vasa2018`
* If you use MNI space transform functionality (``mni_to_mni``):

  * EasyReg: :cite:`iglesias2023`
  * SynthMorph: :cite:`hoffmann2022`
  * SynthSeg: :cite:`billot2023`
* If you use combat-harmonization functionality: :cite:`fortin2017`; :cite:`pomponio2019`
* When fetching included reference datasets, ``NiSpace`` will automatically print information on references we ask you to cite

Furthermore, the preceding tools, JuSpyce and ABAnnotate, were used in the following works:

* JuSpyce: :cite:`lotter2024` (as above); :cite:`lotter_npp2024`
* JuSpyce/ABAnnotate: :cite:`lotter2023` (as above)
* ABAnnotate: :cite:`feng2023`

Lastly, ``NiSpace`` strongly depends on several foundational Python packages, among them:

* numpy: :cite:`harris2020`
* pandas: :cite:`mckinney2010`
* scipy: :cite:`virtanen2020`
* statsmodels: :cite:`seabold2010`
* nibabel: :cite:`brett_nibabel`
* nilearn and scikit-learn: :cite:`abraham2014`
* matplotlib: :cite:`hunter2007`
* seaborn: :cite:`waskom2021`

Full bibliographic details for all references are listed on the :doc:`references` page.
