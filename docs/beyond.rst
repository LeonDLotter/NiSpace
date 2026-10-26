.. _beyond:

Beyond spatial correlation
==========================

``NiSpace`` is built around spatial colocalization, but several of its parts are useful on their
own, independent of any colocalization analysis.

Reference data
--------------

One of the largest available collections of consistently processed, standardized multimodal
brain maps: PET maps of receptors, transporters and metabolism, gene expression and cell types,
cortical features, disorder effect maps (ENIGMA), meta-analytic maps, functional networks,
tissue probabilities, and more. Maps are available as images in several volume and surface
spaces and pre-parcellated for many parcellations, with per-map metadata (sample, tracer,
source, license) and curated collections (subsets and sets of maps).

- :func:`~nispace.datasets.fetch_reference`, :func:`~nispace.datasets.fetch_collection`,
  :func:`~nispace.datasets.fetch_map_info`
- Overview of all datasets: :doc:`datasets`

When you use these data, cite the original sources: :func:`~nispace.datasets.fetch_reference`
prints them (see :doc:`citation`).

Parcellations and templates
---------------------------

Cortical, subcortical and combined parcellations in volume and surface spaces, with consistent
labels and precomputed distance matrices, plus brain templates.

- :func:`~nispace.datasets.fetch_parcellation`, :func:`~nispace.datasets.fetch_template`
- Parcellate your own images: :class:`~nispace.parcellate.Parcellater`
- Overviews: :doc:`parcellations`, :doc:`templates`

Brain plotting
--------------

Publication-ready plots of brain maps and parcellated data on surfaces and volumes.

- :func:`~nispace.plotting.brainplot`; :meth:`NiSpace.plot_brain <nispace.api.NiSpace.plot_brain>`
- Notebook: :doc:`nb_introduction/intro08_brainplot`

Imaging space transformations
-----------------------------

Transformations between MNI volume spaces and fsLR / fsaverage surface spaces.

- :class:`~nispace.transform.SpaceTransformer`, :func:`~nispace.transform.mni_to_mni`,
  :func:`~nispace.transform.surf_to_surf`, :func:`~nispace.transform.mni_to_surf`

Null maps and spatial autocorrelation
-------------------------------------

Spatially constrained null maps (e.g. Moran spectral randomization, spin permutation, BrainSMASH)
for any spatial analysis, and descriptive statistics of spatial autocorrelation (Moran's I,
variograms, Mantel test in ``nispace.stats.autocorr``).

- :func:`~nispace.nulls.generate_null_maps`

Spatial axes
------------

Named spatial axes (tissue probabilities, parcel geometry and coordinates, cortex vs. subcortex,
cortical hierarchy) to describe maps or to use as covariates.

- :func:`~nispace.datasets.fetch_axes`, :func:`~nispace.diagnostics.shared_axes`
