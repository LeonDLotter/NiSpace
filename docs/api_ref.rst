.. _api:

.. currentmodule:: nispace

Reference API
==============

.. contents:: **List of modules**
   :local:


.. _api_api:

:mod:`nispace.api` – NiSpace main class
----------------------------------------
.. currentmodule:: nispace.api

.. autosummary::
   :template: class.rst
   :toctree: generated/

   NiSpace


.. _api_datasets:

:mod:`nispace.datasets` – Dataset fetchers
-------------------------------------------
.. currentmodule:: nispace.datasets

.. autosummary::
   :template: function.rst
   :toctree: generated/

   fetch_template
   fetch_parcellation
   fetch_reference
   fetch_collection
   apply_collection
   fetch_metadata
   fetch_example


.. _api_workflows:

:mod:`nispace.workflows` – Workflows
--------------------------------------
.. currentmodule:: nispace.workflows

.. autosummary::
   :template: function.rst
   :toctree: generated/

   simple_colocalization
   group_comparison
   simple_xsea


.. _api_stats_coloc:

:mod:`nispace.stats.coloc` – Colocalization statistics
--------------------------------------------------------
.. currentmodule:: nispace.stats.coloc

.. autosummary::
   :template: function.rst
   :toctree: generated/

   rank_array
   rank1d
   rank2d
   corr
   pearson
   partialcorr
   partialpearson
   mutualinfo
   mlr
   r2
   beta
   dominance
   pls
   pcr
   fast_pls1
   elasticnet
   lasso
   ridge


.. _api_stats_effectsize:

:mod:`nispace.stats.effectsize` – Effect size calculation
-----------------------------------------------------------
.. currentmodule:: nispace.stats.effectsize

.. autosummary::
   :template: function.rst
   :toctree: generated/

   cohen
   cohen_nan
   cohen_paired
   cohen_paired_nan
   hedges
   hedges_nan
   zscore
   zscore_nan
   rzscore_nan
   prc
   logfc_nan


.. _api_stats_misc:

:mod:`nispace.stats.misc` – Miscellaneous stats functions
-----------------------------------------------------------
.. currentmodule:: nispace.stats.misc

.. autosummary::
   :template: function.rst
   :toctree: generated/

   np_any_axis1
   residuals
   residuals_nan
   partial_residuals_nan
   rho_to_z
   z_to_rho
   zscore_df
   permute_groups
   null_to_p
   mc_correction
   compute_meff
   meff_sidak_correction
   maxT_correction
   step_maxT_correction


.. _api_io:

:mod:`nispace.io` – Imaging data input
----------------------------------------
.. currentmodule:: nispace.io

.. autosummary::
   :template: function.rst
   :toctree: generated/

   parcellate_data
   read_json
   write_json
   load_img
   load_labels
   load_distmat
   load_spinmat
   to_pickle
   from_pickle


.. _api_parcellate:

:mod:`nispace.parcellate` – Parcellation class
------------------------------------------------
.. currentmodule:: nispace.parcellate

.. autosummary::
   :template: class.rst
   :toctree: generated/

   Parcellater


.. _api_nulls:

:mod:`nispace.nulls` – Null map generation
--------------------------------------------
.. currentmodule:: nispace.nulls

.. autosummary::
   :template: function.rst
   :toctree: generated/

   generate_null_maps
   nulls_moran
   nulls_burt2020
   nulls_burt2018
   nulls_random
   generate_spins
   apply_spins
   get_distance_matrix
   find_vol_parc_centroids
   find_surf_parc_centroids
   correlate_hemis_parc
   find_parcel_hemispheres


.. _api_plotting:

:mod:`nispace.plotting` – Plotting functions
----------------------------------------------
.. currentmodule:: nispace.plotting

.. autosummary::
   :template: function.rst
   :toctree: generated/

   brainplot
   view_surf
   catplot
   nullplot
   heatmap
   nice_stats_labels
   print_significance
   move_legend_fig_to_ax
   colors_from_values
   hide_empty_axes
   linewidth_from_data_units


.. _api_transforms:

:mod:`nispace.transforms` – MNI space transforms
--------------------------------------------------
.. currentmodule:: nispace.transforms

.. autosummary::
   :template: function.rst
   :toctree: generated/

   mni_to_mni
   compute_transform_displacement


.. _api_utils_utils:

:mod:`nispace.utils` – Utility functions
------------------------------------------
.. currentmodule:: nispace.utils.utils

.. autosummary::
   :template: function.rst
   :toctree: generated/

   set_log
   nan_detector
   remove_nan
   fill_nan
   mean_by_set_df
   print_arg_pairs
   get_column_names
   lower
   get_background_value
   vect_to_vol_arr
   vol_to_vect_arr
   parc_vect_to_vol
   relabel_gifti_parc
   relabel_nifti_parc
   merge_parcellations
   mirror_nifti
   mirror_gifti
