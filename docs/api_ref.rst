.. _api:

.. currentmodule:: nispace

Reference API
==============

.. contents:: **List of modules**
   :local:


.. _api_api:

:mod:`nispace.api` - NiSpace main class
---------------------------------------
.. autoclass:: nispace.api
   :members:
   :inherited-members:

.. currentmodule:: nispace.api

.. autosummary::
   :template: class.rst
   :toctree: generated/

   NiSpace


.. _api_datasets:

:mod:`nispace.datasets` - Dataset fetchers
------------------------------------------
.. automodule:: nispace.datasets
   :members:
   :inherited-members:

.. currentmodule:: nispace.datasets

.. autosummary::
   :template: function.rst
   :toctree: generated/

   fetch_template
   fetch_parcellation
   fetch_reference
   fetch_metadata
   fetch_example
   

.. _api_workflows:

:mod:`nispace.workflows` - Workflows
------------------------------------
.. automodule:: nispace.workflows
   :members:
   :inherited-members:

.. currentmodule:: nispace.workflows

.. autosummary::
   :template: function.rst
   :toctree: generated/

   simple_colocalization
   group_comparison
   simple_xsea


.. _api_stats_coloc:

:mod:`nispace.stats.coloc` - Colocalization statistics
------------------------------------------------------
.. automodule:: nispace.stats.coloc
   :members:
   :inherited-members:

.. currentmodule:: nispace.stats.coloc

.. autosummary::
   :template: function.rst
   :toctree: generated/

   rank_array
   corr
   partialcorr
   mlr
   r2
   beta
   dominance
   pls
   pcr
   elasticnet
   lasso
   ridge


.. _api_stats_effectsize:

:mod:`nispace.stats.effectsize` - Effect size calculation
---------------------------------------------------------
.. automodule:: nispace.stats.effectsize
   :members:
   :inherited-members:

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
   hedges_paired
   zscore
   zscore_nan
   prc


.. _api_stats_misc:

:mod:`nispace.stats.misc` - Miscellaneous stats functions
---------------------------------------------------------
.. automodule:: nispace.stats.misc
   :members:
   :inherited-members:

.. currentmodule:: nispace.stats.misc

.. autosummary::
   :template: function.rst
   :toctree: generated/

   np_any_axis1
   residuals
   residuals_nan
   rho_to_z
   zscore_df
   permute_groups
   null_to_p
   mc_correction 


.. _api_io:

:mod:`nispace.io` - Imaging data input
--------------------------------------
.. automodule:: nispace.io
   :members:
   :inherited-members:

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


.. _api_parcellate:

:mod:`nispace.parcellate` - Parcellation class
----------------------------------------------
.. automodule:: nispace.parcellate
   :members:
   :inherited-members:

.. currentmodule:: nispace.parcellate

.. autosummary::
   :template: class.rst
   :toctree: generated/

   Parcellater


.. _api_nulls:

:mod:`nispace.nulls` - Null map generation
------------------------------------------
.. automodule:: nispace.nulls
   :members:
   :inherited-members:

.. currentmodule:: nispace.nulls

.. autosummary::
   :template: function.rst
   :toctree: generated/

   generate_null_maps
   nulls_moran
   nulls_burt2020
   nulls_burt2018
   get_distance_matrix
   find_surf_parc_centroids


.. _api_plotting:

:mod:`nispace.plotting` - Plotting functions
--------------------------------------------
.. automodule:: nispace.plotting
   :members:
   :inherited-members:

.. currentmodule:: nispace.plotting

.. autosummary::
   :template: function.rst
   :toctree: generated/

   catplot
   nullplot
   heatmap
   move_legend_fig_to_ax
   colors_from_values
   hide_empty_axes
   linewidth_from_data_units


.. _api_utils_utils:

:mod:`nispace.utils` - Utility functions
----------------------------------------
.. automodule:: nispace.utils.utils
   :members:
   :inherited-members:

.. currentmodule:: nispace.utils.utils

.. autosummary::
   :template: function.rst
   :toctree: generated/

   set_log
   nan_detector
   remove_nan
   fill_nan
   print_arg_pairs
   get_column_names
   lower
   get_background_value
   parc_vect_to_vol
   relabel_gifti_parc


