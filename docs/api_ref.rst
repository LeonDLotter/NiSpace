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
   :no-members:
   :no-inherited-members:

.. currentmodule:: nispace.api

.. autosummary::
   :template: class.rst
   :toctree: generated/

   NiSpace


.. _api_datasets:

:mod:`nispace.datasets` - Dataset fetchers
------------------------------------------
.. automodule:: nispace.datasets
   :no-members:
   :no-inherited-members:

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
   :no-members:
   :no-inherited-members:

.. currentmodule:: nispace.workflows

.. autosummary::
   :template: function.rst
   :toctree: generated/

   simple_colocalization
   group_comparison
   simple_xsea


.. _api_stats:

:mod:`nispace.stats` - Statistics
---------------------------------
.. automodule:: nispace.stats
   :no-members:
   :no-inherited-members:

.. currentmodule:: nispace.stats

.. autosummary::
   :template: function.rst
   :toctree: generated/

   coloc
   effectsize
   misc


.. _api_io:

:mod:`nispace.io` - Imaging data input
--------------------------------------
.. automodule:: nispace.io
   :no-members:
   :no-inherited-members:

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
   :no-members:
   :no-inherited-members:

.. currentmodule:: nispace.parcellate

.. autosummary::
   :template: class.rst
   :toctree: generated/

   Parcellater


.. _api_nulls:

:mod:`nispace.nulls` - Null map generation
------------------------------------------
.. automodule:: nispace.nulls
   :no-members:
   :no-inherited-members:

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
   :no-members:
   :no-inherited-members:

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
   :no-members:
   :no-inherited-members:

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


