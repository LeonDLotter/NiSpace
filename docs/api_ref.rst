.. _api:

.. currentmodule:: nispace

Reference API
==============

.. contents:: **List of modules**
   :local:


.. _api_api:

:mod:`nispace.api` - NiSpace main class
------------------------------------------
.. autoclass:: nispace.api
   :no-members:
   :no-inherited-members:

.. currentclass:: nispace.api

.. autosummary::
   :template: class.rst
   :toctree: generated/

   nispace.api.NiSpace


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

   nispace.datasets.download_datasets
   nispace.datasets.fetch_template
   nispace.datasets.fetch_parcellation
   nispace.datasets.fetch_reference
   nispace.datasets.fetch_example
   nispace.datasets.fetch_metadata
   nispace.datasets.fetch_example


.. _api_workflows:

:mod:`nispace.workflows` - Workflows
------------------------------------------
.. automodule:: nispace.workflows
   :no-members:
   :no-inherited-members:

.. currentmodule:: nispace.workflows

.. autosummary::
   :template: function.rst
   :toctree: generated/

   nispace.workflows.simple_colocalization
   nispace.workflows.group_comparison
   nispace.workflows.simple_xsea


.. _api_stats:

:mod:`nispace.stats` - Statistics
------------------------------------------
.. automodule:: nispace.stats
   :no-members:
   :no-inherited-members:

.. currentmodule:: nispace.stats

.. autosummary::
   :template: function.rst
   :toctree: generated/

   nispace.stats.coloc
   nispace.stats.effectsize
   nispace.stats.misc


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

   nispace.io.parcellate_data
   nispace.io.read_json
   nispace.io.write_json
   nispace.io.load_img
   nispace.io.load_labels
   nispace.io.load_distmat


.. _api_parcellate:

:mod:`nispace.parcellate` - Parcellation class
--------------------------------------
.. automodule:: nispace.parcellate
   :no-members:
   :no-inherited-members:

.. currentmodule:: nispace.parcellate

.. autosummary::
   :template: class.rst
   :toctree: generated/

   nispace.parcellate.Parcellater


.. _api_nulls:

:mod:`nispace.nulls` - Null map generation
--------------------------------------
.. automodule:: nispace.nulls
   :no-members:
   :no-inherited-members:

.. currentmodule:: nispace.nulls

.. autosummary::
   :template: function.rst
   :toctree: generated/

   nispace.nulls.generate_null_maps
   nispace.nulls.nulls_moran
   nispace.nulls.nulls_burt2020
   nispace.nulls.nulls_burt2018
   nispace.nulls.get_distance_matrix
   nispace.nulls.find_surf_parc_centroids


.. _api_plotting:

:mod:`nispace.plotting` - Plotting functions
--------------------------------------
.. automodule:: nispace.plotting
   :no-members:
   :no-inherited-members:

.. currentmodule:: nispace.plotting

.. autosummary::
   :template: function.rst
   :toctree: generated/

   nispace.plotting.catplot
   nispace.plotting.nullplot
   nispace.plotting.heatmap
   nispace.plotting.move_legend_fig_to_ax
   nispace.plotting.colors_from_values
   nispace.plotting.hide_empty_axes
   nispace.plotting.linewidth_from_data_units


.. _api_utils:

:mod:`nispace.utils` - Utility functions
----------------------------------------
.. automodule:: nispace.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: nispace.utils

.. autosummary::
   :template: function.rst
   :toctree: generated/

   nispace.utils.set_log
   nispace.utils.nan_detector
   nispace.utils.remove_nan
   nispace.utils.fill_nan
   nispace.utils.print_arg_pairs
   nispace.utils.get_column_names
   nispace.utils.lower
   nispace.utils.get_background_value
   nispace.utils.parc_vect_to_vol
   nispace.utils.relabel_gifti_parc


