.. _installation:

Installation
============

``NiSpace`` is not yet available on PyPI and has to be installed from GitHub.


.. _installation_requirements:

Requirements
------------

``NiSpace`` requires Python 3.9+. We recommend installation in a dedicated environment (e.g., via `Anaconda <https://www.anaconda.com/>`_).  
The implemented imaging space transformations rely on `neuromaps <https://github.com/netneurolab/neuromaps>`_, which in turn uses `Connectome Workbench <https://www.humanconnectome.org/software/connectome-workbench>`_ to transform data from volume to surface spaces or resample surface files.
If you use this functionality, make sure that you have Workbench installed (see `neuromaps installation instructions <https://netneurolab.github.io/neuromaps/installation.html>`_).


.. _installation_github:

Installation from GitHub
------------------------

The current development version is most conveniently installed from GitHub using pip:

.. code-block:: bash

   pip install git+https://github.com/LeonDLotter/NiSpace.git@dev

There are some optional dependencies. When calling the respective ``NiSpace`` functions, you are prompted to install them. 
However, to avoid this, you can install them directly with the other dependencies using:

.. code-block:: bash

   pip install "git+https://github.com/LeonDLotter/NiSpace.git@dev#egg=nispace[opt]"

Alternatively, you can clone the repository and install ``NiSpace`` manually:

.. code-block:: bash

   git clone https://github.com/LeonDLotter/NiSpace.git
   cd NiSpace
   pip install .


.. _installation_datasets:

Downloading datasets
--------------------

To use the integrated datasets, parcellations, and templates, run the following in Python:

.. code-block:: python

   import nispace
   nispace.datasets.download_datasets()

This will download about 250 MB of data to ``~HOME/nispace-data``. You can adjust the path with the ``nispace_data_path`` keyword argument.
In the future, we will adjust the ``fetch_...`` functions to automatically download requested datasets.
