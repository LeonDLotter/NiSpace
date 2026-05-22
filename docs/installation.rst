.. _installation:

Installation
============

``NiSpace`` is not yet available on PyPI and has to be installed from GitHub.


.. _installation_requirements:

Requirements
------------

``NiSpace`` requires Python 3.9+. We recommend installation in a dedicated environment (e.g., via `Mamba <https://mamba.readthedocs.io/en/latest/>`_).  
The implemented imaging space transformations rely on `neuromaps <https://github.com/netneurolab/neuromaps>`_, which in turn uses `Connectome Workbench <https://www.humanconnectome.org/software/connectome-workbench>`_ to transform data from volume to surface spaces or resample surface files.
If you use this functionality, make sure that you have Workbench installed (see `neuromaps installation instructions <https://netneurolab.github.io/neuromaps/installation.html>`_).


.. _installation_github:

Installation via pip from GitHub
--------------------------------

The current development version is most conveniently installed from GitHub using pip:

.. code-block:: bash

   pip install git+https://github.com/LeonDLotter/NiSpace.git@dev

For reproducibility, consider installing a specific commit:

.. code-block:: bash

   pip install git+https://github.com/LeonDLotter/NiSpace.git@{commit_hash}

There are currently two optional dependencies: factor-analyzer and BrainSmash. When calling the respective ``NiSpace`` functions, you are prompted to install them. 
However, to avoid this, you can install them directly with the other dependencies using:

.. code-block:: bash

   pip install "git+https://github.com/LeonDLotter/NiSpace.git@dev#egg=nispace[opt]"

In the future, factor-analyzer will be removed as a dependency and all non-default null models will be kept as optional dependencies.
The default null model, `moran`, was copied from `BrainSpace <https://github.com/MICA-MNI/BrainSpace/blob/master/brainspace/null_models/moran.py>`_ and integrated into NiSpace to avoid BrainSpace and vtk as dependencies.


.. _installation_manual:

Installation from source
------------------------

Alternatively, you can clone the repository and install ``NiSpace`` manually:

.. code-block:: bash

   git clone https://github.com/LeonDLotter/NiSpace.git
   cd NiSpace
   pip install .


.. _installation_pypi:

Installation via pip from PyPI
------------------------------

``NiSpace`` is not yet available on PyPI.


.. _installation_datasets:

Integrated data
--------------------

Data (parcellations, templates, reference maps, ...) are downloaded automatically when you run ``fetch_...`` functions. 
