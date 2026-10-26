.. _installation:

Installation
============

``NiSpace`` is not yet available on PyPI and has to be installed from GitHub.


.. _installation_requirements:

Requirements
------------

``NiSpace`` requires Python 3.9+. We recommend installation in a dedicated environment (e.g., via `Mamba <https://mamba.readthedocs.io/en/latest/>`_).  
The imaging space transformations (:mod:`nispace.transform`: MNI ↔ MNI, MNI → fsLR / fsaverage, surface ↔ surface) run in pure Python on templates that NiSpace hosts and hash-checks; `Connectome Workbench <https://www.humanconnectome.org/software/connectome-workbench>`_ is not required.
`neuromaps <https://github.com/netneurolab/neuromaps>`_ is still installed as a dependency (e.g., for null models and image loading).
Workbench is only needed to run the optional validation tests against ``wb_command`` (``tests/integration/test_transform_surface_real.py``).


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
