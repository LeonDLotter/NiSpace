.. _installation:

Installation
============

``NiSpace`` is not yet available on PyPI and has to be installed from GitHub.


.. _installation_requirements:

Requirements
------------

``NiSpace`` requires Python 3.9+; we recommend Python 3.12, the newest version our tests run on. Support for Python 3.9 will end in a future release. We recommend installation in a dedicated environment (e.g., via `Mamba <https://mamba.readthedocs.io/en/latest/>`_).  
The imaging space transformations (:mod:`nispace.transform`: MNI ↔ MNI, MNI → fsLR / fsaverage, surface ↔ surface) run in pure Python on templates that NiSpace hosts and hash-checks; `Connectome Workbench <https://www.humanconnectome.org/software/connectome-workbench>`_ is not required.
`neuromaps <https://github.com/netneurolab/neuromaps>`_ is still installed as a dependency (e.g., for null models and image loading).
Workbench is only needed to run the optional validation tests against ``wb_command`` (``tests/integration/test_transform_surface_real.py``).


.. _installation_github:

Installation via pip from GitHub
--------------------------------

The latest release (`0.1.0b1 <https://github.com/LeonDLotter/NiSpace/releases>`__, a beta version) is most conveniently installed from GitHub using pip:

.. code-block:: bash

   pip install git+https://github.com/LeonDLotter/NiSpace.git@v0.1.0b1

The development version has the newest changes, but its behaviour can change without a release:

.. code-block:: bash

   pip install git+https://github.com/LeonDLotter/NiSpace.git@dev

For reproducibility, note the installed version (``nispace.__version__``), or install a specific commit:

.. code-block:: bash

   pip install git+https://github.com/LeonDLotter/NiSpace.git@{commit_hash}

Some functions need optional dependencies: BrainSMASH (variogram null maps), factor-analyzer, nitransforms, NiMARE (meta-analysis helpers), and neuroHarmonize (the deprecated ComBat option of ``clean_y()``). When calling the respective ``NiSpace`` functions, you are prompted to install them.
However, to avoid this, you can install them directly with the other dependencies using:

.. code-block:: bash

   pip install "nispace[opt] @ git+https://github.com/LeonDLotter/NiSpace.git@v0.1.0b1"

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
