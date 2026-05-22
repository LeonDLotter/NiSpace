.. _data_management:

Data Management
===============

``NiSpace`` ships with no data bundled. All datasets — parcellations, brain templates,
reference maps, and example data — are downloaded on demand the first time they are
requested and cached locally.


.. _data_management_location:

Storage location
----------------

By default, data are stored in ``~/nispace-data/`` on all platforms.
Set the environment variable ``NISPACE_DATA_DIR`` to use a different path:

.. code-block:: bash

   export NISPACE_DATA_DIR=/path/to/your/data

or in Python before importing ``NiSpace``:

.. code-block:: python

   import os
   os.environ["NISPACE_DATA_DIR"] = "/path/to/your/data"

The directory structure mirrors the `NiSpace data repository
<https://github.com/LeonDLotter/NiSpace-data>`__:

.. code-block:: text

   ~/nispace-data/
   ├── parcellation/
   ├── reference/
   ├── template/
   └── example/


.. _data_management_versioning:

Data versioning
---------------

The ``NiSpace`` package version and the data version are **tightly coupled**.
Internally, every ``NiSpace`` release pins a specific commit of the
`NiSpace data repository <https://github.com/LeonDLotter/NiSpace-data>`__,
so all users on the same package version receive byte-for-byte identical data.

This means:

- **Upgrading ``NiSpace``** may update or add datasets automatically on next use.
- You **cannot mix data versions**: the toolbox controls which data commit is used.
- For strict reproducibility, pin your ``NiSpace`` installation to a specific
  package commit (see :ref:`installation_github`).


.. _data_management_integrity:

Integrity checking
------------------

Every downloaded file is verified against a SHA-256 hash manifest shipped inside
the package. If a file is missing, incomplete, or corrupted, ``NiSpace`` automatically
re-downloads it — no manual intervention required. A log message marked
``Updating`` indicates that a cached file was replaced.


.. _data_management_reproducibility:

Reproducibility
---------------

``NiSpace`` logs the package version and source-code commit at the start of every
analysis. Include this information when reporting results:

.. code-block:: python

   import nispace
   print(nispace.__version__)   # package version
   print(nispace.__commit__)    # exact source commit

For full reproducibility, report both the ``NiSpace`` version (which implicitly
identifies the data version) and the specific commit hash.
