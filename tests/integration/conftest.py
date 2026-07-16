"""Auto-tag every test collected under this directory as integration.

Tests here may require ~/projects/nispace-data and/or network access and are
excluded from the default `pytest` run (see addopts in pyproject.toml). Run
explicitly with `pytest -m integration`. This exists so placing a test file
under tests/integration/ is enough on its own -- no need to remember the
@pytest.mark.integration decorator on every test.
"""

from pathlib import Path

import pytest

_HERE = Path(__file__).parent


def pytest_collection_modifyitems(items):
    # This hook fires for the whole session (not scoped to this directory),
    # so filter explicitly to items collected under tests/integration/.
    for item in items:
        if _HERE in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.integration)
