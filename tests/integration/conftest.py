"""Auto-tag every test collected under this directory as integration.

Tests here may require the NiSpace data cache and/or network access and are
excluded from the default `pytest` run (see addopts in pyproject.toml). Run
explicitly with `pytest -m integration`. This exists so placing a test file
under tests/integration/ is enough on its own -- no need to remember the
@pytest.mark.integration decorator on every test.

Data directory: single point of truth, deliberately not re-checked/re-set
per test file. `nispace.datasets` sets `NISPACE_DATA_DIR` to `~/nispace-data`
at import time (a runtime download cache, separate from the
`~/projects/nispace-data` git repo -- see [[project_data_repo_ci]]/
[[project_environment]]); individual test files should just call
`fetch_reference()`/etc. directly and trust that env var, not probe/override
paths themselves. If the cache is missing or unpopulated, the whole
integration session is skipped here, once, with one clear reason, rather
than every test failing individually with an obscure network/file error.
"""

import os
from pathlib import Path

import pytest

_HERE = Path(__file__).parent


def pytest_collection_modifyitems(items):
    # This hook fires for the whole session (not scoped to this directory),
    # so filter explicitly to items collected under tests/integration/.
    for item in items:
        if _HERE in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session", autouse=True)
def _require_nispace_data_dir():
    data_dir = Path(os.environ.get("NISPACE_DATA_DIR", Path.home() / "nispace-data"))
    if not (data_dir / "template").exists():
        pytest.skip(
            f"NiSpace data cache not found at '{data_dir}' (no 'template' subdir). "
            "Integration tests need a populated NISPACE_DATA_DIR -- run any fetch_* "
            "call once locally to populate it, or set NISPACE_DATA_DIR to an "
            "existing cache."
        )
