"""Execute NiSpace documentation notebooks in place.

Usage
-----
Run all intro notebooks:
    python docs/run_notebooks.py --intro

Run all example notebooks:
    python docs/run_notebooks.py --examples

Run both:
    python docs/run_notebooks.py

Options:
    --kernel KERNEL   Jupyter kernel name (default: from notebook metadata)
    --timeout SECS    Cell execution timeout in seconds (default: 1200)
    --force           Continue after a notebook failure and report all results at the end
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import nbformat
from nbformat.v4 import new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

# Injected as the very first cell before any NiSpace import so that
# 'from tqdm.auto import tqdm' in NiSpace modules binds our subclass.
# miniters/mininterval=inf suppresses intermediate refresh calls;
# tqdm.close() still calls display() once (the final 100% line).
_TQDM_PATCH_CODE = """\
import tqdm, tqdm.std, tqdm.auto, tqdm.notebook as _tqdm_nb
_OrigTqdm = tqdm.std.tqdm
class _SingleLineTqdm(_OrigTqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("miniters", float("inf"))
        kwargs.setdefault("mininterval", float("inf"))
        super().__init__(*args, **kwargs)
    def display(self, msg=None, pos=None):
        if self.n > 0:  # skip the initial 0% line emitted by __init__
            super().display(msg, pos)
tqdm.tqdm = _SingleLineTqdm
tqdm.std.tqdm = _SingleLineTqdm
tqdm.auto.tqdm = _SingleLineTqdm
_tqdm_nb.tqdm = _SingleLineTqdm
"""

DOCS_DIR = Path(__file__).parent
NB_DIRS = {
    "intro":    DOCS_DIR / "nb_introduction",
    "examples": DOCS_DIR / "nb_examples",
}


@dataclass
class _NBResult:
    name: str
    success: bool
    elapsed: float
    cell_index: Optional[int] = None   # 1-based, None if not determinable
    ename: Optional[str] = None
    evalue: Optional[str] = None

    def summary_line(self) -> str:
        if self.success:
            return f"  {self.name}: ran without error ({self.elapsed:.0f}s)"
        cell = f"cell {self.cell_index}" if self.cell_index is not None else "unknown cell"
        return f"  {self.name}: error in {cell}: {self.ename}: {self.evalue}"


def _find_error_cell(nb) -> tuple:
    """Scan a partially-executed notebook for the first cell with an error output.
    Returns (1-based cell index, ename, evalue) or (None, None, None)."""
    for i, cell in enumerate(nb.cells):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                return i + 1, output.get("ename", "Error"), output.get("evalue", "")
    return None, None, None


def _run_notebook(nb_path: Path, kernel: Optional[str], timeout: int) -> _NBResult:
    """Execute a single notebook in place. Returns a result object."""
    print(f"  {nb_path.name} ... ", end="", flush=True)
    t0 = time.time()
    try:
        with open(nb_path) as f:
            nb = nbformat.read(f, as_version=4)

        k = kernel or nb.get("metadata", {}).get("kernelspec", {}).get("name", "python3")
        ep = ExecutePreprocessor(timeout=timeout, kernel_name=k)
        nb.cells.insert(0, new_code_cell(source=_TQDM_PATCH_CODE))
        try:
            ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})
        finally:
            nb.cells.pop(0)

        with open(nb_path, "w") as f:
            nbformat.write(nb, f)

        elapsed = time.time() - t0
        print(f"OK ({elapsed:.0f}s)")
        return _NBResult(nb_path.name, success=True, elapsed=elapsed)

    except CellExecutionError:
        elapsed = time.time() - t0
        cell_idx, ename, evalue = _find_error_cell(nb)
        print(f"FAILED ({elapsed:.0f}s)")
        return _NBResult(nb_path.name, success=False, elapsed=elapsed,
                         cell_index=cell_idx, ename=ename, evalue=evalue)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAILED ({elapsed:.0f}s)")
        return _NBResult(nb_path.name, success=False, elapsed=elapsed,
                         ename=type(e).__name__, evalue=str(e))


def _run_all(notebooks: List[Path], kernel: Optional[str],
             timeout: int, force: bool) -> int:
    results = []
    for nb_path in notebooks:
        result = _run_notebook(nb_path, kernel, timeout)
        results.append(result)
        if not result.success:
            if not force:
                # report immediately and stop
                cell = f"cell {result.cell_index}" if result.cell_index else "unknown cell"
                print(f"\n  Error in {cell}: {result.ename}: {result.evalue}")
                print("  Use --force to continue past errors and see a full report.")
                break

    if force:
        print("\nResults:")
        for r in results:
            print(r.summary_line())

    failures = sum(1 for r in results if not r.success)
    if failures:
        print(f"\n{failures}/{len(results)} notebook(s) failed.")
    else:
        print(f"\nAll {len(results)} notebook(s) completed successfully.")
    return failures


def run_intro_notebooks(kernel: Optional[str] = None, timeout: int = 1200,
                        force: bool = False) -> int:
    """Execute all intro notebooks in docs/nb_introduction/. Returns number of failures."""
    nb_dir = NB_DIRS["intro"]
    notebooks = sorted(nb_dir.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {nb_dir}")
        return 0
    print(f"Running {len(notebooks)} intro notebook(s):")
    return _run_all(notebooks, kernel, timeout, force)


def run_example_notebooks(kernel: Optional[str] = None, timeout: int = 1200,
                           force: bool = False) -> int:
    """Execute all example notebooks in docs/nb_examples/. Returns number of failures."""
    nb_dir = NB_DIRS["examples"]
    notebooks = sorted(nb_dir.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {nb_dir} — nothing to run.")
        return 0
    print(f"Running {len(notebooks)} example notebook(s):")
    return _run_all(notebooks, kernel, timeout, force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute NiSpace documentation notebooks.")
    parser.add_argument("--intro",    action="store_true", help="Run intro notebooks")
    parser.add_argument("--examples", action="store_true", help="Run example notebooks")
    parser.add_argument("--kernel",   default=None, help="Jupyter kernel name")
    parser.add_argument("--timeout",  default=1200, type=int, help="Cell timeout in seconds")
    parser.add_argument("--force",    action="store_true",
                        help="Continue after failures; print full report at the end")
    args = parser.parse_args()

    run_intro    = args.intro or not (args.intro or args.examples)
    run_examples = args.examples or not (args.intro or args.examples)

    total_failures = 0
    if run_intro:
        total_failures += run_intro_notebooks(args.kernel, args.timeout, args.force)
    if run_examples:
        total_failures += run_example_notebooks(args.kernel, args.timeout, args.force)

    sys.exit(1 if total_failures else 0)
