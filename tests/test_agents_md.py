"""AGENTS.md (the guide for AI assistants) must only mention names and paths that exist.

The file deliberately avoids defaults and other volatile details; what remains are function,
method, parameter and file names. These tests fail when one of them is renamed or removed.
"""
import builtins
import importlib
import inspect
import re
from pathlib import Path

import pytest

import nispace
from nispace import NiSpace

REPO = Path(__file__).resolve().parents[1]
AGENTS_MD = REPO / "AGENTS.md"

# namespaces in which bare names such as `fit()` or `fetch_axes()` may live
_NAMESPACES = ["nispace", "nispace.datasets", "nispace.diagnostics", "nispace.plotting",
               "nispace.nulls", "nispace.transform", "nispace.parcellate", "nispace.stats.autocorr"]


@pytest.fixture(scope="module")
def text():
    return AGENTS_MD.read_text()


@pytest.fixture(scope="module")
def code_spans(text):
    return re.findall(r"`([^`\n]+)`", text)


def _resolve_dotted(name):
    """Import the longest importable module prefix of `name`, then getattr the rest."""
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(name)


def _known_callables():
    objs = {}
    for mod_name in _NAMESPACES:
        mod = importlib.import_module(mod_name)
        objs.update({n: o for n, o in vars(mod).items() if callable(o) and not n.startswith("_")})
    objs.update({n: o for n, o in vars(NiSpace).items() if callable(o) and not n.startswith("_")})
    return objs


def _known_parameters(callables):
    params = set()
    for obj in callables.values():
        try:
            params |= set(inspect.signature(obj).parameters)
        except (TypeError, ValueError):
            pass
    return params


def test_relative_links_exist(text):
    links = [l for l in re.findall(r"\]\(([^)\s]+)\)", text) if "://" not in l]
    assert links
    missing = [l for l in links if not (REPO / l).exists()]
    assert not missing, f"AGENTS.md links to missing paths: {missing}"


def test_dotted_names_exist(code_spans):
    dotted = {m.group(0) for s in code_spans for m in re.finditer(r"nispace(\.\w+)+", s)}
    assert dotted
    missing = []
    for name in sorted(dotted):
        if (REPO / name.replace(".", "/")).with_suffix(".py").exists():
            continue
        try:
            _resolve_dotted(name)
        except (ImportError, AttributeError):
            missing.append(name)
    assert not missing, f"AGENTS.md mentions missing objects: {missing}"


def test_called_names_exist(code_spans):
    # `fit()`, `colocalize("dominance")`, `nsp.get_x()`, `NiSpace(z=...)`, `help(...)`
    called = {m.group(1) for s in code_spans
              for m in re.finditer(r"(?:^|[\s.(])(\w+)\(", s)}
    known = set(_known_callables()) | {"NiSpace"} | set(dir(builtins))
    known |= {"corr", "tanh"}  # pandas / numpy methods used in examples
    missing = sorted(called - known)
    assert not missing, f"AGENTS.md calls unknown functions or methods: {missing}"


def test_parameter_names_exist(code_spans):
    params = {m.group(1) for s in code_spans for m in re.finditer(r"(\w+)\s*[=>]", s)}
    known = _known_parameters(_known_callables())
    missing = sorted(params - known)
    assert not missing, f"AGENTS.md mentions unknown parameters: {missing}"


def test_quoted_colocalization_methods_exist(code_spans):
    from nispace.core.constants import _COLOC_METHODS
    used = {m.group(1) for s in code_spans
            for m in re.finditer(r'colocalize\("(\w+)"\)', s)}
    used |= {s.strip('"') for s in code_spans if re.fullmatch(r'"\w+"', s)}
    missing = sorted(used - set(_COLOC_METHODS))
    assert not missing, f"AGENTS.md mentions unknown colocalization methods: {missing}"
