"""
Pre-load stub so ``core.Memory`` skips optional ``sentence_transformers`` (avoids pulling torch via SBERT).

Executed via ``conftest.py`` before other Haroma imports.
HTTP test modules can call :func:`prepare_test_imports` (after repo is on path).

Torch: use :func:`torch_loads_in_subprocess` or :func:`skip_unless_torch_imports` —
importing ``torch`` in the pytest process can raise :exc:`OSError` or **crash** the
interpreter on some Windows hosts; a subprocess probe avoids that during collection.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Optional

__all__ = [
    "prepare_test_imports",
    "torch_loads_in_subprocess",
    "skip_unless_torch_imports",
]

_TORCH_SUBPROC_OK: Optional[bool] = None


def _ensure_sentence_transformers_stub() -> None:
    if "sentence_transformers" not in sys.modules:
        sys.modules["sentence_transformers"] = types.ModuleType("sentence_transformers")


_ensure_sentence_transformers_stub()


def prepare_test_imports(test_file: str) -> None:
    """Put repo root on ``sys.path`` and ensure the sentence_transformers stub.

    Use at the top of HTTP-focused test modules::
        prepare_test_imports(__file__)
    """
    p = Path(test_file).resolve()
    repo = p.parent.parent
    r = str(repo)
    if r not in sys.path:
        sys.path.insert(0, r)
    _ensure_sentence_transformers_stub()


def torch_loads_in_subprocess() -> bool:
    """Return whether ``import torch`` succeeds in a child interpreter (cached).

    Use with ``@pytest.mark.skipif(not torch_loads_in_subprocess(), ...)`` for tests
    that import modules which load torch lazily at runtime.

    Set ``HAROMA_SKIP_TORCH_TESTS=1`` to treat torch as unavailable without probing.
    """
    global _TORCH_SUBPROC_OK
    if _TORCH_SUBPROC_OK is not None:
        return _TORCH_SUBPROC_OK

    import os
    import subprocess

    raw = str(os.environ.get("HAROMA_SKIP_TORCH_TESTS", "") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        _TORCH_SUBPROC_OK = False
        return False

    try:
        r = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch; assert torch.__version__",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        _TORCH_SUBPROC_OK = False
        return False

    _TORCH_SUBPROC_OK = r.returncode == 0
    return _TORCH_SUBPROC_OK


def skip_unless_torch_imports() -> None:
    """Skip the whole test module if ``torch`` is not loadable (see :func:`torch_loads_in_subprocess`)."""
    import pytest

    if not torch_loads_in_subprocess():
        pytest.skip(
            "torch not loadable (subprocess probe failed or HAROMA_SKIP_TORCH_TESTS=1)",
            allow_module_level=True,
        )
