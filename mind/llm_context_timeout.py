"""Read global packed-LLM context timeout (seconds) for HTTP wait math and ``/status``.

Application code should prefer :func:`mind.cognitive_contracts.llm_context_timeout_seconds`
(same object). This module implements the lazy delegate to the engine helper via
:mod:`mind.prompt_packaging` to avoid import cycles with :mod:`mind.cognitive_contracts`.
"""

from __future__ import annotations

from typing import Optional


def llm_context_timeout_seconds() -> Optional[float]:
    """Return engine cap, or ``None`` if unlimited / unavailable / import failure."""
    try:
        from mind.prompt_packaging import packed_llm_timeout_seconds

        return packed_llm_timeout_seconds()
    except Exception:
        return None
