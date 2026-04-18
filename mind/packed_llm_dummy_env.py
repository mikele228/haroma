"""``HAROMA_LLM_DUMMY_REPLY`` probes — re-export :mod:`mind.haroma_settings`.

Kept so ``mind`` modules and ``engine`` can import a stable, descriptive module
name; canonical definitions are in :mod:`mind.haroma_settings`.
"""

from __future__ import annotations

from mind.haroma_settings import (
    packed_llm_dummy_reply_raw,
    synthetic_llm_dummy_reply_env,
)


def packed_llm_dummy_probe_active() -> bool:
    """Alias for :func:`synthetic_llm_dummy_reply_env`."""
    return synthetic_llm_dummy_reply_env()
