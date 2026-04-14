"""Merge soul + overlay ``identity_query_lexicon`` for bind-time identity state.

Lexicon data is carried on the identity object for prompts and downstream use.
The runtime does **not** map user utterances to fixed reply categories here;
that belongs to the LLM and post-LLM handling.
"""

from __future__ import annotations

import copy
from typing import Any, Dict


def merge_identity_query_lexicon(
    soul_lex: Any,
    overlay_lex: Any,
) -> Dict[str, Any]:
    """Build runtime lexicon: soul keys first, then overlay replaces same keys.

    Soul files stay immutable; tuning lives in ``config/identity_slot_overlay.json``.
    """
    base = soul_lex if isinstance(soul_lex, dict) else {}
    ovl = overlay_lex if isinstance(overlay_lex, dict) else {}
    out: Dict[str, Any] = {k: copy.deepcopy(v) for k, v in base.items()}
    for k, v in ovl.items():
        out[k] = copy.deepcopy(v)
    return out
