"""Stable client identity for HTTP /chat.

Flow: ``POST /chat`` → :func:`sanitize_user_id` at the Flask boundary →
:class:`agents.input_agent.InputAgent` attaches only non-empty ids → TrueSelf
(same ``Message.content``) → :class:`agents.persona_agent.PersonaAgent` uses
:func:`speaker_key` / :func:`user_tag` for ToM, conversation, recall, and memory
nodes. Specialist delegation uses ``copy.deepcopy`` of content; fast path is
unchanged.
"""

from __future__ import annotations

import re
from typing import Optional

_MAX_LEN = 128


def sanitize_user_id(raw: Optional[object]) -> Optional[str]:
    """Return a filesystem-safe token or None if absent/invalid."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s = re.sub(r"[^a-zA-Z0-9_\-@.+]", "_", s).strip("._")
    if not s:
        return None
    return s[:_MAX_LEN]


def user_tag(user_id: Optional[object]) -> Optional[str]:
    uid = sanitize_user_id(user_id)
    if uid:
        return f"user:{uid}"
    return None


def speaker_key(role: str, user_id: Optional[object]) -> str:
    """Interlocutor / conversation speaker: ``user:<id>`` when set, else *role*."""
    t = user_tag(user_id)
    if t:
        return t
    return role
