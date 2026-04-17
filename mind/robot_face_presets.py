"""Shared emotion → ASCII face lines (aligned with ``scripts/haroma_chat_buddy``)."""

from __future__ import annotations

from typing import Dict

# Episode / EmotionEngine labels + melancholy
_EMOTIONS = (
    "neutral",
    "joy",
    "wonder",
    "curiosity",
    "fear",
    "sadness",
    "anger",
    "resolve",
    "peace",
    "surprise",
    "melancholy",
)

_FACES: Dict[str, str] = {
    "neutral": "  ( · · )   ",
    "joy": "  ( ‿‿ )    ",
    "wonder": "  ( ☆‿☆ )  ",
    "curiosity": "  ( ◕‿◕ )  ",
    "fear": "  ( @_@ )   ",
    "sadness": "  ( T_T )   ",
    "anger": "  ( >_< )   ",
    "resolve": "  ( •̀ᴗ•́ ) ",
    "peace": "  ( -.- )   ",
    "surprise": "  ( O_O )   ",
    "melancholy": "  ( ‸ ‸ )  ",
}

_FACES_ASCII: Dict[str, str] = dict(_FACES)
_FACES_ASCII.update(
    {
        "neutral": "  ( o o )   ",
        "resolve": "  ( -_- )b  ",
        "melancholy": "  ( u u )   ",
    }
)


def normalize_emotion_label(raw: object) -> str:
    k = str(raw or "neutral").lower().strip()
    return k if k in _FACES else "neutral"


def ascii_face_line(emo: str, frame: int, *, ascii_only: bool = False) -> str:
    """One face line; *frame* toggles idle nudge for calm moods (buddy script)."""
    table = _FACES_ASCII if ascii_only else _FACES
    key = normalize_emotion_label(emo)
    base = table.get(key, table["neutral"])
    if key in ("neutral", "peace") and (frame % 4) == 2:
        return " " + base
    return base


def emotion_intensity_boost(emo: str, intensity: float, line: str) -> str:
    e = normalize_emotion_label(emo)
    if intensity > 0.65 and e in ("anger", "fear", "surprise"):
        return line.rstrip() + " !"
    return line
