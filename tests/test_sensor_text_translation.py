"""mind.sensor_text_translation — per-reading text lines for multimodal input."""

from __future__ import annotations

import sys

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mind.sensor_text_translation import (
    enrich_sensor_data,
    reading_text_translation,
    sensor_text_translation_digest,
)


def test_chat_reading():
    t = reading_text_translation(
        "chat",
        {"text": "hello", "source": "user", "channel": "chat"},
    )
    assert "hello" in t
    assert "user" in t.lower()


def test_vision_reading():
    t = reading_text_translation(
        "vision",
        {
            "resolution": [640, 480],
            "brightness": 120.5,
            "edge_density": 0.12,
        },
    )
    assert "vision" in t.lower()
    assert "640" in t


def test_enrich_adds_text_translation():
    sd = {
        "vision": [{"brightness": 99.0, "edge_density": 0.1}],
        "chat": [{"text": "hi", "source": "user"}],
    }
    enrich_sensor_data(sd)
    assert "text_translation" in sd["vision"][0]
    assert "text_translation" in sd["chat"][0]


def test_digest_skips_chat_only_modalities():
    sd = {
        "vision": [{"brightness": 1.0}],
        "chat": [{"text": "x", "source": "user"}],
    }
    enrich_sensor_data(sd)
    d = sensor_text_translation_digest(sd)
    assert "vision" in d.lower()
    assert "x" not in d  # chat text lives in user message, not digest
