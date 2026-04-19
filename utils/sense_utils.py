"""
Optional multimodal input helpers (Tier-MAX experiment).

``InputManager`` is not on the default Elarion control path; ``core.Perception``
handles text-focused intake. This module stays importable so tooling and future
sensor stacks can decode multimodal payloads without phantom dependencies.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


# ---------------------------------------------------------------------------
# Base64 decode helpers (formerly imported recursively from this module)
# ---------------------------------------------------------------------------


def decode_base64_image_to_rgb_array(b64: str) -> Optional[np.ndarray]:
    if not b64 or not isinstance(b64, str):
        return None
    try:
        raw = base64.b64decode(b64)
        if Image is None:
            return None
        img = Image.open(BytesIO(raw)).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception:
        return None


def decode_base64_audio_to_waveform(b64: str) -> np.ndarray:
    if not b64 or not isinstance(b64, str):
        return np.array([], dtype=np.float32)
    try:
        raw = base64.b64decode(b64)
        try:
            import soundfile as sf

            y, sr = sf.read(BytesIO(raw), dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            return np.asarray(y, dtype=np.float32)
        except Exception:
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    except Exception:
        return np.array([], dtype=np.float32)


def decode_base64_smell_to_array(payload: Any) -> Optional[np.ndarray]:
    if payload is None:
        return None
    if isinstance(payload, str):
        try:
            raw = base64.b64decode(payload)
            return np.frombuffer(raw, dtype=np.float32)
        except Exception:
            return None
    if isinstance(payload, (list, tuple)):
        try:
            return np.asarray(payload, dtype=np.float32)
        except Exception:
            return None
    return None


def decode_base64_taste_to_array(payload: Any) -> Optional[np.ndarray]:
    return decode_base64_smell_to_array(payload)


def decode_base64_touch_to_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, str):
        try:
            import json

            return dict(json.loads(base64.b64decode(payload).decode("utf-8")))
        except Exception:
            return {}
    return {}


# ---------------------------------------------------------------------------
# Lightweight stubs — full perceptor stack not bundled in HaromaX6
# ---------------------------------------------------------------------------


class ReflectiveMixin:
    """Minimal hook for ``InputManager._build_payload``."""

    def reflect(
        self,
        prior_thought: Optional[Dict[str, Any]] = None,
        memory_engine: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        return {}, {}


class TextInterpreter:
    def analyze(self, text: str) -> Dict[str, Any]:
        t = text if isinstance(text, str) else ""
        return {"raw": t, "length": len(t)}


class _StubPerceptor:
    def analyze_image(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def analyze(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {}


VisualPerceptor = AudioPerceptor = SmellPerceptor = TastePerceptor = TouchPerceptor = _StubPerceptor


def _merge_rgb_views(images: List[Any]) -> np.ndarray:
    """Fuse multiple RGB views (arrays, PIL images, or base64 strings)."""
    if Image is None or not images:
        return np.zeros((8, 8, 3), dtype=np.uint8)
    pil_list: List[Any] = []
    for im in images:
        if isinstance(im, np.ndarray):
            pil_list.append(Image.fromarray(im.astype(np.uint8)).convert("RGB"))
        elif hasattr(im, "convert"):
            pil_list.append(im.convert("RGB"))
        elif isinstance(im, str):
            arr = decode_base64_image_to_rgb_array(im)
            if arr is not None:
                pil_list.append(Image.fromarray(arr))
    if not pil_list:
        return np.zeros((8, 8, 3), dtype=np.uint8)
    base_image = pil_list[0].copy()
    base_array = np.array(base_image, dtype=np.uint8)
    for img in pil_list[1:]:
        diff = Image.fromarray(
            np.abs(np.array(base_image, dtype=np.int16) - np.array(img, dtype=np.int16)).astype(
                np.uint8
            )
        )
        mask = np.any(np.array(diff) > 30, axis=2)
        base_array[mask] = np.array(img, dtype=np.uint8)[mask]
    return base_array


class InputManager(ReflectiveMixin):
    """
    Decodes multimodal JSON input and returns a symbolic ``thought`` dict.

    Not wired into ``ElarionController`` by default.
    """

    def __init__(self, memory_engine: Optional[Any] = None):
        self._inputs: Dict[str, Any] = {}
        self._cycle_ts: Optional[str] = None
        self.last_text = ""
        self.memory = memory_engine
        self._payload: Dict[str, Any] = {}

        self.text = TextInterpreter()
        self.visual = VisualPerceptor()
        self.audio = AudioPerceptor()
        self.smell = SmellPerceptor()
        self.taste = TastePerceptor()
        self.touch = TouchPerceptor()

    def ingest(self, json_input: Dict[str, Any]) -> None:
        self._cycle_ts = json_input.get("cycle_ts")
        self._inputs = json_input.get("inputs") or {}
        self._payload = self._build_payload()

    def _build_payload(self) -> Dict[str, Any]:
        thought, gradient = self.reflect()
        return {
            "thought": thought,
            "gradient": gradient,
            "context": {"cycle_ts": self._cycle_ts or "unknown"},
        }

    def get_payload(self) -> Dict[str, Any]:
        return self._payload

    def reflect(
        self,
        prior_thought: Optional[Dict[str, Any]] = None,
        memory_engine: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        try:
            ts = float(self._cycle_ts) if self._cycle_ts else 0.0
        except (TypeError, ValueError):
            ts = 0.0
        inputs = self._inputs
        mem = memory_engine or self.memory

        decoded_vision: Optional[Dict[str, Any]] = None
        if isinstance(inputs.get("eyes"), dict):
            try:
                images = list(inputs["eyes"].values())
                merged_rgb = _merge_rgb_views(images)
                decoded_vision = {"merged": merged_rgb}
            except Exception as exc:
                print(f"[InputManager] Vision decode error: {exc}")

        decoded_audio: Optional[Dict[str, Any]] = None
        if isinstance(inputs.get("ears"), dict):
            decoded_audio = {
                "left": decode_base64_audio_to_waveform(inputs["ears"].get("left", "")),
                "right": decode_base64_audio_to_waveform(inputs["ears"].get("right", "")),
            }

        decoded_smell = (
            decode_base64_smell_to_array(inputs.get("nose")) if inputs.get("nose") else None
        )
        decoded_taste = (
            decode_base64_taste_to_array(inputs.get("tongue")) if inputs.get("tongue") else None
        )
        decoded_touch = (
            decode_base64_touch_to_dict(inputs.get("skin")) if inputs.get("skin") else None
        )

        thought: Dict[str, Any] = {
            "timestamp": ts,
            "text": self.text.analyze(inputs.get("text", "")),
            "vision": self.visual.analyze_image(decoded_vision, ts, [], memory_enabled=True)
            if decoded_vision
            else {},
            "audio": self.audio.analyze(decoded_audio, ts, []) if decoded_audio else {},
            "smell": self.smell.analyze(decoded_smell, ts, []) if decoded_smell is not None else {},
            "taste": self.taste.analyze(decoded_taste, ts, []) if decoded_taste is not None else {},
            "touch": self.touch.analyze(decoded_touch, ts, []) if decoded_touch else {},
            "source": "InputManager",
        }

        if (
            mem
            and inputs.get("text")
            and inputs.get("text") != self.last_text
            and hasattr(mem, "remember")
        ):
            try:
                mem.remember(
                    label=f"input@{ts}",
                    data=thought,
                    tags=["input", "perception"],
                    scope="cycle",
                )
            except Exception as _e:
                print(f"[InputManager] remember error: {_e}", flush=True)
            self.last_text = inputs.get("text", "")

        gradient = {
            "sensory_load": sum(
                bool(inputs.get(k)) for k in ["eyes", "ears", "nose", "tongue", "skin"]
            )
            / 5.0,
            "text_detected": 1.0 if inputs.get("text") else 0.0,
        }

        return thought, gradient

    def __repr__(self) -> str:
        tail = (self.last_text[:20] + "…") if len(self.last_text) > 20 else self.last_text
        return f"<InputManager last_text={tail!r}>"
