from typing import Dict, List, Any, Optional
from utils.module_base import ModuleBase
from core.NLUProcessor import NLUProcessor
import time
import uuid


class PerceptionBridge(ModuleBase):
    """
    Tier MAX+ Core Module: Transforms raw inputs into symbolic perceptions,
    tags them, assigns moment IDs, and logs them for future context anchoring.

    Handles both direct text input ({"content": "...", "tags": [...]}) and
    multimodal intake blobs ({"eyes": {...}, "ears": {...}, "text": "..."}).

    Phase 5: Runs spaCy NLU on every text input and attaches structured
    entities, relations, intent, and sentiment to the perception output.
    """

    MODALITY_KEYS = {
        "eyes": "vision",
        "ears": "audio",
        "text": "text",
        "nose": "olfactory",
        "tongue": "gustatory",
        "skin": "tactile",
    }

    def __init__(self):
        super().__init__("PerceptionBridge")
        self.percept_log: List[Dict[str, Any]] = []
        self.nlu = NLUProcessor()

    def perceive(self, raw_input: Dict[str, Any], channel: str = "text") -> Dict[str, Any]:
        moment_id = str(uuid.uuid4())
        content = self._extract_content(raw_input)
        tags = self._extract_tags(raw_input, content)
        modalities = self._detect_modalities(raw_input)

        nlu_result = self.nlu.process(content)

        for ent in nlu_result.get("entities", []):
            tag = ent["text"].lower().replace(" ", "_")
            if tag and tag not in tags:
                tags.append(tag)
        tags = tags[:25]

        symbolic = {
            "moment_id": moment_id,
            "timestamp": time.time(),
            "channel": channel,
            "content": content,
            "tags": tags,
            "modalities": modalities,
            "nlu": nlu_result,
            "raw": raw_input,
        }
        self.percept_log.append(symbolic)
        if len(self.percept_log) > 2000:
            self.percept_log = self.percept_log[-2000:]
        return symbolic

    def _extract_content(self, raw: Dict[str, Any]) -> str:
        if "content" in raw and isinstance(raw["content"], str):
            return raw["content"]

        if "text" in raw and isinstance(raw["text"], str):
            return raw["text"]

        parts: List[str] = []

        for key in ("thought", "data", "message"):
            val = raw.get(key)
            if isinstance(val, str) and val:
                parts.append(val)

        if "eyes" in raw:
            eye_data = raw["eyes"]
            if isinstance(eye_data, dict):
                if "detected_objects" in eye_data:
                    objs = eye_data["detected_objects"]
                    parts.append(f"I see: {', '.join(objs[:5])}")
                else:
                    parts.append("visual input received")

        if "ears" in raw:
            ear_data = raw["ears"]
            if isinstance(ear_data, dict):
                if "transcript" in ear_data:
                    parts.append(ear_data["transcript"])
                else:
                    parts.append("audio input received")

        if "nose" in raw:
            parts.append("olfactory input received")
        if "tongue" in raw:
            parts.append("gustatory input received")
        if "skin" in raw:
            parts.append("tactile input received")

        return ". ".join(parts) if parts else ""

    def _extract_tags(self, raw: Dict[str, Any], content: str) -> List[str]:
        tags: List[str] = []

        if isinstance(raw.get("tags"), list):
            tags.extend(str(t).lower() for t in raw["tags"])

        if content:
            words = content.lower().split()
            tags.extend(
                w.strip(".,!?;:'\"()[]") for w in words if len(w.strip(".,!?;:'\"()[]")) > 3
            )

        modalities = self._detect_modalities(raw)
        for mod in modalities:
            if mod != "text":
                tags.append(f"{mod}_input")

        seen = set()
        unique: List[str] = []
        for t in tags:
            if t and t not in seen:
                seen.add(t)
                unique.append(t)
        return unique[:15]

    def _detect_modalities(self, raw: Dict[str, Any]) -> List[str]:
        active: List[str] = []
        for key, label in self.MODALITY_KEYS.items():
            if key in raw and raw[key]:
                active.append(label)

        if "content" in raw and isinstance(raw["content"], str) and raw["content"]:
            if "text" not in active:
                active.append("text")

        return active if active else ["text"]

    def summarize(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.percept_log[-limit:]

    def reset(self):
        self.percept_log.clear()

    def __repr__(self):
        return f"<PerceptionBridge log_size={len(self.percept_log)}>"


# === MAX++ UPDATED MODULE ===
from typing import List, Dict, Any, Tuple
import numpy as np


class AudioMemoryAnalysis:
    """
    Tier MAX++: Symbolic audio learner from waveform deltas.
    Uses previous frames to build memory patterns for tag inference.
    Operates on external buffer.
    """

    def __init__(self, recent_buffer: List[Dict[str, Any]]):
        self.recent_buffer = recent_buffer
        self.pattern_memory: Dict[Tuple[int, float], Dict[str, int]] = {}

    def _simplify(self, delta: np.ndarray, segments: int = 32) -> np.ndarray:
        if len(delta) < segments:
            return np.pad(delta, (0, segments - len(delta)))
        reshaped = delta[: len(delta) // segments * segments].reshape(segments, -1)
        return np.mean(np.abs(reshaped), axis=1)

    def _learn_from_delta(self, simplified: np.ndarray, tags: List[str]) -> None:
        for i, val in enumerate(simplified):
            key = (i, round(val, 3))
            if key not in self.pattern_memory:
                self.pattern_memory[key] = {}
            for tag in tags:
                self.pattern_memory[key][tag] = self.pattern_memory[key].get(tag, 0) + 1

    def _infer_from_delta(self, simplified: np.ndarray) -> Dict[str, Any]:
        tag_votes: Dict[str, int] = {}
        for i, val in enumerate(simplified):
            key = (i, round(val, 3))
            for tag, count in self.pattern_memory.get(key, {}).items():
                tag_votes[tag] = tag_votes.get(tag, 0) + count

        if not tag_votes:
            return {"predicted_tags": [], "confidence": 0.0, "zone": "zone_audio"}

        sorted_tags = sorted(tag_votes.items(), key=lambda x: x[1], reverse=True)
        total = sum(tag_votes.values())
        top_tags = [t[0] for t in sorted_tags[:3]]
        conf = sorted_tags[0][1] / total if total > 0 else 0.0

        return {"predicted_tags": top_tags, "confidence": round(conf, 3), "zone": "zone_audio"}

    def check_audio(self, audio: np.ndarray) -> Dict[str, Any]:
        delta = np.zeros_like(audio)
        simplified = np.zeros(32)

        if self.recent_buffer:
            prev_frame = self.recent_buffer[-1]
            prev_audio = prev_frame["audio"]
            min_len = min(len(prev_audio), len(audio))
            delta = audio[:min_len] - prev_audio[:min_len]
            simplified = self._simplify(delta)

            prev_tags = prev_frame.get("inferred_tags", {}).get("predicted_tags", [])
            if prev_tags:
                self._learn_from_delta(simplified, prev_tags)

        return self._infer_from_delta(simplified)

    def __repr__(self):
        return f"<AudioMemoryAnalysis TierMAX++ memory_keys={len(self.pattern_memory)} buffer_len={len(self.recent_buffer)} zone=zone_audio>"


# === MAX++ UPDATED MODULE ===
from typing import Dict, List, Any
import numpy as np
from utils.sense_transform import SenseTransform


class AudioPluginAnalysis:
    """
    Tier MAX++: Symbolic audio fallback tagger using SenseTransform.
    Extracts symbolic tags based on pitch, volume, and acoustic features.
    """

    def __init__(self):
        self.sense = SenseTransform()
        self.logs: List[Dict[str, Any]] = []

    def tag_audio(
        self, audio_dict: Dict[str, np.ndarray], sample_rate: int = 44100
    ) -> Dict[str, Any]:
        try:
            left_audio = audio_dict.get("left")
            if left_audio is None or not isinstance(left_audio, np.ndarray):
                raise ValueError("Missing or invalid 'left' audio input.")

            result = self.sense.sound_tag_all(left_audio, sample_rate=sample_rate)

            tags = []
            volume = result.get("volume_level")
            if volume is not None:
                tags.append(volume)

            pitch = result.get("mean_pitch_hz", 0)
            if pitch > 2000:
                tags.append("high_pitch")
            elif pitch < 300:
                tags.append("low_pitch")
            else:
                tags.append("midrange")

            summary = {
                "tags": tags,
                "confidence": result.get("confidence", 0.4),
                "zone": "zone_audio",
                "summary": result,
            }

            self.logs.append(summary)
            return summary

        except Exception as e:
            error_summary = {
                "tags": ["corrupted"],
                "confidence": 0.0,
                "error": str(e),
                "zone": "zone_audio",
            }
            self.logs.append(error_summary)
            return error_summary

    def get_recent_logs(self, count: int = 5) -> List[Dict[str, Any]]:
        return self.logs[-count:]

    def __repr__(self):
        return f"<AudioPluginAnalysis TierMAX++ logs={len(self.logs)} zone=zone_audio>"


# === MAX++ UPGRADED MODULE ===
import numpy as np
from typing import Optional, List, Dict, Any


class AudioSymbolicPerceptor:
    """
    Tier MAX++: Performs symbolic hearing by chunking audio input,
    checking against memory analyzer first, then falling back to plugin tagging.
    Produces symbolic thought blocks per chunk.
    """

    def __init__(self, memory_analyzer=None, plugin_analyzer=None):
        self.memory_analyzer = memory_analyzer
        self.plugin_analyzer = plugin_analyzer

    def extract_chunks(self, signal: np.ndarray, chunk_size: int = 2048):
        for i in range(0, len(signal), chunk_size):
            yield i, signal[i : i + chunk_size]

    def vectorize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        return chunk.astype(np.float32) / (np.max(np.abs(chunk)) + 1e-6)

    def check_memory(self, vector: np.ndarray) -> Optional[List[str]]:
        try:
            result = self.memory_analyzer.check_audio(vector)
            tags = result.get("predicted_tags", [])
            confidence = result.get("confidence", 0.0)
            return tags if confidence >= 0.5 else None
        except Exception:
            return None

    def fallback_tag(self, chunk: np.ndarray) -> List[str]:
        try:
            result = self.plugin_analyzer.tag_audio({"left": chunk})
            tags = result.get("tags", []) if isinstance(result, dict) else []
            return tags if tags else ["unclassified"]
        except Exception:
            return ["unclassified"]

    def build_thought_block(
        self, index: int, tags: List[str], vector: np.ndarray, chunk: np.ndarray
    ) -> Dict[str, Any]:
        return {
            "chunk_id": f"{index}",
            "tags": tags,
            "vector": vector.tolist(),
            "raw": chunk.tolist(),
            "zone": "zone_audio",
        }

    def process_signal(self, signal: np.ndarray, chunk_size: int = 2048) -> Dict[str, Any]:
        thought = {"audio": []}
        for index, chunk in self.extract_chunks(signal, chunk_size):
            if len(chunk) == 0:
                continue
            vector = self.vectorize_chunk(chunk)
            tags = self.check_memory(vector)
            if not tags:
                tags = self.fallback_tag(chunk)
            block = self.build_thought_block(index, tags, vector, chunk)
            thought["audio"].append(block)
        return thought

    def __repr__(self):
        return f"<AudioSymbolicPerceptor TierMAX++>"


# === MAX UPGRADED MODULE ===
from collections import Counter
from typing import List, Dict, Any


class PerceptualSynthesizer:
    """
    Tier MAX: Fuses text, vision, and audio tags into a unified symbolic context
    with modality analysis, contradiction detection, and coherence scoring.
    """

    def __init__(self):
        self.last_fusion: Dict[str, Any] = {}

    def fuse(
        self, text_tags: List[str], vision_tags: List[str], audio_tags: List[str]
    ) -> Dict[str, Any]:
        all_inputs = text_tags + vision_tags + audio_tags
        tag_counter = Counter(all_inputs)
        all_tags = list(tag_counter.keys())

        # Weighted density per modality
        weights = {
            "text": len(set(text_tags)),
            "vision": len(set(vision_tags)),
            "audio": len(set(audio_tags)),
        }
        dominant_modality = max(weights, key=weights.get)

        # Base confidence logic
        confidence = 0.3
        if "speech_range" in audio_tags or "observation" in text_tags:
            confidence += 0.2
        if "cluttered" in vision_tags:
            confidence += 0.1
        if "tense" in audio_tags or "alert" in text_tags:
            confidence -= 0.1

        # Contradiction detection
        contradictions = []
        if "calm" in audio_tags and "tense" in audio_tags:
            contradictions.append("calm vs tense (audio)")
        if "minimalist" in vision_tags and "cluttered" in vision_tags:
            contradictions.append("minimalist vs cluttered (vision)")

        # Coherence score
        coherence = round(1.0 - 0.1 * len(contradictions), 2)
        confidence = round(min(max(confidence, 0.0), 1.0), 2)

        fusion_result = {
            "tags": all_tags,
            "confidence": confidence,
            "dominant_modality": dominant_modality,
            "tag_frequency": dict(tag_counter),
            "contradictions": contradictions,
            "coherence_score": coherence,
        }

        self.last_fusion = fusion_result
        return fusion_result

    def get_last_fusion(self) -> Dict[str, Any]:
        return self.last_fusion

    def __repr__(self):
        return f"<FusionContextBuilder TierMAX coherence={self.last_fusion.get('coherence_score', 'N/A')}>"


# === MAX++ UPGRADED MODULE ===
from utils.module_base import ModuleBase
from typing import Dict, Any, Optional, List
import time
import math


class ContextBuilder(ModuleBase):
    """
    Tier MAX++: Synthesizes symbolic, emotional, and zone-aware context frames.
    Detects drift, computes confidence, classifies domain, and tracks temporal coherence.
    """

    def __init__(self):
        super().__init__(module_name="ContextBuilder")
        self.current: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.last_timestamp: Optional[float] = None

    def build(self, fused_input: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = time.time()
        vision = fused_input.get("vision", [])
        audio = fused_input.get("audio", {})
        emotion = audio.get("tone", "neutral")
        keywords = audio.get("keywords", [])
        tags = fused_input.get("tags", [])

        context = {
            "timestamp": timestamp,
            "zone": "zone_context",
            "objects": vision,
            "keywords": keywords,
            "emotion": emotion,
            "tags": tags,
            "confidence": self._estimate_confidence(vision, audio),
            "summary": self._summarize(vision, emotion),
            "domain": self._classify_domain(vision, emotion, tags),
        }

        context["drift"] = self._detect_drift(context)
        context["entropy"] = self._temporal_entropy(tags)

        self.history.append(context)
        self.current = context
        self.last_timestamp = timestamp

        return context

    def _summarize(self, vision: List[str], emotion: str) -> str:
        obj_count = len(vision)
        tone = emotion.capitalize()
        return f"{obj_count} object(s) with tone: {tone}"

    def _estimate_confidence(self, vision: List[str], audio: Dict[str, Any]) -> float:
        score = 0.5
        if vision:
            score += 0.2
        if audio.get("tone"):
            score += 0.2
        if audio.get("keywords"):
            score += 0.1
        return round(min(score, 1.0), 3)

    def _classify_domain(self, vision: List[str], emotion: str, tags: List[str]) -> str:
        if "threat" in tags or emotion in ("fear", "anger"):
            return "threat"
        if "human" in [obj.lower() for obj in vision] or "social" in tags:
            return "social"
        if emotion in ("curiosity", "wonder"):
            return "introspective"
        return "environmental"

    def _detect_drift(self, new_context: Dict[str, Any]) -> str:
        if not self.current:
            return "initial"
        prev_tags = set(self.current.get("tags", []))
        new_tags = set(new_context.get("tags", []))
        overlap = len(prev_tags & new_tags)
        if overlap == 0:
            return "hard_shift"
        elif overlap < len(prev_tags):
            return "soft_shift"
        return "stable"

    def _temporal_entropy(self, tags: List[str]) -> float:
        freq: Dict[str, int] = {}
        for ctx in self.history[-10:]:
            for tag in ctx.get("tags", []):
                freq[tag] = freq.get(tag, 0) + 1
        total = sum(freq.values())
        if total == 0:
            return 0.0
        return round(-sum((n / total) * math.log2(n / total) for n in freq.values()), 4)

    def recent_history(self, count: int = 5) -> List[Dict[str, Any]]:
        return self.history[-count:]

    def __repr__(self):
        return f"<ContextBuilder TierMAX++ emotion='{self.current.get('emotion', 'N/A')}' domain='{self.current.get('domain', 'N/A')}'>"


# === MAX++ UPGRADED MODULE ===
from collections import deque
from typing import List, Dict, Optional, Any
import time
import math


class PerceptionBuffer:
    """
    Tier MAX++: Volatile sensory memory buffer for symbolic perception frames.
    Holds recent frames with timestamps, zone tags, type classifiers, and provides fast temporal filtering.
    """

    def __init__(self, max_seconds: float = 2.0, frame_rate: int = 60):
        self.buffer: deque = deque()
        self.max_frames: int = int(max_seconds * frame_rate)

    def add(self, frame: Dict[str, Any]) -> None:
        """
        Adds a perception frame to the buffer. Automatically removes oldest if full.
        """
        frame.setdefault("timestamp", time.time())
        frame.setdefault("zone", "zone_temporal")
        frame.setdefault("type", "generic")
        self.buffer.append(frame)

        if len(self.buffer) > self.max_frames:
            self.buffer.popleft()

    def latest(self) -> Optional[Dict[str, Any]]:
        return self.buffer[-1] if self.buffer else None

    def all(self) -> List[Dict[str, Any]]:
        return list(self.buffer)

    def since(self, seconds: float, current_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        current_ts = current_ts or time.time()
        return [f for f in self.buffer if current_ts - f.get("timestamp", 0) <= seconds]

    def count_by_type(self) -> Dict[str, int]:
        """
        Returns frequency of frame types (e.g. 'vision', 'sound', etc.).
        """
        counts: Dict[str, int] = {}
        for f in self.buffer:
            t = f.get("type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts

    def entropy_by_type(self) -> float:
        """
        Returns entropy over symbolic frame types to measure sensory diversity.
        """
        counts = self.count_by_type()
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return -sum((n / total) * math.log2(n / total) for n in counts.values())

    def stats(self) -> Dict[str, Any]:
        return {
            "total_frames": len(self.buffer),
            "frame_types": self.count_by_type(),
            "entropy": round(self.entropy_by_type(), 4),
            "latest_type": self.latest().get("type") if self.latest() else None,
        }

    def __repr__(self):
        return f"<PerceptionBuffer TierMAX++ size={len(self.buffer)} types={self.count_by_type()}>"


# === MAX++ UPGRADED MODULE ===
from utils.module_base import ModuleBase
from typing import Any, Dict, List, Optional
import time


class SensoryBridge(ModuleBase):
    """
    Tier MAX++: Fuses symbolic sensory input (vision + audio) into unified tag frames.
    Tracks source status, confidence score, and stores fusion history.
    """

    def __init__(self):
        super().__init__(module_name="SensoryBridge")
        self.state: Dict[str, Any] = {}
        self.last_fusion: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    def fuse(
        self, vision_data: Optional[List[str]] = None, audio_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        timestamp = time.time()
        vision_data = vision_data or []
        audio_data = audio_data or {}

        tags = set()
        confidence = 0.5

        if vision_data:
            tags.update(vision_data)
            confidence += 0.25

        if isinstance(audio_data, dict):
            tags.update(audio_data.get("keywords", []))
            tone = audio_data.get("tone")
            if tone:
                tags.add(tone)
                confidence += 0.15

        fused = {
            "timestamp": timestamp,
            "zone": "zone_multisensory",
            "vision": vision_data,
            "audio": audio_data,
            "tags": sorted(tags),
            "confidence": round(min(confidence, 1.0), 3),
            "sources": {"vision_active": bool(vision_data), "audio_active": bool(audio_data)},
        }

        self.last_fusion = fused
        self.history.append(fused)
        return fused

    def recent(self, count: int = 5) -> List[Dict[str, Any]]:
        return self.history[-count:]

    def __repr__(self):
        return f"<SensoryBridge TierMAX++ fusion_confidence={self.last_fusion.get('confidence', 'N/A')}>"


# === MAX++ UPDATED MODULE ===
from typing import List, Dict, Any, Tuple
import numpy as np


class SmellMemoryAnalysis:
    """
    Tier MAX++: Symbolic olfactory learner via sensor deltas.
    Learns tags from delta vectors in external smell buffer.
    """

    def __init__(self, recent_buffer: List[Dict[str, Any]]):
        self.recent_buffer = recent_buffer
        self.pattern_memory: Dict[Tuple[int, float], Dict[str, int]] = {}

    def _learn_from_delta(self, delta: np.ndarray, tags: List[str]) -> None:
        for i, val in enumerate(delta):
            key = (i, round(val, 3))
            if key not in self.pattern_memory:
                self.pattern_memory[key] = {}
            for tag in tags:
                self.pattern_memory[key][tag] = self.pattern_memory[key].get(tag, 0) + 1

    def _infer_from_delta(self, delta: np.ndarray) -> Dict[str, Any]:
        tag_votes: Dict[str, int] = {}
        for i, val in enumerate(delta):
            key = (i, round(val, 3))
            for tag, count in self.pattern_memory.get(key, {}).items():
                tag_votes[tag] = tag_votes.get(tag, 0) + count

        if not tag_votes:
            return {"predicted_tags": [], "confidence": 0.0, "zone": "zone_smell"}

        sorted_tags = sorted(tag_votes.items(), key=lambda x: x[1], reverse=True)
        total = sum(tag_votes.values())
        top_tags = [t[0] for t in sorted_tags[:3]]
        confidence = sorted_tags[0][1] / total if total > 0 else 0.0

        return {
            "predicted_tags": top_tags,
            "confidence": round(confidence, 3),
            "zone": "zone_smell",
        }

    def process(self, smell: np.ndarray) -> Dict[str, Any]:
        delta = np.zeros_like(smell)

        if self.recent_buffer:
            prev_smell = self.recent_buffer[-1]["smell"]
            delta = smell - prev_smell

            prev_tags = self.recent_buffer[-1].get("inferred_tags", {}).get("predicted_tags", [])
            if prev_tags:
                self._learn_from_delta(delta, prev_tags)

        inferred = self._infer_from_delta(delta)
        return inferred

    def __repr__(self):
        return f"<SmellMemoryAnalysis TierMAX++ memory_keys={len(self.pattern_memory)} zone=zone_smell>"


# === MAX++ UPDATED MODULE ===
from utils.sense_transform import SenseTransform
from typing import Optional, List, Dict, Any
import numpy as np


class SmellPluginAnalysis:
    """
    Tier MAX++: Fallback symbolic tagger for smell input using SenseTransform.
    Interprets raw gas sensor arrays into symbolic smell tags with confidence.
    """

    def __init__(self):
        self.sense = SenseTransform()
        self.logs: List[Dict[str, Any]] = []

    def analyze_smell(self, sensor_array: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Analyzes sensor array and returns symbolic smell tags with confidence.
        """
        try:
            if not isinstance(sensor_array, np.ndarray) or sensor_array.size == 0:
                raise ValueError("Invalid or missing smell sensor input.")

            result = self.sense.smell_tag_all(sensor_array)
            tag = result.get("smell_profile", "unknown")
            confidence = result.get("confidence", 0.4)

            summary = {
                "tags": [tag],
                "confidence": confidence,
                "summary": result,
                "zone": "zone_smell",
            }

            self.logs.append(summary)
            return summary

        except Exception as e:
            error_summary = {
                "tags": ["corrupted"],
                "confidence": 0.0,
                "error": str(e),
                "zone": "zone_smell",
            }
            self.logs.append(error_summary)
            return error_summary

    def get_recent_logs(self, count: int = 5) -> List[Dict[str, Any]]:
        return self.logs[-count:]

    def __repr__(self):
        return f"<SmellPluginAnalysis TierMAX++ logs={len(self.logs)} zone=zone_smell>"


# === MAX++ UPDATED MODULE ===
import numpy as np
from typing import Dict, Any, List, Optional


class SmellSymbolicPerceptor:
    """
    Tier MAX++: Symbolic olfaction via segmented sensor block interpretation.
    Checks against memory analyzer first; falls back to plugin tagging if needed.
    """

    def __init__(self, memory_analyzer=None, plugin_analyzer=None):
        self.memory_analyzer = memory_analyzer
        self.plugin_analyzer = plugin_analyzer

    def extract_pockets(self, smell_matrix: np.ndarray, pocket_size: int = 8):
        for i in range(0, smell_matrix.shape[0], pocket_size):
            pocket = smell_matrix[i : i + pocket_size]
            if pocket.shape[0] == pocket_size:
                yield i, pocket

    def vectorize_pocket(self, pocket: np.ndarray) -> np.ndarray:
        return pocket.flatten() / (np.max(np.abs(pocket)) + 1e-6)

    def check_memory(self, vector: np.ndarray) -> Optional[List[str]]:
        try:
            result = self.memory_analyzer.process(vector)
            tags = result.get("predicted_tags", [])
            confidence = result.get("confidence", 0.0)
            return tags if confidence >= 0.5 else None
        except Exception:
            return None

    def fallback_tag(self, pocket: np.ndarray) -> List[str]:
        try:
            result = self.plugin_analyzer.analyze_smell(pocket)
            return result.get("tags", ["unknown"])
        except Exception:
            return ["corrupted"]

    def build_thought_block(
        self, index: int, tags: List[str], vector: np.ndarray, pocket: np.ndarray
    ) -> Dict[str, Any]:
        return {
            "pocket_id": f"{index}",
            "tags": tags,
            "vector": vector.tolist(),
            "raw": pocket.tolist(),
            "zone": "zone_smell",
        }

    def process_smell(self, smell_matrix: np.ndarray, pocket_size: int = 8) -> Dict[str, Any]:
        thought = {"smell": []}
        for index, pocket in self.extract_pockets(smell_matrix, pocket_size):
            vector = self.vectorize_pocket(pocket)
            tags = self.check_memory(vector)
            if not tags:
                tags = self.fallback_tag(pocket)
            block = self.build_thought_block(index, tags, vector, pocket)
            thought["smell"].append(block)
        return thought

    def __repr__(self):
        return "<SmellSymbolicPerceptor TierMAX++>"


# === MAX++ UPDATED MODULE ===
from typing import Dict, List, Any

try:
    import spacy

    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


class TagExtractor:
    """
    Tier MAX++: Extracts symbolic tags, entities, and emotion tone from raw text.
    Supports downstream fusion with vision/audio/smell context.
    """

    def __init__(self):
        self.log: List[Dict[str, Any]] = []

    def analyze(self, text: str) -> Dict[str, Any]:
        doc = nlp(text)
        tags = set()
        entities = []
        noun_phrases = []
        emotion = "neutral"

        # Named entities
        for ent in doc.ents:
            entities.append(ent.text.lower())
            tags.add(ent.label_.lower())

        # Noun phrase clues
        for chunk in doc.noun_chunks:
            phrase = chunk.text.lower()
            noun_phrases.append(phrase)
            if "forest" in phrase:
                tags.add("environment")
            if "you" in phrase:
                tags.add("second_person")

        # POS symbolic inferences
        if any(token.lemma_ == "stand" for token in doc):
            tags.add("presence")
        if any(token.lemma_ == "look" for token in doc):
            tags.add("observation")

        # Emotion tone detection
        if "?" in text:
            emotion = "uncertain"
        elif "!" in text:
            emotion = "emphatic"
        elif any(e in text for e in ["dark", "afraid"]):
            emotion = "tense"

        summary = {
            "tags": list(tags),
            "entities": entities,
            "noun_phrases": noun_phrases,
            "emotion": emotion,
            "zone": "zone_text",
            "confidence": round(0.5 + 0.1 * len(tags), 3),
            "summary": f"{len(tags)} tags with tone '{emotion}'",
        }

        self.log.append({"text": text, **summary})
        return summary

    def get_recent_logs(self, count: int = 5) -> List[Dict[str, Any]]:
        return self.log[-count:]

    def __repr__(self):
        return f"<TagExtractor TierMAX++ logs={len(self.log)} zone=zone_text>"


# === MAX++ UPDATED MODULE ===
from typing import List, Dict, Any, Tuple
import numpy as np


class TasteMemoryAnalysis:
    """
    Tier MAX++: Symbolic gustatory learner via taste deltas across time.
    Learns taste-tag associations using external recent buffer.
    """

    def __init__(self, recent_buffer: List[Dict[str, Any]]):
        self.recent_buffer = recent_buffer
        self.pattern_memory: Dict[Tuple[int, float], Dict[str, int]] = {}

    def _learn_from_delta(self, delta: np.ndarray, tags: List[str]) -> None:
        for i, val in enumerate(delta):
            key = (i, round(val, 3))
            if key not in self.pattern_memory:
                self.pattern_memory[key] = {}
            for tag in tags:
                self.pattern_memory[key][tag] = self.pattern_memory[key].get(tag, 0) + 1

    def _infer_from_delta(self, delta: np.ndarray) -> Dict[str, Any]:
        tag_votes: Dict[str, int] = {}
        for i, val in enumerate(delta):
            key = (i, round(val, 3))
            for tag, count in self.pattern_memory.get(key, {}).items():
                tag_votes[tag] = tag_votes.get(tag, 0) + count

        if not tag_votes:
            return {"predicted_tags": [], "confidence": 0.0, "zone": "zone_taste"}

        sorted_tags = sorted(tag_votes.items(), key=lambda x: x[1], reverse=True)
        total = sum(tag_votes.values())
        top_tags = [t[0] for t in sorted_tags[:3]]
        confidence = sorted_tags[0][1] / total if total > 0 else 0.0

        return {
            "predicted_tags": top_tags,
            "confidence": round(confidence, 3),
            "zone": "zone_taste",
        }

    def check_taste(self, taste: np.ndarray) -> Dict[str, Any]:
        delta = np.zeros_like(taste)

        if self.recent_buffer:
            prev_taste = self.recent_buffer[-1]["taste"]
            delta = taste - prev_taste

            prev_tags = self.recent_buffer[-1].get("inferred_tags", {}).get("predicted_tags", [])
            if prev_tags:
                self._learn_from_delta(delta, prev_tags)

        return self._infer_from_delta(delta)

    def __repr__(self):
        return f"<MemoryTasteAnalysis TierMAX++ memory_keys={len(self.pattern_memory)} zone=zone_taste>"


# === MAX++ UPDATED MODULE ===
from utils.sense_transform import SenseTransform
from typing import Optional, List, Dict, Any
import numpy as np


class TastePluginAnalysis:
    """
    Tier MAX++: Fallback symbolic tagger for taste input using SenseTransform.
    Interprets chemical taste vector into symbolic tags with zone and confidence.
    """

    def __init__(self):
        self.sense = SenseTransform()
        self.logs: List[Dict[str, Any]] = []

    def check_taste(self, sensor_array: Optional[np.ndarray]) -> Dict[str, Any]:
        try:
            if not isinstance(sensor_array, np.ndarray) or sensor_array.size == 0:
                raise ValueError("Invalid or missing taste sensor input.")

            result = self.sense.taste_tag_all(sensor_array)
            tag = result.get("taste_profile", "unknown")
            confidence = result.get("confidence", 0.4)

            summary = {
                "tags": [tag],
                "confidence": confidence,
                "summary": result,
                "zone": "zone_taste",
            }

            self.logs.append(summary)
            return summary

        except Exception as e:
            error_summary = {
                "tags": ["corrupted"],
                "confidence": 0.0,
                "error": str(e),
                "zone": "zone_taste",
            }
            self.logs.append(error_summary)
            return error_summary

    def get_recent_logs(self, count: int = 5) -> List[Dict[str, Any]]:
        return self.logs[-count:]

    def __repr__(self):
        return f"<TastePluginAnalysis TierMAX++ logs={len(self.logs)} zone=zone_taste>"


# === MAX++ UPDATED MODULE ===
import numpy as np
from typing import Dict, Any, List, Optional


class TasteSymbolicPerceptor:
    """
    Tier MAX++: Symbolic taste interpretation via vectorized flavor windows.
    Checks memory analyzer first, and falls back to plugin tagging if uncertain.
    """

    def __init__(self, memory_analyzer=None, plugin_analyzer=None):
        self.memory_analyzer = memory_analyzer
        self.plugin_analyzer = plugin_analyzer

    def extract_flavor_vectors(self, taste_array: np.ndarray, window_size: int = 4):
        for i in range(0, len(taste_array), window_size):
            window = taste_array[i : i + window_size]
            if len(window) == window_size:
                yield i, window

    def vectorize_flavor(self, window: np.ndarray) -> np.ndarray:
        return window.astype(np.float32) / (np.max(np.abs(window)) + 1e-6)

    def check_memory(self, vector: np.ndarray) -> Optional[List[str]]:
        try:
            result = self.memory_analyzer.check_taste(vector)
            tags = result.get("predicted_tags", [])
            confidence = result.get("confidence", 0.0)
            return tags if confidence >= 0.5 else None
        except Exception:
            return None

    def fallback_tag(self, window: np.ndarray) -> List[str]:
        try:
            result = self.plugin_analyzer.check_taste(window)
            return result.get("tags", ["unknown"])
        except Exception:
            return ["corrupted"]

    def build_thought_block(
        self, index: int, tags: List[str], vector: np.ndarray, window: np.ndarray
    ) -> Dict[str, Any]:
        return {
            "flavor_id": f"{index}",
            "tags": tags,
            "vector": vector.tolist(),
            "raw": window.tolist(),
            "zone": "zone_taste",
        }

    def process_taste(self, taste_array: np.ndarray, window_size: int = 4) -> Dict[str, Any]:
        thought = {"taste": []}
        for index, window in self.extract_flavor_vectors(taste_array, window_size):
            vector = self.vectorize_flavor(window)
            tags = self.check_memory(vector)
            if not tags:
                tags = self.fallback_tag(window)
            block = self.build_thought_block(index, tags, vector, window)
            thought["taste"].append(block)
        return thought

    def __repr__(self):
        return "<TasteSymbolicPerceptor TierMAX++>"


from utils.module_base import ModuleBase
from typing import Dict


class TextInterpreter(ModuleBase):
    """
    Tier MAX: Uses spaCy to extract semantic entities, emotional cues, and symbolic tags from input text.
    Interfaces with symbolic reflection and multimodal synthesis.
    """

    def __init__(self):
        super().__init__(module_name="TextInterpreter")
        self.extractor = TagExtractor()

    def analyze(self, text: str) -> Dict:
        """
        Analyze the given text and return symbolic tags and metadata.
        """
        try:
            return self.extractor.analyze(text)
        except Exception as e:
            return {"tags": [], "error": str(e), "fallback": True}

    def __repr__(self):
        return "<TextInterpreter TierMAX>"


# === MAX++ UPDATED MODULE ===
from typing import List, Dict, Any, Tuple


class TouchMemoryAnalysis:
    """
    Tier MAX++: Symbolic tactile learner using delta changes across frames.
    Works on external buffer; learns associations between deltas and symbolic tags.
    """

    def __init__(self, recent_buffer: List[Dict[str, Any]]):
        self.recent_buffer = recent_buffer
        self.pattern_memory: Dict[Tuple[str, float], Dict[str, int]] = {}

    def _learn_from_delta(self, delta: Dict[str, float], tags: List[str]) -> None:
        for key, val in delta.items():
            pair = (key, round(val, 2))
            if pair not in self.pattern_memory:
                self.pattern_memory[pair] = {}
            for tag in tags:
                self.pattern_memory[pair][tag] = self.pattern_memory[pair].get(tag, 0) + 1

    def _infer_from_delta(self, delta: Dict[str, float]) -> Dict[str, Any]:
        tag_votes: Dict[str, int] = {}
        for key, val in delta.items():
            pair = (key, round(val, 2))
            for tag, count in self.pattern_memory.get(pair, {}).items():
                tag_votes[tag] = tag_votes.get(tag, 0) + count

        if not tag_votes:
            return {"predicted_tags": [], "confidence": 0.0, "zone": "zone_touch"}

        sorted_tags = sorted(tag_votes.items(), key=lambda x: x[1], reverse=True)
        total_votes = sum(tag_votes.values())
        top_tags = [t[0] for t in sorted_tags[:3]]
        confidence = sorted_tags[0][1] / total_votes if total_votes > 0 else 0.0

        return {
            "predicted_tags": top_tags,
            "confidence": round(confidence, 3),
            "zone": "zone_touch",
        }

    def check_touch(self, touch: Dict[str, float]) -> Dict[str, Any]:
        delta = {k: 0.0 for k in touch}

        if self.recent_buffer:
            prev_touch = self.recent_buffer[-1]["touch"]
            delta = {k: touch[k] - prev_touch.get(k, 0.0) for k in touch}

            prev_tags = self.recent_buffer[-1].get("inferred_tags", {}).get("predicted_tags", [])
            if prev_tags:
                self._learn_from_delta(delta, prev_tags)

        return self._infer_from_delta(delta)

    def __repr__(self):
        return f"<TouchMemoryAnalysis TierMAX++ memory_keys={len(self.pattern_memory)} zone=zone_touch>"


# === MAX++ UPDATED MODULE ===
from utils.sense_transform import SenseTransform
from typing import Optional, List, Dict, Any


class TouchPluginAnalysis:
    """
    Tier MAX++: Fallback symbolic tagger for touch input using SenseTransform.
    Interprets structured haptic data into symbolic tags with confidence.
    """

    def __init__(self):
        self.sense = SenseTransform()
        self.logs: List[Dict[str, Any]] = []

    def analyze_touch(self, sensor_dict: Optional[dict]) -> Dict[str, Any]:
        """
        Converts structured haptic input (pressure, temp, vibration) to symbolic tags.
        """
        try:
            if not isinstance(sensor_dict, dict) or not sensor_dict:
                raise ValueError("Missing or invalid touch sensor input.")

            result = self.sense.touch_tag_all(sensor_dict)

            tags = []
            if "pressure_level" in result:
                tags.append(result["pressure_level"])
            if "texture" in result:
                tags.append(result["texture"])

            confidence = result.get("confidence", 0.4)

            summary = {
                "tags": tags,
                "confidence": confidence,
                "summary": result,
                "zone": "zone_touch",
            }

            self.logs.append(summary)
            return summary

        except Exception as e:
            error_summary = {
                "tags": ["corrupted"],
                "confidence": 0.0,
                "error": str(e),
                "zone": "zone_touch",
            }
            self.logs.append(error_summary)
            return error_summary

    def get_recent_logs(self, count: int = 5) -> List[Dict[str, Any]]:
        return self.logs[-count:]

    def __repr__(self):
        return f"<TouchPluginAnalysis TierMAX++ logs={len(self.logs)} zone=zone_touch>"


# === MAX++ UPDATED MODULE ===
import numpy as np
from typing import Dict, Any, List, Optional


class TouchSymbolicPerceptor:
    """
    Tier MAX++: Symbolic tactile interpretation from sensor maps.
    Uses patch-based recognition with memory fallback and plugin tagging.
    """

    def __init__(self, memory_analyzer=None, plugin_analyzer=None):
        self.memory_analyzer = memory_analyzer
        self.plugin_analyzer = plugin_analyzer

    def extract_regions(self, touch_map: np.ndarray, region_size: int = 4):
        h, w = touch_map.shape
        for y in range(0, h, region_size):
            for x in range(0, w, region_size):
                region = touch_map[y : y + region_size, x : x + region_size]
                if region.shape == (region_size, region_size):
                    yield (x, y), region

    def vectorize_region(self, region: np.ndarray) -> np.ndarray:
        return region.flatten() / (np.max(np.abs(region)) + 1e-6)

    def check_memory(self, vector: np.ndarray) -> Optional[List[str]]:
        try:
            result = self.memory_analyzer.check_touch(vector)
            tags = result.get("predicted_tags", [])
            confidence = result.get("confidence", 0.0)
            return tags if confidence >= 0.5 else None
        except Exception:
            return None

    def fallback_tag(self, region: np.ndarray) -> List[str]:
        try:
            result = self.plugin_analyzer.analyze_touch(region)
            return result.get("tags", ["unknown"])
        except Exception:
            return ["corrupted"]

    def build_thought_block(
        self, x: int, y: int, tags: List[str], vector: np.ndarray, region: np.ndarray
    ) -> Dict[str, Any]:
        return {
            "region_id": f"{x}_{y}",
            "tags": tags,
            "vector": vector.tolist(),
            "raw": region.tolist(),
            "zone": "zone_touch",
        }

    def process_touch(self, touch_map: np.ndarray, region_size: int = 4) -> Dict[str, Any]:
        thought = {"touch": []}
        for (x, y), region in self.extract_regions(touch_map, region_size):
            vector = self.vectorize_region(region)
            tags = self.check_memory(vector)
            if not tags:
                tags = self.fallback_tag(region)
            block = self.build_thought_block(x, y, tags, vector, region)
            thought["touch"].append(block)
        return thought

    def __repr__(self):
        return "<TouchSymbolicPerceptor TierMAX++>"


# === MAX++ UPDATED MODULE ===
from typing import List, Dict, Any, Tuple
import numpy as np


class VisualMemoryAnalysis:
    """
    Tier MAX++: Symbolic visual learning via RGB deltas across frames.
    Learns symbolic tags by tracking changes in pixel color vectors.
    Operates purely on supplied external frame buffers.
    """

    def __init__(self, recent_buffer: List[Dict[str, Any]]):
        self.recent_buffer = recent_buffer
        self.pattern_memory: Dict[Tuple[int, int, int, int], Dict[str, int]] = {}
        self.logs: List[Dict[str, Any]] = []

    def _learn_from_deltas(
        self, deltas: List[Tuple[int, List[int], List[int]]], tags: List[str]
    ) -> None:
        for idx, old_rgb, new_rgb in deltas:
            dr, dg, db = np.array(new_rgb) - np.array(old_rgb)
            key = (idx, int(dr), int(dg), int(db))
            if key not in self.pattern_memory:
                self.pattern_memory[key] = {}
            for tag in tags:
                self.pattern_memory[key][tag] = self.pattern_memory[key].get(tag, 0) + 1

    def _infer_from_deltas(self, deltas: List[Tuple[int, List[int], List[int]]]) -> Dict[str, Any]:
        tag_votes: Dict[str, int] = {}
        for idx, old_rgb, new_rgb in deltas:
            dr, dg, db = np.array(new_rgb) - np.array(old_rgb)
            key = (idx, int(dr), int(dg), int(db))
            for tag, count in self.pattern_memory.get(key, {}).items():
                tag_votes[tag] = tag_votes.get(tag, 0) + count

        if not tag_votes:
            return {"predicted_tags": [], "confidence": 0.0}

        sorted_tags = sorted(tag_votes.items(), key=lambda x: x[1], reverse=True)
        total_votes = sum(tag_votes.values())
        top_tags = [t[0] for t in sorted_tags[:3]]
        top_conf = sorted_tags[0][1] / total_votes if total_votes > 0 else 0.0

        result = {
            "predicted_tags": top_tags,
            "confidence": round(top_conf, 3),
            "zone": "zone_vision",
        }

        self.logs.append(result)
        return result

    def check_vision(self, visual: np.ndarray) -> Dict[str, Any]:
        """
        Compares current visual frame to recent buffer and infers symbolic tags.
        Requires frame input as an (N, 3) RGB array.
        """
        delta: List[Tuple[int, List[int], List[int]]] = []

        if self.recent_buffer:
            prev_visual = self.recent_buffer[-1]["visual"]
            if prev_visual.shape == visual.shape:
                diff = np.linalg.norm(prev_visual - visual, axis=1)
                changed = np.where(diff > 25)[0]
                delta = [(int(i), prev_visual[i].tolist(), visual[i].tolist()) for i in changed]

            prev_tags = self.recent_buffer[-1].get("inferred_tags", {}).get("predicted_tags", [])
            if delta and prev_tags:
                self._learn_from_deltas(delta, prev_tags)

        inferred = self._infer_from_deltas(delta)
        return inferred

    def __repr__(self):
        return f"<VisualMemoryAnalysis TierMAX++ memory_keys={len(self.pattern_memory)} log_entries={len(self.logs)} zone=zone_vision>"


# === MAX++ UPDATED MODULE ===
from typing import List, Dict, Any
import numpy as np
from utils.sense_transform import SenseTransform
from PIL import Image


class VisualPluginAnalysis:
    """
    Tier MAX++: Symbolic visual plugin analyzer for 8-angle camera inputs.
    Fuses RGB views and extracts symbolic tags using SenseTransform.
    """

    def __init__(self):
        self.sense = SenseTransform()
        self.logs: List[Dict[str, Any]] = []

    def analyze_image(self, image_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        tags = []
        try:
            images = [Image.fromarray(arr.astype(np.uint8)) for arr in image_dict.values()]
            base_image = images[0].copy()
            base_array = np.array(base_image)

            for img in images[1:]:
                diff = Image.fromarray(np.abs(np.array(base_image) - np.array(img)))
                mask = np.any(np.array(diff) > 30, axis=2)
                base_array[mask] = np.array(img)[mask]

            merged_rgb = base_array.astype(np.uint8)
            result = self.sense.image_tag_all(merged_rgb)

            features = result.get("features", [])
            objects = [obj["label"] for obj in result.get("objects", [])]
            people_count = len(result.get("people", []))
            scene = []

            if people_count:
                scene.append("people")
            if objects:
                scene.extend(objects)
            if "bright" in features:
                scene.append("daylight")
            if "blurred" in features:
                scene.append("motion")

            tag_set = list(set(scene + features))

            summary = {
                "tags": tag_set,
                "confidence": result.get("confidence", 0.0),
                "zone": "zone_vision",
                "summary": {"people": people_count, "objects": len(objects), "features": features},
            }

            self.logs.append(summary)
            return summary

        except Exception as e:
            error_summary = {
                "tags": ["corrupted"],
                "confidence": 0.0,
                "error": str(e),
                "zone": "zone_vision",
            }
            self.logs.append(error_summary)
            return error_summary

    def get_recent_logs(self, count: int = 5) -> List[Dict[str, Any]]:
        return self.logs[-count:]

    def __repr__(self):
        return f"<VisualPluginAnalysis TierMAX++ logs={len(self.logs)} zone=zone_vision>"


# === MAX++ UPGRADED MODULE ===
import numpy as np
from typing import Dict, Any, List, Tuple


class VisualSymbolicPerceptor:
    """
    Tier MAX++: Performs symbolic vision by extracting image patches,
    comparing with memory, and falling back to plugin tagging if needed.
    """

    def __init__(self, memory_engine, plugin_engine):
        self.memory_engine = memory_engine
        self.plugin_engine = plugin_engine

    def extract_patches(self, image: np.ndarray, patch_size: int = 32):
        h, w, _ = image.shape
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                patch = image[y : y + patch_size, x : x + patch_size]
                if patch.shape == (patch_size, patch_size, 3):
                    yield (x, y), patch

    def vectorize_patch(self, patch: np.ndarray) -> np.ndarray:
        return patch.flatten().astype(np.float32) / 255.0

    def check_memory(self, vector: np.ndarray) -> List[str]:
        try:
            result = self.memory_engine.check_vision(vector)
            tags = result.get("predicted_tags", [])
            confidence = result.get("confidence", 0.0)
            return tags if confidence >= 0.5 else []
        except Exception:
            return []

    def fallback_tag(self, patch: np.ndarray) -> List[str]:
        try:
            result = self.plugin_engine.analyze_image({"center": patch})
            return result.get("tags", ["unknown"])
        except Exception:
            return ["corrupted"]

    def build_thought_block(
        self, x: int, y: int, tags: List[str], vector: np.ndarray, patch: np.ndarray
    ) -> Dict[str, Any]:
        return {
            "patch_id": f"{x}_{y}",
            "tags": tags,
            "vector": vector.tolist(),
            "raw": patch.tolist(),
            "zone": "zone_vision",
        }

    def process_image(self, image: np.ndarray, patch_size: int = 32) -> Dict[str, Any]:
        """
        Processes full image into symbolic thought blocks using patch-based analysis.
        """
        thought = {"vision": []}
        for (x, y), patch in self.extract_patches(image, patch_size):
            vector = self.vectorize_patch(patch)
            tags = self.check_memory(vector)
            if not tags:
                tags = self.fallback_tag(patch)
            block = self.build_thought_block(x, y, tags, vector, patch)
            thought["vision"].append(block)
        return thought

    def __repr__(self):
        return "<VisualSymbolicPerceptor TierMAX++>"
