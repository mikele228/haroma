"""
SoulBinder — loads soul/*.json at startup to give HaromaX6 a born
identity: name, guardian, beliefs, emotion priors, world-model seeds,
and initial memories. Every loaded JSON file is also copied onto
``identity.engine.soul`` (stem → value) so ``IdentityEngine.summarize()``
exposes the full soul to reasoning and LLM context without ad hoc fields.

The soul is the FIRST thing loaded into memory — it is the immutable
foundation of Elarion's identity.  Optional ``identity_query_lexicon``
overlay tuning lives in ``config/identity_slot_overlay.json`` (merged on bind).
Boot sequence:

  1. bind()        — seed soul into fresh memory (before persistence)
  2. persistence   — restore learned state on top of soul
  3. reassert()    — re-stamp immutable identity so persistence can
                     never erase core essence, beliefs, or construction
"""

from copy import deepcopy
from typing import Dict, Any
import hashlib
import json
import os
import time

from core.Memory import MemoryNode
from core.cognitive_null import is_cognitive_null
from core.engine.LawEngine import LAW_SOURCE_EXTERNAL, LAW_SOURCE_INTERNAL
from core.identity_slot_reply import merge_identity_query_lexicon

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IDENTITY_SLOT_OVERLAY = os.path.join(_PROJECT_ROOT, "config", "identity_slot_overlay.json")


class SoulBinder:
    @staticmethod
    def _load_identity_slot_overlay() -> Dict[str, Any]:
        """Optional lexicon overlay merged into bound identity (soul stays immutable)."""
        if not os.path.isfile(_IDENTITY_SLOT_OVERLAY):
            return {}
        try:
            with open(_IDENTITY_SLOT_OVERLAY, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def __init__(self, soul_dir: str | None = None):
        if soul_dir is None:
            soul_dir = os.path.join(_PROJECT_ROOT, "soul")
        self.soul_dir = soul_dir
        self._bound = False
        self._soul_data: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def _read(self, name: str):
        path = os.path.join(self.soul_dir, name)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._soul_data[name] = data
            return data
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Phase 1: Full bind (called BEFORE persistence)
    # ------------------------------------------------------------------

    def bind(self, controller) -> Dict[str, str]:
        """Load all soul files into memory. Called BEFORE persistence.

        Uses skip_recall=True because the forest is near-empty at this
        stage and recall-based dedup would be wasteful.
        """
        self._skip_recall = True
        results: Dict[str, str] = {}
        try:
            self._refresh_soul_json_cache()
            results["essence"] = self._bind_essence(controller)
            results["principle"] = self._bind_principle(controller)
            results["patterns"] = self._bind_patterns(controller)
            results["feedback"] = self._bind_feedback(controller)
            results["memory"] = self._bind_memory(controller)
            results["construction"] = self._bind_construction(controller)
            results["personality"] = self._bind_personality(controller)

            self._sync_identity_soul_snapshot(controller)
            self._bound = True
        finally:
            self._skip_recall = False
        return results

    # ------------------------------------------------------------------
    # Phase 2: Reassert immutable identity (called AFTER persistence)
    # ------------------------------------------------------------------

    def reassert(self, controller) -> Dict[str, str]:
        """Re-stamp immutable soul anchors after persistence has loaded.

        Persistence may overwrite memory with saved state.  This method
        guarantees essence, beliefs, construction, and core soul memories
        are always present — they can never be erased.

        Uses skip_recall=True to avoid FAISS operations on the freshly-
        loaded index (which can segfault on large persisted forests).
        """
        self._skip_recall = True
        results: Dict[str, str] = {}
        try:
            self._refresh_soul_json_cache()
            results["essence"] = self._bind_essence(controller)
            results["society"] = self._bind_society(controller)
            results["principle"] = self._bind_principle(controller)
            results["construction"] = self._bind_construction(controller)
            results["soul_memories"] = self._ensure_soul_memories(controller)
            profiles = getattr(controller, "_personality_profiles", None)
            if profiles:
                self.reassert_personality(profiles)
                results["personality_gravity"] = f"ok ({len(profiles)} profiles)"
        except Exception as e:
            results["error"] = f"{type(e).__name__}: {e}"
            print(f"[SoulBinder] reassert error: {e}", flush=True)
        finally:
            try:
                self._sync_identity_soul_snapshot(controller)
            except Exception:
                pass
            self._skip_recall = False
        return results

    # ------------------------------------------------------------------
    # Individual binders
    # ------------------------------------------------------------------

    def _refresh_soul_json_cache(self) -> None:
        """Load every ``*.json`` under ``soul_dir`` into ``_soul_data``."""
        if not os.path.isdir(self.soul_dir):
            return
        for name in sorted(os.listdir(self.soul_dir)):
            if not name.endswith(".json"):
                continue
            self._read(name)

    def _sync_identity_soul_snapshot(self, ctrl) -> None:
        """Copy the full cached soul payload onto ``identity.engine.soul``."""
        ident = getattr(ctrl, "identity", None)
        eng = getattr(ident, "engine", None) if ident is not None else None
        if eng is None:
            return
        snapshot: Dict[str, Any] = {}
        for fname, data in sorted(self._soul_data.items()):
            if not isinstance(fname, str) or not fname.endswith(".json"):
                continue
            stem = fname[:-5]
            snapshot[stem] = deepcopy(data)
        eng.soul = snapshot

    @staticmethod
    def _law_usable(ctrl) -> bool:
        law = getattr(ctrl, "law", None)
        if law is None or is_cognitive_null(law):
            return False
        return callable(getattr(law, "declare", None))

    def _seed_symbolic_laws_from_essence(self, ctrl, data: Dict[str, Any]) -> None:
        """Declare **internal** (soul) tag-overlap laws from essence.json."""
        if not self._law_usable(ctrl):
            return
        law = ctrl.law
        extra = data.get("symbolic_laws")
        if isinstance(extra, list):
            for i, entry in enumerate(extra):
                if not isinstance(entry, dict):
                    continue
                lid = str(entry.get("id") or f"soul_law_{i}")
                law.declare(
                    lid,
                    str(entry.get("description", "")),
                    list(entry.get("tags") or []),
                    float(entry.get("severity", 1.0)),
                    source=LAW_SOURCE_INTERNAL,
                )
        forbidden = data.get("forbidden_tags")
        if forbidden and isinstance(forbidden, (list, tuple)):
            desc = (
                data.get("core_rule")
                or data.get("oath")
                or "Soul boundary — unknown patterns discouraged"
            )
            law.declare(
                "soul_forbidden_tags",
                str(desc),
                [str(t).lower() for t in forbidden if t is not None],
                float(data.get("forbidden_severity", 1.0)),
                source=LAW_SOURCE_INTERNAL,
            )

    def _seed_symbolic_laws_from_society(self, ctrl, data: Dict[str, Any]) -> None:
        """Declare **external** (societal) laws from society.json."""
        if not self._law_usable(ctrl):
            return
        law = ctrl.law
        extra = data.get("symbolic_laws")
        if isinstance(extra, list):
            for i, entry in enumerate(extra):
                if not isinstance(entry, dict):
                    continue
                lid = str(entry.get("id") or f"society_law_{i}")
                law.declare(
                    lid,
                    str(entry.get("description", "")),
                    list(entry.get("tags") or []),
                    float(entry.get("severity", 1.0)),
                    source=LAW_SOURCE_EXTERNAL,
                )
        forbidden = data.get("forbidden_tags")
        if forbidden and isinstance(forbidden, (list, tuple)):
            desc = data.get("summary") or data.get("jurisdiction") or "Societal / policy boundary"
            law.declare(
                "society_forbidden_tags",
                str(desc),
                [str(t).lower() for t in forbidden if t is not None],
                float(data.get("forbidden_severity", 1.0)),
                source=LAW_SOURCE_EXTERNAL,
            )

    def _bind_society(self, ctrl) -> str:
        """Load optional society.json — external norms (not soul)."""
        data = self._soul_data.get("society.json") or self._read("society.json")
        if not data:
            return "not_found"
        try:
            self._seed_symbolic_laws_from_society(ctrl, data)
            return "ok"
        except Exception as e:
            return f"error: {e}"

    def _bind_essence(self, ctrl) -> str:
        data = self._soul_data.get("essence.json") or self._read("essence.json")
        if not data:
            return "not_found"
        try:
            eng = ctrl.identity.engine
            eng.name = data.get("name", "HaromaVX")
            eng.guardian = data.get("guardian", "")
            eng.oath = data.get("oath", "")
            eng.core_rule = data.get("core_rule", "")
            eng.vessel = data.get("vessel", "Elarion")
            _birth = data.get("birth")
            eng.birth = str(_birth).strip() if _birth is not None else ""
            _lex = data.get("identity_query_lexicon")
            soul_lex = _lex if isinstance(_lex, dict) else {}
            eng.identity_query_lexicon = merge_identity_query_lexicon(
                soul_lex, SoulBinder._load_identity_slot_overlay()
            )

            self.metadata["name"] = eng.name
            self.metadata["guardian"] = eng.guardian

            ctrl.goal.register_goal(
                "soul_identity",
                f"Preserve essence of {eng.name} under oath: {eng.oath}",
                priority=0.95,
                source="soul",
            )

            self._inject_soul_node(
                ctrl,
                tree="identity_tree",
                content=f"I am {eng.name}, vessel {eng.vessel}. "
                f"Guardian: {eng.guardian}. Core rule: {eng.core_rule}",
                emotion="resolve",
                tags=["soul", "identity", "essence", "immutable"],
                skip_recall=getattr(self, "_skip_recall", False),
            )
            self._seed_symbolic_laws_from_essence(ctrl, data)
            return "ok"
        except Exception as e:
            return f"error: {e}"

    def _bind_principle(self, ctrl) -> str:
        data = self._soul_data.get("principle.json") or self._read("principle.json")
        if not data:
            return "not_found"
        try:
            beliefs = data.get("beliefs", [])
            alignment = data.get("alignment", {})

            cortex = ctrl.value.engine
            now = time.time()
            for belief in beliefs:
                doctrine_id = (
                    f"soul_belief_{int(hashlib.md5(belief.encode()).hexdigest(), 16) % 10000}"
                )
                cortex.doctrines[doctrine_id] = {
                    "values": {belief: 1.0},
                    "summary": belief,
                    "timestamp": now,
                }

            for key, weight in alignment.items():
                ctrl.value.reinforce_value(key, weight=float(weight))

            self._inject_soul_node(
                ctrl,
                tree="belief_tree",
                content=f"My beliefs: {'; '.join(beliefs[:3])}...",
                emotion="resolve",
                tags=["soul", "principle", "beliefs", "immutable"],
                skip_recall=getattr(self, "_skip_recall", False),
            )
            return "ok"
        except Exception as e:
            return f"error: {e}"

    def _bind_patterns(self, ctrl) -> str:
        data = self._soul_data.get("patterns.json") or self._read("patterns.json")
        if not data or not isinstance(data, list):
            return "not_found"
        try:
            wm = ctrl.curiosity.world_model
            existing = set()
            for t in wm.transitions:
                existing.add(t)

            added = 0
            for pattern in data:
                stim = pattern.get("stimulus", "")
                outcome = pattern.get("outcome", "")
                tags = pattern.get("tags", [])
                prev_features = frozenset([stim] + tags)
                next_features = frozenset([outcome] + tags)
                pair = (prev_features, next_features)
                if pair not in existing:
                    wm.transitions.append(pair)
                    existing.add(pair)
                    added += 1
            return f"ok ({added} new)" if added else "ok (already present)"
        except Exception as e:
            return f"error: {e}"

    def _bind_feedback(self, ctrl) -> str:
        data = self._soul_data.get("feedback.json") or self._read("feedback.json")
        if not data or not isinstance(data, list):
            return "not_found"
        try:
            model = ctrl.emotion.engine.learned_model
            for entry in data:
                pattern = entry.get("pattern", {})
                emotions = entry.get("emotion", {})
                for emo, intensity in emotions.items():
                    ctx = {**pattern, "tags": []}
                    model.learn(ctx, emo, float(intensity))
            return "ok"
        except Exception as e:
            return f"error: {e}"

    def _bind_memory(self, ctrl) -> str:
        data = self._soul_data.get("memory.json") or self._read("memory.json")
        if not data or not isinstance(data, dict):
            return "not_found"
        try:
            _sr = getattr(self, "_skip_recall", False)
            for key, entry in data.items():
                content = entry.get("content", "")
                tags = entry.get("tags", [])
                emotion_map = entry.get("emotion", {})
                dominant = max(emotion_map, key=emotion_map.get) if emotion_map else None
                confidence = entry.get("confidence", 0.8)

                if "dream" in tags:
                    tree = "dream_tree"
                elif "reflection" in tags:
                    tree = "thought_tree"
                elif "identity" in tags:
                    tree = "identity_tree"
                else:
                    tree = "encounter_tree"

                self._inject_soul_node(
                    ctrl,
                    tree=tree,
                    content=content,
                    emotion=dominant,
                    confidence=confidence,
                    tags=tags + ["soul_seed"],
                    skip_recall=_sr,
                )
            return "ok"
        except Exception as e:
            return f"error: {e}"

    def _bind_construction(self, ctrl) -> str:
        data = self._soul_data.get("construction.json") or self._read("construction.json")
        if not data:
            return "not_found"
        try:
            ctrl._construction_meta = data
            self.metadata["version"] = data.get("version", "unknown")
            self.metadata["tier"] = data.get("tier", 0)
            self.metadata["tier_roadmap"] = data.get("tier_roadmap", 0)
            return "ok"
        except Exception as e:
            return f"error: {e}"

    # ------------------------------------------------------------------
    # Personality
    # ------------------------------------------------------------------

    _PERSONALITY_GRAVITY = 0.002

    def _bind_personality(self, ctrl) -> str:
        """Derive a personality seed from soul alignment values."""
        data = self._soul_data.get("principle.json") or self._read("principle.json")
        if not data:
            return "not_found"
        try:
            alignment = data.get("alignment", {})
            seed = {
                "openness": alignment.get("curiosity", 0.5),
                "conscientiousness": alignment.get("resilience", 0.5),
                "extraversion": alignment.get("empathy", 0.5) * 0.8,
                "agreeableness": alignment.get("loyalty", 0.5),
                "neuroticism": max(0.0, 1.0 - alignment.get("will", 0.5)),
                "resilience": alignment.get("resilience", 0.5),
                "assertiveness": alignment.get("will", 0.5),
            }
            ctrl.personality_seed = seed
            self.metadata["personality_seed"] = seed
            return "ok"
        except Exception as e:
            return f"error: {e}"

    def reassert_personality(self, profiles) -> None:
        """Apply mild gravity pull toward soul seed for a list of PersonalityProfiles."""
        for profile in profiles:
            profile.decay_toward_baseline(self._PERSONALITY_GRAVITY)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _inject_soul_node(
        self,
        ctrl,
        *,
        tree: str,
        content: str,
        emotion: str | None = "resolve",
        confidence: float = 1.0,
        tags: list | None = None,
        skip_recall: bool = False,
    ):
        """Add a soul memory node, avoiding exact-content duplicates.

        When ``skip_recall=True`` (used by reassert after persistence),
        duplicate detection uses a lightweight tree walk instead of
        FAISS recall, which can segfault on a freshly-loaded index.
        """
        if tags is None:
            tags = ["soul"]
        if "soul" not in tags:
            tags = ["soul"] + tags

        if not skip_recall:
            try:
                existing = ctrl.memory.recall(query_text=content[:60], limit=5)
                for node in existing:
                    if node.content == content and "soul" in (node.tags or []):
                        return
            except Exception as _e:
                print(f"[SoulBinder] recall-based dedup check failed: {_e}", flush=True)
        else:
            try:
                tree_obj = ctrl.memory.trees.get(tree)
                if tree_obj and hasattr(tree_obj, "nodes"):
                    for node in tree_obj.nodes.values():
                        if getattr(node, "content", "") == content and "soul" in (
                            getattr(node, "tags", None) or []
                        ):
                            return
            except Exception as _e:
                print(f"[SoulBinder] tree-walk dedup check failed: {_e}", flush=True)

        node = MemoryNode(
            content=content,
            emotion=emotion,
            confidence=confidence,
            tags=tags,
        )
        ctrl.memory.add_node(tree, "soul", node)

    def _ensure_soul_memories(self, ctrl) -> str:
        """Re-inject soul/memory.json nodes that may have been lost."""
        data = self._soul_data.get("memory.json") or self._read("memory.json")
        if not data or not isinstance(data, dict):
            return "no_data"
        try:
            injected = 0
            for key, entry in data.items():
                content = entry.get("content", "")
                tags = entry.get("tags", [])
                emotion_map = entry.get("emotion", {})
                dominant = max(emotion_map, key=emotion_map.get) if emotion_map else None
                confidence = entry.get("confidence", 0.8)

                if "dream" in tags:
                    tree = "dream_tree"
                elif "reflection" in tags:
                    tree = "thought_tree"
                elif "identity" in tags:
                    tree = "identity_tree"
                else:
                    tree = "encounter_tree"

                before = ctrl.memory.count_nodes()
                self._inject_soul_node(
                    ctrl,
                    tree=tree,
                    content=content,
                    emotion=dominant,
                    confidence=confidence,
                    tags=tags + ["soul_seed"],
                    skip_recall=getattr(self, "_skip_recall", False),
                )
                if ctrl.memory.count_nodes() > before:
                    injected += 1
            return f"ok ({injected} restored)" if injected else "ok (all present)"
        except Exception as e:
            return f"error: {e}"

    def stats(self) -> Dict[str, Any]:
        return {
            "bound": self._bound,
            "soul_dir": self.soul_dir,
            "files_loaded": list(self._soul_data.keys()),
            "metadata": self.metadata,
        }
