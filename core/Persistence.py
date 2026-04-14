"""
CognitivePersistence — durable state for HaromaX6.

Saves and loads all accumulated cognitive state so the system
survives process restarts with its memories, learned models,
narrative, and cycle history intact.

Boot performance:
  - Memory forest uses a pickle cache (_forest_cache.pkl) for fast
    reload (~0.5s vs ~14s JSON).  JSON remains the source of truth;
    the pickle cache is rebuilt on save or invalidated on mismatch.
  - Every load step is timed and logged.
"""

from typing import Any, Dict, List, Optional
import json
import os
import pickle
import threading
import time


class CognitivePersistence:
    def __init__(self, data_dir: str = "data/cognitive"):
        self.data_dir = data_dir
        self._last_save_cycle = 0
        self._save_interval = 10
        self._save_lock = threading.Lock()

    @staticmethod
    def apply_emotion_snapshot(emotion_mgr: Any, emotion_data: Dict[str, Any]) -> None:
        """Restore learned emotion associations from persisted JSON."""
        from collections import defaultdict

        model = emotion_mgr.engine.learned_model
        raw = emotion_data.get("associations", {})
        model.associations = defaultdict(
            lambda: defaultdict(float),
            {k: defaultdict(float, v) for k, v in raw.items()},
        )
        model.experience_count = emotion_data.get("experience_count", 0)

    def _ensure_dir(self):
        os.makedirs(self.data_dir, exist_ok=True)

    def _path(self, name: str) -> str:
        return os.path.join(self.data_dir, name)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, controller) -> Dict[str, Any]:
        """Persist cognitive state. Serialized so overlapping saves never interleave."""
        with self._save_lock:
            return self._save_impl(controller)

    def _save_impl(self, controller) -> Dict[str, Any]:
        self._ensure_dir()
        saved: Dict[str, str] = {}

        if not hasattr(controller, "memory") or controller.memory is None:
            saved["memory_forest"] = "skipped (no memory)"
        else:
            try:
                dirty = controller.memory.get_dirty_tree_names()
                if dirty:
                    tree_dir = os.path.join(self.data_dir, "memory_trees")
                    os.makedirs(tree_dir, exist_ok=True)
                    for tree_name in dirty:
                        tree_data = controller.memory.tree_to_dict(tree_name)
                        if tree_data is not None:
                            path = os.path.join(tree_dir, f"{tree_name}.json")
                            tmp_path = path + ".tmp"
                            try:
                                with open(tmp_path, "w", encoding="utf-8") as f:
                                    json.dump(tree_data, f, ensure_ascii=False, default=str)
                                    f.flush()
                                    os.fsync(f.fileno())
                                os.replace(tmp_path, path)
                            except Exception:
                                try:
                                    os.remove(tmp_path)
                                except OSError:
                                    pass
                                raise
                    controller.memory.mark_trees_clean(dirty)
                    self._invalidate_forest_cache()
                    self._invalidate_bulk_cache()
                    saved["memory_forest"] = f"ok ({len(dirty)} trees)"
                else:
                    saved["memory_forest"] = "ok (no changes)"
            except Exception as e:
                saved["memory_forest"] = f"error: {e}"

        _emo = getattr(controller, "emotion", None) or getattr(controller, "persist_emotion", None)
        if _emo is not None and hasattr(_emo, "engine"):
            try:
                model = _emo.engine.learned_model
                emotion_data = {
                    "associations": {k: dict(v) for k, v in model.associations.items()},
                    "experience_count": model.experience_count,
                }
                self._write_json("emotion_model.json", emotion_data)
                saved["emotion_model"] = "ok"
            except Exception as e:
                saved["emotion_model"] = f"error: {e}"
        else:
            saved["emotion_model"] = "skipped (no learned emotion target)"

        if hasattr(controller, "curiosity") and controller.curiosity is not None:
            try:
                wm = controller.curiosity.world_model
                _trans_snap = list(wm.transitions)
                transitions = [[sorted(prev), sorted(nxt)] for prev, nxt in _trans_snap]
                world_data = {
                    "transitions": transitions,
                    "total_error": wm.total_error,
                    "total_predictions": wm.total_predictions,
                }
                novelty_data = dict(controller.curiosity.novelty_baseline)
                self._write_json("world_model.json", world_data)
                self._write_json("curiosity.json", novelty_data)
                saved["world_model"] = "ok"
            except Exception as e:
                saved["world_model"] = f"error: {e}"
        else:
            saved["world_model"] = "skipped (no curiosity)"

        if hasattr(controller, "action_memory") and controller.action_memory:
            try:
                _entries = list(controller.action_memory.entries)
                self._write_json("action_memory.json", _entries)
                saved["action_memory"] = "ok"
            except Exception as e:
                saved["action_memory"] = f"error: {e}"
        else:
            saved["action_memory"] = "skipped (no action_memory)"

        _nbuf = None
        if hasattr(controller, "_narrative_buffer"):
            _nbuf = controller._narrative_buffer
        else:
            _src = getattr(controller, "persist_narrative_source", None)
            if _src is not None:
                _nbuf = getattr(_src, "_narrative_buffer", None)
        if _nbuf is not None:
            try:
                _nbuf = list(_nbuf)
                self._write_json("narrative.json", _nbuf)
                saved["narrative"] = "ok"
            except Exception as e:
                saved["narrative"] = f"error: {e}"
        else:
            saved["narrative"] = "skipped (no narrative buffer)"

        try:
            meta = {
                "cycle_count": getattr(controller, "cycle_count", 0),
                "timestamp": time.time(),
                "version": "X6-phase16",
            }
            if hasattr(controller, "drives"):
                _drives_snap = list(controller.drives.drives)
                meta["drives"] = {d.name: d.level for d in _drives_snap}
            if hasattr(controller, "metacognition"):
                try:
                    meta["meta_history_len"] = len(controller.metacognition._history)
                except Exception:
                    meta["meta_history_len"] = 0
            if hasattr(controller, "dream_consolidator"):
                meta["reconsolidation_count"] = controller.dream_consolidator._reconsolidation_count
            if hasattr(controller, "_current_plan"):
                meta["current_plan"] = list(controller._current_plan)
                meta["plan_step"] = controller._plan_step
            self._write_json("cycle_meta.json", meta)
            saved["cycle_meta"] = "ok"
        except Exception as e:
            saved["cycle_meta"] = f"error: {e}"

        # metacognition history saved as part of metacognition_learned.json via to_dict()

        if hasattr(controller, "working_memory"):
            try:
                self._write_json("working_memory.json", controller.working_memory.to_dict())
                saved["working_memory"] = "ok"
            except Exception as e:
                saved["working_memory"] = f"error: {e}"

        if hasattr(controller, "conversation"):
            try:
                self._write_json("conversation.json", controller.conversation.to_dict())
                saved["conversation"] = "ok"
            except Exception as e:
                saved["conversation"] = f"error: {e}"

        if hasattr(controller, "temporal"):
            try:
                self._write_json("temporal.json", controller.temporal.to_dict())
                saved["temporal"] = "ok"
            except Exception as e:
                saved["temporal"] = f"error: {e}"

        if hasattr(controller, "knowledge"):
            try:
                self._write_json("knowledge_graph.json", controller.knowledge.to_dict())
                saved["knowledge_graph"] = "ok"
            except Exception as e:
                saved["knowledge_graph"] = f"error: {e}"

        if hasattr(controller, "interlocutor_model"):
            try:
                self._write_json("interlocutor_model.json", controller.interlocutor_model.to_dict())
                saved["interlocutor_model"] = "ok"
            except Exception as e:
                saved["interlocutor_model"] = f"error: {e}"

        if hasattr(controller, "identity") and hasattr(controller.identity, "to_dict"):
            try:
                self._write_json("identity_engine.json", controller.identity.to_dict())
                saved["identity_engine"] = "ok"
            except Exception as e:
                saved["identity_engine"] = f"error: {e}"

        if (
            hasattr(controller, "meaning_lexicon")
            and getattr(controller, "meaning_lexicon", None) is not None
        ):
            try:
                self._write_json(
                    "meaning_lexicon.json",
                    controller.meaning_lexicon.to_dict(),
                )
                saved["meaning_lexicon"] = "ok"
            except Exception as e:
                saved["meaning_lexicon"] = f"error: {e}"

        if hasattr(controller, "reasoning") and hasattr(controller.reasoning, "rule_learner"):
            try:
                self._write_json(
                    "reasoning_rules.json", controller.reasoning.rule_learner.to_dict()
                )
                saved["reasoning_rules"] = "ok"
            except Exception as e:
                saved["reasoning_rules"] = f"error: {e}"

        if hasattr(controller, "appraisal"):
            try:
                self._write_json("appraisal.json", controller.appraisal.to_dict())
                saved["appraisal"] = "ok"
            except Exception as e:
                saved["appraisal"] = f"error: {e}"

        if hasattr(controller, "encoder"):
            try:
                self._write_json("neural_encoder.json", controller.encoder.to_dict())
                saved["neural_encoder"] = "ok"
            except Exception as e:
                saved["neural_encoder"] = f"error: {e}"

        if hasattr(controller, "self_model"):
            try:
                self._write_json("self_model.json", controller.self_model.to_dict())
                saved["self_model"] = "ok"
            except Exception as e:
                saved["self_model"] = f"error: {e}"

        if hasattr(controller, "composer"):
            try:
                self._write_json("language_composer.json", controller.composer.to_dict())
                saved["language_composer"] = "ok"
            except Exception as e:
                saved["language_composer"] = f"error: {e}"

        if hasattr(controller, "attention"):
            try:
                self._write_json("learned_attention.json", controller.attention.to_dict())
                saved["learned_attention"] = "ok"
            except Exception as e:
                saved["learned_attention"] = f"error: {e}"

        if hasattr(controller, "modulation") and hasattr(controller.modulation, "to_dict"):
            try:
                self._write_json("embodied_modulation.json", controller.modulation.to_dict())
                saved["embodied_modulation"] = "ok"
            except Exception as e:
                saved["embodied_modulation"] = f"error: {e}"

        if hasattr(controller, "drives") and hasattr(controller.drives, "to_dict"):
            try:
                self._write_json("homeostatic_drives.json", controller.drives.to_dict())
                saved["homeostatic_drives"] = "ok"
            except Exception as e:
                saved["homeostatic_drives"] = f"error: {e}"

        if hasattr(controller, "perception"):
            try:
                nlu = controller.perception.engine.nlu
                if hasattr(nlu, "lexicon"):
                    self._write_json("nlu_lexicon.json", nlu.lexicon.to_dict())
                    saved["nlu_lexicon"] = "ok"
            except Exception as e:
                saved["nlu_lexicon"] = f"error: {e}"

        if hasattr(controller, "imagination"):
            try:
                self._write_json("imagination.json", controller.imagination.to_dict())
                saved["imagination"] = "ok"
            except Exception as e:
                saved["imagination"] = f"error: {e}"

        if hasattr(controller, "goal_synthesizer"):
            try:
                self._write_json("goal_synthesizer.json", controller.goal_synthesizer.to_dict())
                saved["goal_synthesizer"] = "ok"
            except Exception as e:
                saved["goal_synthesizer"] = f"error: {e}"

        if hasattr(controller, "metacognition") and hasattr(controller.metacognition, "to_dict"):
            try:
                self._write_json("metacognition_learned.json", controller.metacognition.to_dict())
                saved["metacognition_learned"] = "ok"
            except Exception as e:
                saved["metacognition_learned"] = f"error: {e}"

        if hasattr(controller, "counterfactual") and hasattr(controller.counterfactual, "to_dict"):
            try:
                self._write_json("counterfactual_gate.json", controller.counterfactual.to_dict())
                saved["counterfactual_gate"] = "ok"
            except Exception as e:
                saved["counterfactual_gate"] = f"error: {e}"

        if hasattr(controller, "process_gate") and hasattr(controller.process_gate, "to_dict"):
            try:
                self._write_json("process_gate.json", controller.process_gate.to_dict())
                saved["process_gate"] = "ok"
            except Exception as e:
                saved["process_gate"] = f"error: {e}"

        if hasattr(controller, "backbone") and hasattr(controller.backbone, "to_dict"):
            try:
                self._write_json("cognitive_backbone.json", controller.backbone.to_dict())
                saved["cognitive_backbone"] = "ok"
            except Exception as e:
                saved["cognitive_backbone"] = f"error: {e}"

        if hasattr(controller, "discourse") and hasattr(controller.discourse, "to_dict"):
            try:
                self._write_json("discourse_state.json", controller.discourse.to_dict())
                saved["discourse_state"] = "ok"
            except Exception as e:
                saved["discourse_state"] = f"error: {e}"

        if hasattr(controller, "grounder") and hasattr(controller.grounder, "to_dict"):
            try:
                self._write_json("environment_grounder.json", controller.grounder.to_dict())
                saved["environment_grounder"] = "ok"
            except Exception as e:
                saved["environment_grounder"] = f"error: {e}"

        if hasattr(controller, "mental_simulator") and hasattr(
            controller.mental_simulator, "to_dict"
        ):
            try:
                self._write_json("mental_simulator.json", controller.mental_simulator.to_dict())
                saved["mental_simulator"] = "ok"
            except Exception as e:
                saved["mental_simulator"] = f"error: {e}"

        if hasattr(controller, "arch_searcher") and hasattr(controller.arch_searcher, "to_dict"):
            try:
                self._write_json("arch_searcher.json", controller.arch_searcher.to_dict())
                saved["arch_searcher"] = "ok"
            except Exception as e:
                saved["arch_searcher"] = f"error: {e}"

        if hasattr(controller, "training_scheduler") and hasattr(
            controller.training_scheduler, "to_dict"
        ):
            try:
                self._write_json("training_scheduler.json", controller.training_scheduler.to_dict())
                saved["training_scheduler"] = "ok"
            except Exception as e:
                saved["training_scheduler"] = f"error: {e}"

        profiles = getattr(controller, "_personality_profiles", None) or []
        if profiles:
            try:
                all_snaps = {}
                for prof in profiles:
                    snap = prof.snapshot()
                    base_keys = list(snap.get("baseline", {}).keys())
                    agent_id = getattr(prof, "_owner_id", None) or f"idx_{profiles.index(prof)}"
                    all_snaps[agent_id] = snap
                self._write_json("personality_profiles.json", all_snaps)
                saved["personality_profiles"] = f"ok ({len(all_snaps)})"
            except Exception as e:
                saved["personality_profiles"] = f"error: {e}"

        if hasattr(controller, "autonomy_persist_blob"):
            try:
                blob = controller.autonomy_persist_blob()
                if isinstance(blob, dict):
                    self._write_json("autonomy_state.json", blob)
                    saved["autonomy_state"] = "ok"
                    self._invalidate_bulk_cache()
            except Exception as e:
                saved["autonomy_state"] = f"error: {e}"

        self._last_save_cycle = getattr(controller, "cycle_count", 0)
        return saved

    def load_personality_profiles(self, profiles) -> Dict[str, str]:
        """Restore personality snapshots onto existing PersonalityProfile instances."""
        loaded: Dict[str, str] = {}
        data = self._read_json("personality_profiles.json")
        if not data or not isinstance(data, dict):
            return {"personality_profiles": "no data"}
        restored = 0
        for prof in profiles:
            owner = getattr(prof, "_owner_id", None)
            snap = data.get(owner) or data.get(f"idx_{profiles.index(prof)}")
            if snap:
                prof.load_snapshot(snap)
                restored += 1
        loaded["personality_profiles"] = f"ok ({restored} restored)"
        return loaded

    def save_environment(self, environment) -> Dict[str, str]:
        """Save TextEnvironment state separately."""
        self._ensure_dir()
        saved: Dict[str, str] = {}
        try:
            self._write_json("text_environment.json", environment.to_dict())
            saved["text_environment"] = "ok"
        except Exception as e:
            saved["text_environment"] = f"error: {e}"
        return saved

    def save_grounder(self, grounder) -> Dict[str, str]:
        """Save EnvironmentGrounder state."""
        self._ensure_dir()
        saved: Dict[str, str] = {}
        try:
            self._write_json("environment_grounder.json", grounder.to_dict())
            saved["environment_grounder"] = "ok"
        except Exception as e:
            saved["environment_grounder"] = f"error: {e}"
        return saved

    def load_grounder(self, grounder) -> Dict[str, str]:
        """Load EnvironmentGrounder state."""
        loaded: Dict[str, str] = {}
        data = self._read_json("environment_grounder.json")
        if data and isinstance(data, dict):
            try:
                grounder.from_dict(data)
                loaded["environment_grounder"] = "ok"
            except Exception as e:
                loaded["environment_grounder"] = f"error: {e}"
        return loaded

    def load_environment(self, environment) -> Dict[str, str]:
        """Load TextEnvironment state."""
        loaded: Dict[str, str] = {}
        env_data = self._read_json("text_environment.json")
        if env_data and isinstance(env_data, dict):
            try:
                environment.from_dict(env_data)
                loaded["text_environment"] = "ok"
            except Exception as e:
                loaded["text_environment"] = f"error: {e}"
        return loaded

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    _BULK_CACHE = "_boot_cache.pkl"

    def load(self, controller) -> Dict[str, Any]:
        loaded: Dict[str, str] = {}
        t0 = time.time()

        def _timed(label, fn):
            """Run fn and log timing."""
            mt = time.time()
            try:
                result = fn()
            except Exception as e:
                dt = time.time() - mt
                print(f"    [LOAD] FAIL  {label:25s} {dt:.2f}s  {e}", flush=True)
                return f"error: {e}"
            dt = time.time() - mt
            tag = "SLOW!" if dt > 2.0 else "ok"
            print(f"    [LOAD] {tag:5s} {label:25s} {dt:.2f}s", flush=True)
            return result

        # -- try bulk pickle cache first (covers all JSON files) -------
        bulk_cache = self._try_load_bulk_cache()
        if bulk_cache is not None:
            print(f"    [LOAD] Using bulk pickle cache", flush=True)

        def _cached_read_json(name):
            """Read from bulk cache if available, else from disk."""
            if bulk_cache is not None:
                return bulk_cache.get(name)
            return self._read_json(name)

        # -- memory forest (try pickle cache first) --------------------
        def _load_forest():
            cache_path = os.path.join(self.data_dir, "_forest_cache.pkl")
            if os.path.exists(cache_path):
                try:
                    ct = time.time()
                    with open(cache_path, "rb") as f:
                        forest_data = pickle.load(f)
                    print(
                        f"    [LOAD]       pickle cache read in {time.time() - ct:.2f}s", flush=True
                    )
                    self._restore_forest(controller.memory, forest_data)
                    return "ok (pickle cache)"
                except Exception:
                    print("    [LOAD]       pickle cache invalid, falling back to JSON", flush=True)

            tree_dir = os.path.join(self.data_dir, "memory_trees")
            if os.path.isdir(tree_dir):
                tree_count = 0
                combined = {}
                for fname in os.listdir(tree_dir):
                    if not fname.endswith(".json"):
                        continue
                    path = os.path.join(tree_dir, fname)
                    with open(path, "r", encoding="utf-8") as f:
                        tree_data = json.load(f)
                    tree_name = fname[:-5]
                    combined[tree_name] = tree_data
                    tree_count += 1
                self._restore_forest(controller.memory, combined)
                self._write_forest_cache(combined)
                return f"ok ({tree_count} trees, cache written)"
            else:
                forest_data = self._read_json("memory_forest.json")
                if forest_data:
                    self._restore_forest(controller.memory, forest_data)
                    self._write_forest_cache(forest_data)
                    return "ok (cache written)"
            return "no data"

        loaded["memory_forest"] = _timed("memory_forest", _load_forest)

        def _load_emotion():
            emotion_data = self._read_json("emotion_model.json")
            if not emotion_data:
                return "no data"
            if hasattr(controller, "emotion") and controller.emotion is not None:
                CognitivePersistence.apply_emotion_snapshot(controller.emotion, emotion_data)
                return "ok"
            if type(controller).__name__ == "SharedResources":
                controller._pending_emotion_restore = emotion_data
                return "ok (deferred to executive)"
            return "no data"

        if hasattr(controller, "emotion") or type(controller).__name__ == "SharedResources":
            loaded["emotion_model"] = _timed("emotion_model", _load_emotion)

        def _load_world_model():
            if not hasattr(controller, "curiosity") or controller.curiosity is None:
                return "skipped (no curiosity)"
            world_data = self._read_json("world_model.json")
            if world_data:
                wm = controller.curiosity.world_model
                wm.transitions = [
                    (frozenset(t[0]), frozenset(t[1])) for t in world_data.get("transitions", [])
                ]
                wm.total_error = world_data.get("total_error", 0.0)
                wm.total_predictions = world_data.get("total_predictions", 0)
                return "ok"
            return "no data"

        loaded["world_model"] = _timed("world_model", _load_world_model)

        def _load_curiosity():
            if not hasattr(controller, "curiosity") or controller.curiosity is None:
                return "skipped (no curiosity)"
            novelty_data = self._read_json("curiosity.json")
            if novelty_data and isinstance(novelty_data, dict):
                controller.curiosity.novelty_baseline = {k: int(v) for k, v in novelty_data.items()}
                return "ok"
            return "no data"

        loaded["curiosity"] = _timed("curiosity", _load_curiosity)

        def _load_action_memory():
            action_data = self._read_json("action_memory.json")
            if action_data and isinstance(action_data, list):
                _am = controller.action_memory
                _lk = getattr(_am, "_lock", None)
                if _lk:
                    with _lk:
                        _am.entries = action_data
                        _am._rebuild_index()
                else:
                    _am.entries = action_data
                    _am._rebuild_index()
                return "ok"
            return "no data"

        loaded["action_memory"] = _timed("action_memory", _load_action_memory)

        def _load_narrative():
            narrative_data = self._read_json("narrative.json")
            if narrative_data and isinstance(narrative_data, list):
                if hasattr(controller, "_narrative_buffer"):
                    controller._narrative_buffer = narrative_data
                    return "ok"
                if type(controller).__name__ == "SharedResources":
                    controller._pending_narrative_restore = narrative_data
                    return "ok (deferred to executive)"
            return "no data"

        if (
            hasattr(controller, "_narrative_buffer")
            or type(controller).__name__ == "SharedResources"
        ):
            loaded["narrative"] = _timed("narrative", _load_narrative)

        _deferred: list = []

        def _load_json_into(label, json_file, target, attr_or_fn, *, check_type=dict, defer=False):
            """Generic: read JSON -> call from_dict or set attr."""

            def _do():
                data = _cached_read_json(json_file)
                if data is None:
                    return "no data"
                if check_type and not isinstance(data, check_type):
                    return "no data"
                if callable(attr_or_fn):
                    attr_or_fn(data)
                else:
                    target.from_dict(data)
                return "ok"

            if target is None:
                return
            if defer and (time.time() - t0) > 5.0:
                _deferred.append((label, _do))
                loaded[label] = "deferred"
                print(f"    [LOAD] DEFER {label:25s} (past 5s)", flush=True)
            else:
                loaded[label] = _timed(label, _do)

        # -- cycle_meta (fast, needed for cycle_count) ------------------
        def _load_cycle_meta():
            meta = self._read_json("cycle_meta.json")
            if not meta:
                return "no data"
            controller.cycle_count = meta.get("cycle_count", 0)
            if hasattr(controller, "dream_consolidator"):
                controller.dream_consolidator._reconsolidation_count = meta.get(
                    "reconsolidation_count", 0
                )
            if hasattr(controller, "_current_plan"):
                controller._current_plan = meta.get("current_plan", [])
                controller._plan_step = meta.get("plan_step", 0)
            if hasattr(controller, "drives") and "drives" in meta:
                for d in controller.drives.drives:
                    if d.name in meta["drives"]:
                        d.level = meta["drives"][d.name]
            return "ok"

        loaded["cycle_meta"] = _timed("cycle_meta", _load_cycle_meta)

        # -- remaining modules (generic from_dict pattern) -------------
        if not os.path.exists(self._path("metacognition_learned.json")):
            _load_json_into(
                "metacognition",
                "metacognition.json",
                getattr(controller, "metacognition", None),
                lambda d: setattr(controller.metacognition, "_history", d.get("history", [])),
            )
        _load_json_into(
            "working_memory",
            "working_memory.json",
            getattr(controller, "working_memory", None),
            lambda d: controller.working_memory.from_dict(d),
            check_type=list,
        )
        _load_json_into(
            "conversation",
            "conversation.json",
            getattr(controller, "conversation", None),
            lambda d: controller.conversation.from_dict(d),
        )
        _load_json_into(
            "temporal",
            "temporal.json",
            getattr(controller, "temporal", None),
            lambda d: controller.temporal.from_dict(d),
        )
        _load_json_into(
            "knowledge_graph",
            "knowledge_graph.json",
            getattr(controller, "knowledge", None),
            lambda d: controller.knowledge.from_dict(d),
        )
        _load_json_into(
            "interlocutor_model",
            "interlocutor_model.json",
            getattr(controller, "interlocutor_model", None),
            lambda d: controller.interlocutor_model.from_dict(d),
        )
        _id_eng = getattr(controller, "identity", None)
        if _id_eng is not None and hasattr(_id_eng, "from_dict"):
            _load_json_into(
                "identity_engine",
                "identity_engine.json",
                _id_eng,
                _id_eng.from_dict,
            )
        _ml = getattr(controller, "meaning_lexicon", None)
        if _ml is not None and hasattr(_ml, "from_dict"):
            _load_json_into("meaning_lexicon", "meaning_lexicon.json", _ml, _ml.from_dict)
        if hasattr(controller, "reasoning") and hasattr(controller.reasoning, "rule_learner"):
            _load_json_into(
                "reasoning_rules",
                "reasoning_rules.json",
                controller.reasoning.rule_learner,
                lambda d: controller.reasoning.rule_learner.from_dict(d),
            )
        _load_json_into(
            "appraisal",
            "appraisal.json",
            getattr(controller, "appraisal", None),
            lambda d: controller.appraisal.from_dict(d),
        )
        _load_json_into(
            "neural_encoder",
            "neural_encoder.json",
            getattr(controller, "encoder", None),
            lambda d: controller.encoder.from_dict(d),
        )
        _load_json_into(
            "self_model",
            "self_model.json",
            getattr(controller, "self_model", None),
            lambda d: controller.self_model.from_dict(d),
            defer=True,
        )
        _load_json_into(
            "language_composer",
            "language_composer.json",
            getattr(controller, "composer", None),
            lambda d: controller.composer.from_dict(d),
            defer=True,
        )
        _load_json_into(
            "learned_attention",
            "learned_attention.json",
            getattr(controller, "attention", None),
            lambda d: controller.attention.from_dict(d),
            defer=True,
        )
        if hasattr(controller, "modulation") and hasattr(controller.modulation, "from_dict"):
            _load_json_into(
                "embodied_modulation",
                "embodied_modulation.json",
                controller.modulation,
                lambda d: controller.modulation.from_dict(d),
            )
        if hasattr(controller, "drives") and hasattr(controller.drives, "from_dict"):
            _load_json_into(
                "homeostatic_drives",
                "homeostatic_drives.json",
                controller.drives,
                lambda d: controller.drives.from_dict(d),
            )
        if hasattr(controller, "perception"):

            def _load_nlu(data):
                nlu = controller.perception.engine.nlu
                if hasattr(nlu, "lexicon"):
                    nlu.lexicon.from_dict(data)

            _load_json_into(
                "nlu_lexicon", "nlu_lexicon.json", controller.perception, _load_nlu, defer=True
            )
        _load_json_into(
            "imagination",
            "imagination.json",
            getattr(controller, "imagination", None),
            lambda d: controller.imagination.from_dict(d),
            defer=True,
        )
        _load_json_into(
            "goal_synthesizer",
            "goal_synthesizer.json",
            getattr(controller, "goal_synthesizer", None),
            lambda d: controller.goal_synthesizer.from_dict(d),
            defer=True,
        )
        if hasattr(controller, "metacognition") and hasattr(controller.metacognition, "from_dict"):
            _load_json_into(
                "metacognition_learned",
                "metacognition_learned.json",
                controller.metacognition,
                lambda d: controller.metacognition.from_dict(d),
                defer=True,
            )
        if hasattr(controller, "counterfactual") and hasattr(
            controller.counterfactual, "from_dict"
        ):
            _load_json_into(
                "counterfactual_gate",
                "counterfactual_gate.json",
                controller.counterfactual,
                lambda d: controller.counterfactual.from_dict(d),
                defer=True,
            )
        if hasattr(controller, "process_gate") and hasattr(controller.process_gate, "from_dict"):
            _load_json_into(
                "process_gate",
                "process_gate.json",
                controller.process_gate,
                lambda d: controller.process_gate.from_dict(d),
                defer=True,
            )
        if hasattr(controller, "backbone") and hasattr(controller.backbone, "from_dict"):
            _load_json_into(
                "cognitive_backbone",
                "cognitive_backbone.json",
                controller.backbone,
                lambda d: controller.backbone.from_dict(d),
                defer=True,
            )
        if hasattr(controller, "discourse") and hasattr(controller.discourse, "from_dict"):
            _load_json_into(
                "discourse_state",
                "discourse_state.json",
                controller.discourse,
                lambda d: controller.discourse.from_dict(d),
            )
        if hasattr(controller, "grounder") and hasattr(controller.grounder, "from_dict"):
            _load_json_into(
                "environment_grounder",
                "environment_grounder.json",
                controller.grounder,
                lambda d: controller.grounder.from_dict(d),
            )
        if hasattr(controller, "mental_simulator") and hasattr(
            controller.mental_simulator, "from_dict"
        ):
            _load_json_into(
                "mental_simulator",
                "mental_simulator.json",
                controller.mental_simulator,
                lambda d: controller.mental_simulator.from_dict(d),
            )
        if hasattr(controller, "arch_searcher") and hasattr(controller.arch_searcher, "from_dict"):
            _load_json_into(
                "arch_searcher",
                "arch_searcher.json",
                controller.arch_searcher,
                lambda d: controller.arch_searcher.from_dict(d),
            )
        if hasattr(controller, "training_scheduler") and hasattr(
            controller.training_scheduler, "from_dict"
        ):
            _load_json_into(
                "training_scheduler",
                "training_scheduler.json",
                controller.training_scheduler,
                lambda d: controller.training_scheduler.from_dict(d),
            )

        def _load_autonomy_state():
            data = _cached_read_json("autonomy_state.json")
            if not data or not isinstance(data, dict):
                return "no data"
            if hasattr(controller, "autonomy_restore_blob"):
                controller.autonomy_restore_blob(data)
                return "ok"
            return "skipped"

        loaded["autonomy_state"] = _timed("autonomy_state", _load_autonomy_state)

        # -- run deferred loads in background thread --------------------
        if _deferred:
            import threading

            _deferred_done = threading.Event()

            def _run_deferred():
                for label, fn in _deferred:
                    mt = time.time()
                    try:
                        fn()
                        dt = time.time() - mt
                        print(f"    [LOAD] bg    {label:25s} {dt:.2f}s", flush=True)
                    except Exception as e:
                        print(f"    [LOAD] bg-FAIL {label:25s} {e}", flush=True)
                print(f"    [LOAD] Deferred loads complete", flush=True)
                _deferred_done.set()

            t = threading.Thread(target=_run_deferred, daemon=True)
            t.start()
            self._deferred_thread = t
            self._deferred_done = _deferred_done
            print(f"    [LOAD] {len(_deferred)} loads deferred to background thread", flush=True)

        total = time.time() - t0
        print(
            f"    [LOAD] TOTAL persistence_load {total:.2f}s ({len(_deferred)} deferred)",
            flush=True,
        )
        return loaded

    def wait_for_deferred(self, timeout: float = 60.0) -> bool:
        """Block until deferred loads complete. Returns True if done, False on timeout."""
        ev = getattr(self, "_deferred_done", None)
        if ev is None:
            return True
        return ev.wait(timeout=timeout)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _try_load_bulk_cache(self) -> Optional[Dict[str, Any]]:
        """Try to load bulk JSON cache (all .json files pre-parsed)."""
        cache_path = os.path.join(self.data_dir, self._BULK_CACHE)
        if not os.path.exists(cache_path):
            return None
        try:
            ct = time.time()
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            print(f"    [LOAD] Bulk cache loaded in {time.time() - ct:.2f}s", flush=True)
            return data
        except Exception:
            print("    [LOAD] Bulk cache corrupted, using JSON", flush=True)
            return None

    def write_bulk_cache(self):
        """Snapshot all JSON files into a single pickle for fast boot."""
        with self._save_lock:
            cache_path = os.path.join(self.data_dir, self._BULK_CACHE)
            blob: Dict[str, Any] = {}
            for fname in os.listdir(self.data_dir):
                if fname.endswith(".json"):
                    data = self._read_json(fname)
                    if data is not None:
                        blob[fname] = data
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"    [LOAD] Bulk cache written ({len(blob)} files)", flush=True)
            except Exception as e:
                print(f"    [LOAD] Bulk cache write failed: {e}", flush=True)

    def _write_forest_cache(self, forest_data: Dict[str, Any]):
        """Write pickle cache for fast subsequent boots."""
        cache_path = os.path.join(self.data_dir, "_forest_cache.pkl")
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(forest_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as _e:
            print(f"[Persistence] forest cache write failed: {_e}", flush=True)

    def _invalidate_forest_cache(self):
        """Delete stale pickle cache after save (new trees on disk)."""
        cache_path = os.path.join(self.data_dir, "_forest_cache.pkl")
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
        except Exception as _e:
            print(f"[Persistence] forest cache invalidation failed: {_e}", flush=True)

    def _invalidate_bulk_cache(self):
        """Delete stale bulk cache after save."""
        cache_path = os.path.join(self.data_dir, self._BULK_CACHE)
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
        except Exception as _e:
            print(f"[Persistence] bulk cache invalidation failed: {_e}", flush=True)

    def should_save(self, cycle_count: int, interval: int = 10) -> bool:
        return (cycle_count - self._last_save_cycle) >= interval

    def _restore_forest(self, forest, data: Dict[str, Any]):
        from core.Memory import MemoryNode

        for tree_name, tree_data in data.items():
            if not isinstance(tree_data, dict):
                continue
            branches = tree_data.get("branches", {})
            for branch_name, branch_data in branches.items():
                if not isinstance(branch_data, dict):
                    continue
                for node_dict in branch_data.get("nodes", []):
                    node = MemoryNode.from_dict(node_dict)
                    forest.add_node(tree_name, branch_name, node)

    def _write_json(self, name: str, data):
        path = self._path(name)
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    def _read_json(self, name: str):
        path = self._path(name)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            print(
                f"[Persistence] WARNING: failed to read {name}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return None

    def stats(self) -> Dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "last_save_cycle": self._last_save_cycle,
            "save_interval": self._save_interval,
            "files_on_disk": (
                len(os.listdir(self.data_dir)) if os.path.isdir(self.data_dir) else 0
            ),
        }
