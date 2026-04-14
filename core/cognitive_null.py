"""
Unified sentinel for cognitive modules that failed to load or were skipped
(resource tier, boot deadline, optional organs).

``isinstance`` / :func:`is_cognitive_null` replace string type checks.
``engine`` returns ``self`` so ``manager.engine`` mirrors real wrappers.

Unknown attribute names **raise** :exc:`AttributeError` (so :func:`hasattr` is
accurate).  Whitelisted method names resolve to small callables with
type-appropriate defaults (empty str/list/dict, 0.0, False, etc.).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional


def _z_dict(*_a: Any, **_k: Any) -> Dict[str, Any]:
    return {}


def _z_list(*_a: Any, **_k: Any) -> List[Any]:
    return []


def _z_str(*_a: Any, **_k: Any) -> str:
    return ""


def _z_none(*_a: Any, **_k: Any) -> None:
    return None


def _z_false(*_a: Any, **_k: Any) -> bool:
    return False


def _z_true(*_a: Any, **_k: Any) -> bool:
    return True


def _z_float(*_a: Any, **_k: Any) -> float:
    return 0.0


def _z_int(*_a: Any, **_k: Any) -> int:
    return 0


def _evaluate_shared(*_a: Any, **_k: Any) -> Dict[str, Any]:
    """Curiosity + appraisal both call ``evaluate``; return a merged safe shape."""
    return {
        "questions": [],
        "novelty": 0.0,
        "curiosity_score": 0.0,
        "overrides": False,
        "emotion": None,
    }


def _drives_update(*_a: Any, **_k: Any) -> Dict[str, Any]:
    return {
        "drives": {},
        "dominant_drive": "rest",
        "dominant_level": 0.0,
        "urgency": 0.0,
        "urgent_drives": [],
    }


def _reasoning_reason(*_a: Any, **_k: Any) -> Dict[str, Any]:
    return {"reasoning_depth": 0}


def _imagination_simulate(*_a: Any, **_k: Any) -> Dict[str, Any]:
    return {"scenarios": [], "imagined_plan": []}


# Methods sometimes invoked on stubbed engines before/without ``_module_real`` guards.
_NULL_METHODS: Dict[str, Callable[..., Any]] = {
    # LLM / composer / generative
    "generate": _z_str,
    "generate_chat": _z_str,
    "generate_streaming": _z_str,
    "generate_async": _z_str,
    "generate_best_of_n": _z_str,
    "score_with_trained_heads": _z_float,
    "build_prompt": _z_str,
    "record_outcome": _z_none,
    "train_reward_model": _z_float,
    "save_finetune_data": _z_int,
    "train_step": _z_float,
    "train_generative_step": _z_float,
    "extract_phrases": _z_none,
    "_build_context": _z_dict,
    "compose": _z_str,
    # Knowledge graph
    "integrate": _z_none,
    "summary": _z_dict,
    "diff": _z_dict,
    "find_gaps": _z_list,
    "add_entity": _z_none,
    "add_relation": _z_none,
    # Temporal
    "get_temporal_position": _z_dict,
    "summarize_arc": _z_str,
    "record": _z_none,
    "get_recent_sequence": _z_list,
    "detect_repetition": _z_dict,
    "bind_event": _z_none,
    # Appraisal / modulation / discourse / perception
    "evaluate": _evaluate_shared,
    "compute": _z_dict,
    "process": _z_dict,
    "perceive": _z_dict,
    # Identity / law / myth / fusion (manager surface)
    "update": _drives_update,
    "declare": _z_none,
    "revoke": _z_none,
    "check": _z_list,
    "check_compliance": _z_list,
    "bind": _z_str,
    "fuse": _z_dict,
    "resolve": _z_dict,
    # Reasoning / counterfactual / imagination / metacognition
    "reason": _reasoning_reason,
    "simulate": _imagination_simulate,
    "get_strategy_recommendation": _z_none,
    "propose": _z_list,
    "learn_from_cycle": _z_none,
    "inspect": _z_dict,
    # Self-model / attention / process gate / backbone
    "forward": _z_dict,
    "predict": _z_dict,
    "allocate": _z_dict,
    "adjust_salience": _z_float,
    # Encoder / grounder / simulator
    "encode": _z_none,
    "ground": _z_dict,
    "simulate_turn": _z_dict,
    # Goal synthesis / reconciliation / emotion engine surface
    "synthesize": _z_list,
    "reconcile": _z_dict,
    "ingest": _z_none,
    "apply_decay": _z_none,
    "update_emotion": _z_none,
    # Dream / consolidator
    "consolidate": _z_dict,
    "should_dream": _z_false,
    "bias_goals": _z_list,
    # Misc
    "reset": _z_none,
    "export": _z_str,
    "import_data": _z_none,
    "learn": _z_none,
    "get_summary": _z_dict,
    "forecast_identity": _z_str,
    "forecast_next": _z_dict,
    "forecast_sequence": _z_list,
    "detect_drift": _z_dict,
    "detect_collapse": _z_dict,
    "detect": _z_dict,
    "log_loop_event": _z_str,
    "reflect_on_state": _z_dict,
    "timeline": _z_list,
    "memory_threads": _z_list,
    "define_doctrine": _z_none,
    "evaluate_conflict": _z_dict,
    "reinforce_value": _z_none,
    "classify_dream": _z_str,
    "generate_dream": _z_dict,
    "run_cycle": _z_dict,
    "dispatch": _z_dict,
}


class _NullLearnedModel:
    def learn(self, *a: Any, **k: Any) -> None:
        return None


class _NullLexicon:
    total_words = 0
    grown_words = 0

    def stats(self) -> Dict[str, Any]:
        return {}


class _NullNLU:
    def __init__(self) -> None:
        self.lexicon = _NullLexicon()

    def learn_from_emotion(self, *a: Any, **k: Any) -> Any:
        return {}

    def stats(self) -> Dict[str, Any]:
        return {}


def _make_drive(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        level=0.0,
        is_urgent=False,
        rise_rate=0.03,
        decay_rate=0.05,
        urgency_threshold=0.7,
        rise=lambda *a, **k: None,
        decay=lambda *a, **k: None,
        to_dict=lambda: {"name": name, "level": 0.0},
    )


class CognitiveNull:
    """Stub module.

    Unknown names raise :exc:`AttributeError`, so :func:`hasattr` is accurate.
    For "is this a real subsystem?" checks, use :func:`is_cognitive_null` or
    :func:`has_runtime_feature`.
    """

    available = False
    learned_weight = 0.0
    model_name: str = ""

    def __init__(self) -> None:
        self.goals: Dict[str, Any] = {}
        self.soul: Dict[str, Any] = {}
        self.learned_model = _NullLearnedModel()
        self._nlu = _NullNLU()
        self._history: List[Any] = []
        self._active_proposals: List[Any] = []
        self.episode_timeline: List[Any] = []
        self.relations: List[Any] = []
        drive_names = (
            "understanding",
            "coherence",
            "expression",
            "rest",
            "connection",
        )
        self.drives: List[Any] = [_make_drive(n) for n in drive_names]
        self._drive_map: Dict[str, Any] = {d.name: d for d in self.drives}
        self.adaptation = SimpleNamespace(
            stats=_z_dict,
            record=lambda *a, **k: None,
            adapt=lambda *a, **k: None,
            _adaptation_steps=0,
        )

    @property
    def engine(self) -> CognitiveNull:
        return self

    @property
    def nlu(self) -> _NullNLU:
        return self._nlu

    def get(self, name: str, default: Any = None) -> Any:
        return self._drive_map.get(name, default)

    def prioritize(self, *a: Any, **k: Any) -> List[str]:
        return []

    def register_goal(self, *a: Any, **k: Any) -> Any:
        return None

    def register_input_goal(self, *a: Any, **k: Any) -> Any:
        return None

    def complete_input_goal(self, *a: Any, **k: Any) -> bool:
        return False

    def current_input_goal(self, *a: Any, **k: Any) -> Optional[str]:
        return None

    def activate(self, *a: Any, **k: Any) -> Dict[str, Any]:
        return {}

    def record_mission(self, *a: Any, **k: Any) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        fn = _NULL_METHODS.get(name)
        if fn is not None:
            return fn
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r} "
            "(cognitive module unavailable)"
        )

    def stats(self) -> Dict[str, Any]:
        return {"available": False, "skipped": True, "entity_count": 0}

    def summarize(self) -> Dict[str, Any]:
        """Match engine ``summarize() -> dict`` so ``.summarize().get(...)`` and
        ``EpisodeContext.bind_identity`` never see a bare string from a stub."""
        drive_levels = {str(d.name): 0.0 for d in self.drives}
        return {
            "available": False,
            "skipped": True,
            "summary": "module skipped (resource tier)",
            "curiosity_score": 0.0,
            "novelty": 0.0,
            "prediction_error": 0.0,
            "questions": [],
            "current_role": "observer",
            "current_phase": "stable",
            "snapshot_count": 0,
            "forecast": "",
            "essence_name": "",
            "vessel": "",
            "drives": drive_levels,
            "dominant": "rest",
            "urgent": [],
            "adaptation": {},
            "history_length": 0,
            "history": [],
        }


def is_cognitive_null(obj: Any) -> bool:
    return isinstance(obj, CognitiveNull)


def has_runtime_feature(obj: Any, attr: str) -> bool:
    """True if *obj* is a live (non-null) object and exposes *attr*.

    Prefer this over bare :func:`hasattr` when *obj* may be a skipped module:
    short-circuits on :class:`CognitiveNull` without probing dynamic stubs.
    """
    if obj is None or is_cognitive_null(obj):
        return False
    return hasattr(obj, attr)
