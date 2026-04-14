import copy
from typing import Dict, Any, Optional, List
from utils.module_base import ModuleBase
import time


class IdentityEngine(ModuleBase):
    """Tracks identity snapshots, role transitions, and phase shifts."""

    def __init__(self):
        super().__init__("IdentityEngine")
        self.snapshots: List[Dict[str, Any]] = []
        self.current_role: str = "observer"
        self.current_phase: str = "awakening"
        self.principles: List[str] = []
        #: Full soul directory payload (stem → parsed JSON). Set by SoulBinder.
        self.soul: Dict[str, Any] = {}
        #: Runtime merge: soul ``identity_query_lexicon`` + config overlay.
        self.identity_query_lexicon: Dict[str, Any] = {}

    _MAX_SNAPSHOTS = 500

    def record_snapshot(self, state: Dict[str, Any], role: Optional[str] = None):
        if role:
            self.current_role = role
        self.snapshots.append(
            {
                "timestamp": time.time(),
                "role": self.current_role,
                "phase": self.current_phase,
                "state_keys": list(state.keys()),
            }
        )
        if len(self.snapshots) > self._MAX_SNAPSHOTS:
            self.snapshots = self.snapshots[-self._MAX_SNAPSHOTS :]

    def assign_role(self, role: str):
        self.current_role = role

    def set_phase(self, phase: str):
        self.current_phase = phase

    def forecast_identity(self) -> str:
        if len(self.snapshots) < 2:
            return "stable"
        seen = set()
        for s in self.snapshots[-5:]:
            seen.add(s["role"])
            if len(seen) > 2:
                return "shifting"
        return "stable"

    def get_snapshot(self, agent_id: str = None) -> Dict[str, Any]:
        return self.summarize()

    def summarize(self) -> Dict[str, Any]:
        # SoulBinder sets ``name`` / ``vessel`` from essence.json when present.
        _n = getattr(self, "name", None)
        _v = getattr(self, "vessel", None)
        _b = getattr(self, "birth", None)
        out: Dict[str, Any] = {
            "current_role": self.current_role,
            "current_phase": self.current_phase,
            "snapshot_count": len(self.snapshots),
            "forecast": self.forecast_identity(),
            "essence_name": (_n or "") if isinstance(_n, str) else "",
            "vessel": (_v or "") if isinstance(_v, str) else "",
        }
        if _b is not None and str(_b).strip():
            out["birth"] = str(_b).strip()
        if self.soul:
            out["soul"] = copy.deepcopy(self.soul)
        _lex = self.identity_query_lexicon
        if _lex:
            out["identity_query_lexicon"] = copy.deepcopy(_lex)
        return out

    def to_dict(self) -> Dict[str, Any]:
        """Serialize runtime state for persistence."""
        return {
            "current_role": self.current_role,
            "current_phase": self.current_phase,
            "principles": list(self.principles),
            "snapshots": self.snapshots[-self._MAX_SNAPSHOTS :],
        }

    def from_dict(self, data: Dict[str, Any]):
        """Restore runtime state from persistence."""
        if not isinstance(data, dict):
            return
        self.current_role = data.get("current_role", self.current_role)
        self.current_phase = data.get("current_phase", self.current_phase)
        self.principles = data.get("principles", self.principles)
        saved_snaps = data.get("snapshots")
        if isinstance(saved_snaps, list):
            self.snapshots = saved_snaps[-self._MAX_SNAPSHOTS :]

    def reset(self):
        self.snapshots.clear()
        self.current_role = "observer"
        self.current_phase = "awakening"
        self.soul = {}
        self.identity_query_lexicon = {}
