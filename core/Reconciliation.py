"""Domain reflectors and SymbolicReconciliationEngine for X7-style multi-agent reconciliation.

Each domain reflector reads per-agent branches from a symbolic tree, reconciles
them into a merged common view, and returns a list of new common MemoryNodes.
The engine orchestrates all reflectors and calls ``MemoryCore.commit_agent_tree``
to atomically apply results.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional

from core.Memory import MemoryNode
from core.MemoryCore import MemoryCore, AGENT_PREFIX, COMMON_BRANCH


# ═══════════════════════════════════════════════════════════════════════
# Base reflector
# ═══════════════════════════════════════════════════════════════════════


class DomainReflector(ABC):
    """Abstract base: reads agent branches of a single tree, returns merged nodes."""

    tree_name: str = ""

    def __init__(self, memory_core: MemoryCore):
        self.mc = memory_core

    def _agent_nodes(self) -> Dict[str, List[MemoryNode]]:
        """Collect nodes per agent branch."""
        result: Dict[str, List[MemoryNode]] = {}
        for branch in self.mc.list_agent_branches(self.tree_name):
            agent_id = branch[len(AGENT_PREFIX) :]
            result[agent_id] = list(self.mc.forest.get_nodes(self.tree_name, branch))
        return result

    @abstractmethod
    def reconcile_agents(self) -> List[MemoryNode]:
        """Return list of new common-branch nodes merging all agent perspectives."""
        ...


# ═══════════════════════════════════════════════════════════════════════
# Concrete reflectors
# ═══════════════════════════════════════════════════════════════════════


class BeliefReconciler(DomainReflector):
    """Majority-vote on conflicting belief keys across agents."""

    tree_name = "belief_tree"

    def reconcile_agents(self) -> List[MemoryNode]:
        agent_nodes = self._agent_nodes()
        if not agent_nodes:
            return []

        votes: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        confidence_sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for _aid, nodes in agent_nodes.items():
            for node in nodes:
                key = (node.content or "")[:120]
                if not key:
                    continue
                conf = node.confidence if node.confidence is not None else 0.5
                votes[key][node.emotion or "neutral"] += 1
                confidence_sums[key][node.emotion or "neutral"] += conf

        merged: List[MemoryNode] = []
        for key, emotion_votes in votes.items():
            best_emotion = max(emotion_votes, key=emotion_votes.get)
            count = sum(emotion_votes.values())
            avg_conf = confidence_sums[key][best_emotion] / emotion_votes[best_emotion]
            merged.append(
                MemoryNode(
                    content=key,
                    emotion=best_emotion,
                    confidence=min(1.0, avg_conf),
                    tags=["reconciled", "belief", f"votes:{count}"],
                )
            )
        return merged


class GoalReconciler(DomainReflector):
    """Union of goals with priority averaging across agents."""

    tree_name = "goal_tree"

    def reconcile_agents(self) -> List[MemoryNode]:
        agent_nodes = self._agent_nodes()
        if not agent_nodes:
            return []

        goal_pool: Dict[str, List[float]] = defaultdict(list)
        goal_emotions: Dict[str, List[str]] = defaultdict(list)

        for _aid, nodes in agent_nodes.items():
            for node in nodes:
                key = node.content or ""
                if not key:
                    continue
                conf = node.confidence if node.confidence is not None else 0.5
                goal_pool[key].append(conf)
                if node.emotion:
                    goal_emotions[key].append(node.emotion)

        merged: List[MemoryNode] = []
        for content, confs in goal_pool.items():
            avg_conf = sum(confs) / len(confs)
            emotions = goal_emotions.get(content, [])
            dominant = max(set(emotions), key=emotions.count) if emotions else None
            merged.append(
                MemoryNode(
                    content=content,
                    emotion=dominant,
                    confidence=round(avg_conf, 3),
                    tags=["reconciled", "goal"],
                )
            )
        return merged


class ValueReconciler(DomainReflector):
    """Weighted merge of value scores, preserving soul-sourced values at full weight."""

    tree_name = "value_tree"

    def reconcile_agents(self) -> List[MemoryNode]:
        agent_nodes = self._agent_nodes()
        if not agent_nodes:
            return []

        value_confs: Dict[str, List[float]] = defaultdict(list)
        value_emotions: Dict[str, str] = {}
        soul_values: Dict[str, MemoryNode] = {}

        for _aid, nodes in agent_nodes.items():
            for node in nodes:
                key = node.content or ""
                if not key:
                    continue
                conf = node.confidence if node.confidence is not None else 0.5
                if "soul" in (node.tags or []):
                    soul_values[key] = node
                else:
                    value_confs[key].append(conf)
                    if node.emotion:
                        value_emotions[key] = node.emotion

        merged: List[MemoryNode] = []
        for content, node in soul_values.items():
            merged.append(
                MemoryNode(
                    content=content,
                    emotion=node.emotion,
                    confidence=1.0,
                    tags=["reconciled", "value", "soul"],
                )
            )

        for content, confs in value_confs.items():
            if content in soul_values:
                continue
            merged.append(
                MemoryNode(
                    content=content,
                    emotion=value_emotions.get(content),
                    confidence=round(sum(confs) / len(confs), 3),
                    tags=["reconciled", "value"],
                )
            )
        return merged


class EmotionReconciler(DomainReflector):
    """Blended emotion state across agents (intensity averaging)."""

    tree_name = "emotion_tree"

    def reconcile_agents(self) -> List[MemoryNode]:
        agent_nodes = self._agent_nodes()
        if not agent_nodes:
            return []

        emotion_pool: Dict[str, List[float]] = defaultdict(list)

        for _aid, nodes in agent_nodes.items():
            for node in nodes:
                key = node.emotion or (node.content or "")[:60]
                if not key:
                    continue
                conf = node.confidence if node.confidence is not None else 0.5
                emotion_pool[key].append(conf)

        merged: List[MemoryNode] = []
        for emotion_key, intensities in emotion_pool.items():
            avg_intensity = sum(intensities) / len(intensities)
            merged.append(
                MemoryNode(
                    content=f"blended emotion: {emotion_key}",
                    emotion=emotion_key,
                    confidence=round(avg_intensity, 3),
                    tags=["reconciled", "emotion"],
                )
            )
        return merged


class DreamReconciler(DomainReflector):
    """Merges dream motifs from per-agent dream branches into common themes."""

    tree_name = "dream_tree"

    def reconcile_agents(self) -> List[MemoryNode]:
        agent_nodes = self._agent_nodes()
        if not agent_nodes:
            return []

        motif_pool: Dict[str, List[MemoryNode]] = defaultdict(list)
        for _aid, nodes in agent_nodes.items():
            for node in nodes:
                motif = (node.content or "")[:80]
                if motif:
                    motif_pool[motif].append(node)

        merged: List[MemoryNode] = []
        for motif, nodes in motif_pool.items():
            confs = [n.confidence for n in nodes if n.confidence is not None]
            avg_conf = sum(confs) / len(confs) if confs else 0.5
            emotions = [n.emotion for n in nodes if n.emotion]
            dominant = max(set(emotions), key=emotions.count) if emotions else None
            merged.append(
                MemoryNode(
                    content=motif,
                    emotion=dominant,
                    confidence=round(avg_conf, 3),
                    tags=["reconciled", "dream", f"sources:{len(nodes)}"],
                )
            )
        return merged


class EncounterReconciler(DomainReflector):
    """Merge per-agent encounter branches by deduplicating and averaging confidence."""

    tree_name = "encounter_tree"

    def reconcile_agents(self) -> List[MemoryNode]:
        agent_nodes = self._agent_nodes()
        if not agent_nodes:
            return []

        pool: Dict[str, List[MemoryNode]] = defaultdict(list)
        for _aid, nodes in agent_nodes.items():
            for node in nodes:
                key = (node.content or "")[:120]
                if key:
                    pool[key].append(node)

        merged: List[MemoryNode] = []
        for content, nodes in pool.items():
            confs = [n.confidence for n in nodes if n.confidence is not None]
            avg_conf = sum(confs) / len(confs) if confs else 0.5
            emotions = [n.emotion for n in nodes if n.emotion]
            dominant = max(set(emotions), key=emotions.count) if emotions else None
            all_tags = set()
            for n in nodes:
                all_tags.update(n.tags or [])
            all_tags.add("reconciled")
            all_tags.add("encounter")
            merged.append(
                MemoryNode(
                    content=content,
                    emotion=dominant,
                    confidence=round(avg_conf, 3),
                    tags=sorted(all_tags),
                )
            )
        return merged


class ThoughtReconciler(DomainReflector):
    """Merge per-agent thought branches, preserving diverse perspectives."""

    tree_name = "thought_tree"

    def reconcile_agents(self) -> List[MemoryNode]:
        agent_nodes = self._agent_nodes()
        if not agent_nodes:
            return []

        pool: Dict[str, List[MemoryNode]] = defaultdict(list)
        for _aid, nodes in agent_nodes.items():
            for node in nodes:
                key = (node.content or "")[:120]
                if key:
                    pool[key].append(node)

        merged: List[MemoryNode] = []
        for content, nodes in pool.items():
            confs = [n.confidence for n in nodes if n.confidence is not None]
            max_conf = max(confs) if confs else 0.5
            emotions = [n.emotion for n in nodes if n.emotion]
            dominant = max(set(emotions), key=emotions.count) if emotions else None
            sources = len(nodes)
            all_tags = set()
            for n in nodes:
                all_tags.update(n.tags or [])
            all_tags.update(["reconciled", "thought", f"perspectives:{sources}"])
            merged.append(
                MemoryNode(
                    content=content,
                    emotion=dominant,
                    confidence=round(max_conf, 3),
                    tags=sorted(all_tags),
                )
            )
        return merged


class ActionReconciler(DomainReflector):
    """Merge per-agent action branches, keeping highest-scoring outcomes."""

    tree_name = "action_tree"

    def reconcile_agents(self) -> List[MemoryNode]:
        agent_nodes = self._agent_nodes()
        if not agent_nodes:
            return []

        pool: Dict[str, List[MemoryNode]] = defaultdict(list)
        for _aid, nodes in agent_nodes.items():
            for node in nodes:
                key = (node.content or "")[:120]
                if key:
                    pool[key].append(node)

        merged: List[MemoryNode] = []
        for content, nodes in pool.items():
            best = max(nodes, key=lambda n: n.confidence if n.confidence is not None else 0.0)
            all_tags = set()
            for n in nodes:
                all_tags.update(n.tags or [])
            all_tags.update(["reconciled", "action"])
            merged.append(
                MemoryNode(
                    content=content,
                    emotion=best.emotion,
                    confidence=best.confidence if best.confidence is not None else 0.5,
                    tags=sorted(all_tags),
                )
            )
        return merged


# ═══════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════


class SymbolicReconciliationEngine:
    """Orchestrates all domain reflectors and commits reconciled results."""

    def __init__(self, memory_core: MemoryCore):
        self.mc = memory_core
        self.reflectors: List[DomainReflector] = [
            BeliefReconciler(memory_core),
            GoalReconciler(memory_core),
            ValueReconciler(memory_core),
            EmotionReconciler(memory_core),
            DreamReconciler(memory_core),
            EncounterReconciler(memory_core),
            ThoughtReconciler(memory_core),
            ActionReconciler(memory_core),
        ]
        self._last_reconcile_time: float = 0.0
        self._reconcile_count: int = 0

    def _belief_cohesion_snapshot(self, belief_reflector: BeliefReconciler) -> Dict[str, Any]:
        """Pre-commit overlap / consensus / disagreement over per-agent belief strings."""
        ctx = {"cycle_ts": time.time()}
        agent_nodes = belief_reflector._agent_nodes()
        belief_map: Dict[str, List[str]] = {}
        for aid, nodes in agent_nodes.items():
            beliefs: List[str] = []
            for n in nodes:
                key = (n.content or "").strip()[:120]
                if key:
                    beliefs.append(key)
            if beliefs:
                belief_map[aid] = beliefs
        if len(belief_map) < 2:
            return {
                "belief_cohesion": {
                    "agent_count": len(belief_map),
                    "note": "overlap metrics need 2+ agents with beliefs",
                }
            }
        from core.BeliefCohesion import (
            BeliefOverlapAnalyzer,
            ConsensusSynthesizer,
            DisagreementClassifier,
        )

        overlap_an = BeliefOverlapAnalyzer(context=ctx)
        syn = ConsensusSynthesizer(context=ctx)
        diss = DisagreementClassifier(context=ctx)
        overlap = overlap_an.analyze(belief_map)
        consensus = syn.synthesize(belief_map, min_support=2)
        disagree = diss.classify(belief_map, consensus)
        return {
            "belief_cohesion": {
                "agent_count": len(belief_map),
                "shared_belief_count": len(overlap["shared_beliefs"]),
                "shared_preview": overlap["shared_beliefs"][:12],
                "consensus_count": len(consensus),
                "consensus_preview": consensus[:12],
                "unresolved_counts": {k: len(v) for k, v in disagree.items()},
            }
        }

    def has_agent_branches(self) -> bool:
        """Quick check: is there any agent branch across reconcilable trees?"""
        for r in self.reflectors:
            if self.mc.list_agent_branches(r.tree_name):
                return True
        return False

    def reconcile_all(self) -> Dict[str, Any]:
        """Run all reflectors and commit results. Returns per-domain summary."""
        results: Dict[str, Any] = {}
        total_removed = 0

        for reflector in self.reflectors:
            agent_branches = self.mc.list_agent_branches(reflector.tree_name)
            if not agent_branches:
                results[reflector.tree_name] = "no agents"
                continue

            try:
                extra: Dict[str, Any] = {}
                if reflector.tree_name == "belief_tree":
                    extra = self._belief_cohesion_snapshot(
                        reflector  # type: ignore[arg-type]
                    )
                new_nodes = reflector.reconcile_agents()
                removed = self.mc.commit_agent_tree(reflector.tree_name, new_nodes)
                total_removed += removed
                results[reflector.tree_name] = {
                    "merged_nodes": len(new_nodes),
                    "agent_nodes_removed": removed,
                    **extra,
                }
            except Exception as e:
                results[reflector.tree_name] = f"error: {e}"

        self._last_reconcile_time = time.time()
        self._reconcile_count += 1
        results["_meta"] = {
            "total_removed": total_removed,
            "reconcile_count": self._reconcile_count,
        }
        return results

    def stats(self) -> Dict[str, Any]:
        return {
            "reconcile_count": self._reconcile_count,
            "last_reconcile_time": self._last_reconcile_time,
            "has_agent_branches": self.has_agent_branches(),
        }


# Dedupe: controller + background may both call materialize after the same reconcile pass.
_LAST_MATERIALIZE_RECONCILE_COUNT: int = -1
_MATERIALIZE_LOCK = threading.Lock()


def belief_cohesion_summary_lines(results: Dict[str, Any]) -> List[str]:
    """Compact lines for working-memory hooks (personas, diagnostics)."""
    lines: List[str] = []
    bt = results.get("belief_tree")
    if not isinstance(bt, dict):
        return lines
    co = bt.get("belief_cohesion")
    if not isinstance(co, dict):
        return lines
    ac = int(co.get("agent_count") or 0)
    if ac < 1:
        return lines
    sb = int(co.get("shared_belief_count") or 0)
    cc = int(co.get("consensus_count") or 0)
    ur = co.get("unresolved_counts") or {}
    mx = max(ur.values()) if ur else 0
    lines.append(f"Belief cohesion: {ac} view(s), {sb} overlap, {cc} consensus, max split {mx}.")
    previews = co.get("shared_preview") or []
    if isinstance(previews, list) and previews:
        lines.append(f"Shared beliefs include: {', '.join(previews[:4])}.")
    return lines


def materialize_reconciliation_experience(
    memory: Any,
    branch_name: str,
    results: Dict[str, Any],
    *,
    cycle: int = 0,
) -> Optional[str]:
    """Persist reconciliation + belief cohesion into ``thought_tree``; return narrative snippet."""
    global _LAST_MATERIALIZE_RECONCILE_COUNT
    if not results or not callable(getattr(memory, "add_node", None)):
        return None
    meta = results.get("_meta") or {}
    rc = int(meta.get("reconcile_count", -1))
    if rc < 0:
        return None
    with _MATERIALIZE_LOCK:
        if rc == _LAST_MATERIALIZE_RECONCILE_COUNT:
            return None
        _LAST_MATERIALIZE_RECONCILE_COUNT = rc

    narrative_parts: List[str] = []
    wrote = False
    bt = results.get("belief_tree")
    if isinstance(bt, dict):
        co = bt.get("belief_cohesion")
        if isinstance(co, dict) and int(co.get("agent_count") or 0) >= 1:
            ac = int(co.get("agent_count") or 0)
            sb = int(co.get("shared_belief_count") or 0)
            cc = int(co.get("consensus_count") or 0)
            ur = co.get("unresolved_counts") or {}
            mx = max(ur.values()) if ur else 0
            summary = (
                f"[cycle {cycle}] Multi-agent beliefs: {ac} inner view(s), "
                f"{sb} overlapping proposition(s), {cc} consensus; "
                f"max unresolved per view {mx}."
            )
            memory.add_node(
                "thought_tree",
                branch_name,
                MemoryNode(
                    content=summary[:500],
                    emotion="reflective",
                    confidence=0.75,
                    tags=[
                        "reconciliation",
                        "belief_cohesion",
                        "self_model",
                        "multi_agent",
                        f"cycle:{cycle}",
                    ],
                ),
            )
            wrote = True
            narrative_parts.append(
                f"I integrated inner voices: {sb} shared belief(s) across {ac} perspectives."
            )

    if not wrote:
        merged_any = sum(
            int(v.get("merged_nodes", 0)) for v in results.values() if isinstance(v, dict)
        )
        if merged_any > 0:
            sentence = (
                f"[cycle {cycle}] Reconciliation merged {merged_any} node(s) "
                f"into common symbolic memory."
            )
            memory.add_node(
                "thought_tree",
                branch_name,
                MemoryNode(
                    content=sentence,
                    emotion="neutral",
                    confidence=0.6,
                    tags=["reconciliation", "self_model", f"cycle:{cycle}"],
                ),
            )
            narrative_parts.append(sentence)

    return " ".join(narrative_parts) if narrative_parts else None
