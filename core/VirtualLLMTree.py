"""
VirtualLLMTree — treat the LLM as a first-class MemoryForest tree.

All persisted LLM-facing memory lives under ``llm_tree`` with conventional
branches:

  * ``learning`` — critique / lexicon consolidation summaries (written by callers)
  * ``transcript`` — optional persisted prompt/response turns from ``generate*``
  * ``registry`` — optional model/backend snapshot nodes

Code that should "talk to the LLM through the forest" uses :class:`VirtualLLMTree`
on ``SharedResources.virtual_llm_tree`` instead of calling ``llm_backend`` and
``memory.add_node`` separately.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.Memory import MemoryForest, MemoryNode

if TYPE_CHECKING:
    pass

LLM_TREE_NAME = "llm_tree"
LLM_BRANCH_LEARNING = "learning"
LLM_BRANCH_TRANSCRIPT = "transcript"
LLM_BRANCH_REGISTRY = "registry"


class VirtualLLMTree:
    """Facade: ``llm_tree`` branches + optional ``LLMBackend`` inference."""

    def __init__(self, forest: MemoryForest, llm_backend: Any):
        self._forest = forest
        self._llm = llm_backend

    @property
    def tree_name(self) -> str:
        return LLM_TREE_NAME

    @property
    def llm(self) -> Any:
        """Underlying :class:`LLMBackend` (may be stub or unavailable)."""
        return self._llm

    @property
    def available(self) -> bool:
        return self._llm is not None and bool(getattr(self._llm, "available", False))

    def list_branches(self) -> List[str]:
        tree = self._forest.get_tree(LLM_TREE_NAME)
        if not tree:
            return []
        return sorted(tree.branches.keys())

    def get_nodes(self, branch: str) -> List[MemoryNode]:
        return self._forest.get_nodes(LLM_TREE_NAME, branch)

    def append(
        self,
        branch: str,
        content: str,
        *,
        emotion: Optional[str] = None,
        confidence: float = 0.75,
        tags: Optional[List[str]] = None,
    ) -> MemoryNode:
        """Append a node under ``llm_tree`` / *branch*."""
        base_tags = ["virtual_llm_tree", f"branch:{branch}"]
        if tags:
            base_tags.extend(tags)
        node = MemoryNode(
            content=content,
            emotion=emotion,
            confidence=confidence,
            tags=base_tags,
        )
        self._forest.add_node(LLM_TREE_NAME, branch, node)
        return node

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        persist: bool = False,
        transcript_branch: str = LLM_BRANCH_TRANSCRIPT,
    ) -> Optional[str]:
        """Run ``LLMBackend.generate``; optionally record a turn on *transcript*."""
        if not self.available:
            return None
        text = self._llm.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        if persist and text:
            p = (prompt or "").strip()[:1200]
            t = (text or "").strip()[:2000]
            self.append(
                transcript_branch,
                f"[user]\n{p}\n\n[assistant]\n{t}",
                tags=["llm_turn", "generate"],
                confidence=0.85,
            )
        return text

    def generate_chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        persist: bool = False,
        transcript_branch: str = LLM_BRANCH_TRANSCRIPT,
    ) -> Optional[str]:
        """Run ``LLMBackend.generate_chat``; optionally flatten to *transcript*."""
        if not self.available:
            return None
        text = self._llm.generate_chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        if persist and text:
            lines = []
            for m in messages:
                role = str(m.get("role", "user")).strip()
                body = str(m.get("content", "") or "").strip()[:800]
                if body:
                    lines.append(f"[{role}]\n{body}")
            lines.append(f"[assistant]\n{(text or '').strip()[:2000]}")
            self.append(
                transcript_branch,
                "\n\n".join(lines),
                tags=["llm_turn", "generate_chat"],
                confidence=0.85,
            )
        return text

    def refresh_registry_snapshot(self) -> Optional[MemoryNode]:
        """Append a one-line backend description under ``registry``."""
        if self._llm is None:
            return None
        name = getattr(self._llm, "model_name", "") or "none"
        bt = getattr(self._llm, "backend_type", "?")
        content = f"t={time.time():.0f} backend={bt} model={name} available={self.available}"
        return self.append(
            LLM_BRANCH_REGISTRY,
            content,
            tags=["registry_snapshot"],
            confidence=1.0,
        )

    def summary(self) -> Dict[str, Any]:
        """Lightweight introspection for HTTP / debugging."""
        branches = self.list_branches()
        counts = {b: len(self.get_nodes(b)) for b in branches}
        return {
            "tree": LLM_TREE_NAME,
            "branches": branches,
            "node_counts": counts,
            "llm_available": self.available,
            "model_name": getattr(self._llm, "model_name", None) if self._llm else None,
        }
