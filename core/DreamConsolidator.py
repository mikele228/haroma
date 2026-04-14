"""
DreamConsolidator — real memory consolidation for HaromaX6.

Six consolidation operations:
  1. Replay    — re-score high-salience memories under current emotion
  2. Compress  — merge highly similar memory pairs into abstractions
  3. Pattern   — detect recurring (tag, emotion) co-occurrences
  4. Prune     — forget low-significance old memories
  5. Episodic replay with reconsolidation (Phase 14)
  6. Neural abstraction via autoencoder clustering (Upgrade 9)

The autoencoder learns a compressed latent space from memory embeddings,
discovers clusters of related memories, and creates abstract schema
nodes that generalize across specific experiences.
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import time
import random

try:
    import torch
    import torch.nn as nn
    import numpy as np

    _TORCH = True
except ImportError:
    _TORCH = False
    np = None  # type: ignore

from core.Memory import (
    MemoryForest,
    MemoryNode,
    MemorySignificanceScorer,
    ForgettingCandidateClassifier,
)

_AE_INPUT_DIM = 256
_AE_LATENT_DIM = 32
_AE_HIDDEN_DIM = 128
_AE_REPLAY_CAP = 2048
_CLUSTER_THRESHOLD = 0.8
_MIN_CLUSTER_SIZE = 3

if _TORCH:

    class _MemoryAutoencoder(nn.Module):
        """Autoencoder that compresses memory embeddings into a latent space.

        The latent space naturally clusters related memories; decoding
        from cluster centroids produces abstract schema representations.
        """

        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(_AE_INPUT_DIM, _AE_HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(_AE_HIDDEN_DIM, 64),
                nn.ReLU(),
                nn.Linear(64, _AE_LATENT_DIM),
            )
            self.decoder = nn.Sequential(
                nn.Linear(_AE_LATENT_DIM, 64),
                nn.ReLU(),
                nn.Linear(64, _AE_HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(_AE_HIDDEN_DIM, _AE_INPUT_DIM),
            )

        def forward(self, x: torch.Tensor):
            z = self.encoder(x)
            x_hat = self.decoder(z)
            return x_hat, z

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            return self.decoder(z)


class MemoryAbstractionEngine:
    """Discovers abstract memory schemas via autoencoder latent clustering.

    Workflow during consolidation:
    1. Encode all memory embeddings to latent vectors
    2. Cluster latent vectors by cosine similarity
    3. For each cluster, decode the centroid to get an abstract embedding
    4. Create schema memory nodes that generalize across the cluster
    """

    def __init__(self):
        self._available = _TORCH
        self._net = None
        self._optimizer = None
        self._replay: List[List[float]] = []
        self._train_steps = 0
        self._schemas_created = 0

        if _TORCH:
            self._net = _MemoryAutoencoder()
            self._optimizer = torch.optim.Adam(self._net.parameters(), lr=1e-3)

    @property
    def available(self) -> bool:
        return self._available and self._net is not None

    def observe(self, embedding: List[float]):
        """Buffer a memory embedding for autoencoder training."""
        if not self.available or len(embedding) != _AE_INPUT_DIM:
            return
        self._replay.append(embedding)
        if len(self._replay) > _AE_REPLAY_CAP:
            self._replay = self._replay[-_AE_REPLAY_CAP:]

    def train_step(self) -> float:
        """Train the autoencoder on buffered embeddings."""
        if not self.available or len(self._replay) < 16:
            return 0.0
        batch_size = min(64, len(self._replay))
        batch = random.sample(self._replay, batch_size)
        x = torch.tensor(batch, dtype=torch.float32)

        self._net.train()
        x_hat, z = self._net(x)
        recon_loss = nn.functional.mse_loss(x_hat, x)

        self._optimizer.zero_grad()
        recon_loss.backward()
        self._optimizer.step()
        self._net.eval()
        self._train_steps += 1
        return recon_loss.item()

    def discover_clusters(
        self,
        embeddings: List[Tuple[str, List[float], "MemoryNode"]],
    ) -> List[List[Tuple[str, "MemoryNode"]]]:
        """Find clusters of related memories in latent space."""
        if not self.available or self._train_steps < 20 or len(embeddings) < 5:
            return []

        ids_and_nodes = [(mid, node) for mid, emb, node in embeddings]
        raw_embs = [emb for _, emb, _ in embeddings]
        x = torch.tensor(raw_embs, dtype=torch.float32)

        with torch.no_grad():
            z = self._net.encode(x)
            z_normed = z / (z.norm(dim=1, keepdim=True) + 1e-8)
            sim_matrix = torch.mm(z_normed, z_normed.t())

        n = len(embeddings)
        assigned = set()
        clusters: List[List[int]] = []

        for i in range(n):
            if i in assigned:
                continue
            cluster = [i]
            assigned.add(i)
            for j in range(i + 1, n):
                if j in assigned:
                    continue
                if float(sim_matrix[i, j].item()) > _CLUSTER_THRESHOLD:
                    cluster.append(j)
                    assigned.add(j)
            if len(cluster) >= _MIN_CLUSTER_SIZE:
                clusters.append(cluster)

        result = []
        for cluster_indices in clusters:
            group = [(ids_and_nodes[idx][0], ids_and_nodes[idx][1]) for idx in cluster_indices]
            result.append(group)
        return result

    def create_schema(
        self,
        cluster: List[Tuple[str, "MemoryNode"]],
    ) -> Optional[Dict[str, Any]]:
        """Generate an abstract schema from a memory cluster."""
        if not self.available or len(cluster) < _MIN_CLUSTER_SIZE:
            return None

        all_tags: Dict[str, int] = defaultdict(int)
        all_emotions: Dict[str, int] = defaultdict(int)
        contents: List[str] = []
        total_confidence = 0.0

        for _, node in cluster:
            for tag in node.tags:
                all_tags[tag] += 1
            if node.emotion:
                all_emotions[node.emotion] += 1
            contents.append(node.content[:50])
            total_confidence += node.confidence

        shared_tags = [tag for tag, count in all_tags.items() if count >= len(cluster) // 2]
        dominant_emotion = max(all_emotions, key=all_emotions.get) if all_emotions else "neutral"

        summary_parts = contents[:3]
        schema_content = f"[schema] Abstraction over {len(cluster)} memories: " + " | ".join(
            summary_parts
        )

        self._schemas_created += 1
        return {
            "content": schema_content,
            "emotion": dominant_emotion,
            "confidence": min(0.9, total_confidence / len(cluster)),
            "tags": shared_tags + ["schema", "abstraction", f"cluster_size_{len(cluster)}"],
            "cluster_size": len(cluster),
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "train_steps": self._train_steps,
            "replay_size": len(self._replay),
            "schemas_created": self._schemas_created,
        }


class DreamConsolidator:
    def __init__(self, memory: MemoryForest, encoder=None):
        self.memory = memory
        self._encoder = encoder
        self.scorer = MemorySignificanceScorer()
        self.classifier = ForgettingCandidateClassifier()
        self.dream_count: int = 0
        self.total_pruned: int = 0
        self.total_compressed: int = 0
        self.last_insights: List[str] = []
        self._reconsolidation_count: int = 0
        self.abstraction_engine = MemoryAbstractionEngine()

    def consolidate(
        self,
        emotion_summary: Optional[Dict[str, Any]] = None,
        meta_assessment: Optional[Dict[str, Any]] = None,
        controller=None,
    ) -> Dict[str, Any]:
        emotion_summary = emotion_summary or {}
        meta_assessment = meta_assessment or {}

        replayed = self._replay(emotion_summary)
        compressed = self._compress()
        insights = self._extract_patterns()
        pruned = self._prune()

        reconsolidation = {"replayed": 0, "reconsolidated": 0, "insights": []}
        if controller is not None:
            reconsolidation = self.replay_episodic(controller, emotion_summary)
            insights.extend(reconsolidation.get("insights", []))

        # Neural abstraction (Upgrade 9)
        abstraction_result = self._abstract_memories(controller)
        if abstraction_result.get("schemas_created", 0) > 0:
            insights.append(
                f"Discovered {abstraction_result['schemas_created']} "
                f"abstract patterns across memories"
            )

        dream_narrative = self._compose_dream(replayed, insights, emotion_summary, reconsolidation)

        self.dream_count += 1
        self.last_insights = insights

        if dream_narrative:
            node = MemoryNode(
                content=dream_narrative,
                emotion=emotion_summary.get("dominant", "peace"),
                confidence=0.7,
                tags=["dream", "consolidation", f"dream_{self.dream_count}"],
            )
            self.memory.add_node("dream_tree", "dreamer", node)

        return {
            "replayed": replayed,
            "compressed": compressed,
            "insights": insights,
            "pruned": pruned,
            "dream_narrative": dream_narrative,
            "dream_number": self.dream_count,
            "reconsolidation": reconsolidation,
            "abstraction": abstraction_result,
        }

    # ------------------------------------------------------------------
    # 1. Replay
    # ------------------------------------------------------------------

    def _replay(self, emotion_summary: Dict[str, Any]) -> int:
        current_emotion = emotion_summary.get("dominant", "neutral")
        all_nodes = self._collect_all_nodes()
        if not all_nodes:
            return 0

        scored = []
        for node in all_nodes:
            sal = node.confidence
            if node.emotion and node.emotion == current_emotion:
                sal += 0.3
            if node.emotion and node.emotion != "neutral":
                sal += 0.1
            scored.append((sal, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        replayed = 0
        for sal, node in scored[:10]:
            boost = min(0.15, sal * 0.05)
            node.confidence = min(1.0, node.confidence + boost)
            replayed += 1

        return replayed

    # ------------------------------------------------------------------
    # 2. Compress
    # ------------------------------------------------------------------

    def _compress(self) -> int:
        sem = self.memory.semantic_index
        if sem.tfidf_matrix is None or len(sem.nodes) < 10:
            return 0

        try:
            import numpy as np
        except ImportError:
            return 0

        if sem._dirty:
            sem._rebuild()

        n = len(sem.nodes)
        if n < 2:
            return 0

        use_neural = (
            self._encoder is not None
            and self._encoder.available
            and self._encoder.learned_weight > 0.3
            and getattr(sem, "_dense_vectors", None) is not None
            and sem._dense_vectors.shape[0] == n
        )

        if use_neural:
            mat = sem._dense_vectors
            threshold = 0.75
        else:
            mat = sem.tfidf_matrix
            threshold = 0.85

        if mat is None or mat.shape[0] == 0:
            return 0

        mat_rows = mat.shape[0]
        mat_offset = n - mat_rows if mat_rows < n else 0

        merged = 0
        used = set()
        merge_pairs: List[Tuple[int, int]] = []

        def _row(idx):
            r = mat[idx]
            if hasattr(r, "toarray"):
                return np.asarray(r.toarray()).ravel()
            return np.asarray(r).ravel()

        batch = min(mat_rows, 200)
        indices = list(range(max(0, mat_rows - batch), mat_rows))
        for i in indices:
            if i in used:
                continue
            ri = _row(i)
            for j in indices:
                if j <= i or j in used:
                    continue
                sim = float(np.dot(ri, _row(j)))
                if sim > threshold:
                    merge_pairs.append((i, j))
                    used.add(j)
                    break

        for i, j in merge_pairs[:5]:
            node_a = sem.nodes[i + mat_offset]
            node_b = sem.nodes[j + mat_offset]
            combined_tags = list(set(node_a.tags + node_b.tags))
            merged_content = node_a.content[:60] + " | " + node_b.content[:60]
            avg_conf = (node_a.confidence + node_b.confidence) / 2.0

            abstract_node = MemoryNode(
                content=f"[abstraction] {merged_content}",
                emotion=node_a.emotion or node_b.emotion,
                confidence=avg_conf,
                tags=combined_tags + ["compressed"],
            )
            self.memory.add_node("thought_tree", "consolidation", abstract_node)
            node_b.confidence *= 0.3
            merged += 1

        self.total_compressed += merged
        return merged

    # ------------------------------------------------------------------
    # 3. Pattern extraction
    # ------------------------------------------------------------------

    def _extract_patterns(self) -> List[str]:
        all_nodes = self._collect_all_nodes()
        if len(all_nodes) < 5:
            return []

        tag_emotion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for node in all_nodes:
            if not node.emotion or node.emotion == "neutral":
                continue
            for tag in node.tags:
                if tag.startswith("salience:") or tag.startswith("outcome:"):
                    continue
                tag_emotion[tag][node.emotion] += 1

        insights: List[str] = []
        for tag, emotions in tag_emotion.items():
            for emo, count in emotions.items():
                if count >= 3:
                    insights.append(f"'{tag}' is often paired with {emo} ({count} times)")

        def _insight_count(s: str) -> int:
            try:
                return int(s.split("(")[-1].split()[0])
            except (ValueError, IndexError):
                return 0

        insights.sort(key=_insight_count, reverse=True)
        return insights[:8]

    # ------------------------------------------------------------------
    # 4. Prune
    # ------------------------------------------------------------------

    def _prune(self) -> int:
        total_nodes = self.memory.count_nodes()
        if total_nodes < 100:
            return 0

        pruned = 0
        cutoff = time.time() - 600

        # Collect moment_ids to remove (don't mutate during iteration)
        to_remove_ids: List[str] = []
        for tree in self.memory.trees.values():
            for branch in tree.branches.values():
                branch_removals = 0
                for node in branch.nodes:
                    if branch_removals >= 10:
                        break
                    if node.timestamp > cutoff:
                        continue
                    if node.confidence < 0.2 and "immutable" not in node.tags:
                        if "soul" not in node.tags and "essence" not in node.tags:
                            to_remove_ids.append(node.moment_id)
                            branch_removals += 1

        for mid in to_remove_ids:
            if self.memory.remove_node(mid):
                pruned += 1

        self.total_pruned += pruned
        return pruned

    # ------------------------------------------------------------------
    # 5. Episodic replay with reconsolidation (Phase 14)
    # ------------------------------------------------------------------

    def replay_episodic(self, controller, emotion_summary: Dict[str, Any]) -> Dict[str, Any]:
        experience_nodes = self._find_experience_nodes(limit=50)
        if not experience_nodes:
            return {"replayed": 0, "reconsolidated": 0, "insights": []}

        max_replay = min(3, 1 + self._reconsolidation_count // 10)
        selected = experience_nodes[:max_replay]

        reconsolidated = 0
        replay_insights: List[str] = []

        for node in selected:
            try:
                result = self._reconsolidate_node(node, controller, emotion_summary)
                if result.get("changed"):
                    reconsolidated += 1
                    if result.get("insight"):
                        replay_insights.append(result["insight"])
            except Exception as _dc_exc:
                print(f"[DreamConsolidator] replay_episodic node error: {_dc_exc}", flush=True)
                continue

        self._reconsolidation_count += reconsolidated
        return {
            "replayed": len(selected),
            "reconsolidated": reconsolidated,
            "insights": replay_insights,
        }

    def _find_experience_nodes(self, limit: int = 50) -> List[MemoryNode]:
        all_nodes = self._collect_all_nodes()
        experience = [n for n in all_nodes if "experience" in n.tags]
        if not experience:
            return []

        now = time.time()
        scored = []
        for node in experience:
            age = max(1.0, now - node.timestamp)
            recency = 1.0 / (1.0 + age / 300.0)
            surprise_val = 0.0
            for tag in node.tags:
                if tag.startswith("surprise:"):
                    try:
                        surprise_val = float(tag.split(":")[1])
                    except (ValueError, IndexError):
                        pass
            score = node.confidence * 0.4 + recency * 0.3 + surprise_val * 0.3
            scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:limit]]

    def _reconsolidate_node(
        self, node: MemoryNode, controller, emotion_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        original_emotion = node.emotion or "neutral"

        new_embedding = None
        if hasattr(controller, "encoder") and controller.encoder.available:
            new_embedding = controller.encoder.encode(node.content)

        new_appraisal = {}
        if hasattr(controller, "appraisal"):
            active_goals = []
            if hasattr(controller, "goal"):
                priorities = controller.goal.prioritize()
                active_goals = [
                    {"goal_id": gid, "priority": i} for i, gid in enumerate(priorities[:3])
                ]

            knowledge_summary = {}
            knowledge_diff = {"changed": False}
            if hasattr(controller, "knowledge"):
                knowledge_summary = controller.knowledge.summary()
                knowledge_diff = controller.knowledge.diff()

            identity_summary = {}
            if hasattr(controller, "identity"):
                identity_summary = controller.identity.summarize()

            nlu_result = {"text": node.content, "entities": [], "relations": []}

            new_appraisal = controller.appraisal.evaluate(
                nlu_result=nlu_result,
                active_goals=active_goals,
                knowledge_summary=knowledge_summary,
                knowledge_diff=knowledge_diff,
                identity_summary=identity_summary,
                emotion_summary=emotion_summary,
                drift_score=0.0,
                action_memory_stats={},
                working_memory_load=0.0,
                interlocutor={},
            )

        new_emotion = new_appraisal.get("emotion", original_emotion)
        new_valence = new_appraisal.get("valence", 0.0)
        old_valence = 0.0
        _EMOTION_VALENCE = {
            "joy": 0.8,
            "wonder": 0.6,
            "curiosity": 0.4,
            "peace": 0.3,
            "neutral": 0.0,
            "surprise": 0.1,
            "resolve": 0.2,
            "sadness": -0.5,
            "fear": -0.6,
            "anger": -0.4,
        }
        old_valence = _EMOTION_VALENCE.get(original_emotion, 0.0)

        delta = abs(new_valence - old_valence)
        changed = delta > 0.3

        insight = ""
        if changed:
            if new_valence > old_valence:
                node.confidence = min(1.0, node.confidence + 0.1)
                insight = (
                    f"Reconsolidated: '{node.content[:40]}' "
                    f"shifted from {original_emotion} to {new_emotion} "
                    f"(resolved positively)"
                )
            else:
                node.confidence = max(0.1, node.confidence - 0.1)
                insight = (
                    f"Reconsolidated: '{node.content[:40]}' "
                    f"shifted from {original_emotion} to {new_emotion} "
                    f"(unresolved conflict)"
                )
            node.emotion = new_emotion
            if "reconsolidated" not in node.tags:
                node.tags.append("reconsolidated")

            insight_node = MemoryNode(
                content=f"[insight] {insight}",
                emotion=new_emotion,
                confidence=0.6,
                tags=["reconsolidation", "insight", new_emotion],
            )
            self.memory.add_node("thought_tree", "reconsolidation", insight_node)

        return {"changed": changed, "insight": insight, "delta": delta}

    # ------------------------------------------------------------------
    # Dream narrative
    # ------------------------------------------------------------------

    def _compose_dream(
        self,
        replayed: int,
        insights: List[str],
        emotion_summary: Dict[str, Any],
        reconsolidation: Optional[Dict[str, Any]] = None,
    ) -> str:
        reconsolidation = reconsolidation or {}
        recon_count = reconsolidation.get("reconsolidated", 0)

        if replayed == 0 and not insights and recon_count == 0:
            return ""

        emo = emotion_summary.get("dominant", "peace")
        parts = [f"In a dream colored by {emo},"]

        if replayed > 0:
            parts.append(f"I revisited {replayed} memories,")
            parts.append("strengthening what mattered.")

        if recon_count > 0:
            parts.append(f"I re-experienced {recon_count} past moments with new eyes.")

        if insights:
            parts.append(f"I noticed: {insights[0]}.")
            if len(insights) > 1:
                parts.append(f"Also: {insights[1]}.")

        if self.total_pruned > 0:
            parts.append("Some fading traces were released.")

        parts.append("I awoke with clearer understanding.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # 6. Neural memory abstraction (Upgrade 9)
    # ------------------------------------------------------------------

    def _abstract_memories(self, controller=None) -> Dict[str, Any]:
        """Encode memories through the autoencoder, discover clusters,
        and create schema nodes for each cluster."""
        ae = self.abstraction_engine
        if not ae.available:
            return {"schemas_created": 0, "clusters_found": 0}

        encoder = self._encoder
        if encoder is None or not getattr(encoder, "available", False):
            return {"schemas_created": 0, "clusters_found": 0}

        all_nodes = self._collect_all_nodes()
        embeddings: List[Tuple[str, List[float], MemoryNode]] = []

        for node in all_nodes:
            if "schema" in node.tags or "abstraction" in node.tags:
                continue
            try:
                emb = encoder.encode(node.content)
                if emb is not None and len(emb) == _AE_INPUT_DIM:
                    ae.observe(list(emb))
                    embeddings.append((node.moment_id, list(emb), node))
            except Exception as _dc_exc:
                print(f"[DreamConsolidator] abstract_memories encode error: {_dc_exc}", flush=True)
                continue

        if len(embeddings) < 10:
            return {"schemas_created": 0, "clusters_found": 0}

        ae.train_step()

        clusters = ae.discover_clusters(embeddings)
        schemas_created = 0
        for cluster in clusters:
            schema = ae.create_schema(cluster)
            if schema is None:
                continue
            schema_node = MemoryNode(
                content=schema["content"],
                emotion=schema["emotion"],
                confidence=schema["confidence"],
                tags=schema["tags"],
            )
            self.memory.add_node("thought_tree", "abstraction", schema_node)
            schemas_created += 1

            for _, node in cluster:
                node.confidence = max(0.2, node.confidence * 0.7)

        return {
            "schemas_created": schemas_created,
            "clusters_found": len(clusters),
            "embeddings_processed": len(embeddings),
        }

    # ------------------------------------------------------------------

    def _collect_all_nodes(self) -> List[MemoryNode]:
        if hasattr(self.memory, "get_all_nodes"):
            return self.memory.get_all_nodes()
        nodes: List[MemoryNode] = []
        for tree in self.memory.trees.values():
            for branch in tree.branches.values():
                nodes.extend(branch.nodes)
        return nodes

    def summarize(self) -> Dict[str, Any]:
        return {
            "dream_count": self.dream_count,
            "total_pruned": self.total_pruned,
            "total_compressed": self.total_compressed,
            "last_insights": self.last_insights[:3],
            "reconsolidation_count": self._reconsolidation_count,
            "abstraction": self.abstraction_engine.stats(),
        }
