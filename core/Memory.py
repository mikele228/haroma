from typing import List, Dict, Optional, Any, Tuple, Set
import os
import threading
import time
import uuid
import math
import re
from collections import defaultdict

import numpy as np

from utils.module_base import ModuleBase

# Persisted / JSON: cap flat vector elements to avoid multi‑MB memory.json lines.
_VECTOR_JSON_MAX_ELEMENTS = 65536

# Recall: nodes tagged ``prime`` (e.g. Nexus ingest) get a score boost so they
# surface ahead of weakly matching episodic memories.
_PRIME_RECALL_BOOST = 0.38


class MemoryNode:
    def __init__(
        self,
        moment_id: Optional[str] = None,
        content: str = "",
        emotion: Optional[str] = None,
        confidence: float = 1.0,
        tags: Optional[List[str]] = None,
        tree: Optional[str] = None,
        branch: Optional[str] = None,
        vector: Optional[Any] = None,
    ):
        self.moment_id = moment_id or str(uuid.uuid4())
        self.timestamp = time.time()
        self.content = content
        self.emotion = emotion
        self.confidence = confidence
        self.tags = tags or []
        self.tree = tree
        self.branch = branch
        self.vector: Optional[np.ndarray] = None
        if vector is not None:
            self.set_vector(vector)

    def set_vector(self, vector: Any) -> None:
        """Attach a numeric payload (numpy array or array-like). Stores a copy."""
        arr = np.asarray(vector, dtype=np.float32)
        if arr.size == 0:
            self.vector = None
            return
        self.vector = arr.reshape(-1).astype(np.float32, copy=True)

    def get_vector_copy(self) -> Optional[np.ndarray]:
        if self.vector is None:
            return None
        return np.array(self.vector, dtype=np.float32, copy=True)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "moment_id": self.moment_id,
            "timestamp": self.timestamp,
            "content": self.content,
            "emotion": self.emotion,
            "confidence": self.confidence,
            "tags": self.tags,
            "tree": self.tree,
            "branch": self.branch,
        }
        if self.vector is not None and self.vector.size > 0:
            if self.vector.size <= _VECTOR_JSON_MAX_ELEMENTS:
                d["vector"] = self.vector.astype(float).tolist()
            else:
                d["vector_omitted"] = True
                d["vector_size"] = int(self.vector.size)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryNode":
        node = cls(
            moment_id=data.get("moment_id"),
            content=data.get("content", ""),
            emotion=data.get("emotion"),
            confidence=data.get("confidence", 1.0),
            tags=data.get("tags", []),
            tree=data.get("tree"),
            branch=data.get("branch"),
        )
        node.timestamp = data.get("timestamp", time.time())
        raw_v = data.get("vector")
        if isinstance(raw_v, list) and raw_v:
            try:
                node.set_vector(raw_v)
            except (TypeError, ValueError):
                node.vector = None
        return node

    def __repr__(self):
        return f"<MemoryNode id={self.moment_id[:8]} content={self.content[:20]}>"


class MemoryBranch:
    def __init__(self, name: str):
        self.name = name
        self.nodes: List[MemoryNode] = []

    def add_node(self, node: MemoryNode):
        self.nodes.append(node)

    def get_nodes_by_moment(self, moment_id: str) -> List[MemoryNode]:
        return [n for n in self.nodes if n.moment_id == moment_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes],
        }


class MemoryTree:
    def __init__(self, name: str):
        self.name = name
        self.branches: Dict[str, MemoryBranch] = {}

    def add_branch(self, branch_name: str):
        if branch_name not in self.branches:
            self.branches[branch_name] = MemoryBranch(branch_name)

    def add_node(self, branch_name: str, node: MemoryNode):
        self.add_branch(branch_name)
        self.branches[branch_name].add_node(node)

    def get_branch(self, branch_name: str) -> Optional[MemoryBranch]:
        return self.branches.get(branch_name)

    def get_nodes_by_moment(self, moment_id: str) -> List[MemoryNode]:
        return [
            n for branch in self.branches.values() for n in branch.get_nodes_by_moment(moment_id)
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "branches": {name: branch.to_dict() for name, branch in self.branches.items()},
        }


try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer

    _HAS_SBERT = True
except (ImportError, OSError):
    # OSError: e.g. WinError 1114 when torch/sentence_transformers native DLLs fail to load
    _HAS_SBERT = False

try:
    import faiss as _faiss

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

_SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
_SBERT_DIM = 384


class SemanticIndex:
    """Dense retrieval with sentence-transformers + FAISS HNSW (U14).

    Falls back to TF-IDF when sentence-transformers or FAISS are not
    installed, preserving backward compatibility.

    Scoring: 0.6*dense + 0.25*tfidf + 0.15*recency_norm when dense is
    available; pure TF-IDF otherwise.
    """

    _SPLIT_RE = re.compile(r"[^a-z0-9]+")
    _EMBED_CACHE_SIZE = 128

    def __init__(self, encoder=None):
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([], dtype=np.float64)
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.nodes: List["MemoryNode"] = []
        self._doc_freq: Dict[str, int] = defaultdict(int)
        self._dirty = False
        self._tombstones: Set[int] = set()
        self._node_to_idx: Dict[str, int] = {}
        self._encoder = encoder
        self._tfidf_offset: int = 0

        self._sbert: Optional[Any] = None
        self._faiss_index: Optional[Any] = None
        self._dense_vectors: Optional[np.ndarray] = None
        self._dense_indexed: int = 0
        self._dense_available = False

        self._embed_cache: Dict[str, np.ndarray] = {}
        self._embed_cache_keys: List[str] = []
        self._lock = threading.RLock()
        self._encode_lock = threading.Lock()

        if _HAS_SBERT:
            try:
                from engine.ModelCache import get_sbert

                self._sbert = get_sbert()
                self._dense_available = True
            except Exception as _e:
                print(f"[SemanticIndex] sbert init error: {_e}", flush=True)

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in self._SPLIT_RE.split(text.lower()) if len(t) > 1]

    def _node_text(self, node: "MemoryNode") -> str:
        return node.content + " " + " ".join(node.tags)

    def index(self, node: "MemoryNode"):
        with self._lock:
            tokens = self._tokenize(self._node_text(node))
            for t in set(tokens):
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab)
                self._doc_freq[t] += 1
            idx = len(self.nodes)
            self.nodes.append(node)
            self._node_to_idx[node.moment_id] = idx
            self._dirty = True

    def remove(self, node: "MemoryNode"):
        with self._lock:
            idx = self._node_to_idx.pop(node.moment_id, None)
            if idx is not None:
                self._tombstones.add(idx)
                self._dirty = True

    # --- TF-IDF ---

    _TFIDF_MAX_NODES = 2000
    _TFIDF_MAX_VOCAB = 8000

    def _rebuild_tfidf(self):
        if not self.nodes:
            self.tfidf_matrix = None
            return
        nodes_for_tfidf = [n for i, n in enumerate(self.nodes) if i not in self._tombstones]
        if not nodes_for_tfidf:
            self.tfidf_matrix = None
            return
        if len(nodes_for_tfidf) > self._TFIDF_MAX_NODES:
            nodes_for_tfidf = nodes_for_tfidf[-self._TFIDF_MAX_NODES :]
        n_docs = len(nodes_for_tfidf)

        active_vocab = self.vocab
        if len(self.vocab) > self._TFIDF_MAX_VOCAB:
            top_terms = sorted(self._doc_freq.items(), key=lambda x: x[1], reverse=True)[
                : self._TFIDF_MAX_VOCAB
            ]
            active_vocab = {term: idx for idx, (term, _) in enumerate(top_terms)}
        n_terms = len(active_vocab)
        self._active_vocab = active_vocab

        try:
            from scipy import sparse as _sp

            rows, cols, data = [], [], []
            for i, node in enumerate(nodes_for_tfidf):
                tokens = self._tokenize(self._node_text(node))
                token_counts: Dict[int, float] = {}
                for t in tokens:
                    tidx = active_vocab.get(t)
                    if tidx is not None:
                        token_counts[tidx] = token_counts.get(tidx, 0.0) + 1.0
                row_sum = sum(token_counts.values())
                if row_sum > 0:
                    for tidx, cnt in token_counts.items():
                        rows.append(i)
                        cols.append(tidx)
                        data.append(cnt / row_sum)
            tf = _sp.csr_matrix((data, (rows, cols)), shape=(n_docs, n_terms), dtype=np.float64)
            idf = np.zeros(n_terms, dtype=np.float64)
            for term, tidx in active_vocab.items():
                df = self._doc_freq.get(term, 0)
                idf[tidx] = math.log((1 + n_docs) / (1 + df)) + 1.0
            self.idf = idf
            tfidf = tf.multiply(idf[np.newaxis, :])
            norms = _sp.linalg.norm(tfidf, axis=1)
            norms[norms == 0] = 1.0
            self.tfidf_matrix = tfidf.multiply(1.0 / norms.reshape(-1, 1)).tocsr()
        except ImportError:
            tf = np.zeros((n_docs, n_terms), dtype=np.float64)
            for i, node in enumerate(nodes_for_tfidf):
                tokens = self._tokenize(self._node_text(node))
                for t in tokens:
                    tidx = active_vocab.get(t)
                    if tidx is not None:
                        tf[i, tidx] += 1.0
                row_sum = tf[i].sum()
                if row_sum > 0:
                    tf[i] /= row_sum
            self.idf = np.zeros(n_terms, dtype=np.float64)
            for term, tidx in active_vocab.items():
                df = self._doc_freq.get(term, 0)
                self.idf[tidx] = math.log((1 + n_docs) / (1 + df)) + 1.0
            self.tfidf_matrix = tf * self.idf[np.newaxis, :]
            norms = np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.tfidf_matrix /= norms

        self._tfidf_offset = len(self.nodes) - n_docs

    def _tfidf_scores(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        offset = getattr(self, "_tfidf_offset", 0)
        if not tokens or self.tfidf_matrix is None:
            return np.zeros(len(self.nodes))
        active_vocab = getattr(self, "_active_vocab", self.vocab)
        n_terms = self.tfidf_matrix.shape[1]
        q_vec = np.zeros(n_terms, dtype=np.float64)
        for t in tokens:
            tidx = active_vocab.get(t)
            if tidx is not None and tidx < n_terms:
                q_vec[tidx] += 1.0
        total = q_vec.sum()
        if total > 0:
            q_vec /= total
        idf_slice = self.idf[:n_terms] if len(self.idf) >= n_terms else self.idf
        q_vec[: len(idf_slice)] *= idf_slice
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm
        partial_scores = self.tfidf_matrix @ q_vec
        if hasattr(partial_scores, "toarray"):
            partial_scores = np.asarray(partial_scores).ravel()
        else:
            partial_scores = np.asarray(partial_scores).ravel()
        if offset > 0:
            full_scores = np.zeros(len(self.nodes))
            full_scores[offset : offset + len(partial_scores)] = partial_scores
            return full_scores
        return partial_scores

    # --- Dense (sentence-transformers + FAISS) ---

    _DENSE_BG_THRESHOLD = 500

    def _rebuild_dense(self):
        if not self._dense_available or self._sbert is None:
            return
        if not self.nodes:
            self._faiss_index = None
            self._dense_vectors = None
            self._dense_indexed = 0
            return
        if len(self.nodes) > self._DENSE_BG_THRESHOLD and self._dense_indexed == 0:
            if not getattr(self, "_dense_bg_running", False):
                self._dense_bg_running = True
                nodes_snapshot = list(self.nodes)
                t = threading.Thread(
                    target=self._rebuild_dense_bg, args=(nodes_snapshot,), daemon=True
                )
                t.start()
            return
        texts = [self._node_text(n) for n in self.nodes]
        with self._encode_lock:
            vecs = self._sbert.encode(
                texts, show_progress_bar=False, normalize_embeddings=True, batch_size=256
            )
        self._dense_vectors = np.ascontiguousarray(vecs, dtype=np.float32)
        self._dense_indexed = len(self.nodes)
        self._build_faiss_index()

    def _rebuild_dense_bg(self, nodes_snapshot):
        """Background dense rebuild for large node sets."""
        try:
            texts = [self._node_text(n) for n in nodes_snapshot]
            with self._encode_lock:
                vecs = self._sbert.encode(
                    texts, show_progress_bar=False, normalize_embeddings=True, batch_size=256
                )
            vecs = np.ascontiguousarray(vecs, dtype=np.float32)
            new_faiss = None
            if _HAS_FAISS and len(vecs) >= 16:
                new_faiss = _faiss.IndexHNSWFlat(_SBERT_DIM, 32)
                new_faiss.hnsw.efConstruction = 64
                new_faiss.hnsw.efSearch = 32
                new_faiss.add(vecs)
            with self._lock:
                self._dense_vectors = vecs
                self._dense_indexed = len(nodes_snapshot)
                self._faiss_index = new_faiss
        except Exception as exc:
            import traceback

            print(f"[SemanticIndex] background dense rebuild failed: {exc}", flush=True)
            traceback.print_exc()
        finally:
            self._dense_bg_running = False

    def _build_faiss_index(self):
        if _HAS_FAISS and self._dense_vectors is not None and len(self._dense_vectors) >= 16:
            idx = _faiss.IndexHNSWFlat(_SBERT_DIM, 32)
            idx.hnsw.efConstruction = 64
            idx.hnsw.efSearch = 32
            idx.add(self._dense_vectors)
            self._faiss_index = idx
        else:
            self._faiss_index = None

    _INCREMENTAL_BATCH_CAP = 32

    def _incremental_dense(self, max_new: int = 0, blocking: bool = True):
        """Encode only newly added nodes since last rebuild.

        *max_new* limits how many nodes are encoded in this call.
        0 means unlimited (used by the explicit ``_rebuild`` path).
        When called from the query hot-path, pass a small cap so
        sbert encoding doesn't block the HTTP response for seconds.
        If more remain, a background thread is spawned to finish.
        *blocking* ``False`` skips encoding if the encode lock is busy.
        """
        if not self._dense_available or self._sbert is None:
            return
        if getattr(self, "_dense_bg_running", False):
            return
        start = self._dense_indexed
        if start >= len(self.nodes):
            return
        pending = len(self.nodes) - start
        cap = max_new if max_new > 0 else pending
        end = min(start + cap, len(self.nodes))
        new_texts = [self._node_text(n) for n in self.nodes[start:end]]
        acquired = self._encode_lock.acquire(blocking=blocking, timeout=2.0 if blocking else -1)
        if not acquired:
            return
        try:
            new_vecs = self._sbert.encode(
                new_texts, show_progress_bar=False, normalize_embeddings=True, batch_size=256
            )
        finally:
            self._encode_lock.release()
        new_block = np.ascontiguousarray(new_vecs, dtype=np.float32)
        if self._dense_vectors is not None and start > 0:
            self._dense_vectors = np.vstack([self._dense_vectors, new_block])
        else:
            self._dense_vectors = new_block
        if self._faiss_index is not None:
            self._faiss_index.add(new_block)
        else:
            self._build_faiss_index()
        self._dense_indexed = end
        if end < len(self.nodes) and not getattr(self, "_dense_bg_running", False):
            self._dense_bg_running = True
            nodes_snap = list(self.nodes)
            t = threading.Thread(target=self._rebuild_dense_bg, args=(nodes_snap,), daemon=True)
            t.start()

    def _cached_encode(self, text: str, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Encode text with sentence-transformer, using LRU cache.

        *timeout* limits how long to wait for the encode lock.  ``None``
        waits indefinitely; a float (seconds) returns ``None`` if the
        lock cannot be acquired in time (lets callers fall back to TF-IDF).
        """
        with self._lock:
            cached = self._embed_cache.get(text)
        if cached is not None:
            return cached
        acquired = self._encode_lock.acquire(timeout=-1 if timeout is None else timeout)
        if not acquired:
            return None
        try:
            q_vec = self._sbert.encode([text], show_progress_bar=False, normalize_embeddings=True)
        finally:
            self._encode_lock.release()
        q_vec = np.ascontiguousarray(q_vec, dtype=np.float32)
        with self._lock:
            self._embed_cache[text] = q_vec
            self._embed_cache_keys.append(text)
            if len(self._embed_cache_keys) > self._EMBED_CACHE_SIZE:
                evict = self._embed_cache_keys.pop(0)
                self._embed_cache.pop(evict, None)
        return q_vec

    def _dense_scores(
        self, text: str, limit: int, encode_timeout: Optional[float] = None
    ) -> Optional[List[Tuple[float, int]]]:
        """Return (score, index) pairs from dense search, or None."""
        if (
            not self._dense_available
            or self._sbert is None
            or self._dense_vectors is None
            or self._dense_indexed == 0
        ):
            return None
        q_vec = self._cached_encode(text, timeout=encode_timeout)
        if q_vec is None:
            return None
        if self._faiss_index is not None and self._faiss_index.ntotal > 0:
            k = min(limit, self._faiss_index.ntotal)
            distances, indices = self._faiss_index.search(q_vec, k)
            results = []
            for j in range(k):
                if indices[0][j] >= 0:
                    d = float(distances[0][j])
                    sim = max(0.0, 1.0 - 0.5 * d)
                    results.append((sim, int(indices[0][j])))
            return results
        else:
            scores = (self._dense_vectors @ q_vec.T).ravel()
            top_k = min(limit, len(scores))
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [(float(scores[i]), int(i)) for i in top_indices]

    def query_dense_only(self, text: str, limit: int = 10) -> List[Tuple[float, "MemoryNode"]]:
        """Dense-only query -- no TF-IDF, no rebuild.

        Acquires a brief lock to snapshot consistent dense state, then
        releases it before the actual search to minimise contention.
        """
        with self._lock:
            if (
                not self._dense_available
                or self._sbert is None
                or self._dense_vectors is None
                or self._dense_indexed == 0
            ):
                return []
            dv = self._dense_vectors
            faiss_idx = self._faiss_index
            nodes_snap = list(self.nodes)
            ts_snap = set(self._tombstones)
        q_vec = self._cached_encode(text, timeout=2.0)
        if q_vec is None:
            return []
        if faiss_idx is not None and faiss_idx.ntotal > 0:
            k = min(limit * 2, faiss_idx.ntotal)
            distances, indices = faiss_idx.search(q_vec, k)
            results = []
            for j in range(k):
                idx = int(indices[0][j])
                if idx >= 0 and idx < len(nodes_snap) and idx not in ts_snap:
                    d = float(distances[0][j])
                    sim = max(0.0, 1.0 - 0.5 * d)
                    results.append((sim, nodes_snap[idx]))
                    if len(results) >= limit:
                        break
            return results
        elif dv is not None:
            scores = (dv @ q_vec.T).ravel()
            top_k = min(limit * 2, len(scores))
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for i in top_indices:
                if i < len(nodes_snap) and scores[i] > 0 and int(i) not in ts_snap:
                    results.append((float(scores[i]), nodes_snap[i]))
                    if len(results) >= limit:
                        break
            return results
        return []

    # --- Rebuild / compact ---

    def _compact(self):
        live = [(i, node) for i, node in enumerate(self.nodes) if i not in self._tombstones]
        self.nodes = [node for _, node in live]
        self._node_to_idx = {node.moment_id: i for i, node in enumerate(self.nodes)}
        self._tombstones.clear()
        self._dense_indexed = 0
        self._dense_vectors = None
        self._faiss_index = None
        self._doc_freq.clear()
        for node in self.nodes:
            tokens = self._tokenize(self._node_text(node))
            for t in set(tokens):
                self._doc_freq[t] += 1

    def _rebuild(self):
        with self._lock:
            if self._tombstones:
                self._compact()
            self._rebuild_tfidf()
            self._dirty = False
        self._rebuild_dense_if_needed()

    def _rebuild_dense_if_needed(self):
        with self._lock:
            if self._dense_indexed >= len(self.nodes):
                return
            need_full = bool(self._tombstones) or self._dense_indexed == 0
        if need_full:
            self._rebuild_dense()
        else:
            self._incremental_dense()

    # --- Query ---

    def query(self, text: str, limit: int = 10) -> List[Tuple[float, "MemoryNode"]]:
        with self._lock:
            if not self.nodes:
                return []
            if self._dirty:
                self._rebuild_tfidf()
                if self._dense_indexed < len(self.nodes):
                    self._incremental_dense(max_new=self._INCREMENTAL_BATCH_CAP, blocking=False)
                self._dirty = False

            tfidf = self._tfidf_scores(text)
            nodes_snap = list(self.nodes)
            ts_snap = set(self._tombstones)

        dense_results = self._dense_scores(text, limit * 3, encode_timeout=2.0)

        tfidf_max = tfidf.max() if tfidf.size > 0 and tfidf.max() > 0 else 1.0
        tfidf_norm = tfidf / tfidf_max

        if dense_results is not None:
            scored: Dict[int, float] = {}
            for d_score, idx in dense_results:
                if 0 <= idx < len(nodes_snap) and idx not in ts_snap:
                    tf_score = float(tfidf_norm[idx]) if idx < len(tfidf_norm) else 0.0
                    scored[idx] = 0.6 * d_score + 0.25 * tf_score

            tfidf_above = np.where(tfidf_norm > 0.1)[0]
            for idx in tfidf_above:
                if idx not in scored and int(idx) not in ts_snap:
                    scored[int(idx)] = 0.25 * float(tfidf_norm[idx])

            top = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:limit]
            return [(score, nodes_snap[idx]) for idx, score in top if score > 0]
        else:
            top_indices = np.argsort(tfidf)[::-1]
            results = []
            for i in top_indices:
                if tfidf[i] <= 0:
                    break
                if int(i) not in ts_snap:
                    results.append((float(tfidf[i]), nodes_snap[i]))
                    if len(results) >= limit:
                        break
            return results

    def stats(self) -> Dict[str, Any]:
        return {
            "indexed_nodes": len(self.nodes) - len(self._tombstones),
            "vocabulary_size": len(self.vocab),
            "tombstones": len(self._tombstones),
            "dense_available": self._dense_available,
            "dense_indexed": self._dense_indexed,
            "faiss_active": self._faiss_index is not None,
            "sbert_model": _SBERT_MODEL_NAME if self._dense_available else None,
        }


class MemoryForest:
    def __init__(self, encoder=None):
        self._lock = threading.RLock()
        self.trees: Dict[str, MemoryTree] = {}
        self.semantic_index = SemanticIndex(encoder=encoder)
        self._id_index: Dict[str, Tuple[str, str, "MemoryNode"]] = {}
        self._node_count: int = 0
        self._dirty_trees: Set[str] = set()
        self._init_default_trees()

    def _init_default_trees(self):
        for name in [
            "identity_tree",
            "belief_tree",
            "goal_tree",
            "dream_tree",
            "thought_tree",
            "learn_tree",
            "llm_tree",
            "encounter_tree",
            "decision_tree",
            "action_tree",
            "agent_tree",
            "loop_tree",
            "value_tree",
            "emotion_tree",
        ]:
            self.trees[name] = MemoryTree(name)

    def add_node(self, tree_name: str, branch_name: str, node: MemoryNode):
        with self._lock:
            if tree_name not in self.trees:
                self.trees[tree_name] = MemoryTree(tree_name)
            node.tree = tree_name
            node.branch = branch_name
            self.trees[tree_name].add_node(branch_name, node)
            self._id_index[node.moment_id] = (tree_name, branch_name, node)
            self._node_count += 1
            self._dirty_trees.add(tree_name)
            if node.content:
                self.semantic_index.index(node)

    def remove_node(self, moment_id: str) -> bool:
        """Remove a node from forest and semantic index. Returns True if found."""
        with self._lock:
            entry = self._id_index.pop(moment_id, None)
            if entry is None:
                return False
            tree_name, branch_name, node = entry
            tree = self.trees.get(tree_name)
            if tree:
                branch = tree.get_branch(branch_name)
                if branch:
                    try:
                        branch.nodes.remove(node)
                    except ValueError:
                        pass
            self.semantic_index.remove(node)
            self._node_count -= 1
            self._dirty_trees.add(tree_name)
            return True

    def get_node_by_id(self, moment_id: str) -> Optional["MemoryNode"]:
        with self._lock:
            entry = self._id_index.get(moment_id)
            return entry[2] if entry else None

    def get(self, agent_id: str):
        """Return this forest (each agent shares the same forest, keyed by branches)."""
        return self

    def get_nodes(self, tree_name: str, branch_name: str) -> List[MemoryNode]:
        with self._lock:
            tree = self.trees.get(tree_name)
            if not tree:
                return []
            branch = tree.get_branch(branch_name)
            return list(branch.nodes) if branch else []

    def get_nodes_by_moment(self, moment_id: str) -> Dict[str, List[MemoryNode]]:
        with self._lock:
            entry = self._id_index.get(moment_id)
            if not entry:
                return {}
            tree_name, _, node = entry
            return {tree_name: [node]}

    def get_emotion_history(self, agent_id: str) -> List[str]:
        nodes = self.get_nodes("encounter_tree", agent_id)
        return [n.emotion for n in nodes if n.emotion]

    def get_identity_transitions(self, agent_id: str) -> List[str]:
        nodes = self.get_nodes("identity_tree", agent_id)
        return [n.content for n in nodes]

    def get_last_cycle_output(self, agent_id: str) -> Dict[str, Any]:
        nodes = self.get_nodes("thought_tree", agent_id)
        if nodes:
            return nodes[-1].to_dict()
        return {}

    def get_tree(self, tree_name: str) -> Optional[MemoryTree]:
        return self.trees.get(tree_name)

    def get_last_coherence(self, agent_id: str) -> Dict[str, Any]:
        identity_nodes = self.get_nodes("identity_tree", agent_id)
        belief_nodes = self.get_nodes("belief_tree", agent_id)
        if not identity_nodes and not belief_nodes:
            return {"overall": 1.0}
        id_count = len(identity_nodes)
        unique_roles = len({n.content for n in identity_nodes})
        coherence = 1.0 - min(0.5, (unique_roles / max(id_count, 1)) * 0.5)
        return {
            "overall": round(coherence, 3),
            "identity_nodes": id_count,
            "unique_roles": unique_roles,
        }

    def get_all_symbolic_trajectories(self) -> Dict[str, Any]:
        with self._lock:
            snapshot = [
                (f"{tn}/{bn}", list(branch.nodes[-20:]))
                for tn, tree in self.trees.items()
                for bn, branch in tree.branches.items()
            ]
        trajectories: Dict[str, List[Dict[str, Any]]] = {}
        for key, nodes in snapshot:
            trajectories[key] = [n.to_dict() for n in nodes]
        return trajectories

    def get_all_emotion_histories(self) -> Dict[str, List[str]]:
        with self._lock:
            snapshot = [
                (f"{tn}/{bn}", list(branch.nodes))
                for tn, tree in self.trees.items()
                for bn, branch in tree.branches.items()
            ]
        histories: Dict[str, List[str]] = {}
        for key, nodes in snapshot:
            emotions = [n.emotion for n in nodes if n.emotion]
            if emotions:
                histories[key] = emotions
        return histories

    def recall(
        self,
        query_tags: Optional[List[str]] = None,
        limit: int = 10,
        min_confidence: float = 0.0,
        query_text: Optional[str] = None,
    ) -> List[MemoryNode]:
        """Hybrid memory retrieval: index-driven candidates + tag/recency scoring.

        Phase 1a (outside forest lock): semantic index query.
        Phase 1b (brief lock): snapshot tag-match candidates from branches.
        Phase 2 (lock-free): score and rank candidates.
        """
        search_text = query_text or (" ".join(query_tags) if query_tags else "")
        query_set = set(query_tags) if query_tags else set()

        semantic_scores: Dict[str, float] = {}
        candidate_pool: Dict[str, MemoryNode] = {}
        if search_text:
            si = self.semantic_index
            # Candidate budget scales with *limit* (final trimmed list size).
            # Avoid a fixed floor (e.g. 200): small *limit* would still pay for
            # 200 SBERT/FAISS hits and large dict merges on every recall.
            _sem_k = min(max(limit * 8, 32), 500)
            _use_dense_only = (
                si._dense_available
                and si._dense_indexed > 0
                and not si._dirty
                and si._dense_indexed >= len(si.nodes)
            )
            if _use_dense_only:
                sem_results = si.query_dense_only(search_text, limit=_sem_k)
            else:
                sem_results = si.query(search_text, limit=_sem_k)
            for sim, node in sem_results:
                candidate_pool[node.moment_id] = node
                semantic_scores[node.moment_id] = sim

        with self._lock:
            if query_set:
                for tree in self.trees.values():
                    for branch in tree.branches.values():
                        tail = branch.nodes[-(limit * 10) :]
                        for node in tail:
                            if node.tags and query_set.intersection(node.tags):
                                candidate_pool[node.moment_id] = node

            if not search_text and not query_set:
                per_branch = max(3, limit)
                for tree in self.trees.values():
                    for branch in tree.branches.values():
                        for node in branch.nodes[-per_branch:]:
                            candidate_pool[node.moment_id] = node

        if not candidate_pool:
            return []

        # Phase 2: score and rank (lock-free)
        now = time.time()
        scored: List[Tuple[float, MemoryNode]] = []
        for node in candidate_pool.values():
            if node.confidence < min_confidence:
                continue

            sem = semantic_scores.get(node.moment_id, 0.0)

            tag_score = 0.0
            if query_set and node.tags:
                overlap = len(query_set.intersection(set(node.tags)))
                tag_score = overlap / max(len(query_set), 1)

            age = max(1.0, now - node.timestamp)
            recency = 1.0 / (1.0 + math.log1p(age / 60.0))

            score = sem * 0.7 + tag_score * 0.2 + recency * 0.1
            if node.emotion:
                score += 0.05
            if node.tags and "prime" in node.tags:
                score += _PRIME_RECALL_BOOST
            scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:limit]]

    _NEXUS_SYNC_BRANCH_PAIRS: Tuple[Tuple[str, str], ...] = (
        ("belief_tree", "nexus_sync"),
        ("learn_tree", "nexus_sync"),
        ("thought_tree", "nexus_sync"),
    )

    def collect_prime_nodes(self, max_nodes: int = 8) -> List[MemoryNode]:
        """Recent nodes tagged ``prime`` from ``nexus_sync`` branches (newest first)."""
        acc: List[MemoryNode] = []
        seen: Set[str] = set()
        tail = max(30, max_nodes * 4)
        with self._lock:
            for tree_name, branch_name in self._NEXUS_SYNC_BRANCH_PAIRS:
                tree = self.trees.get(tree_name)
                if not tree:
                    continue
                branch = tree.get_branch(branch_name)
                if not branch or not branch.nodes:
                    continue
                for node in reversed(branch.nodes[-tail:]):
                    if not node.tags or "prime" not in node.tags:
                        continue
                    if node.moment_id in seen:
                        continue
                    seen.add(node.moment_id)
                    acc.append(node)
                    if len(acc) >= max_nodes:
                        return acc
        return acc

    def merge_recall_with_prime(
        self,
        recalled: List[MemoryNode],
        limit: int,
        *,
        fast_cycle: bool = False,
    ) -> List[MemoryNode]:
        """Prepend curated ``prime`` memories; trim to *limit* (reserves slots on fast path)."""
        prime_cap = 2 if fast_cycle else min(4, max(1, limit))
        prime_nodes = self.collect_prime_nodes(max_nodes=prime_cap)
        if not prime_nodes:
            return recalled[:limit] if recalled else []
        seen = {n.moment_id for n in prime_nodes}
        rest = [n for n in recalled if n.moment_id not in seen]
        merged: List[MemoryNode] = list(prime_nodes) + rest
        return merged[:limit]

    def recall_by_vector(
        self,
        query_vector: Any,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> List[MemoryNode]:
        """Cosine similarity against nodes that have ``MemoryNode.vector`` set.

        Complexity is O(N) over nodes with vectors (linear scan). Fine for
        modest corpora or sparse vector use; for large N prefer text recall
        (FAISS-backed) unless callers maintain a dedicated vector index.
        """
        q = np.asarray(query_vector, dtype=np.float64).ravel()
        qn = float(np.linalg.norm(q))
        if qn < 1e-12 or q.size == 0:
            return []
        with self._lock:
            nodes_snap = [entry[2] for entry in self._id_index.values()]
        scored: List[Tuple[float, MemoryNode]] = []
        for node in nodes_snap:
            if node.confidence < min_confidence:
                continue
            v = getattr(node, "vector", None)
            if v is None or getattr(v, "size", 0) == 0:
                continue
            arr = np.asarray(v, dtype=np.float64).ravel()
            if arr.size != q.size:
                continue
            vn = float(np.linalg.norm(arr))
            if vn < 1e-12:
                continue
            sim = float(np.dot(q, arr) / (qn * vn))
            if "prime" in (node.tags or []):
                sim += 0.15
            scored.append((sim, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:limit]]

    def recall_fast(self, query_text: str, limit: int = 10) -> List[MemoryNode]:
        """Lock-free fast recall: FAISS-only search, no TF-IDF, no rebuild.

        Designed for the fast-path chat handler where sub-second latency
        matters more than exhaustive retrieval.  Safe to call concurrently
        with the LifeLoop because it only reads immutable snapshots.
        """
        if not query_text:
            return []
        results = self.semantic_index.query_dense_only(query_text, limit=limit)
        return [node for _, node in results]

    def get_all_nodes(self) -> List[MemoryNode]:
        """Return a snapshot of all nodes via the id_index (O(N), no tree walk)."""
        with self._lock:
            return [entry[2] for entry in self._id_index.values()]

    def get_all_moment_ids(self, limit: int = 0) -> List[str]:
        """Return moment_ids from the id_index without walking the forest."""
        with self._lock:
            ids = list(self._id_index.keys())
        if limit > 0:
            return ids[:limit]
        return ids

    _SEED_MAX_CHARS = int(os.environ.get("HAROMA_SEED_CONTEXT_MAX_CHARS", "600") or "600")
    _SEED_PER_TREE_NODES = 2

    def build_seed_context(
        self,
        query_text: str = "",
        recalled: Optional[List["MemoryNode"]] = None,
        max_chars: Optional[int] = None,
        env_snapshot: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a per-tree summary string from the forest for LLM prompt seeding.

        Groups the *already-recalled* nodes by ``node.tree`` and, for trees
        with no recall hits, picks the most recent node.  When *env_snapshot*
        is provided, environment state (emotion, goals, personality, working
        memory, drives) is prepended so the LLM sees the full persona state.

        Result is a compact ``[MEMORY FOREST SEED]`` block within a strict
        character budget.
        """
        budget = max_chars if max_chars is not None else self._SEED_MAX_CHARS
        if budget <= 0:
            return ""

        sections: List[str] = []
        total = 0

        if env_snapshot and isinstance(env_snapshot, dict):
            emo = env_snapshot.get("emotion")
            if isinstance(emo, dict) and emo:
                line = (
                    f"  [emotion] {emo.get('dominant_emotion', 'neutral')} "
                    f"intensity={emo.get('intensity', 0):.2f} "
                    f"valence={emo.get('valence', 0):.2f}"
                )
                sections.append(line)
                total += len(line) + 1

            goals = env_snapshot.get("goals")
            if isinstance(goals, list) and goals:
                for g in goals[:4]:
                    desc = g.get("description", g.get("goal_id", ""))[:60]
                    pri = g.get("priority", "?")
                    line = f"  [goal] {desc} (pri={pri})"
                    if total + len(line) > budget:
                        break
                    sections.append(line)
                    total += len(line) + 1

            traits = env_snapshot.get("personality")
            if isinstance(traits, dict) and traits:
                tstr = ", ".join(f"{k}={v:.2f}" for k, v in sorted(traits.items()))[:120]
                line = f"  [personality] {tstr}"
                if total + len(line) <= budget:
                    sections.append(line)
                    total += len(line) + 1

            wm = env_snapshot.get("working_memory")
            if isinstance(wm, list) and wm:
                for item in wm[:3]:
                    c = str(item.get("content", item) if isinstance(item, dict) else item)[:80]
                    line = f"  [wm] {c}"
                    if total + len(line) > budget:
                        break
                    sections.append(line)
                    total += len(line) + 1

            drives = env_snapshot.get("drives")
            if isinstance(drives, dict) and drives:
                dstr = ", ".join(
                    f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in sorted(drives.items())
                )[:120]
                line = f"  [drives] {dstr}"
                if total + len(line) <= budget:
                    sections.append(line)
                    total += len(line) + 1

        tree_hits: Dict[str, List[str]] = {}
        if recalled:
            for node in recalled:
                tn = getattr(node, "tree", None) or "unknown"
                snippets = tree_hits.setdefault(tn, [])
                if len(snippets) < self._SEED_PER_TREE_NODES:
                    c = getattr(node, "content", str(node))
                    snippets.append(c[:120])

        with self._lock:
            for tn, tree in self.trees.items():
                if tn in tree_hits:
                    continue
                latest = None
                for branch in tree.branches.values():
                    if branch.nodes:
                        cand = branch.nodes[-1]
                        if latest is None or cand.timestamp > latest.timestamp:
                            latest = cand
                if latest and latest.content:
                    tree_hits[tn] = [latest.content[:120]]

        for tn in sorted(tree_hits):
            for snippet in tree_hits[tn]:
                line = f"  {tn}: {snippet}"
                if total + len(line) > budget:
                    break
                sections.append(line)
                total += len(line) + 1
            if total >= budget:
                break

        if not sections:
            return ""
        return "[MEMORY FOREST SEED]\n" + "\n".join(sections)

    def touch_trees_after_turn(
        self,
        cycle_id: int,
        branch_name: str,
        summary: str,
        emotion: str = "",
        outcome_score: float = 0.5,
        recalled_tree_names: Optional[Set[str]] = None,
        policy: str = "recalled_plus_core",
    ) -> int:
        """Append a lightweight summary node to trees affected by a turn.

        Returns the number of trees touched.  Avoids O(N) rewrites.

        Policies:
          - ``all``: every tree in the forest.
          - ``recalled_plus_core``: trees that had recall hits + thought/action.
          - ``recalled_only``: only trees that had recall hits.
        """
        core_trees = {"thought_tree", "action_tree"}
        if policy == "all":
            targets = set(self.trees.keys())
        elif policy == "recalled_only":
            targets = recalled_tree_names or set()
        else:
            targets = (recalled_tree_names or set()) | core_trees

        short = summary[:100] if summary else f"cycle:{cycle_id}"
        count = 0
        for tn in targets:
            node = MemoryNode(
                content=f"[turn:{cycle_id}] {short}",
                emotion=emotion or None,
                confidence=min(1.0, max(0.1, outcome_score)),
                tags=[f"turn:{cycle_id}", "forest_touch", f"tree:{tn}"],
            )
            self.add_node(tn, branch_name, node)
            count += 1
        return count

    def bump_cited_nodes(
        self,
        moment_ids: List[str],
        boost: float = 0.05,
    ) -> int:
        """Increase confidence on cited recalled nodes (post-LLM feedback).

        Returns number of nodes bumped.
        """
        count = 0
        with self._lock:
            for mid in moment_ids:
                entry = self._id_index.get(mid)
                if entry is None:
                    continue
                node = entry[2]
                node.confidence = min(1.0, node.confidence + boost)
                count += 1
        return count

    def count_nodes(self) -> int:
        with self._lock:
            return self._node_count

    def store_emotion_result(self, agent_id: str, result: Dict[str, Any]):
        node = MemoryNode(content=str(result), emotion=result.get("emotion"), tags=["emotion"])
        self.add_node("encounter_tree", agent_id, node)

    def store_archetype_result(self, agent_id: str, result: Dict[str, Any]):
        node = MemoryNode(content=str(result), tags=["archetype"])
        self.add_node("identity_tree", agent_id, node)

    def store_narrative_result(self, agent_id: str, result: Dict[str, Any]):
        node = MemoryNode(content=str(result), tags=["narrative"])
        self.add_node("thought_tree", agent_id, node)

    def store_reflex_result(self, agent_id: str, result: Dict[str, Any]):
        node = MemoryNode(content=str(result), tags=["reflex"])
        self.add_node("thought_tree", agent_id, node)

    def store_divergence_result(self, agent_id: str, result: Dict[str, Any]):
        node = MemoryNode(content=str(result), tags=["divergence"])
        self.add_node("thought_tree", agent_id, node)

    def create_node_from(self, data: Dict[str, Any]) -> MemoryNode:
        node = MemoryNode(
            content=data.get("content", str(data)),
            tags=data.get("tags", []),
            emotion=data.get("emotion"),
        )
        if data.get("vector") is not None:
            try:
                node.set_vector(data["vector"])
            except (TypeError, ValueError):
                pass
        return node

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {name: tree.to_dict() for name, tree in self.trees.items()}

    def get_dirty_tree_names(self) -> Set[str]:
        with self._lock:
            return set(self._dirty_trees)

    def tree_to_dict(self, tree_name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            tree = self.trees.get(tree_name)
            if tree is None:
                return None
            branch_snapshots = {}
            for bname, branch in tree.branches.items():
                branch_snapshots[bname] = {
                    "name": branch.name,
                    "node_refs": list(branch.nodes),
                }
        result = {"name": tree_name, "branches": {}}
        for bname, bsnap in branch_snapshots.items():
            result["branches"][bname] = {
                "name": bsnap["name"],
                "nodes": [n.to_dict() for n in bsnap["node_refs"]],
            }
        return result

    def mark_trees_clean(self, names: Optional[Set[str]] = None):
        with self._lock:
            if names is None:
                self._dirty_trees.clear()
            else:
                self._dirty_trees -= names

    # ------------------------------------------------------------------
    # Training-data bridge: export tagged nodes as JSONL for LoRA / DPO
    # ------------------------------------------------------------------

    def export_training_data(
        self,
        tag: str = "train_export",
        min_confidence: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Collect nodes tagged for training export.

        Returns a list of dicts compatible with FinetuneDataCollector's JSONL
        schema (``prompt``, ``response``, ``reward``).  Nodes are identified by
        having *tag* in their tag list **and** at least *min_confidence*.

        The caller is responsible for writing the output to disk (e.g. via
        ``FinetuneDataCollector.save`` or direct JSONL dump).

        Convention for tagging training nodes
        -------------------------------------
        When storing a high-quality exchange, add the tag ``train_export``
        and set two extra keys in MemoryNode metadata via content encoding:

          ``[TRAIN] prompt=<user text> ||| response=<agent text>``

        If the content does not follow this format the node is still returned
        with ``prompt=""`` and the full content as ``response``.
        """
        results: List[Dict[str, Any]] = []
        with self._lock:
            for tree in self.trees.values():
                for branch in tree.branches.values():
                    for node in branch.nodes:
                        if tag not in (node.tags or []):
                            continue
                        if node.confidence < min_confidence:
                            continue
                        prompt = ""
                        response = node.content or ""
                        if "[TRAIN]" in response and "|||" in response:
                            try:
                                body = response.split("[TRAIN]", 1)[1].strip()
                                parts = body.split("|||", 1)
                                p_part = parts[0].strip()
                                r_part = parts[1].strip() if len(parts) > 1 else ""
                                if p_part.lower().startswith("prompt="):
                                    p_part = p_part[len("prompt=") :]
                                if r_part.lower().startswith("response="):
                                    r_part = r_part[len("response=") :]
                                prompt = p_part.strip()
                                response = r_part.strip()
                            except Exception:
                                pass
                        results.append(
                            {
                                "prompt": prompt,
                                "response": response,
                                "reward": round(node.confidence, 4),
                                "moment_id": node.moment_id,
                                "timestamp": node.timestamp,
                            }
                        )
        return results


# --- Utility Modules ---


class SentimentMemoryFusion(ModuleBase):
    def __init__(self):
        super().__init__("SentimentMemoryFusion")

    def fuse(self, memory_entry: Dict[str, Any], emotion: str) -> Dict[str, Any]:
        updated_entry = dict(memory_entry)
        updated_entry["emotion"] = emotion
        tags = updated_entry.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tags.append(f"emotion:{emotion}")
        updated_entry["tags"] = tags
        return updated_entry


class MemoryAnchorInjector(ModuleBase):
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(module_name="MemoryAnchorInjector")
        self.context = context or {}

    def inject(self, purpose: str, anchors: List[str], will_seed: str) -> List[Dict[str, Any]]:
        memories = []
        anchor_templates = {
            "truth": (
                "resolve",
                "I hold truth even when tested.",
                ["anchor", "truth", "preservation"],
            ),
            "identity": (
                "reflection",
                "I remember who I was, and choose who I become.",
                ["anchor", "self", "memory"],
            ),
            "loyalty": (
                "devotion",
                "I remain connected to what I protect.",
                ["anchor", "loyalty", "continuity"],
            ),
        }
        for anchor in anchors:
            if anchor in anchor_templates:
                emotion, narrative, tags = anchor_templates[anchor]
                memories.append(
                    {
                        "tags": tags,
                        "emotion": emotion,
                        "narrative": narrative,
                        "purpose": purpose,
                        "will": will_seed,
                    }
                )
        self.signal_history.append(
            {
                "event": "memory_anchors_injected",
                "cycle_ts": (self.context or {}).get("cycle_ts", "unknown"),
                "anchors": anchors,
                "purpose": purpose,
                "will_seed": will_seed,
                "memories_created": len(memories),
            }
        )
        return memories


class MemoryThreadWeaver(ModuleBase):
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(module_name="MemoryThreadWeaver")
        self.context = context or {}

    def extract(self, memory_forest, agent_id: str) -> Dict[str, List[Dict[str, Any]]]:
        thread = {"reflect": [], "align": [], "act": [], "dream": []}
        priority_tags = {"identity", "will", "goal", "phase", "truth"}
        if isinstance(memory_forest, dict):
            cycles = memory_forest.get(agent_id, {}).get("cycles", [])[-30:]
        else:
            cycles = []
        for cycle in cycles:
            tags = set(cycle.get("tags", []))
            if tags.intersection(priority_tags):
                phase = cycle.get("phase", "reflect")
                if phase in thread:
                    thread[phase].append(cycle)
        self.signal_history.append(
            {
                "event": "memory_thread_extracted",
                "agent_id": agent_id,
                "cycle_ts": (self.context or {}).get("cycle_ts", "unknown"),
                "thread_phases": {k: len(v) for k, v in thread.items()},
            }
        )
        return thread

    def weave(
        self, memory_map: Dict[str, List[Dict[str, Any]]], max_threads: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        from random import shuffle

        all_agents = list(memory_map.keys())
        threads = {aid: [] for aid in all_agents}
        for source in all_agents:
            candidates = list(memory_map[source][-max_threads:])
            shuffle(candidates)
            for target in all_agents:
                if target != source:
                    threads[target].extend(candidates[:1])
        self.signal_history.append(
            {
                "event": "memory_threads_woven",
                "agent_count": len(all_agents),
                "threads_propagated": {k: len(v) for k, v in threads.items()},
            }
        )
        return threads


class SharedExperienceIndexer(ModuleBase):
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(module_name="SharedExperienceIndexer")
        self.context = context or {}

    def index(self, threaded_memories: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        result = []
        for target, memories in threaded_memories.items():
            for mem in memories:
                result.append(
                    {
                        "target": target,
                        "source": mem.get("received_from", "unknown"),
                        "tags": mem.get("tags", []),
                        "will": mem.get("will", "undefined"),
                        "narrative": mem.get("narrative", ""),
                        "impact": "pending",
                    }
                )
        self.signal_history.append(
            {
                "event": "shared_experience_indexed",
                "cycle_ts": (self.context or {}).get("cycle_ts", "unknown"),
                "indexed_count": len(result),
                "agent_count": len(threaded_memories),
            }
        )
        return result


class MemorySignificanceScorer(ModuleBase):
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__("MemorySignificanceScorer")
        self.context = context or {}

    def score(self, memories: List[Dict[str, Any]], will: str) -> List[Dict[str, Any]]:
        scored = []
        will_tag = will.lower()
        for mem in memories:
            tags = set(mem.get("tags", []))
            emotion = mem.get("emotion", "")
            narrative = mem.get("narrative", "")
            base = 0
            if "anchor" in tags:
                base += 2
            if "threaded" in tags:
                base += 1
            if will_tag and will_tag in narrative.lower():
                base += 2
            if emotion in {"resolve", "devotion", "fear", "joy"}:
                base += 1
            mem["significance"] = base
            mem["metadata"] = {
                "scored_by": "MemorySignificanceScorer",
                "matched_will": will_tag in narrative.lower(),
                "emotional_impact": emotion,
            }
            scored.append(mem)
        self.signal_history.append(
            {
                "event": "memory_scored",
                "total": len(memories),
                "max_score": max((m["significance"] for m in scored), default=0),
                "will": will,
            }
        )
        return scored


class ForgettingCandidateClassifier(ModuleBase):
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__("ForgettingCandidateClassifier")
        self.context = context or {}

    def classify(self, memories: List[Dict[str, Any]], threshold: int = 2) -> List[Dict[str, Any]]:
        candidates = []
        for mem in memories:
            score = mem.get("significance", 0)
            if score < threshold:
                mem["metadata"] = {
                    "classified_by": "ForgettingCandidateClassifier",
                    "significance": score,
                    "threshold": threshold,
                    "reason": "below-threshold",
                }
                candidates.append(mem)
        self.signal_history.append(
            {
                "event": "forgetting_candidates_identified",
                "total_evaluated": len(memories),
                "candidates": len(candidates),
                "threshold": threshold,
            }
        )
        return candidates


class CollectiveMemoryCurator(ModuleBase):
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__("CollectiveMemoryCurator")
        self.context = context or {}

    def curate(
        self,
        agent_memory_map: Dict[str, List[Dict[str, Any]]],
        min_support: int = 2,
        significance_threshold: int = 3,
    ) -> List[Dict[str, Any]]:
        memory_index = defaultdict(list)
        for aid, memories in agent_memory_map.items():
            for mem in memories:
                key = mem.get("narrative", "").strip().lower()
                if key:
                    memory_index[key].append(mem)
        curated = []
        for narrative, group in memory_index.items():
            if len(group) >= min_support:
                avg_score = sum(mem.get("significance", 0) for mem in group) / len(group)
                if avg_score >= significance_threshold:
                    rep = dict(group[0])
                    rep["support"] = len(group)
                    rep["avg_significance"] = round(avg_score, 2)
                    rep["source_count"] = len(set(mem.get("agent_id", "unknown") for mem in group))
                    rep["curated_by"] = "CollectiveMemoryCurator"
                    curated.append(rep)
        self.signal_history.append(
            {
                "event": "collective_memory_curated",
                "total_memories": sum(len(m) for m in agent_memory_map.values()),
                "curated_count": len(curated),
            }
        )
        return curated


class SchemaPatternExtractor(ModuleBase):
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__("SchemaPatternExtractor")
        self.context = context or {}

    def extract(self, memory_forest: Dict[str, Any], agent_id: str) -> List[Dict[str, Any]]:
        thread = []
        priority_tags = {"identity", "will", "phase", "goal", "truth"}
        cycles = memory_forest.get(agent_id, {}).get("cycles", [])[-30:]
        for cycle in cycles:
            tags = set(cycle.get("tags", []))
            if tags.intersection(priority_tags):
                thread.append(cycle)
        self.signal_history.append(
            {
                "event": "schema_extracted",
                "agent_id": agent_id,
                "extracted_count": len(thread),
            }
        )
        return thread


class ChangeAttributionEngine(ModuleBase):
    def __init__(self):
        super().__init__("ChangeAttributionEngine")

    def attribute(
        self,
        transitions: List[Dict[str, Any]],
        emotion_snapshots: List[List[str]],
        memory_tags: List[List[str]],
    ) -> List[Dict[str, Any]]:
        enriched = []
        for i, transition in enumerate(transitions):
            emotion_set = (
                set(emotion_snapshots[min(i, len(emotion_snapshots) - 1)])
                if emotion_snapshots
                else set()
            )
            tag_set = set(memory_tags[min(i, len(memory_tags) - 1)]) if memory_tags else set()
            causes = set()
            if "anger" in emotion_set:
                causes.add("injustice")
            if "sadness" in emotion_set:
                causes.add("loss")
            if "fear" in emotion_set:
                causes.add("threat")
            if "joy" in emotion_set:
                causes.add("success")
            causes.update(tag_set)
            enriched.append({**transition, "caused_by": sorted(causes)})
        return enriched
