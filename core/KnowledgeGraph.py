"""
KnowledgeGraph — Relational world knowledge for HaromaX6.

Stores entities and typed relations extracted from NLU output.
Replaces the flat frozenset(tags) world model with structured
entity-relation knowledge that supports querying, gap detection,
state diffing, and persistence.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import threading
import time


@dataclass
class Entity:
    id: str
    name: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    mention_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "properties": self.properties,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "mention_count": self.mention_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Entity":
        return cls(
            id=d["id"],
            name=d["name"],
            entity_type=d.get("entity_type", "CONCEPT"),
            properties=d.get("properties", {}),
            first_seen=d.get("first_seen", 0.0),
            last_seen=d.get("last_seen", 0.0),
            mention_count=d.get("mention_count", 1),
        )


@dataclass
class Relation:
    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    source_cycle: int = 0
    source: str = "nlu"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "source_cycle": self.source_cycle,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Relation":
        return cls(
            subject_id=d["subject_id"],
            predicate=d["predicate"],
            object_id=d["object_id"],
            confidence=d.get("confidence", 1.0),
            timestamp=d.get("timestamp", 0.0),
            source_cycle=d.get("source_cycle", 0),
            source=d.get("source", "nlu"),
        )

    def triple_key(self) -> Tuple[str, str, str]:
        return (self.subject_id, self.predicate, self.object_id)


class KnowledgeGraph:
    def __init__(self, max_entities: int = 2000, max_relations: int = 5000):
        self._lock = threading.RLock()
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self._adjacency: Dict[str, List[int]] = defaultdict(list)
        self._relation_index: Dict[Tuple[str, str, str], int] = {}
        self.max_entities = max_entities
        self.max_relations = max_relations
        self._prev_signature: str = ""
        self._last_diff: Dict[str, Any] = {}

    @staticmethod
    def _entity_id(name: str, entity_type: str) -> str:
        raw = f"{name.lower().strip()}|{entity_type}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def integrate(self, nlu_output: Dict[str, Any], cycle_id: int = 0):
        """Merge entities and relations from an NLU parse into the graph."""
        if not nlu_output:
            return
        with self._lock:
            self._integrate_impl(nlu_output, cycle_id)

    def _integrate_impl(self, nlu_output: Dict[str, Any], cycle_id: int = 0):
        old_sig = self.compute_state_signature()
        now = time.time()

        name_to_id: Dict[str, str] = {}

        for ent_info in nlu_output.get("entities", []):
            name = ent_info.get("text", "").strip()
            etype = ent_info.get("type", "CONCEPT")
            if not name:
                continue
            eid = self._entity_id(name, etype)
            name_to_id[name.lower()] = eid

            if eid in self.entities:
                existing = self.entities[eid]
                existing.last_seen = now
                existing.mention_count += 1
                role = ent_info.get("role", "")
                if role:
                    existing.properties["last_role"] = role
            else:
                self.entities[eid] = Entity(
                    id=eid,
                    name=name,
                    entity_type=etype,
                    properties={"last_role": ent_info.get("role", "mention")},
                    first_seen=now,
                    last_seen=now,
                    mention_count=1,
                )

        for rel_info in nlu_output.get("relations", []):
            subj_name = rel_info.get("subject", "").strip()
            obj_name = rel_info.get("object", "").strip()
            pred = rel_info.get("predicate", "")
            if not subj_name or not pred:
                continue

            subj_id = name_to_id.get(subj_name.lower())
            if not subj_id:
                subj_id = self._entity_id(subj_name, "CONCEPT")
                if subj_id not in self.entities:
                    self.entities[subj_id] = Entity(
                        id=subj_id,
                        name=subj_name,
                        entity_type="CONCEPT",
                        first_seen=now,
                        last_seen=now,
                    )
                name_to_id[subj_name.lower()] = subj_id

            obj_id = ""
            if obj_name:
                obj_id = name_to_id.get(obj_name.lower())
                if not obj_id:
                    obj_id = self._entity_id(obj_name, "CONCEPT")
                    if obj_id not in self.entities:
                        self.entities[obj_id] = Entity(
                            id=obj_id,
                            name=obj_name,
                            entity_type="CONCEPT",
                            first_seen=now,
                            last_seen=now,
                        )
                    name_to_id[obj_name.lower()] = obj_id

            if not obj_id:
                continue

            triple = (subj_id, pred, obj_id)
            if triple in self._relation_index:
                idx = self._relation_index[triple]
                if idx < len(self.relations):
                    self.relations[idx].confidence = min(1.0, self.relations[idx].confidence + 0.1)
                    self.relations[idx].timestamp = now
                    self.relations[idx].source_cycle = cycle_id
                continue

            rel = Relation(
                subject_id=subj_id,
                predicate=pred,
                object_id=obj_id,
                confidence=0.8 if not rel_info.get("negated") else 0.3,
                timestamp=now,
                source_cycle=cycle_id,
                source="nlu",
            )
            idx = len(self.relations)
            self.relations.append(rel)
            self._adjacency[subj_id].append(idx)
            self._adjacency[obj_id].append(idx)
            self._relation_index[triple] = idx

        self._enforce_limits()
        new_sig = self.compute_state_signature()
        self._last_diff = self._compute_diff(old_sig, new_sig, cycle_id)
        self._prev_signature = new_sig

    def integrate_world_state(self, triples: List[Dict[str, Any]], cycle_id: int = 0):
        """Inject structured triples from the environment with high confidence."""
        if not triples:
            return
        with self._lock:
            self._integrate_world_state_impl(triples, cycle_id)

    def _integrate_world_state_impl(self, triples: List[Dict[str, Any]], cycle_id: int = 0):
        now = time.time()
        for triple in triples:
            subj_name = triple.get("subject", "").strip()
            pred = triple.get("predicate", "")
            obj_name = triple.get("object", "").strip()
            if not subj_name or not pred or not obj_name:
                continue

            conf = triple.get("confidence", 0.9)
            source = triple.get("source", "environment")

            subj_id = self._entity_id(subj_name, "ENV")
            if subj_id not in self.entities:
                self.entities[subj_id] = Entity(
                    id=subj_id, name=subj_name, entity_type="ENV", first_seen=now, last_seen=now
                )
            else:
                self.entities[subj_id].last_seen = now
                self.entities[subj_id].mention_count += 1

            obj_id = self._entity_id(obj_name, "ENV")
            if obj_id not in self.entities:
                self.entities[obj_id] = Entity(
                    id=obj_id, name=obj_name, entity_type="ENV", first_seen=now, last_seen=now
                )
            else:
                self.entities[obj_id].last_seen = now
                self.entities[obj_id].mention_count += 1

            tkey = (subj_id, pred, obj_id)
            if tkey in self._relation_index:
                idx = self._relation_index[tkey]
                if idx < len(self.relations):
                    self.relations[idx].confidence = conf
                    self.relations[idx].timestamp = now
                continue

            rel = Relation(
                subject_id=subj_id,
                predicate=pred,
                object_id=obj_id,
                confidence=conf,
                timestamp=now,
                source_cycle=cycle_id,
                source=source,
            )
            idx = len(self.relations)
            self.relations.append(rel)
            self._adjacency[subj_id].append(idx)
            self._adjacency[obj_id].append(idx)
            self._relation_index[tkey] = idx

        self._enforce_limits()

    def query_entity(self, name: str) -> Optional[Dict[str, Any]]:
        for eid, ent in self.entities.items():
            if ent.name.lower() == name.lower():
                rels = self._get_entity_relations(eid)
                return {
                    "entity": ent.to_dict(),
                    "relations": [r.to_dict() for r in rels],
                }
        return None

    def query_relations(
        self, subject: str = None, predicate: str = None, obj: str = None
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for rel in self.relations:
            if subject and not self._name_matches(rel.subject_id, subject):
                continue
            if obj and not self._name_matches(rel.object_id, obj):
                continue
            if predicate and rel.predicate != predicate:
                continue
            results.append(
                {
                    "subject": self._entity_name(rel.subject_id),
                    "predicate": rel.predicate,
                    "object": self._entity_name(rel.object_id),
                    "confidence": rel.confidence,
                    "source": rel.source,
                }
            )
        return results

    def get_neighborhood(self, entity_id: str, depth: int = 1) -> Dict[str, Any]:
        visited_entities: Set[str] = {entity_id}
        visited_relations: List[Relation] = []
        frontier = {entity_id}

        for _ in range(depth):
            next_frontier: Set[str] = set()
            for eid in frontier:
                for idx in self._adjacency.get(eid, []):
                    if idx >= len(self.relations):
                        continue
                    rel = self.relations[idx]
                    visited_relations.append(rel)
                    for neighbor in (rel.subject_id, rel.object_id):
                        if neighbor not in visited_entities:
                            visited_entities.add(neighbor)
                            next_frontier.add(neighbor)
            frontier = next_frontier

        return {
            "entities": {
                eid: self.entities[eid].to_dict()
                for eid in visited_entities
                if eid in self.entities
            },
            "relations": [r.to_dict() for r in visited_relations],
        }

    def find_gaps(self, max_gaps: int = 5) -> List[Dict[str, Any]]:
        """Entities with few relations — candidates for curiosity-driven inquiry."""
        gaps: List[Dict[str, Any]] = []
        for eid, ent in self.entities.items():
            rel_count = len(self._adjacency.get(eid, []))
            if rel_count < 2:
                gaps.append(
                    {
                        "entity": ent.name,
                        "entity_type": ent.entity_type,
                        "relation_count": rel_count,
                        "mention_count": ent.mention_count,
                        "gap_score": max(0.0, 1.0 - rel_count * 0.5),
                    }
                )
        gaps.sort(key=lambda g: (-g["gap_score"], -g["mention_count"]))
        return gaps[:max_gaps]

    def compute_state_signature(self) -> str:
        parts = sorted(f"{e.id}:{e.mention_count}" for e in self.entities.values())
        parts.extend(sorted(f"{r.subject_id}-{r.predicate}-{r.object_id}" for r in self.relations))
        raw = "|".join(parts)
        return hashlib.md5(raw.encode()).hexdigest()

    def diff(self) -> Dict[str, Any]:
        """Return the diff computed during the last integrate() call."""
        return self._last_diff

    def get_entity_names_for(self, entity_ids: List[str]) -> List[str]:
        return [self.entities[eid].name for eid in entity_ids if eid in self.entities]

    def summary(self) -> Dict[str, Any]:
        """Compact snapshot of graph state for EpisodeContext."""
        with self._lock:
            return self._summary_impl()

    def _summary_impl(self) -> Dict[str, Any]:
        top_entities = sorted(
            self.entities.values(),
            key=lambda e: e.mention_count,
            reverse=True,
        )[:10]
        predicates = set(r.predicate for r in self.relations)
        return {
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "top_entities": [e.name for e in top_entities],
            "predicate_types": sorted(predicates)[:15],
        }

    def _get_entity_relations(self, entity_id: str) -> List[Relation]:
        indices = self._adjacency.get(entity_id, [])
        return [self.relations[i] for i in indices if i < len(self.relations)]

    def _name_matches(self, entity_id: str, name: str) -> bool:
        ent = self.entities.get(entity_id)
        return ent is not None and ent.name.lower() == name.lower()

    def _entity_name(self, entity_id: str) -> str:
        ent = self.entities.get(entity_id)
        return ent.name if ent else entity_id

    def _compute_diff(self, old_sig: str, new_sig: str, cycle_id: int) -> Dict[str, Any]:
        if old_sig == new_sig:
            return {"changed": False, "new_entities": 0, "new_relations": 0, "knowledge_gain": 0.0}

        new_ents = sum(
            1
            for e in self.entities.values()
            if e.mention_count == 1 and abs(e.first_seen - e.last_seen) < 0.01
        )
        new_rels = sum(1 for r in self.relations if r.source_cycle == cycle_id)

        gain = min(1.0, (new_ents * 0.3 + new_rels * 0.2))
        return {
            "changed": True,
            "new_entities": new_ents,
            "new_relations": new_rels,
            "knowledge_gain": round(gain, 3),
        }

    def _enforce_limits(self):
        if len(self.entities) > self.max_entities:
            by_seen = sorted(self.entities.values(), key=lambda e: e.last_seen)
            to_remove = len(self.entities) - self.max_entities
            for ent in by_seen[:to_remove]:
                self._remove_entity(ent.id)

        if len(self.relations) > self.max_relations:
            by_time = sorted(range(len(self.relations)), key=lambda i: self.relations[i].timestamp)
            remove_count = len(self.relations) - self.max_relations
            remove_set = set(by_time[:remove_count])
            self._rebuild_after_removal(remove_set)

    def _remove_entity(self, entity_id: str):
        self.entities.pop(entity_id, None)
        remove_indices: Set[int] = set()
        for idx in self._adjacency.get(entity_id, []):
            remove_indices.add(idx)
        if remove_indices:
            self._rebuild_after_removal(remove_indices)
        self._adjacency.pop(entity_id, None)

    def _rebuild_after_removal(self, remove_indices: Set[int]):
        new_relations: List[Relation] = []
        old_to_new: Dict[int, int] = {}
        for old_idx, rel in enumerate(self.relations):
            if old_idx in remove_indices:
                continue
            old_to_new[old_idx] = len(new_relations)
            new_relations.append(rel)
        self.relations = new_relations

        self._adjacency.clear()
        self._relation_index.clear()
        for idx, rel in enumerate(self.relations):
            self._adjacency[rel.subject_id].append(idx)
            self._adjacency[rel.object_id].append(idx)
            self._relation_index[rel.triple_key()] = idx

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
                "relations": [r.to_dict() for r in self.relations],
            }

    def from_dict(self, data: Dict[str, Any]):
        with self._lock:
            self.entities.clear()
            self.relations.clear()
            self._adjacency.clear()
            self._relation_index.clear()

            for eid, ed in data.get("entities", {}).items():
                self.entities[eid] = Entity.from_dict(ed)

            for rd in data.get("relations", []):
                rel = Relation.from_dict(rd)
                idx = len(self.relations)
                self.relations.append(rel)
                self._adjacency[rel.subject_id].append(idx)
                self._adjacency[rel.object_id].append(idx)
                self._relation_index[rel.triple_key()] = idx

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entity_count": len(self.entities),
                "relation_count": len(self.relations),
                "unique_predicates": len(set(r.predicate for r in self.relations)),
            }
