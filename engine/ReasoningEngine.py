"""
ReasoningEngine — Symbolic + Neural reasoning over the KnowledgeGraph.

Six capabilities:
  1. Forward-chaining inference (transitivity, reciprocity, causal chains)
  2. Analogy detection via structural similarity of entity neighborhoods
  3. Goal decomposition / means-ends planning
  4. Self-modifying rule learning from KG predicate co-occurrence
  5. LLM chain-of-thought reasoning (Upgrade 7)
  6. Neural rule discovery via link-prediction MLP (Upgrade 7)

KG → symbolic law: triples using predicates ``forbids_tag`` or ``symbolic_forbids``
(object entity name becomes a forbidden tag) are stored as **external**
(societal / world-model) laws when a ``law_manager`` is wired in.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import Counter

from core.cognitive_null import is_cognitive_null
from core.engine.LawEngine import LAW_SOURCE_EXTERNAL

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except (ImportError, OSError):
    _TORCH = False


@dataclass
class Rule:
    name: str
    pattern_a: Tuple[str, str, str]  # (subj_var, pred_literal, obj_var)
    pattern_b: Tuple[str, str, str]
    conclusion: Tuple[str, str, str]  # (subj_var, pred_literal, obj_var)
    source: str = "built_in"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pattern_a": list(self.pattern_a),
            "pattern_b": list(self.pattern_b),
            "conclusion": list(self.conclusion),
            "source": self.source,
        }


@dataclass
class Analogy:
    source_entity: str
    target_entity: str
    shared_predicates: List[str]
    similarity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_entity,
            "target": self.target_entity,
            "shared": self.shared_predicates,
            "similarity": round(self.similarity, 3),
        }


@dataclass
class PlanStep:
    goal_id: str
    description: str
    preconditions: List[str]
    expected_effects: List[str]
    priority: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal_id,
            "step": self.description,
            "preconditions": self.preconditions,
            "expected_effects": self.expected_effects,
            "priority": self.priority,
        }


_KG_LAW_PREDICATES = frozenset({"forbids_tag", "symbolic_forbids"})

_DEFAULT_RULES = [
    Rule(
        name="transitivity_is_a",
        pattern_a=("X", "is_a", "Y"),
        pattern_b=("Y", "is_a", "Z"),
        conclusion=("X", "is_a", "Z"),
    ),
    Rule(
        name="transitivity_part_of",
        pattern_a=("X", "part_of", "Y"),
        pattern_b=("Y", "part_of", "Z"),
        conclusion=("X", "part_of", "Z"),
    ),
    Rule(
        name="causal_chain",
        pattern_a=("X", "cause", "Y"),
        pattern_b=("Y", "cause", "Z"),
        conclusion=("X", "indirect_cause", "Z"),
    ),
    Rule(
        name="reciprocity_like",
        pattern_a=("X", "like", "Y"),
        pattern_b=("Y", "like", "X"),
        conclusion=("X", "mutual_affinity", "Y"),
    ),
    Rule(
        name="reciprocity_help",
        pattern_a=("X", "help", "Y"),
        pattern_b=("Y", "help", "X"),
        conclusion=("X", "mutual_support", "Y"),
    ),
    Rule(
        name="possession_through_give",
        pattern_a=("X", "give", "Z"),
        pattern_b=("Z", "belong_to", "Y"),
        conclusion=("X", "transfer_to", "Y"),
    ),
]


class RuleLearner:
    """Discovers inference rules from predicate co-occurrence in the KG.

    Scans entity neighborhoods for pairs of predicates that co-occur
    frequently, then validates them against existing data to compute
    confidence.  Promoted rules are used alongside built-in rules.
    """

    MAX_LEARNED = 20

    def __init__(self, min_support: int = 5, min_confidence: float = 0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.learned_rules: List[Rule] = []
        self._cooccurrence: Counter = Counter()
        self._observe_count: int = 0

    def observe(self, kg):
        self._observe_count += 1
        self._cooccurrence.clear()

        entity_preds = self._build_entity_predicates(kg)
        for preds in entity_preds.values():
            pred_list = sorted(preds)
            for i in range(len(pred_list)):
                for j in range(i + 1, len(pred_list)):
                    self._cooccurrence[(pred_list[i], pred_list[j])] += 1

    @staticmethod
    def _build_entity_predicates(kg) -> Dict[str, Set[str]]:
        result: Dict[str, Set[str]] = {}
        for eid in kg.entities:
            indices = kg._adjacency.get(eid, [])
            predicates: Set[str] = set()
            for idx in indices:
                if idx < len(kg.relations):
                    predicates.add(kg.relations[idx].predicate)
            if predicates:
                result[eid] = predicates
        return result

    def propose_rules(self, kg) -> List[Rule]:
        candidates: List[Rule] = []
        existing_names = {r.name for r in self.learned_rules}

        for (p1, p2), support in self._cooccurrence.most_common(30):
            if support < self.min_support:
                break
            name = f"learned_{p1}_{p2}"
            if name in existing_names:
                continue

            confidence = self._validate(kg, p1, p2)
            if confidence < self.min_confidence:
                continue

            rule = Rule(
                name=name,
                pattern_a=("X", p1, "Y"),
                pattern_b=("X", p2, "Z"),
                conclusion=("Y", f"cooccurs_with_{p2}", "Z"),
                source="learned",
            )
            candidates.append(rule)

        for rule in candidates:
            if len(self.learned_rules) >= self.MAX_LEARNED:
                self.learned_rules.pop(0)
            self.learned_rules.append(rule)

        return candidates

    def _validate(self, kg, p1: str, p2: str) -> float:
        entity_preds = self._build_entity_predicates(kg)
        count_p1 = 0
        count_both = 0
        for preds in entity_preds.values():
            if p1 in preds:
                count_p1 += 1
                if p2 in preds:
                    count_both += 1
        if count_p1 == 0:
            return 0.0
        return count_both / count_p1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learned_rules": [r.to_dict() for r in self.learned_rules],
            "observe_count": self._observe_count,
            "min_support": self.min_support,
            "min_confidence": self.min_confidence,
        }

    def from_dict(self, data: Dict[str, Any]):
        self.learned_rules = []
        self._observe_count = data.get("observe_count", 0)
        self.min_support = data.get("min_support", self.min_support)
        self.min_confidence = data.get("min_confidence", self.min_confidence)
        for rd in data.get("learned_rules", []):
            self.learned_rules.append(
                Rule(
                    name=rd["name"],
                    pattern_a=tuple(rd["pattern_a"]),
                    pattern_b=tuple(rd["pattern_b"]),
                    conclusion=tuple(rd["conclusion"]),
                    source=rd.get("source", "learned"),
                )
            )


_LINK_EMBED_DIM = 32
_LINK_HIDDEN = 128
_LINK_REPLAY_CAP = 4096

if _TORCH:

    class _LinkPredictorNet(nn.Module):
        """Learns to predict missing KG links from entity-pair embeddings.

        Input: concat(entity_a_embed, entity_b_embed) -> 2*LINK_EMBED_DIM
        Output: predicate logits over a dynamic vocabulary of predicates.
        """

        def __init__(self, max_predicates: int = 128):
            super().__init__()
            self.max_predicates = max_predicates
            self.encoder = nn.Sequential(
                nn.Linear(2 * _LINK_EMBED_DIM, _LINK_HIDDEN),
                nn.ReLU(),
                nn.Linear(_LINK_HIDDEN, _LINK_HIDDEN),
                nn.ReLU(),
            )
            self.pred_head = nn.Linear(_LINK_HIDDEN, max_predicates)
            self.conf_head = nn.Sequential(
                nn.Linear(_LINK_HIDDEN, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor):
            h = self.encoder(x)
            return self.pred_head(h), self.conf_head(h)


class NeuralLinkPredictor:
    """Learns predicate distributions over entity pairs from KG experience.

    Trains on observed (entity_a, predicate, entity_b) triples and can
    propose new links for entity pairs that lack a direct relation.
    """

    _MAX_ENTITY_EMBEDS = 5000

    def __init__(self):
        self._available = _TORCH
        self._entity_embeds: Dict[str, List[float]] = {}
        self._predicate_vocab: Dict[str, int] = {}
        self._predicate_names: List[str] = []
        self._replay: List[Tuple[List[float], int]] = []
        self._train_steps = 0
        self._net = None
        self._optimizer = None

        if _TORCH:
            self._net = _LinkPredictorNet()
            self._optimizer = torch.optim.Adam(self._net.parameters(), lr=1e-3)

    @property
    def available(self) -> bool:
        return self._available and self._net is not None

    def _entity_embed(self, eid: str) -> List[float]:
        if eid in self._entity_embeds:
            return self._entity_embeds[eid]
        import hashlib

        h = hashlib.sha256(eid.encode()).digest()
        embed = [(b / 255.0) * 2.0 - 1.0 for b in h[:_LINK_EMBED_DIM]]
        self._entity_embeds[eid] = embed
        if len(self._entity_embeds) > self._MAX_ENTITY_EMBEDS:
            keys = list(self._entity_embeds.keys())
            self._entity_embeds = {k: self._entity_embeds[k] for k in keys[-4000:]}
        return embed

    def _pred_idx(self, predicate: str) -> int:
        if predicate in self._predicate_vocab:
            return self._predicate_vocab[predicate]
        idx = len(self._predicate_names)
        if idx >= 128:
            import hashlib

            return int(hashlib.md5(predicate.encode()).hexdigest(), 16) % 128
        self._predicate_vocab[predicate] = idx
        self._predicate_names.append(predicate)
        return idx

    def observe_triple(self, subj_id: str, predicate: str, obj_id: str):
        if not self.available:
            return
        ea = self._entity_embed(subj_id)
        eb = self._entity_embed(obj_id)
        pidx = self._pred_idx(predicate)
        self._replay.append((ea + eb, pidx))
        if len(self._replay) > _LINK_REPLAY_CAP:
            self._replay = self._replay[-_LINK_REPLAY_CAP:]

    def train_step(self) -> float:
        if not self.available or len(self._replay) < 16:
            return 0.0
        import random

        batch_size = min(64, len(self._replay))
        batch = random.sample(self._replay, batch_size)
        inputs = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        targets = torch.tensor([b[1] for b in batch], dtype=torch.long)

        self._net.train()
        pred_logits, conf = self._net(inputs)
        pred_loss = nn.functional.cross_entropy(pred_logits, targets)
        conf_targets = torch.ones(conf.shape[0], 1, dtype=torch.float32)
        conf_loss = nn.functional.mse_loss(conf, conf_targets)
        loss = pred_loss + 0.5 * conf_loss
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._net.eval()
        self._train_steps += 1
        return loss.item()

    def propose_links(self, kg, max_proposals: int = 5) -> List[Dict[str, Any]]:
        """Propose missing links between entity pairs."""
        if not self.available or not self._predicate_names or self._train_steps < 10:
            return []

        proposals: List[Dict[str, Any]] = []
        eids = list(kg.entities.keys())
        if len(eids) < 2:
            return []

        import random as rnd

        pairs = []
        for _ in range(min(50, len(eids) * 2)):
            a, b = rnd.sample(eids, 2)
            if not any(r.subject_id == a and r.object_id == b for r in kg.relations):
                pairs.append((a, b))
        if not pairs:
            return []

        inputs = torch.tensor(
            [self._entity_embed(a) + self._entity_embed(b) for a, b in pairs],
            dtype=torch.float32,
        )
        self._net.eval()
        with torch.no_grad():
            pred_logits, conf = self._net(inputs)
            probs = torch.softmax(pred_logits, dim=-1)

        for i, (a, b) in enumerate(pairs):
            confidence = float(conf[i].item())
            if confidence < 0.4:
                continue
            top_pred_idx = int(probs[i].argmax().item())
            top_prob = float(probs[i, top_pred_idx].item())
            if top_prob < 0.3 or top_pred_idx >= len(self._predicate_names):
                continue
            proposals.append(
                {
                    "subject_id": a,
                    "predicate": self._predicate_names[top_pred_idx],
                    "object_id": b,
                    "confidence": round(confidence * top_prob, 3),
                    "source": "neural_link_predictor",
                }
            )

        proposals.sort(key=lambda p: p["confidence"], reverse=True)
        return proposals[:max_proposals]

    def stats(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "train_steps": self._train_steps,
            "predicate_vocab_size": len(self._predicate_names),
            "entity_embeds_cached": len(self._entity_embeds),
            "replay_size": len(self._replay),
        }


class LLMReasoner:
    """LLM-based reasoning with tool use and self-verification (U15).

    Builds a structured prompt from KG triples, goals, and a question.
    The LLM can call tools (KG query, memory recall, math eval) and
    its inferences go through a verification step before commitment.
    """

    def __init__(self, llm_backend=None):
        self._llm = None
        self._available = False
        self._call_count = 0
        self._verified_count = 0
        self._tool_calls = 0
        self._rules_induced = 0
        self._inferences_accumulated = 0
        self._RULE_INDUCTION_THRESHOLD = 50
        if llm_backend is not None:
            self._llm = llm_backend
            self._available = self._llm.available
        else:
            try:
                from engine.LLMBackend import LLMBackend

                self._llm = LLMBackend()
                self._available = self._llm.available
            except Exception as _e:
                print(f"[ReasoningEngine] LLMBackend init error: {_e}", flush=True)

    @property
    def available(self) -> bool:
        return self._available and self._llm is not None and self._llm.available

    def _build_tool_results(self, response: str, kg, memory=None) -> str:
        """Parse TOOL_CALL lines and execute them, returning results."""
        tool_outputs = []
        for line in response.split("\n"):
            stripped = line.strip()
            if not stripped.upper().startswith("TOOL_CALL:"):
                continue
            call = stripped[len("TOOL_CALL:") :].strip()
            self._tool_calls += 1

            if call.startswith("kg_query(") and kg is not None:
                try:
                    args = call[len("kg_query(") : -1]
                    parts = [p.strip().strip('"').strip("'") for p in args.split(",")]
                    subject = parts[0] if len(parts) > 0 else ""
                    predicate = parts[1] if len(parts) > 1 else None
                    matches = []
                    for r in kg.relations[-100:]:
                        sn = kg._entity_name(r.subject_id)
                        if subject.lower() in sn.lower():
                            if predicate is None or predicate.lower() in r.predicate.lower():
                                on = kg._entity_name(r.object_id)
                                matches.append(f"{sn} --[{r.predicate}]--> {on}")
                    result = "\n".join(matches[:10]) if matches else "No matches found."
                    tool_outputs.append(f"[TOOL_RESULT: kg_query]\n{result}")
                except Exception:
                    tool_outputs.append("[TOOL_RESULT: kg_query]\nError executing query.")

            elif call.startswith("memory_recall(") and memory is not None:
                try:
                    query = call[len("memory_recall(") : -1].strip('"').strip("'")
                    nodes = memory.recall(query_text=query, limit=5)
                    results = [n.content[:200] for n in nodes]
                    result = "\n".join(results) if results else "No memories found."
                    tool_outputs.append(f"[TOOL_RESULT: memory_recall]\n{result}")
                except Exception:
                    tool_outputs.append("[TOOL_RESULT: memory_recall]\nError executing recall.")

            elif call.startswith("math_eval("):
                try:
                    expr = call[len("math_eval(") : -1].strip()
                    import ast

                    tree = ast.parse(expr, mode="eval")
                    _SAFE_NODES = (
                        ast.Expression,
                        ast.BinOp,
                        ast.UnaryOp,
                        ast.Num,
                        ast.Constant,
                        ast.Add,
                        ast.Sub,
                        ast.Mult,
                        ast.Div,
                        ast.Pow,
                        ast.Mod,
                        ast.USub,
                        ast.UAdd,
                        ast.Call,
                        ast.Name,
                        ast.Load,
                    )
                    for node in ast.walk(tree):
                        if not isinstance(node, _SAFE_NODES):
                            raise ValueError(f"Unsafe node: {type(node).__name__}")

                    def _safe_pow(base, exp, *mod):
                        if isinstance(exp, (int, float)) and abs(exp) > 1000:
                            raise ValueError(f"exponent too large: {exp}")
                        return pow(base, exp, *mod)

                    _SAFE_FUNCS = {
                        "abs": abs,
                        "max": max,
                        "min": min,
                        "round": round,
                        "sum": sum,
                        "len": len,
                        "pow": _safe_pow,
                        "int": int,
                        "float": float,
                    }
                    result = str(
                        eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, _SAFE_FUNCS)
                    )
                    if len(result) > 1000:
                        result = result[:1000] + "..."
                    tool_outputs.append(f"[TOOL_RESULT: math_eval]\n{result}")
                except Exception as e:
                    tool_outputs.append(f"[TOOL_RESULT: math_eval]\nError: {e}")

        return "\n".join(tool_outputs)

    def _verify_inferences(self, inferences: List[Dict[str, Any]], kg) -> List[Dict[str, Any]]:
        """Ask the LLM to verify its own inferences for consistency."""
        if not inferences or not self.available:
            return inferences

        existing_triples = []
        for r in kg.relations[-20:]:
            sn = kg._entity_name(r.subject_id)
            on = kg._entity_name(r.object_id)
            existing_triples.append(f"  {sn} --[{r.predicate}]--> {on}")

        proposed = []
        for inf in inferences:
            proposed.append(
                f"  {inf['subject_name']} --[{inf['predicate']}]--> {inf['object_name']} "
                f"(confidence: {inf['confidence']:.2f})"
            )

        prompt = (
            "You are a verification engine. Check these PROPOSED inferences "
            "against the existing knowledge graph for consistency.\n\n"
            "[Existing Knowledge]\n" + "\n".join(existing_triples[-15:]) + "\n\n"
            "[Proposed Inferences]\n" + "\n".join(proposed) + "\n\n"
            "For each proposed inference, write:\n"
            "VERIFIED: <index> (the inference is consistent)\n"
            "REJECTED: <index> <reason>\n"
            "where <index> is 1-based.\n"
        )

        response = self._llm.generate(prompt, max_tokens=512, temperature=0.2)
        if not response:
            return inferences

        verified = set()
        rejected = set()
        for line in response.split("\n"):
            line = line.strip().upper()
            if line.startswith("VERIFIED:"):
                try:
                    idx = int(line.split(":")[1].strip().split()[0]) - 1
                    verified.add(idx)
                except (ValueError, IndexError):
                    pass
            elif line.startswith("REJECTED:"):
                try:
                    idx = int(line.split(":")[1].strip().split()[0]) - 1
                    rejected.add(idx)
                except (ValueError, IndexError):
                    pass

        result = []
        for i, inf in enumerate(inferences):
            if i in rejected:
                continue
            if i in verified:
                inf["verified"] = True
                inf["confidence"] = min(0.95, inf["confidence"] + 0.1)
            else:
                inf["verified"] = False
                inf["confidence"] = inf.get("confidence", 0.5) * 0.7
            result.append(inf)
        self._verified_count += len(result)
        return result

    def reason(
        self,
        kg,
        active_goals: List[Dict[str, Any]],
        symbolic_inferences: List[Dict[str, Any]],
        max_new: int = 10,
        memory=None,
    ) -> Dict[str, Any]:
        """Run LLM chain-of-thought with tool use and self-verification."""
        if not self.available:
            return {"inferences": [], "explanation": "", "source": "llm_unavailable"}

        triples = []
        for r in kg.relations[-50:]:
            sn = kg._entity_name(r.subject_id)
            on = kg._entity_name(r.object_id)
            triples.append(f"  {sn} --[{r.predicate}]--> {on}")

        goal_strs = []
        for g in active_goals[:10]:
            desc = g.get("description", g.get("goal_id", ""))
            goal_strs.append(f"  - {desc}")

        sym_strs = []
        for inf in symbolic_inferences[:10]:
            s = inf.get("subject", inf.get("subject_name", ""))
            p = inf.get("predicate", "")
            o = inf.get("object", inf.get("object_name", ""))
            sym_strs.append(f"  {s} --[{p}]--> {o}")

        prompt = (
            "You are a reasoning engine with access to tools. Given the "
            "knowledge graph, goals, and existing inferences, derive NEW "
            "inferences using step-by-step reasoning.\n\n"
            "[Available Tools]\n"
            '  TOOL_CALL: kg_query("subject", "predicate") - query the knowledge graph\n'
            '  TOOL_CALL: memory_recall("query text") - search memories\n'
            '  TOOL_CALL: math_eval("expression") - evaluate math\n\n'
            "[Knowledge Graph Triples]\n" + "\n".join(triples[-30:]) + "\n\n"
            "[Active Goals]\n" + "\n".join(goal_strs) + "\n\n"
        )
        if sym_strs:
            prompt += "[Existing Inferences]\n" + "\n".join(sym_strs) + "\n\n"

        prompt += (
            "Think step-by-step. Use tools when needed. For each new "
            "inference:\n"
            "INFERENCE: subject --[predicate]--> object (confidence: 0.X)\n"
            "\nDerive up to 10 new inferences not already listed.\n"
        )

        response = self._llm.generate(prompt, max_tokens=2048, temperature=0.3)
        self._call_count += 1

        if not response:
            return {"inferences": [], "explanation": "", "source": "llm_empty"}

        tool_results = self._build_tool_results(response, kg, memory)
        if tool_results:
            followup_prompt = (
                response + "\n\n" + tool_results + "\n\n"
                "Based on the tool results above, derive additional inferences.\n"
                "INFERENCE: subject --[predicate]--> object (confidence: 0.X)\n"
            )
            followup = self._llm.generate(followup_prompt, max_tokens=1024, temperature=0.3)
            if followup:
                response = response + "\n" + followup

        raw_inferences = self._parse_inferences(response, kg)
        verified = self._verify_inferences(raw_inferences[:max_new], kg)
        self._inferences_accumulated += len(verified)

        result: Dict[str, Any] = {
            "inferences": verified,
            "explanation": response[:800],
            "source": "llm_chain_of_thought_verified",
            "tool_calls": self._tool_calls,
        }

        if self._inferences_accumulated >= self._RULE_INDUCTION_THRESHOLD:
            try:

                new_rules = self.induce_rules(kg, None)
                if new_rules:
                    result["induced_rules"] = new_rules
            except Exception as _e:
                print(f"[ReasoningEngine] induce_rules error: {_e}", flush=True)

        return result

    def induce_rules(self, kg, rule_learner) -> List[Dict[str, Any]]:
        """After accumulating enough inferences, ask LLM to generalize rules."""
        if not self.available or self._inferences_accumulated < self._RULE_INDUCTION_THRESHOLD:
            return []

        recent_relations = []
        for r in kg.relations[-40:]:
            sn = kg._entity_name(r.subject_id)
            on = kg._entity_name(r.object_id)
            recent_relations.append(f"{sn} --[{r.predicate}]--> {on}")

        prompt = (
            "Given these knowledge graph relationships, identify general "
            "RULES (if A relates to B in way X, then B likely relates to C "
            "in way Y).\n\n"
            "[Relationships]\n" + "\n".join(recent_relations) + "\n\n"
            "For each rule, write:\n"
            "RULE: IF (A --[pred1]--> B) THEN (B --[pred2]--> A) "
            "NAMED: rule_name\n"
        )
        response = self._llm.generate(prompt, max_tokens=1024, temperature=0.4)
        if not response:
            return []

        new_rules = []
        for line in response.split("\n"):
            line = line.strip()
            if not line.upper().startswith("RULE:"):
                continue
            self._rules_induced += 1
            new_rules.append({"text": line, "source": "llm_induction"})
        self._inferences_accumulated = 0
        return new_rules

    @staticmethod
    def _parse_inferences(text: str, kg) -> List[Dict[str, Any]]:
        results = []
        for line in text.split("\n"):
            line = line.strip()
            if not line.upper().startswith("INFERENCE:"):
                continue
            body = line[len("INFERENCE:") :].strip()
            conf = 0.6
            if "(confidence:" in body.lower():
                try:
                    conf_str = body.lower().split("(confidence:")[1].split(")")[0]
                    conf = float(conf_str.strip())
                except (ValueError, IndexError):
                    pass
                body = body.split("(confidence")[0].strip()
            parts = body.split("--[")
            if len(parts) != 2:
                continue
            subj = parts[0].strip()
            rest = parts[1]
            pred_parts = rest.split("]-->")
            if len(pred_parts) != 2:
                continue
            pred = pred_parts[0].strip()
            obj = pred_parts[1].strip()
            if not subj or not pred or not obj:
                continue
            subj_id = subj.lower().replace(" ", "_")
            obj_id = obj.lower().replace(" ", "_")
            results.append(
                {
                    "subject_id": subj_id,
                    "subject_name": subj,
                    "object_id": obj_id,
                    "object_name": obj,
                    "predicate": pred,
                    "confidence": min(0.85, conf),
                    "rule": "llm_chain_of_thought",
                    "source": "llm",
                }
            )
        return results

    def stats(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "call_count": self._call_count,
            "verified_inferences": self._verified_count,
            "tool_calls": self._tool_calls,
            "rules_induced": self._rules_induced,
            "inferences_accumulated": self._inferences_accumulated,
        }


class ReasoningEngine:
    """Symbolic + neural reasoning: inference, analogy, planning, rule learning,
    LLM chain-of-thought, and neural link prediction."""

    RULE_LEARN_INTERVAL = 10
    NEURAL_LINK_INTERVAL = 5
    LLM_REASON_INTERVAL = 3

    def __init__(self, max_inference_steps: int = 10, llm_backend=None, law_manager=None):
        self.built_in_rules: List[Rule] = list(_DEFAULT_RULES)
        self.rules: List[Rule] = list(_DEFAULT_RULES)
        self.rule_learner = RuleLearner()
        self.max_inference_steps = max_inference_steps
        self._inference_history: List[Dict[str, Any]] = []
        self._cycle_count: int = 0
        self.link_predictor = NeuralLinkPredictor()
        self.llm_reasoner = LLMReasoner(llm_backend=llm_backend)
        self._law_manager = law_manager

    def _sync_laws_from_knowledge_graph(self, knowledge_graph) -> int:
        """Promote KG triples (subject --[forbids_tag|symbolic_forbids]--> tag)
        into LawEngine constraints (tag = object entity name)."""
        mgr = self._law_manager
        if mgr is None or is_cognitive_null(mgr):
            return 0
        rels = getattr(knowledge_graph, "relations", None) or []
        if not rels:
            return 0
        touched = 0
        for rel in rels[-400:]:
            if rel.predicate not in _KG_LAW_PREDICATES:
                continue
            try:
                tag = knowledge_graph._entity_name(rel.object_id)
            except Exception:
                continue
            tag = (tag or "").strip().lower()
            if not tag:
                continue
            try:
                anchor = knowledge_graph._entity_name(rel.subject_id)
            except Exception:
                anchor = "kg"
            safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in tag)
            lid = f"kg_forbid_{safe[:48]}"
            try:
                mgr.declare(
                    lid,
                    f"KG: {anchor} forbids pattern `{tag}`",
                    [tag],
                    0.85,
                    source=LAW_SOURCE_EXTERNAL,
                )
                touched += 1
            except Exception:
                continue
        return touched

    def register_causal_rules(self, rules: List[Dict[str, Any]]):
        """Merge environment-learned causal rules into the rule set."""
        for rd in rules:
            name = rd.get("name", "env_rule")
            existing_names = {r.name for r in self.rule_learner.learned_rules}
            if name in existing_names:
                continue
            try:
                rule = Rule(
                    name=name,
                    pattern_a=tuple(rd.get("pattern_a", ("", "", ""))),
                    pattern_b=tuple(rd.get("pattern_b", ("", "", ""))),
                    conclusion=tuple(rd.get("conclusion", ("", "", ""))),
                    source=rd.get("source", "environment"),
                )
                self.rule_learner.learned_rules.append(rule)
                if len(self.rule_learner.learned_rules) > self.rule_learner.MAX_LEARNED:
                    self.rule_learner.learned_rules = self.rule_learner.learned_rules[
                        -self.rule_learner.MAX_LEARNED :
                    ]
            except Exception as _e:
                print(f"[ReasoningEngine] causal rule parse error: {_e}", flush=True)

    def reason(
        self,
        knowledge_graph,
        active_goals: List[Dict[str, Any]],
        nlu_result: Optional[Dict[str, Any]] = None,
        max_depth: Optional[int] = None,
        memory=None,
    ) -> Dict[str, Any]:
        """Run all reasoning capabilities and return combined results."""
        self._cycle_count += 1

        self.rules = list(self.built_in_rules) + list(self.rule_learner.learned_rules)

        inferences = self._forward_chain(knowledge_graph, max_depth=max_depth)
        analogies = self._detect_analogies(knowledge_graph)
        plan_steps = self._decompose_goals(knowledge_graph, active_goals)

        for inf in inferences:
            self._apply_inference(knowledge_graph, inf)

        new_rules: List[Dict[str, Any]] = []
        if self._cycle_count % self.RULE_LEARN_INTERVAL == 0:
            self.rule_learner.observe(knowledge_graph)
            proposed = self.rule_learner.propose_rules(knowledge_graph)
            new_rules = [r.to_dict() for r in proposed]

        # Neural link prediction (Upgrade 7)
        neural_links: List[Dict[str, Any]] = []
        if self._cycle_count % self.NEURAL_LINK_INTERVAL == 0:
            for r in knowledge_graph.relations[-50:]:
                self.link_predictor.observe_triple(r.subject_id, r.predicate, r.object_id)
            self.link_predictor.train_step()
            neural_links = self.link_predictor.propose_links(knowledge_graph)
            for nl in neural_links:
                self._apply_inference(knowledge_graph, nl)

        # LLM chain-of-thought reasoning (Upgrade 7)
        llm_result: Dict[str, Any] = {}
        if self.llm_reasoner.available and self._cycle_count % self.LLM_REASON_INTERVAL == 0:
            symbolic = [self._inf_to_dict(i) for i in inferences]
            llm_result = self.llm_reasoner.reason(
                knowledge_graph, active_goals, symbolic, memory=memory
            )
            for llm_inf in llm_result.get("inferences", []):
                self._apply_inference(knowledge_graph, llm_inf)
                inferences.append(llm_inf)
            for rule_dict in llm_result.get("induced_rules", []):
                try:
                    text = rule_dict.get("text", "")
                    if "--[" in text and "NAMED:" in text:
                        name_part = text.split("NAMED:")[-1].strip()
                        r = Rule(
                            name=name_part or "llm_rule",
                            pattern_a=("?A", "relates_to", "?B"),
                            pattern_b=("?B", "relates_to", "?C"),
                            conclusion=("?A", "llm_inferred", "?C"),
                            source="llm_induction",
                        )
                        if len(self.rule_learner.learned_rules) >= self.rule_learner.MAX_LEARNED:
                            self.rule_learner.learned_rules.pop(0)
                        self.rule_learner.learned_rules.append(r)
                except Exception as _e:
                    print(f"[ReasoningEngine] LLM rule merge error: {_e}", flush=True)

        all_inferences = [self._inf_to_dict(i) for i in inferences]
        depth = len(inferences) + len(analogies) + len(plan_steps)

        kg_symbolic_laws = self._sync_laws_from_knowledge_graph(knowledge_graph)

        return {
            "inferences": all_inferences,
            "analogies": [a.to_dict() for a in analogies],
            "plan_steps": [p.to_dict() for p in plan_steps],
            "reasoning_depth": depth,
            "learned_rules_count": len(self.rule_learner.learned_rules),
            "new_rules": new_rules,
            "neural_links": neural_links,
            "llm_reasoning": llm_result,
            "kg_symbolic_laws_touched": kg_symbolic_laws,
        }

    def _forward_chain(self, kg, max_depth: Optional[int] = None) -> List[Dict[str, Any]]:
        new_inferences: List[Dict[str, Any]] = []
        existing_triples: Set[Tuple[str, str, str]] = set()

        pred_index: Dict[str, list] = {}
        for rel in kg.relations:
            existing_triples.add((rel.subject_id, rel.predicate, rel.object_id))
            pred_index.setdefault(rel.predicate, []).append(rel)

        step_limit = max_depth if max_depth is not None else self.max_inference_steps
        steps = 0
        found_new = True
        while found_new and steps < step_limit:
            found_new = False
            steps += 1

            for rule in self.rules:
                matches = self._match_rule(kg, rule, existing_triples, pred_index)
                for match in matches:
                    triple = (match["subject_id"], match["predicate"], match["object_id"])
                    if triple not in existing_triples:
                        existing_triples.add(triple)
                        new_inferences.append(match)
                        found_new = True

        return new_inferences

    def _match_rule(
        self,
        kg,
        rule: Rule,
        existing: Set[Tuple[str, str, str]],
        pred_index: Optional[Dict[str, list]] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        pa_pred = rule.pattern_a[1]
        pb_pred = rule.pattern_b[1]

        if pred_index is not None:
            rels_a = pred_index.get(pa_pred, [])
            rels_b = pred_index.get(pb_pred, [])
        else:
            rels_a = [r for r in kg.relations if r.predicate == pa_pred]
            rels_b = [r for r in kg.relations if r.predicate == pb_pred]

        if not rels_a or not rels_b:
            return results

        pa_subj_var, _, pa_obj_var = rule.pattern_a
        pb_subj_var, _, pb_obj_var = rule.pattern_b
        cc_subj_var, cc_pred, cc_obj_var = rule.conclusion

        shared_var = None
        if pb_subj_var in (pa_subj_var, pa_obj_var):
            shared_var = pb_subj_var
        elif pb_obj_var in (pa_subj_var, pa_obj_var):
            shared_var = pb_obj_var

        if shared_var is not None and len(rels_b) > 20:
            b_by_shared: Dict[str, list] = {}
            for rb in rels_b:
                key = rb.subject_id if shared_var == pb_subj_var else rb.object_id
                b_by_shared.setdefault(key, []).append(rb)

            for ra in rels_a:
                bindings_a = {pa_subj_var: ra.subject_id, pa_obj_var: ra.object_id}
                join_val = bindings_a.get(shared_var)
                if join_val is None:
                    continue
                for rb in b_by_shared.get(join_val, []):
                    self._try_bind_and_emit(
                        ra,
                        rb,
                        bindings_a,
                        pb_subj_var,
                        pb_obj_var,
                        cc_subj_var,
                        cc_pred,
                        cc_obj_var,
                        existing,
                        kg,
                        results,
                        rule.name,
                    )
        else:
            for ra in rels_a:
                bindings_a = {pa_subj_var: ra.subject_id, pa_obj_var: ra.object_id}
                for rb in rels_b:
                    self._try_bind_and_emit(
                        ra,
                        rb,
                        bindings_a,
                        pb_subj_var,
                        pb_obj_var,
                        cc_subj_var,
                        cc_pred,
                        cc_obj_var,
                        existing,
                        kg,
                        results,
                        rule.name,
                    )

        return results

    @staticmethod
    def _try_bind_and_emit(
        ra,
        rb,
        bindings_a,
        pb_subj_var,
        pb_obj_var,
        cc_subj_var,
        cc_pred,
        cc_obj_var,
        existing,
        kg,
        results,
        rule_name,
    ):
        bindings = dict(bindings_a)
        b_subj = rb.subject_id
        b_obj = rb.object_id

        if pb_subj_var in bindings:
            if bindings[pb_subj_var] != b_subj:
                return
        else:
            bindings[pb_subj_var] = b_subj

        if pb_obj_var in bindings:
            if bindings[pb_obj_var] != b_obj:
                return
        else:
            bindings[pb_obj_var] = b_obj

        c_subj = bindings.get(cc_subj_var)
        c_obj = bindings.get(cc_obj_var)
        if c_subj and c_obj and c_subj != c_obj:
            triple = (c_subj, cc_pred, c_obj)
            if triple not in existing:
                subj_name = kg._entity_name(c_subj)
                obj_name = kg._entity_name(c_obj)
                results.append(
                    {
                        "subject_id": c_subj,
                        "object_id": c_obj,
                        "predicate": cc_pred,
                        "rule": rule_name,
                        "subject_name": subj_name,
                        "object_name": obj_name,
                        "confidence": min(ra.confidence, rb.confidence) * 0.9,
                    }
                )

    def _apply_inference(self, kg, inference: Dict[str, Any]):
        from core.KnowledgeGraph import Relation as KGRelation

        triple = (inference["subject_id"], inference["predicate"], inference["object_id"])
        if triple in kg._relation_index:
            return

        rel = KGRelation(
            subject_id=inference["subject_id"],
            predicate=inference["predicate"],
            object_id=inference["object_id"],
            confidence=inference.get("confidence", 0.7),
            source="inference",
        )
        idx = len(kg.relations)
        kg.relations.append(rel)
        kg._adjacency[rel.subject_id].append(idx)
        kg._adjacency[rel.object_id].append(idx)
        kg._relation_index[triple] = idx

    _ANALOGY_ENTITY_CAP = 200

    def _detect_analogies(self, kg, max_analogies: int = 3) -> List[Analogy]:
        if len(kg.entities) < 2:
            return []

        entity_predicates: Dict[str, Set[str]] = {}
        for eid in kg.entities:
            preds: Set[str] = set()
            for idx in kg._adjacency.get(eid, []):
                if idx < len(kg.relations):
                    preds.add(kg.relations[idx].predicate)
            if preds:
                entity_predicates[eid] = preds

        eids = list(entity_predicates.keys())
        if len(eids) > self._ANALOGY_ENTITY_CAP:
            import random

            eids = sorted(eids, key=lambda e: len(entity_predicates[e]), reverse=True)
            top = eids[: self._ANALOGY_ENTITY_CAP // 2]
            rest = eids[self._ANALOGY_ENTITY_CAP // 2 :]
            random.shuffle(rest)
            eids = top + rest[: self._ANALOGY_ENTITY_CAP // 2]

        analogies: List[Analogy] = []
        for i in range(len(eids)):
            if len(analogies) >= max_analogies * 3:
                break
            for j in range(i + 1, len(eids)):
                preds_a = entity_predicates[eids[i]]
                preds_b = entity_predicates[eids[j]]
                shared = preds_a & preds_b
                if not shared:
                    continue
                union = preds_a | preds_b
                sim = len(shared) / len(union)
                if sim >= 0.3:
                    analogies.append(
                        Analogy(
                            source_entity=kg._entity_name(eids[i]),
                            target_entity=kg._entity_name(eids[j]),
                            shared_predicates=sorted(shared),
                            similarity=sim,
                        )
                    )
                    if len(analogies) >= max_analogies * 3:
                        break

        analogies.sort(key=lambda a: a.similarity, reverse=True)
        return analogies[:max_analogies]

    def _decompose_goals(
        self, kg, active_goals: List[Dict[str, Any]], max_depth: int = 2
    ) -> List[PlanStep]:
        plan_steps: List[PlanStep] = []

        for goal in active_goals[:3]:
            goal_id = goal.get("goal_id", "")
            desc = goal.get("description", goal_id)
            if not desc:
                desc = goal_id

            desc_lower = desc.lower()
            words = set(desc_lower.replace("_", " ").split())

            relevant_entities: List[str] = []
            for eid, ent in kg.entities.items():
                if ent.name.lower() in desc_lower or any(
                    w in ent.name.lower() for w in words if len(w) > 3
                ):
                    relevant_entities.append(eid)

            if not relevant_entities:
                plan_steps.append(
                    PlanStep(
                        goal_id=goal_id,
                        description=f"Gather information about: {desc}",
                        preconditions=["insufficient knowledge"],
                        expected_effects=["knowledge acquisition"],
                        priority=goal.get("priority", 0.5),
                    )
                )
                continue

            for eid in relevant_entities[:2]:
                rels = kg._adjacency.get(eid, [])
                ent = kg.entities.get(eid)
                if not ent:
                    continue

                if len(rels) < 2:
                    plan_steps.append(
                        PlanStep(
                            goal_id=goal_id,
                            description=f"Learn more about {ent.name} "
                            f"(only {len(rels)} known relations)",
                            preconditions=[f"{ent.name} has sparse knowledge"],
                            expected_effects=[f"richer understanding of {ent.name}"],
                            priority=goal.get("priority", 0.5) + 0.1,
                        )
                    )

                connected_names: List[str] = []
                for idx in rels:
                    if idx < len(kg.relations):
                        r = kg.relations[idx]
                        other = r.object_id if r.subject_id == eid else r.subject_id
                        connected_names.append(kg._entity_name(other))

                if connected_names and max_depth > 0:
                    plan_steps.append(
                        PlanStep(
                            goal_id=goal_id,
                            description=f"Explore connections of {ent.name}: "
                            f"{', '.join(connected_names[:3])}",
                            preconditions=[f"{ent.name} is known"],
                            expected_effects=[f"deeper understanding via {ent.name}'s connections"],
                            priority=goal.get("priority", 0.5),
                        )
                    )

        # Same sparse-KG or gather-information line is often emitted once per
        # matching goal; dedupe by description so plan_steps / UI are not spammed.
        seen_desc: set[str] = set()
        deduped: List[PlanStep] = []
        for p in plan_steps:
            if p.description in seen_desc:
                continue
            seen_desc.add(p.description)
            deduped.append(p)
        return deduped[:6]

    def _inf_to_dict(self, inf: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "subject": inf.get("subject_name", inf.get("subject_id", "")),
            "predicate": inf.get("predicate", ""),
            "object": inf.get("object_name", inf.get("object_id", "")),
            "rule": inf.get("rule", ""),
            "confidence": inf.get("confidence", 0.7),
        }

    def add_rule(self, rule: Rule):
        self.built_in_rules.append(rule)

    def stats(self) -> Dict[str, Any]:
        return {
            "built_in_rules": len(self.built_in_rules),
            "learned_rules": len(self.rule_learner.learned_rules),
            "total_rules": len(self.built_in_rules) + len(self.rule_learner.learned_rules),
            "max_steps": self.max_inference_steps,
            "observe_cycles": self.rule_learner._observe_count,
            "link_predictor": self.link_predictor.stats(),
            "llm_reasoner": self.llm_reasoner.stats(),
        }
