"""
MetaCognitionEngine — self-monitoring for HaromaX6
(Phase 3 → Phase 12).

Tracks rolling trends of the system's own learning performance,
detects cognitive anomalies, generates self-assessments and
meta-level goals that feed back into the cognitive cycle.

Phase 12 upgrade: The metacognition system now LEARNS which of its
own assessments, concerns, and insights actually predict good outcomes.
A transformer over recent cycle features predicts outcomes; thresholds
adapt from concern/insight predictive accuracy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, List, Optional, Tuple

if TYPE_CHECKING:
    from core.KnowledgeGraph import KnowledgeGraph
import time

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except (ImportError, OSError):
    _TORCH = False

from engine.ComputeFabric import get_fabric as _get_fabric

_META_INPUT_DIM = 10
_META_SEQ_LEN = 128
_META_D_MODEL = 256
_META_N_HEADS = 8
_META_FF_DIM = 512
_META_N_LAYERS = 4

if _TORCH:

    class _MetaTransformerNet(nn.Module):
        """Transformer encoder over a sequence of cycle summaries.

        Attends over the last N cycle snapshots to detect temporal
        patterns in cognitive performance.

        Input : (batch, seq_len, _META_INPUT_DIM)
        Output: outcome prediction (scalar), anomaly logits (4-class),
                confidence (scalar)
        """

        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(_META_INPUT_DIM, _META_D_MODEL)
            self.pos_embed = nn.Embedding(_META_SEQ_LEN, _META_D_MODEL)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=_META_D_MODEL,
                nhead=_META_N_HEADS,
                dim_feedforward=_META_FF_DIM,
                dropout=0.1,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=_META_N_LAYERS)

            self.outcome_head = nn.Sequential(
                nn.Linear(_META_D_MODEL, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Tanh(),
            )
            self.anomaly_head = nn.Sequential(
                nn.Linear(_META_D_MODEL, 32),
                nn.ReLU(),
                nn.Linear(32, 4),
            )
            self.confidence_head = nn.Sequential(
                nn.Linear(_META_D_MODEL, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor):
            """x: (batch, seq_len, input_dim)"""
            seq_len = x.size(1)
            h = self.input_proj(x)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            h = h + self.pos_embed(positions)
            h = self.encoder(h)
            pooled = h[:, -1, :]  # use last token as summary
            outcome = self.outcome_head(pooled).squeeze(-1)
            anomaly = self.anomaly_head(pooled)
            confidence = self.confidence_head(pooled).squeeze(-1)
            return outcome, anomaly, confidence


class MetaCognitionEngine:
    _RAMP_STEPS = 80
    _MAX_WEIGHT = 0.9
    _MAX_HISTORY = 2000
    _MAX_ACCURACY_ENTRIES = 1000
    _ACCURACY_KEEP = 500

    def __init__(self, window: int = 50):
        self.window = window
        self._history: List[Dict[str, Any]] = []

        self._emotion_streak: int = 0
        self._last_emotion: str = "neutral"

        self.available = _TORCH
        self._train_steps = 0
        self._train_buffer: List[Tuple[List[float], float]] = []
        self._buffer_cap = 256
        self._rolling_accuracy: List[float] = []
        self._accuracy_window = 50

        self._concern_accuracy: Dict[str, List[bool]] = {}
        self._insight_accuracy: Dict[str, List[bool]] = {}
        self._threshold_adjustments: Dict[str, float] = {}

        self._last_prediction: float = 0.5
        self._last_concerns: List[str] = []
        self._last_insights: List[str] = []

        self._self_score_weights = {
            "prediction_error": 0.3,
            "action_score": 0.3,
            "emotion_stability": 0.2,
            "no_drift": 0.2,
        }

        self._inspection_count: int = 0
        self._inspection_interval: int = 5
        self._inspection_benefit_history: List[float] = []

        self._anomaly_labels = [
            "prediction_collapse",
            "emotion_loop",
            "action_decline",
            "identity_drift",
        ]

        if _TORCH:
            self._transformer = _MetaTransformerNet()
            _fab = _get_fabric()
            if _fab:
                self._transformer = _fab.register("meta_transformer", self._transformer)
            self._transformer.eval()
            self._transformer_optim = torch.optim.Adam(self._transformer.parameters(), lr=5e-4)
            self._transformer_train_steps = 0
            self._transformer_seq_buffer: List[List[float]] = []
            self._last_anomaly_logits = None
            self._last_meta_confidence = 0.5
        else:
            self._transformer = None
            self._transformer_optim = None
            self._transformer_train_steps = 0
            self._transformer_seq_buffer = []
            self._last_anomaly_logits = None
            self._last_meta_confidence = 0.5

    @property
    def learned_weight(self) -> float:
        progress = min(1.0, self._train_steps / max(self._RAMP_STEPS, 1))
        return progress * self._MAX_WEIGHT

    def assess(
        self,
        episode_payload: Dict[str, Any],
        emotion_summary: Dict[str, Any],
        curiosity_result: Dict[str, Any],
        outcome: Dict[str, Any],
        self_inspection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        appraisal = episode_payload.get("appraisal", {})
        snapshot = {
            "cycle": episode_payload.get("cycle_id", 0),
            "timestamp": time.time(),
            "emotion": emotion_summary.get("dominant", "neutral"),
            "intensity": emotion_summary.get("intensity", 0.0),
            "prediction_error": curiosity_result.get("prediction_error", 0.5),
            "action_score": outcome.get("score", 0.0),
            "memory_count": episode_payload.get("salience", 0.0),
            "drift": episode_payload.get("drift_score", 0.0),
            "curiosity_score": curiosity_result.get("curiosity_score", 0.0),
            "learned_weight": emotion_summary.get("learned_model", {}).get("learned_weight", 0.0),
            "appraisal_coping": appraisal.get("coping", 0.5),
            "appraisal_relevance": appraisal.get("relevance", 0.0),
            "appraisal_overrides": appraisal.get("overrides", False),
            "self_surprise": episode_payload.get("self_surprise", {}).get("overall_surprise", 0.0),
            "self_prediction_accuracy": episode_payload.get("self_surprise", {}).get(
                "accuracy", 0.5
            ),
        }
        self._history.append(snapshot)
        if len(self._history) > self._MAX_HISTORY:
            self._history = self._history[-self._MAX_HISTORY :]

        current_emo = snapshot["emotion"]
        if current_emo == self._last_emotion:
            self._emotion_streak += 1
        else:
            self._emotion_streak = 1
            self._last_emotion = current_emo

        trends = self._compute_trends()
        insights = self._generate_insights(trends, snapshot)
        concerns = self._generate_concerns(trends, snapshot)
        reco_i, reco_c = self._reconciliation_meta(episode_payload.get("reconciliation") or {})
        insights.extend(reco_i)
        concerns.extend(reco_c)
        self_score = self._compute_self_score(trends)

        meta_features = self._extract_meta_features(trends, snapshot)
        learned_prediction = self._predict_outcome(meta_features)

        if self_inspection:
            for expl in self_inspection.get("causal_explanations", []):
                insights.append(f"[self-inspection] {expl}")

        anomalies = self.detect_anomalies()
        for anom in anomalies:
            concerns.append(
                f"[anomaly] {anom['type']} detected "
                f"(p={anom['probability']:.2f}, conf={anom['confidence']:.2f})"
            )

        self._last_concerns = [c[:40] for c in concerns]
        self._last_insights = [i[:40] for i in insights]
        self._last_prediction = learned_prediction

        return {
            "insights": insights,
            "concerns": concerns,
            "self_score": round(self_score, 3),
            "trends": trends,
            "emotion_streak": self._emotion_streak,
            "emotion_streak_label": self._last_emotion,
            "predicted_outcome": round(learned_prediction, 3),
            "meta_learned_weight": round(self.learned_weight, 3),
            "self_inspection": self_inspection,
            "anomalies": anomalies,
        }

    def learn_from_outcome(self, actual_outcome: float):
        if self._last_concerns or self._last_insights:
            for concern_key in self._last_concerns:
                acc_list = self._concern_accuracy.setdefault(concern_key, [])
                was_right = actual_outcome < 0.5
                acc_list.append(was_right)
                if len(acc_list) > 100:
                    self._concern_accuracy[concern_key] = acc_list[-100:]

            for insight_key in self._last_insights:
                acc_list = self._insight_accuracy.setdefault(insight_key, [])
                was_right = actual_outcome > 0.4
                acc_list.append(was_right)
                if len(acc_list) > 100:
                    self._insight_accuracy[insight_key] = acc_list[-100:]

            self._adapt_thresholds()

        if len(self._concern_accuracy) > self._MAX_ACCURACY_ENTRIES:
            keys = list(self._concern_accuracy.keys())
            self._concern_accuracy = {
                k: self._concern_accuracy[k] for k in keys[-self._ACCURACY_KEEP :]
            }
        if len(self._insight_accuracy) > self._MAX_ACCURACY_ENTRIES:
            keys = list(self._insight_accuracy.keys())
            self._insight_accuracy = {
                k: self._insight_accuracy[k] for k in keys[-self._ACCURACY_KEEP :]
            }

        if self.available and len(self._history) >= 2:
            prev = self._history[-2]
            meta_features = self._extract_meta_features(self._compute_trends(), prev)
            self._train_buffer.append((meta_features, actual_outcome))
            if len(self._train_buffer) > self._buffer_cap:
                self._train_buffer = self._train_buffer[-self._buffer_cap :]

        self._adapt_self_score_weights(actual_outcome)

    def _adapt_thresholds(self):
        for key, acc_list in self._concern_accuracy.items():
            if len(acc_list) < 10:
                continue
            accuracy = sum(acc_list[-20:]) / len(acc_list[-20:])
            if accuracy < 0.3:
                self._threshold_adjustments[key] = self._threshold_adjustments.get(key, 0.0) + 0.01
            elif accuracy > 0.7:
                self._threshold_adjustments[key] = self._threshold_adjustments.get(key, 0.0) - 0.005

    def _adapt_self_score_weights(self, outcome: float):
        if len(self._history) < 5:
            return
        last = self._history[-1]
        pe = 1.0 - last.get("prediction_error", 0.5)
        action = last.get("action_score", 0.5)
        stability = 1.0 if self._emotion_streak < 5 else 0.5
        no_drift = 1.0 - min(1.0, last.get("drift", 0.0))

        components = {
            "prediction_error": pe,
            "action_score": action,
            "emotion_stability": stability,
            "no_drift": no_drift,
        }

        for key, val in components.items():
            correlation = val * outcome
            old_w = self._self_score_weights[key]
            self._self_score_weights[key] = old_w + 0.005 * (correlation - old_w)

        total = sum(self._self_score_weights.values())
        if total > 0:
            self._self_score_weights = {k: v / total for k, v in self._self_score_weights.items()}

    def _extract_meta_features(self, trends: Dict, snapshot: Dict) -> List[float]:
        return [
            snapshot.get("prediction_error", 0.5),
            snapshot.get("action_score", 0.5),
            trends.get("emotion_stability", 0.5),
            snapshot.get("drift", 0.0),
            snapshot.get("curiosity_score", 0.0),
            snapshot.get("appraisal_coping", 0.5),
            snapshot.get("self_surprise", 0.0),
            snapshot.get("self_prediction_accuracy", 0.5),
            trends.get("appraisal_override_rate", 0.0),
            min(1.0, self._emotion_streak / 20.0),
        ]

    def _predict_outcome(self, features: List[float]) -> float:
        if not self.available:
            return 0.5
        if self.learned_weight < 0.05:
            return 0.5

        self._transformer_seq_buffer.append(features)
        if len(self._transformer_seq_buffer) > _META_SEQ_LEN:
            self._transformer_seq_buffer = self._transformer_seq_buffer[-_META_SEQ_LEN:]

        if (
            self._transformer is not None
            and len(self._transformer_seq_buffer) >= 5
            and self._transformer_train_steps >= 10
        ):
            with torch.no_grad():
                seq = torch.tensor([self._transformer_seq_buffer], dtype=torch.float32)
                outcome, anomaly, confidence = self._transformer(seq)
                pred = (float(outcome.item()) + 1.0) / 2.0
                self._last_anomaly_logits = anomaly.squeeze(0)
                self._last_meta_confidence = float(confidence.item())
                return pred

        return 0.5

    def train_step(self) -> Optional[float]:
        if not self.available or not self._train_buffer:
            return None
        if len(self._train_buffer) < 8:
            return None

        transformer_loss_val = 0.0
        out_loss_scalar = 0.0
        if self._transformer is not None and len(self._transformer_seq_buffer) >= 8:
            self._transformer.train()
            seq_data = self._transformer_seq_buffer
            _dev = next(self._transformer.parameters()).device
            seq_tensor = torch.tensor([seq_data], dtype=torch.float32, device=_dev)

            last_outcome = self._train_buffer[-1][1]
            outcome_target = torch.tensor(
                [last_outcome * 2.0 - 1.0], dtype=torch.float32, device=_dev
            )

            anomaly_target = self._compute_anomaly_target().to(_dev)

            out_pred, anom_logits, conf_pred = self._transformer(seq_tensor)
            out_loss = nn.functional.mse_loss(out_pred, outcome_target)
            out_loss_scalar = float(out_loss.detach().item())
            anom_loss = nn.functional.cross_entropy(anom_logits, anomaly_target.unsqueeze(0))
            conf_target = torch.tensor(
                [1.0 - min(1.0, out_loss_scalar)], dtype=torch.float32, device=_dev
            )
            conf_loss = nn.functional.mse_loss(conf_pred, conf_target)

            total = 0.5 * out_loss + 0.3 * anom_loss + 0.2 * conf_loss
            self._transformer_optim.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(self._transformer.parameters(), 1.0)
            self._transformer_optim.step()
            self._transformer.eval()
            self._transformer_train_steps += 1
            transformer_loss_val = total.item()

        self._train_steps += 1

        loss_val = transformer_loss_val
        accuracy = 1.0 - min(1.0, out_loss_scalar) if out_loss_scalar > 0 else 0.5
        self._rolling_accuracy.append(accuracy)
        if len(self._rolling_accuracy) > self._accuracy_window:
            self._rolling_accuracy = self._rolling_accuracy[-self._accuracy_window :]

        return loss_val

    def _compute_anomaly_target(self) -> torch.Tensor:
        """Determine the dominant anomaly from recent trends."""
        trends = self._compute_trends()
        if not trends:
            return torch.tensor(0, dtype=torch.long)

        scores = [
            trends.get("prediction_error_avg", 0.0),
            1.0 if self._emotion_streak > 10 else 0.0,
            max(0.0, -trends.get("action_score_trend", 0.0)),
            trends.get("drift_avg", 0.0),
        ]
        return torch.tensor(int(max(range(4), key=lambda i: scores[i])), dtype=torch.long)

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Use the transformer to detect cognitive anomalies."""
        if (
            not self.available
            or self._transformer is None
            or self._transformer_train_steps < 10
            or len(self._transformer_seq_buffer) < 5
        ):
            return []

        with torch.no_grad():
            _dev = next(self._transformer.parameters()).device
            seq = torch.tensor([self._transformer_seq_buffer], dtype=torch.float32, device=_dev)
            _, anomaly_logits, confidence = self._transformer(seq)
            probs = torch.softmax(anomaly_logits.squeeze(0), dim=0)

        anomalies = []
        conf = float(confidence.item())
        for i, (label, prob) in enumerate(zip(self._anomaly_labels, probs.tolist())):
            if prob > 0.35 and conf > 0.3:
                anomalies.append(
                    {
                        "type": label,
                        "probability": round(prob, 3),
                        "confidence": round(conf, 3),
                    }
                )
        return anomalies

    def generate_meta_goals(self, assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        goals: List[Dict[str, Any]] = []
        trends = assessment.get("trends", {})

        pe_trend = trends.get("prediction_error_trend", 0.0)
        pe_thresh = 0.05 + self._threshold_adjustments.get("prediction_error_worsening", 0.0)
        if pe_trend > pe_thresh:
            goals.append(
                {
                    "goal_id": "meta_recalibrate_worldmodel",
                    "description": "Recalibrate world model -- prediction error is worsening",
                    "priority": 0.7,
                    "source": "metacognition",
                }
            )

        streak_thresh = max(5, 10 + int(self._threshold_adjustments.get("emotion_loop", 0.0) * 100))
        if assessment.get("emotion_streak", 0) > streak_thresh:
            label = assessment.get("emotion_streak_label", "unknown")
            goals.append(
                {
                    "goal_id": "meta_break_emotion_loop",
                    "description": f"Seek novel input to break {label} loop "
                    f"(stuck for {assessment['emotion_streak']} cycles)",
                    "priority": 0.6,
                    "source": "metacognition",
                }
            )

        as_trend = trends.get("action_score_trend", 0.0)
        as_thresh = -0.03 + self._threshold_adjustments.get("action_declining", 0.0)
        if as_trend < as_thresh:
            goals.append(
                {
                    "goal_id": "meta_improve_strategy",
                    "description": "Reflect on strategy effectiveness -- action scores declining",
                    "priority": 0.5,
                    "source": "metacognition",
                }
            )

        drift_avg = trends.get("drift_avg", 0.0)
        if drift_avg > 0.3:
            goals.append(
                {
                    "goal_id": "meta_stabilize_identity",
                    "description": "Stabilize identity -- persistent drift detected",
                    "priority": 0.65,
                    "source": "metacognition",
                }
            )

        coping_avg = trends.get("coping_avg", 0.5)
        if coping_avg < 0.35:
            goals.append(
                {
                    "goal_id": "meta_improve_coping",
                    "description": "Acquire knowledge or refine strategy "
                    "-- coping potential is consistently low",
                    "priority": 0.6,
                    "source": "metacognition",
                }
            )

        ss_avg = trends.get("self_surprise_avg", 0.5)
        if ss_avg > 0.5:
            goals.append(
                {
                    "goal_id": "meta_improve_self_knowledge",
                    "description": "Develop better self-understanding "
                    "-- self-prediction is consistently poor",
                    "priority": 0.55,
                    "source": "metacognition",
                }
            )

        predicted = assessment.get("predicted_outcome", 0.5)
        if predicted < 0.3 and self.learned_weight > 0.2:
            goals.append(
                {
                    "goal_id": "meta_preemptive_correction",
                    "description": "Preemptively adjust strategy -- "
                    "metacognition predicts poor outcome",
                    "priority": 0.7 * self.learned_weight,
                    "source": "metacognition_learned",
                }
            )

        return goals

    # ------------------------------------------------------------------
    # Recursive self-inspection (Phase 14)
    # ------------------------------------------------------------------

    def should_inspect(self, cycle_count: int) -> bool:
        return cycle_count % max(1, self._inspection_interval) == 0

    def inspect_self(
        self, assessment: Dict[str, Any], self_surprise: Dict[str, Any], controller
    ) -> Dict[str, Any]:

        temp_kg = self._build_self_kg(assessment, self_surprise, controller)

        meta_goals = [
            {
                "goal_id": "understand_self",
                "description": "Understand current cognitive state",
                "priority": 0.8,
            }
        ]

        reasoning_result = {}
        if hasattr(controller, "reasoning"):
            try:
                reasoning_result = controller.reasoning.reason(temp_kg, meta_goals, max_depth=3)
            except Exception as _e:
                print(f"[MetaCognition] inspect_self reasoning error: {_e}", flush=True)

        causal_explanations = []
        recommended_adjustments = []

        inferences = reasoning_result.get("inferences", [])
        for inf in inferences:
            subj = inf.get("subject", "")
            pred = inf.get("predicate", "")
            obj = inf.get("object", "")
            if subj and pred and obj:
                causal_explanations.append(f"{subj} {pred} {obj}")

        trends = assessment.get("trends", {})

        surprise_level = self_surprise.get("overall_surprise", 0.0)
        outcome_avg = trends.get("action_score_avg", 0.5)

        if surprise_level > 0.5 and outcome_avg < 0.4:
            causal_explanations.append(
                "High self-surprise correlates with poor outcomes "
                "-- self-prediction accuracy should be prioritized"
            )
            recommended_adjustments.append("increase_self_model_training")

        emotion_stability = trends.get("emotion_stability", 0.5)
        emotion_streak = assessment.get("emotion_streak", 0)
        if emotion_streak > 10 and emotion_stability > 0.9:
            causal_explanations.append(
                f"Emotion stuck in {assessment.get('emotion_streak_label', '?')} "
                f"for {emotion_streak} cycles -- novelty-seeking may help"
            )
            recommended_adjustments.append("increase_novelty_bias")

        drift_avg = trends.get("drift_avg", 0.0)
        if drift_avg > 0.3:
            causal_explanations.append(
                "Identity drift is elevated -- reconsolidation or "
                "reflection may stabilize self-model"
            )
            recommended_adjustments.append("prioritize_reflection")

        coping = trends.get("coping_avg", 0.5)
        if coping < 0.35:
            causal_explanations.append(
                "Low coping suggests inadequate knowledge or strategies for current challenges"
            )
            recommended_adjustments.append("seek_knowledge")

        self._inspection_count += 1

        return {
            "causal_explanations": causal_explanations,
            "recommended_adjustments": recommended_adjustments,
            "inspection_count": self._inspection_count,
            "temp_kg_entities": len(temp_kg.entities),
        }

    def _build_self_kg(
        self, assessment: Dict[str, Any], self_surprise: Dict[str, Any], controller
    ) -> KnowledgeGraph:
        from core.KnowledgeGraph import KnowledgeGraph, Entity, Relation
        import time as _time

        kg = KnowledgeGraph(max_entities=50, max_relations=100)
        now = _time.time()

        self_entities = {
            "self_model": {
                "accuracy": self_surprise.get("accuracy", 0.5),
                "surprise": self_surprise.get("overall_surprise", 0.0),
            },
            "emotion_system": {
                "streak": assessment.get("emotion_streak", 0),
                "label": assessment.get("emotion_streak_label", "neutral"),
                "stability": assessment.get("trends", {}).get("emotion_stability", 0.5),
            },
            "action_system": {
                "score_avg": assessment.get("trends", {}).get("action_score_avg", 0.5),
                "score_trend": assessment.get("trends", {}).get("action_score_trend", 0.0),
            },
            "knowledge_system": {
                "prediction_error": assessment.get("trends", {}).get("prediction_error_avg", 0.5),
            },
            "coping": {
                "level": assessment.get("trends", {}).get("coping_avg", 0.5),
            },
        }

        if hasattr(controller, "counterfactual"):
            self_entities["counterfactual_gate"] = {
                "weight": controller.counterfactual.learned_weight,
            }
        if hasattr(controller, "process_gate"):
            self_entities["process_gate"] = {
                "weight": controller.process_gate.learned_weight,
            }
        if hasattr(controller, "imagination"):
            self_entities["imagination"] = {
                "weight": controller.imagination.learned_weight,
                "accuracy": controller.imagination.overall_accuracy,
            }

        for name, props in self_entities.items():
            eid = f"self:{name}"
            e = Entity(id=eid, name=name, entity_type="self_component")
            for k, v in props.items():
                e.properties[k] = v
            e.last_updated = now
            kg.entities[eid] = e

        def _add_rel(subj, pred, obj, conf=0.8):
            r = Relation(
                subject_id=f"self:{subj}",
                predicate=pred,
                object_id=f"self:{obj}",
                confidence=conf,
            )
            r.timestamp = now
            kg.relations.append(r)

        surprise = self_surprise.get("overall_surprise", 0.0)
        outcome_avg = assessment.get("trends", {}).get("action_score_avg", 0.5)
        if surprise > 0.5 and outcome_avg < 0.4:
            _add_rel("self_model", "negatively_impacts", "action_system")
        if surprise < 0.3 and outcome_avg > 0.6:
            _add_rel("self_model", "supports", "action_system")

        streak = assessment.get("emotion_streak", 0)
        if streak > 8:
            _add_rel("emotion_system", "shows_rigidity_in", "coping")

        _add_rel("knowledge_system", "informs", "action_system")
        _add_rel("emotion_system", "modulates", "action_system")

        return kg

    def record_inspection_benefit(self, outcome_before: float, outcome_after: float):
        benefit = outcome_after - outcome_before
        self._inspection_benefit_history.append(benefit)
        if len(self._inspection_benefit_history) > 50:
            self._inspection_benefit_history = self._inspection_benefit_history[-50:]

        if len(self._inspection_benefit_history) >= 10:
            avg_benefit = sum(self._inspection_benefit_history[-10:]) / 10.0
            if avg_benefit > 0.05:
                self._inspection_interval = max(2, self._inspection_interval - 1)
            elif avg_benefit < -0.02:
                self._inspection_interval = min(20, self._inspection_interval + 1)

    # ------------------------------------------------------------------
    # Trend computation (unchanged baseline)
    # ------------------------------------------------------------------

    def _compute_trends(self) -> Dict[str, Any]:
        if len(self._history) < 3:
            return {}

        recent = self._history[-self.window :]
        half = len(recent) // 2
        if half == 0:
            return {}
        first_half = recent[:half]
        second_half = recent[half:]

        def avg(lst, key):
            vals = [s.get(key, 0.0) for s in lst]
            return sum(vals) / max(len(vals), 1)

        pe_first = avg(first_half, "prediction_error")
        pe_second = avg(second_half, "prediction_error")

        as_first = avg(first_half, "action_score")
        as_second = avg(second_half, "action_score")

        emotions = [s["emotion"] for s in recent]
        unique_emotions = len(set(emotions))
        emotion_stability = 1.0 - min(1.0, unique_emotions / max(len(emotions), 1))

        override_count = sum(1 for s in recent if s.get("appraisal_overrides"))
        override_rate = override_count / max(len(recent), 1)

        _ss_first = avg(first_half, "self_surprise")
        ss_second = avg(second_half, "self_surprise")
        spa_first = avg(first_half, "self_prediction_accuracy")
        spa_second = avg(second_half, "self_prediction_accuracy")

        return {
            "prediction_error_trend": round(pe_second - pe_first, 4),
            "prediction_error_avg": round(pe_second, 4),
            "action_score_trend": round(as_second - as_first, 4),
            "action_score_avg": round(as_second, 4),
            "emotion_stability": round(emotion_stability, 3),
            "drift_avg": round(avg(second_half, "drift"), 4),
            "curiosity_avg": round(avg(second_half, "curiosity_score"), 4),
            "learned_weight": round(avg(second_half, "learned_weight"), 4),
            "coping_avg": round(avg(second_half, "appraisal_coping"), 4),
            "appraisal_override_rate": round(override_rate, 3),
            "self_surprise_avg": round(ss_second, 4),
            "self_knowledge_trend": round(spa_second - spa_first, 4),
            "self_prediction_accuracy": round(spa_second, 4),
            "sample_size": len(recent),
        }

    def _reconciliation_meta(self, reconciliation: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Turn multi-agent belief merge + cohesion into explicit meta insights/concerns."""
        insights: List[str] = []
        concerns: List[str] = []
        if not reconciliation:
            return insights, concerns
        bt = reconciliation.get("belief_tree")
        if not isinstance(bt, dict):
            return insights, concerns
        co = bt.get("belief_cohesion")
        if isinstance(co, dict):
            ac = int(co.get("agent_count") or 0)
            sb = int(co.get("shared_belief_count") or 0)
            cc = int(co.get("consensus_count") or 0)
            ur = co.get("unresolved_counts") or {}
            mx = int(max(ur.values()) if ur else 0)
            if ac >= 2 and sb > 0:
                insights.append(
                    f"Across {ac} inner voice(s), {sb} belief(s) align "
                    f"({cc} reach consensus) - partial unity in my model of truth."
                )
            if ac >= 2 and sb == 0:
                concerns.append(
                    "My inner voices share no overlapping beliefs this cycle - "
                    "possible fragmentation."
                )
            if mx >= 3:
                concerns.append(
                    f"Belief splits run deep: up to {mx} unresolved proposition(s) per inner voice."
                )
        merged = int(bt.get("merged_nodes", 0) or 0)
        if merged > 0:
            insights.append(f"I merged {merged} belief-related memory node(s) into a common view.")
        return insights, concerns

    def _generate_insights(self, trends: Dict, snapshot: Dict) -> List[str]:
        insights: List[str] = []
        if not trends:
            return insights

        pe_trend = trends.get("prediction_error_trend", 0.0)
        if pe_trend < -0.03:
            insights.append(f"My predictions are improving -- error trend {pe_trend:+.3f}")
        elif pe_trend > 0.03:
            insights.append(f"My predictions are degrading -- error trend {pe_trend:+.3f}")

        as_trend = trends.get("action_score_trend", 0.0)
        if as_trend > 0.02:
            insights.append(
                f"My actions are becoming more effective -- score trend {as_trend:+.3f}"
            )

        lw = trends.get("learned_weight", 0.0)
        if lw > 0.3:
            insights.append(f"I am relying more on learned emotion (weight {lw:.2f})")

        stab = trends.get("emotion_stability", 0.0)
        if stab > 0.8:
            insights.append("My emotional state has been very stable recently")

        override_rate = trends.get("appraisal_override_rate", 0.0)
        if override_rate > 0.6:
            insights.append(
                f"Appraisal is overriding keyword emotion "
                f"{override_rate:.0%} of the time -- "
                f"goals and context are driving my affect"
            )

        coping_avg = trends.get("coping_avg", 0.5)
        if coping_avg > 0.7:
            insights.append(
                f"My coping potential is high ({coping_avg:.2f}) "
                f"-- I feel capable of handling what arises"
            )

        ss_avg = trends.get("self_surprise_avg", 0.5)
        sk_trend = trends.get("self_knowledge_trend", 0.0)
        sample = trends.get("sample_size", 0)
        if ss_avg < 0.3 and sample > 20:
            insights.append("I am becoming predictable to myself -- I know my own patterns")
        if sk_trend > 0.02 and sample > 10:
            insights.append(f"My self-knowledge is improving (accuracy trend {sk_trend:+.3f})")

        if self.learned_weight > 0.3:
            insights.append(
                f"My metacognition is learning from experience "
                f"({self.learned_weight:.0%} self-calibrated)"
            )

        return insights

    def _generate_concerns(self, trends: Dict, snapshot: Dict) -> List[str]:
        concerns: List[str] = []
        if not trends:
            return concerns

        if self._emotion_streak > 15:
            concerns.append(
                f"I have been in '{self._last_emotion}' for "
                f"{self._emotion_streak} consecutive cycles"
            )

        pe = trends.get("prediction_error_avg", 0.0)
        if pe > 0.7:
            concerns.append(f"World model prediction error is very high ({pe:.2f})")

        drift = trends.get("drift_avg", 0.0)
        if drift > 0.4:
            concerns.append(f"Persistent identity drift ({drift:.2f})")

        asc = trends.get("action_score_avg", 0.0)
        if asc < 0.3:
            concerns.append(f"Action outcomes are consistently poor ({asc:.2f})")

        coping_avg = trends.get("coping_avg", 0.5)
        if coping_avg < 0.35:
            concerns.append(
                f"Coping potential is consistently low ({coping_avg:.2f}) "
                f"-- may need new knowledge or strategy"
            )

        ss_avg = trends.get("self_surprise_avg", 0.5)
        sample = trends.get("sample_size", 0)
        if ss_avg > 0.6 and sample > 20:
            concerns.append(
                f"I keep surprising myself -- my self-model is unreliable ({ss_avg:.2f})"
            )

        return concerns

    def _compute_self_score(self, trends: Dict) -> float:
        if not trends:
            return 0.5

        pe = 1.0 - trends.get("prediction_error_avg", 0.5)
        action = trends.get("action_score_avg", 0.5)
        stability = trends.get("emotion_stability", 0.5)
        no_drift = 1.0 - min(1.0, trends.get("drift_avg", 0.0))

        w = self._self_score_weights
        return (
            pe * w["prediction_error"]
            + action * w["action_score"]
            + stability * w["emotion_stability"]
            + no_drift * w["no_drift"]
        )

    def summarize(self) -> Dict[str, Any]:
        trends = self._compute_trends()
        return {
            "history_length": len(self._history),
            "self_score": round(self._compute_self_score(trends), 3),
            "emotion_streak": self._emotion_streak,
            "emotion_streak_label": self._last_emotion,
            "trends": trends,
            "learned_weight": round(self.learned_weight, 3),
            "self_score_weights": {k: round(v, 3) for k, v in self._self_score_weights.items()},
            "transformer_train_steps": self._transformer_train_steps,
            "transformer_available": self._transformer is not None,
            "seq_buffer_size": len(self._transformer_seq_buffer),
        }

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "history": self._history[-(self.window * 2) :],
            "train_steps": self._train_steps,
            "emotion_streak": self._emotion_streak,
            "last_emotion": self._last_emotion,
            "self_score_weights": self._self_score_weights,
            "threshold_adjustments": self._threshold_adjustments,
            "inspection_count": self._inspection_count,
            "inspection_interval": self._inspection_interval,
        }
        if self.available and self._transformer is not None:
            data["transformer_state"] = {
                k: v.tolist() for k, v in self._transformer.state_dict().items()
            }
            data["transformer_train_steps"] = self._transformer_train_steps
        return data

    def from_dict(self, data: Dict[str, Any]):
        self._history = data.get("history", [])
        self._train_steps = data.get("train_steps", 0)
        self._emotion_streak = data.get("emotion_streak", 0)
        self._last_emotion = data.get("last_emotion", "neutral")
        self._inspection_count = data.get("inspection_count", 0)
        self._inspection_interval = data.get("inspection_interval", 5)

        saved_weights = data.get("self_score_weights")
        if saved_weights and isinstance(saved_weights, dict):
            self._self_score_weights.update(saved_weights)

        saved_thresh = data.get("threshold_adjustments")
        if saved_thresh and isinstance(saved_thresh, dict):
            self._threshold_adjustments = saved_thresh

        self._transformer_train_steps = data.get("transformer_train_steps", 0)
        if self.available and self._transformer is not None:
            t_state = data.get("transformer_state")
            if t_state:
                try:
                    state = {k: torch.tensor(v) for k, v in t_state.items()}
                    self._transformer.load_state_dict(state)
                    self._transformer.eval()
                except Exception as _e:
                    print(f"[MetaCognition] from_dict transformer load failed: {_e}", flush=True)
