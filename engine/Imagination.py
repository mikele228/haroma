"""
Imagination — Internal Simulation for HaromaX6 (Phase 12).

Runs lightweight "mental cycles" using the system's own learned models
to simulate hypothetical scenarios before committing to action.  This
enables:

  Foresight    — predict outcomes of candidate actions
  Planning     — evaluate multi-step action sequences
  Creativity   — generate novel scenarios from memory + curiosity
  Counterfactual replay — re-imagine past events with different choices

The simulation engine does NOT run the full cognitive loop.  Instead it
uses the SelfModel, NeuralEncoder, and EmbodiedModulation as fast
forward-models to approximate what would happen, then scores each
scenario with a learned quality predictor.

If PyTorch is unavailable, the engine degrades to a heuristic
imagination based on keyword/memory overlap.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import hashlib
import random
import math
import threading

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except ImportError:
    _TORCH = False

from engine.ComputeFabric import get_fabric as _get_fabric


_SCENARIO_INPUT_DIM = 16
_STRATEGIES = ["inform", "inquire", "empathize", "advance_goal", "reflect"]

if _TORCH:

    class _OutcomeSimulatorNet(nn.Module):
        """Predicts (outcome_quality, emotional_shift, surprise_level) for
        a hypothetical state-action pair.

        Input (16-d): content_embed_summary(4) + valence + arousal +
                      curiosity + dominant_drive + wm_load + outcome_prev +
                      strategy_onehot(5) + has_external
        Output (3-d): predicted outcome score, predicted valence shift,
                      predicted surprise
        """

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(_SCENARIO_INPUT_DIM, 32),
                nn.ReLU(),
                nn.Linear(32, 24),
                nn.ReLU(),
                nn.Linear(24, 3),
                nn.Tanh(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


_ROLLOUT_INPUT_DIM = 16
_ROLLOUT_HIDDEN = 32

if _TORCH:

    class _SequenceRolloutNet(nn.Module):
        """GRU that predicts multi-step outcomes by feeding predicted state
        back as input for the next step."""

        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                input_size=_ROLLOUT_INPUT_DIM,
                hidden_size=_ROLLOUT_HIDDEN,
                num_layers=1,
                batch_first=True,
            )
            self.output_proj = nn.Linear(_ROLLOUT_HIDDEN, 3)
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

        def forward(
            self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """x: (1, seq_len, 16), returns (1, seq_len, 3), hidden."""
            out, h = self.gru(x, hidden)
            pred = torch.tanh(self.output_proj(out))
            return pred, h


_STATE_DIM = 256
_STRATEGY_EMBED_DIM = 32
_TRANSITION_INPUT_DIM = _STATE_DIM + _STRATEGY_EMBED_DIM
_TRANSITION_HIDDEN = 256
_N_LLM_ACTION_SLOTS = 16
_N_STRATEGIES_FULL = len(_STRATEGIES) + _N_LLM_ACTION_SLOTS
_DISCOUNT = 0.9
_ROLLOUT_STEPS = 3

if _TORCH:

    class _StateTransitionNet(nn.Module):
        """GRU-based state transition model that predicts next-state
        embeddings in full 256-d space with a separate reward head.

        Supports multi-step rollouts by feeding predicted states back.
        """

        def __init__(self, num_strategies: int = _N_STRATEGIES_FULL):
            super().__init__()
            self.strategy_embed = nn.Embedding(num_strategies, _STRATEGY_EMBED_DIM)
            self.gru = nn.GRU(
                input_size=_TRANSITION_INPUT_DIM,
                hidden_size=_TRANSITION_HIDDEN,
                num_layers=1,
                batch_first=True,
            )
            self.state_proj = nn.Linear(_TRANSITION_HIDDEN, _STATE_DIM)
            self.reward_head = nn.Sequential(
                nn.Linear(_TRANSITION_HIDDEN, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh(),
            )
            nn.init.xavier_uniform_(self.state_proj.weight)

        def forward(
            self,
            state: torch.Tensor,
            strategy_idx: torch.Tensor,
            hidden: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Single-step transition.

            Args:
                state: (batch, state_dim)
                strategy_idx: (batch,) long
                hidden: optional GRU hidden

            Returns:
                next_state: (batch, state_dim)
                reward: (batch, 1)
                new_hidden: GRU hidden
            """
            s_emb = self.strategy_embed(strategy_idx)  # (batch, 32)
            inp = torch.cat([state, s_emb], dim=-1).unsqueeze(1)  # (batch, 1, input)
            out, h = self.gru(inp, hidden)
            out = out.squeeze(1)  # (batch, hidden)
            next_state = self.state_proj(out)
            reward = self.reward_head(out)
            return next_state, reward, h

        def rollout(
            self,
            state: torch.Tensor,
            strategy_idx: int,
            steps: int = _ROLLOUT_STEPS,
        ) -> List[Tuple[torch.Tensor, float]]:
            """Multi-step rollout feeding predicted state back."""
            results = []
            current = state.unsqueeze(0) if state.dim() == 1 else state
            s_idx = torch.tensor([strategy_idx], dtype=torch.long)
            hidden = None
            for _ in range(steps):
                next_s, reward, hidden = self.forward(current, s_idx, hidden)
                results.append((next_s.squeeze(0).detach(), reward.item()))
                current = next_s.detach()
            return results


class _MCTSNode:
    """Node in the Monte Carlo Tree Search tree."""

    __slots__ = ("state", "strategy", "parent", "children", "visits", "value", "depth")

    def __init__(self, state, strategy: Optional[str], parent: Optional["_MCTSNode"]):
        self.state = state
        self.strategy = strategy
        self.parent = parent
        self.children: List["_MCTSNode"] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.depth: int = (parent.depth + 1) if parent else 0

    def ucb1(self, c: float, parent_visits: int) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.value / self.visits
        explore = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploit + explore


class Scenario:
    """A single imagined scenario with its predicted consequences."""

    __slots__ = (
        "label",
        "strategy",
        "features",
        "predicted_outcome",
        "predicted_valence_shift",
        "predicted_surprise",
        "confidence",
        "source",
    )

    def __init__(
        self, label: str, strategy: str, features: List[float], source: str = "imagination"
    ):
        self.label = label
        self.strategy = strategy
        self.features = features
        self.source = source
        self.predicted_outcome = 0.0
        self.predicted_valence_shift = 0.0
        self.predicted_surprise = 0.0
        self.confidence = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "strategy": self.strategy,
            "predicted_outcome": round(self.predicted_outcome, 3),
            "predicted_valence_shift": round(self.predicted_valence_shift, 3),
            "predicted_surprise": round(self.predicted_surprise, 3),
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }


class Imagination:
    _RAMP_STEPS = 100
    _MAX_WEIGHT = 0.9
    _MAX_SCENARIOS = 5

    def __init__(
        self, encoder=None, llm_backend=None, buffer_lock: Optional[threading.Lock] = None
    ):
        self._encoder = encoder
        self.available = _TORCH
        self._train_steps = 0
        self._train_buffer: List[Tuple[List[float], List[float]]] = []
        self._buffer_cap = 256
        self._rolling_accuracy: List[float] = []
        self._accuracy_window = 50
        self._buf_lock = buffer_lock if buffer_lock is not None else threading.Lock()

        self._last_simulation: List[Dict[str, Any]] = []
        self._best_imagined_strategy: str = ""
        self._simulation_count = 0

        self._rollout_train_steps = 0
        self._rollout_buffer: List[Dict[str, Any]] = []
        self._last_plan: List[str] = []
        self._plan_horizon = 3

        self._transition_net = None
        self._transition_optim = None
        self._transition_train_steps = 0
        self._transition_buffer: List[Tuple[Any, int, Any, float]] = []
        self._transition_buffer_cap = 512

        self._llm_planner = None
        self._plan_context: Optional[Dict[str, Any]] = None
        if llm_backend is not None and llm_backend.available:
            self._llm_planner = llm_backend
        elif llm_backend is None:
            try:
                from engine.LLMBackend import LLMBackend

                self._llm_planner = LLMBackend()
                if not self._llm_planner.available:
                    self._llm_planner = None
            except Exception as _e:
                print(f"[Imagination] LLM planner init error: {_e}", flush=True)

        if _TORCH:
            self._net = _OutcomeSimulatorNet()
            self._rollout_net = _SequenceRolloutNet()
            self._transition_net = _StateTransitionNet()
            _fab = _get_fabric()
            if _fab:
                self._net = _fab.register("imagination_sim", self._net)
                self._rollout_net = _fab.register("imagination_rollout", self._rollout_net)
                self._transition_net = _fab.register("imagination_transition", self._transition_net)
            self._optim = torch.optim.Adam(self._net.parameters(), lr=1e-3)
            self._rollout_optim = torch.optim.Adam(self._rollout_net.parameters(), lr=5e-4)
            self._transition_optim = torch.optim.Adam(self._transition_net.parameters(), lr=5e-4)
        else:
            self._net = None
            self._optim = None
            self._rollout_net = None
            self._rollout_optim = None

    @property
    def learned_weight(self) -> float:
        progress = min(1.0, self._train_steps / max(self._RAMP_STEPS, 1))
        return progress * self._MAX_WEIGHT

    def _compress_embedding(self, embedding) -> List[float]:
        if embedding is None:
            return [0.0, 0.0, 0.0, 0.0]
        emb = list(embedding)
        if len(emb) < 4:
            return emb + [0.0] * (4 - len(emb))
        quarter = len(emb) // 4
        return [
            sum(emb[:quarter]) / max(quarter, 1),
            sum(emb[quarter : quarter * 2]) / max(quarter, 1),
            sum(emb[quarter * 2 : quarter * 3]) / max(quarter, 1),
            sum(emb[quarter * 3 :]) / max(len(emb) - quarter * 3, 1),
        ]

    def _build_scenario_features(
        self,
        embedding_summary: List[float],
        valence: float,
        arousal: float,
        curiosity: float,
        dominant_drive: float,
        wm_load: float,
        outcome_prev: float,
        strategy: str,
        has_external: float,
    ) -> List[float]:
        strat_oh = [0.0] * len(_STRATEGIES)
        if strategy in _STRATEGIES:
            strat_oh[_STRATEGIES.index(strategy)] = 1.0

        return (
            embedding_summary[:4]
            + [valence, arousal, curiosity, dominant_drive, wm_load, outcome_prev]
            + strat_oh
            + [has_external]
        )

    def _simulate_via_transition(self, content_embedding, **kwargs) -> Optional[Dict[str, Any]]:
        """State-transition simulation using full 256-d embeddings and
        multi-step GRU rollouts (Upgrade 4)."""
        if self._transition_net is None or content_embedding is None:
            return None

        try:
            state = torch.tensor(list(content_embedding), dtype=torch.float32)
            if state.shape[0] != _STATE_DIM:
                return None
        except Exception:
            return None

        scenarios: List[Scenario] = []

        for si, strategy in enumerate(_STRATEGIES):
            scenario = Scenario(
                label=f"What if I {strategy}?",
                strategy=strategy,
                features=[],
                source="transition_rollout",
            )

            with torch.no_grad():
                rollout = self._transition_net.rollout(state, si, steps=_ROLLOUT_STEPS)

            discounted_reward = 0.0
            for step_i, (_, r) in enumerate(rollout):
                discounted_reward += r * (_DISCOUNT**step_i)

            scenario.predicted_outcome = (discounted_reward / _ROLLOUT_STEPS + 1.0) / 2.0
            scenario.predicted_outcome = max(0.0, min(1.0, scenario.predicted_outcome))
            scenario.confidence = min(0.9, self._transition_train_steps / 200.0)
            scenarios.append(scenario)

        self._apply_temporal_penalties(
            scenarios,
            kwargs.get("recent_emotions"),
            kwargs.get("repetition_flag"),
        )
        scenarios.sort(key=lambda s: s.predicted_outcome, reverse=True)
        scenarios = scenarios[: self._MAX_SCENARIOS]

        best = scenarios[0] if scenarios else None
        self._best_imagined_strategy = best.strategy if best else ""
        self._last_simulation = [s.to_dict() for s in scenarios]
        self._simulation_count += 1

        worst = scenarios[-1] if len(scenarios) > 1 else None
        spread = (best.predicted_outcome - worst.predicted_outcome) if best and worst else 0.0

        _quality = (
            min(
                1.0,
                (len(scenarios) / max(1, self._MAX_SCENARIOS))
                * (0.5 + 0.5 * spread)
                * self.learned_weight,
            )
            if scenarios
            else 0.0
        )

        result = {
            "scenarios": self._last_simulation,
            "best_strategy": self._best_imagined_strategy,
            "best_predicted_outcome": round(best.predicted_outcome if best else 0.0, 3),
            "scenario_count": len(scenarios),
            "outcome_spread": round(spread, 3),
            "simulation_count": self._simulation_count,
            "imagination_weight": round(self.learned_weight, 3),
            "quality": round(_quality, 3),
            "method": "state_transition",
        }

        mcts_context = {
            k: kwargs[k]
            for k in ("emotion", "goals", "recent_emotions", "drives", "working_memory")
            if k in kwargs
        } or None
        mcts_result = self.mcts_plan(content_embedding, context=mcts_context)
        result["imagined_plan"] = mcts_result.get("plan", [])
        result["plan_score"] = mcts_result.get("plan_score", 0.0)
        result["mcts"] = mcts_result
        if mcts_result.get("plan"):
            self._last_plan = mcts_result["plan"]

        return result

    def simulate(
        self,
        *,
        content_embedding,
        valence: float,
        arousal: float,
        curiosity: float,
        dominant_drive: float,
        wm_load: float,
        outcome_prev: float,
        has_external: float,
        recalled_memories: List[Any] = None,
        curiosity_questions: List[str] = None,
        recent_emotions: Optional[List[str]] = None,
        repetition_flag: Optional[str] = None,
        emotion: Optional[Dict[str, Any]] = None,
        goals: Optional[List[Dict[str, Any]]] = None,
        drives: Optional[Dict[str, Any]] = None,
        working_memory: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:

        if self._transition_net is not None and content_embedding is not None:
            transition_result = self._simulate_via_transition(
                content_embedding,
                recent_emotions=recent_emotions,
                repetition_flag=repetition_flag,
                emotion=emotion,
                goals=goals,
                drives=drives,
                working_memory=working_memory,
            )
            if transition_result is not None:
                return transition_result

        embed_summary = self._compress_embedding(content_embedding)

        scenarios: List[Scenario] = []

        for strategy in _STRATEGIES:
            features = self._build_scenario_features(
                embed_summary,
                valence,
                arousal,
                curiosity,
                dominant_drive,
                wm_load,
                outcome_prev,
                strategy,
                has_external,
            )

            scenario = Scenario(
                label=f"What if I {strategy}?",
                strategy=strategy,
                features=features,
                source="strategy_exploration",
            )
            self._predict_scenario(scenario)
            scenarios.append(scenario)

        if curiosity_questions:
            for q in curiosity_questions[:2]:
                features = self._build_scenario_features(
                    embed_summary,
                    valence,
                    arousal,
                    min(1.0, curiosity + 0.2),
                    dominant_drive,
                    wm_load,
                    outcome_prev,
                    "inquire",
                    has_external,
                )
                scenario = Scenario(
                    label=f"What if I pursue: {q[:60]}?",
                    strategy="inquire",
                    features=features,
                    source="curiosity_driven",
                )
                self._predict_scenario(scenario)
                scenarios.append(scenario)

        if recalled_memories and len(recalled_memories) > 0:
            mem = recalled_memories[0]
            mem_content = mem.get("content", "") if isinstance(mem, dict) else str(mem)
            if mem_content and self._encoder:
                mem_embed = self._encoder.encode(mem_content[:80])
                mem_summary = self._compress_embedding(mem_embed)
            else:
                mem_summary = embed_summary

            features = self._build_scenario_features(
                mem_summary,
                valence,
                arousal,
                curiosity,
                dominant_drive,
                wm_load,
                outcome_prev,
                "reflect",
                has_external,
            )
            scenario = Scenario(
                label="What if I revisit this memory?",
                strategy="reflect",
                features=features,
                source="memory_replay",
            )
            self._predict_scenario(scenario)
            scenarios.append(scenario)

        self._apply_temporal_penalties(scenarios, recent_emotions, repetition_flag)

        scenarios.sort(key=lambda s: s.predicted_outcome, reverse=True)
        scenarios = scenarios[: self._MAX_SCENARIOS]

        best = scenarios[0] if scenarios else None
        self._best_imagined_strategy = best.strategy if best else ""
        self._last_simulation = [s.to_dict() for s in scenarios]
        self._simulation_count += 1

        worst = scenarios[-1] if len(scenarios) > 1 else None
        spread = (best.predicted_outcome - worst.predicted_outcome) if best and worst else 0.0

        best_plan: List[str] = []
        plan_score = 0.0
        horizon = min(self._plan_horizon, 1 + self._rollout_train_steps // 20)
        if horizon > 1 and self._rollout_net is not None and self.available:
            top_strats = [s.strategy for s in scenarios[:3]]
            candidate_plans = self._generate_candidate_plans(top_strats, horizon)

            best_total = -999.0
            for plan in candidate_plans:
                result = self.simulate_sequence(
                    embed_summary,
                    plan,
                    valence=valence,
                    arousal=arousal,
                    curiosity=curiosity,
                    dominant_drive=dominant_drive,
                    wm_load=wm_load,
                    outcome_prev=outcome_prev,
                    has_external=has_external,
                )
                total = result.get("total_predicted_outcome", 0.0)
                if total > best_total:
                    best_total = total
                    best_plan = list(plan)
                    plan_score = total

        self._last_plan = best_plan

        _quality = (
            min(
                1.0,
                (len(scenarios) / max(1, self._MAX_SCENARIOS))
                * (0.5 + 0.5 * spread)
                * self.learned_weight,
            )
            if scenarios
            else 0.0
        )

        return {
            "scenarios": self._last_simulation,
            "best_strategy": self._best_imagined_strategy,
            "best_predicted_outcome": round(best.predicted_outcome if best else 0.0, 3),
            "scenario_count": len(scenarios),
            "outcome_spread": round(spread, 3),
            "simulation_count": self._simulation_count,
            "imagination_weight": round(self.learned_weight, 3),
            "imagined_plan": best_plan,
            "plan_score": round(plan_score, 3),
            "quality": round(_quality, 3),
            "method": "scenarios",
        }

    def _predict_scenario(self, scenario: Scenario):
        if not self.available or self._net is None:
            scenario.predicted_outcome = 0.5
            scenario.confidence = 0.1
            return

        _fab = _get_fabric()
        with torch.no_grad():
            x = (
                _fab.tensor([scenario.features])
                if _fab
                else torch.tensor([scenario.features], dtype=torch.float32)
            )
            out = self._net(x).squeeze(0)

        scenario.predicted_outcome = (out[0].item() + 1.0) / 2.0
        scenario.predicted_valence_shift = out[1].item()
        scenario.predicted_surprise = (out[2].item() + 1.0) / 2.0
        scenario.confidence = self.learned_weight

    def get_strategy_recommendation(self) -> Optional[str]:
        if self.learned_weight < 0.15:
            return None
        return self._best_imagined_strategy or None

    def get_plan_recommendation(self) -> Optional[List[str]]:
        if not self._last_plan or self.learned_weight < 0.15:
            return None
        return list(self._last_plan)

    # ------------------------------------------------------------------
    # Multi-step temporal imagination (Phase 14)
    # ------------------------------------------------------------------

    def simulate_sequence(
        self,
        base_features: List[float],
        strategy_sequence: List[str],
        valence: float = 0.0,
        arousal: float = 0.0,
        curiosity: float = 0.0,
        dominant_drive: float = 0.0,
        wm_load: float = 0.0,
        outcome_prev: float = 0.5,
        has_external: float = 0.0,
    ) -> Dict[str, Any]:
        if not self.available or self._rollout_net is None:
            return {"steps": [], "total_predicted_outcome": 0.5}

        embed_summary = base_features[:4] if len(base_features) >= 4 else [0.0] * 4
        step_results: List[Dict[str, Any]] = []

        running_valence = valence
        running_outcome = outcome_prev

        _fab = _get_fabric()
        with torch.no_grad():
            hidden = None
            for step_i, strategy in enumerate(strategy_sequence):
                features = self._build_scenario_features(
                    embed_summary,
                    running_valence,
                    arousal,
                    curiosity,
                    dominant_drive,
                    wm_load,
                    running_outcome,
                    strategy,
                    has_external,
                )

                x = (
                    _fab.tensor([[features]])
                    if _fab
                    else torch.tensor([[features]], dtype=torch.float32)
                )
                pred, hidden = self._rollout_net(x, hidden)
                pred = pred.squeeze(0).squeeze(0)

                pred_outcome = (pred[0].item() + 1.0) / 2.0
                pred_valence_shift = pred[1].item()
                pred_surprise = (pred[2].item() + 1.0) / 2.0

                step_results.append(
                    {
                        "step": step_i,
                        "strategy": strategy,
                        "predicted_outcome": round(pred_outcome, 3),
                        "predicted_valence_shift": round(pred_valence_shift, 3),
                        "predicted_surprise": round(pred_surprise, 3),
                    }
                )

                running_valence = max(-1.0, min(1.0, running_valence + pred_valence_shift * 0.5))
                running_outcome = pred_outcome

        total = sum(s["predicted_outcome"] for s in step_results) / max(len(step_results), 1)

        return {
            "steps": step_results,
            "total_predicted_outcome": round(total, 3),
            "strategy_sequence": list(strategy_sequence),
        }

    def _generate_candidate_plans(
        self, top_strategies: List[str], horizon: int = 3
    ) -> List[List[str]]:
        plans: List[List[str]] = []
        for lead in top_strategies[:3]:
            others = [s for s in _STRATEGIES if s != lead]
            random.shuffle(others)
            plan = [lead] + others[: horizon - 1]
            plans.append(plan)
        plans.append(top_strategies[:horizon])
        if len(top_strategies) >= 2:
            plans.append([top_strategies[1], top_strategies[0], top_strategies[0]])
        return plans[:5]

    # ------------------------------------------------------------------
    # MCTS Planning (Upgrade 11)
    # ------------------------------------------------------------------

    def mcts_plan(
        self,
        state_embedding,
        horizon: int = 4,
        simulations: int = 100,
        exploration_weight: float = 1.4,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Monte Carlo Tree Search with LLM-generated action candidates (U17).

        Builds a tree of strategy sequences using the transition net
        for simulation and UCB1 for selection.  At the root level,
        the LLM proposes additional action candidates beyond the fixed
        strategy set.
        """
        self._plan_context = context
        if (
            not self.available
            or self._transition_net is None
            or state_embedding is None
            or len(state_embedding) != _STATE_DIM
        ):
            return self._mcts_fallback(state_embedding)

        root = _MCTSNode(state=state_embedding, strategy=None, parent=None)

        for _ in range(simulations):
            node = self._mcts_select(root, exploration_weight)
            if node.visits > 0 and node.depth < horizon:
                node = self._mcts_expand(node, state_embedding)
            reward = self._mcts_simulate(node, horizon)
            self._mcts_backprop(node, reward)

        best_child = max(root.children, key=lambda c: c.visits) if root.children else None
        if best_child is None:
            return self._mcts_fallback(state_embedding)

        plan = []
        node = best_child
        while node is not None and node.strategy is not None:
            plan.append(node.strategy)
            node = max(node.children, key=lambda c: c.visits) if node.children else None

        plan_scores = []
        for child in root.children:
            plan_scores.append(
                {
                    "strategy": child.strategy,
                    "visits": child.visits,
                    "mean_reward": round(child.value / max(child.visits, 1), 3),
                }
            )
        plan_scores.sort(key=lambda x: x["visits"], reverse=True)

        return {
            "plan": plan,
            "plan_score": round(best_child.value / max(best_child.visits, 1), 3),
            "total_simulations": simulations,
            "tree_depth": horizon,
            "root_children": plan_scores[:5],
            "method": "mcts",
        }

    def _mcts_select(self, node: "_MCTSNode", c: float) -> "_MCTSNode":
        """Walk down the tree using UCB1 until reaching a leaf."""
        while node.children and node.visits > 0:
            node = max(node.children, key=lambda ch: ch.ucb1(c, node.visits))
        return node

    def _mcts_generate_llm_actions(self, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Ask the LLM for candidate actions beyond fixed strategies."""
        if self._llm_planner is None:
            return []
        try:
            prompt = (
                "Given the current cognitive state, suggest 3-5 concrete "
                "actions I could take. Each should be a short phrase.\n"
            )
            if context:
                if context.get("emotion"):
                    prompt += f"Current emotion: {context['emotion']}\n"
                if context.get("goals"):
                    goals = context["goals"][:3]
                    prompt += "Active goals: " + ", ".join(str(g) for g in goals) + "\n"
            prompt += (
                "\nList actions one per line, starting with ACTION:\nACTION: <action phrase>\n"
            )
            response = self._llm_planner.generate(prompt, max_tokens=256, temperature=0.7)
            if not response:
                return []
            actions = []
            for line in response.split("\n"):
                line = line.strip()
                if line.upper().startswith("ACTION:"):
                    action = line[len("ACTION:") :].strip()
                    if action and len(action) < 100:
                        actions.append(action)
            return actions[:5]
        except Exception:
            return []

    def _mcts_expand(self, node: "_MCTSNode", root_state) -> "_MCTSNode":
        """Add child nodes for fixed strategies + LLM-generated actions."""
        if node.children:
            return random.choice(node.children)

        state = node.state if node.state is not None else root_state

        for i, strategy in enumerate(_STRATEGIES):
            child_state = self._mcts_transition(state, i)
            child = _MCTSNode(state=child_state, strategy=strategy, parent=node)
            node.children.append(child)

        if node.depth == 0:
            llm_actions = self._mcts_generate_llm_actions(self._plan_context)
            for j, action in enumerate(llm_actions):
                si = len(_STRATEGIES) + (
                    int(hashlib.md5(action.encode()).hexdigest(), 16) % _N_LLM_ACTION_SLOTS
                )
                child_state = self._mcts_transition(state, si)
                child = _MCTSNode(state=child_state, strategy=action, parent=node)
                node.children.append(child)

        return random.choice(node.children) if node.children else node

    def _mcts_simulate(self, node: "_MCTSNode", horizon: int) -> float:
        """Random rollout from node to get a reward estimate."""
        if node.state is None or self._transition_net is None:
            return 0.5

        state_t = torch.tensor(list(node.state), dtype=torch.float32)
        total_reward = 0.0
        steps = horizon - node.depth

        with torch.no_grad():
            for _ in range(max(1, steps)):
                s_idx = random.randint(0, len(_STRATEGIES) - 1)
                results = self._transition_net.rollout(state_t, s_idx, steps=1)
                if results:
                    state_t = results[0][0]
                    total_reward += results[0][1]

        return max(0.0, min(1.0, (total_reward / max(steps, 1) + 1.0) / 2.0))

    def _mcts_transition(self, state, strategy_idx: int):
        """Single-step transition for tree expansion."""
        if self._transition_net is None:
            return state
        state_t = torch.tensor(list(state), dtype=torch.float32)
        with torch.no_grad():
            results = self._transition_net.rollout(state_t, strategy_idx, steps=1)
        if results:
            return results[0][0].tolist()
        return state

    @staticmethod
    def _mcts_backprop(node: "_MCTSNode", reward: float):
        """Propagate reward up the tree."""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _mcts_fallback(self, state_embedding) -> Dict[str, Any]:
        """Fallback when MCTS prerequisites not met."""
        return {
            "plan": [],
            "plan_score": 0.0,
            "total_simulations": 0,
            "tree_depth": 0,
            "root_children": [],
            "method": "mcts_unavailable",
        }

    def _apply_temporal_penalties(
        self,
        scenarios: List[Scenario],
        recent_emotions: Optional[List[str]] = None,
        repetition_flag: Optional[str] = None,
    ):
        if not recent_emotions and not repetition_flag:
            return

        if repetition_flag:
            pattern_type = repetition_flag.split(":")[0] if ":" in repetition_flag else ""
            for s in scenarios:
                if pattern_type == "action" and s.strategy == repetition_flag.split(":")[-1]:
                    s.predicted_outcome *= 0.7
                elif pattern_type:
                    s.predicted_outcome *= 0.9

    def record_sequence_outcome(
        self,
        plan: List[str],
        actual_outcomes: List[float],
        actual_valences: List[float],
        actual_surprises: List[float],
        base_features: List[float],
    ):
        if not self.available or not plan:
            return
        self._rollout_buffer.append(
            {
                "plan": plan,
                "outcomes": actual_outcomes,
                "valences": actual_valences,
                "surprises": actual_surprises,
                "base_features": base_features[:4],
            }
        )
        if len(self._rollout_buffer) > 128:
            self._rollout_buffer = self._rollout_buffer[-128:]

    def train_rollout_step(self) -> Optional[float]:
        if (
            not self.available
            or not self._rollout_buffer
            or self._rollout_net is None
            or len(self._rollout_buffer) < 4
        ):
            return None

        _fab = _get_fabric()
        batch = self._rollout_buffer[-16:]
        self._rollout_net.train()
        total_loss = 0.0
        n = 0

        for sample in batch:
            plan = sample["plan"]
            outcomes = sample["outcomes"]
            valences = sample["valences"]
            surprises = sample["surprises"]
            embed_summary = sample["base_features"]

            seq_len = min(len(plan), len(outcomes), len(valences), len(surprises))
            if seq_len == 0:
                continue

            input_features = []
            running_valence = 0.0
            running_outcome = 0.5

            for i in range(seq_len):
                f = self._build_scenario_features(
                    embed_summary,
                    running_valence,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    running_outcome,
                    plan[i],
                    0.5,
                )
                input_features.append(f)
                running_valence = valences[i] if i < len(valences) else 0.0
                running_outcome = outcomes[i] if i < len(outcomes) else 0.5

            targets = []
            for i in range(seq_len):
                targets.append(
                    [
                        outcomes[i] * 2.0 - 1.0,
                        max(-1.0, min(1.0, valences[i])),
                        surprises[i] * 2.0 - 1.0,
                    ]
                )

            x = (
                _fab.tensor([input_features])
                if _fab
                else torch.tensor([input_features], dtype=torch.float32)
            )
            tgt = _fab.tensor([targets]) if _fab else torch.tensor([targets], dtype=torch.float32)

            pred, _ = self._rollout_net(x)
            loss = nn.functional.mse_loss(pred, tgt)

            self._rollout_optim.zero_grad()
            if _fab:
                _fab.scale_loss(loss).backward()
                nn.utils.clip_grad_norm_(self._rollout_net.parameters(), 1.0)
                _fab.scaler_step(self._rollout_optim)
                _fab.scaler_update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self._rollout_net.parameters(), 1.0)
                self._rollout_optim.step()

            total_loss += loss.item()
            n += 1

        self._rollout_net.eval()
        self._rollout_train_steps += 1
        return total_loss / max(n, 1)

    def record_outcome(
        self,
        features: List[float],
        actual_outcome: float,
        actual_valence_shift: float,
        actual_surprise: float,
    ):
        if not self.available:
            return

        targets = [
            actual_outcome * 2.0 - 1.0,
            max(-1.0, min(1.0, actual_valence_shift)),
            actual_surprise * 2.0 - 1.0,
        ]
        with self._buf_lock:
            self._train_buffer.append((features, targets))
            if len(self._train_buffer) > self._buffer_cap:
                self._train_buffer = self._train_buffer[-self._buffer_cap :]

    def train_step(self) -> Optional[float]:
        if not self.available or self._net is None:
            return None
        with self._buf_lock:
            if len(self._train_buffer) < 8:
                return None
            batch_size = min(32, len(self._train_buffer))
            batch = random.sample(self._train_buffer, batch_size)

        _fab = _get_fabric()
        features = (
            _fab.tensor([b[0] for b in batch])
            if _fab
            else torch.tensor([b[0] for b in batch], dtype=torch.float32)
        )
        targets = (
            _fab.tensor([b[1] for b in batch])
            if _fab
            else torch.tensor([b[1] for b in batch], dtype=torch.float32)
        )

        predictions = self._net(features)
        loss = nn.functional.mse_loss(predictions, targets)

        self._optim.zero_grad()
        if _fab:
            _fab.scale_loss(loss).backward()
            nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
            _fab.scaler_step(self._optim)
            _fab.scaler_update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
            self._optim.step()

        self._train_steps += 1

        loss_val = loss.item()
        accuracy = 1.0 - min(1.0, loss_val)
        self._rolling_accuracy.append(accuracy)
        if len(self._rolling_accuracy) > self._accuracy_window:
            self._rolling_accuracy = self._rolling_accuracy[-self._accuracy_window :]

        return loss_val

    def record_transition(
        self,
        state_embedding,
        strategy: str,
        actual_next_state,
        actual_reward: float,
    ):
        """Record a real state transition for training _StateTransitionNet."""
        if self._transition_net is None or state_embedding is None or actual_next_state is None:
            return
        if strategy in _STRATEGIES:
            si = _STRATEGIES.index(strategy)
        else:
            si = len(_STRATEGIES) + (
                int(hashlib.md5(strategy.encode()).hexdigest(), 16) % _N_LLM_ACTION_SLOTS
            )
        with self._buf_lock:
            self._transition_buffer.append(
                (list(state_embedding), si, list(actual_next_state), actual_reward)
            )
            if len(self._transition_buffer) > self._transition_buffer_cap:
                self._transition_buffer = self._transition_buffer[-self._transition_buffer_cap :]

    def train_transition_step(self) -> Optional[float]:
        """Train the state-transition GRU from recorded transitions."""
        with self._buf_lock:
            if self._transition_net is None or len(self._transition_buffer) < 16:
                return None
            batch_size = min(32, len(self._transition_buffer))
            batch = random.sample(self._transition_buffer, batch_size)

        states = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        strategies = torch.tensor([b[1] for b in batch], dtype=torch.long)
        next_states = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        rewards = torch.tensor([[b[3]] for b in batch], dtype=torch.float32)

        self._transition_net.train()
        pred_next, pred_reward, _ = self._transition_net(states, strategies)

        loss_state = nn.functional.mse_loss(pred_next, next_states)
        loss_reward = nn.functional.mse_loss(pred_reward, rewards)
        loss = loss_state + 0.5 * loss_reward

        self._transition_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._transition_net.parameters(), 1.0)
        self._transition_optim.step()
        self._transition_net.eval()

        self._transition_train_steps += 1
        return loss.item()

    @property
    def overall_accuracy(self) -> float:
        if not self._rolling_accuracy:
            return 0.0
        return sum(self._rolling_accuracy) / len(self._rolling_accuracy)

    def stats(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "train_steps": self._train_steps,
            "learned_weight": round(self.learned_weight, 3),
            "simulation_count": self._simulation_count,
            "buffer_size": len(self._train_buffer),
            "accuracy": round(self.overall_accuracy, 3),
            "last_best_strategy": self._best_imagined_strategy,
            "last_scenario_count": len(self._last_simulation),
            "rollout_train_steps": self._rollout_train_steps,
            "rollout_buffer_size": len(self._rollout_buffer),
            "plan_horizon": self._plan_horizon,
            "last_plan": self._last_plan,
            "transition_train_steps": self._transition_train_steps,
            "transition_buffer_size": len(self._transition_buffer),
            "transition_available": self._transition_net is not None,
            "mcts_available": self._transition_net is not None,
        }

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "train_steps": self._train_steps,
            "simulation_count": self._simulation_count,
            "rollout_train_steps": self._rollout_train_steps,
            "transition_train_steps": self._transition_train_steps,
        }
        if self.available and self._net is not None:
            data["net_state"] = {k: v.tolist() for k, v in self._net.state_dict().items()}
        if self.available and self._rollout_net is not None:
            data["rollout_state"] = {
                k: v.tolist() for k, v in self._rollout_net.state_dict().items()
            }
        if self.available and self._transition_net is not None:
            data["transition_state"] = {
                k: v.tolist() for k, v in self._transition_net.state_dict().items()
            }
        return data

    def from_dict(self, data: Dict[str, Any]):
        self._train_steps = data.get("train_steps", 0)
        self._simulation_count = data.get("simulation_count", 0)
        self._rollout_train_steps = data.get("rollout_train_steps", 0)
        self._transition_train_steps = data.get("transition_train_steps", 0)

        if self.available and self._net is not None:
            net_state = data.get("net_state")
            if net_state:
                try:
                    state = {k: torch.tensor(v) for k, v in net_state.items()}
                    self._net.load_state_dict(state)
                except Exception as _e:
                    print(f"[Imagination] from_dict _net load failed: {_e}", flush=True)

        if self.available and self._rollout_net is not None:
            rollout_state = data.get("rollout_state")
            if rollout_state:
                try:
                    state = {k: torch.tensor(v) for k, v in rollout_state.items()}
                    self._rollout_net.load_state_dict(state)
                except Exception as _e:
                    print(f"[Imagination] from_dict _rollout_net load failed: {_e}", flush=True)

        if self.available and self._transition_net is not None:
            transition_state = data.get("transition_state")
            if transition_state:
                try:
                    state = {k: torch.tensor(v) for k, v in transition_state.items()}
                    self._transition_net.load_state_dict(state)
                except Exception as _e:
                    print(f"[Imagination] from_dict _transition_net load failed: {_e}", flush=True)
