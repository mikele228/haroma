# Reference: training, RL, simulation, and HTTP bridges

[<- Back to Index](index.md)

This page consolidates **installation**, **background training modules**, **environment variables**, and **integration entry points** for Gymnasium/RLlib, Vowpal Wabbit, JSONL ingest, pluggable simulation backends, and the stdlib HTTP client. For step-by-step robot wiring, see [Robot integration](robot-integration.md).

---

## 1. Installation matrix

| File / command | Contents |
|----------------|----------|
| [`requirements.txt`](../requirements.txt) | Full stack including `llama-cpp-python` (may need a C++ toolchain on Windows). |
| [`requirements-core.txt`](../requirements-core.txt) | Same without local GGUF bindings; use with API-only LLM. |
| [`requirements-rl.txt`](../requirements-rl.txt) | `gymnasium`, `scikit-learn` (Gymnasium bridge / offline scorer). |
| [`requirements-training-extras.txt`](../requirements-training-extras.txt) | Includes `-r requirements-rl.txt` + `vowpalwabbit` (optional VW online reward). |
| [`requirements-dev.txt`](../requirements-dev.txt) | Dev/test tooling (if present). |
| [`scripts/install_max_training.ps1`](../scripts/install_max_training.ps1) | Windows one-shot: `requirements.txt` → training extras → spaCy model → `download_training_data.py`. |

**Recommended order for “maximum training surfaces”:**

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install -r requirements-training-extras.txt
python scripts/download_training_data.py
```

**spaCy:** `python -m spacy download en_core_web_sm`

**Optional Ray (offline RLlib workflows on JSONL, not in-process training):** `pip install "ray[rllib]"`

**Vowpal Wabbit on Windows:** If `import vowpalwabbit` crashes, omit VW or fix VC++ runtime; Haroma runs without it.

---

## 2. Resource tier and scheduler registration

[`engine/ResourceAdaptiveConfig.py`](../engine/ResourceAdaptiveConfig.py) sets `training.enabled` from **hardware tier**:

| Tier | Condition (simplified) | `training.enabled` |
|------|-------------------------|---------------------|
| **0** | RAM &lt; 4 GB **or** Raspberry Pi | **false** — `TrainingScheduler` does **not** register neural modules in [`agents/shared_resources.py`](../agents/shared_resources.py). |
| **≥ 1** | Otherwise | **true** — modules registered with base intervals (encoder, backbone, …, `llm_reward`). |

**Background agent:** [`soul/agents.json`](../soul/agents.json) → `background.training_enabled` (default `true`). Env **`HAROMA_BENCH_DISABLE_BG_TRAINING=1`** disables `_run_training()` in [`agents/background_agent.py`](../agents/background_agent.py).

---

## 3. Background training map (full list)

One pass = ordered `train_step` calls from [`core/training_surface.py`](../core/training_surface.py) `build_background_train_map`, gated by [`engine/TrainingScheduler.py`](../engine/TrainingScheduler.py) `should_train(name)` per module.

| `module_name` | Source |
|----------------|--------|
| `encoder` | Contrastive step from memory + action-memory text (`_encoder_train_step`). |
| `backbone` | `shared.backbone.train_step()` |
| `attention` | `shared.attention.train_step()` |
| `process_gate` | `shared.process_gate.train_step()` |
| `self_model` | `_self_model_background_train_step` from `_self_model_last_train_ctx` |
| `appraisal` | `shared.appraisal.train_step()` |
| `modulation` | `shared.modulation.train_step()` |
| `goal_synth` | `shared.goal_synthesizer.train_step()` |
| `imagination` | `shared.imagination.train_step()` if available |
| `metacog` | `shared.metacognition.train_step()` |
| `composer` | `shared.composer.train_step()` |
| `generative` | `shared.composer.train_generative_step()` |
| `counterfactual` | `shared.counterfactual.train_step()` if gate available |
| `grounder` | `_grounder_train_step` |
| `mental_sim` | `shared.mental_simulator.train_step()` if available |
| `arch_search` | `shared.arch_searcher.train_step()` if available |
| `llm_reward` | `shared.llm_backend.train_reward_model()` — PyTorch reward + optional VW + optional [`vw_jsonl_ingest`](../mind/training/vw_jsonl_ingest.py) |

Many steps return **`None`** if preconditions fail (insufficient data, `CognitiveNull`, `.available` false). [`core/cognitive_null.py`](../core/cognitive_null.py) and `skip_modules` in resource config can stub engines.

---

## 4. LLM reward path (`llm_reward`)

[`engine/LLMBackend.py`](../engine/LLMBackend.py):

- **`record_outcome(...)`** — feeds PyTorch `RewardModel`, optional `VowpalWabbitRewardTrainer.record`, optional `RLlibTransitionLogger.record`.
- **`train_reward_model()`** — `reward_model.train_step()`; optional `ingest_bandit_jsonl_into_vw` when `HAROMA_VW_BANDIT_INGEST_PATH` is set; then `VowpalWabbitRewardTrainer.train_step()`.

**Finetune / LoRA data:** `FinetuneDataCollector` + `save_finetune_data()` / background flush — exports JSONL for **external** training, not full in-process LoRA.

---

## 5. Environment variables (complete tables)

### 5.1 Vowpal Wabbit + RLlib JSONL ([`mind/training/vw_rl_bridge.py`](../mind/training/vw_rl_bridge.py))

| Variable | Role |
|----------|------|
| `HAROMA_VW_REWARD` | Enable VW learner (`1` / `true`). Requires `pip install vowpalwabbit`. |
| `HAROMA_VW_OPTS` | Extra CLI args for `pyvw.vw(...)` (default squared loss). |
| `HAROMA_RLLIB_LOG_TRANSITIONS` | Append `bandit_step` JSONL (`1` / `true`). |
| `HAROMA_RLLIB_TRANSITIONS_PATH` | Output file (default `data/rllib/transitions.jsonl`). |
| `HAROMA_VW_SCORE_WEIGHT` | Blend VW `predict` into composite score `[0,1]` (`0` = torch only). |
| `HAROMA_RLLIB_SCORE_FN` | `module:callable` → `(prompt, response) -> float` for runtime blend. |
| `HAROMA_RLLIB_SCORE_WEIGHT` | Weight for that callable (`0` = off). |
| `HAROMA_VW_ENV_CONTEXT_TTL_SEC` | Expire cached environment text for VW scoring (`0` = no TTL). |
| `HAROMA_RLLIB_LOG_FULL_AGENT_ENV` | `1` = full `agent_environment` in JSONL `info`. |
| `HAROMA_RLLIB_ENV_SUMMARY_LOG_CHARS` | Cap for `environment_summary` in JSONL. |
| `HAROMA_RLLIB_LOG_FULL_ALIGNMENT_DIAG` | `1` = full `alignment` dict in JSONL. |

### 5.2 Bandit JSONL → VW ingest ([`mind/training/vw_jsonl_ingest.py`](../mind/training/vw_jsonl_ingest.py))

| Variable | Role |
|----------|------|
| `HAROMA_VW_BANDIT_INGEST_PATH` | JSONL file of `bandit_step` lines to feed into VW before each VW `train_step`. |
| `HAROMA_VW_BANDIT_INGEST_MAX_LINES` | Max rows per `train_reward_model` call (default `64`). |
| `HAROMA_VW_BANDIT_INGEST_OFFSET_PATH` | Cursor file (default `data/rllib/vw_ingest_line.offset`). |
| `HAROMA_VW_INGEST_LOG` | `1` = print ingest counts. |

Use a **dedicated** ingest file if you also append the same stream via `RLlibTransitionLogger` to avoid double-counting.

### 5.3 Pluggable simulation ([`integrations/sim/registry.py`](../integrations/sim/registry.py))

| Variable | Role |
|----------|------|
| `HAROMA_SIM_BACKEND` | `null`, `http_json`, or `package.module:ClassName`. |
| `HAROMA_SIM_BACKEND_KWARGS` | JSON object for importable class constructor. |
| `HAROMA_SIM_HTTP_BASE_URL` | Base URL for `http_json` backend. |
| `HAROMA_SIM_HTTP_STEP_PATH` | Default `/sim/step` — POST `{"action": {...}}`. |
| `HAROMA_SIM_HTTP_RESET_PATH` | Default `/sim/reset` — POST body may include `seed`. Empty string = skip. |
| `HAROMA_SIM_HTTP_OBSERVE_PATH` | Default `/sim/observe` — GET. Empty = skip. |
| `HAROMA_SIM_HTTP_TIMEOUT_SEC` | HTTP timeout seconds (default `30`). |

### 5.4 HTTP client auth ([`bridge/haroma_client.py`](../bridge/haroma_client.py))

| Variable | Role |
|----------|------|
| `HAROMA_HTTP_BEARER_TOKEN` | Optional `Authorization: Bearer …` for `_post_json` / `_get_json` / `health_ping`. |

---

## 6. Python API index

| Area | Key symbols |
|------|-------------|
| **Sim** | `integrations.sim.load_backend_from_env`, `create_backend`, `register_backend`, `SimulationBackend`, `merge_simulation_into_extensions`, `simulation_summary_for_prompt` |
| **Gymnasium / JSONL** | `mind.training.gymnasium_offline_bridge`, `mind.training.haroma_gym_env.HaromaBanditChatEnv`, `mind.training.rllib_jsonl_offline` |
| **VW / RLlib hooks** | `mind.training.vw_rl_bridge` (`composite_trained_scores`, `RLlibTransitionLogger`, `load_rllib_score_callable`) |
| **Ingest** | `mind.training.vw_jsonl_ingest.ingest_bandit_jsonl_into_vw` |
| **Bridge HTTP** | `bridge.haroma_client.post_chat`, `get_chat_result`, `post_chat_wait_result`, `post_robot_bridge_feedback`, `health_ping` |
| **Robot merge** | `integrations.robot_http_bridge.merge_feedback_into_agent_environment`, `append_robot_bridge_to_extensions` |

---

## 7. Related focused guides

| Doc | Topic |
|-----|--------|
| [Gymnasium bridge](gymnasium-bridge.md) | Bandit JSONL schema, HTTP `Env`, limitations |
| [Simulation backends](simulation-backends.md) | Protocol, HTTP JSON contract, importable backends |
| [Robot cognitive / control split](robot-cognitive-control-split.md) | RT vs cognitive |
| [Getting started](getting-started.md) | Hardware, wizard, launch |

---

## 8. Tests (sanity)

```bash
pytest tests/test_sim_backends.py tests/test_gymnasium_offline_bridge.py tests/test_haroma_gym_env.py tests/test_vw_jsonl_ingest.py tests/test_trained_score_integration.py tests/test_background_training_surface.py -q
```

`tests/test_haroma_gym_env.py` skips if `gymnasium` is not installed.
