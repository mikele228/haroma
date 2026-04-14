# Gymnasium and offline RL bridge

[<- Back to Index](index.md) | [Full training & integration reference](reference-training-integrations.md)

Haroma does not embed OpenAI Gym or Farama Gymnasium in the core cognitive loop. This document describes how to connect **external** Gymnasium / Stable-Baselines3 (or other) trainers with Haroma using the existing **bandit JSONL** format and **`HAROMA_RLLIB_SCORE_FN`** hook.

## When to use which path

| Path | Use case |
|------|----------|
| **Offline JSONL** | Train or analyze on `data/rllib/transitions.jsonl` (or any path) without running Flask. Export a Python scorer for `HAROMA_RLLIB_SCORE_FN`. |
| **`HAROMA_RLLIB_SCORE_FN`** | At runtime, blend an external score into `composite_trained_scores` (see `mind/training/vw_rl_bridge.py`). |
| **HTTP Gymnasium env** | Low-step experiments: one `step` = one `POST /chat` (see `mind/training/haroma_gym_env.py`). The env maps `depth=fast` to `normal` (server chat semantics). Use a single trainer thread. |

Haroma’s natural RL shape is **contextual bandit / text bandit**: observation and action are **text** (mapped to vectors outside Haroma if you use SB3).

---

## Bandit JSONL record (`type: bandit_step`)

Each line is one JSON object. [`RLlibTransitionLogger`](../mind/training/vw_rl_bridge.py) appends these when `HAROMA_RLLIB_LOG_TRANSITIONS=1`.

| Field | Meaning |
|-------|---------|
| `type` | Must be `"bandit_step"`. |
| `obs` | Prompt / context string (sanitized, max 4000 chars in the logger). |
| `action` | Model / assistant response text (sanitized, max 4000 chars). |
| `reward` | Float in **[0, 1]** (clamped by the logger). |
| `done` | Boolean; logger sets `true` (one step per line). |
| `info` | Optional metadata dict; may include flattened `agent_environment_*` keys (see `_transition_info_payload` in `vw_rl_bridge.py`). |

Load lines without Ray: [`iter_bandit_steps` / `load_bandit_steps` / `summarize_file`](../mind/training/rllib_jsonl_offline.py).

You may **append compatible lines** from Gymnasium rollouts using [`write_bandit_steps_to_jsonl`](../mind/training/gymnasium_offline_bridge.py) so offline tools and Haroma agree on one format.

---

## `HAROMA_RLLIB_SCORE_FN` contract

Environment variable: `HAROMA_RLLIB_SCORE_FN=importable.module:callable`

The callable must be importable from `PYTHONPATH` (repo root):

```text
(prompt: str, response: str) -> float
```

Return value is **clamped to [0, 1]** inside `composite_trained_scores`. Typical use: load a model trained offline on `bandit_step` rows and map `(prompt, response)` to a predicted reward.

Blend weight: `HAROMA_RLLIB_SCORE_WEIGHT` in **[0, 1]** (0 = disable external head).

---

## Online training via BackgroundAgent

Reward-related **online** learning already runs in the background tick, not on the HTTP hot path:

1. **Foreground** — Cognitive cycles call `LLMBackend.record_outcome(prompt, response, outcome_score, …)` when an outcome is available. That enqueues PyTorch reward-model samples, optional **Vowpal Wabbit** lines (`HAROMA_VW_REWARD=1`, `pip install vowpalwabbit`), and optional **RLlib JSONL** (`HAROMA_RLLIB_LOG_TRANSITIONS=1`) for export only.
2. **Background** — `BackgroundAgent._run_training()` walks `core.training_surface.build_background_train_map`, including the **`llm_reward`** step: `LLMBackend.train_reward_model()` runs `reward_model.train_step()` and, if enabled, `VowpalWabbitRewardTrainer.train_step()`.

So **VW + PyTorch reward heads train online** whenever the scheduler runs `llm_reward` (default base interval registered in `mind/control.py` / `SharedResources`). **`HAROMA_RLLIB_LOG_TRANSITIONS`** does **not** train Ray/RLlib inside the process; it only appends JSONL for offline tools.

Disable background neural training for benchmarks: `HAROMA_BENCH_DISABLE_BG_TRAINING=1`. Tune deferral when HTTP chat is in flight: see `HAROMA_BG_DEFER_*` in `mind/elarion_server_v2.py` / `agents/background_agent.py`.

### Optional: feed a bandit JSONL file into VW on each `llm_reward` step

For an **external** writer (e.g. another process appending `bandit_step` lines), set:

| Env | Meaning |
|-----|---------|
| `HAROMA_VW_BANDIT_INGEST_PATH` | JSONL file to read (dedicated file recommended; avoid duplicating lines you already log via `RLlibTransitionLogger` on the same path). |
| `HAROMA_VW_BANDIT_INGEST_MAX_LINES` | Max bandit rows to enqueue per `train_reward_model` call (default `64`). |
| `HAROMA_VW_BANDIT_INGEST_OFFSET_PATH` | Persisted line cursor (default `data/rllib/vw_ingest_line.offset`). |
| `HAROMA_VW_INGEST_LOG` | Set to `1` to print how many rows were ingested. |

Implementation: [`mind/training/vw_jsonl_ingest.py`](../mind/training/vw_jsonl_ingest.py), invoked from `LLMBackend.train_reward_model()` before `VowpalWabbitRewardTrainer.train_step()`.

---

## Optional dependencies

Install RL extras (Gymnasium, sklearn for the linear scorer helper; SB3 is commented in the file):

```bash
pip install -r requirements-rl.txt
```

Without `gymnasium`, `tests/test_haroma_gym_env.py` skips the env tests (`pytest.importorskip`).

---

## Python modules

| Module | Role |
|--------|------|
| [`mind/training/gymnasium_offline_bridge.py`](../mind/training/gymnasium_offline_bridge.py) | Write/read bandit JSONL; optional sklearn linear scorer export. |
| [`mind/training/haroma_gym_env.py`](../mind/training/haroma_gym_env.py) | `gymnasium.Env` over sync `POST /chat`. |
| [`bridge/haroma_client.py`](../bridge/haroma_client.py) | `post_chat`, `get_chat_result` (stdlib HTTP). |
| [`mind/training/vw_jsonl_ingest.py`](../mind/training/vw_jsonl_ingest.py) | Optional bandit JSONL → VW queue (background `llm_reward`). |

---

## Limitations

- **Latency**: Each env step waits for a full chat cycle; do not expect Atari-scale sample counts.
- **SB3**: Standard policies expect vector observations; wrap text with your own encoder or use discrete task indices as in `HaromaBanditChatEnv`.
- **Concurrency**: Prefer a single training thread against one Haroma process unless you understand HTTP and cognitive-slot behavior.
