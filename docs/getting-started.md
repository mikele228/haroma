# Getting Started

[<- Back to Index](index.md)

This guide walks you through installing HaromaX6, downloading training data, and launching Elarion for the first time.

**Conceptual frame:** Elarion is documented as a **minded agent** — **LLM** as **Brain CPU**, **Memory Forest** as **memory**, the **cognitive cycle** as **law**, **goals/drives** as **fuel**, one cycle as one **Atomos**. Read [Minded architecture](minded-architecture-metaphor.md) before deep dives.

---

## Prerequisites

### Hardware minimum requirements

HaromaX6 runs a **multi-threaded Flask server**, **several agent loops**, **PyTorch-backed modules**, and optionally a **local GGUF LLM** via `llama-cpp-python`. Requirements depend on whether you use a **local** large language model or an external API-only path.

| Tier | RAM | CPU | Storage | GPU | Typical use |
|------|-----|-----|---------|-----|-------------|
| **Minimum** | **8 GB** system RAM | 64-bit **x86_64** or **ARM64**, 2+ cores | **15 GB** free disk (deps + small models + `data/` growth) | None (CPU inference) | Light dev, short sessions, smaller / quantized GGUF or heavy swapping |
| **Recommended** | **16 GB+** | 4+ cores | **25 GB+** SSD | **NVIDIA GPU** with CUDA for `llama-cpp-python` GPU layers | Comfortable local chat, vision/audio stacks, training data on disk |
| **Tight / edge** | **4 GB** | 4+ cores (ARM OK) | **10 GB+** | None | **Not** recommended for full local LLM; use a **remote** LLM endpoint or a very small quantized model and expect swap/latency |

**Notes:**

- **Disk:** First-time setup pulls Python packages, spaCy, sentence-transformers cache, optional **ConceptNet / corpora** via `scripts/download_training_data.py`, optional **BERT-family** checkpoints via `scripts/download_bert_models.py` (DistilBERT, MobileBERT, BERT-tiny into the Hugging Face cache), and any **GGUF** you configure. Plan extra space for **persisted memory** under `data/cognitive/`.
- **GPU:** Optional but strongly improves local LLM throughput. CPU-only PyTorch installs are supported; reinstall PyTorch with CUDA if you use GPU offload for the local backend.
- **Raspberry Pi / SBC:** Suitable as a **sensor bridge** or thin client; running the **full** stack + large local LLM on-device is usually impractical—prefer Haroma on a **desktop/laptop/NUC** or **remote** inference.
- **Network:** First install needs internet; ongoing chat may use **local** models without cloud access.

### Software

- **Python 3.10+** (3.12 recommended)
- **Internet connection** for first-time model and data downloads

---

## 1. Automated Setup

### Windows

Open PowerShell as Administrator and run:

```powershell
cd c:\Project\HaromaX6
.\setup_windows.ps1
```

This script will:
- Download and install Python 3.12 if not found
- Upgrade pip and install all dependencies (CPU-only PyTorch, Flask, NumPy, spaCy, transformers, etc.)
- Download the spaCy `en_core_web_sm` language model
- Download the `sentence-transformers/all-MiniLM-L6-v2` embedding model
- Create required data directories
- Run `scripts/download_training_data.py` for public knowledge data

### Linux

```bash
cd /path/to/HaromaX6
chmod +x setup_linux.sh
./setup_linux.sh
```

Supports apt (Debian/Ubuntu), dnf (Fedora), pacman (Arch), and apk (Alpine). On Raspberry Pi, also installs I2C/GPIO sensor libraries.

---

## 2. Manual Setup

If you prefer manual installation:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/download_training_data.py
```

Optional — **maximum training surfaces** (Gymnasium/sklearn bridge, Vowpal Wabbit online reward):

```bash
pip install -r requirements-training-extras.txt
```

Windows: `.\scripts\install_max_training.ps1`. Env tables and API index: [Training & integrations reference](reference-training-integrations.md).

Optional — prefetch **DistilBERT**, **MobileBERT**, and a **BERT-tiny** checkpoint (Hugging Face `transformers` cache; used by parts of the stack such as [`engine/EmotionEngine.py`](../engine/EmotionEngine.py) for DistilBERT):

```bash
python scripts/download_bert_models.py
python scripts/download_bert_models.py --list   # show IDs
```

The emotion encoder defaults to DistilBERT; set `HAROMA_EMOTION_ENCODER=google/mobilebert-uncased` (or another Hugging Face encoder id) to switch the emotion stack to MobileBERT. Chat generation still uses your configured GGUF/API LLM.

The training data script downloads:
- **ConceptNet** — common-sense knowledge graph
- **NRC EmoLex** — emotion lexicon (10,000+ words with valence scores)
- **English word frequencies** — vocabulary with frequency rankings
- **WordNet** — lexical database via NLTK
- **DailyDialog** — conversational training corpus

---

## 2.5 Deployment wizard (optional)

After dependencies are installed, generate a **`.env`** file with interactive prompts (bind host, port, bearer token, rate limits, structured logs, optional robot bridge URL):

```bash
python scripts/setup_wizard.py
```

Profiles:

| # | Mode | Typical use |
|---|------|-------------|
| 1 | Local development | `127.0.0.1`, no bearer |
| 2 | LAN server | `0.0.0.0`, bearer + rate limit + access logs |
| 3 | Robot / edge | Like LAN, higher rate limit + bridge URL hint |
| 4 | Custom | You choose each value |

`python main.py` loads `.env` from the project root via [`mind/deploy_config.py`](../mind/deploy_config.py). You can also start from [`.env.example`](../.env.example) and edit by hand.

---

## 3. Launch

```bash
python main.py
```

You should see:

```
[HaromaX6] Elarion system ignition...
[HaromaX6] Gradient-Driven Wire Loop: ONLINE
[HaromaX6] MemoryForest: ONLINE
[HaromaX6] Smart Autonomy: ARMED
[HaromaX6] Soul Identity: LOADED
[server] Elarion listening on http://0.0.0.0:8193
```

---

## 4. Chat with Elarion

Open your browser to **http://localhost:8193**. The web UI features:

- Dark theme chat interface
- Real-time status indicators (cycle count, emotion, memory nodes)
- Automatic reconnection on server restart

Or use the API directly:

```bash
curl -X POST http://localhost:8193/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello Elarion, who are you?"}'
```

---

## 5. Check System Status

```bash
curl http://localhost:8193/status
```

The JSON is built by `mind.system_snapshot.build_http_status_payload` (multi-agent v2). Useful top-level keys include **`cycle_count`**, **`memory_nodes`**, **`llm`**, **`health`** (e.g. `llm_ready`, last agent-environment receive time), **`embodiment_readiness`** (heuristic scores), **`organs`**, **`symbolic_queue`**, **`reconciliation`**, **`agent_environment`**, and optional **`status_build_notes`** when a subsection could not be read.

For REST details, optional **bearer auth**, **rate limiting**, and **structured logging** env vars, see [API Reference](api-reference.md).

---

## 6. Push Sensor Data

External sensors can push data to Elarion:

```bash
curl -X POST http://localhost:8193/sensor \
  -H "Content-Type: application/json" \
  -d '{"channel": "temperature", "data": {"celsius": 22.5, "location": "room"}}'
```

See [Sensor Integration](sensors.md) for hardware adapter configuration.

---

## 7. Save State

Elarion auto-saves periodically, but you can trigger a manual save:

```bash
curl -X POST http://localhost:8193/save
```

State is saved as sharded JSON files in `data/cognitive/memory_trees/` (one file per memory tree) plus module-specific state files.

---

## Project Structure Quick Reference

| Directory | Purpose |
|-----------|---------|
| `mind/` | Controller, server, managers |
| `core/` | Stateful cognitive modules (memory, knowledge, etc.) |
| `engine/` | Processing engines (neural, reasoning, emotion) |
| `soul/` | Immutable identity files |
| `sensors/` | Hardware adapters |
| `boot/` | Sensory intake clients |
| `web/` | Chat UI |
| `data/` | Runtime persisted state |
| `scripts/` | Setup and data download scripts |
| `docs/` | This documentation |

---

## Next Steps

- [API Reference](api-reference.md) — all HTTP routes, auth, rate limits, lab hooks
- [Robot integration (step-by-step)](robot-integration.md) — connect Haroma to hardware via HTTP bridge / ROS 2
- [Minded architecture](minded-architecture-metaphor.md) — Brain CPU, Memory, Law, Fuel, Atomos
- [Architecture Overview](architecture.md) — understand the full system topology
- [The Cognitive Cycle](cognitive-cycle.md) — see what happens each tick
- [Soul System](soul-system.md) — learn about Elarion's identity
