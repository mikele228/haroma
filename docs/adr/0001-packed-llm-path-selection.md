# ADR 0001: Packed-context LLM path selection

**Status:** Accepted  
**Context:** Haroma can run `generate_chat` with a packed prompt (memories, KG, agent state) from two entry points: **PersonaAgent** (HTTP / multi-agent chat) and **ElarionController.run_cycle** (embedded / library).

**Decision:**

1. **Gate computation** lives in `mind/packed_llm_paths.py` (`compute_packed_llm_path_state`, `PackedLlmPathState`). Flags such as `llm_ctx_enabled`, organic vs packed-eligible paths, and chat-primary routing are derived there from cycle inputs (`build_packed_llm_cycle_inputs`).

2. **Persona and controller** both call `invoke_run_llm_context_reasoning_phase` after symbolic reasoning when their respective env gates allow (`HAROMA_CHAT_LLM_PRIMARY` for chat-primary behavior; `HAROMA_CONTROLLER_PACKED_LLM` for the embedded bridge). The controller path is implemented in `mind/packed_llm_controller_bridge.py` to mirror Persona wiring.

3. **Discourse shaping** (`discourse_context_for_packed_llm` → `enrich_discourse_for_dialogue_phases`) is orthogonal to path selection but receives **`llm_ctx_enabled`** (and other cycle metadata) so the prompt can carry a compact `[Pack] llm_ctx=…` tag when the dialogue roadmap tier is high enough—see ADR 0002.

**Consequences:** Any new condition that should affect *whether* packed LLM runs belongs in `packed_llm_paths` / cycle inputs; discourse tags must not duplicate that logic beyond passing through the resolved boolean.
