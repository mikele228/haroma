# ADR 0002: Dialogue discourse phase ordering

**Status:** Accepted  
**Context:** `HAROMA_DIALOGUE_PHASE` (clamped by `HAROMA_DIALOGUE_PHASE_MAX` in `mind/haroma_settings.py`) enables incremental **suffix segments** on the packed discourse string for session discipline, eval tokens, persona hints, and telemetry-style tags.

**Decision:**

1. **Single enrichment function:** `enrich_discourse_for_dialogue_phases` in `mind/dialogue_phases.py` appends segments in a **fixed order** (after optional `base` conversation summary):

   **Session → cycle role → rel → turn → (correction hint if matched) → eval → cog → pack → voice**

   Joining uses `_join_discourse` (`" | "` between non-empty parts).

2. **Tiers are inclusive:** higher phase numbers imply lower-tier segments when the corresponding env tier is enabled (e.g. phase ≥ 9 still includes session when phase ≥ 1).

3. **Correction** is special: it is **not** a separate tier line; it inserts the static correction *discourse note* only when phase ≥ 2 **and** user text matches heuristics (`_correction_markers_in_text`). It appears **after** `[Turn]` and **before** `[Eval]`.

4. **Changing order or semantics** requires updating **golden / contract tests** (`tests/test_dialogue_discourse_contract.py`) and any consumers that parse bracket tokens.

**Consequences:** New roadmap tiers should extend `PHASE_*` constants, bump `HAROMA_DIALOGUE_PHASE_MAX`, document the tag in `dialogue_phases` module docstring, wire parameters through `packed_llm_discourse` and callers (Persona + controller bridge), and add barrel exports in `mind/cognitive_contracts`.
