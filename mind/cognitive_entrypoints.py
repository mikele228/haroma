"""Canonical notes on **which code paths run packed-context LLM** (``generate_chat``).

Haroma has two main ways to advance cognition:

1. **Embedded / library —** :class:`~mind.control.ElarionController` ``run_cycle``

   Runs perception → workspace → recall → reasoning → (optional) **13.2b packed
   LLM** → counterfactual → … → deliberative action. By default it does **not**
   call :func:`~mind.cycle_flow.run_llm_context_reasoning_phase`. Set
   ``HAROMA_CONTROLLER_PACKED_LLM=1`` to run the same phase as Persona (after
   reasoning, before counterfactual); see :mod:`mind.packed_llm_controller_bridge`.
   Otherwise ``episode.llm_context`` is typically unused unless another layer
   sets it.

2. **Multi-agent chat —** :class:`~agents.persona_agent.PersonaAgent` (TrueSelf delegation)

   Runs the same *family* of phases via :mod:`mind.cycle_flow`, and when gates
   allow, calls ``run_llm_context_reasoning_phase`` so
   :mod:`engine.LLMContextReasoner` can fill ``episode.llm_context`` for HTTP
   ``response`` resolution (:func:`~mind.chat_visibility.resolve_chat_visible_text`).

**Import convention:** Application code that shapes chat text or packed LLM should
prefer :mod:`mind.cognitive_contracts` (re-exports visibility, merge helpers, and
packed-LLM symbols). Low-level modules under ``mind.packed_llm_*`` remain the
source of truth; tests may import them directly for identity checks.

**Configuration:** Prefer :mod:`mind.haroma_settings` for grouped ``HAROMA_*`` /
``ELARION_*`` reads; :mod:`mind.config_env` for generic ``env_truthy`` / ``env_int``.
Additional keys still exist in submodules (e.g. ``mind.packed_llm_*``).
"""
