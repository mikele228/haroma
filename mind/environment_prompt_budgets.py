"""Single place for environment-text size limits (packed LLM, VW, metadata)."""

from __future__ import annotations

# Packed-context LLM: [ENVIRONMENT STATE] block
PACKED_LLM_ENV_SUMMARY_MAX_CHARS = 3200

# Packed-context LLM: [ROBOT BODY STATE] JSON blob (extensions.robot_body)
PACKED_LLM_ROBOT_BODY_MAX_CHARS = 4000

# Packed-context LLM: [ROBOT BRIDGE FEEDBACK] (extensions.robot_bridge — executor acks)
PACKED_LLM_ROBOT_BRIDGE_MAX_CHARS = 2000

# Persona → alignment_metadata.environment_summary before record_outcome
PERSONA_ALIGNMENT_ENV_SUMMARY_MAX_CHARS = 800

# LLMBackend.record_outcome: VW cache + finetune metadata string cap
RECORD_OUTCOME_ENV_SUMMARY_MAX_CHARS = 600

# Vowpal Wabbit |e namespace (must match train vs predict)
VW_ENV_NAMESPACE_MAX_CHARS = 400
