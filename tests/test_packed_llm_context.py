"""mind.packed_llm_context — host sections and optional LLM payload slices."""

from mind.packed_llm_context import (
    host_environment_sections_for_prompt,
    optional_llm_structured_fields,
)


def test_host_sections_includes_bridge_feedback():
    secs = host_environment_sections_for_prompt(
        {
            "domain": "lab",
            "extensions": {
                "robot_bridge": {
                    "bridge_schema_version": 1,
                    "correlation_id": "c1",
                    "results": [{"command_id": "cmd_0", "status": "completed", "detail": "ok"}],
                }
            },
        }
    )
    assert any(s.startswith("[ROBOT BRIDGE FEEDBACK]") for s in secs)
    assert any("cmd_0" in s for s in secs)


def test_host_sections_env_then_body():
    secs = host_environment_sections_for_prompt(
        {
            "domain": "x",
            "extensions": {
                "robot_body": {
                    "body_defined": True,
                    "readings": {"hardware": {"approx_height_m": 1.7}},
                }
            },
        }
    )
    assert len(secs) == 2
    assert secs[0].startswith("[ENVIRONMENT STATE]")
    assert secs[1].startswith("[ROBOT BODY STATE]")
    assert "approx_height_m" in secs[1]


def test_optional_llm_structured_fields_nonempty():
    d = optional_llm_structured_fields(
        {
            "answer": "a",
            "candidate_actions": [{"label": "x"}],
            "body_actions": [],
            "chosen_action": None,
        }
    )
    assert "candidate_actions" in d
    assert "body_actions" not in d
    assert "chosen_action" not in d


def test_optional_llm_structured_fields_empty_ctx():
    assert optional_llm_structured_fields(None) == {}
