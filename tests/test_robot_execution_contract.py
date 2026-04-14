"""mind.robot_execution_contract + integrations.robot_http_bridge."""

from mind.robot_execution_contract import (
    MAX_ROBOT_BRIDGE_RESULTS,
    build_executor_command_batch,
    merge_robot_bridge_history,
    normalize_feedback_payload,
    robot_command_from_body_action,
    summarize_robot_bridge,
    validate_executor_command_batch,
    validate_feedback_entry,
)
from integrations.robot_http_bridge import merge_feedback_into_agent_environment


def test_build_batch_maps_body_action():
    batch = build_executor_command_batch(
        [
            {
                "label": "Go",
                "command": "move_base",
                "resource": "base",
                "priority": 0.8,
                "cancel_current": True,
                "safety_class": "caution",
                "parameters": {"vx": 0.2},
                "coordinate_frame": "map",
                "supports_goal_id": "g1",
                "rationale": "advance",
                "duration_hint_sec": 5.0,
                "confidence": 0.7,
            }
        ],
        source="test",
        correlation_id="corr-fixed-uuid",
    )
    assert batch["bridge_schema_version"] >= 1
    assert batch["correlation_id"] == "corr-fixed-uuid"
    assert len(batch["commands"]) == 1
    c0 = batch["commands"][0]
    assert c0["type"] == "move_base"
    assert c0["cancel_previous_on_resource"] is True
    assert c0["resource"] == "base"
    assert c0["parameters"] == {"vx": 0.2}


def test_robot_command_alias_nav():
    out = robot_command_from_body_action(
        {"label": "x", "command": "nav", "layer": "localization"},
        0,
        correlation_id="abc",
    )
    assert out["type"] == "move_base"


def test_feedback_normalize():
    fb, err = normalize_feedback_payload(
        {
            "bridge_schema_version": 1,
            "correlation_id": "c1",
            "results": [
                {"command_id": "cmd_0_c1", "status": "completed", "detail": "ok", "t": 1.0}
            ],
        }
    )
    assert err is None
    assert fb["results"][0]["status"] == "completed"


def test_validate_feedback_entry_rejects_bad_status():
    assert validate_feedback_entry({"command_id": "x", "status": "bogus"}) is not None


def test_merge_into_agent_environment():
    env = {"domain": "lab", "extensions": {"foo": 1}}
    merged, err = merge_feedback_into_agent_environment(
        env,
        {
            "bridge_schema_version": 1,
            "correlation_id": "c2",
            "results": [{"command_id": "cmd_0", "status": "running"}],
        },
    )
    assert err is None
    assert merged["extensions"]["foo"] == 1
    assert merged["extensions"]["robot_bridge"]["correlation_id"] == "c2"


def test_merge_robot_bridge_history_dedupes_and_prefers_incoming_status():
    prev = {
        "bridge_schema_version": 1,
        "correlation_id": "old",
        "received_at_epoch": 10.0,
        "results": [
            {"command_id": "a", "status": "running", "t": 1.0},
            {"command_id": "b", "status": "pending", "t": 2.0},
        ],
    }
    inc = {
        "bridge_schema_version": 1,
        "correlation_id": "new",
        "received_at_epoch": 20.0,
        "results": [{"command_id": "a", "status": "completed", "t": 3.0}],
    }
    m = merge_robot_bridge_history(prev, inc)
    assert m["correlation_id"] == "new"
    assert m["received_at_epoch"] == 20.0
    by_id = {r["command_id"]: r["status"] for r in m["results"]}
    assert by_id["a"] == "completed"
    assert by_id["b"] == "pending"


def test_merge_robot_bridge_history_caps_length():
    rows = [
        {"command_id": f"id{i}", "status": "completed", "t": float(i)} for i in range(MAX_ROBOT_BRIDGE_RESULTS + 12)
    ]
    m = merge_robot_bridge_history(None, {"bridge_schema_version": 1, "correlation_id": "x", "results": rows})
    assert len(m["results"]) == MAX_ROBOT_BRIDGE_RESULTS


def test_validate_executor_command_batch():
    batch = build_executor_command_batch([{"label": "x", "command": "noop"}], correlation_id="c")
    assert validate_executor_command_batch(batch) is None
    assert validate_executor_command_batch([]) == "expected_object"
    assert validate_executor_command_batch({"bridge_schema_version": 99, "commands": []}) == "bridge_schema_mismatch"


def test_summarize_robot_bridge():
    s = summarize_robot_bridge(
        {
            "correlation_id": "corr",
            "results": [
                {"command_id": "c1", "status": "completed"},
                {"command_id": "c2", "status": "failed"},
            ],
        }
    )
    assert s["has_bridge"] is True
    assert s["results_count"] == 2
    assert s["status_counts"]["completed"] == 1
    assert s["last_failed_command_id"] == "c2"
