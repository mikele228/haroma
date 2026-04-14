"""Tests for mind.robot_body_state (layered merge + last-known fallbacks)."""

from mind.robot_body_state import (
    LAYER_KEYS,
    build_robot_body_extension,
    merge_robot_body_readings,
)


def test_no_data_body_undefined():
    eff, ev, stamps = merge_robot_body_readings({}, False, None)
    assert eff == {}
    assert ev is False
    assert stamps == {}
    ext = build_robot_body_extension(eff, ev, stamps)
    assert ext["body_defined"] is False
    assert "undefined" in ext["interpretation"].lower() or "undefined" in ext["interpretation"]


def test_first_successful_reading_defines_body():
    eff, ev, stamps = merge_robot_body_readings(
        {},
        False,
        {"readings": {"height_m": 1.7, "posture": "standing"}, "sensor_available": {"height_m": True, "posture": True}},
        now=1000.0,
    )
    assert ev is True
    assert eff["height_m"] == 1.7
    assert eff["posture"] == "standing"
    assert "height_m" in stamps
    ext = build_robot_body_extension(eff, ev, stamps, now=1000.0)
    assert ext["body_defined"] is True
    assert ext["schema_version"] == 4
    assert ext["architecture"] == "layered_v4"
    assert "cognitive_summary" in ext
    assert len(ext["cognitive_summary"]) > 5
    assert ext.get("operational_mode") == "nominal"
    assert ext.get("risk_posture") == "normal"


def test_unavailable_sensor_keeps_last():
    eff, ev, _ = merge_robot_body_readings(
        {"posture": "standing", "height_m": 1.6},
        True,
        {
            "readings": {"posture": "sitting", "height_m": 9.99},
            "sensor_available": {"posture": False, "height_m": True},
        },
        now=1.0,
    )
    assert eff["posture"] == "standing"
    assert eff["height_m"] == 9.99


def test_unavailable_skips_garbage():
    eff, ev, _ = merge_robot_body_readings(
        {"height_m": 1.5},
        True,
        {"readings": {"height_m": 999.0}, "sensor_available": {"height_m": False}},
        now=1.0,
    )
    assert eff["height_m"] == 1.5


def test_nested_readings_and_dot_path_availability():
    eff, ev, stamps = merge_robot_body_readings(
        {},
        False,
        {
            "readings": {
                "localization": {
                    "frame_id": "map",
                    "pose": {"x": 1.0, "y": 2.0, "theta": 0.5, "valid": True},
                },
            },
            "sensor_available": {"localization.pose": True, "localization.frame_id": True},
        },
        now=10.0,
    )
    assert ev is True
    assert eff["localization"]["frame_id"] == "map"
    assert eff["localization"]["pose"]["x"] == 1.0
    assert "localization.pose.x" in stamps


def test_layer_subtree_off_blocks_children():
    eff, ev, _ = merge_robot_body_readings(
        {"localization": {"pose": {"x": 0.0, "y": 0.0}}},
        True,
        {
            "readings": {"localization": {"pose": {"x": 99.0, "y": 99.0}}},
            "sensor_available": {"localization": False},
        },
        now=20.0,
    )
    assert eff["localization"]["pose"]["x"] == 0.0


def test_stale_fields_when_max_age_set(monkeypatch):
    monkeypatch.setenv("HAROMA_ROBOT_FIELD_MAX_AGE_SEC", "10")
    eff = {"hardware": {"approx_height_m": 1.7}}
    field_stamps = {"hardware.approx_height_m": 100.0}
    ext = build_robot_body_extension(eff, True, field_stamps, now=200.0)
    assert ext["field_stale_max_age_sec"] == 10.0
    assert "hardware.approx_height_m" in ext["stale_fields"]


def test_layer_keys_constant():
    assert "hardware" in LAYER_KEYS
    assert "scene" in LAYER_KEYS
    assert "control" in LAYER_KEYS
    assert "safety" in LAYER_KEYS
    assert "perception" in LAYER_KEYS


def test_coordination_hint_in_extension():
    eff = {
        "localization": {
            "frame_id": "odom",
            "pose": {"x": 0, "y": 0, "theta": 0, "valid": True},
        }
    }
    ext = build_robot_body_extension(eff, True, {}, now=1.0)
    assert ext.get("coordination_hint") == "frame=odom | pose_valid"


def test_cognitive_summary_includes_safety_flags():
    eff = {
        "hardware": {"end_effectors": ["g"]},
        "safety": {"estop": True},
    }
    ext = build_robot_body_extension(eff, True, {}, now=1.0)
    assert "ESTOP" in ext["cognitive_summary"]
    assert ext.get("operational_mode") == "fault_estop"
    assert ext.get("risk_posture") == "critical"


def test_perception_digest_and_layer():
    eff = {
        "perception": {
            "attention_target": "cup",
            "fused_objects": [{"id": "a"}],
            "semantic_scene": "lab",
        }
    }
    ext = build_robot_body_extension(eff, True, {}, now=1.0)
    assert ext["layers_present"].get("perception") is True
    assert "cup" in (ext.get("perception_digest") or "")
    assert "lab" in ext["cognitive_summary"]
