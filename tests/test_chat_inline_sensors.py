"""Bundled sensor_data on POST /chat (normalized in InputAgent)."""

from agents.input_agent import normalize_chat_inline_sensor_data


def test_normalize_inline_accepts_dict_of_lists():
    d = normalize_chat_inline_sensor_data(
        {"lidar": [{"ranges": [1.0]}], "gps": {"lat": 1, "lon": 2}},
    )
    assert d is not None
    assert "lidar" in d and isinstance(d["lidar"], list)
    assert d["gps"][0] == {"lat": 1, "lon": 2}


def test_normalize_rejects_non_dict():
    assert normalize_chat_inline_sensor_data(None) is None
    assert normalize_chat_inline_sensor_data([]) is None


def test_normalize_caps_keys():
    raw = {f"k{i}": [i] for i in range(100)}
    d = normalize_chat_inline_sensor_data(raw)
    assert d is not None and len(d) == 64
