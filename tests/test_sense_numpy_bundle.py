"""Canonical NumPy sense bundles (:mod:`mind.sense_numpy_bundle`)."""

import numpy as np

from mind.sense_numpy_bundle import SENSE_NUMPY_KEYS, build_senses_numpy_bundle


def test_all_keys_present_empty_without_input():
    b = build_senses_numpy_bundle(None, text_embedding=None)
    assert set(b.keys()) == set(SENSE_NUMPY_KEYS)
    for k, arr in b.items():
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.ndim == 1
        assert arr.size == 0


def test_text_embedding_passed_through():
    emb = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    b = build_senses_numpy_bundle({}, text_embedding=emb)
    assert np.allclose(b["text_embedding"], emb.astype(np.float32))


def test_lidar_gps_mapped():
    sd = {
        "lidar": [{"ranges": [1.0, 2.0, 3.0]}],
        "gps": {"lat": 10.0, "lon": -20.0},
    }
    b = build_senses_numpy_bundle(sd, text_embedding=None)
    assert b["lidar"].size == 3
    assert np.allclose(b["lidar"], [1.0, 2.0, 3.0])
    assert b["gps"].size >= 2
    assert b["vision"].size == 0


def test_unknown_channel_goes_to_other():
    b = build_senses_numpy_bundle({"weird_xyz": [1.0, 2.0]}, text_embedding=None)
    assert b["other"].size >= 2
