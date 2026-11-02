"""Detector: finds a synthetic flyover (two harmonics) and ignores a steady hum in noise."""

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from dopploc import detect, physics  # noqa: E402

SR = 4000


def _write(path, pass_=True, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 300.0, 1 / SR)
    x = rng.normal(0, 1.0, len(t))
    x += 0.08 * np.sin(2 * np.pi * 200.0 * t)            # steady hum
    if pass_:
        f = physics.linear_flyby(t, 110.0, 60.0, 800.0, 150.0)
        ph = 2 * np.pi * np.cumsum(f) / SR
        x += 0.15 * np.sin(ph) + 0.09 * np.sin(2 * ph)
    sf.write(path, (x / np.abs(x).max() * 0.9).astype(np.float32), SR)
    return path


def test_finds_flyover(tmp_path):
    res = detect.detect(_write(tmp_path / "pass.wav"), log=lambda *a: None)
    good = [f for f in res.flyovers if f.confidence != "low"]
    assert res.found and len(good) == 1, detect.report(res)
    f = good[0]
    assert abs(f.t_cpa - 150.0) < 3.0
    assert abs(f.f0 - 110.0) < 2.0
    assert set(f.harmonics) >= {1, 2}
    assert 45 < f.v < 75 and 500 < f.d < 1200


def test_steady_hum_is_not_a_flyover(tmp_path):
    res = detect.detect(_write(tmp_path / "hum.wav", pass_=False, seed=1), log=lambda *a: None)
    assert not res.found, detect.report(res)
