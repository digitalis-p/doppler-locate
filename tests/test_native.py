import numpy as np
import pytest

from dopploc import native, score, tracks

pytestmark = pytest.mark.skipif(native.backend() == "numba", reason="native library not built")


def _track(fid, seed):
    rng = np.random.default_rng(seed)
    dt = 0.5
    t = np.arange(0, 1200, dt)
    v = rng.uniform(50, 140)
    hdg = rng.uniform(0, 2 * np.pi)
    turn = rng.uniform(-0.002, 0.002)
    ang = hdg + turn * t
    vx, vy = v * np.cos(ang), v * np.sin(ang)
    P = np.stack([np.cumsum(vx) * dt, np.cumsum(vy) * dt, np.full_like(t, 1500.0)], 1) + [6.37e6, 0, 0]
    V = np.stack([vx, vy, np.zeros_like(t)], 1)
    return tracks.Sampled(fid, 1.7e9, dt, P, V, t, t, t, np.full_like(t, v))


@pytest.fixture(scope="module")
def setup():
    flights = {f"f{i}": _track(f"f{i}", i) for i in range(3)}
    rng = np.random.default_rng(7)
    n = 200
    fids = np.array([f"f{i % 3}" for i in range(n)])
    obs = np.empty((n, 3))
    t_exp = np.empty(n)
    for i, f in enumerate(fids):
        sm = flights[f]
        k = rng.integers(400, 2000)
        obs[i] = sm.P[k] + [rng.uniform(-3000, 3000), rng.uniform(-3000, 3000), -1500 + rng.uniform(-50, 50)]
        t_exp[i] = sm.t0 + k * sm.dt + 5
    u = np.arange(-65, 65, 0.1)
    w = rng.uniform(-12, 12, (n, 3)) * [1, 1, 0.1]
    atm = np.c_[rng.uniform(325, 350, n), w, rng.uniform(330, 350, n), 0.5 * w]
    return flights, fids, obs, t_exp, u, atm


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_model_and_sweep_match_numba(setup, device, monkeypatch):
    if device == "cuda" and native.backend() != "cuda":
        pytest.skip("no GPU")
    flights, fids, obs, t_exp, u, atm = setup
    ref = {}
    for f in np.unique(fids):
        jj = np.flatnonzero(fids == f)
        ref[f] = (jj, *score.model_ratios(obs[jj], flights[f], t_exp[jj], u, 343.0, atm[jj]))
    monkeypatch.setattr(native, "_dev", device)
    pack = native.TrackPack(flights)
    tc, dc, R = native.model(pack, fids, obs, t_exp, u, 343.0, atm=atm)
    for f, (jj, tcr, dcr, Rr) in ref.items():
        np.testing.assert_allclose(tc[jj], tcr, atol=1e-6)
        np.testing.assert_allclose(dc[jj], dcr, atol=1e-6)
        np.testing.assert_allclose(R[jj], Rr, atol=2e-6, equal_nan=True)

    rng = np.random.default_rng(3)
    tau = np.arange(1000.0, 1120.0, 0.25)
    B = rng.random((400, len(tau))).astype(np.float32)
    deltas = np.arange(-3, 3.01, 0.25)
    f0s = np.arange(98, 102, 0.2)
    harms = np.array([1.0, 2.0])
    base = np.full(len(fids), 1060.0)
    S, A = native.score_brightness(R, u[0], u[1] - u[0], base, deltas, f0s, harms, tau, B, 60.0, 0.5)
    Sr, Ar = score.score_brightness(R.astype(np.float64), u[0], u[1] - u[0], base, deltas, f0s, harms, tau,
                                    B.astype(np.float64), 60.0, 0.5)
    np.testing.assert_allclose(S, Sr, atol=2e-4)
    assert (A == Ar).mean() > 0.98
