import numpy as np

from dopploc import measure, physics
from dopploc.audio import spectrogram


def straight_track(v=90.0, d=800.0, h=600.0, dt=0.5, T=400.0):
    """Straight level track along x, passing the origin observer at horizontal offset."""
    t = np.arange(-T, T + dt / 2, dt)
    y = np.sqrt(d * d - h * h)
    P = np.stack([v * t, np.full_like(t, y), np.full_like(t, h)], 1)
    V = np.tile([v, 0.0, 0.0], (len(t), 1))
    return t, P, V


def test_linear_flyby_matches_track_model():
    c = 343.0
    t, P, V = straight_track()
    obs = np.zeros(3)
    tau, d, tdel = physics.track_cpa(obs, t[0], 0.5, P, c)
    assert abs(tau) < 1e-6 and abs(d - 800.0) < 1e-6 and abs(tdel - 800.0 / c) < 1e-6
    tr = np.linspace(-60, 60, 241)
    r, _, ok = physics.track_doppler_ratio(tr, obs, t[0], 0.5, P, V, physics.still_air(c))
    ra = physics.linear_flyby_ratio(tr, 90.0, 800.0, tdel, c)
    assert ok.all()
    assert np.max(np.abs(r - ra)) < 1e-6


def test_retarded_time_is_not_a_shift():
    """The naive model (velocity at receive time) is a differently shaped curve."""
    c = 343.0
    tt = np.linspace(-40, 40, 801)
    good = physics.linear_flyby(tt, 100.0, 90.0, 800.0, 0.0, c)
    naive = physics.linear_flyby_naive(tt, 100.0, 90.0, 800.0, 0.0, c)
    best = min(np.max(np.abs(np.interp(tt, tt + s, naive) - good)[100:-100]) for s in np.linspace(-5, 5, 201))
    assert best > 0.5  # Hz: no time shift reconciles them


def test_fit_recovers_parameters():
    c = 343.0
    t = np.arange(1000.0, 1120.0, 0.25)
    rng = np.random.default_rng(0)
    f = 2 * physics.linear_flyby(t, 120.0, 85.0, 1200.0, 1060.0, c) + rng.normal(0, 0.3, len(t))
    fit = measure.fit_flyby(t, f, harmonic=2, c=c)
    assert abs(fit.f0 - 120.0) < 0.3
    assert abs(fit.v - 85.0) < 3
    assert abs(fit.d - 1200.0) < 120
    assert abs(fit.t_cpa - 1060.0) < 0.5


def test_synthetic_audio_ridge_and_fit():
    """Render a pass as audio (phase = f0 * emission time), trace it, fit it."""
    c = 343.0
    sr = 2000
    t, P, V = straight_track(v=80.0, d=900.0, h=700.0, dt=0.1, T=200.0)
    obs = np.zeros(3)
    ta = np.arange(-90.0, 90.0, 1 / sr)
    tau, R = physics.track_emission(ta, obs, t[0], 0.1, P, V, c)
    x = np.sin(2 * np.pi * 110.0 * tau) * 300 / R + np.random.default_rng(1).normal(0, 0.3, len(ta))
    sp = spectrogram(x, sr, win_s=1.0, hop_s=0.25, fmin=60, fmax=180, t_offset=-90.0).whiten()
    tt, ff, _ = measure.trace_ridge(sp, min_z=2.0)
    fit = measure.fit_flyby(tt, ff, 1, c=c)
    assert abs(fit.f0 - 110.0) < 0.5
    assert abs(fit.v - 80.0) < 4
    assert abs(fit.t_cpa - 900.0 / c) < 0.7
