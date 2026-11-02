"""Synthetic recordings from real ADS-B tracks, for validating the pipeline end to end.

A random observer is placed beside a real propeller flight; a second real flight that
passes the same observer 5-25 minutes later is found; every propeller flight passing within
``audible_m`` is rendered as a harmonic source (phase = 2*pi*f0*emission_time, amplitude 1/R)
over coloured noise. The ground truth is written next to the audio.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from . import geo, physics, search, tracks
from .config import Case
from .dem import DEM


def render_pass(sm: tracks.Sampled, obs, t_start, duration, sr, f0, amps, c, ref_m=300.0, rng=None):
    """Audio of one flight heard at ``obs`` (ECEF) for UTC [t_start, t_start + duration)."""
    rng = rng or np.random.default_rng()
    tq = t_start + np.arange(0.0, duration + 0.02, 0.01)
    tau, R = physics.track_emission(tq, obs, sm.t0, sm.dt, sm.P, sm.V, c)
    t_end = sm.t0 + sm.dt * (len(sm.gs) - 1)
    inside = (tau >= sm.t0) & (tau <= t_end)
    env = np.where(inside, np.minimum(1.0, ref_m / R), 0.0)
    # soften the edges where the track starts / ends
    k = int(5.0 / 0.01)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    ta = np.arange(int(duration * sr)) / sr
    tau_a = np.interp(ta, tq - t_start, tau - t_start)
    env_a = np.interp(ta, tq - t_start, env)
    # slow engine-speed wander (+-0.15 %)
    wander = np.cumsum(rng.normal(0, 1, len(tq)))
    wander = 0.0015 * wander / (np.abs(wander).max() + 1e-9)
    w_a = np.interp(ta, tq - t_start, wander)
    # source phase time: integral of (1 + wander) over emission time
    src_t = tau_a + np.concatenate([[0.0], np.cumsum(w_a[1:] * np.diff(tau_a))])
    x = np.zeros_like(ta)
    for h, a in amps.items():
        x += a * np.sin(2 * np.pi * h * f0 * src_t + rng.uniform(0, 2 * np.pi))
    return (x * env_a).astype(np.float32)


def coloured_noise(n, sr, rng, level=1.0):
    w = rng.normal(0, 1, n)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / sr)
    W /= np.maximum(f, 20.0) ** 0.5  # pink-ish
    x = np.fft.irfft(W, n)
    return (level * x / x.std()).astype(np.float32)


def make(case: Case, out_dir: Path, seed: int = 1, utc_hour: float = 18.0, sr: int = 4000,
         audible_m: float = 4000.0, snr_db: float = -6.0, log=print) -> dict:
    """Create a synthetic recording + truth. Returns the truth dict."""
    rng = np.random.default_rng(seed)
    day0 = datetime.fromisoformat(case.day).replace(tzinfo=timezone.utc).timestamp()
    t_lo = day0 + utc_hour * 3600 - 1800
    t_hi = t_lo + 3 * 3600
    trs = search.load_tracks(case, t_range=(t_lo - 600, t_hi + 600), log=log)
    sms = {}
    for tr in trs:
        sm = tracks.resample(tr, 1.0)
        if sm is not None and len(sm.gs) > 300:
            sms[tr.fid] = sm
    dem = DEM(case.cache / "dem", res=30)
    fids = list(sms)
    for attempt in range(500):
        fa = fids[rng.integers(len(fids))]
        A = sms[fa]
        k = int(rng.integers(120, len(A.gs) - 120))
        tA = A.t0 + k * A.dt
        if not (t_lo < tA < t_hi - 1800) or not (45 < A.gs[k] < 110):
            continue
        # observer perpendicular to the track, 300-1500 m to one side
        x, y = geo.to_grid([A.lon[k - 1], A.lon[k + 1]], [A.lat[k - 1], A.lat[k + 1]])
        dx, dy = x[1] - x[0], y[1] - y[0]
        nrm = np.hypot(dx, dy)
        if nrm < 1:
            continue
        side = rng.choice([-1, 1]) * rng.uniform(300, 1500)
        xo = (x[0] + x[1]) / 2 - dy / nrm * side
        yo = (y[0] + y[1]) / 2 + dx / nrm * side
        lon, lat = geo.from_grid(xo, yo)
        lon, lat = float(lon), float(lat)
        elev = float(dem.elevation(np.array([lat]), np.array([lon]))[0])
        if A.h[k] - elev > 2500 or A.h[k] - elev < 150:
            continue
        obs = geo.lla_to_ecef(lat, lon, elev + 2.0)
        # every flight's CPA to the observer in the window
        passes = []
        for fid, sm in sms.items():
            d = np.linalg.norm(sm.P - obs, axis=1)
            j = int(np.argmin(d))
            if d[j] < audible_m and 0 < j < len(d) - 1:
                tau, dd, tdel = physics.track_cpa(obs, sm.t0, sm.dt, sm.P, case.c)
                passes.append((tdel, dd, fid, float(sm.gs[j])))
        passes.sort()
        a_pass = [p for p in passes if p[2] == fa]
        if not a_pass:
            continue
        t1 = a_pass[0][0]
        later = [p for p in passes if p[2] != fa and 300 < p[0] - t1 < 1500 and p[1] < 2500 and 45 < p[3] < 110]
        if not later:
            continue
        b = later[0]
        between = [p for p in passes if t1 + 60 < p[0] < b[0] - 60]
        if between:
            continue
        break
    else:
        raise RuntimeError("no suitable synthetic geometry found")

    t_start = t1 - 240.0
    duration = (b[0] - t1) + 480.0
    x = coloured_noise(int(duration * sr), sr, rng)
    sig = np.zeros_like(x)
    heard = []
    for tdel, dd, fid, gs in passes:
        if not (t_start - 120 < tdel < t_start + duration + 120):
            continue
        f0 = float(rng.uniform(70, 140))
        amps = {1: 1.0, 2: 0.6, 3: 0.35, 4: 0.2}
        sig += render_pass(sms[fid], obs, t_start, duration, sr, f0, amps, case.c, rng=rng)
        heard.append({"fid": fid, "t_cpa_delayed": tdel, "audio_cpa": tdel - t_start, "d": dd, "gs": gs, "f0": f0})
    p_sig = np.mean(sig[np.abs(sig) > 0] ** 2) if np.any(sig) else 1.0
    x *= np.sqrt(p_sig / 10 ** (snr_db / 10))
    y = sig + x
    y /= np.abs(y).max() * 1.05
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / "synthetic.wav", y, sr, subtype="FLOAT")
    truth = {"lat": lat, "lon": lon, "elev": elev, "t_audio0_utc": t_start, "event1": fa, "event2": b[2],
             "offset_s": b[0] - t1, "passes": heard, "seed": seed}
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=1))
    log(f"synthetic observer {lat:.5f},{lon:.5f}  e1={fa}  e2={b[2]}  offset={b[0] - t1:.1f}s  "
        f"{len(heard)} audible passes")
    return truth
