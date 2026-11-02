"""Acoustic Doppler physics for a moving source and a stationary observer in still air.

Conventions used everywhere in this package:

* ``v_r`` is the radial velocity of the source relative to the observer, positive when the
  range is increasing (receding). Observed frequency ``f = f0 * c / (c + v_r)``.
* Every frequency an observer hears at receive time ``t`` was emitted at an earlier time
  ``tau`` with ``t = tau + R(tau) / c`` (retarded / emission time). The radial velocity that
  sets the pitch is the one at ``tau``, not at ``t``. Using ``t`` produces a curve with the
  wrong shape, not just a delayed one.
* "CPA" is the closest point of approach of the flight path to the observer. The
  "delayed CPA time" is when the sound emitted at CPA reaches the observer:
  ``t_cpa_delayed = tau_cpa + d_cpa / c``.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

C_SOUND = 343.0  # m/s at ~20 C


def speed_of_sound(temp_c: float) -> float:
    """Speed of sound in dry air (m/s)."""
    return 331.3 * np.sqrt(1.0 + temp_c / 273.15)


def doppler(f0, v_r, c=C_SOUND):
    """Observed frequency of a source with radial velocity ``v_r`` (positive = receding)."""
    return f0 * c / (c + v_r)


# --------------------------------------------------------------------------- straight-line flyby

def linear_flyby_ratio(t, v, d, t_cpa, c=C_SOUND):
    """f/f0 heard at receive times ``t`` for a straight, constant-speed pass.

    v: speed (m/s); d: slant distance at CPA (m); t_cpa: *delayed* CPA time (receive time of
    the sound emitted at CPA, same clock as ``t``). Solves the emission time exactly.
    """
    t = np.asarray(t, dtype=float)
    tau0 = t_cpa - d / c
    T = t - tau0
    a = c * c - v * v
    u = (c * c * T - np.sqrt(c * c * v * v * T * T + a * d * d)) / a  # emission time - tau0
    R = np.sqrt(d * d + (v * u) ** 2)
    v_r = v * v * u / R
    return c / (c + v_r)


def linear_flyby(t, f0, v, d, t_cpa, c=C_SOUND):
    """Observed frequency (Hz) for a straight, constant-speed pass. See ``linear_flyby_ratio``."""
    return f0 * linear_flyby_ratio(t, v, d, t_cpa, c)


def linear_flyby_naive(t, f0, v, d, t_cpa, c=C_SOUND):
    """The *wrong* model (radial velocity at receive time, then shifted by d/c). For comparison."""
    t = np.asarray(t, dtype=float)
    u = t - t_cpa
    R = np.sqrt(d * d + (v * u) ** 2)
    return f0 * c / (c + v * v * u / R)


# --------------------------------------------------------------------------- track-based model

@njit(cache=True)
def _interp_track(tq, t0, dt, P, V):
    """Linear interpolation of uniformly sampled position/velocity at time ``tq``."""
    n = P.shape[0]
    x = (tq - t0) / dt
    if x <= 0.0:
        i = 0
        w = 0.0
    elif x >= n - 1:
        i = n - 2
        w = 1.0
    else:
        i = int(x)
        w = x - i
    px = P[i, 0] + w * (P[i + 1, 0] - P[i, 0])
    py = P[i, 1] + w * (P[i + 1, 1] - P[i, 1])
    pz = P[i, 2] + w * (P[i + 1, 2] - P[i, 2])
    vx = V[i, 0] + w * (V[i + 1, 0] - V[i, 0])
    vy = V[i, 1] + w * (V[i + 1, 1] - V[i, 1])
    vz = V[i, 2] + w * (V[i + 1, 2] - V[i, 2])
    inside = 0.0 <= x <= n - 1
    return px, py, pz, vx, vy, vz, inside


def still_air(c: float = C_SOUND) -> np.ndarray:
    """Atmosphere vector for ``track_doppler_ratio``: [c_dop, w_src(3), c_path, w_path(3)]."""
    return np.array([c, 0.0, 0.0, 0.0, c, 0.0, 0.0, 0.0])


@njit(cache=True)
def track_doppler_ratio(t_recv, obs, t0, dt, P, V, atm):
    """f/f0 heard by ``obs`` at receive times ``t_recv`` from a sampled track.

    P, V: (n, 3) positions / ground velocities in a Cartesian frame (ECEF metres), sampled
    every ``dt`` seconds from ``t0``. ``atm`` = [c_e, w_s(3), c_p, w_m(3)] (see atmos.py):
    with d = unit(observer -> aircraft), f/f0 = (c_e - w_s.d) / (c_e + V.d - w_s.d) and the
    travel time is R / (c_p - w_m.d). Still air: ``still_air(c)``.
    The emission time is solved by fixed-point iteration (a contraction since |v_r| < c).
    Returns (ratio, range_m, valid); valid=False where the emission time falls outside
    the track.
    """
    m = t_recv.shape[0]
    ratio = np.empty(m)
    rng = np.empty(m)
    valid = np.zeros(m, dtype=np.bool_)
    ce = atm[0]
    cp = atm[4]
    for j in range(m):
        tau = t_recv[j] - 2000.0 / cp
        for _ in range(60):
            px, py, pz, vx, vy, vz, inside = _interp_track(tau, t0, dt, P, V)
            dx = px - obs[0]
            dy = py - obs[1]
            dz = pz - obs[2]
            R = np.sqrt(dx * dx + dy * dy + dz * dz)
            wm = (atm[5] * dx + atm[6] * dy + atm[7] * dz) / R
            tau_new = t_recv[j] - R / (cp - wm)
            if abs(tau_new - tau) < 1e-5:
                tau = tau_new
                break
            tau = tau_new
        px, py, pz, vx, vy, vz, inside = _interp_track(tau, t0, dt, P, V)
        dx = px - obs[0]
        dy = py - obs[1]
        dz = pz - obs[2]
        R = np.sqrt(dx * dx + dy * dy + dz * dz)
        vr = (dx * vx + dy * vy + dz * vz) / R
        ws = (atm[1] * dx + atm[2] * dy + atm[3] * dz) / R
        ratio[j] = (ce - ws) / (ce + vr - ws)
        rng[j] = R
        valid[j] = inside
    return ratio, rng, valid


@njit(cache=True)
def track_cpa(obs, t0, dt, P, c):
    """Closest point of approach of a sampled track to ``obs``.

    Returns (tau_cpa, d_cpa, t_cpa_delayed). Sub-sample refinement by a parabola through
    the squared ranges of the three samples around the minimum (exact for straight flight).
    """
    n = P.shape[0]
    best = 1e30
    k = 0
    for i in range(n):
        dx = P[i, 0] - obs[0]
        dy = P[i, 1] - obs[1]
        dz = P[i, 2] - obs[2]
        r2 = dx * dx + dy * dy + dz * dz
        if r2 < best:
            best = r2
            k = i
    tau = t0 + k * dt
    d2 = best
    if 0 < k < n - 1:
        r = np.empty(3)
        for q in range(3):
            i = k - 1 + q
            dx = P[i, 0] - obs[0]
            dy = P[i, 1] - obs[1]
            dz = P[i, 2] - obs[2]
            r[q] = dx * dx + dy * dy + dz * dz
        den = r[0] - 2.0 * r[1] + r[2]
        if den > 0:
            off = 0.5 * (r[0] - r[2]) / den
            tau = t0 + (k + off) * dt
            d2 = r[1] - 0.25 * (r[0] - r[2]) * off
    d = np.sqrt(max(d2, 0.0))
    return tau, d, tau + d / c


@njit(parallel=True, cache=True)
def many_track_doppler_ratio(t_recv, obs, t0, dt, P, V, atm):
    """``track_doppler_ratio`` for many observers (rows of ``obs``) on one track."""
    k = obs.shape[0]
    m = t_recv.shape[0]
    out = np.empty((k, m))
    for i in prange(k):
        r, _, ok = track_doppler_ratio(t_recv, obs[i], t0, dt, P, V, atm)
        for j in range(m):
            out[i, j] = r[j] if ok[j] else np.nan
    return out


@njit(parallel=True, cache=True)
def track_emission(t_recv, obs, t0, dt, P, V, c):
    """Emission time and range for each receive time (for synthesising audio).

    The received waveform of a source emitting phase 2*pi*f0*tau is
    sin(2*pi*f0*tau(t)) / R(t); its instantaneous frequency is the Doppler-shifted one.
    """
    m = t_recv.shape[0]
    tau_out = np.empty(m)
    rng = np.empty(m)
    for j in prange(m):
        tau = t_recv[j] - 2000.0 / c
        R = 2000.0
        for _ in range(60):
            px, py, pz, vx, vy, vz, inside = _interp_track(tau, t0, dt, P, V)
            dx = px - obs[0]
            dy = py - obs[1]
            dz = pz - obs[2]
            R = np.sqrt(dx * dx + dy * dy + dz * dz)
            tau_new = t_recv[j] - R / c
            if abs(tau_new - tau) < 1e-7:
                tau = tau_new
                break
            tau = tau_new
        tau_out[j] = tau
        rng[j] = R
    return tau_out, rng
