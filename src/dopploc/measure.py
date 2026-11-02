"""Measure a Doppler curve on the spectrogram: trace the ridge, fit a straight-line flyby,
and turn the fit into the speed / slant-distance / CPA-time ranges the search uses.

A straight, constant-speed pass is fully described by (f0, v, d, t_cpa): v sets the size of
the frequency drop, d sets how quickly it happens. Real flights turn and change speed, so
the ranges should be generous; the exact geometry is checked later by ``score``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.optimize import least_squares

from . import physics
from .audio import Spectrogram


@njit(cache=True)
def _viterbi(B, max_step, penalty, start_bin, end_bin):
    nb, nt = B.shape
    score = np.full((nt, nb), -1e18)
    back = np.zeros((nt, nb), np.int32)
    for b in range(nb):
        if start_bin < 0 or abs(b - start_bin) <= max_step:
            score[0, b] = B[b, 0]
    for j in range(1, nt):
        for b in range(nb):
            best = -1e18
            arg = 0
            for d in range(-max_step, max_step + 1):
                bp = b + d
                if bp < 0 or bp >= nb:
                    continue
                v = score[j - 1, bp] - penalty * d * d
                if v > best:
                    best = v
                    arg = bp
            score[j, b] = best + B[b, j]
            back[j, b] = arg
    path = np.empty(nt, np.int64)
    if end_bin >= 0:
        lo = max(0, end_bin - max_step)
        hi = min(nb, end_bin + max_step + 1)
        path[nt - 1] = lo + np.argmax(score[nt - 1, lo:hi])
    else:
        path[nt - 1] = np.argmax(score[nt - 1])
    for j in range(nt - 1, 0, -1):
        path[j - 1] = back[j, path[j]]
    return path


def trace_ridge(spec: Spectrogram, f_start: float | None = None, f_end: float | None = None,
                max_slope_hz_s: float = 8.0, penalty: float = 0.05, min_z: float = 1.0):
    """Follow the brightest smooth ridge through a (whitened) spectrogram window.

    f_start / f_end pin the ridge near those frequencies at the window edges (useful when
    several curves cross). Returns (t, f, z) with sub-bin peak interpolation; points whose
    brightness is below ``min_z`` are dropped.
    """
    B = spec.S.astype(np.float64)
    max_step = max(1, int(np.ceil(max_slope_hz_s * spec.dt / spec.df)))
    sb = -1 if f_start is None else int(round((f_start - spec.f[0]) / spec.df))
    eb = -1 if f_end is None else int(round((f_end - spec.f[0]) / spec.df))
    path = _viterbi(B, max_step, penalty, sb, eb)
    j = np.arange(len(path))
    b = np.clip(path, 1, len(spec.f) - 2)
    y0, y1, y2 = B[b - 1, j], B[b, j], B[b + 1, j]
    den = y0 - 2 * y1 + y2
    off = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1), 0.0)
    f = spec.f[0] + (b + np.clip(off, -0.5, 0.5)) * spec.df
    z = B[path, j]
    # keep the longest run where the (smoothed) ridge is bright; ends usually wander in noise
    zs = np.convolve(z, np.ones(9) / 9, mode="same")
    good = zs >= min_z
    best, cur, start, best_rng = 0, 0, 0, (0, 0)
    for i, g in enumerate(np.append(good, False)):
        if g:
            if cur == 0:
                start = i
            cur += 1
        else:
            if cur > best:
                best, best_rng = cur, (start, i)
            cur = 0
    keep = np.zeros(len(z), bool)
    keep[best_rng[0]:best_rng[1]] = True
    keep &= z >= 0.5 * min_z
    return spec.t[keep], f[keep], z[keep]


@dataclass
class Fit:
    f0: float
    v: float
    d: float
    t_cpa: float
    rms: float
    harmonic: int

    def curve(self, t, c=physics.C_SOUND):
        return self.harmonic * physics.linear_flyby(t, self.f0, self.v, self.d, self.t_cpa, c)


def fit_flyby(t, f, harmonic: int = 1, c: float = physics.C_SOUND, w=None) -> Fit:
    """Robust least-squares straight-line flyby fit to a traced curve of harmonic ``harmonic``."""
    t = np.asarray(t, float)
    fh = np.asarray(f, float) / harmonic
    w = np.ones_like(t) if w is None else np.asarray(w, float)
    hi, lo = np.percentile(fh, 97), np.percentile(fh, 3)
    f0_init = 2 * hi * lo / (hi + lo)
    v_init = np.clip(c * (hi - lo) / (hi + lo), 15, 200)
    t_init = t[np.argmin(np.abs(fh - f0_init))]

    def resid(p):
        f0, v, d, tc = p
        return w * (physics.linear_flyby(t, f0, v, d, tc, c) - fh)

    w0 = w.copy()
    best = None
    for _ in range(4):  # fit, then drop > 4 sigma (MAD) outliers such as hops onto hum lines
        best = None
        for d0 in (200, 500, 1000, 2000, 4000, 8000):
            r = least_squares(resid, [f0_init, v_init, d0, t_init], loss="soft_l1", f_scale=0.5,
                              bounds=([1, 5, 30, t[0] - 120], [5000, 300, 50000, t[-1] + 120]))
            if best is None or r.cost < best.cost:
                best = r
        res = physics.linear_flyby(t, *best.x[:2], best.x[2], best.x[3], c) - fh
        mad = np.median(np.abs(res[w > 0])) * 1.4826 + 1e-3
        new_w = w0 * (np.abs(res) < 4 * mad)
        if np.array_equal(new_w, w):
            break
        w[:] = new_w
    f0, v, d, tc = best.x
    rms = float(np.sqrt(np.sum(resid(best.x) ** 2) / max(1, np.sum(w > 0))) * harmonic)
    return Fit(float(f0), float(v), float(d), float(tc), rms, harmonic)


def profile_ranges(t, f, fit: Fit, c: float = physics.C_SOUND, rel: float = 0.5, abs_hz: float = 0.3,
                   v_grid=None, d_grid=None):
    """Grid (v, d), re-fitting f0 and t_cpa at each node; accept nodes with
    rms <= rms_best * (1 + rel) + abs_hz. Returns dict of ranges and the rms map."""
    t = np.asarray(t, float)
    fh = np.asarray(f, float) / fit.harmonic
    v_grid = np.arange(max(10, fit.v - 40), fit.v + 40.1, 2.0) if v_grid is None else v_grid
    d_grid = np.geomspace(max(50, fit.d / 4), fit.d * 4, 41) if d_grid is None else d_grid
    rms = np.full((len(v_grid), len(d_grid)), np.inf)
    extra = {}
    for i, v in enumerate(v_grid):
        for j, d in enumerate(d_grid):
            r = least_squares(lambda p: physics.linear_flyby(t, p[0], v, d, p[1], c) - fh,
                              [fit.f0, fit.t_cpa], loss="soft_l1", f_scale=0.5)
            rms[i, j] = np.sqrt(np.mean(r.fun ** 2)) * fit.harmonic
            extra[(i, j)] = r.x
    thr = fit.rms * (1 + rel) + abs_hz
    ok = rms <= thr
    if not ok.any():
        ok = rms <= rms.min() * (1 + rel) + abs_hz
    vi, dj = np.nonzero(ok)
    f0s = np.array([extra[(a, b)][0] for a, b in zip(vi, dj)])
    tcs = np.array([extra[(a, b)][1] for a, b in zip(vi, dj)])
    return {
        "speed_ms": (float(v_grid[vi].min()), float(v_grid[vi].max())),
        "slant_m": (float(d_grid[dj].min()), float(d_grid[dj].max())),
        "f0": (float(f0s.min()), float(f0s.max())),
        "audio_cpa": (float(tcs.min()), float(tcs.max())),
        "threshold_hz": float(thr),
        "rms": rms, "v_grid": v_grid, "d_grid": d_grid,
    }


def fit_trace(t, f, harmonic: int = 1, c: float = physics.C_SOUND, rel: float = 0.5, abs_hz: float = 0.3):
    """Fit a traced curve, keep the points near the fit, profile the ranges on those.
    Returns (fit, ranges, inlier_mask)."""
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    fit = fit_flyby(t, f, harmonic, c=c)
    res = fit.curve(t, c) - f
    inl = np.abs(res) < max(4 * 1.4826 * np.median(np.abs(res)), 0.5)
    return fit, profile_ranges(t[inl], f[inl], fit, c=c, rel=rel, abs_hz=abs_hz), inl


def event_dict(name: str, fit: Fit, ranges: dict, harmonics, half_width: float = 60.0,
               speed_pad: float = 5.0, slant_pad_frac: float = 0.15) -> dict:
    """An [[event]] table: the fit's CPA and f0, padded speed and slant ranges."""
    v0, v1 = ranges["speed_ms"]
    d0, d1 = ranges["slant_m"]
    return {"name": name, "audio_cpa": round(float(fit.t_cpa), 2), "f0": round(float(fit.f0), 3),
            "harmonics": [int(h) for h in harmonics],
            "speed_ms": [round(max(0.0, v0 - speed_pad), 1), round(v1 + speed_pad, 1)],
            "slant_m": [float(round(d0 * (1 - slant_pad_frac))), float(round(d1 * (1 + slant_pad_frac)))],
            "half_width_s": float(round(half_width))}


def event_toml(name: str, fit: Fit, ranges: dict, harmonics, half_width: float = 60.0,
               speed_pad: float = 5.0, slant_pad_frac: float = 0.15) -> str:
    e = event_dict(name, fit, ranges, harmonics, half_width, speed_pad, slant_pad_frac)
    return "\n".join([
        "[[event]]",
        f'name = "{name}"',
        f"audio_cpa = {e['audio_cpa']:.2f}",
        f"f0 = {e['f0']:.3f}",
        f"harmonics = {e['harmonics']}",
        f"speed_ms = [{e['speed_ms'][0]:.1f}, {e['speed_ms'][1]:.1f}]",
        f"slant_m = [{e['slant_m'][0]:.0f}, {e['slant_m'][1]:.0f}]",
        f"half_width_s = {half_width:.0f}",
        f"# fit: v={fit.v:.1f} m/s d={fit.d:.0f} m rms={fit.rms:.2f} Hz on harmonic {fit.harmonic}",
    ])
