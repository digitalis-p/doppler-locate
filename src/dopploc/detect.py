"""Find aircraft Doppler curves in a recording, with no flight data: ``dopploc detect``.

Why it works: a tone from a source passing in a straight line at constant speed is heard as
f(t) = f0 * g((t - t_cpa) / tau, v / c) with tau = d / v. On a log-frequency axis the
fundamental f0 is only a vertical shift, so every (v, tau) pair is one fixed curve template.
Sliding each template over a whitened log-frequency spectrogram (a 2D cross-correlation, done
with FFTs) scores every (t_cpa, f0) at once. Steady hums are removed first (per-bin running
median), so only tones that sweep the way a flyby does score high.

Peaks are then
  1. checked: the curve's steep middle part must be brighter than a flat line at the same
     place (rejects partly suppressed hums and broadband bursts),
  2. grouped into harmonic sets (same CPA time, same shape, integer frequency ratios),
  3. refined with the ridge tracer and straight-line flyby fit used to measure events, which
     also yields ready-to-use ``[[event]]`` blocks for a case file.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import scipy.fft as sfft
from numba import njit, prange
from scipy.ndimage import maximum_filter, maximum_filter1d

from . import audio, measure, physics


@dataclass
class Params:
    sr: int = 4000
    win_s: float = 1.0
    hop_s: float = 0.25
    fmin: float = 30.0
    fmax: float = 1000.0
    line_s: float = 20.0            # steady-tone suppression (see Spectrogram.whiten)
    log_step: float = 0.002         # log-frequency bin (0.2 %)
    speeds: tuple = tuple(np.round(np.geomspace(20.0, 160.0, 9), 1))   # m/s
    taus: tuple = tuple(np.round(np.geomspace(2.5, 120.0, 12), 2))      # d / v (s)
    span_tau: float = 1.5           # template half-width in units of tau (the flat tails carry no
                                    # signal after steady-tone suppression)
    threshold: float = 8.0          # matched-filter score (robust z per frequency column)
    min_contrast: float = 0.5       # curve middle minus flat line (column-standardised units)
    chunk_s: float = 1800.0
    margin_s: float = 150.0
    c: float = physics.C_SOUND
    refine: bool = True
    max_candidates: int = 400


@dataclass
class Flyover:
    t_cpa: float                    # audio time of the delayed CPA (s)
    f0: float                       # fundamental (Hz)
    v: float                        # m/s
    d: float                        # CPA slant distance (m)
    z: float                        # matched-filter score of the strongest component
    contrast: float                 # curve middle minus flat line (whitened units)
    harmonics: list[int] = field(default_factory=list)
    freqs: list[float] = field(default_factory=list)   # component frequencies at CPA (Hz)
    refined: bool = False
    rms: float = float("nan")       # flyby-fit residual (Hz) when refined
    points: int = 0                 # traced points used by the fit
    ranges: dict | None = None      # profile ranges (speed, slant, f0, audio_cpa) when refined
    confidence: str = "low"
    f_main: float = 0.0             # the strongest component's frequency at CPA (Hz)
    kind: str = "aircraft"          # "aircraft" or "close" (road vehicle or other source < ~150 m)

    def event(self, name: str, half_width: float = 60.0) -> dict | None:
        if not self.refined or self.ranges is None:
            return None
        fit = measure.Fit(self.f0, self.v, self.d, self.t_cpa, self.rms, 1)
        return measure.event_dict(name, fit, self.ranges, self.harmonics or [1], half_width=half_width)


# ------------------------------------------------------------------------------------- spectra

def to_log(sp: audio.Spectrogram, fmin: float, fmax: float, step: float):
    """(T, F) image on a log-frequency axis: max over linear bins inside a log bin, linear
    interpolation where a log bin is narrower than a linear one."""
    lf = np.arange(np.log(fmin), np.log(fmax), step)
    S = np.ascontiguousarray(sp.S.T)  # (T, Flin)
    centres = np.exp(lf + step / 2)
    out = np.empty((S.shape[0], len(lf)), np.float32)
    x = (centres - sp.f[0]) / sp.df
    i = np.clip(np.floor(x).astype(int), 0, len(sp.f) - 2)
    w = np.clip(x - i, 0, 1).astype(np.float32)
    out[:] = (1 - w) * S[:, i] + w * S[:, i + 1]
    idx = np.floor((np.log(np.maximum(sp.f, 1e-6)) - lf[0]) / step).astype(int)
    ok = (idx >= 0) & (idx < len(lf))
    if ok.any():
        cols = np.flatnonzero(ok)
        bins = idx[ok]
        starts = np.flatnonzero(np.r_[True, np.diff(bins) > 0])
        mx = np.maximum.reduceat(S[:, cols], starts, axis=1)
        tgt = bins[starts]
        out[:, tgt] = np.maximum(out[:, tgt], mx)
    return out, lf


def prepare(img):
    """+-2 log-bin (0.4 %) tolerance for the curve, then per-column robust standardisation: pooling
    several linear bins into one log bin (high frequencies) raises the noise floor there."""
    img = maximum_filter1d(img, 5, axis=1)
    med = np.median(img, axis=0, keepdims=True)
    mad = np.median(np.abs(img - med), axis=0, keepdims=True) * 1.4826 + 0.05
    return ((img - med) / mad).astype(np.float32)


@dataclass
class Shape:
    v: float
    tau: float
    k: np.ndarray        # frame offsets
    dl: np.ndarray       # log-bin offsets of the curve relative to f0
    mid: np.ndarray      # mask of the steep middle part

    @property
    def d(self):
        return self.v * self.tau


def shapes(p: Params) -> list[Shape]:
    out = []
    for v in p.speeds:
        for tau in p.taus:
            half = float(np.clip(p.span_tau * tau, 8.0, 90.0))
            K = int(round(half / p.hop_s))
            k = np.arange(-K, K + 1)
            r = physics.linear_flyby_ratio(k * p.hop_s, v, v * tau, 0.0, p.c)
            dl = np.round(np.log(r) / p.log_step).astype(int)
            mid = np.abs(k * p.hop_s) <= max(tau, 4.0)
            out.append(Shape(float(v), float(tau), k, dl, mid))
    return out


# ------------------------------------------------------------------------------------- matched filter

def _scan(img: np.ndarray, bank: list[Shape]):
    """Best robust z over all shapes for every (frame, log bin), and which shape gave it."""
    T, F = img.shape
    kmax = max(int(s.k.max()) for s in bank)
    lo = min(int(s.dl.min()) for s in bank)
    hi = max(int(s.dl.max()) for s in bank)
    P = sfft.next_fast_len(T + 2 * kmax + 1, real=True)
    Q = sfft.next_fast_len(F + (hi - lo) + 1, real=True)
    A = np.zeros((P, Q), np.float32)
    A[:T, :F] = img
    FA = sfft.rfft2(A, workers=-1)
    del A
    best = np.full((T, F), -np.inf, np.float32)
    which = np.zeros((T, F), np.int16)
    rows = np.arange(0, T, max(1, T // 1500))  # frames used for the per-column statistics
    for n, s in enumerate(bank):
        K = int(s.k.max())
        dmin = int(s.dl.min())
        tmpl = np.zeros((P, Q), np.float32)
        np.add.at(tmpl, (s.k + K, s.dl - dmin), 1.0 / len(s.k))
        C = sfft.irfft2(FA * np.conj(sfft.rfft2(tmpl, workers=-1)), s=(P, Q), workers=-1)
        # C[a, b] = sum img[a + i, b + j] tmpl[i, j]  ->  response at (t, l) = C[t - K, l + dmin]
        _update(C, K, dmin, rows, best, which, n)
    return best, which


@njit(parallel=True, cache=True)
def _update(C, K, dmin, rows, best, which, n):
    """Constant false-alarm rate: the response's spread differs with frequency (log bins are
    interpolated below ~250 Hz and pooled above), so each column is normalised by its own
    median / MAD; then the running best score and its shape are updated."""
    P, Q = C.shape
    T, F = best.shape
    m = len(rows)
    for j in prange(F):
        cj = (j + dmin) % Q
        smp = np.empty(m, np.float32)
        for a in range(m):
            smp[a] = C[(rows[a] - K) % P, cj]
        med = np.median(smp)
        mad = np.median(np.abs(smp - med)) * 1.4826 + 1e-6
        for t in range(T):
            z = (C[(t - K) % P, cj] - med) / mad
            if z > best[t, j]:
                best[t, j] = z
                which[t, j] = n


def _along(img, t_idx, l_idx, s: Shape, mask=None, flat=False):
    """Mean image value along a shape's curve (or a flat line) anchored at (t_idx, l_idx)."""
    k, dl = (s.k, s.dl) if mask is None else (s.k[mask], s.dl[mask])
    tt = t_idx + k
    ll = l_idx + (0 if flat else dl)
    ok = (tt >= 0) & (tt < img.shape[0]) & (ll >= 0) & (ll < img.shape[1])
    if not ok.any():
        return 0.0
    ll = np.broadcast_to(ll, tt.shape)
    return float(img[tt[ok], ll[ok]].mean())


def _curve_cells(s: Shape, t_idx, l_idx, shape, mask=None):
    k, dl = (s.k, s.dl) if mask is None else (s.k[mask], s.dl[mask])
    tt, ll = t_idx + k, l_idx + dl
    ok = (tt >= 0) & (tt < shape[0]) & (ll >= 0) & (ll < shape[1])
    return tt[ok], ll[ok]


def _candidates(img, lf, t0, best, which, bank, p: Params, keep_range):
    """Peaks of the best-shape score, strongest first. A peak whose curve mostly runs along an
    already accepted curve is the same ridge matched by a mismatched shape: dropped."""
    T, F = img.shape
    ft = 2 * int(round(6.0 / p.hop_s)) + 1
    fl = 2 * int(round(0.02 / p.log_step)) + 1
    peak = (best >= maximum_filter(best, size=(ft, fl), mode="nearest")) & (best >= p.threshold)
    ti, li = np.nonzero(peak)
    order = np.argsort(-best[ti, li])[: p.max_candidates]
    claimed = np.zeros((T, F), bool)
    wl = int(round(0.015 / p.log_step))  # +-1.5 % around an accepted curve
    out = []
    for t_idx, l_idx in zip(ti[order], li[order]):
        s = bank[which[t_idx, l_idx]]
        tt, ll = _curve_cells(s, t_idx, l_idx, (T, F), s.mid)
        if len(tt) and claimed[tt, ll].mean() > 0.25:
            continue
        ta, la = _curve_cells(s, t_idx, l_idx, (T, F))
        if not len(ta):
            continue
        # explain away: a curve that only crosses an accepted ridge borrows its brightness
        vals = img[ta, la]
        full = vals.mean()
        z_eff = best[t_idx, l_idx] * (np.where(claimed[ta, la], 0.0, vals).mean() / full if full > 0 else 0.0)
        if z_eff < p.threshold:
            continue
        for d in range(-wl, wl + 1):
            claimed[ta, np.clip(la + d, 0, F - 1)] = True
        t = t0 + t_idx * p.hop_s
        if not keep_range[0] <= t < keep_range[1]:
            continue
        mid = _along(img, t_idx, l_idx, s, s.mid)
        flat = max(_along(img, t_idx, l_idx + d, s, s.mid, flat=True) for d in (-2, 0, 2))
        out.append(dict(t=t, f=float(np.exp(lf[l_idx] + p.log_step / 2)), z=float(best[t_idx, l_idx]),
                        shape=s, contrast=mid - flat))
    return out


def _group(cands, p: Params) -> list[Flyover]:
    """Merge components of one pass (same CPA time and shape, integer frequency ratios)."""
    groups: list[list[dict]] = []
    for c in sorted(cands, key=lambda c: -c["z"]):
        if c["contrast"] < p.min_contrast:
            continue
        home = None
        for g in groups:
            r = g[0]
            dt_ok = abs(c["t"] - r["t"]) <= max(3.0, 0.15 * r["shape"].tau)
            v_ok = 1 / 1.4 <= c["shape"].v / r["shape"].v <= 1.4
            q = c["f"] / r["f"]
            ratio_ok = any(abs(q - a / b) < 0.015 * q for a in range(1, 7) for b in range(1, 7))
            if dt_ok and v_ok and ratio_ok:
                home = g
                break
        if home is None:
            groups.append([c])
        elif all(abs(c["f"] - m["f"]) > 0.02 * m["f"] for m in home):
            home.append(c)
    out = []
    for g in groups:
        r = g[0]
        f0, h, fs = _fundamental([m["f"] for m in g])
        out.append(Flyover(t_cpa=r["t"], f0=f0, v=r["shape"].v, d=r["shape"].d, z=r["z"],
                           contrast=float(r["contrast"]), harmonics=h, freqs=fs, f_main=float(r["f"])))
    return out


def _fundamental(freqs, tol=0.03):
    """(f0, harmonic numbers, sorted distinct freqs) for components of one pass."""
    fs = np.unique(np.round(np.asarray(freqs, float), 2))
    keep = [fs[0]]
    for f in fs[1:]:
        if f > keep[-1] * (1 + tol):
            keep.append(f)
    fs = np.array(keep)
    f0 = fs.min()
    for m in (1, 2, 3, 4):
        base = fs.min() / m
        n = fs / base
        if np.all(np.abs(n - np.round(n)) < tol * n):
            f0 = base
            break
    h = np.maximum(1, np.round(fs / f0).astype(int))
    return float(f0), [int(x) for x in h], [float(x) for x in fs]


def _trace_for(fo: Flyover, sp: audio.Spectrogram, p: Params):
    """Trace the strongest component of a detection: (t, f, harmonic) or None."""
    fc = fo.f_main
    h = max(1, int(round(fc / fo.f0)))
    half = float(np.clip(3.0 * fo.d / fo.v, 20.0, 90.0))
    tt = np.array([-half, half])
    r = physics.linear_flyby_ratio(tt, fo.v, fo.d, 0.0, p.c)
    lo, hi = fc * r.min() * 0.95, fc * r.max() * 1.05
    win = sp.window(fo.t_cpa - half, fo.t_cpa + half).band(lo, hi)
    if win.S.shape[1] < 20 or win.S.shape[0] < 5:
        return None
    t, f, _ = measure.trace_ridge(win, fc * r[0], fc * r[1], max_slope_hz_s=max(8.0, 0.2 * fc))
    return (t, f, h, half) if len(t) >= 20 else None


def _fit_job(args):
    """Worker: straight-line flyby fit of one trace -> (fit, inlier mask) or None."""
    t, f, h, c = args
    try:
        fit = measure.fit_flyby(t, f, h, c=c)
    except Exception:  # noqa: BLE001 -- a failed fit leaves the detection unrefined
        return None
    res = fit.curve(t, c) - f
    return fit, np.abs(res) < max(4 * 1.4826 * np.median(np.abs(res)), 0.5)


def _ranges_job(args):
    t, f, fit, c = args
    try:
        rng = measure.profile_ranges(t, f, fit, c=c)
    except Exception:  # noqa: BLE001
        return None
    return {k: [float(x) for x in rng[k]] for k in ("speed_ms", "slant_m", "f0", "audio_cpa")}


def _pool():
    """Worker processes for the fits (least squares is Python-bound). Spawned, not forked:
    numba / FFT thread pools are not fork-safe."""
    import multiprocessing as mp
    import os
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 2), mp_context=mp.get_context("spawn"))


def _refine_all(found: list[Flyover], sp: audio.Spectrogram, p: Params, ex):
    traces = [(fo, _trace_for(fo, sp, p)) for fo in found]
    jobs = [(fo, tr) for fo, tr in traces if tr is not None]
    results = ex.map(_fit_job, [(t, f, h, p.c) for _, (t, f, h, _half) in jobs])
    for (fo, (t, f, h, half)), r in zip(jobs, results):
        if r is None:
            continue
        fit, inl = r
        if abs(fit.t_cpa - fo.t_cpa) > max(10.0, 0.5 * half) or not (6 < fit.v < 290 and 35 < fit.d < 30000):
            continue  # fit ran off or onto its bounds: keep the matched-filter estimate
        if fit.rms > 1.5 * h:
            continue
        fo.refined = True
        fo.f0, fo.v, fo.d, fo.t_cpa, fo.rms = fit.f0, fit.v, fit.d, fit.t_cpa, fit.rms
        fo.points = int(inl.sum())
        fo.freqs = [fo.f0 * hh for hh in fo.harmonics]
        fo._trace = (t[inl], f[inl], fit)


def _curve(f: Flyover, tt, c):
    return physics.linear_flyby(tt, f.f0, f.v, f.d, f.t_cpa, c)


def _dedupe(found: list[Flyover], c: float) -> list[Flyover]:
    """Partial matches of one ridge (other shapes, other anchors) and harmonics found on their
    own: drop a detection when most of its curve runs along a stronger one's harmonics."""
    keep: list[Flyover] = []
    for f in sorted(found, key=lambda f: (-f.refined, -f.z)):
        half = float(np.clip(1.5 * f.d / max(f.v, 1.0), 6.0, 60.0))
        tt = np.linspace(f.t_cpa - half, f.t_cpa + half, 41)
        mine = [h * _curve(f, tt, c) for h in f.harmonics]
        same = None
        for k in keep:
            base = _curve(k, tt, c)
            for curve in mine:
                n = curve / base
                near = np.abs(n - np.round(n)) < 0.03 * n
                if near.mean() >= 0.6 and np.round(n[near]).min() >= 1:
                    same = k
                    break
            if same is not None:
                break
        if same is None:
            keep.append(f)
            continue
        same.z = max(same.z, f.z)
        n = np.array([h * f.f0 for h in f.harmonics]) / same.f0  # attach as harmonics if they line up
        ok = (np.abs(n - np.round(n)) < 0.03 * n) & (np.round(n) >= 1) & (np.round(n) <= 6)
        same.harmonics = sorted(set(same.harmonics) | {int(round(x)) for x in n[ok]})
        same.freqs = [same.f0 * h for h in same.harmonics]
    return keep


def _classify(fo: Flyover) -> str:
    """Aircraft pass well above the microphone; road vehicles, people and machinery pass within
    tens of metres at low speed and make the same kind of curve."""
    return "aircraft" if fo.d >= 150.0 and fo.v >= 30.0 else "close"


def _confidence(fo: Flyover, p: Params) -> str:
    """Calibrated on the video's recording against ADS-B and visual checks: what separates real
    flyovers from other sweeping sounds is a straight-line flyby fit that follows the traced
    ridge closely (residual <= 0.5 % of the frequency for the real ones, >= 1.2 % for the
    rest), not the matched-filter score alone. Three or more harmonics are a strong sign."""
    if fo.kind != "aircraft" or not fo.refined or fo.points < 40:
        return "low"
    rel = fo.rms / max(fo.f_main, 1e-6)
    if rel > 0.010:
        return "low"
    if (fo.z >= p.threshold + 5 and rel <= 0.006) or len(fo.harmonics) >= 3:
        return "high"
    return "medium"


def _merge_components(fl: list[Flyover]) -> list[Flyover]:
    """One aircraft often shows several tonal components (blade-pass harmonics, shaft-rate
    harmonics such as 4:5) whose fits land on slightly different shapes: merge detections
    with the same CPA time and a small-integer frequency ratio into the strongest."""
    order = sorted(fl, key=lambda f: (-(f.kind == "aircraft"), -f.refined, -f.z))
    keep: list[Flyover] = []
    for f in order:
        home = None
        for k in keep:
            # fits of a weaker component land on a biased shape and CPA time (up to ~0.5 tau)
            if abs(f.t_cpa - k.t_cpa) > max(25.0, 0.5 * k.d / max(k.v, 1.0)):
                continue
            q = f.f_main / k.f_main
            if any(abs(q - a / b) < 0.04 * q for a in range(1, 7) for b in range(1, 7)):
                home = k
                break
        if home is None:
            keep.append(f)
            continue
        home.z = max(home.z, f.z)
        n = f.f_main / home.f0
        if abs(n - round(n)) < 0.03 * n and 1 <= round(n) <= 6:
            home.harmonics = sorted(set(home.harmonics) | {int(round(n))})
        home.freqs = sorted(set(home.freqs) | {round(float(f.f_main), 2)})
    return keep


# ------------------------------------------------------------------------------------- driver

def duration(path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], check=True, capture_output=True, text=True).stdout.strip()
    return float(out)


@dataclass
class Result:
    flyovers: list[Flyover]
    overview: audio.Spectrogram | None
    source: str
    start: float
    end: float
    params: Params
    meta: dict = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return any(f.confidence in ("high", "medium") for f in self.flyovers)


def detect(path, start: float = 0.0, dur: float | None = None, p: Params | None = None, log=print,
           meta: dict | None = None) -> Result:
    p = p or Params()
    total = duration(path)
    end = total if dur is None else min(total, start + dur)
    bank = shapes(p)
    flyovers: list[Flyover] = []
    ov_parts = []
    ex = _pool() if p.refine else None
    try:
        s = start
        while s < end - 1:
            e = min(end, s + p.chunk_s)
            a = max(0.0, s - p.margin_s)
            b = min(total, e + p.margin_s)
            log(f"  {s:.0f}-{e:.0f} s: spectrogram")
            x = audio.load_audio(path, sr=p.sr, start=a, duration=b - a)
            sp = audio.spectrogram(x, p.sr, win_s=p.win_s, hop_s=p.hop_s, fmin=p.fmin * 0.8, fmax=p.fmax,
                                   t_offset=a).whiten(line_s=p.line_s)
            img, lf = to_log(sp, p.fmin, p.fmax, p.log_step)
            img = prepare(img)
            log(f"  {s:.0f}-{e:.0f} s: matched filter, {len(bank)} shapes")
            best, which = _scan(img, bank)
            cands = _candidates(img, lf, float(sp.t[0]), best, which, bank, p, (s, e))
            found = _group(cands, p)
            if ex is not None:
                log(f"  {s:.0f}-{e:.0f} s: fitting {len(found)} detections")
                _refine_all(found, sp, p, ex)
            flyovers += _dedupe(found, p.c)
            ov = sp.window(s, e)
            step = max(1, int(round(1.0 / p.hop_s)))
            ov_parts.append(audio.Spectrogram(ov.f, ov.t[::step], ov.S[:, ::step], ov.df, ov.dt * step))
            s = e
        for fo in flyovers:
            fo.kind = _classify(fo)
        flyovers = _merge_components(flyovers)
        for fo in flyovers:
            fo.confidence = _confidence(fo, p)
        keep = [fo for fo in flyovers if fo.confidence != "low" and hasattr(fo, "_trace")]
        if ex is not None and keep:
            log(f"  speed / distance ranges for {len(keep)} flyovers")
            for fo, rng in zip(keep, ex.map(_ranges_job, [(*fo._trace, p.c) for fo in keep])):
                fo.ranges = rng
    finally:
        if ex is not None:
            ex.shutdown()
    flyovers.sort(key=lambda f: f.t_cpa)
    overview = None
    if ov_parts:
        o = ov_parts[0]
        overview = audio.Spectrogram(o.f, np.concatenate([q.t for q in ov_parts]),
                                     np.concatenate([q.S for q in ov_parts], axis=1), o.df, o.dt)
    return Result(flyovers, overview, str(path), start, end, p, meta or {})


# ------------------------------------------------------------------------------------- sources

URL = re.compile(r"^https?://", re.I)


def fetch(url: str, root: Path, log=print):
    """Download the audio track of a web video with yt-dlp into its own directory under ``root``
    (once; later calls reuse it). Returns (audio path, metadata dict). Downloads are untrusted
    data: nothing in them is run.

    Always the whole track: yt-dlp's ``--download-sections`` goes through ffmpeg reading one long
    HTTP response, which YouTube throttles to a standstill on long live archives, and a section
    file would put detection times on the section's clock instead of the video's. Sections are
    cut locally by ``detect(path, start, dur)``."""
    root = Path(root)
    info = json.loads(subprocess.run(["yt-dlp", "-J", "--no-playlist", "--", url], check=True,
                                     capture_output=True, text=True).stdout)
    vid = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{info.get('extractor_key', 'web')}-{info.get('id', 'x')}")
    out = root / vid
    out.mkdir(parents=True, exist_ok=True)
    meta = {k: info.get(k) for k in ("id", "title", "channel", "upload_date", "timestamp", "release_timestamp",
                                     "duration", "was_live", "webpage_url")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    have = sorted(p for p in out.glob("audio.*") if p.suffix not in (".part", ".ytdl", ".json"))
    if have:  # already downloaded
        return have[0], meta
    cmd = ["yt-dlp", "--no-playlist", "--no-progress", "-f", "bestaudio/best", "-o", str(out / "audio.%(ext)s")]
    log(f"  downloading audio of {meta.get('title')!r}")
    subprocess.run(cmd + ["--", url], check=True, capture_output=True)
    got = sorted(p for p in out.glob("audio.*") if p.suffix not in (".part", ".ytdl", ".json"))
    if not got:
        raise RuntimeError("yt-dlp produced no audio file")
    return got[0], meta


# ------------------------------------------------------------------------------------- output

def fmt_time(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(int(round(t)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def report(res: Result) -> str:
    fl = res.flyovers
    strong = [f for f in fl if f.confidence != "low"]
    lines = [f"source: {res.source}  ({fmt_time(res.start)}-{fmt_time(res.end)})"]
    if res.meta.get("title"):
        lines.append(f"title: {res.meta['title']}")
    if res.meta.get("release_timestamp") or res.meta.get("timestamp"):
        from datetime import datetime, timezone
        ts = res.meta.get("release_timestamp") or res.meta.get("timestamp")
        lines.append(f"web metadata time: {datetime.fromtimestamp(ts, timezone.utc):%Y-%m-%d %H:%M:%SZ} "
                     f"({'stream start' if res.meta.get('release_timestamp') else 'upload'})")
    air = [f for f in fl if f.kind == "aircraft"]
    close = [f for f in fl if f.kind != "aircraft"]
    verdict = "YES" if res.found else "NO"
    lines.append(f"aircraft Doppler curves found: {verdict} -- {len(strong)} high/medium, "
                 f"{len(air) - len(strong)} low-confidence; {len(close)} close-range sources "
                 f"(road vehicles etc., < 150 m or < 30 m/s) listed separately")
    head = ["| # | CPA | f0 Hz | harmonics | speed m/s (kt) | CPA dist m | score | fit rms Hz | confidence |",
            "|---|---|---|---|---|---|---|---|---|"]

    def row(i, f):
        rms = f"{f.rms:.2f}" if f.refined else "-"
        return (f"| {i} | {fmt_time(f.t_cpa)} ({f.t_cpa:.1f} s) | {f.f0:.1f} | {f.harmonics} | "
                f"{f.v:.0f} ({f.v / 0.514444:.0f}) | {f.d:.0f} | {f.z:.1f} | {rms} | {f.confidence} |")

    if air:
        lines += ["", "Aircraft"] + head + [row(i, f) for i, f in enumerate(fl, 1) if f.kind == "aircraft"]
    if close:
        lines += ["", "Close-range sources"] + head + [row(i, f) for i, f in enumerate(fl, 1) if f.kind != "aircraft"]
    return "\n".join(lines)


def save(res: Result, out: Path, log=print):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in res.flyovers:
        r = asdict(f)
        r["ranges"] = f.ranges
        rows.append(r)
    (out / "detections.json").write_text(json.dumps({"source": res.source, "start": res.start, "end": res.end,
                                                     "meta": res.meta, "found": res.found,
                                                     "flyovers": rows}, indent=1))
    blocks = []
    n = 0
    for f in res.flyovers:
        ev = f.event(f"D{n + 1}") if f.confidence != "low" else None
        if ev is None:
            continue
        n += 1
        blocks.append("[[event]]\n" + "\n".join(f"{k} = {json.dumps(v)}" for k, v in ev.items())
                      + f"\n# detected: score {f.z:.1f}, fit rms {f.rms:.2f} Hz, {f.confidence} confidence")
    (out / "events.toml").write_text("\n\n".join(blocks) + ("\n" if blocks else ""))
    (out / "report.md").write_text(report(res) + "\n")
    if res.overview is not None:
        plot(res, out / "detections.png")
    log(f"wrote {out}/detections.json, events.toml, report.md" + (", detections.png" if res.overview else ""))


def plot(res: Result, path: Path, max_rows_s: float = 1200.0):
    """Overview spectrogram(s) with each detection's model curve; one row per ``max_rows_s``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sp = res.overview
    n = max(1, math.ceil((sp.t[-1] - sp.t[0]) / max_rows_s))
    fig, axes = plt.subplots(n, 1, figsize=(16, 3.2 * n), squeeze=False)
    colours = {"high": "#4fd18b", "medium": "#fcbf55", "low": "#9a9a9a"}
    strong = [f for f in res.flyovers if f.confidence != "low"]
    fmax = min(sp.f[-1], max([f.f0 * min(max(f.harmonics or [1]), 3) * 1.4 for f in strong] + [600.0]))
    for i in range(n):
        ax = axes[i, 0]
        a = sp.t[0] + i * max_rows_s
        b = min(sp.t[-1], a + max_rows_s)
        w = sp.window(a, b).band(sp.f[0], fmax)
        ax.imshow(w.S, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=5, interpolation="nearest",
                  extent=[w.t[0], w.t[-1], w.f[0], w.f[-1]])
        for k, f in enumerate(res.flyovers, 1):
            if not a <= f.t_cpa < b or f.kind != "aircraft":
                continue
            weak = f.confidence == "low"
            half = float(np.clip(3.0 * f.d / f.v, 20.0, 90.0))
            tt = np.linspace(f.t_cpa - half, f.t_cpa + half, 200)
            base = physics.linear_flyby(tt, f.f0, f.v, f.d, f.t_cpa, res.params.c)
            hs = [h for h in (f.harmonics or [1]) if h * f.f0 <= fmax] or [1]
            for h in hs:
                ax.plot(tt, h * base, color=colours[f.confidence], lw=0.6 if weak else 1.2,
                        alpha=0.35 if weak else 0.95)
            if not weak:
                ax.text(f.t_cpa, min(fmax * 0.93, f.f0 * max(hs) * 1.2), str(k), color=colours[f.confidence],
                        ha="center", va="bottom", fontsize=10, weight="bold")
        ax.set_xlim(a, b)
        ax.set_ylim(w.f[0], fmax)
        ax.set_ylabel("Hz")
    axes[-1, 0].set_xlabel("audio time (s)")
    axes[0, 0].set_title(f"{Path(res.source).name}: {sum(f.confidence != 'low' for f in res.flyovers)} "
                         f"aircraft flyovers (green high, yellow medium confidence; faint grey: weak)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
