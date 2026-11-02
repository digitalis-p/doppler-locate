"""Fine scoring: model the exact Doppler curve each candidate would hear and compare it
with the spectrogram.

For a candidate observer and a flight, the model is built from the real (interpolated)
ADS-B track with emission-time radial velocity (``physics.track_doppler_ratio``). It is
placed on the spectrogram by its delayed CPA time.

Locked two-event scoring: event 1 is swept over a small time shift (measurement error of
its CPA time on the audio clock) and over the fundamental frequency; the best shift fixes
the audio <-> UTC clock. Event 2 is then placed purely by flight geometry (its delayed CPA
relative to event 1's) and only its fundamental is swept. A wrong candidate cannot "fix"
event 2's timing.

Brightness score: mean whitened spectrogram brightness under the model curve (all scored
harmonics). Trace score (faint curves): mean Gaussian agreement between the model and a
traced (t, f) line.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import polars as pl
from numba import njit, prange
from scipy.ndimage import maximum_filter1d

from . import audio, geo, physics, search, tracks
from .config import Case, Event
from .dem import DEM

OBS_HEIGHT_M = 2.0


# --------------------------------------------------------------------------- kernels


@njit(cache=True)
def _lin(Rrow, u0, du, u):
    x = (u - u0) / du
    n = Rrow.shape[0]
    if x < 0.0 or x > n - 1:
        return np.nan
    i = min(int(x), n - 2)
    w = x - i
    return Rrow[i] + w * (Rrow[i + 1] - Rrow[i])


@njit(parallel=True, cache=True)
def score_brightness(R, u0, du, base, deltas, f0s, harms, tau, B, fb0, dfb):
    """Brightness score for every candidate, time shift and fundamental.

    R: (ncand, nu) f/f0 on the receive-time grid u0 + du*k relative to the delayed CPA.
    base: (ncand,) audio time where u = 0 sits before the shift (u_j = tau_j - base - delta).
    Returns S (ncand, ndelta) = best mean brightness over f0s and the arg index into f0s.
    """
    nc = R.shape[0]
    nd = deltas.shape[0]
    nf = f0s.shape[0]
    nt = tau.shape[0]
    nh = harms.shape[0]
    nbin = B.shape[0]
    S = np.full((nc, nd), -1.0)
    A = np.zeros((nc, nd), np.int32)
    for c in prange(nc):
        ratio = np.empty(nt)
        for di in range(nd):
            for j in range(nt):
                ratio[j] = _lin(R[c], u0, du, tau[j] - base[c] - deltas[di])
            for fi in range(nf):
                acc = 0.0
                for j in range(nt):
                    r = ratio[j]
                    if np.isnan(r):
                        continue
                    for hh in range(nh):
                        y = (harms[hh] * f0s[fi] * r - fb0) / dfb
                        if y < 0.0 or y > nbin - 1:
                            continue
                        yi = min(int(y), nbin - 2)
                        w = y - yi
                        acc += B[yi, j] + w * (B[yi + 1, j] - B[yi, j])
                val = acc / (nt * nh)
                if val > S[c, di]:
                    S[c, di] = val
                    A[c, di] = fi
    return S, A


@njit(parallel=True, cache=True)
def score_trace(R, u0, du, base, deltas, f0s, tt, tf, th, sigma):
    """Trace-line score: mean exp(-0.5 ((h f0 r(u) - f_trace) / sigma)^2) over trace points."""
    nc = R.shape[0]
    nd = deltas.shape[0]
    nf = f0s.shape[0]
    nt = tt.shape[0]
    S = np.full((nc, nd), -1.0)
    A = np.zeros((nc, nd), np.int32)
    for c in prange(nc):
        ratio = np.empty(nt)
        for di in range(nd):
            for j in range(nt):
                ratio[j] = _lin(R[c], u0, du, tt[j] - base[c] - deltas[di])
            for fi in range(nf):
                acc = 0.0
                for j in range(nt):
                    r = ratio[j]
                    if np.isnan(r):
                        continue
                    e = (th[j] * f0s[fi] * r - tf[j]) / sigma
                    acc += np.exp(-0.5 * e * e)
                val = acc / nt
                if val > S[c, di]:
                    S[c, di] = val
                    A[c, di] = fi
    return S, A


# --------------------------------------------------------------------------- model curves


@dataclass
class FlightModel:
    fid: str
    sm: tracks.Sampled


def load_flights(case: Case, fids, t_lo: float, t_hi: float, dt: float = 0.5, log=print):
    """Resampled ECEF tracks for flight ids produced by the search."""
    icaos = sorted({f.split(":")[0] for f in fids})
    trs = search.load_tracks(case, icaos=icaos, log=lambda *_: None)
    want = set(fids)
    out = {}
    for tr in trs:
        if tr.fid in want:
            sm = tracks.resample(tr, dt, t_range=(t_lo, t_hi))
            if sm is not None:
                out[tr.fid] = sm
    missing = want - set(out)
    if missing:
        log(f"  warning: {len(missing)} flights not reconstructed (e.g. {sorted(missing)[:3]})")
    return out


def model_ratios(obs_ecef, sm: tracks.Sampled, t_expect, u_grid, c):
    """Delayed CPA and f/f0 on ``u_grid`` (s relative to delayed CPA) for many observers.

    t_expect: approximate delayed CPA per observer (from the search) to pick the right pass.
    Returns (t_cpa (n,), d_cpa (n,), R (n, len(u_grid))).
    """
    n = obs_ecef.shape[0]
    t_cpa = np.empty(n)
    d_cpa = np.empty(n)
    R = np.full((n, len(u_grid)), np.nan)
    tt = sm.t
    for i in range(n):
        # restrict the CPA search to +-90 s around the expected pass
        lo = np.searchsorted(tt, t_expect[i] - 90.0)
        hi = np.searchsorted(tt, t_expect[i] + 90.0)
        if hi - lo < 3:
            t_cpa[i] = np.nan
            d_cpa[i] = np.nan
            continue
        tau, d, tdel = physics.track_cpa(obs_ecef[i], tt[lo], sm.dt, sm.P[lo:hi], c)
        t_cpa[i], d_cpa[i] = tdel, d
        r, _, ok = physics.track_doppler_ratio(tdel + u_grid, obs_ecef[i], sm.t0, sm.dt, sm.P, sm.V, c)
        R[i] = np.where(ok, r, np.nan)
    return t_cpa, d_cpa, R


# --------------------------------------------------------------------------- spectrogram prep


@dataclass
class EventSpec:
    ev: Event
    tau: np.ndarray        # frame times (audio s)
    B: np.ndarray | None   # whitened brightness (nbin, nt)
    fb0: float
    dfb: float
    harms: np.ndarray
    trace: tuple | None    # (t, f, h)


def prepare_event(case: Case, ev: Event, pad: float) -> EventSpec:
    sc = case.score
    hmax = max(ev.harmonics)
    fband = ev.fband or (max(15.0, 0.6 * ev.f0 * min(ev.harmonics)), 1.6 * ev.f0 * hmax)
    sp = audio.clip_spectrogram(case.audio, ev.audio_cpa, half_width=ev.half_width_s + pad + 2 * sc.win_s,
                                sr=4000 if fband[1] < 1800 else 16000, win_s=sc.win_s, hop_s=sc.hop_s,
                                fmin=fband[0], fmax=fband[1])
    sp = sp.window(ev.audio_cpa - ev.half_width_s, ev.audio_cpa + ev.half_width_s).whiten()
    B = sp.S
    if sc.freq_tol_bins > 0:
        B = maximum_filter1d(B, size=2 * sc.freq_tol_bins + 1, axis=0)
    trace = None
    if ev.trace:
        tr = np.loadtxt(ev.trace, delimiter=",", ndmin=2, comments="#")
        h = tr[:, 2] if tr.shape[1] > 2 else np.ones(len(tr))
        trace = (tr[:, 0].copy(), tr[:, 1].copy(), h.astype(float))
    return EventSpec(ev, sp.t.astype(float), np.ascontiguousarray(B, dtype=np.float64), float(sp.f[0]),
                     float(sp.df), np.asarray(ev.harmonics, dtype=np.float64), trace)


def _sweep(case: Case, spec: EventSpec, R, u0, du, base, deltas, f0s):
    if spec.trace is not None:
        tt, tf, th = spec.trace
        return score_trace(R, u0, du, base, deltas, f0s, tt, tf, th, case.score.trace_sigma_hz)
    return score_brightness(R, u0, du, base, deltas, f0s, spec.harms, spec.tau, spec.B, spec.fb0, spec.dfb)


# --------------------------------------------------------------------------- driver


def refine_points(cand: pl.DataFrame, grid_m: float, refine_m: float) -> pl.DataFrame:
    """Split each grid cell into sub-cells of ``refine_m`` (e.g. 200 m -> four 100 m points)."""
    k = int(round(grid_m / refine_m))
    if k <= 1:
        return cand.with_columns(pl.lit(0).alias("sub"))
    offs = (np.arange(k) + 0.5) * refine_m - grid_m / 2
    ox, oy = np.meshgrid(offs, offs, indexing="ij")
    sub = pl.DataFrame({"sub": np.arange(k * k), "dx": ox.ravel(), "dy": oy.ravel()})
    out = cand.join(sub, how="cross").with_columns((pl.col("x") + pl.col("dx")).alias("x"),
                                                   (pl.col("y") + pl.col("dy")).alias("y")).drop("dx", "dy")
    return out


def score_candidates(case: Case, cand: pl.DataFrame, log=print, refine: bool = True) -> pl.DataFrame:
    sc = case.score
    t0 = time.time()
    ev = case.events
    two = len(ev) >= 2 and "f2" in cand.columns
    if refine:
        cand = refine_points(cand, case.search.grid_m, sc.refine_m)
    lon, lat = geo.from_grid(cand["x"].to_numpy(), cand["y"].to_numpy())
    lon, lat = np.asarray(lon), np.asarray(lat)
    dem = DEM(case.cache / "dem", res=sc.dem_res)
    elev = dem.elevation(lat, lon)
    obs = geo.lla_to_ecef(lat, lon, elev + OBS_HEIGHT_M)
    cand = cand.with_columns(pl.Series("lon", lon), pl.Series("lat", lat), pl.Series("elev", elev))
    log(f"scoring {len(cand):,} points ({cand.select('f1').n_unique()} event-1 flights)")

    d0, d1, dstep = sc.dt_sweep
    deltas = np.arange(d0, d1 + 1e-9, dstep)
    pad = max(abs(d0), abs(d1)) + 2.0
    u_grid = np.arange(-ev[0].half_width_s - pad, ev[0].half_width_s + pad + 1e-9, sc.model_dt)
    specs = [prepare_event(case, e, pad) for e in ev[: (2 if two else 1)]]

    def f0_grid(e: Event):
        span = e.f0 * sc.f0_sweep_frac
        return np.arange(e.f0 - span, e.f0 + span + 1e-9, sc.f0_step_hz)

    n = len(cand)
    t_lo = float(cand["t1"].min()) - 600
    t_hi = float((cand["t2"] if two else cand["t1"]).max()) + 600
    fids = set(cand["f1"].to_list()) | (set(cand["f2"].to_list()) if two else set())
    log(f"  reconstructing {len(fids)} flights")
    flights = load_flights(case, fids, t_lo, t_hi, log=log)

    out = {k: np.full(n, np.nan) for k in ("t1m", "d1m", "S1", "delta", "f01", "t2m", "d2m", "S2", "f02",
                                           "S2_joint", "S1_joint", "delta_joint")}
    f1 = cand["f1"].to_numpy()
    f2 = cand["f2"].to_numpy() if two else None
    t1s = cand["t1"].to_numpy()
    t2s = cand["t2"].to_numpy() if two else None
    f0g1 = f0_grid(ev[0])
    f0g2 = f0_grid(ev[1]) if two else None
    key = np.array([f"{a}|{b}" for a, b in zip(f1, f2)]) if two else f1
    for k in np.unique(key):
        idx = np.flatnonzero(key == k)
        a = f1[idx[0]]
        if a not in flights or (two and f2[idx[0]] not in flights):
            continue
        tc1, dc1, R1 = model_ratios(obs[idx], flights[a], t1s[idx], u_grid, case.c)
        base1 = np.full(len(idx), ev[0].audio_cpa)
        S1, A1 = _sweep(case, specs[0], R1, u_grid[0], sc.model_dt, base1, deltas, f0g1)
        best = np.argmax(S1, axis=1)
        r = np.arange(len(idx))
        out["t1m"][idx], out["d1m"][idx] = tc1, dc1
        out["S1"][idx] = S1[r, best]
        out["delta"][idx] = deltas[best]
        out["f01"][idx] = f0g1[A1[r, best]]
        if two:
            b = f2[idx[0]]
            tc2, dc2, R2 = model_ratios(obs[idx], flights[b], t2s[idx], u_grid, case.c)
            base2 = ev[0].audio_cpa + (tc2 - tc1)  # locked: geometry sets event 2's position
            S2, A2 = _sweep(case, specs[1], R2, u_grid[0], sc.model_dt, base2, deltas, f0g2)
            out["t2m"][idx], out["d2m"][idx] = tc2, dc2
            out["S2"][idx] = S2[r, best]
            out["f02"][idx] = f0g2[A2[r, best]]
            J = S1 + S2
            bj = np.argmax(J, axis=1)
            out["S1_joint"][idx] = S1[r, bj]
            out["S2_joint"][idx] = S2[r, bj]
            out["delta_joint"][idx] = deltas[bj]
    res = cand.with_columns(*[pl.Series(k, v) for k, v in out.items()])
    if two:
        if sc.mode == "joint":
            res = res.with_columns((pl.col("S1_joint") + pl.col("S2_joint")).alias("score"))
        else:
            res = res.with_columns((pl.col("S1") + pl.col("S2")).alias("score"))
        res = res.with_columns((pl.col("t2m") - pl.col("t1m") - case.offset_s).alias("offset_err_model"))
    else:
        res = res.with_columns(pl.col("S1").alias("score"))
    res = res.sort("score", descending=True, nulls_last=True)
    log(f"  scored in {time.time() - t0:.0f} s")
    return res


def best_per_pair(res: pl.DataFrame) -> pl.DataFrame:
    keys = ["f1", "f2"] if "f2" in res.columns else ["f1"]
    return (res.filter(pl.col("score").is_finite()).sort("score", descending=True)
            .group_by(keys, maintain_order=True).first().sort("score", descending=True))


def heatmap(case: Case, row: dict, half_m: float = 1500.0, step_m: float = 25.0, log=print):
    """Score a dense grid around one candidate for its flight pair (finer than the search)."""
    xs = np.arange(-half_m, half_m + 1e-9, step_m)
    X, Y = np.meshgrid(row["x"] + xs, row["y"] + xs, indexing="ij")
    base = {k: v for k, v in row.items() if k in ("f1", "f2", "t1", "t2")}
    df = pl.DataFrame({"x": X.ravel(), "y": Y.ravel()}).with_columns(
        *[pl.lit(v).alias(k) for k, v in base.items()])
    res = score_candidates(case, df, log=log, refine=False)
    return res
