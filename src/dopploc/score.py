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

from . import audio, geo, native, physics, search, tracks
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


@njit(parallel=True, cache=True)
def _model_many(obs, t_expect, u_grid, t0, dt, P, V, c, win, atm):
    n = obs.shape[0]
    nu = u_grid.shape[0]
    np_ = P.shape[0]
    t_cpa = np.full(n, np.nan)
    d_cpa = np.full(n, np.nan)
    R = np.full((n, nu), np.nan)
    for i in prange(n):
        lo = max(0, int((t_expect[i] - win - t0) / dt))
        hi = min(np_, int((t_expect[i] + win - t0) / dt) + 1)
        if hi - lo < 3:
            continue
        tau, d, tdel = physics.track_cpa(obs[i], t0 + lo * dt, dt, P[lo:hi], c)
        t_cpa[i] = tdel
        d_cpa[i] = d
        r, _, ok = physics.track_doppler_ratio(tdel + u_grid, obs[i], t0, dt, P, V, atm[i])
        for j in range(nu):
            if ok[j]:
                R[i, j] = r[j]
    return t_cpa, d_cpa, R


def model_ratios(obs_ecef, sm: tracks.Sampled, t_expect, u_grid, c, atm=None, win: float = 90.0):
    """Delayed CPA and f/f0 on ``u_grid`` (s relative to delayed CPA) for many observers.

    t_expect: approximate delayed CPA per observer (from the search) picks the right pass
    (CPA searched within +-win s). ``atm``: (8,) or (n, 8) atmosphere rows (physics.still_air).
    Returns (t_cpa (n,), d_cpa (n,), R (n, len(u_grid))).
    """
    obs = np.ascontiguousarray(obs_ecef, dtype=np.float64)
    a = physics.still_air(c) if atm is None else atm
    a = np.ascontiguousarray(np.broadcast_to(a, (len(obs), 8)), dtype=np.float64)
    return _model_many(obs, np.asarray(t_expect, np.float64), np.asarray(u_grid, np.float64), sm.t0, sm.dt, sm.P,
                       sm.V, c, win, a)


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
    sp = audio.brightness(case.audio, ev.audio_cpa - ev.half_width_s, ev.audio_cpa + ev.half_width_s,
                          sr=4000 if fband[1] < 1800 else 16000, win_s=sc.win_s, hop_s=sc.hop_s,
                          fmin=fband[0], fmax=fband[1], line_s=sc.line_suppress_s)
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
    nat = native.backend() != "numba"
    if spec.trace is not None:
        tt, tf, th = spec.trace
        if nat:
            return native.score_trace(R, u0, du, base, deltas, f0s, tt, tf, th, case.score.trace_sigma_hz)
        return score_trace(R.astype(np.float64), u0, du, base, deltas, f0s, tt, tf, th, case.score.trace_sigma_hz)
    if nat:
        return native.score_brightness(R, u0, du, base, deltas, f0s, spec.harms, spec.tau, spec.B, spec.fb0, spec.dfb)
    return score_brightness(R.astype(np.float64), u0, du, base, deltas, f0s, spec.harms, spec.tau, spec.B,
                            spec.fb0, spec.dfb)


class Models:
    """Forward model for arbitrary (flight, observer) rows: native batch or numba per flight."""

    def __init__(self, flights: dict, c: float):
        self.flights = flights
        self.c = c
        self.pack = native.TrackPack(flights) if native.backend() != "numba" and flights else None

    def __call__(self, fids, obs, t_expect, u_grid, atm=None):
        """atm: (n, 8) atmosphere rows (see physics.track_doppler_ratio), or None for still air."""
        n = len(fids)
        ok = np.array([f in self.flights for f in fids], bool)
        t_cpa = np.full(n, np.nan)
        d_cpa = np.full(n, np.nan)
        R = np.full((n, len(u_grid)), np.nan, np.float32)
        if not ok.any():
            return t_cpa, d_cpa, R
        ii = np.flatnonzero(ok)
        w = np.tile(physics.still_air(self.c), (n, 1)) if atm is None else atm
        if self.pack is not None:
            t_cpa[ii], d_cpa[ii], R[ii] = native.model(self.pack, fids[ii], obs[ii], t_expect[ii], u_grid, self.c,
                                                       atm=w[ii])
            return t_cpa, d_cpa, R
        for f in np.unique(fids[ii]):
            jj = ii[fids[ii] == f]
            a, b, r = model_ratios(obs[jj], self.flights[f], t_expect[jj], u_grid, self.c, w[jj])
            t_cpa[jj], d_cpa[jj], R[jj] = a, b, r
        return t_cpa, d_cpa, R


# --------------------------------------------------------------------------- driver


def _aircraft_h(flights, fids, taus):
    """Aircraft height (m MSL) of each row's flight at emission time tau."""
    out = np.full(len(fids), np.nan)
    for f in np.unique(fids):
        sm = flights.get(f)
        if sm is None:
            continue
        m = fids == f
        out[m] = np.interp(taus[m], sm.t, sm.h)
    return out


def case_atmosphere(case: Case, lat, lon, elev, flights, rows: dict, log=print):
    """Per event, (n, 8) atmosphere rows [c_e, w_s(3), c_p, w_m(3)] in ECEF, plus a summary
    dict of columns to keep. rows: {"f1", "tau1", "t1", ["f2", "tau2", "t2"]}."""
    n = len(lat)
    zero = np.zeros(n)
    out, cols = [], {}
    mode = case.atmosphere
    if mode in ("asos", "fixed"):
        if mode == "fixed":
            we, wn = np.full(n, case.wind_ms[0]), np.full(n, case.wind_ms[1])
        else:
            from . import wind as wmod
            W = wmod.Wind(wmod.fetch_day(case.day, case.cache, log=log))
            we, wn = W.at(lat, lon, rows["t1"])
            log(f"  atmosphere: ASOS surface wind, median {np.median(np.hypot(we, wn)):.1f} m/s, c = {case.c:g}")
        w = geo.enu_to_ecef_vec(lat, lon, we, wn, zero)
        a = np.c_[np.full(n, case.c), w, np.full(n, case.c), w]
        cols.update(wind_e=we, wind_n=wn)
        out = [a, a]
    elif mode == "gfs":
        from . import atmos
        A = atmos.Atmosphere(case.day, case.cache, log=log)
        for k in (1, 2):
            if f"f{k}" not in rows:
                break
            h_ac = _aircraft_h(flights, rows[f"f{k}"], rows[f"tau{k}"])
            h_ac = np.where(np.isfinite(h_ac), h_ac, elev + 1000.0)
            ce, ua, va, um, vm = A.effective(lat, lon, rows[f"t{k}"], elev + OBS_HEIGHT_M, h_ac)
            ws = geo.enu_to_ecef_vec(lat, lon, ua, va, zero)
            wm = geo.enu_to_ecef_vec(lat, lon, um, vm, zero)
            out.append(np.c_[ce, ws, ce, wm])
            cols.update({f"c_e{k}": ce, f"wind_e{k}": ua, f"wind_n{k}": va, f"h_ac{k}": h_ac})
            log(f"  atmosphere (GFS) event {k}: median c_e {np.median(ce):.1f} m/s, "
                f"median wind at aircraft {np.median(np.hypot(ua, va)):.1f} m/s")
        out.append(out[0])
    else:
        a = np.tile(physics.still_air(case.c), (n, 1))
        out = [a, a]
    return out, cols


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


def _sweeps(case: Case, fine: bool):
    sc = case.score
    d0, d1, dstep = sc.dt_sweep
    if not fine:
        dstep = max(dstep, 0.25)
    deltas = np.arange(d0, d1 + 1e-9, dstep)

    def f0_grid(e: Event):
        span = e.f0 * sc.f0_sweep_frac
        step = sc.f0_step_hz if fine else max(sc.f0_step_hz, 0.2)
        return np.arange(e.f0 - span, e.f0 + span + 1e-9, step)
    return deltas, f0_grid


def _score_block(case, specs, models: Models, obs, f1, f2, t1s, t2s, deltas, f0_grid, frame_step, u_grid,
                 out, idx_all, atm=None, chunk: int = 32768):
    """Score rows ``idx_all`` into the arrays of ``out`` (locked two-event scoring)."""
    ev = case.events
    two = f2 is not None
    sp = []
    for s_ in specs:
        if s_.trace is None and frame_step > 1:
            sp.append(EventSpec(s_.ev, s_.tau[::frame_step].copy(), np.ascontiguousarray(s_.B[:, ::frame_step]),
                                s_.fb0, s_.dfb, s_.harms, None))
        else:
            sp.append(s_)
    f0g1 = f0_grid(ev[0])
    f0g2 = f0_grid(ev[1]) if two else None
    u0, du = u_grid[0], u_grid[1] - u_grid[0]
    for s0 in range(0, len(idx_all), chunk):
        idx = idx_all[s0:s0 + chunk]
        r = np.arange(len(idx))
        tc1, dc1, R1 = models(f1[idx], obs[idx], t1s[idx], u_grid, None if atm is None else atm[0][idx])
        S1, A1 = _sweep(case, sp[0], R1, u0, du, np.full(len(idx), ev[0].audio_cpa), deltas, f0g1)
        out["t1m"][idx], out["d1m"][idx] = tc1, dc1
        b1 = np.argmax(S1, axis=1)  # "locked": event 1 alone sets the clock shift
        sel_d = b1
        if two:
            tc2, dc2, R2 = models(f2[idx], obs[idx], t2s[idx], u_grid, None if atm is None else atm[1][idx])
            base2 = ev[0].audio_cpa + (tc2 - tc1)  # event 2 placed by flight geometry relative to event 1
            base2 = np.where(np.isfinite(base2), base2, 1e12)
            S2, A2 = _sweep(case, sp[1], R2, u0, du, base2, deltas, f0g2)
            out["t2m"][idx], out["d2m"][idx] = tc2, dc2
            bj = np.argmax(S1 + S2, axis=1)  # "joint": one shared shift for both events
            out["S_locked"][idx] = S1[r, b1] + S2[r, b1]
            out["S_joint"][idx] = S1[r, bj] + S2[r, bj]
            if case.score.mode == "joint":
                sel_d = bj
            out["S2"][idx] = np.where(np.isfinite(tc2), S2[r, sel_d], np.nan)
            out["f02"][idx] = f0g2[A2[r, sel_d]]
        out["S1"][idx] = np.where(np.isfinite(tc1), S1[r, sel_d], np.nan)
        out["delta"][idx] = deltas[sel_d]
        out["f01"][idx] = f0g1[A1[r, sel_d]]


def score_candidates(case: Case, cand: pl.DataFrame, log=print, refine: bool = True,
                     fine_top: int = 5000) -> pl.DataFrame:
    """Model and score every candidate: a coarse sweep for all rows, then a fine sweep for
    the best ``fine_top`` rows plus the best row of every flight pair."""
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

    pad = max(abs(sc.dt_sweep[0]), abs(sc.dt_sweep[1])) + 2.0
    hw = max(e.half_width_s for e in ev)
    u_grid = np.arange(-hw - pad, hw + pad + 1e-9, sc.model_dt)
    specs = [prepare_event(case, e, pad) for e in ev[: (2 if two else 1)]]

    n = len(cand)
    fids = set(cand["f1"].to_list()) | (set(cand["f2"].to_list()) if two else set())
    log(f"  reconstructing {len(fids)} flights")
    flights = load_flights(case, fids, -np.inf, np.inf, log=log)
    models = Models(flights, case.c)
    rows = {k: cand[k].to_numpy() for k in ("f1", "tau1", "t1", "f2", "tau2", "t2") if k in cand.columns}
    for k in ("f1", "f2"):
        if k in rows:
            rows[k] = rows[k].astype(str)
    atm, atm_cols = case_atmosphere(case, lat, lon, elev, flights, rows, log=log)
    cand = cand.with_columns(*[pl.Series(k, v) for k, v in atm_cols.items()])
    log(f"  compute backend: {native.backend()}")
    f1 = cand["f1"].to_numpy().astype(str)
    f2 = cand["f2"].to_numpy().astype(str) if two else None
    t1s = cand["t1"].to_numpy()
    t2s = cand["t2"].to_numpy() if two else None
    cols = ("t1m", "d1m", "S1", "delta", "f01", "t2m", "d2m", "S2", "f02", "S_locked", "S_joint")
    coarse = {k: np.full(n, np.nan) for k in cols}
    deltas, f0_grid = _sweeps(case, fine=False)
    log(f"  coarse sweep: {len(deltas)} shifts x {len(f0_grid(ev[0]))} f0 steps")
    _score_block(case, specs, models, obs, f1, f2, t1s, t2s, deltas, f0_grid, 2, u_grid, coarse, np.arange(n), atm)
    tot = coarse["S1"] + (coarse["S2"] if two else 0)
    tot = np.where(np.isfinite(tot), tot, -np.inf)
    sel = set(np.argsort(-tot)[:fine_top].tolist())
    keyc = np.char.add(np.char.add(f1, "|"), f2) if two else f1
    for k in np.unique(keyc):  # best coarse row of every flight pair
        ii = np.flatnonzero(keyc == k)
        sel.add(int(ii[np.argmax(tot[ii])]))
    sel = np.array(sorted(sel))
    fine = {k: v.copy() for k, v in coarse.items()}
    deltas, f0_grid = _sweeps(case, fine=True)
    log(f"  fine sweep on {len(sel):,} rows: {len(deltas)} shifts x {len(f0_grid(ev[0]))} f0 steps")
    _score_block(case, specs, models, obs, f1, f2, t1s, t2s, deltas, f0_grid, 1, u_grid, fine, sel, atm)
    res = cand.with_columns(*[pl.Series(k, v) for k, v in fine.items()],
                            pl.Series("score_coarse", tot), pl.Series("refined", np.isin(np.arange(n), sel)))
    if two:
        res = res.with_columns((pl.col("S1") + pl.col("S2")).alias("score"))
        res = res.with_columns((pl.col("t2m") - pl.col("t1m") - case.offset_s).alias("offset_err_model"))
    else:
        res = res.with_columns(pl.col("S1").alias("score"))
    # unrefined rows keep coarse scores, which are not comparable; rank refined rows first
    res = res.sort(["refined", "score"], descending=[True, True], nulls_last=True)
    log(f"  scored in {time.time() - t0:.0f} s")
    return res


def best_per_pair(res: pl.DataFrame) -> pl.DataFrame:
    keys = ["f1", "f2"] if "f2" in res.columns else ["f1"]
    if "refined" in res.columns:
        res = res.filter(pl.col("refined"))
    return (res.filter(pl.col("score").is_finite()).sort("score", descending=True)
            .group_by(keys, maintain_order=True).first().sort("score", descending=True))


def heatmap(case: Case, row: dict, half_m: float = 1500.0, step_m: float = 25.0, log=print):
    """Score a dense grid around one candidate for its flight pair (finer than the search)."""
    xs = np.arange(-half_m, half_m + 1e-9, step_m)
    X, Y = np.meshgrid(row["x"] + xs, row["y"] + xs, indexing="ij")
    # the base pass times (t: delayed CPA, tau: emission CPA) stand in for every grid point; the
    # GFS atmosphere lookup needs tau for the aircraft altitude
    base = {k: v for k, v in row.items() if k in ("f1", "f2", "t1", "t2", "tau1", "tau2")}
    df = pl.DataFrame({"x": X.ravel(), "y": Y.ravel()}).with_columns(
        *[pl.lit(v).alias(k) for k, v in base.items()])
    res = score_candidates(case, df, log=log, refine=False, fine_top=len(df))
    return res
