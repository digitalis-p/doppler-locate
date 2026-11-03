"""Coarse search: candidate grid points that could have heard the measured flyovers.

Stages (as in the video):

1. Propeller flights (known jet types excluded), resampled every ``sample_dt`` seconds.
2. For every 200 m cell of the CONUS Albers lattice near a track, each *pass* of a flight
   (contiguous stretch within range) yields a CPA record: emission time, 3D slant distance
   (cell height from the DEM, aircraft height above MSL), ground speed at CPA, delayed CPA
   time = emission time + distance / c.
3. Event match: CPA records whose speed and slant distance fall in an event's measured ranges.
4. Offset filter: cells hearing an event-1 match and, ``offset`` +- tol seconds later, an
   event-2 match. (Single-event mode: event-1 delayed CPA inside a known UTC window.)
5. Silence filter: no other audible aircraft within ``radius_m`` whose sound arrives
   between the two events (outside ``margin_s`` of each).

The lattice is tiled; each tile is processed independently with a numba kernel.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass

import numpy as np
import polars as pl
from numba import njit

from . import geo, tracks
from .config import Case
from .dem import DEM

# --------------------------------------------------------------------------- sample table


@dataclass
class Samples:
    fid: list[str]          # flight ids, indexed by fi
    fi: np.ndarray          # int32 flight index per sample
    t: np.ndarray           # float64 emission time (unix s)
    x: np.ndarray           # float64 Albers m
    y: np.ndarray
    h: np.ndarray           # float32 m MSL
    gs: np.ndarray          # float32 m/s
    dt: float

    def __len__(self):
        return len(self.t)


def build_samples(case: Case, exclude_engines=None, t_range=None, dt=None, max_alt_ft=None,
                  only_engines=None, log=print) -> Samples:
    dt = dt or case.search.sample_dt
    trs = load_tracks(case, exclude_engines=exclude_engines, t_range=t_range, max_alt_ft=max_alt_ft,
                      only_engines=only_engines, log=log)
    fids, parts = [], []
    for tr in trs:
        sm = tracks.resample(tr, dt)
        if sm is None:
            continue
        fids.append(sm.fid)
        parts.append((len(fids) - 1, sm))
    n = sum(len(sm.gs) for _, sm in parts)
    fi = np.empty(n, np.int32)
    t = np.empty(n)
    lat = np.empty(n)
    lon = np.empty(n)
    hh = np.empty(n, np.float32)
    gs = np.empty(n, np.float32)
    o = 0
    for i, sm in parts:
        m = len(sm.gs)
        fi[o:o + m] = i
        t[o:o + m] = sm.t
        lat[o:o + m] = sm.lat
        lon[o:o + m] = sm.lon
        hh[o:o + m] = sm.h
        gs[o:o + m] = sm.gs
        o += m
    x, y = geo.to_grid(lon, lat)
    log(f"  {n:,} samples at {dt:g} s")
    return Samples(fids, fi, t, np.asarray(x), np.asarray(y), hh, gs, dt)


def load_tracks(case: Case, icaos=None, exclude_engines=None, t_range=None, max_alt_ft=None,
                only_engines=None, log=print) -> list[tracks.Track]:
    """Track segments exactly as the search sees them (scoring reuses this for stable ids)."""
    s = case.search
    t_range = t_range or _day_range(case)
    df = tracks.load_points(case.adsb, t_range=t_range, bbox=s.bbox, icaos=icaos,
                            exclude_engines=s.exclude_engines if exclude_engines is None else exclude_engines,
                            max_alt_ft=s.max_alt_ft if max_alt_ft is None else max_alt_ft)
    if only_engines is not None:
        df = df.filter(pl.col("engine").is_in(list(only_engines)))
    geoid = geo.Geoid(case.cache) if case.altitude != "baro" else None
    h = tracks.altitude_msl(df, geoid, case.altitude)
    trs = tracks.split_tracks(df, h)
    log(f"  {len(df):,} points -> {len(trs):,} track segments")
    return trs


def _day_range(case: Case):
    from datetime import datetime, timezone
    t0 = datetime.fromisoformat(case.day).replace(tzinfo=timezone.utc).timestamp()
    return (t0 - 3600.0, t0 + 86400.0 + 3600.0)


# --------------------------------------------------------------------------- numba kernels


@njit(cache=True)
def _keep(b, m, a, d2lo, d2hi, gsk, vlo, vhi):
    """Emit a pass only if it is a true local minimum whose refined distance and speed are in band."""
    if not (np.isfinite(b) and np.isfinite(a)):
        return False
    if gsk < vlo or gsk > vhi:
        return False
    den = b - 2.0 * m + a
    d2 = m
    if den > 0:
        off = 0.5 * (b - a) / den
        if off > 1.0:
            off = 1.0
        elif off < -1.0:
            off = -1.0
        d2 = m - 0.25 * (b - a) * off
    return d2lo <= d2 <= d2hi


@njit(cache=True)
def _cpa_tile(sidx, fi, x, y, h, gs, elev, ox, oy, g, nc, rmax, d2lo, d2hi, vlo, vhi, mask):
    """CPA records for one tile.

    sidx: global sample indices for this tile, in track/time order (consecutive samples of a
    track have consecutive indices). elev: (nc, nc) cell elevations; cell (i, j) centre is
    (ox + (i + .5) g, oy + (j + .5) g). Only cells with mask != 0 are visited. A pass is
    emitted when its refined CPA distance is in [sqrt(d2lo), sqrt(d2hi)] and the ground speed
    at the CPA sample is in [vlo, vhi] (filtering here keeps memory bounded in busy tiles).
    Returns arrays (cell, kmin, d2b, d2m, d2a).
    """
    ncell = nc * nc
    last_k = np.full(ncell, -10, np.int64)
    min_d2 = np.full(ncell, np.inf)
    min_k = np.full(ncell, -1, np.int64)
    d2_b = np.full(ncell, np.nan)
    d2_a = np.full(ncell, np.nan)
    prev_d2 = np.full(ncell, np.nan)
    cap = 1 << 14
    o_c = np.empty(cap, np.int64)
    o_k = np.empty(cap, np.int64)
    o_b = np.empty(cap)
    o_m = np.empty(cap)
    o_a = np.empty(cap)
    n_out = 0
    r2 = rmax * rmax
    for q in range(sidx.shape[0]):
        k = sidx[q]
        px = x[k] - ox
        py = y[k] - oy
        i0 = max(0, int((px - rmax) / g))
        i1 = min(nc - 1, int((px + rmax) / g))
        j0 = max(0, int((py - rmax) / g))
        j1 = min(nc - 1, int((py + rmax) / g))
        for i in range(i0, i1 + 1):
            dx = px - (i + 0.5) * g
            for j in range(j0, j1 + 1):
                c = i * nc + j
                if mask[c] == 0:
                    continue
                dy = py - (j + 0.5) * g
                hz = dx * dx + dy * dy
                if hz > r2:
                    continue
                dz = h[k] - elev[i, j]
                d2 = hz + dz * dz
                if last_k[c] == k - 1 and fi[k] == fi[k - 1]:
                    if d2 < min_d2[c]:
                        min_d2[c] = d2
                        min_k[c] = k
                        d2_b[c] = prev_d2[c]
                        d2_a[c] = np.nan
                    elif k == min_k[c] + 1:
                        d2_a[c] = d2
                else:
                    # close the previous pass on this cell
                    if min_k[c] >= 0 and _keep(d2_b[c], min_d2[c], d2_a[c], d2lo, d2hi, gs[min_k[c]], vlo, vhi):
                        if n_out == cap:
                            cap *= 2
                            o_c = _grow_i(o_c, cap)
                            o_k = _grow_i(o_k, cap)
                            o_b = _grow_f(o_b, cap)
                            o_m = _grow_f(o_m, cap)
                            o_a = _grow_f(o_a, cap)
                        o_c[n_out] = c
                        o_k[n_out] = min_k[c]
                        o_b[n_out] = d2_b[c]
                        o_m[n_out] = min_d2[c]
                        o_a[n_out] = d2_a[c]
                        n_out += 1
                    min_d2[c] = d2
                    min_k[c] = k
                    d2_b[c] = np.nan
                    d2_a[c] = np.nan
                prev_d2[c] = d2
                last_k[c] = k
    for c in range(ncell):
        if min_k[c] >= 0 and _keep(d2_b[c], min_d2[c], d2_a[c], d2lo, d2hi, gs[min_k[c]], vlo, vhi):
            if n_out == cap:
                cap *= 2
                o_c = _grow_i(o_c, cap)
                o_k = _grow_i(o_k, cap)
                o_b = _grow_f(o_b, cap)
                o_m = _grow_f(o_m, cap)
                o_a = _grow_f(o_a, cap)
            o_c[n_out] = c
            o_k[n_out] = min_k[c]
            o_b[n_out] = d2_b[c]
            o_m[n_out] = min_d2[c]
            o_a[n_out] = d2_a[c]
            n_out += 1
    return o_c[:n_out], o_k[:n_out], o_b[:n_out], o_m[:n_out], o_a[:n_out]


@njit(cache=True)
def _grow_i(a, cap):
    b = np.empty(cap, a.dtype)
    b[:a.shape[0]] = a
    return b


@njit(cache=True)
def _grow_f(a, cap):
    b = np.empty(cap, a.dtype)
    b[:a.shape[0]] = a
    return b


@njit(cache=True, nogil=True)
def _pair_offsets(c1, t1, c2, t2, offset, tol):
    """All (i, j) with c1[i] == c2[j] and |t2[j] - t1[i] - offset| <= tol.

    c2/t2 must be sorted by (c2, t2)."""
    out_i = []
    out_j = []
    for i in range(c1.shape[0]):
        lo = np.searchsorted(c2, c1[i], side="left")
        hi = np.searchsorted(c2, c1[i], side="right")
        if lo == hi:
            continue
        a = t1[i] + offset - tol
        b = t1[i] + offset + tol
        for j in range(lo, hi):
            if a <= t2[j] <= b:
                out_i.append(i)
                out_j.append(j)
    return np.array(out_i, np.int64), np.array(out_j, np.int64)


@njit(cache=True, nogil=True)
def _silence_cpa(qcell, w_lo, w_hi, f1, f2, thr, ocell, ot, od, ofi):
    """Video rule: count other flights' CPA records in the same cell inside [w_lo, w_hi]
    and nearer than thr. ocell must be sorted."""
    m = qcell.shape[0]
    out = np.zeros(m, np.int32)
    for q in range(m):
        lo = np.searchsorted(ocell, qcell[q], side="left")
        hi = np.searchsorted(ocell, qcell[q], side="right")
        cnt = 0
        for k in range(lo, hi):
            if ofi[k] == f1[q] or ofi[k] == f2[q]:
                continue
            if w_lo[q] <= ot[k] <= w_hi[q] and od[k] < thr[q]:
                cnt += 1
        out[q] = cnt
    return out


@njit(cache=True)
def _silence_hits(cx, cy, cz, ta, tb, st, sx, sy, sh, sfi, f1, f2, radius, c, excl_s, t1e, t2e):
    """Count audible-aircraft samples per survivor.

    For survivor m (cell centre cx, cy, cz; quiet window [ta, tb] in receive time), count
    samples within ``radius`` (3D) whose sound arrives inside the window. Samples of the two
    event flights within ``excl_s`` of their own CPA emission times (t1e, t2e) are ignored.
    st must be sorted ascending.
    """
    m = cx.shape[0]
    out = np.zeros(m, np.int32)
    r2 = radius * radius
    for q in range(m):
        lo = np.searchsorted(st, ta[q] - radius / c - 1.0)
        hi = np.searchsorted(st, tb[q])
        cnt = 0
        for k in range(lo, hi):
            dx = sx[k] - cx[q]
            dy = sy[k] - cy[q]
            hz = dx * dx + dy * dy
            if hz > r2:
                continue
            dz = sh[k] - cz[q]
            d2 = hz + dz * dz
            if d2 > r2:
                continue
            tr = st[k] + np.sqrt(d2) / c
            if tr < ta[q] or tr > tb[q]:
                continue
            if sfi[k] == f1[q] and abs(st[k] - t1e[q]) < excl_s:
                continue
            if sfi[k] == f2[q] and abs(st[k] - t2e[q]) < excl_s:
                continue
            cnt += 1
        out[q] = cnt
    return out


# --------------------------------------------------------------------------- tiling

_G: dict = {}  # worker globals (inherited through fork)
_OFF = 1 << 19
_K = 1 << 20


def _tile_key(x, y, tile):
    tx = np.floor(np.asarray(x) / tile).astype(np.int64) + _OFF
    ty = np.floor(np.asarray(y) / tile).astype(np.int64) + _OFF
    return tx * _K + ty


def _key_xy(key):
    return int(key // _K) - _OFF, int(key % _K) - _OFF


class _TileIndex:
    """Samples bucketed by base tile; neighbourhood gathers return sorted sample indices."""

    def __init__(self, x, y, tile, subset=None):
        idx = np.arange(len(x)) if subset is None else np.asarray(subset)
        k = _tile_key(x[idx], y[idx], tile)
        o = np.argsort(k, kind="stable")
        self.keys = k[o]
        self.idx = idx[o]
        self.tile = tile

    def gather(self, tx, ty, x, y, margin):
        parts = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (tx + dx + _OFF) * _K + (ty + dy + _OFF)
                lo = np.searchsorted(self.keys, key, "left")
                hi = np.searchsorted(self.keys, key, "right")
                if hi > lo:
                    parts.append(self.idx[lo:hi])
        if not parts:
            return np.empty(0, np.int64)
        ii = np.sort(np.concatenate(parts))
        T = self.tile
        ox, oy = tx * T, ty * T
        ok = (x[ii] >= ox - margin) & (x[ii] <= ox + T + margin) & (y[ii] >= oy - margin) & (y[ii] <= oy + T + margin)
        return ii[ok]

    def tiles(self):
        return np.unique(self.keys)


def _tile_elev(tx, ty, tile, g, nc, dem: DEM):
    ox, oy = tx * tile, ty * tile
    lat_ = _G.get("lattice")
    if lat_ is not None and abs(lat_.g - g) < 1e-9:
        return lat_.block(int(round(ox / g)), int(round(oy / g)), nc), ox, oy
    i = np.arange(nc)
    X, Y = np.meshgrid(ox + (i + 0.5) * g, oy + (i + 0.5) * g, indexing="ij")
    lon, lat = geo.from_grid(X.ravel(), Y.ravel())
    return dem.elevation(lat, lon).reshape(nc, nc), ox, oy


def _cpa_records(S, sidx, elev, ox, oy, g, nc, rmax, c, dband=None, vband=None, mask=None):
    """CPA records with parabolic sub-sample refinement (see ``_cpa_tile``)."""
    d2lo, d2hi = (0.0, rmax * rmax) if dband is None else (dband[0] ** 2, dband[1] ** 2)
    vlo, vhi = (-np.inf, np.inf) if vband is None else vband
    if mask is None:
        mask = np.ones(nc * nc, np.uint8)
    cc, kk, d2b, d2m, d2a = _cpa_tile(sidx, S.fi, S.x, S.y, S.h, S.gs, elev, ox, oy, g, nc, rmax, d2lo, d2hi,
                                      float(vlo), float(vhi), mask)
    if len(cc) == 0:
        return None
    den = d2b - 2 * d2m + d2a
    off = np.where(den > 0, 0.5 * (d2b - d2a) / np.where(den > 0, den, 1), 0.0)
    off = np.clip(off, -1, 1)
    d = np.sqrt(np.maximum(d2m - 0.25 * (d2b - d2a) * off, 0.0))
    tau = S.t[kk] + off * S.dt
    nb = np.where(off > 0, kk + 1, kk - 1)
    gs = S.gs[kk] + np.abs(off) * (S.gs[nb] - S.gs[kk])
    trecv = tau + d / c
    ci, cj = cc // nc, cc % nc
    gx = (round(ox / g) + ci).astype(np.int64)  # global lattice indices
    gy = (round(oy / g) + cj).astype(np.int64)
    cell = (gx + _OFF) * (_K * 4) + (gy + _OFF)
    return cc, kk, d, tau, gs, trecv, gx, gy, cell


def _process_tile(key):
    G = _G
    S: Samples = G["S"]
    case: Case = G["case"]
    s = case.search
    tile, g, nc = G["tile"], s.grid_m, G["nc"]
    tx, ty = _key_xy(key)
    ev = case.events
    nev = min(2, len(ev))
    sidx = [G["SI"][k].gather(tx, ty, S.x, S.y, G["rmax"][k]) for k in range(nev)]
    if any(len(x) < 3 for x in sidx):
        return None
    elev, ox, oy = _tile_elev(tx, ty, tile, g, nc, G["dem"])
    stats = {}
    recs = []
    for k in range(nev):
        e = ev[k]
        rec = _cpa_records(S, sidx[k], elev, ox, oy, g, nc, G["rmax"][k], case.c, dband=e.slant_m,
                           vband=(e.speed_ms[0] - 5.0, e.speed_ms[1] + 5.0))
        if rec is None:
            return stats, None
        cc, kk, d, tau, gs, trecv, gx, gy, cell = rec
        m = (gs >= e.speed_ms[0]) & (gs <= e.speed_ms[1]) & (d >= e.slant_m[0]) & (d <= e.slant_m[1])
        if k == 0 and s.utc_window is not None:
            m &= (trecv >= s.utc_window[0]) & (trecv <= s.utc_window[1])
        i = np.flatnonzero(m)
        stats[f"cpa_records_e{k + 1}"] = len(cc)
        stats[f"e{k + 1}"] = len(i)
        stats[f"e{k + 1}_cells"] = len(np.unique(cell[i]))
        if not len(i):
            return stats, None
        recs.append({"cc": cc[i], "kk": kk[i], "d": d[i], "tau": tau[i], "gs": gs[i], "t": trecv[i],
                     "gx": gx[i], "gy": gy[i], "cell": cell[i]})
    R1 = recs[0]
    if nev == 2:
        R2 = recs[1]
        o = np.lexsort((R2["t"], R2["cell"]))
        R2 = {k: v[o] for k, v in R2.items()}
        a, b = _pair_offsets(R1["cell"], R1["t"], R2["cell"], R2["t"], case.offset_s, s.offset_tol_s)
        same = (S.fi[R1["kk"][a]] == S.fi[R2["kk"][b]]) & (np.abs(R1["tau"][a] - R2["tau"][b]) < 1.0)
        r1, r2 = a[~same], b[~same]
    else:
        r1, r2 = np.arange(len(R1["t"])), None
    stats["offset"] = len(r1)
    stats["offset_cells"] = len(np.unique(R1["cell"][r1]))
    if not len(r1):
        return stats, None
    cell = R1["cell"]
    ci, cj = R1["cc"] // nc, R1["cc"] % nc
    res = {
        "gx": R1["gx"][r1], "gy": R1["gy"][r1],
        "x": ox + (ci[r1] + 0.5) * g, "y": oy + (cj[r1] + 0.5) * g, "elev": elev[ci[r1], cj[r1]],
        "f1": S.fi[R1["kk"][r1]], "tau1": R1["tau"][r1], "t1": R1["t"][r1], "d1": R1["d"][r1], "v1": R1["gs"][r1],
    }
    if r2 is not None:
        res |= {"f2": S.fi[R2["kk"][r2]], "tau2": R2["tau"][r2], "t2": R2["t"][r2], "d2": R2["d"][r2],
                "v2": R2["gs"][r2]}
        sil = case.silence
        if sil.enabled and sil.mode == "cpa":
            qcell = cell[r1]
            w_lo = np.minimum(res["t1"], res["t2"])
            w_hi = np.maximum(res["t1"], res["t2"])
            # only survivor cells, only samples whose sound could arrive inside the quiet windows
            si = G["SA"].gather(tx, ty, S.x, S.y, G["rsil"])
            si = si[(S.t[si] >= w_lo.min() - 120.0) & (S.t[si] <= w_hi.max() + 30.0)]
            mask = np.zeros(nc * nc, np.uint8)
            mask[R1["cc"][r1]] = 1
            orec = _cpa_records(S, si, elev, ox, oy, g, nc, G["rsil"], case.c, mask=mask)
            thr = sil.ratio * np.maximum(res["d1"], res["d2"])
            if orec is None:
                hits = np.zeros(len(qcell), np.int32)
            else:
                _, okk, od, _, _, ot, _, _, ocell = orec
                o = np.argsort(ocell, kind="stable")
                hits = _silence_cpa(qcell, w_lo, w_hi, res["f1"], res["f2"], thr, ocell[o], ot[o], od[o],
                                    S.fi[okk[o]])
            res["heard_between"] = hits
            quiet = hits == 0
            stats["silent"] = int(quiet.sum())
            stats["silent_cells"] = len(np.unique(qcell[quiet]))
        elif sil.enabled:
            ta = res["t1"] + sil.margin_s
            tb = res["t2"] - sil.margin_s
            A = G["A"]
            ai = G["AI"].gather(tx, ty, A["x"], A["y"], sil.radius_m)
            hits = _silence_hits(res["x"], res["y"], res["elev"], ta, tb, A["t"][ai], A["x"][ai], A["y"][ai],
                                 A["h"][ai], A["fi"][ai], res["f1"], res["f2"], sil.radius_m, case.c,
                                 sil.margin_s, res["tau1"], res["tau2"])
            if "J" in G:
                J = G["J"]
                ji = G["JI"].gather(tx, ty, J["x"], J["y"], sil.jet_radius_m)
                none = np.full(len(ta), -1, np.int32)
                hits += _silence_hits(res["x"], res["y"], res["elev"], ta, tb, J["t"][ji], J["x"][ji], J["y"][ji],
                                      J["h"][ji], J["fi"][ji], none, none, sil.jet_radius_m, case.c, 0.0,
                                      res["tau1"], res["tau2"])
            res["heard_between"] = hits
            quiet = hits == 0
            stats["silent"] = int(quiet.sum())
            stats["silent_cells"] = len(np.unique(cell[r1][quiet]))
    return stats, res


def _default_workers(gb_per_worker: float = 1.5) -> int:
    """CPU count minus two, capped so workers fit in currently available memory."""
    try:
        with open("/proc/meminfo") as f:
            avail_kb = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable:"))
        by_mem = int(avail_kb / 1e6 / gb_per_worker)
    except (OSError, StopIteration):
        by_mem = 8
    return max(1, min(mp.cpu_count() - 2, by_mem))


def run(case: Case, log=print) -> pl.DataFrame:
    """Run the coarse search and write ``<out>/candidates.parquet``."""
    s = case.search
    if s.gpu and (not case.silence.enabled or case.silence.mode == "cpa"):
        from . import search_gpu
        if search_gpu.available():
            log("search backend: cuda")
            return search_gpu.run(case, log=log)
    t_start = time.time()
    ev = case.events
    vmax = max(e.speed_ms[1] for e in ev) + s.speed_margin_ms
    dmax = max(e.slant_m[1] for e in ev[:2])
    log("loading propeller tracks")
    S = build_samples(case, log=log)
    keep = []
    for e in ev[:2]:
        lo, hi = e.speed_ms[0] - s.speed_margin_ms, e.speed_ms[1] + s.speed_margin_ms
        keep.append((S.gs >= lo) & (S.gs <= hi))
        log(f"  {e.name}: {keep[-1].sum():,} samples within {lo:.0f}-{hi:.0f} m/s, slant <= {e.slant_m[1]:.0f} m")

    A = None
    J = None
    if len(ev) >= 2 and case.silence.enabled and case.silence.mode != "cpa":
        o = np.argsort(S.t, kind="stable")
        A = {"t": S.t[o], "x": S.x[o], "y": S.y[o], "h": S.h[o], "fi": S.fi[o]}
        if case.silence.jets_below_ft > 0:
            log("loading low jets for the silence filter")
            SJ = build_samples(case, exclude_engines=(), only_engines=("J",), max_alt_ft=case.silence.jets_below_ft,
                               log=log)
            oj = np.argsort(SJ.t, kind="stable")
            J = {"t": SJ.t[oj], "x": SJ.x[oj], "y": SJ.y[oj], "h": SJ.h[oj], "fi": SJ.fi[oj] + 10_000_000}

    tile = s.tile_km * 1000.0
    nc = int(round(tile / s.grid_m))
    rmax = [e.slant_m[1] + (e.speed_ms[1] + s.speed_margin_ms) * S.dt for e in ev[:2]]
    rsil = case.silence.ratio * dmax + vmax * S.dt
    if max(max(rmax), rsil, case.silence.radius_m) >= tile:
        raise ValueError("tile_km must exceed the search / silence radius")
    SI = [_TileIndex(S.x, S.y, tile, np.flatnonzero(k)) for k in keep]

    def dilate(keys):
        tx, ty = keys // _K, keys % _K
        return set(np.unique(np.concatenate([(tx + dx) * _K + (ty + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)])).tolist())

    work = dilate(SI[0].tiles())
    for idx in SI[1:]:
        work &= dilate(idx.tiles())
    work = np.array(sorted(work), dtype=np.int64)
    log(f"  {len(work):,} tiles of {s.tile_km:g} km")

    dem = DEM(case.cache / "dem", res=90)
    tiles_1deg = set()
    work_for_dem = work if not s.elevation_lattice else []
    for k in work_for_dem:
        tx, ty = _key_xy(k)
        cx = np.array([tx, tx + 1, tx, tx + 1]) * tile
        cy = np.array([ty, ty, ty + 1, ty + 1]) * tile
        lon, lat = geo.from_grid(cx, cy)
        for la in range(int(np.floor(lat.min())), int(np.floor(lat.max())) + 1):
            for lo in range(int(np.floor(lon.min())), int(np.floor(lon.max())) + 1):
                tiles_1deg.add((la, lo))
    if tiles_1deg:
        log(f"  DEM: {len(tiles_1deg)} GLO-90 tiles (cached under {dem.dir})")
        dem.prefetch(tiles_1deg)

    _G.clear()
    from .dem import ElevationLattice
    lattice = ElevationLattice.build(case.cache, s.grid_m, log=log) if s.elevation_lattice else None
    _G.update(S=S, SI=SI, case=case, tile=tile, nc=nc, rmax=rmax, dem=dem, rsil=rsil, lattice=lattice)
    if len(ev) >= 2 and case.silence.enabled and case.silence.mode == "cpa":
        _G["SA"] = _TileIndex(S.x, S.y, tile)
    if A is not None:
        _G["A"], _G["AI"] = A, _TileIndex(A["x"], A["y"], tile)
    if J is not None:
        _G["J"], _G["JI"] = J, _TileIndex(J["x"], J["y"], tile)
    work = [int(k) for k in work]
    nproc = s.workers or _default_workers()
    totals: dict[str, int] = {}
    results = []
    log(f"  scanning tiles on {nproc} processes")
    ctx = mp.get_context("fork")
    with ctx.Pool(nproc) as pool:
        for out in pool.imap_unordered(_process_tile, work, chunksize=4):
            if out is None:
                continue
            st, res = out
            for k, v in st.items():
                totals[k] = totals.get(k, 0) + v
            if res is not None:
                results.append(res)
    _G.clear()
    log("  counts: " + ", ".join(f"{k}={v:,}" for k, v in totals.items()))
    if not results:
        log("no candidates")
        return pl.DataFrame()
    cols = {k: np.concatenate([r[k] for r in results]) for k in results[0]}
    lon, lat = geo.from_grid(cols["x"], cols["y"])
    cols["lon"], cols["lat"] = np.asarray(lon), np.asarray(lat)
    fid = np.array(S.fid, dtype=object)
    cols["f1"] = fid[cols["f1"]]
    if "f2" in cols:
        cols["f2"] = fid[cols["f2"]]
        cols["offset_err"] = cols["t2"] - cols["t1"] - case.offset_s
    df = pl.DataFrame(cols)
    case.out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(case.out / "candidates.parquet")
    (case.out / "search_counts.json").write_text(__import__("json").dumps(totals, indent=1))
    log(f"wrote {len(df):,} candidate rows in {time.time() - t_start:.0f} s")
    return df
