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
def _cpa_tile(sidx, fi, x, y, h, elev, ox, oy, g, nc, rmax, d2max):
    """CPA records for one tile.

    sidx: global sample indices for this tile, in track/time order (consecutive samples of a
    track have consecutive indices). elev: (nc, nc) cell elevations; cell (i, j) centre is
    (ox + (i + .5) g, oy + (j + .5) g). Returns arrays (ci, cj, kmin, d2b, d2m, d2a).
    """
    ncell = nc * nc
    last_k = np.full(ncell, -10, np.int64)
    min_d2 = np.full(ncell, np.inf)
    min_k = np.full(ncell, -1, np.int64)
    d2_b = np.full(ncell, np.nan)
    d2_a = np.full(ncell, np.nan)
    prev_d2 = np.full(ncell, np.nan)
    start_k = np.full(ncell, -1, np.int64)
    cap = 1 << 16
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
                dy = py - (j + 0.5) * g
                hz = dx * dx + dy * dy
                if hz > r2:
                    continue
                c = i * nc + j
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
                    if min_k[c] >= 0 and min_d2[c] <= d2max and np.isfinite(d2_b[c]) and np.isfinite(d2_a[c]):
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
                    start_k[c] = k
                    min_d2[c] = d2
                    min_k[c] = k
                    d2_b[c] = np.nan
                    d2_a[c] = np.nan
                prev_d2[c] = d2
                last_k[c] = k
    for c in range(ncell):
        if min_k[c] >= 0 and min_d2[c] <= d2max and np.isfinite(d2_b[c]) and np.isfinite(d2_a[c]):
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


@njit(cache=True)
def _pair_offsets(c1, t1, c2, t2, offset, tol):
    """All (i, j) with c1[i] == c2[j] and |t2[j] - t1[i] - offset| <= tol.

    c2/t2 must be sorted by (c2, t2)."""
    out_i = []
    out_j = []
    n2 = c2.shape[0]
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
    i = np.arange(nc)
    X, Y = np.meshgrid(ox + (i + 0.5) * g, oy + (i + 0.5) * g, indexing="ij")
    lon, lat = geo.from_grid(X.ravel(), Y.ravel())
    return dem.elevation(lat, lon).reshape(nc, nc), ox, oy


def _process_tile(key):
    G = _G
    S: Samples = G["S"]
    case: Case = G["case"]
    s = case.search
    tile, g, nc = G["tile"], s.grid_m, G["nc"]
    tx, ty = _key_xy(key)
    sidx = G["SI"].gather(tx, ty, S.x, S.y, G["rmax"])
    if len(sidx) < 3:
        return None
    elev, ox, oy = _tile_elev(tx, ty, tile, g, nc, G["dem"])
    cc, kk, d2b, d2m, d2a = _cpa_tile(sidx, S.fi, S.x, S.y, S.h, elev, ox, oy, g, nc, G["rmax"], G["d2max"])
    if len(cc) == 0:
        return None
    den = d2b - 2 * d2m + d2a
    off = np.where(den > 0, 0.5 * (d2b - d2a) / np.where(den > 0, den, 1), 0.0)
    off = np.clip(off, -1, 1)
    d = np.sqrt(np.maximum(d2m - 0.25 * (d2b - d2a) * off, 0.0))
    tau = S.t[kk] + off * S.dt
    nb = np.where(off > 0, kk + 1, kk - 1)
    gs = S.gs[kk] + np.abs(off) * (S.gs[nb] - S.gs[kk])
    trecv = tau + d / case.c
    ci, cj = cc // nc, cc % nc
    gx = (round(ox / g) + ci).astype(np.int64)  # global lattice indices
    gy = (round(oy / g) + cj).astype(np.int64)
    cell = (gx + _OFF) * (_K * 4) + (gy + _OFF)
    ev = case.events
    stats = {"cpa_records": len(cc)}

    def match(e):
        return (gs >= e.speed_ms[0]) & (gs <= e.speed_ms[1]) & (d >= e.slant_m[0]) & (d <= e.slant_m[1])

    m1 = match(ev[0])
    if s.utc_window is not None:
        m1 &= (trecv >= s.utc_window[0]) & (trecv <= s.utc_window[1])
    i1 = np.flatnonzero(m1)
    stats["e1"] = len(i1)
    stats["e1_cells"] = len(np.unique(cell[i1]))
    if len(ev) >= 2:
        i2 = np.flatnonzero(match(ev[1]))
        stats["e2"] = len(i2)
        stats["e2_cells"] = len(np.unique(cell[i2]))
        if not len(i1) or not len(i2):
            return stats, None
        o = np.lexsort((trecv[i2], cell[i2]))
        i2 = i2[o]
        a, b = _pair_offsets(cell[i1], trecv[i1], cell[i2], trecv[i2], case.offset_s, s.offset_tol_s)
        r1, r2 = i1[a], i2[b]
        same = (S.fi[kk[r1]] == S.fi[kk[r2]]) & (np.abs(tau[r1] - tau[r2]) < 1.0)
        r1, r2 = r1[~same], r2[~same]
    else:
        r1, r2 = i1, None
    stats["offset"] = len(r1)
    stats["offset_cells"] = len(np.unique(cell[r1]))
    if not len(r1):
        return stats, None
    res = {
        "gx": gx[r1], "gy": gy[r1],
        "x": ox + (ci[r1] + 0.5) * g, "y": oy + (cj[r1] + 0.5) * g, "elev": elev[ci[r1], cj[r1]],
        "f1": S.fi[kk[r1]], "tau1": tau[r1], "t1": trecv[r1], "d1": d[r1], "v1": gs[r1],
    }
    if r2 is not None:
        res |= {"f2": S.fi[kk[r2]], "tau2": tau[r2], "t2": trecv[r2], "d2": d[r2], "v2": gs[r2]}
        sil = case.silence
        if sil.enabled:
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


def run(case: Case, log=print) -> pl.DataFrame:
    """Run the coarse search and write ``<out>/candidates.parquet``."""
    s = case.search
    t_start = time.time()
    ev = case.events
    vmin = min(e.speed_ms[0] for e in ev) - s.speed_margin_ms
    vmax = max(e.speed_ms[1] for e in ev) + s.speed_margin_ms
    dmax = max(e.slant_m[1] for e in ev)
    log("loading propeller tracks")
    S = build_samples(case, log=log)
    keep_speed = (S.gs >= vmin) & (S.gs <= vmax)
    log(f"  {keep_speed.sum():,} samples within {vmin:.0f}-{vmax:.0f} m/s")

    A = None
    J = None
    if len(ev) >= 2 and case.silence.enabled:
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
    rmax = dmax + vmax * S.dt
    if rmax >= tile or case.silence.radius_m >= tile:
        raise ValueError("tile_km must exceed the search radius")
    SI = _TileIndex(S.x, S.y, tile, np.flatnonzero(keep_speed))
    base = SI.tiles()
    btx = base // _K
    bty = base % _K
    work = np.unique(np.concatenate([(btx + dx) * _K + (bty + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]))
    log(f"  {len(work):,} tiles of {s.tile_km:g} km")

    dem = DEM(case.cache / "dem", res=90)
    tiles_1deg = set()
    for k in work:
        tx, ty = _key_xy(k)
        cx = np.array([tx, tx + 1, tx, tx + 1]) * tile
        cy = np.array([ty, ty, ty + 1, ty + 1]) * tile
        lon, lat = geo.from_grid(cx, cy)
        for la in range(int(np.floor(lat.min())), int(np.floor(lat.max())) + 1):
            for lo in range(int(np.floor(lon.min())), int(np.floor(lon.max())) + 1):
                tiles_1deg.add((la, lo))
    log(f"  DEM: {len(tiles_1deg)} GLO-90 tiles (cached under {dem.dir})")
    dem.prefetch(tiles_1deg)

    _G.clear()
    _G.update(S=S, SI=SI, case=case, tile=tile, nc=nc, rmax=rmax, d2max=dmax * dmax, dem=dem)
    if A is not None:
        _G["A"], _G["AI"] = A, _TileIndex(A["x"], A["y"], tile)
    if J is not None:
        _G["J"], _G["JI"] = J, _TileIndex(J["x"], J["y"], tile)
    work = [int(k) for k in work]
    nproc = s.workers or max(1, mp.cpu_count() - 2)
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
