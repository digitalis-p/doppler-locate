"""GPU coarse search (CUDA kernel in native/dopploc_search.cu). Same stages and output as
``search.run``; one process, samples resident on the GPU, CONUS processed in large chunks.

Differences from the CPU path: every local minimum of the cell-to-track distance is a CPA
record (the CPU path keeps one per contiguous pass), and the neighbours used for the minimum
test and the parabola are the flight's own adjacent samples even when they fall outside the
event's speed band.
"""

from __future__ import annotations

import ctypes as C
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl

from . import geo, native
from .config import Case
from .dem import ElevationLattice
from .search import _K, _TileIndex, _key_xy, _pair_offsets, _silence_cpa, build_samples


def _lib():
    lib = native._load()
    if lib is None or native.backend() != "cuda" or not hasattr(lib, "dl_search_chunk"):
        return None
    P = np.ctypeslib.ndpointer
    d, f, i32, i64 = (P(np.float64, flags="C"), P(np.float32, flags="C"), P(np.int32, flags="C"),
                      P(np.int64, flags="C"))
    lib.dl_search_upload.argtypes = [i32, d, d, d, f, f, C.c_int64, C.c_int]
    lib.dl_search_upload.restype = C.c_int
    lib.dl_search_free.restype = C.c_int
    lib.dl_search_chunk.argtypes = [i64, C.c_int64, f, C.c_int32, C.c_int32, C.c_double, C.c_double, C.c_double,
                                    C.c_void_p, C.c_double, C.c_double, C.c_double, C.c_double, C.c_double,
                                    C.c_double, C.c_int64, i64, i64, d, d, f, f, C.POINTER(C.c_int64)]
    lib.dl_search_chunk.restype = C.c_int
    return lib


def available() -> bool:
    return _lib() is not None


def _records(lib, sidx, elev, ox, oy, g, rmax, dband, vband, c, mask=None, cap=1 << 22):
    nx, ny = elev.shape
    sidx = np.ascontiguousarray(sidx, np.int64)
    mptr = None if mask is None else np.ascontiguousarray(mask, np.uint8).ctypes.data_as(C.c_void_p)
    while True:
        out = dict(cell=np.empty(cap, np.int64), k=np.empty(cap, np.int64), tau=np.empty(cap),
                   t=np.empty(cap), d=np.empty(cap, np.float32), gs=np.empty(cap, np.float32))
        n = C.c_int64(0)
        rc = lib.dl_search_chunk(sidx, len(sidx), elev, nx, ny, ox, oy, g, mptr, rmax, dband[0], dband[1],
                                 vband[0], vband[1], c, cap, out["cell"], out["k"], out["tau"], out["t"], out["d"],
                                 out["gs"], C.byref(n))
        if rc:
            raise RuntimeError(f"dl_search_chunk failed ({rc})")
        if n.value <= cap:
            return {k: v[:n.value] for k, v in out.items()}
        cap = int(n.value * 1.1) + 1024


def run(case: Case, log=print, chunk_km: float = 250.0, threads: int = 8) -> pl.DataFrame:
    lib = _lib()
    if lib is None:
        raise RuntimeError("GPU search needs the CUDA build of the native library")
    s = case.search
    sil = case.silence
    if sil.enabled and sil.mode != "cpa":
        raise RuntimeError("GPU search implements the 'cpa' silence rule only")
    t_start = time.time()
    ev = case.events[:2]
    two = len(ev) == 2
    g = s.grid_m
    log("loading propeller tracks")
    S = build_samples(case, log=log)
    rc = lib.dl_search_upload(S.fi, S.x, S.y, S.t, S.h.astype(np.float32), S.gs.astype(np.float32), len(S), 0)
    if rc:
        raise RuntimeError(f"dl_search_upload failed ({rc})")
    lattice = ElevationLattice.build(case.cache, g, log=log)
    chunk = chunk_km * 1000.0
    nc = int(round(chunk / g))
    rmax = [e.slant_m[1] + (e.speed_ms[1] + s.speed_margin_ms) * S.dt for e in ev]
    vmax = max(e.speed_ms[1] for e in ev) + s.speed_margin_ms
    rsil = sil.ratio * max(e.slant_m[1] for e in ev) + vmax * S.dt
    SI = []
    for e in ev:
        keep = (S.gs >= e.speed_ms[0] - s.speed_margin_ms) & (S.gs <= e.speed_ms[1] + s.speed_margin_ms)
        SI.append(_TileIndex(S.x, S.y, chunk, np.flatnonzero(keep)))
    SA = _TileIndex(S.x, S.y, chunk) if two and sil.enabled else None

    def dilate(keys):
        tx, ty = keys // _K, keys % _K
        return set(np.unique(np.concatenate([(tx + dx) * _K + (ty + dy) for dx in (-1, 0, 1)
                                             for dy in (-1, 0, 1)])).tolist())

    work = dilate(SI[0].tiles())
    for idx in SI[1:]:
        work &= dilate(idx.tiles())
    work = sorted(work)
    log(f"  {len(work)} chunks of {chunk_km:g} km on the GPU")
    def do_chunk(key):
        """Records, pairing and silence for one chunk -> (columns or None, counts)."""
        cnt: dict[str, int] = {}

        def add(k, v):
            cnt[k] = cnt.get(k, 0) + int(v)

        tx, ty = _key_xy(key)
        ox, oy = tx * chunk, ty * chunk
        gx0, gy0 = int(round(ox / g)), int(round(oy / g))
        elev = np.ascontiguousarray(lattice.block(gx0, gy0, nc), np.float32)
        recs = []
        cmask = None
        for k, e in enumerate(ev):
            sidx = SI[k].gather(tx, ty, S.x, S.y, rmax[k])
            if len(sidx) < 3:
                break
            r = _records(lib, sidx, elev, ox, oy, g, rmax[k], e.slant_m, e.speed_ms, case.c, mask=cmask)
            if k == 0 and s.utc_window is not None:
                m = (r["t"] >= s.utc_window[0]) & (r["t"] <= s.utc_window[1])
                r = {a: b[m] for a, b in r.items()}
            add(f"e{k + 1}", len(r["t"]))
            if not len(r["t"]):
                break
            recs.append(r)
            if k == 0 and two:  # event 2 only matters in cells that have an event-1 record
                cmask = np.zeros(nc * nc, np.uint8)
                cmask[r["cell"]] = 1
        if len(recs) < len(ev):
            return None, cnt
        R1 = recs[0]
        if two:
            R2 = recs[1]
            o = np.lexsort((R2["t"], R2["cell"]))
            R2 = {a: b[o] for a, b in R2.items()}
            a, b = _pair_offsets(R1["cell"], R1["t"], R2["cell"], R2["t"], case.offset_s, s.offset_tol_s)
            same = (S.fi[R1["k"][a]] == S.fi[R2["k"][b]]) & (np.abs(R1["tau"][a] - R2["tau"][b]) < 1.0)
            r1, r2 = a[~same], b[~same]
        else:
            r1, r2 = np.arange(len(R1["t"])), None
        add("offset", len(r1))
        if not len(r1):
            return None, cnt
        ci, cj = R1["cell"][r1] // nc, R1["cell"][r1] % nc
        res = {
            "gx": gx0 + ci, "gy": gy0 + cj, "x": ox + (ci + 0.5) * g, "y": oy + (cj + 0.5) * g,
            "elev": elev[ci, cj].astype(np.float64),
            "f1": S.fi[R1["k"][r1]], "tau1": R1["tau"][r1], "t1": R1["t"][r1], "d1": R1["d"][r1].astype(np.float64),
            "v1": R1["gs"][r1].astype(np.float64),
        }
        if two:
            res |= {"f2": S.fi[R2["k"][r2]], "tau2": R2["tau"][r2], "t2": R2["t"][r2],
                    "d2": R2["d"][r2].astype(np.float64), "v2": R2["gs"][r2].astype(np.float64)}
            if sil.enabled:
                w_lo = np.minimum(res["t1"], res["t2"])
                w_hi = np.maximum(res["t1"], res["t2"])
                si = SA.gather(tx, ty, S.x, S.y, rsil)
                si = si[(S.t[si] >= w_lo.min() - 120.0) & (S.t[si] <= w_hi.max() + 30.0)]
                mask = np.zeros(nc * nc, np.uint8)
                qcell = R1["cell"][r1]
                mask[qcell] = 1
                orc = _records(lib, si, elev, ox, oy, g, rsil, (0.0, rsil), (-np.inf, np.inf), case.c, mask=mask)
                o = np.argsort(orc["cell"], kind="stable")
                thr = sil.ratio * np.maximum(res["d1"], res["d2"])
                hits = _silence_cpa(qcell, w_lo, w_hi, res["f1"], res["f2"], thr, orc["cell"][o], orc["t"][o],
                                    orc["d"][o].astype(np.float64), S.fi[orc["k"][o]])
                res["heard_between"] = hits
                add("silent", (hits == 0).sum())
        return res, cnt

    # Host-side pairing dominates (the kernels take seconds), so chunks run on a thread pool:
    # numpy sorts, the nogil numba kernels and the ctypes calls all release the GIL.
    totals: dict[str, int] = {}
    results = []
    t_w = time.time()
    with ThreadPoolExecutor(threads) as ex:
        for res, cnt in ex.map(do_chunk, work):
            for k, v in cnt.items():
                totals[k] = totals.get(k, 0) + v
            if res is not None:
                results.append(res)
    lib.dl_search_free()
    log("  counts: " + ", ".join(f"{k}={v:,}" for k, v in totals.items()))
    log(f"  chunks done in {time.time() - t_w:.0f} s on {threads} threads")
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
    (case.out / "search_counts.json").write_text(json.dumps(totals | {"backend": "cuda"}, indent=1))
    log(f"wrote {len(df):,} candidate rows in {time.time() - t_start:.0f} s")
    return df
