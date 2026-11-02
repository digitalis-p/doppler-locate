"""Flyovers a recording *should* contain, from ADS-B, for a known place and audio clock.

Used as ground truth for ``dopploc detect`` and to sanity-check a solved location: every
aircraft segment near the place is resampled at 1 s, and each local minimum of its range to
the observer within ``radius_m`` is one expected pass (emission time, slant distance, delayed
arrival time = emission + distance / c, mapped to audio time with ``audio0_utc``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from . import geo, tracks
from .dem import DEM

OBS_HEIGHT_M = 2.0


def passes(adsb_dir: Path, cache: Path, lat: float, lon: float, t0_utc: float, t1_utc: float,
           radius_m: float = 10000.0, c: float = 343.0, altitude: str = "auto",
           audio0_utc: float | None = None) -> pl.DataFrame:
    dlat = radius_m / 111_000.0 + 0.05
    dlon = dlat / max(0.2, np.cos(np.radians(lat)))
    df = tracks.load_points(adsb_dir, t_range=(t0_utc - 900, t1_utc + 900),
                            bbox=(lon - dlon, lat - dlat, lon + dlon, lat + dlat), exclude_engines=(),
                            max_alt_ft=None)
    if df.is_empty():
        return pl.DataFrame()
    geoid = geo.Geoid(cache) if altitude != "baro" else None
    h = tracks.altitude_msl(df, geoid, altitude)
    elev = float(DEM(Path(cache) / "dem", res=90).elevation(np.array([lat]), np.array([lon]))[0])
    obs = geo.lla_to_ecef(lat, lon, elev + OBS_HEIGHT_M)
    rows = []
    for tr in tracks.split_tracks(df, h):
        sm = tracks.resample(tr, dt=1.0)
        if sm is None or len(sm.gs) < 3:
            continue
        R = np.linalg.norm(sm.P - obs, axis=1)
        i = np.flatnonzero((R[1:-1] <= R[:-2]) & (R[1:-1] < R[2:]) & (R[1:-1] < radius_m)) + 1
        for k in i:
            a, b, cc = R[k - 1] ** 2, R[k] ** 2, R[k + 1] ** 2
            den = a - 2 * b + cc
            off = float(np.clip(0.5 * (a - cc) / den, -1, 1)) if den > 0 else 0.0
            d = float(np.sqrt(max(b - 0.25 * (a - cc) * off, 0.0)))
            tau = sm.t0 + (k + off) * sm.dt
            rows.append(dict(fid=tr.fid, icao=tr.icao, type=tr.type, engine=tr.engine, callsign=tr.callsign,
                             tau=tau, t_arrival=tau + d / c, d=d, speed=float(sm.gs[k]), alt_m=float(sm.h[k]),
                             ground_m=float(np.sqrt(max(d * d - (sm.h[k] - elev) ** 2, 0.0)))))
    out = pl.DataFrame(rows)
    if out.is_empty():
        return out
    out = out.filter(pl.col("t_arrival").is_between(t0_utc, t1_utc)).sort("t_arrival")
    if audio0_utc is not None:
        out = out.with_columns((pl.col("t_arrival") - audio0_utc).alias("audio_t"))
    return out


def match(detected_t: np.ndarray, expected_t: np.ndarray, tol_s: float = 15.0, ok=None):
    """Greedy one-to-one matching by time; ``ok(i_det, j_exp)`` can add a physical consistency
    test. Returns (pairs [(i_det, j_exp)], unmatched det, unmatched exp)."""
    pairs = []
    used_e: set[int] = set()
    used_d: set[int] = set()
    cand = sorted(((abs(a - b), i, j) for i, a in enumerate(detected_t) for j, b in enumerate(expected_t)
                   if abs(a - b) <= tol_s and (ok is None or ok(i, j))))
    for _, i, j in cand:
        if i in used_d or j in used_e:
            continue
        pairs.append((i, j))
        used_d.add(i)
        used_e.add(j)
    return (pairs, [i for i in range(len(detected_t)) if i not in used_d],
            [j for j in range(len(expected_t)) if j not in used_e])
