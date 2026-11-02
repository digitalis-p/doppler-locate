"""Flight tracks from ingested ADS-B points: segmentation, altitude datum, resampling.

Positions are resampled with cubic Hermite interpolation in ECEF, using the reported
ground speed / track / vertical rate as the end-point derivatives, so turns and speed
changes between sparse ADS-B reports are represented smoothly. These derivatives are
also what the Doppler model uses as the aircraft velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from . import geo
from .adsb import FPM, FT, KT

DEFAULT_EXCLUDE_ENGINES = ("J", "R")  # the video's rule: drop known jets, keep unknown types


@dataclass
class Track:
    """One continuous airborne segment of one airframe (raw ADS-B points)."""
    fid: str               # "<icao>:<unix start of segment>"
    icao: str
    type: str | None
    engine: str
    callsign: str | None
    t: np.ndarray          # unix s
    lat: np.ndarray
    lon: np.ndarray
    h: np.ndarray          # metres above mean sea level (see altitude_msl)
    gs: np.ndarray         # m/s
    trk: np.ndarray        # deg
    vr: np.ndarray         # m/s


@dataclass
class Sampled:
    """Uniformly resampled track."""
    fid: str
    t0: float
    dt: float
    P: np.ndarray          # (n, 3) ECEF m
    V: np.ndarray          # (n, 3) ECEF m/s
    lat: np.ndarray
    lon: np.ndarray
    h: np.ndarray          # m MSL
    gs: np.ndarray         # m/s horizontal

    @property
    def t(self):
        return self.t0 + self.dt * np.arange(len(self.gs))


def scan_points(adsb_dir: Path) -> pl.LazyFrame:
    return pl.scan_parquet(str(Path(adsb_dir) / "points-*.parquet"))


def aircraft(adsb_dir: Path) -> pl.DataFrame:
    return pl.read_parquet(Path(adsb_dir) / "aircraft.parquet")


def load_points(adsb_dir: Path, t_range=None, bbox=None, exclude_engines=DEFAULT_EXCLUDE_ENGINES,
                icaos=None, max_alt_ft: float | None = None) -> pl.DataFrame:
    """Points joined with aircraft info, filtered. bbox = (lon0, lat0, lon1, lat1)."""
    q = scan_points(adsb_dir)
    if t_range is not None:
        q = q.filter(pl.col("t").is_between(t_range[0], t_range[1]))
    if bbox is not None:
        q = q.filter(pl.col("lon").is_between(bbox[0], bbox[2]) & pl.col("lat").is_between(bbox[1], bbox[3]))
    if icaos is not None:
        q = q.filter(pl.col("icao").is_in(list(icaos)))
    ac = pl.scan_parquet(str(Path(adsb_dir) / "aircraft.parquet")).select("icao", "type", "engine")
    q = q.join(ac, on="icao", how="left").with_columns(pl.col("engine").fill_null("?"))
    if exclude_engines:
        q = q.filter(~pl.col("engine").is_in(list(exclude_engines)))
    if max_alt_ft is not None:
        q = q.filter(pl.coalesce("alt_baro_ft", "alt_geom_ft") <= max_alt_ft)
    # total order: duplicate timestamps otherwise arrive in scan order, which varies run to run,
    # and split_tracks keeps the first of each duplicate
    return q.sort("icao", "t", "lat", "lon", "alt_baro_ft", "alt_geom_ft", "gs_kt", "track_deg",
                  nulls_last=True).collect()


def altitude_msl(df: pl.DataFrame, geoid: geo.Geoid | None, mode: str = "auto") -> np.ndarray:
    """Aircraft height above mean sea level (m).

    mode "baro": pressure altitude as-is (what most tools plot; off by the local QNH
    difference, typically tens of metres). "geom": GNSS ellipsoidal altitude minus geoid.
    "auto": GNSS where present; where only baro exists, baro + the airframe's own median
    (GNSS - baro) offset, else raw baro.
    """
    baro = df["alt_baro_ft"].fill_nan(None).to_numpy().astype(float) * FT
    hae = df["alt_geom_ft"].fill_nan(None).to_numpy().astype(float) * FT
    if mode == "baro" or geoid is None:
        out = np.where(np.isfinite(baro), baro, hae)
        return out
    N = geoid(df["lat"].to_numpy(), df["lon"].to_numpy())
    geom = hae - N
    if mode == "geom":
        return np.where(np.isfinite(geom), geom, baro)
    out = geom.copy()
    icao = df["icao"].to_numpy()
    miss = ~np.isfinite(out)
    if miss.any():
        diff = pl.DataFrame({"icao": icao, "d": geom - baro}).filter(pl.col("d").is_finite())
        off = dict(diff.group_by("icao").agg(pl.col("d").median()).iter_rows())
        corr = np.array([off.get(i, 0.0) for i in icao[miss]])
        out[miss] = baro[miss] + corr
    return out


def split_tracks(df: pl.DataFrame, h_msl: np.ndarray, max_gap_s: float = 90.0,
                 min_points: int = 3) -> list[Track]:
    """Split sorted points into continuous segments (gap > max_gap_s or new-leg flag)."""
    icao = df["icao"].to_numpy()
    t = df["t"].to_numpy()
    flags = df["flags"].to_numpy()
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    gs = df["gs_kt"].fill_nan(None).to_numpy().astype(float) * KT
    trk = df["track_deg"].fill_nan(None).to_numpy().astype(float)
    vr = df["vrate_fpm"].fill_nan(None).fill_null(0.0).to_numpy().astype(float) * FPM
    typ = df["type"].to_numpy() if "type" in df.columns else np.full(len(df), None)
    eng = df["engine"].to_numpy() if "engine" in df.columns else np.full(len(df), "?")
    cs = df["callsign"].to_numpy()
    n = len(t)
    brk = np.zeros(n, dtype=bool)
    if n:
        brk[0] = True
        brk[1:] = (icao[1:] != icao[:-1]) | (np.diff(t) > max_gap_s) | ((flags[1:] & 2) > 0)
    starts = np.flatnonzero(brk)
    ends = np.append(starts[1:], n)
    out = []
    for a, b in zip(starts, ends):
        ok = np.isfinite(h_msl[a:b]) & np.isfinite(lat[a:b])
        if ok.sum() < min_points:
            continue
        sl = np.arange(a, b)[ok]
        # drop duplicate timestamps
        keep = np.append(True, np.diff(t[sl]) > 0.05)
        sl = sl[keep]
        if len(sl) < min_points:
            continue
        # stable id: airframe + segment start time (independent of the loaded window)
        out.append(Track(f"{icao[a]}:{int(t[sl[0]])}", icao[a], typ[a], eng[a],
                         next((c for c in cs[sl] if c), None),
                         t[sl], lat[sl], lon[sl], h_msl[sl], gs[sl], trk[sl], vr[sl]))
    return out


def _velocities(tr: Track):
    """ENU velocity per point; fall back to finite differences where gs/track missing."""
    e = tr.gs * np.sin(np.radians(tr.trk))
    n = tr.gs * np.cos(np.radians(tr.trk))
    u = tr.vr.copy()
    bad = ~(np.isfinite(e) & np.isfinite(n))
    if bad.any():
        x, y = geo.to_grid(tr.lon, tr.lat)
        ve = np.gradient(x, tr.t)
        vn = np.gradient(y, tr.t)
        e[bad], n[bad] = ve[bad], vn[bad]
    return e, n, u


def resample(tr: Track, dt: float = 1.0, t_range=None) -> Sampled | None:
    """Cubic Hermite resampling in ECEF on a grid aligned to multiples of ``dt``."""
    lo, hi = tr.t[0], tr.t[-1]
    if t_range is not None:
        lo, hi = max(lo, t_range[0]), min(hi, t_range[1])
    t0 = np.ceil(lo / dt) * dt
    if hi - t0 < dt:
        return None
    tq = np.arange(t0, hi + 1e-9, dt)
    P = geo.lla_to_ecef(tr.lat, tr.lon, tr.h)
    e, n, u = _velocities(tr)
    V = geo.enu_to_ecef_vec(tr.lat, tr.lon, e, n, u)
    k = np.clip(np.searchsorted(tr.t, tq, side="right") - 1, 0, len(tr.t) - 2)
    h = tr.t[k + 1] - tr.t[k]
    s = ((tq - tr.t[k]) / h)[:, None]
    hh = h[:, None]
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    # long gaps: Hermite with stale velocities can overshoot; blend to linear
    lin = (h > 30.0)[:, None]
    Pq = np.where(lin, (1 - s) * P[k] + s * P[k + 1],
                  h00 * P[k] + h10 * hh * V[k] + h01 * P[k + 1] + h11 * hh * V[k + 1])
    d00 = 6 * s**2 - 6 * s
    d10 = 3 * s**2 - 4 * s + 1
    d01 = -6 * s**2 + 6 * s
    d11 = 3 * s**2 - 2 * s
    Vh = (d00 * P[k] + d10 * hh * V[k] + d01 * P[k + 1] + d11 * hh * V[k + 1]) / hh
    Vq = np.where(lin, (1 - s) * V[k] + s * V[k + 1], Vh)
    lat = np.interp(tq, tr.t, tr.lat)
    lon = np.interp(tq, tr.t, tr.lon)
    hq = np.interp(tq, tr.t, tr.h)
    # horizontal speed from the velocity vector, projected on the local vertical
    up = geo.lla_to_ecef(lat, lon, 1.0) - geo.lla_to_ecef(lat, lon, 0.0)
    vz = np.einsum("ij,ij->i", Vq, up)
    gsq = np.sqrt(np.maximum(np.einsum("ij,ij->i", Vq, Vq) - vz * vz, 0.0))
    return Sampled(tr.fid, float(t0), dt, Pq, Vq, lat, lon, hq, gsq)
