"""Atmosphere for the forward model: sound speed and wind profiles from NOAA GFS analyses.

Why: the received pitch depends on the aircraft velocity relative to the *air* and on the
speed of sound along the path. For an aircraft at 5 km, winds of 15+ m/s and a sound speed
~15 m/s below the surface value change the Doppler curve by far more than the residuals
that separate candidates (validated on the video's recording: the truth's event-1 residual
drops from 0.55 Hz in still air to 0.21 Hz with this model).

Model used per (observer, aircraft pass):
  * Doppler factor  f/f0 = (c_e - w_s.d) / (c_e + v.d - w_s.d)
      c_e: mean sound speed between the ground and the aircraft altitude,
      w_s: wind at the aircraft altitude, d: unit vector observer -> aircraft.
  * travel time     R / (c_e - w_m.d), w_m: mean wind along the vertical path.

Data: GFS 0.25 deg analyses (f000, 6-hourly) from the public AWS bucket, fetched with HTTP
range requests for only the needed GRIB2 messages (~40 MB per cycle), cropped to CONUS and
cached as .npz. Nothing about the search area is sent anywhere.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from numba import njit, prange

GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d}/{h:02d}/atmos/gfs.t{h:02d}z.pgrb2.0p25.f000"
LEVELS = (1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300)
CROP = (20.0, 55.0, 230.0, 300.0)  # lat0, lat1, lon0, lon1 (0-360)


def speed_of_sound(t_k):
    return 331.3 * np.sqrt(np.asarray(t_k) / 273.15)


def _cycle_path(cache: Path, cyc: datetime) -> Path:
    return Path(cache) / "gfs" / f"gfs_{cyc:%Y%m%d%H}.npz"


def fetch_cycle(cyc: datetime, cache: Path) -> Path:
    import eccodes
    out = _cycle_path(cache, cyc)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    url = GFS.format(d=f"{cyc:%Y%m%d}", h=cyc.hour)
    idx = requests.get(url + ".idx", timeout=120)
    idx.raise_for_status()
    lines = [l.split(":") for l in idx.text.strip().splitlines()]
    starts = [int(l[1]) for l in lines]
    want = {}
    for i, l in enumerate(lines):
        var, lev = l[3], l[4]
        key = None
        mb = float(lev.split()[0]) if lev.endswith(" mb") else None
        if var in ("HGT", "TMP", "UGRD", "VGRD") and mb is not None and mb in LEVELS:
            key = (var, int(mb))
        elif (var, lev) in (("TMP", "2 m above ground"), ("UGRD", "10 m above ground"),
                            ("VGRD", "10 m above ground"), ("HGT", "surface")):
            key = (var, 0)
        if key and key not in want:
            end = starts[i + 1] - 1 if i + 1 < len(starts) else None
            want[key] = (starts[i], end)

    def get(item):
        key, (a, b) = item
        rng = f"bytes={a}-" + ("" if b is None else str(b))
        for attempt in range(5):
            try:
                r = requests.get(url, headers={"Range": rng}, timeout=300)
                r.raise_for_status()
                return key, r.content
            except requests.RequestException:
                if attempt == 4:
                    raise
        return key, b""

    with ThreadPoolExecutor(8) as ex:
        msgs = dict(ex.map(get, want.items()))
    fields = {}
    lat = lon = None
    for key, buf in msgs.items():
        gid = eccodes.codes_new_from_message(buf)
        try:
            ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
            vals = eccodes.codes_get_values(gid).reshape(nj, ni)
            if lat is None:
                la0 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
                la1 = eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees")
                lo0 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
                lo1 = eccodes.codes_get(gid, "longitudeOfLastGridPointInDegrees")
                lat = np.linspace(la0, la1, nj)
                lon = np.linspace(lo0, lo1, ni)
        finally:
            eccodes.codes_release(gid)
        fields[key] = vals
    yi = np.flatnonzero((lat >= CROP[0]) & (lat <= CROP[1]))
    xi = np.flatnonzero((lon >= CROP[2]) & (lon <= CROP[3]))
    crop = lambda a: a[yi[0]:yi[-1] + 1, xi[0]:xi[-1] + 1].astype(np.float32)
    lv = list(LEVELS)
    np.savez_compressed(
        out, lat=lat[yi], lon=lon[xi], levels=np.array(lv),
        hgt=np.stack([crop(fields[("HGT", p)]) for p in lv]), tmp=np.stack([crop(fields[("TMP", p)]) for p in lv]),
        u=np.stack([crop(fields[("UGRD", p)]) for p in lv]), v=np.stack([crop(fields[("VGRD", p)]) for p in lv]),
        zs=crop(fields[("HGT", 0)]), ts=crop(fields[("TMP", 0)]), us=crop(fields[("UGRD", 0)]),
        vs=crop(fields[("VGRD", 0)]))
    return out


def cycles_for(day: str):
    d0 = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    return [d0 + timedelta(hours=h) for h in range(-6, 31, 6)]


class Atmosphere:
    """GFS profiles interpolated bilinearly in space and linearly in time."""

    def __init__(self, day: str, cache: Path, log=print):
        self.cyc = []
        self.data = []
        for c in cycles_for(day):
            try:
                p = fetch_cycle(c, cache)
            except Exception as e:  # missing cycle: interpolate across the gap
                log(f"  GFS {c:%Y-%m-%d %H}Z unavailable ({e})")
                continue
            self.cyc.append(c.timestamp())
            self.data.append(dict(np.load(p)))
        if not self.data:
            raise RuntimeError("no GFS cycles available")
        self.cyc = np.array(self.cyc)
        d = self.data[0]
        self.lat, self.lon = d["lat"], d["lon"]

    def _sample(self, k, name, lat, lon):
        a = self.data[k][name]
        y = (lat - self.lat[0]) / (self.lat[1] - self.lat[0])
        x = ((lon % 360.0) - self.lon[0]) / (self.lon[1] - self.lon[0])
        y0 = np.clip(np.floor(y).astype(int), 0, len(self.lat) - 2)
        x0 = np.clip(np.floor(x).astype(int), 0, len(self.lon) - 2)
        wy = np.clip(y - y0, 0, 1)
        wx = np.clip(x - x0, 0, 1)
        if a.ndim == 2:
            a = a[None]
        out = ((1 - wy) * (1 - wx) * a[:, y0, x0] + (1 - wy) * wx * a[:, y0, x0 + 1]
               + wy * (1 - wx) * a[:, y0 + 1, x0] + wy * wx * a[:, y0 + 1, x0 + 1])
        return out.T  # (n, L)

    def profiles(self, lat, lon, t):
        """(z, T, u, v) arrays (n, L+1): surface first, then pressure levels; z in m MSL."""
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        t = np.asarray(t, float)
        k = np.clip(np.searchsorted(self.cyc, t) - 1, 0, max(0, len(self.cyc) - 2))
        out = []
        for name_s, name in (("zs", "hgt"), ("ts", "tmp"), ("us", "u"), ("vs", "v")):
            res = np.empty((len(lat), len(LEVELS) + 1))
            for kk in np.unique(k):
                m = k == kk
                k1 = min(kk + 1, len(self.cyc) - 1)
                w = 0.0 if k1 == kk else np.clip((t[m] - self.cyc[kk]) / (self.cyc[k1] - self.cyc[kk]), 0, 1)
                w = np.asarray(w)[:, None] if np.ndim(w) else w
                a = np.concatenate([self._sample(kk, name_s, lat[m], lon[m]), self._sample(kk, name, lat[m], lon[m])], 1)
                b = np.concatenate([self._sample(k1, name_s, lat[m], lon[m]), self._sample(k1, name, lat[m], lon[m])], 1)
                res[m] = (1 - w) * a + w * b
            out.append(res)
        z = out[0]
        z[:, 0] += 2.0  # 2 m / 10 m surface values sit just above the model ground
        return z, out[1], out[2], out[3]

    def effective(self, lat, lon, t, h_obs, h_ac):
        """Per row: c_e (path-mean sound speed), (ue, ve) wind at aircraft altitude,
        (um, vm) path-mean wind. All m/s; winds east / north components (toward)."""
        z, T, u, v = self.profiles(lat, lon, t)
        c = speed_of_sound(T)
        h0 = np.asarray(h_obs, float)
        h1 = np.asarray(h_ac, float)
        ce = _path_mean(z, c, h0, h1)
        um = _path_mean(z, u, h0, h1)
        vm = _path_mean(z, v, h0, h1)
        ua = _at(z, u, h1)
        va = _at(z, v, h1)
        return ce, ua, va, um, vm


@njit(cache=True)
def _interp_sorted(zr, vr, h):
    """Linear interpolation in a profile; levels below the surface (z < z[0]) are skipped."""
    n = zr.shape[0]
    if h <= zr[0]:
        return vr[0]
    prev = 0
    for k in range(1, n):
        if zr[k] <= zr[prev]:
            continue
        if h <= zr[k]:
            w = (h - zr[prev]) / (zr[k] - zr[prev])
            return vr[prev] + w * (vr[k] - vr[prev])
        prev = k
    return vr[prev]


@njit(parallel=True, cache=True)
def _at(z, val, h):
    n = z.shape[0]
    out = np.empty(n)
    for i in prange(n):
        out[i] = _interp_sorted(z[i], val[i], h[i])
    return out


@njit(parallel=True, cache=True)
def _path_mean(z, val, h0, h1, steps=40):
    n = z.shape[0]
    out = np.empty(n)
    for i in prange(n):
        a, b = h0[i], h1[i]
        if b < a:
            a, b = b, a
        acc = 0.0
        for s in range(steps):
            acc += _interp_sorted(z[i], val[i], a + (s + 0.5) * (b - a) / steps)
        out[i] = acc / steps
    return out
