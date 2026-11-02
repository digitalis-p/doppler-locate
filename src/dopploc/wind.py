"""Surface wind from ASOS/AWOS reports (Iowa Environmental Mesonet), as the video did.

The Doppler shift depends on the aircraft's velocity relative to the air, and sound travels
at c relative to the air, so a moving air mass shifts and skews the curve. With wind w
(m/s, the direction the air moves toward) and d the observer->aircraft unit vector:

    f = f0 (c - w.d) / (c + v.d - w.d),   travel time = R / (c - w.d)

Surface wind stands in for the wind along the sound path (a simplification: winds aloft are
usually stronger).
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import requests

from .adsb import KT

IEM = "https://mesonet.agron.iastate.edu"
CONUS_STATES = ("AL AZ AR CA CO CT DE FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY "
                "NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY").split()


def _stations(state: str) -> list[str]:
    r = requests.get(f"{IEM}/geojson/network/{state}_ASOS.geojson", timeout=60)
    r.raise_for_status()
    return [f["properties"]["sid"] for f in r.json()["features"]]


def fetch_day(day: str, cache_dir: Path, margin_h: int = 3, log=print) -> Path:
    """All CONUS ASOS wind reports for the UTC day (+- margin), cached as CSV."""
    out = Path(cache_dir) / "wind" / f"asos_{day}.csv"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    d0 = datetime.fromisoformat(day).replace(tzinfo=timezone.utc) - timedelta(hours=margin_h)
    d1 = d0 + timedelta(hours=24 + 2 * margin_h)
    frames = []
    for st in CONUS_STATES:
        sids = _stations(st)
        for i in range(0, len(sids), 80):
            q = [("station", s) for s in sids[i:i + 80]]
            q += [("data", "sknt"), ("data", "drct"), ("tz", "Etc/UTC"), ("format", "onlycomma"),
                  ("latlon", "yes"), ("missing", "empty"), ("trace", "empty"),
                  ("report_type", "3"), ("report_type", "4"),
                  ("year1", d0.year), ("month1", d0.month), ("day1", d0.day), ("hour1", d0.hour),
                  ("year2", d1.year), ("month2", d1.month), ("day2", d1.day), ("hour2", d1.hour)]
            r = requests.get(f"{IEM}/cgi-bin/request/asos.py", params=q, timeout=300)
            r.raise_for_status()
            if r.text.count("\n") > 1:
                frames.append(pl.read_csv(io.StringIO(r.text), infer_schema_length=0))
        log(f"  wind: {st} done")
    df = pl.concat(frames)
    df.write_csv(out)
    return out


class Wind:
    """Nearest-station, nearest-time surface wind (east, north) in m/s toward which air moves."""

    def __init__(self, csv: Path, max_km: float = 120.0, max_dt_s: float = 2 * 3600):
        from scipy.spatial import cKDTree
        from . import geo
        df = pl.read_csv(csv, infer_schema_length=0).with_columns(
            pl.col("lon").cast(pl.Float64), pl.col("lat").cast(pl.Float64),
            pl.col("sknt").cast(pl.Float64, strict=False), pl.col("drct").cast(pl.Float64, strict=False),
            pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC").dt.epoch("s").alias("t"))
        df = df.filter(pl.col("sknt").is_not_null())
        df = df.with_columns(pl.when(pl.col("sknt") == 0).then(0.0).otherwise(pl.col("drct")).alias("drct"))
        df = df.filter(pl.col("drct").is_not_null()).sort("station", "t")
        spd = df["sknt"].to_numpy() * KT
        toward = np.radians(df["drct"].to_numpy() + 180.0)
        self.e = spd * np.sin(toward)
        self.n = spd * np.cos(toward)
        st = df["station"].to_numpy()
        self.t = df["t"].to_numpy().astype(float)
        uniq, first = np.unique(st, return_index=True)
        self.sid = np.searchsorted(uniq, st)
        x, y = geo.to_grid(df["lon"].to_numpy()[first], df["lat"].to_numpy()[first])
        self.tree = cKDTree(np.c_[x, y])
        self.key = self.sid * 1e10 + self.t
        self.max_m = max_km * 1000.0
        self.max_dt = max_dt_s
        self.names = uniq

    def at(self, lat, lon, t, k: int = 3):
        """Inverse-distance blend of the k nearest stations' reports nearest in time."""
        from . import geo
        x, y = geo.to_grid(np.asarray(lon, float), np.asarray(lat, float))
        dist, idx = self.tree.query(np.c_[x, y], k=k, distance_upper_bound=self.max_m)
        t = np.broadcast_to(np.asarray(t, float), np.shape(x))
        we = np.zeros(len(x))
        wn = np.zeros(len(x))
        wsum = np.zeros(len(x))
        nst = len(self.names)
        for j in range(k):
            ok = idx[:, j] < nst
            s = np.where(ok, idx[:, j], 0)
            q = s * 1e10 + t
            p = np.clip(np.searchsorted(self.key, q), 1, len(self.key) - 1)
            cand = np.stack([p - 1, p], 1)
            dtc = np.abs(self.t[cand] - t[:, None])
            same = self.sid[cand] == s[:, None]
            dtc = np.where(same, dtc, np.inf)
            b = cand[np.arange(len(p)), np.argmin(dtc, 1)]
            ok &= np.min(dtc, 1) <= self.max_dt
            w = np.where(ok, 1.0 / np.maximum(dist[:, j], 1000.0), 0.0)
            we += w * self.e[b]
            wn += w * self.n[b]
            wsum += w
        good = wsum > 0
        we[good] /= wsum[good]
        wn[good] /= wsum[good]
        return we, wn
