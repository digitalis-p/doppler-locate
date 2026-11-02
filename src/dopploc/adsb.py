"""ADS-B history: fetch adsb.lol daily readsb dumps and ingest them to parquet.

adsb.lol publishes one GitHub release per day (repo ``adsblol/globe_history_<year>``,
tag ``v<YYYY.MM.DD>-planes-readsb-prod-0``) holding a split tar of readsb
``traces/xx/trace_full_<icao>.json`` files (gzip JSON). Each trace row is::

    [sec_after_timestamp, lat, lon, alt_baro_ft | "ground" | null, gs_kt, track_deg,
     flags, baro_rate_fpm, details | null, source, alt_geom_ft, geom_rate_fpm, ias_kt, roll]

flags: 1 stale position, 2 new leg, 4 vertical rate is geometric, 8 altitude is geometric.

The ingest keeps airborne points inside a bounding box (CONUS by default) and writes
``points-*.parquet`` (one row per position) plus ``aircraft.parquet`` (one row per airframe,
joined with the ICAO type designator table so propeller and jet types can be told apart).
"""

from __future__ import annotations

import gzip
import io
import json
import multiprocessing as mp
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

CONUS_BBOX = (-126.0, 23.5, -65.5, 50.5)  # lon_min, lat_min, lon_max, lat_max
TYPES_URL = "https://raw.githubusercontent.com/wiedehopf/tar1090-db/master/db/icao_aircraft_types2.js"

FT = 0.3048
KT = 0.514444
FPM = 0.00508

POINT_SCHEMA = pa.schema([
    ("icao", pa.string()),
    ("t", pa.float64()),          # unix seconds UTC
    ("lat", pa.float64()),
    ("lon", pa.float64()),
    ("alt_baro_ft", pa.float32()),  # NaN when not reported
    ("alt_geom_ft", pa.float32()),  # GNSS altitude (WGS84 ellipsoid per readsb docs); NaN when absent
    ("gs_kt", pa.float32()),
    ("track_deg", pa.float32()),
    ("vrate_fpm", pa.float32()),    # geometric rate preferred, else baro rate
    ("flags", pa.uint8()),
    ("src", pa.string()),
    ("callsign", pa.string()),
])


# --------------------------------------------------------------------------- fetch

def release_tag(day: str) -> tuple[str, str]:
    """('adsblol/globe_history_2026', 'v2026.09.08-planes-readsb-prod-0') for '2026-09-08'."""
    y, m, d = day.split("-")
    return f"adsblol/globe_history_{y}", f"v{y}.{m}.{d}-planes-readsb-prod-0"


def fetch_day(day: str, out_dir: Path) -> list[Path]:
    """Download the split tar for one UTC day with the GitHub CLI (``gh``)."""
    repo, tag = release_tag(day)
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gh", "release", "download", tag, "-R", repo, "-D", str(out_dir),
                    "--skip-existing", "-p", "*.tar.*"], check=True)
    return sorted(out_dir.glob(f"{tag}.tar.*"))


def fetch_types(path: Path) -> Path:
    """Download the tar1090-db ICAO type table (gzip JSON: type -> [name, 'L1P', wtc])."""
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(TYPES_URL, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def load_types(path: Path) -> dict[str, tuple[str, str, str]]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    out = {}
    for k, v in json.loads(raw).items():
        if isinstance(v, list) and len(v) >= 3 and v[1]:
            out[k] = (v[0], v[1], v[2])
    return out


def engine_class(icao_desc: str | None) -> str:
    """Engine letter of an ICAO description code ('L1P' -> 'P'); '?' when unknown.

    P piston, T turboprop/turboshaft, J jet, E electric, R rocket.
    """
    if not icao_desc or len(icao_desc) < 3:
        return "?"
    return icao_desc[2]


# --------------------------------------------------------------------------- ingest

class _Cat(io.RawIOBase):
    """Read several files back to back as one stream (split tar parts)."""

    def __init__(self, paths):
        self.files = [open(p, "rb") for p in paths]
        self.i = 0

    def readable(self):
        return True

    def readinto(self, b):
        while self.i < len(self.files):
            n = self.files[self.i].readinto(b)
            if n:
                return n
            self.files[self.i].close()
            self.i += 1
        return 0


def _iter_traces(parts):
    import tarfile
    stream = io.BufferedReader(_Cat(parts), 1 << 22)
    with tarfile.open(fileobj=stream, mode="r|") as tf:
        for m in tf:
            if m.isfile() and "/trace_full_" in m.name:
                yield tf.extractfile(m).read()


def _f(x):
    return np.nan if x is None else float(x)


def _parse_trace(args):
    raw, bbox = args
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    d = json.loads(raw)
    icao = d.get("icao", "")
    t0 = float(d.get("timestamp", 0.0))
    lon0, lat0, lon1, lat1 = bbox
    cols = {k: [] for k in ("t", "lat", "lon", "alt_baro_ft", "alt_geom_ft", "gs_kt", "track_deg",
                            "vrate_fpm", "flags", "src", "callsign")}
    callsign = None
    cats = {}
    for row in d.get("trace", ()):
        det = row[8] if len(row) > 8 else None
        if det:
            fl = det.get("flight")
            if fl:
                callsign = fl.strip()
            c = det.get("category")
            if c:
                cats[c] = cats.get(c, 0) + 1
        lat, lon, alt = row[1], row[2], row[3]
        if lat is None or lon is None or alt == "ground":
            continue
        if not (lat0 <= lat <= lat1 and lon0 <= lon <= lon1):
            continue
        flags = int(row[6] or 0)
        alt_geom = row[10] if len(row) > 10 else None
        if flags & 8:  # altitude field is geometric, not barometric
            alt_geom, alt = alt, None
        rate = row[11] if len(row) > 11 and row[11] is not None else row[7]
        cols["t"].append(t0 + row[0])
        cols["lat"].append(lat)
        cols["lon"].append(lon)
        cols["alt_baro_ft"].append(_f(alt))
        cols["alt_geom_ft"].append(_f(alt_geom))
        cols["gs_kt"].append(_f(row[4]))
        cols["track_deg"].append(_f(row[5]))
        cols["vrate_fpm"].append(_f(rate))
        cols["flags"].append(flags)
        cols["src"].append(row[9] if len(row) > 9 else None)
        cols["callsign"].append(callsign)
    n = len(cols["t"])
    tbl = None
    if n:
        arrays = [pa.array([icao] * n, type=pa.string())]
        for f in POINT_SCHEMA:
            if f.name != "icao":
                arrays.append(pa.array(cols[f.name], type=f.type, from_pandas=True))
        tbl = pa.Table.from_arrays(arrays, schema=POINT_SCHEMA)
    meta = {
        "icao": icao, "reg": d.get("r"), "type": d.get("t"), "desc": d.get("desc"),
        "owner": d.get("ownOp"), "year": d.get("year"), "db_flags": d.get("dbFlags"),
        "category": max(cats, key=cats.get) if cats else None,
        "n_points": len(cols["t"]),
    }
    return tbl, meta


def ingest(parts: list[Path], out_dir: Path, types_path: Path, bbox=CONUS_BBOX,
           workers: int | None = None, rows_per_file: int = 20_000_000) -> dict:
    """Stream a readsb history tar into parquet. Returns summary counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("points-*.parquet"):
        old.unlink()
    types = load_types(types_path)
    workers = workers or max(1, mp.cpu_count() - 2)

    pending: list[pa.Table] = []
    npending = 0
    file_no = 0
    metas = []
    n_traces = 0
    n_points = 0

    def flush():
        nonlocal pending, npending, file_no
        if not npending:
            return
        tbl = pa.concat_tables(pending)
        pq.write_table(tbl, out_dir / f"points-{file_no:03d}.parquet", compression="zstd")
        file_no += 1
        pending, npending = [], 0

    with mp.Pool(workers) as pool:
        it = pool.imap_unordered(_parse_trace, ((raw, bbox) for raw in _iter_traces(parts)), chunksize=64)
        for tbl, meta in it:
            n_traces += 1
            if tbl is None:
                continue
            metas.append(meta)
            pending.append(tbl)
            npending += tbl.num_rows
            n_points += tbl.num_rows
            if npending >= rows_per_file:
                flush()
    flush()

    for m in metas:
        tdesc = types.get(m["type"] or "")
        m["type_name"] = tdesc[0] if tdesc else None
        m["icao_class"] = tdesc[1] if tdesc else None
        m["wtc"] = tdesc[2] if tdesc else None
        m["engine"] = engine_class(tdesc[1] if tdesc else None)
    pq.write_table(pa.Table.from_pylist(metas), out_dir / "aircraft.parquet", compression="zstd")
    return {"traces": n_traces, "aircraft_in_bbox": len(metas), "points": n_points, "files": file_no}
