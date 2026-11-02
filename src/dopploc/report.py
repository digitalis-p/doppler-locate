"""Outputs for humans and GIS: ranked tables, GeoJSON, score scatter, curve overlays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from . import audio, geo, physics, score
from .config import Case
from .dem import DEM


def write_geojson(df: pl.DataFrame, path: Path, props=None):
    props = props or [c for c in df.columns if c not in ("lon", "lat")]
    feats = []
    for r in df.iter_rows(named=True):
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                      "properties": {k: (None if isinstance(r[k], float) and not np.isfinite(r[k]) else r[k])
                                     for k in props}})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def scatter(res: pl.DataFrame, best: pl.DataFrame, path: Path, truth=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    two = "S2" in res.columns and res["S2"].is_not_nan().any()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, df, title in ((axes[0], res, "all scored points"), (axes[1], best, "best point per flight pair")):
        if two:
            ax.scatter(df["S1"], df["S2"], s=4, alpha=0.5)
            top = df.head(5)
            ax.scatter(top["S1"], top["S2"], s=30, c="red")
            for i, r in enumerate(top.iter_rows(named=True)):
                ax.annotate(f"#{i + 1} {r['score']:.2f}", (r["S1"], r["S2"]), fontsize=8)
            ax.set_xlabel("event 1 score")
            ax.set_ylabel("event 2 score")
        else:
            ax.hist(df["S1"].drop_nans(), bins=100)
            ax.set_xlabel("score")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def overlay_data(case: Case, row: dict) -> list[dict]:
    """Per event: the whitened spectrogram around it and the candidate's modelled harmonic
    curves at the scored fit: [{"spec", "curves": [(t, f)], "title"}]."""
    ev = case.events
    two = "f2" in row and row.get("f2") is not None
    sc = case.score
    dem = DEM(case.cache / "dem", res=sc.dem_res)
    elev = float(dem.elevation(np.array([row["lat"]]), np.array([row["lon"]]))[0])
    obs = geo.lla_to_ecef(row["lat"], row["lon"], elev + score.OBS_HEIGHT_M)
    fids = [row["f1"]] + ([row["f2"]] if two else [])
    flights = score.load_flights(case, fids, row["t1"] - 900, (row["t2"] if two else row["t1"]) + 900,
                                 log=lambda *_: None)
    out = []
    for k in range(2 if two else 1):
        e = ev[k]
        fid = fids[k]
        sm = flights.get(fid)
        hmax = max(e.harmonics)
        fband = e.fband or (max(15.0, 0.6 * e.f0 * min(e.harmonics)), 1.6 * e.f0 * hmax)
        sp = audio.brightness(case.audio, e.audio_cpa - e.half_width_s - 5, e.audio_cpa + e.half_width_s + 5,
                              win_s=sc.win_s, hop_s=sc.hop_s, fmin=fband[0], fmax=fband[1])
        curves = []
        if sm is not None:
            tdel = row["t1m"] if k == 0 else row["t2m"]
            # audio time of u=0: event 1 at audio_cpa + delta; event 2 locked by geometry
            a0 = ev[0].audio_cpa + row["delta"] + (0 if k == 0 else row["t2m"] - row["t1m"])
            ta = sp.t
            atm = _row_atm(case, row, k + 1)
            r, _, ok = physics.track_doppler_ratio(tdel + (ta - a0), obs, sm.t0, sm.dt, sm.P, sm.V, atm)
            f0 = row["f01"] if k == 0 else row["f02"]
            curves = [(ta[ok], h * f0 * r[ok]) for h in e.harmonics]
        out.append({"spec": sp, "curves": curves,
                    "title": f"event {k + 1} · {fid} · S={row['S1' if k == 0 else 'S2']:.2f}"})
    return out


def overlay(case: Case, row: dict, path: Path, rank: int):
    """Spectrograms of both events with the candidate's modelled curves at the scored fit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = overlay_data(case, row)
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 6), squeeze=False)
    for k, d in enumerate(data):
        sp = d["spec"]
        ax = axes[0, k]
        ax.pcolormesh(sp.t, sp.f, sp.S, vmin=0, vmax=5, cmap="magma", shading="auto")
        for t, f in d["curves"]:
            ax.plot(t, f, color="cyan", lw=0.8, alpha=0.6)
        ax.set_title(f"#{rank} {d['title']}")
        ax.set_xlabel("audio s")
        ax.set_ylabel("Hz")
    fig.suptitle(f"{row['lat']:.5f}, {row['lon']:.5f}  total {row['score']:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _row_atm(case: Case, row: dict, k: int) -> np.ndarray:
    """Atmosphere vector for one scored row (columns written by score.case_atmosphere)."""
    lat, lon = row["lat"], row["lon"]
    if row.get(f"c_e{k}") is not None:  # GFS: wind at aircraft altitude; path wind not stored -> half of it
        we, wn, ce = row[f"wind_e{k}"], row[f"wind_n{k}"], row[f"c_e{k}"]
        ws = geo.enu_to_ecef_vec(lat, lon, we, wn, 0.0)
        return np.r_[ce, ws, ce, 0.5 * ws]
    if row.get("wind_e") is not None:
        w = geo.enu_to_ecef_vec(lat, lon, row["wind_e"], row["wind_n"], 0.0)
        return np.r_[case.c, w, case.c, w]
    return physics.still_air(case.c)


def write(case: Case, res: pl.DataFrame, top: int = 5, log=print, truth: dict | None = None):
    out = case.out
    out.mkdir(parents=True, exist_ok=True)
    res.write_parquet(out / "scored.parquet")
    best = score.best_per_pair(res)
    best.write_csv(out / "best_per_pair.csv")
    keep = [c for c in ("score", "S1", "S2", "f1", "f2", "t1m", "t2m", "d1m", "d2m", "delta", "f01", "f02",
                        "offset_err_model", "elev") if c in best.columns]
    write_geojson(best.head(500), out / "best_per_pair.geojson", keep)
    write_geojson(res.head(20000), out / "scored_top.geojson", keep)
    scatter(res, best, out / "scatter.png", truth)
    for i, r in enumerate(best.head(top).iter_rows(named=True)):
        overlay(case, r, out / f"rank{i + 1}.png", i + 1)
    lines = [f"# {case.name}", "", "| rank | score | S1 | S2 | lat | lon | event-1 flight | event-2 flight |",
             "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(best.head(20).iter_rows(named=True)):
        lines.append(f"| {i + 1} | {r['score']:.3f} | {r['S1']:.3f} | {r.get('S2', float('nan')):.3f} | "
                     f"{r['lat']:.5f} | {r['lon']:.5f} | {r['f1']} | {r.get('f2', '')} |")
    if truth:
        d = haversine(best["lat"].to_numpy(), best["lon"].to_numpy(), truth["lat"], truth["lon"])
        within = np.flatnonzero(d < 300)
        lines += ["", f"truth {truth['lat']:.5f},{truth['lon']:.5f}: best pair within 300 m ranks "
                  f"{int(within[0]) + 1 if len(within) else 'n/a'}; top-1 is {d[0]:.0f} m away"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    return best


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))
