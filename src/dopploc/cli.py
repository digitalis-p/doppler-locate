"""Geolocate a recording from propeller-aircraft Doppler curves matched to ADS-B tracks.

Run "dopploc <command> -h" for the options of each command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _p(s):
    return Path(s).expanduser()


def cmd_fetch_adsb(a):
    from . import adsb
    out = a.out or Path("data/adsb/raw") / a.day
    for p in adsb.fetch_day(a.day, out):
        print(p)


def cmd_fetch_types(a):
    from . import adsb
    print(adsb.fetch_types(a.out))


def cmd_ingest(a):
    from . import adsb
    raw = a.raw or Path("data/adsb/raw") / a.day
    parts = sorted(raw.glob("*.tar.*"))
    if not parts:
        sys.exit(f"no tar parts in {raw}")
    bbox = tuple(a.bbox) if a.bbox else adsb.CONUS_BBOX
    res = adsb.ingest(parts, a.out or Path("data/adsb") / a.day, a.types, bbox=bbox, workers=a.workers)
    print(json.dumps(res))


def cmd_spec(a):
    from . import audio
    x = audio.load_audio(a.audio, sr=a.sr, start=a.t0, duration=a.t1 - a.t0)
    sp = audio.spectrogram(x, a.sr, win_s=a.win, hop_s=a.hop or a.win / 8, fmin=a.fmin, fmax=a.fmax, t_offset=a.t0)
    audio.render(sp, a.out, title=f"{a.audio.name} {a.t0:.0f}-{a.t1:.0f}s win {a.win}s")
    print(a.out)


def cmd_trace(a):
    from . import audio, measure
    sp = audio.brightness(a.audio, a.t0, a.t1, sr=a.sr, win_s=a.win, hop_s=a.hop, fmin=a.fmin, fmax=a.fmax)
    t, f, z = measure.trace_ridge(sp, a.f_start, a.f_end, max_slope_hz_s=a.max_slope, min_z=a.min_z)
    np.savetxt(a.out, np.c_[t, f, np.full(len(t), a.harmonic), z], delimiter=",", fmt="%.4f",
               header="t,f,harmonic,z")
    print(f"{len(t)} points -> {a.out}")
    if a.png:
        audio.render(sp, a.png, whiten=False, overlays=[(t, f, {"color": "cyan", "lw": 0.8})],
                     title=f"trace h={a.harmonic}")


def cmd_fit(a):
    from . import audio, measure
    tr = np.loadtxt(a.trace, delimiter=",", ndmin=2)
    t, f = tr[:, 0], tr[:, 1]
    h = int(a.harmonic or (tr[0, 2] if tr.shape[1] > 2 else 1))
    sel = np.ones(len(t), bool)
    if a.t0 is not None:
        sel &= t >= a.t0
    if a.t1 is not None:
        sel &= t <= a.t1
    t, f = t[sel], f[sel]
    fit, rng, inl = measure.fit_trace(t, f, h, c=a.c, rel=a.rel, abs_hz=a.abs_hz)
    print(f"{inl.sum()}/{len(t)} trace points within the inlier band")
    t, f = t[inl], f[inl]
    print(f"fit: f0={fit.f0:.3f} Hz  v={fit.v:.1f} m/s ({fit.v / 0.514444:.0f} kt)  d={fit.d:.0f} m  "
          f"t_cpa={fit.t_cpa:.2f} s  rms={fit.rms:.2f} Hz")
    print(f"ranges (rms <= {rng['threshold_hz']:.2f} Hz): speed {rng['speed_ms']}, slant {rng['slant_m']}, "
          f"f0 {rng['f0']}, audio_cpa {rng['audio_cpa']}")
    print()
    print(measure.event_toml(a.name, fit, rng, a.harmonics or [h], half_width=a.half_width))
    if a.png and a.audio:
        hmax = max(a.harmonics or [h])
        x = audio.load_audio(a.audio, sr=4000, start=t[0] - 20, duration=t[-1] - t[0] + 40)
        sp = audio.spectrogram(x, 4000, win_s=1.0, hop_s=0.125, fmin=max(10, 0.5 * fit.f0),
                               fmax=1.5 * fit.f0 * hmax, t_offset=t[0] - 20)
        tt = sp.t
        ov = [(t, f, {"color": "white", "lw": 0, "marker": ".", "ms": 2})]
        for hh in (a.harmonics or [h]):
            ov.append((tt, hh * measure.physics.linear_flyby(tt, fit.f0, fit.v, fit.d, fit.t_cpa, a.c),
                       {"color": "cyan", "lw": 0.8}))
        audio.render(sp, a.png, overlays=ov, title=f"fit v={fit.v:.1f} d={fit.d:.0f} f0={fit.f0:.2f}")


def cmd_search(a):
    from . import config, search
    case = config.load(a.case)
    search.run(case)


def cmd_score(a):
    import polars as pl
    from . import config, report, score
    case = config.load(a.case)
    cand = pl.read_parquet(case.out / "candidates.parquet")
    if "heard_between" in cand.columns and not a.keep_noisy:
        cand = cand.filter(pl.col("heard_between") == 0)
    if a.limit:
        cand = cand.head(a.limit)
    res = score.score_candidates(case, cand)
    truth = json.loads(Path(a.truth).read_text()) if a.truth else None
    report.write(case, res, top=a.top, truth=truth)


def cmd_run(a):
    cmd_search(a)
    cmd_score(a)


def cmd_heatmap(a):
    import polars as pl
    from . import config, report, score
    case = config.load(a.case)
    best = pl.read_csv(case.out / "best_per_pair.csv")
    row = best.row(a.rank - 1, named=True)
    res = score.heatmap(case, row, half_m=a.half, step_m=a.step)
    out = case.out / f"heatmap_rank{a.rank}.parquet"
    res.write_parquet(out)
    report.write_geojson(res, case.out / f"heatmap_rank{a.rank}.geojson", ["score", "S1", "S2"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(res["lon"], res["lat"], c=res["score"], s=6, cmap="viridis")
    ax.plot(row["lon"], row["lat"], "r+", ms=14)
    fig.colorbar(sc, label="score")
    ax.set_title(f"rank {a.rank}: {row['f1']} / {row.get('f2', '')}")
    fig.savefig(case.out / f"heatmap_rank{a.rank}.png", dpi=110)
    top = res.row(0, named=True)
    print(f"heatmap peak {top['lat']:.6f},{top['lon']:.6f} score {top['score']:.3f} -> {out}")


def cmd_detect(a):
    from . import detect
    p = detect.Params(threshold=a.threshold, fmin=a.fmin, fmax=a.fmax, refine=not a.no_refine)
    src = a.source
    meta = {}
    if detect.URL.match(src):
        path, meta = detect.fetch(src, a.downloads)
        stem = path.parent.name
    else:
        path = _p(src)
        stem = path.stem
    start, dur = a.start or 0.0, a.duration
    out = a.out or Path("out/detect") / stem
    res = detect.detect(path, start, dur, p, meta=meta)
    print(detect.report(res))
    detect.save(res, out)


def cmd_synth(a):
    from . import config, synth
    case = config.load(a.case)
    synth.make(case, a.out, seed=a.seed, utc_hour=a.hour, snr_db=a.snr)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dopploc", description=__doc__)
    sub = ap.add_subparsers(required=True)

    p = sub.add_parser("fetch-adsb", help="download one UTC day of adsb.lol history (gh CLI)")
    p.add_argument("day")
    p.add_argument("--out", type=_p)
    p.set_defaults(fn=cmd_fetch_adsb)

    p = sub.add_parser("fetch-types", help="download the ICAO aircraft type table")
    p.add_argument("--out", type=_p, default=Path("data/ref/icao_aircraft_types2.js.gz"))
    p.set_defaults(fn=cmd_fetch_types)

    p = sub.add_parser("ingest", help="adsb.lol tar -> parquet (bbox-filtered airborne points)")
    p.add_argument("day")
    p.add_argument("--raw", type=_p)
    p.add_argument("--out", type=_p)
    p.add_argument("--types", type=_p, default=Path("data/ref/icao_aircraft_types2.js.gz"))
    p.add_argument("--bbox", type=float, nargs=4, metavar=("LON0", "LAT0", "LON1", "LAT1"))
    p.add_argument("--workers", type=int)
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("spec", help="render a zoomed spectrogram PNG")
    p.add_argument("audio", type=_p)
    p.add_argument("--t0", type=float, required=True)
    p.add_argument("--t1", type=float, required=True)
    p.add_argument("--fmin", type=float, default=30)
    p.add_argument("--fmax", type=float, default=600)
    p.add_argument("--win", type=float, default=1.0)
    p.add_argument("--hop", type=float)
    p.add_argument("--sr", type=int, default=4000)
    p.add_argument("--out", type=_p, default=Path("spec.png"))
    p.set_defaults(fn=cmd_spec)

    p = sub.add_parser("trace", help="follow a Doppler ridge and save it as CSV")
    p.add_argument("audio", type=_p)
    p.add_argument("--t0", type=float, required=True)
    p.add_argument("--t1", type=float, required=True)
    p.add_argument("--fmin", type=float, required=True)
    p.add_argument("--fmax", type=float, required=True)
    p.add_argument("--f-start", type=float)
    p.add_argument("--f-end", type=float)
    p.add_argument("--harmonic", type=int, default=1)
    p.add_argument("--win", type=float, default=1.0)
    p.add_argument("--hop", type=float, default=0.25)
    p.add_argument("--sr", type=int, default=4000)
    p.add_argument("--max-slope", type=float, default=8.0, help="Hz/s")
    p.add_argument("--min-z", type=float, default=1.0)
    p.add_argument("--out", type=_p, default=Path("trace.csv"))
    p.add_argument("--png", type=_p)
    p.set_defaults(fn=cmd_trace)

    p = sub.add_parser("fit", help="fit a straight-line flyby to a trace; print an [[event]] block")
    p.add_argument("trace", type=_p)
    p.add_argument("--name", default="E1")
    p.add_argument("--harmonic", type=int)
    p.add_argument("--harmonics", type=int, nargs="+")
    p.add_argument("--t0", type=float)
    p.add_argument("--t1", type=float)
    p.add_argument("--c", type=float, default=343.0)
    p.add_argument("--rel", type=float, default=0.5)
    p.add_argument("--abs-hz", type=float, default=0.3)
    p.add_argument("--half-width", type=float, default=60)
    p.add_argument("--audio", type=_p)
    p.add_argument("--png", type=_p)
    p.set_defaults(fn=cmd_fit)

    for name, fn, hlp in (("search", cmd_search, "coarse search -> candidates.parquet"),
                          ("score", cmd_score, "model + score candidates -> scored.parquet, report"),
                          ("run", cmd_run, "search then score")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("case", type=_p)
        p.add_argument("--top", type=int, default=5)
        p.add_argument("--limit", type=int)
        p.add_argument("--truth", type=_p)
        p.add_argument("--keep-noisy", action="store_true", help="also score rows failing the silence filter")
        p.set_defaults(fn=fn)

    p = sub.add_parser("heatmap", help="dense score map around a ranked flight pair")
    p.add_argument("case", type=_p)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--half", type=float, default=1500)
    p.add_argument("--step", type=float, default=25)
    p.set_defaults(fn=cmd_heatmap)

    p = sub.add_parser("detect", help="find aircraft Doppler curves in an audio/video file or web video link")
    p.add_argument("source", help="file path, or a URL that yt-dlp understands (YouTube, ...)")
    p.add_argument("--start", type=float, help="analyse from this many seconds into the source")
    p.add_argument("--duration", type=float, help="seconds to analyse (default: to the end)")
    p.add_argument("--fmin", type=float, default=30.0)
    p.add_argument("--fmax", type=float, default=1000.0)
    p.add_argument("--threshold", type=float, default=8.0, help="matched-filter score threshold")
    p.add_argument("--no-refine", action="store_true", help="skip the trace + flyby fit of each detection")
    p.add_argument("--out", type=_p, help="output dir (default out/detect/<name>)")
    p.add_argument("--downloads", type=_p, default=Path("data/downloads"), help="where web audio is saved")
    p.set_defaults(fn=cmd_detect)

    p = sub.add_parser("synth", help="synthetic recording from real tracks (validation)")
    p.add_argument("case", type=_p)
    p.add_argument("--out", type=_p, default=Path("out/synth"))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--hour", type=float, default=18.0)
    p.add_argument("--snr", type=float, default=-6.0)
    p.set_defaults(fn=cmd_synth)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
