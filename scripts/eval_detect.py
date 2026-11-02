"""Known-answer test of ``dopploc detect``: compare its detections with the flyovers ADS-B says
the recording must contain, given the true camera position and the solved audio clock.

  python -P scripts/eval_detect.py CASE.toml DETECTIONS.json TRUTH.json [--radius 10000]

The audio clock comes from the rank-1 pair of the case's scoring (best_per_pair.csv):
UTC = audio time + audio0, audio0 = t1m - (event-1 audio_cpa + delta).

A detection matches a pass when the times agree within --tol seconds AND the detection's
speed and CPA distance are physically consistent with the aircraft's (straight-line fits are
rough: distance within a factor 3, speed within -40 %/+60 %). Chance agreement is measured by
repeating the matching with the detection times shifted by several minutes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from dopploc import config, expect

KINDS = {"J": "jet", "P": "piston", "T": "turboprop", "?": "unknown", "R": "rocket", "E": "electric"}


def run_match(dets, exp, tol, shift=0.0):
    ed, ev = exp["d"].to_numpy(), exp["speed"].to_numpy()

    def ok(i, j):
        f = dets[i]
        return 1 / 3 <= f["d"] / max(ed[j], 1.0) <= 3 and 0.6 <= f["v"] / max(ev[j], 1.0) <= 1.6

    return expect.match(np.array([f["t_cpa"] + shift for f in dets]), exp["audio_t"].to_numpy(), tol, ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", type=Path)
    ap.add_argument("detections", type=Path)
    ap.add_argument("truth", type=Path)
    ap.add_argument("--radius", type=float, default=10000.0)
    ap.add_argument("--tol", type=float, default=12.0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    case = config.load(a.case)
    truth = json.loads(a.truth.read_text())
    best = pl.read_csv(case.out / "best_per_pair.csv").row(0, named=True)
    audio0 = best["t1m"] - (case.events[0].audio_cpa + best["delta"])
    det = json.loads(a.detections.read_text())
    exp = expect.passes(case.adsb, case.cache, truth["lat"], truth["lon"], audio0 + det["start"],
                        audio0 + det["end"], radius_m=a.radius, c=case.c, audio0_utc=audio0)
    exp = exp.with_columns(pl.col("engine").replace(KINDS).alias("kind"))
    fl = det["flyovers"]
    air = [f for f in fl if f.get("kind", "aircraft") == "aircraft"]
    print(f"audio0 = {audio0:.1f} UTC; {len(exp)} ADS-B passes within {a.radius / 1000:.0f} km; "
          f"{len(fl)} detections, {len(air)} aircraft-like "
          f"({sum(f['confidence'] != 'low' for f in air)} high/medium)")
    rows = []
    for level in ("high/medium", "all aircraft-like"):
        sel = [f for f in air if level != "high/medium" or f["confidence"] != "low"]
        pairs, fa, _ = run_match(sel, exp, a.tol)
        chance = np.mean([len(run_match(sel, exp, a.tol, s)[0]) for s in (-600, -450, -300, -200, 200, 300, 450, 600)])
        hit = np.zeros(len(exp), bool)
        for _, j in pairs:
            hit[j] = True
        e = exp.with_columns(pl.Series("hit", hit))
        print(f"\n== {level}: {len(sel)} detections, {len(pairs)} match an ADS-B pass "
              f"(chance level {chance:.1f}), {len(fa)} without one")
        for kind in ("piston", "turboprop", "unknown", "jet"):
            for lo, hi in ((0, 2000), (2000, 4000), (4000, 7000), (7000, a.radius)):
                q = e.filter((pl.col("kind") == kind) & (pl.col("d") >= lo) & (pl.col("d") < hi))
                if len(q):
                    print(f"  {kind:9s} {lo / 1000:>4.0f}-{hi / 1000:<4.0f} km: {int(q['hit'].sum())}/{len(q)} detected")
                    rows.append(dict(level=level, kind=kind, lo=lo, hi=hi, n=len(q), hit=int(q["hit"].sum())))
        print("  matched:")
        for i, j in sorted(pairs, key=lambda p: sel[p[0]]["t_cpa"]):
            r = e.row(j, named=True)
            f = sel[i]
            print(f"    {f['t_cpa']:7.1f}s {f['f0']:6.1f}Hz v={f['v']:3.0f} d={f['d']:5.0f} {f['confidence']:6s} | "
                  f"ADS-B {r['audio_t']:7.1f}s {r['callsign'] or r['icao']:8s} {r['type'] or '?':5s} {r['kind']:9s} "
                  f"v={r['speed']:3.0f} d={r['d']:5.0f} alt={r['alt_m']:5.0f}m")
        if fa:
            print("  no ADS-B pass: " + ", ".join(f"{sel[i]['t_cpa']:.0f}s/{sel[i]['f0']:.0f}Hz/{sel[i]['v']:.0f}m/s/"
                                                  f"{sel[i]['d']:.0f}m" for i in fa))
    if a.out:
        exp.write_csv(a.out / "expected_passes.csv")
        pl.DataFrame(rows).write_csv(a.out / "recall.csv")


if __name__ == "__main__":
    main()
