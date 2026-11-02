# NOTES — doppler-locate

Durable facts about this project that are not obvious from the code.

## Source material
- Video: https://www.youtube.com/watch?v=UytYrhYLOKI (colsto, uploaded 2026-11-02). Local copy:
  `~/Videos/I discovered a sound that leaks your exact location (and it's everywhere) [UytYrhYLOKI].webm`.
- Challenge audio: https://youtu.be/4CxVxuMVkMc ("Unknown location - raw audio", 60 min). It is the
  60-120 min chunk of a 2-hour security-camera recording ("Filmed 9/8/26", continental US).
  Video times minus 3600 s = times in this file. Local: `data/recording/recording_4CxVxuMVkMc.webm`.
- Answer shown in the video: Oyster Bay, NY. Camera [position omitted] (QGIS copy at 25:27);
  the video's best candidate [position omitted]; single-event best [position omitted].
- The video used Flightradar24 (pyfr24, paid) data, a 30 m DEM, ASOS surface wind, c = 343 m/s,
  offset 920 s +-5 s, silence rule "other non-jet CPA between the events nearer than max(d1, d2)".

## The two events (this file's clock)
| | audio CPA | aircraft | ADS-B | measured here |
|---|---|---|---|---|
| E1 | ~2153-2156 s | PC-12, 17,250 ft, 137 m/s, ~20:53 UTC | 6.2 km slant | f0 112.7 Hz, v 140.7, d 6.4 km |
| E2 | ~3072-3079 s | King Air 350, ~3,200 ft descending, 103 m/s, CPA 21:08:39 UTC | 3.5 km | v 92.9, d 2.2 km (straight-line fit) |
| faint | ~1279 s | single-event mode, "8:37 p.m. UTC" | | v 71.8, d 716 m, f0 400 (traced) |

Geometry offset between the delayed CPAs at the camera: ~920.6-924 s (audio midpoints: 916.6 s by
straight-line fits; the video measured 920-921 s).

## Lessons (cost real time)
- **Winds aloft and sound speed matter.** At the truth, the E1 straight-line residual is 0.55 Hz in
  still air, 0.21 Hz with GFS wind at aircraft altitude + path-mean sound speed. A free fit of the
  audio alone asks for 14.4 m/s from 316 deg; GFS says 15.1 m/s from 322 deg at 5.5 km.
- **Straight-line fits are biased**: midpoint times by several seconds (w*d/v^2) and f0 by up to ~6 %
  (E2 descends and slows: fit says 134.4 Hz, true-geometry model needs ~126 Hz). Hence the +-8 s
  offset tolerance and +-8 % f0 sweep.
- Event 1 (far, slow curve) cannot anchor the audio clock alone; a single shared shift fitted to
  both events ("joint") is required.
- Intermittent stationary tones (107, 160, 241 Hz here) attract wrong candidates; the scoring
  spectrogram removes tones steadier than ~10 s (running median over 20 s per bin).
- adsb.lol thins straight-and-level traces to ~20 s; Hermite interpolation with reported velocity
  handles it.
- An altitude cap of 15,000 ft silently dropped the event-1 aircraft; the cap is now FL350.
- The tar1090-db type table lacks some common types (e.g. PA28): exclude known jets, keep unknowns.
- **The faint single-event curve (~1279 s) is not in adsb.lol.** Between 20:30 and 20:46 UTC the
  only non-jet within 8 km of the camera is an R-44 at 4.0 km (20:35:53Z); the curve needs ~0.7 km
  at ~72 m/s. The video's FR24 data had it; adsb.lol (fewer feeders, no FR24 MLAT) does not, so
  the single-event result cannot be reproduced from free data (nearest candidate 2.7 km).
- Never edit a bash script while a job is executing it: bash reads scripts incrementally, and
  editing `run-capped.sh` mid-run broke the tail of a running job (exit status was kept).

## Machine / ops
- The CONUS search with 24-30 workers peaks near 20 GB (including page cache). On 2026-11-02 a
  search running beside a scoring job exhausted RAM+swap and systemd-oomd killed the whole terminal
  scope. Run heavy jobs one at a time via `scripts/run-capped.sh <cap> dopploc ...` (cgroup cap).
- Per-worker GDAL cache is capped (32 MB) and the DEM tile LRU is 8 tiles; GDAL's default (5 % of RAM
  per process) was a large part of the blow-up.
- GPU: RTX 4090; CUDA 13.4 needs `-allow-unsupported-compiler` with gcc 16. Build: `make -C native`.
- GUI tests run headless: `gtk4-broadwayd --address 127.0.0.1 :7` and `GDK_BACKEND=broadway
  BROADWAY_DISPLAY=:7`; render widgets to PNG with Gtk.WidgetPaintable + Gsk.CairoRenderer
  (take the shot one frame after a page change, or the snapshot node is empty).
- CONUS elevation lattice (200 m, int16, 1.0 GB): `data/cache/elev/conus_albers_200m.npy`, built
  once in ~7 min from GLO-90.
- Caches: `data/cache/dem/glo90` (~4.5 GB, CONUS), `data/cache/gfs` (~32 MB/day), `data/cache/wind`.

## Results log
- 2026-11-02 synthetic (seed 3, real tracks, still air): truth rank 1, 52 m.
- 2026-11-02 real, own measurements (`cases/oyster.toml`): rank 1, 67 m, score 3.662 vs 3.504.
- 2026-11-02 real, video's ranges (`cases/oyster_video.toml`): rank 1, 67 m, 3.683 vs 3.464.
- 2026-11-02 single-event (`cases/oyster_single.toml`, 20:35-20:41Z): not reproducible, see Lessons.
- 2026-11-02 GPU search candidates (superset of CPU) scored: rank 1, 67 m, 3.662 vs 3.504 (same top 20).
- 2026-11-02 heat map around rank 1 (25 m, GLO-90): peak 13 m from the camera.
- 2026-11-02 synthetic re-run with current defaults (GPU search, joint shift, line suppression,
  f0 +-8 %): rank 1, 52 m, 4.378 vs 3.546.
- Launcher: `scripts/install-desktop.sh` installed ~/.local/share/applications/io.github.digitalis_p.dopploc.desktop
  and the hicolor icon (2026-11-02); `--uninstall` removes both. GUI state: ~/.local/state/dopploc/gui.json.
