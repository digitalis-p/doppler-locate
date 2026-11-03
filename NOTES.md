# Development notes

Facts about this project that are not obvious from the code, lessons that cost real time, and
the results log. The README has the overview.

## Source material
- Video: https://www.youtube.com/watch?v=UytYrhYLOKI (colsto, uploaded 2026-11-02).
- Challenge audio: https://youtu.be/4CxVxuMVkMc ("Unknown location - raw audio", 60 min). It is the
  60-120 min chunk of a 2-hour security-camera recording ("Filmed 9/8/26", continental US), so
  times quoted in the video minus 3600 s are times in this file. Expected locally at
  `data/recording/recording_4CxVxuMVkMc.webm`.
- Answer shown in the video: Oyster Bay, NY (camera position on screen at 25:27). The position used
  for distances is kept locally in `out/oyster/truth.json`, which is not in git.
- The video used Flightradar24 (pyfr24, paid) data, a 30 m DEM, ASOS surface wind, c = 343 m/s,
  offset 920 s +-5 s, and the silence rule "another non-jet CPA between the events nearer than
  max(d1, d2)".

## The two events (this file's clock)
| | audio CPA | aircraft | ADS-B | measured here |
|---|---|---|---|---|
| E1 | ~2153-2156 s | PC-12, 17,250 ft, 137 m/s, ~20:53 UTC | 6.2 km slant | f0 112.7 Hz, v 140.7, d 6.4 km |
| E2 | ~3072-3079 s | King Air 350, ~3,200 ft descending, 103 m/s, CPA 21:08:39 UTC | 3.5 km | v 92.9, d 2.2 km (straight-line fit) |
| faint | ~1279 s | single-event mode, "8:37 p.m. UTC" | | v 71.8, d 716 m, f0 400 (traced) |

Geometry offset between the delayed CPAs at the camera: ~920.6-924 s (audio midpoints: 916.6 s by
straight-line fits; the video measured 920-921 s).

## Lessons
- **Winds aloft and sound speed matter.** At the truth, the E1 straight-line residual is 0.55 Hz in
  still air and 0.21 Hz with the GFS wind at aircraft altitude and the path-mean sound speed. A free
  fit of the audio alone asks for 14.4 m/s from 316 deg; GFS says 15.1 m/s from 322 deg at 5.5 km.
- **Straight-line fits are biased**: midpoint times by several seconds (w*d/v^2) and f0 by up to ~6 %
  (E2 descends and slows: the fit says 134.4 Hz, the true-geometry model needs ~126 Hz). Hence the
  +-8 s offset tolerance and the +-8 % f0 sweep.
- Event 1 (far, slow curve) cannot anchor the audio clock alone; a single shift fitted to both
  events ("joint") is required.
- Intermittent stationary tones (107, 160, 241 Hz here) attract wrong candidates; the scoring
  spectrogram removes tones steadier than ~10 s (running median over 20 s per bin).
- adsb.lol thins straight-and-level traces to ~20 s; Hermite interpolation with the reported
  velocity handles it.
- An altitude cap of 15,000 ft silently dropped the event-1 aircraft; the cap is now FL350.
- The tar1090-db type table lacks some common types (e.g. PA28): exclude known jets, keep unknowns.
- **The faint single-event curve (~1279 s) is not in adsb.lol.** Between 20:30 and 20:46 UTC the
  only non-jet within 8 km of the camera is an R-44 at 4.0 km (20:35:53Z); the curve needs ~0.7 km
  at ~72 m/s. The video's FR24 data had it; adsb.lol (fewer feeders, no FR24 MLAT) does not, so
  the single-event result cannot be reproduced from free data (nearest candidate 2.7 km).
- Never edit a bash script while a job is executing it: bash reads scripts incrementally, and
  editing `run-capped.sh` mid-run broke the tail of a running job.

## Flyover detector (dopploc detect)
- Road vehicles make the same Doppler curves (v 10-30 m/s, d < 100 m): classify by fitted d and v.
- The matched-filter score alone is a poor confidence measure on real audio: 39 "medium+" hits,
  only 4 matching ADS-B. The flyby-fit residual relative to frequency separates them (real
  <= 0.5 %, junk >= 1.2 %); the cut is at 1 %. Unrefined detections (fit failed or at bounds) are low.
- Time-only matching against ADS-B is meaningless (68 detections x 49 passes in 1 h give ~28
  chance matches): eval_detect.py requires consistent speed and distance and reports the chance level.
- The per-detection fit (profile_ranges) dominated the run time (95 s per 10 min of audio);
  fits now run in a spawned process pool, with ranges only for kept detections: 60 min in ~70 s.
- Use spawn, not fork, for process pools once numba or FFT threads have run.
- One aircraft shows several components (PC-12: 113 Hz blade pass and 141 Hz, a 5:4 shaft
  harmonic) whose fits disagree by ~20 s in CPA time: merge by small-integer ratio within
  max(25 s, tau/2).
- HWNDU (Mar 2017): only short clips survive (YouTube HxRLsWKYA6Q, W6uGP8hHwrQ, -C9uac3DsBA;
  archive.org HeWillNotDivideUsTheFalg is 40 s); no Doppler curves in them; adsb.lol starts in 2023.
- Webcam archives as test data: 24/7 streams leave no archive on YouTube (streams over 12 h are not
  kept), so usable ones restart every <= 12 h. For a live archive, `release_timestamp` is the UTC
  start (`eval_detect.py --audio0`); the clock line it prints measures any remaining offset.
- yt-dlp metadata requests in parallel (8 at once, ~240 videos) triggered YouTube HTTP 429 and a
  "confirm you're not a bot" block that cleared within ~20 min. Fetch one at a time, ~10 s apart.

## Development machine and operations
- The continental-US search with 24-30 workers peaks near 20 GB (including page cache). A search
  running beside a scoring job once exhausted RAM and swap, and systemd-oomd killed the whole
  terminal scope. Run heavy jobs one at a time via `scripts/run-capped.sh <cap> dopploc ...`.
- The per-worker GDAL cache is capped (32 MB) and the DEM tile LRU holds 8 tiles; GDAL's default
  (5 % of RAM per process) was a large part of that blow-up.
- GPU: RTX 4090; CUDA 13.4 needs `-allow-unsupported-compiler` with gcc 16. Build: `make -C native`.
- GUI tests run headless: `gtk4-broadwayd --address 127.0.0.1 :7` with `GDK_BACKEND=broadway
  BROADWAY_DISPLAY=:7`; render widgets to PNG with Gtk.WidgetPaintable + Gsk.CairoRenderer
  (take the shot one frame after a page change, or the snapshot node is empty).
- Icons in ~/.local/share/icons/hicolor/scalable/apps stay invisible to a running GNOME Shell until
  the theme's top directory changes (it compares only that mtime); install-desktop.sh touches
  ~/.local/share/icons/hicolor. Do not create a user icon cache.
- Icon names: the breeze-dark theme lacks web-browser-symbolic, sidebar-show-right-symbolic and
  audio-x-generic-symbolic; check names with Gtk.IconTheme.has_icon.
- GUI state (last case): ~/.local/state/dopploc/gui.json.
- Continental-US elevation lattice (200 m, int16, 1.0 GB): `data/cache/elev/conus_albers_200m.npy`,
  built once in ~7 min from GLO-90.
- Caches: `data/cache/dem/glo90` (~4.5 GB for the continental US), `data/cache/gfs` (~32 MB per
  day), `data/cache/wind`.

## Results log
- 2026-11-02 synthetic (seed 3, real tracks, still air): truth rank 1, 52 m.
- 2026-11-02 real, own measurements (`cases/oyster.toml`): rank 1, 67 m, score 3.662 vs 3.504.
- 2026-11-02 real, video's ranges (`cases/oyster_video.toml`): rank 1, 67 m, 3.683 vs 3.464.
- 2026-11-02 single-event (`cases/oyster_single.toml`, 20:35-20:41Z): not reproducible, see Lessons.
- 2026-11-02 GPU search candidates (a superset of the CPU ones) scored: rank 1, 67 m, 3.662 vs
  3.504 (same top 20).
- 2026-11-02 heat map around rank 1 (25 m, GLO-90): peak 13 m from the camera.
- 2026-11-02 synthetic re-run with current defaults (GPU search, joint shift, line suppression,
  f0 +-8 %): rank 1, 52 m, 4.378 vs 3.546.
- 2026-11-02 detector, video's recording: 8 flyovers (E1, E2, faint 21:18, 2 jets, 3 curves with no
  ADS-B aircraft); 4/8 match ADS-B (chance 0.6). Synthetic: 2/2 passes within 1 km, 0 false.
