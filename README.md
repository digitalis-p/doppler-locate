# doppler-locate

Reproduction of the audio-geolocation method shown in colsto's video
["I discovered a sound that leaks your exact location (and it's everywhere)"](https://www.youtube.com/watch?v=UytYrhYLOKI):
find where a recording was made from the Doppler curves of propeller aircraft hidden in its
spectrogram, matched against public ADS-B flight tracks. The video's author released no code;
this is an independent re-implementation from the narration and what is shown on screen, using
only free data.

## Results

The video's challenge recording (60 min, "filmed 9/8/26", somewhere in the continental US;
answer: Oyster Bay, NY), searched over the whole CONUS with no location prior:

| Run | Truth rank | Top-1 distance from the camera | Score, 1st vs 2nd |
|---|---|---|---|
| Own measurements of both flyovers (`cases/oyster.toml`) | **1** of 1,060 flight pairs | 67 m | 3.662 vs 3.504 |
| The video's measured ranges (`cases/oyster_video.toml`) | **1** | 67 m | 3.683 vs 3.464 |
| Synthetic recording rendered from real tracks (`cases/synth.toml`) | **1** | 52 m | 4.378 vs 3.546 |
| Single faint flyover + known time (`cases/oyster_single.toml`) | not found | | see below |

The two flyovers are a Pilatus PC-12 at 17,250 ft (6.2 km slant) and a King Air 350 descending
through ~3,200 ft (3.5 km), heard 920 s apart.
A 25 m heat map around rank 1 (`dopploc heatmap cases/oyster.toml --rank 1`) peaks 13 m from
the camera. The GPU and CPU searches give the same ranking (the GPU search keeps every local
minimum of the distance, so it returns a superset of the CPU candidates).

What was needed beyond the video's description to rank the truth first with free data:
- **Winds and temperature aloft** (NOAA GFS analysis). For the PC-12 at 5.5 km the curve's
  residual at the truth drops from 0.55 Hz (still air, c = 343 m/s) to 0.21 Hz. A wind fitted to
  the audio alone (14.4 m/s from 316°) agrees with GFS (15.1 m/s from 322°).
- **One clock shift fitted to both events** rather than letting the far, flat event 1 set it.
- **Wide f0 and offset tolerances** (±8 %, ±8 s): straight-line fits are biased when the
  aircraft descends or slows (event 2: fit 134.4 Hz, true-geometry 126 Hz).
- **Removing steady tones** (running median per frequency bin) before scoring; intermittent
  hums at 107, 160 and 241 Hz otherwise attract candidates.

**Single-event mode** could not be reproduced. The faint curve at ~1279 s (≈20:38 UTC, ~0.7 km,
~72 m/s) has no aircraft in the adsb.lol data: the only non-jet within 8 km of the camera
between 20:30 and 20:46 UTC is an R-44 at 4 km. The video used Flightradar24, which has more
receivers and MLAT coverage. The code path (trace scoring, `[[event]] trace = ...`) is
implemented but not yet validated on a case where the aircraft is in the data.

The margin over the runner-up is narrower than in the video (3.66 vs 3.50 here; the video shows
20.5 vs 15.3 on its own scale, so the numbers are not comparable directly).

## Method

1. **Measure** each flyover on a zoomed, whitened spectrogram (prop blade-pass tones sit around
   50–600 Hz and are usually inaudible). Trace the ridge and fit a straight-line, constant-speed
   flyby: speed sets the size of the frequency drop, CPA slant distance how fast it happens.
   Profile the fit to get generous speed and distance ranges.
2. **ADS-B**: one UTC day of every aircraft over the continental US. Known jet types are
   removed with the ICAO type table (unknown types are kept, as in the video).
3. **Candidates**: a 200 m Albers grid. For every flight pass near every cell: 3D slant distance
   at the closest point of approach (aircraft altitude above MSL, cell height from a DEM) and the
   **delayed CPA time** (CPA emission time + distance / c). Keep cells that hear an event-1
   match and, `offset` ± tolerance later, an event-2 match.
4. **Silence**: drop candidates that would have heard another non-jet flight between the two
   events nearer than the farther of the two events (the video's rule).
5. **Model and score**: for each surviving point (refined to 100 m), model the exact curve it
   would have heard from the real interpolated track, with the **emission-time** radial velocity
   (`t_receive = t_emit + R(t_emit)/c`), the GFS wind at the aircraft's altitude and the
   path-mean sound speed and wind. Score = mean whitened spectrogram brightness under the
   curve's harmonics, with one audio-clock shift shared by both events and f0 swept per event.
   Coarse pass on every candidate, fine pass on the best.
6. **Single-event mode**: one faint curve plus a known time window; the traced line is compared
   with each candidate's modelled curve.

## Data sources (all free)

| What | Source |
|---|---|
| ADS-B tracks | [adsb.lol globe history](https://github.com/adsblol) daily readsb dumps (ODbL), via `gh release download` |
| Aircraft type -> engine class | [tar1090-db](https://github.com/wiedehopf/tar1090-db) `icao_aircraft_types2.js` (ICAO Doc 8643 codes such as `L1P`) |
| Terrain | Copernicus GLO-90 / GLO-30 DEM, AWS open data bucket |
| Geoid | EGM96 grid from cdn.proj.org (GNSS altitude -> MSL) |
| Winds / temperature aloft | NOAA GFS 0.25° analyses, AWS open data (byte-range requests for the needed fields only) |
| Surface wind (optional) | Iowa Environmental Mesonet ASOS archive |
| Map tiles (GUI only) | OpenStreetMap standard tiles via libshumate |

The video used Flightradar24 (`pyfr24`, paid API) and a 30 m DEM. Nothing about a search area
is sent anywhere by the pipeline; the GUI's map view requests OpenStreetMap tiles for the area
being viewed.

## Install

```
uv venv --python 3.13 .venv
uv pip install -e '.[test,gui]'
make -C native            # CUDA + OpenMP library (needs nvcc); `make -C native cpu` for OpenMP only
```

The GUI needs the system GTK 4, libadwaita (≥ 1.6) and libshumate typelibs. Without the native
library everything falls back to numba.

## Desktop app

```
dopploc-gui cases/oyster.toml
scripts/install-desktop.sh      # "Doppler Locate" in the app launcher (per user; --uninstall removes it)
```

Started without a case (e.g. from the launcher) it reopens the last case used.

![Audio page: event 1 traced (white) and the fitted flyby curve (cyan)](docs/gui_audio.png)
![Results page: ranked pairs, rank 1 next to the camera (green) on the map](docs/gui_results.png)

- **Audio**: whole-recording overview; click to zoom. Trace a ridge automatically (Viterbi) or
  by clicking, fit a flyby, adjust the model with sliders (f0, speed, CPA distance, CPA time)
  over the spectrogram, and add the result to the case as an event.
- **Case**: every setting of the case file. The TOML is edited in place (comments kept; a `.bak`
  copy is made the first time a hand-written file is saved).
- **Run**: fetch and ingest ADS-B, search, score and heat maps as child processes with a live
  log. Heavy steps run inside a memory-capped systemd scope (`scripts/run-capped.sh`), so an
  overrun kills only the job.
- **Results**: ranked flight pairs with distance to `truth.json` when present, the modelled curves
  over both events' spectrograms (rendered on demand for any row), an OpenStreetMap view of the
  top candidates, the event-score scatter, heat maps and the summary.

## Command line

```
dopploc fetch-adsb 2026-09-08              # ~3.9 GB split tar into data/adsb/raw/2026-09-08
dopploc fetch-types
dopploc ingest 2026-09-08                  # -> data/adsb/2026-09-08/points-*.parquet

dopploc spec REC.webm --t0 2050 --t1 2260 --fmin 40 --fmax 600 --out e1.png
dopploc trace REC.webm --t0 2085 --t1 2245 --fmin 70 --fmax 200 --f-start 177 --out e1.csv --png e1_trace.png
dopploc fit e1.csv --name E1 --harmonics 1 2 3     # prints an [[event]] block for the case file

scripts/run-capped.sh 26G dopploc search cases/oyster.toml   # -> out/oyster/candidates.parquet
scripts/run-capped.sh 26G dopploc score cases/oyster.toml --truth out/oyster/truth.json
dopploc heatmap cases/oyster.toml --rank 1

dopploc synth cases/synth_base.toml --out out/synth --seed 3   # synthetic recording from real tracks
```

Case files (`cases/*.toml`) hold the audio path, the ADS-B day, the measured events and the
filter / scoring settings (see `src/dopploc/config.py` for every option).

## Performance (RTX 4090, 32-thread CPU, 62 GB RAM)

| Step | Time |
|---|---|
| Ingest one day (53 M points) | ~40 s |
| CONUS search, CPU (numba, 24 processes; `[search] gpu = false`) | ~126 s, peak ~20 GB |
| CONUS search, GPU (default when the CUDA library is built) | ~80 s, peak 12 GB |
| Scoring 85 k candidates (CUDA forward model + sweeps) | 40–120 s (numba: ~350 s) |

The first search builds a 200 m CONUS elevation lattice from the DEM (~7 min, 1 GB, cached).

## Layout

| Module | Role |
|---|---|
| `adsb.py` | adsb.lol fetch, readsb trace parsing, parquet ingest, type table |
| `audio.py` | ffmpeg decode, STFT, whitening (`brightness`) |
| `measure.py` | Viterbi ridge tracing, straight-line flyby fit, profile ranges |
| `physics.py` | Doppler with wind and sound-speed profile, emission-time solver, CPA (numba) |
| `atmos.py`, `wind.py` | GFS winds / temperature aloft, ASOS surface wind |
| `tracks.py` | track segmentation, altitude datum, Hermite resampling in ECEF |
| `dem.py`, `geo.py` | DEM tile cache, elevation lattice, ECEF / Albers grid / geoid |
| `search.py`, `search_gpu.py` | CPA records, event matching, offset and silence filters (CPU / CUDA) |
| `score.py`, `native.py` | forward model on real tracks, joint two-event / trace scoring, coarse-to-fine |
| `report.py` | ranked tables, GeoJSON for QGIS, scatter and curve-overlay plots |
| `synth.py` | synthetic recordings rendered from real tracks for validation |
| `gui/` | GTK 4 / libadwaita app |
| `native/` | CUDA / OpenMP kernels (forward model, scoring sweeps, search) |
