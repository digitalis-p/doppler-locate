# doppler-locate

Reproduction of the audio-geolocation method shown in colsto's video
["I discovered a sound that leaks your exact location (and it's everywhere)"](https://www.youtube.com/watch?v=UytYrhYLOKI):
find where a recording was made from the Doppler curves of propeller aircraft hidden in its
spectrogram, matched against public ADS-B flight tracks. The video's author released no code;
this is an independent re-implementation from the narration and what is shown on screen.

## Method

1. **Measure** each flyover on a zoomed, whitened, high-resolution spectrogram (prop blade-pass
   tones sit around 50-600 Hz and are usually inaudible). Trace the ridge and fit a
   straight-line, constant-speed flyby: speed sets the size of the frequency drop, CPA slant
   distance sets how fast it happens. Keep generous ranges (real flights turn and change speed).
2. **ADS-B**: one UTC day of every aircraft over the continental US. Known jet types are
   removed with the ICAO type table (unknown types are kept, as in the video).
3. **Candidates**: a 200 m Albers grid. For every flight pass near every cell: 3D slant distance
   at the closest point of approach (aircraft altitude above MSL, cell height from a DEM) and the
   **delayed CPA time** (CPA emission time + distance / c). Keep cells that hear an event-1
   match (speed and distance in range) and, `offset` +- 5 s later, an event-2 match.
4. **Silence**: drop candidates that would have heard another non-jet flight between the two
   events nearer than the farther of the two events (the video's rule).
5. **Model and score**: for each surviving point (refined to 100 m), model the exact curve it
   would have heard from the real interpolated track using the **emission-time** radial velocity
   (`t_receive = t_emit + R(t_emit)/c`; using the receive-time velocity gives a wrongly shaped
   curve, not just a delayed one). Score = mean whitened spectrogram brightness under the curve's
   harmonics. Event 1 is swept over a few seconds of time shift and +-1.5 % of f0; that fixes the
   audio clock, and event 2 is then placed purely by flight geometry ("locked" scoring), with
   only its f0 swept.
6. **Single-event mode**: one faint curve plus a known time of day; the curve is traced and each
   candidate is scored by how well its modelled curve follows the traced line.

## Data sources (all free)

| What | Source |
|---|---|
| ADS-B tracks | [adsb.lol globe history](https://github.com/adsblol) daily readsb dumps (ODbL), via `gh release download` |
| Aircraft type -> engine class | [tar1090-db](https://github.com/wiedehopf/tar1090-db) `icao_aircraft_types2.js` (ICAO Doc 8643 codes such as `L1P`) |
| Terrain | Copernicus GLO-90 / GLO-30 DEM, AWS open data bucket |
| Geoid | EGM96 grid from cdn.proj.org (GNSS altitude -> MSL) |

The video used Flightradar24 (`pyfr24`, paid API) and a 30 m DEM.

## Install

```
uv venv --python 3.13 .venv
uv pip install -e '.[test]'
```

## Usage

```
dopploc fetch-adsb 2026-09-08              # ~3.9 GB split tar into data/adsb/raw/2026-09-08
dopploc fetch-types
dopploc ingest 2026-09-08                  # -> data/adsb/2026-09-08/points-*.parquet (~40 s)

dopploc spec REC.webm --t0 2050 --t1 2260 --fmin 40 --fmax 600 --out e1.png
dopploc trace REC.webm --t0 2085 --t1 2245 --fmin 70 --fmax 200 --f-start 177 --out e1.csv --png e1_trace.png
dopploc fit e1.csv --name E1 --harmonics 1 2 3     # prints an [[event]] block for the case file

dopploc search cases/oyster.toml          # coarse search -> out/.../candidates.parquet
dopploc score cases/oyster.toml           # model + score -> scored.parquet, best_per_pair.csv/.geojson, rank*.png
dopploc heatmap cases/oyster.toml --rank 1

dopploc synth cases/synth_base.toml --out out/synth --seed 3   # synthetic recording from real tracks
```

Case files (`cases/*.toml`) hold the audio path, the ADS-B day, the measured events and the
filter / scoring settings (see `src/dopploc/config.py` for every option).

## Layout

| Module | Role |
|---|---|
| `adsb.py` | adsb.lol fetch, readsb trace parsing, parquet ingest, type table |
| `audio.py` | ffmpeg decode, STFT, whitening (`brightness`) |
| `measure.py` | Viterbi ridge tracing, straight-line flyby fit, profile ranges |
| `physics.py` | Doppler, emission-time solver, CPA (numba) |
| `tracks.py` | track segmentation, altitude datum, Hermite resampling in ECEF |
| `dem.py`, `geo.py` | DEM tile cache, ECEF / Albers grid / geoid |
| `search.py` | tiled CPA records, event matching, offset and silence filters |
| `score.py` | forward model on real tracks, locked two-event / trace scoring, coarse-to-fine |
| `report.py` | ranked tables, GeoJSON for QGIS, scatter and curve-overlay plots |
| `synth.py` | synthetic recordings rendered from real tracks for validation |
