"""Case configuration (TOML): audio, ADS-B day, measured events, filter and scoring settings."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Event:
    name: str
    audio_cpa: float                 # audio-clock seconds of the delayed CPA (curve midpoint)
    f0: float                        # source fundamental (Hz) from the fit
    speed_ms: tuple[float, float]    # ground-speed range accepted (m/s)
    slant_m: tuple[float, float]     # CPA slant-distance range accepted (m)
    harmonics: list[int] = field(default_factory=lambda: [1])
    half_width_s: float = 60.0       # scoring clip: audio_cpa +- half_width
    fband: tuple[float, float] | None = None   # scoring band (Hz); default from f0 and harmonics
    trace: str | None = None         # CSV (t,f[,h]) of a hand/auto traced curve: line scoring


@dataclass
class Silence:
    """Disqualify candidates that would have heard other aircraft between the two events.

    mode "cpa" (the video's rule): another non-jet flight whose delayed CPA falls between the
    two events' delayed CPAs, closer than ``ratio`` * max(d1, d2) (the farther event was
    still audible, so anything nearer would have been too). Jets are ignored.
    mode "samples": any non-jet track sample within ``radius_m`` whose sound arrives in
    (t1 + margin_s, t2 - margin_s); optionally low jets within ``jet_radius_m``.
    """
    enabled: bool = True
    mode: str = "cpa"
    ratio: float = 1.0
    radius_m: float = 3000.0
    margin_s: float = 60.0
    jets_below_ft: float = 0.0
    jet_radius_m: float = 2000.0


@dataclass
class Search:
    bbox: tuple[float, float, float, float] = (-126.0, 23.5, -65.5, 50.5)
    utc_window: tuple[float, float] | None = None  # constrain event-1 delayed CPA (unix s)
    grid_m: float = 200.0
    tile_km: float = 20.0
    sample_dt: float = 3.0
    offset_tol_s: float = 6.0
    exclude_engines: tuple[str, ...] = ("J", "R")
    max_alt_ft: float = 35000.0       # props up to FL350; the slant band rejects anything too high
    speed_margin_ms: float = 15.0
    workers: int = 0
    elevation_lattice: bool = True   # cached CONUS terrain on the search lattice (built once)
    gpu: bool = True                 # CUDA search when the native CUDA library is built


@dataclass
class Score:
    win_s: float = 1.0
    hop_s: float = 0.25
    dt_sweep: tuple[float, float, float] = (-3.0, 3.0, 0.1)
    f0_sweep_frac: float = 0.08      # straight-line f0 can be off ~6% (descent, deceleration)
    f0_step_hz: float = 0.05
    refine_m: float = 100.0
    model_dt: float = 0.1
    freq_tol_bins: int = 1
    mode: str = "joint"              # "joint": one shared clock shift for both events (locked offset);
                                     # "locked": event 1 alone sets the clock shift
    line_suppress_s: float = 20.0    # remove tones steadier than ~this/2 s before scoring (0 = off)
    trace_sigma_hz: float = 1.5
    dem_res: int = 90               # 30 for local heat maps; 90 keeps CONUS-wide scoring cheap


@dataclass
class Case:
    name: str
    root: Path
    audio: Path
    adsb: Path
    cache: Path
    out: Path
    day: str
    c: float = 343.0
    altitude: str = "auto"
    atmosphere: str = "gfs"          # "gfs" (winds/temperature aloft), "asos" (surface wind, as the video),
                                     # "fixed" (wind_ms), "none" (still air at c)
    wind_ms: tuple[float, float] | None = None   # for "fixed": (east, north) m/s the air moves toward
    events: list[Event] = field(default_factory=list)
    search: Search = field(default_factory=Search)
    silence: Silence = field(default_factory=Silence)
    score: Score = field(default_factory=Score)

    @property
    def offset_s(self) -> float | None:
        if len(self.events) < 2:
            return None
        return self.events[1].audio_cpa - self.events[0].audio_cpa


def _ts(x) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, datetime):
        return x.replace(tzinfo=x.tzinfo or timezone.utc).timestamp()
    return datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp()


def load(path: str | Path) -> Case:
    path = Path(path).resolve()
    d = tomllib.loads(path.read_text())
    root = Path(d.get("root", path.parent.parent)).resolve()

    def p(key, default):
        v = Path(d.get(key, default))
        return v if v.is_absolute() else root / v

    s = dict(d.get("search", {}))
    if s.get("utc_window"):
        s["utc_window"] = tuple(_ts(x) for x in s["utc_window"])
    else:
        s.pop("utc_window", None)
    for k in ("bbox", "exclude_engines"):
        if k in s:
            s[k] = tuple(s[k])
    sc = dict(d.get("score", {}))
    if "dt_sweep" in sc:
        sc["dt_sweep"] = tuple(sc["dt_sweep"])
    events = []
    for e in d.get("event", []):
        e = dict(e)
        e["speed_ms"] = tuple(e["speed_ms"])
        e["slant_m"] = tuple(e["slant_m"])
        if e.get("fband"):
            e["fband"] = tuple(e["fband"])
        if e.get("trace"):
            tp = Path(e["trace"])
            e["trace"] = str(tp if tp.is_absolute() else root / tp)
        events.append(Event(**e))
    return Case(
        name=d.get("name", path.stem), root=root,
        audio=p("audio", "audio.wav"), adsb=p("adsb", "data/adsb"), cache=p("cache", "data/cache"),
        out=p("out", f"out/{path.stem}"), day=d["day"], c=float(d.get("c", 343.0)),
        altitude=d.get("altitude", "auto"), events=events,
        atmosphere=d.get("atmosphere", d.get("wind", "fixed" if d.get("wind_ms") else "gfs")),
        wind_ms=tuple(d["wind_ms"]) if d.get("wind_ms") else None,
        search=Search(**s), silence=Silence(**d.get("silence", {})), score=Score(**sc),
    )
