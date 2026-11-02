"""Edit case TOML files in place with tomlkit, so hand-written comments and layout survive.

Values are read and written as TOML literals (``8.0``, ``[1, 2]``, ``true``); fields whose
type is a string take raw text. The first save over a file not written by the GUI keeps a
``.bak`` copy next to it.
"""

from __future__ import annotations

import dataclasses
import shutil
import tomllib
from pathlib import Path

import tomlkit

from .. import config

TOP = (("name", "str", "case name"), ("day", "str", "UTC day of the ADS-B data, YYYY-MM-DD"),
       ("audio", "str", "recording (any ffmpeg format)"), ("adsb", "str", "ingested ADS-B parquet dir"),
       ("cache", "str", "cache dir (DEM, GFS, wind)"), ("out", "str", "output dir"),
       ("c", "float", "surface speed of sound (m/s)"), ("atmosphere", "str", "gfs | asos | fixed | none"),
       ("wind_ms", "list", "fixed wind [east, north] m/s (atmosphere = fixed)"),
       ("altitude", "str", "auto | geom | baro"))
SECTIONS = {"search": config.Search, "silence": config.Silence, "score": config.Score}
MARK = "# dopploc case (edited by dopploc-gui)"


def _is_str(f: dataclasses.Field) -> bool:
    t = str(f.type).replace(" ", "")
    return t in ("str", "str|None")


def fields(cls):
    """(name, is_string, default_text) for a config dataclass."""
    out = []
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            dflt = f.default
        elif f.default_factory is not dataclasses.MISSING:
            dflt = f.default_factory()
        else:
            dflt = None
        out.append((f.name, _is_str(f), "" if dflt is None else to_text(dflt, _is_str(f))))
    return out


def to_text(v, is_str: bool) -> str:
    if hasattr(v, "unwrap"):
        v = v.unwrap()
    if is_str and isinstance(v, str):
        return v
    if isinstance(v, tuple):
        v = list(v)
    return tomlkit.item(v).as_string()


def parse(text: str, is_str: bool):
    """TOML literal -> Python value; raises ValueError. Empty text means 'unset' (None)."""
    text = text.strip()
    if not text:
        return None
    if is_str:
        return text
    try:
        return tomllib.loads("v = " + text)["v"]
    except tomllib.TOMLDecodeError as e:
        raise ValueError(str(e)) from None


EVENT_FIELDS = fields(config.Event)


def load(path: str | Path) -> tomlkit.TOMLDocument:
    return tomlkit.parse(Path(path).read_text())


def save(doc: tomlkit.TOMLDocument, path: str | Path):
    path = Path(path)
    text = tomlkit.dumps(doc)
    if not text.startswith(MARK):
        text = MARK + "\n" + text
    if path.exists() and not path.read_text().startswith(MARK):
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)
    path.write_text(text)


def is_table(v) -> bool:
    return isinstance(v, (tomlkit.items.Table, tomlkit.items.AoT, tomlkit.items.InlineTable))


def get(doc, section: str | None, key: str):
    tbl = doc if section is None else doc.get(section, {})
    return tbl.get(key)


def put(doc, section: str | None, key: str, value):
    """Set (or with value None remove) a key; creates the section on demand."""
    if section is None:
        tbl = doc
    else:
        if section not in doc:
            if value is None:
                return
            doc[section] = tomlkit.table()
        tbl = doc[section]
    if value is None:
        if key in tbl:
            del tbl[key]
    else:
        tbl[key] = list(value) if isinstance(value, tuple) else value


def events(doc) -> list:
    return list(doc.get("event", []))


def add_event(doc, ev: dict):
    t = tomlkit.table()
    for k, v in ev.items():
        if v is not None:
            t[k] = list(v) if isinstance(v, tuple) else v
    if "event" not in doc:
        doc["event"] = tomlkit.aot()
    elif len(doc["event"]):
        doc["event"][-1].add(tomlkit.nl())
    doc["event"].append(t)


def remove_event(doc, i: int):
    del doc["event"][i]
    if not len(doc["event"]):
        del doc["event"]


def new_case(audio: str = "", day: str = "") -> tomlkit.TOMLDocument:
    doc = tomlkit.document()
    doc.add(tomlkit.comment("dopploc case (edited by dopploc-gui)"))
    doc["name"] = "untitled"
    doc["day"] = day
    doc["audio"] = audio
    doc["adsb"] = f"data/adsb/{day}" if day else "data/adsb"
    doc["cache"] = "data/cache"
    doc["out"] = "out/untitled"
    doc["atmosphere"] = "gfs"
    s = tomlkit.table()
    s["offset_tol_s"] = 8.0
    doc["search"] = s
    sc = tomlkit.table()
    sc["dt_sweep"] = [-8.0, 8.0, 0.1]
    doc["score"] = sc
    return doc
