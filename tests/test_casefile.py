"""Case-file editing used by the GUI: comments survive, values round-trip, config.load accepts the result."""

from pathlib import Path

import pytest

pytest.importorskip("tomlkit")

from dopploc import config  # noqa: E402
from dopploc.gui import casefile  # noqa: E402

CASE = Path(__file__).resolve().parents[1] / "cases" / "oyster.toml"


def test_roundtrip_keeps_comments_and_loads(tmp_path):
    doc = casefile.load(CASE)
    casefile.put(doc, "search", "grid_m", casefile.parse("150", False))
    casefile.put(doc, None, "atmosphere", casefile.parse("asos", True))
    casefile.put(doc, "score", "dt_sweep", casefile.parse("[-4.0, 4.0, 0.2]", False))
    casefile.add_event(doc, {"name": "E3", "audio_cpa": 100.0, "f0": 90.0, "harmonics": [1],
                             "speed_ms": [40.0, 80.0], "slant_m": [300.0, 900.0]})
    casefile.remove_event(doc, 0)
    out = tmp_path / "cases" / "edited.toml"
    out.parent.mkdir()
    casefile.save(doc, out)
    text = out.read_text()
    assert text.startswith(casefile.MARK)
    assert "# event-1 clock sweep" in text  # a comment from the original file
    c = config.load(out)
    assert c.search.grid_m == 150.0 and c.atmosphere == "asos" and c.score.dt_sweep == (-4.0, 4.0, 0.2)
    assert [e.name for e in c.events] == ["E2", "E3"]


def test_backup_on_first_save(tmp_path):
    p = tmp_path / "x.toml"
    p.write_text(CASE.read_text())
    casefile.save(casefile.load(p), p)
    assert (tmp_path / "x.toml.bak").read_text() == CASE.read_text()
    casefile.save(casefile.load(p), p)  # second save: file now carries the GUI mark, backup untouched
    assert (tmp_path / "x.toml.bak").read_text() == CASE.read_text()


def test_parse_and_text():
    assert casefile.parse("", False) is None
    assert casefile.parse("[1, 2]", False) == [1, 2]
    assert casefile.parse("true", False) is True
    with pytest.raises(ValueError):
        casefile.parse("[1, 2", False)
    assert casefile.to_text((1.0, 2.0), False) == "[1.0, 2.0]"
    assert casefile.to_text("gfs", True) == "gfs"
    names = [n for n, _, _ in casefile.EVENT_FIELDS]
    assert names[:3] == ["name", "audio_cpa", "f0"]
