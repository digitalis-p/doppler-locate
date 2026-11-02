"""dopploc-gui: GTK4 / libadwaita front end for the dopploc pipeline.

Pages
  Audio    whole-recording overview and a zoomed, whitened spectrogram; ridge tracing (automatic
           or by clicking), straight-line flyby fit, live model sliders, "add as event".
  Case     every case setting; the TOML file is edited in place (comments kept, .bak on first save).
  Run      fetch / ingest / search / score / heat map as child processes with a live log; heavy
           steps run in a memory-capped cgroup scope (scripts/run-capped.sh).
  Results  ranked flight pairs, modelled curves over the spectrograms, OpenStreetMap view,
           event-score scatter, heat map, summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
try:
    gi.require_version("Shumate", "1.0")
    from gi.repository import Shumate
except (ValueError, ImportError):
    Shumate = None
from gi.repository import Adw, Gdk, Gio, GLib, Gtk

import numpy as np

from . import casefile
from .plots import ScatterView, SpecView
from .runner import CAPPED, Job, capped, dopploc_argv

CSS = b"""
.pin { background: #c01c28; color: white; border-radius: 9px; padding: 0 5px; font-size: 8pt; font-weight: bold; }
.pin.truth { background: #26a269; }
.pin.sel { background: #1c71d8; font-size: 10pt; }
.log { font-family: monospace; font-size: 9pt; }
"""


def in_thread(fn, done, fail):
    """Run fn() off the main loop, then done(result) or fail(exception) back on it."""
    def run():
        try:
            r = fn()
        except Exception as e:  # shown to the user by fail()
            GLib.idle_add(lambda: fail(e) and False)
            return
        GLib.idle_add(lambda: done(r) and False)
    threading.Thread(target=run, daemon=True).start()


def spin(lo, hi, step, digits, value, width=7):
    s = Gtk.SpinButton.new_with_range(lo, hi, step)
    s.set_digits(digits)
    s.set_value(value)
    s.set_width_chars(width)
    return s


def labeled(text, w, tip=None):
    b = Gtk.Box(spacing=4)
    lab = Gtk.Label(label=text)
    if tip:
        lab.set_tooltip_text(tip)
        w.set_tooltip_text(tip)
    b.append(lab)
    b.append(w)
    return b


def row(*widgets, spacing=10):
    b = Gtk.Box(spacing=spacing)
    for w in widgets:
        b.append(w)
    return b


def button(label, cb, style=None, tip=None):
    b = Gtk.Button(label=label)
    b.connect("clicked", lambda *_: cb())
    if style:
        b.add_css_class(style)
    if tip:
        b.set_tooltip_text(tip)
    return b


def file_dialog(title, patterns=None, folder: Path | None = None, name: str | None = None):
    d = Gtk.FileDialog(title=title, modal=True)
    if patterns:
        store = Gio.ListStore.new(Gtk.FileFilter)
        for label, pats in patterns:
            f = Gtk.FileFilter()
            f.set_name(label)
            for p in pats:
                f.add_pattern(p)
            store.append(f)
        d.set_filters(store)
    if folder and folder.is_dir():
        d.set_initial_folder(Gio.File.new_for_path(str(folder)))
    if name:
        d.set_initial_name(name)
    return d


# --------------------------------------------------------------------------------------------- audio

class AudioPage(Gtk.ScrolledWindow):
    def __init__(self, win: "Window"):
        super().__init__(hscrollbar_policy=Gtk.PolicyType.NEVER)
        self.win = win
        self.path: Path | None = None
        self.tr_t: list[float] = []
        self.tr_f: list[float] = []
        self.tr_z: list[float] = []
        self.fit = None
        self.ranges = None
        self.manual = False
        self._setting = False
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin_top=10, margin_bottom=10,
                      margin_start=12, margin_end=12)
        self.set_child(box)

        self.file_label = Gtk.Label(label="no audio loaded", xalign=0, hexpand=True,
                                    ellipsize=3)  # Pango.EllipsizeMode.END
        box.append(row(button("Open audio…", self.on_open), self.file_label,
                       button("Reload overview", self.load_overview)))
        self.overview = SpecView(height=170, title="whole recording, whitened (click to zoom there)")
        self.overview.on_click(self.on_overview_click)
        box.append(self.overview.canvas)

        self.t0 = spin(0, 1e6, 1, 1, 0)
        self.t1 = spin(0, 1e6, 1, 1, 120)
        self.fmin = spin(1, 4000, 5, 0, 30, 5)
        self.fmax = spin(1, 4000, 5, 0, 600, 5)
        self.win_s = spin(0.1, 4, 0.1, 2, 1.0, 4)
        box.append(row(labeled("from (s)", self.t0), labeled("to (s)", self.t1), labeled("Hz", self.fmin),
                       labeled("–", self.fmax), labeled("window (s)", self.win_s, "STFT window; hop = window / 4"),
                       button("Show", self.load_zoom, "suggested-action")))
        self.zoom = SpecView(height=380, title="zoom")
        self.zoom.on_click(self.on_zoom_click)
        box.append(self.zoom.canvas)

        self.harm = spin(1, 12, 1, 0, 1, 3)
        self.fs = spin(0, 4000, 1, 1, 0, 6)
        self.fe = spin(0, 4000, 1, 1, 0, 6)
        self.click_mode = Gtk.ToggleButton(label="Click to trace")
        self.click_mode.set_tooltip_text("left click adds a point, right click removes the nearest")
        box.append(row(labeled("harmonic", self.harm, "which harmonic the traced line is"),
                       labeled("start Hz", self.fs, "pin the ridge near this frequency at the left edge (0 = free)"),
                       labeled("end Hz", self.fe, "pin the ridge near this frequency at the right edge (0 = free)"),
                       button("Trace ridge", self.on_trace), self.click_mode,
                       button("Clear", self.on_clear), button("Save trace…", self.on_save_trace)))

        self.harms = Gtk.Entry(text="1", width_chars=8)
        self.harms.set_tooltip_text("harmonics drawn and stored in the event, e.g. 1 2 3")
        self.harms.connect("changed", lambda *_: self.draw_model())
        self.fit_label = Gtk.Label(label="trace a curve, then fit", xalign=0, hexpand=True, wrap=True,
                                   selectable=True)
        box.append(row(button("Fit flyby", self.on_fit, "suggested-action"), labeled("harmonics", self.harms),
                       self.fit_label))

        grid = Gtk.Grid(column_spacing=10, row_spacing=2)
        self.sl = {}
        for i, (key, lbl, lo, hi, step, val, dig) in enumerate((
                ("f0", "f0 (Hz)", 5, 2000, 0.05, 100, 2), ("v", "speed (m/s)", 10, 260, 0.5, 80, 1),
                ("d", "CPA distance (m)", 50, 20000, 10, 2000, 0), ("t", "CPA audio time (s)", 0, 1e6, 0.05, 60, 2))):
            adj = Gtk.Adjustment(value=val, lower=lo, upper=hi, step_increment=step, page_increment=step * 20)
            sc = Gtk.Scale(adjustment=adj, hexpand=True, digits=dig, draw_value=True,
                           value_pos=Gtk.PositionType.RIGHT)
            sc.connect("value-changed", self.on_slider)
            grid.attach(Gtk.Label(label=lbl, xalign=0), 0, i, 1, 1)
            grid.attach(sc, 1, i, 1, 1)
            self.sl[key] = sc
        self.show_model = Gtk.CheckButton(label="show model curve")
        self.show_model.connect("toggled", lambda *_: self.draw_model())
        grid.attach(self.show_model, 0, 4, 2, 1)
        box.append(grid)

        self.ev_name = Gtk.Entry(text="E1", width_chars=6)
        self.half = spin(10, 300, 5, 0, 60, 4)
        self.use_trace = Gtk.CheckButton(label="score against the trace (faint curve)")
        box.append(row(labeled("event name", self.ev_name), labeled("half width (s)", self.half),
                       self.use_trace, button("Add to case", self.on_add, "suggested-action")))

    # loading
    def on_open(self):
        d = file_dialog("Open recording", [("Audio / video", ["*.wav", "*.flac", "*.mp3", "*.ogg", "*.opus",
                                                              "*.m4a", "*.webm", "*.mkv", "*.mp4", "*.mov"]),
                                           ("All files", ["*"])], self.win.root())
        d.open(self.win, None, lambda dlg, res: self._opened(dlg, res))

    def _opened(self, dlg, res):
        try:
            f = dlg.open_finish(res)
        except GLib.Error:
            return
        self.set_audio(Path(f.get_path()))
        if self.win.case_path and casefile.get(self.win.doc, None, "audio") in (None, ""):
            self.win.set_value(None, "audio", self.win.rel(self.path))

    def set_audio(self, path: Path, focus=None):
        self.path = path
        self.file_label.set_text(str(path))
        self.load_overview()
        if focus:
            t0, t1, f0, f1 = focus
            self.t0.set_value(max(0.0, t0))
            self.t1.set_value(t1)
            self.fmin.set_value(f0)
            self.fmax.set_value(f1)
            self.load_zoom()

    def load_overview(self):
        if not self.path:
            return
        from .. import audio
        path = self.path

        def work():
            x = audio.load_audio(path, sr=2000)
            return audio.spectrogram(x, 2000, win_s=1.0, hop_s=1.0, fmin=20, fmax=600).whiten()

        self.win.busy(True, "decoding recording")
        in_thread(work, self._overview_done, self.win.error)

    def _overview_done(self, sp):
        self.win.busy(False)
        self.overview.show(sp)
        self.overview.vspan("win", self.t0.get_value(), self.t1.get_value())

    def on_overview_click(self, t, f, button, mods):
        half = max(10.0, (self.t1.get_value() - self.t0.get_value()) / 2)
        self.t0.set_value(max(0.0, t - half))
        self.t1.set_value(t + half)
        self.load_zoom()

    def load_zoom(self):
        if not self.path:
            self.win.toast("open a recording first")
            return
        from .. import audio
        t0, t1 = self.t0.get_value(), self.t1.get_value()
        if t1 - t0 < 5:
            self.win.toast("zoom window must be at least 5 s")
            return
        w = self.win_s.get_value()
        kw = dict(win_s=w, hop_s=w / 4, fmin=self.fmin.get_value(), fmax=self.fmax.get_value())
        path = self.path
        self.win.busy(True, "spectrogram")
        in_thread(lambda: audio.brightness(path, t0, t1, **kw), self._zoom_done, self.win.error)

    def _zoom_done(self, sp):
        self.win.busy(False)
        self.zoom.title = f"{sp.t[0]:.0f}–{sp.t[-1]:.0f} s, whitened"
        self.zoom.show(sp)
        self.overview.vspan("win", sp.t[0], sp.t[-1])
        adj = self.sl["t"].get_adjustment()
        self._setting = True
        adj.set_lower(sp.t[0])
        adj.set_upper(sp.t[-1])
        self._setting = False
        self.draw_trace()
        self.draw_model()

    # tracing
    def on_trace(self):
        sp = self.zoom.spec
        if sp is None:
            self.win.toast("show a zoomed spectrogram first")
            return
        from .. import measure
        fs, fe = self.fs.get_value() or None, self.fe.get_value() or None
        in_thread(lambda: measure.trace_ridge(sp, fs, fe), self._traced, self.win.error)

    def _traced(self, res):
        t, f, z = res
        self.tr_t, self.tr_f, self.tr_z = list(map(float, t)), list(map(float, f)), list(map(float, z))
        self.win.toast(f"traced {len(t)} points")
        self.draw_trace()

    def on_zoom_click(self, t, f, button, mods):
        if not self.click_mode.get_active():
            return
        if button == 1:
            i = int(np.searchsorted(self.tr_t, t))
            self.tr_t.insert(i, t)
            self.tr_f.insert(i, f)
            self.tr_z.insert(i, 0.0)
        elif button == 3 and self.tr_t:
            sp = self.zoom.spec
            dt = (np.array(self.tr_t) - t) / max(1e-9, sp.t[-1] - sp.t[0])
            df = (np.array(self.tr_f) - f) / max(1e-9, sp.f[-1] - sp.f[0])
            i = int(np.argmin(dt * dt + df * df))
            for a in (self.tr_t, self.tr_f, self.tr_z):
                a.pop(i)
        self.draw_trace()

    def on_clear(self):
        self.tr_t, self.tr_f, self.tr_z = [], [], []
        self.draw_trace()

    def draw_trace(self):
        self.zoom.layer("trace", [(self.tr_t, self.tr_f, {"color": "white", "lw": 0, "marker": ".", "ms": 3,
                                                          "alpha": 0.9})] if self.tr_t else [])

    def trace_array(self):
        h = int(self.harm.get_value())
        return np.c_[self.tr_t, self.tr_f, np.full(len(self.tr_t), h), self.tr_z]

    def on_save_trace(self):
        if not self.tr_t:
            self.win.toast("nothing traced")
            return
        d = file_dialog("Save trace", [("CSV", ["*.csv"])], self.win.out_dir(), f"{self.ev_name.get_text()}.csv")
        d.save(self.win, None, lambda dlg, res: self._save_trace(dlg, res))

    def _save_trace(self, dlg, res):
        try:
            f = dlg.save_finish(res)
        except GLib.Error:
            return
        self.write_trace(Path(f.get_path()))

    def write_trace(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, self.trace_array(), delimiter=",", fmt="%.4f", header="t,f,harmonic,z")
        self.win.toast(f"trace saved: {path.name}")

    # fitting
    def on_fit(self):
        if len(self.tr_t) < 8:
            self.win.toast("need at least 8 traced points")
            return
        from .. import measure
        t, f = np.array(self.tr_t), np.array(self.tr_f)
        h, c = int(self.harm.get_value()), self.win.c()
        self.fit_label.set_text("fitting…")
        in_thread(lambda: measure.fit_trace(t, f, h, c=c), self._fitted, self.win.error)

    def _fitted(self, res):
        fit, rng, inl = res
        self.fit, self.ranges, self.manual = fit, rng, False
        self._setting = True
        adj = self.sl["t"].get_adjustment()
        adj.set_lower(min(adj.get_lower(), fit.t_cpa - 1))
        adj.set_upper(max(adj.get_upper(), fit.t_cpa + 1))
        for k, v in (("f0", fit.f0), ("v", fit.v), ("d", fit.d), ("t", fit.t_cpa)):
            self.sl[k].set_value(v)
        self._setting = False
        sp, dd = rng["speed_ms"], rng["slant_m"]
        self.fit_label.set_text(
            f"f0 {fit.f0:.2f} Hz · v {fit.v:.1f} m/s ({fit.v / 0.514444:.0f} kt) · d {fit.d:.0f} m · "
            f"CPA {fit.t_cpa:.2f} s · rms {fit.rms:.2f} Hz · {int(inl.sum())}/{len(inl)} points\n"
            f"accepted: speed {sp[0]:.0f}–{sp[1]:.0f} m/s, slant {dd[0]:.0f}–{dd[1]:.0f} m")
        self.show_model.set_active(True)
        self.draw_model()

    def on_slider(self, *_):
        if not self._setting:
            self.manual = True
            self.show_model.set_active(True)
        self.draw_model()

    def harmonics(self) -> list[int]:
        try:
            hs = [int(x) for x in self.harms.get_text().replace(",", " ").split()]
        except ValueError:
            hs = []
        return [h for h in hs if 0 < h < 20] or [int(self.harm.get_value())]

    def draw_model(self):
        sp = self.zoom.spec
        if sp is None or self._setting:
            return
        if not self.show_model.get_active():
            self.zoom.layer("model", [])
            return
        from .. import physics
        v = {k: s.get_value() for k, s in self.sl.items()}
        t = sp.t
        base = physics.linear_flyby(t, v["f0"], v["v"], v["d"], v["t"], self.win.c())
        self.zoom.layer("model", [(t, h * base, {"color": "cyan", "lw": 0.9, "alpha": 0.7})
                                  for h in self.harmonics()])

    def on_add(self):
        from .. import measure
        name = self.ev_name.get_text().strip() or "E"
        hs = self.harmonics()
        if self.fit is not None and self.ranges is not None and not self.manual:
            ev = measure.event_dict(name, self.fit, self.ranges, hs, half_width=self.half.get_value())
        else:
            v = {k: s.get_value() for k, s in self.sl.items()}
            fit = measure.Fit(v["f0"], v["v"], v["d"], v["t"], float("nan"), int(self.harm.get_value()))
            rng = {"speed_ms": (v["v"] * 0.85, v["v"] * 1.15), "slant_m": (v["d"] * 0.6, v["d"] * 1.5)}
            ev = measure.event_dict(name, fit, rng, hs, half_width=self.half.get_value())
        if self.use_trace.get_active():
            if not self.tr_t:
                self.win.toast("no trace to score against")
                return
            p = self.win.out_dir() / f"{name}_trace.csv"
            self.write_trace(p)
            ev["trace"] = self.win.rel(p)
        self.win.add_event(ev)
        n = len(casefile.events(self.win.doc))
        self.ev_name.set_text(f"E{n + 1}")


# --------------------------------------------------------------------------------------------- case

class CasePage(Gtk.Box):
    def __init__(self, win: "Window"):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.win = win
        self.page = Adw.PreferencesPage(vexpand=True)
        self.append(self.page)
        self.groups = []

    def rebuild(self):
        for g in self.groups:
            self.page.remove(g)
        self.groups = []
        doc = self.win.doc
        g = Adw.PreferencesGroup(title="Case", description=str(self.win.case_path or "not saved yet"))
        for key, kind, hint in casefile.TOP:
            g.add(self._entry(doc, key, kind == "str", hint))
        self._add(g)
        for sec, cls in casefile.SECTIONS.items():
            g = Adw.PreferencesGroup(title=sec.capitalize(), description="empty = default")
            for name, is_str, dflt in casefile.fields(cls):
                g.add(self._entry(doc.get(sec), name, is_str, f"default {dflt}" if dflt else "default unset",
                                  section=sec))
            self._add(g)
        g = Adw.PreferencesGroup(title="Events", description="add events from the Audio page")
        g.set_header_suffix(button("Add blank event", self._blank_event))
        for i, ev in enumerate(casefile.events(doc)):
            exp = Adw.ExpanderRow(title=str(ev.get("name", f"event {i + 1}")), subtitle=self._ev_summary(ev))
            rm = Gtk.Button(icon_name="user-trash-symbolic", valign=Gtk.Align.CENTER, tooltip_text="remove event")
            rm.add_css_class("flat")
            rm.connect("clicked", lambda _b, i=i: self._remove(i))
            exp.add_suffix(rm)
            for name, is_str, dflt in casefile.EVENT_FIELDS:
                exp.add_row(self._entry(ev, name, is_str, f"default {dflt}" if dflt else "required"
                                        if name in ("name", "audio_cpa", "f0", "speed_ms", "slant_m") else "unset",
                                        event=i))
            g.add(exp)
        self._add(g)

    def _add(self, g):
        self.page.add(g)
        self.groups.append(g)

    @staticmethod
    def _ev_summary(ev):
        def f(k, fmt):
            v = ev.get(k)
            if v is None:
                return "?"
            v = v.unwrap() if hasattr(v, "unwrap") else v
            return "–".join(fmt.format(x) for x in v) if isinstance(v, list) else fmt.format(v)
        return (f"CPA {f('audio_cpa', '{:.1f}')} s · f0 {f('f0', '{:.2f}')} Hz · speed {f('speed_ms', '{:.0f}')} m/s"
                f" · slant {f('slant_m', '{:.0f}')} m")

    def _entry(self, tbl, key, is_str, hint, section=None, event=None):
        r = Adw.EntryRow(title=f"{key}  ·  {hint}")
        v = None if tbl is None else tbl.get(key)
        if v is not None:
            r.set_text(casefile.to_text(v, is_str))

        def changed(entry):
            try:
                val = casefile.parse(entry.get_text(), is_str)
            except ValueError:
                entry.add_css_class("error")
                return
            entry.remove_css_class("error")
            if event is not None:
                casefile.put(self.win.doc["event"][event], None, key, val)
            else:
                casefile.put(self.win.doc, section, key, val)
            self.win.set_dirty(True)

        r.connect("changed", changed)
        return r

    def _blank_event(self):
        n = len(casefile.events(self.win.doc))
        self.win.add_event({"name": f"E{n + 1}", "audio_cpa": 0.0, "f0": 100.0, "harmonics": [1],
                            "speed_ms": [40.0, 120.0], "slant_m": [300.0, 5000.0]})

    def _remove(self, i):
        casefile.remove_event(self.win.doc, i)
        self.win.set_dirty(True)
        self.rebuild()


# --------------------------------------------------------------------------------------------- run

class RunPage(Gtk.Box):
    def __init__(self, win: "Window"):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin_top=10, margin_bottom=10,
                         margin_start=12, margin_end=12)
        self.win = win
        self.job: Job | None = None
        self.queue: list[tuple[str, list, bool]] = []
        self.cap = Gtk.Entry(text="26G", width_chars=6)
        self.use_cap = Gtk.CheckButton(label="run heavy steps in a memory-capped scope",
                                       active=CAPPED.exists())
        self.use_cap.set_sensitive(CAPPED.exists())
        self.truth = Gtk.CheckButton(label="compare with <out>/truth.json when present", active=True)
        self.noisy = Gtk.CheckButton(label="also score rows that fail the silence rule")
        self.rank = spin(1, 500, 1, 0, 1, 4)
        self.append(row(labeled("memory cap", self.cap), self.use_cap, self.truth, self.noisy))
        self.buttons = [
            button("Fetch ADS-B day", self.fetch, tip="type table + one UTC day of adsb.lol history (gh CLI)"),
            button("Ingest", self.ingest, tip="tar -> parquet for the case's day"),
            button("Search", lambda: self.start([("search", ["search", self.case()], True)]), "suggested-action"),
            button("Score", lambda: self.start([("score", self.score_args(), True)]), "suggested-action"),
            button("Search + score", lambda: self.start([("search", ["search", self.case()], True),
                                                         ("score", self.score_args(), True)])),
            button("Heat map", lambda: self.start([("heatmap", ["heatmap", self.case(), "--rank",
                                                               str(int(self.rank.get_value()))], True)])),
        ]
        self.stop_b = button("Stop", self.stop, "destructive-action")
        self.stop_b.set_sensitive(False)
        self.append(row(*self.buttons, labeled("rank", self.rank), self.stop_b, spacing=6))
        self.status = Gtk.Label(label="idle", xalign=0)
        self.append(self.status)
        self.view = Gtk.TextView(editable=False, monospace=True, cursor_visible=False, vexpand=True)
        self.view.add_css_class("log")
        self.buf = self.view.get_buffer()
        sw = Gtk.ScrolledWindow(vexpand=True)
        sw.set_child(self.view)
        self.append(sw)

    def case(self) -> str:
        return str(self.win.case_path)

    def score_args(self):
        a = ["score", self.case()]
        t = self.win.out_dir() / "truth.json"
        if self.truth.get_active() and t.exists():
            a += ["--truth", str(t)]
        if self.noisy.get_active():
            a.append("--keep-noisy")
        return a

    def fetch(self):
        day = str(casefile.get(self.win.doc, None, "day") or "")
        if not day:
            self.win.toast("set the case's day first")
            return
        steps = []
        if not (self.win.root() / "data/ref/icao_aircraft_types2.js.gz").exists():
            steps.append(("fetch-types", ["fetch-types"], False))
        steps.append(("fetch-adsb", ["fetch-adsb", day], False))
        self.start(steps)

    def ingest(self):
        day = str(casefile.get(self.win.doc, None, "day") or "")
        if not day:
            self.win.toast("set the case's day first")
            return
        self.start([("ingest", ["ingest", day, "--out", str(self.win.resolve("adsb", f"data/adsb/{day}"))], True)])

    def log(self, line: str):
        end = self.buf.get_end_iter()
        self.buf.insert(end, line + "\n")
        self.view.scroll_to_mark(self.buf.get_insert(), 0.0, False, 0, 0)
        mark = self.buf.create_mark(None, self.buf.get_end_iter(), False)
        self.view.scroll_to_mark(mark, 0.0, False, 0, 0)
        self.buf.delete_mark(mark)

    def start(self, steps):
        if self.job is not None:
            self.win.toast("a job is already running")
            return
        if not self.win.ensure_saved(lambda: self.start(steps)):
            return
        self.queue = list(steps)
        self._next()

    def _next(self):
        if not self.queue:
            self._idle("done")
            return
        name, args, heavy = self.queue.pop(0)
        argv = dopploc_argv(*args)
        if heavy and self.use_cap.get_active():
            argv = capped(argv, self.cap.get_text().strip() or None)
        self.log(f"$ {' '.join(argv)}")
        self.status.set_text(f"running {name}…")
        for b in self.buttons:
            b.set_sensitive(False)
        self.stop_b.set_sensitive(True)
        self.current = name
        try:
            self.job = Job(argv, self.win.root(), self.log, self._exited)
        except GLib.Error as e:
            self.job = None
            self.queue = []
            self._idle(f"could not start {name}: {e.message}")

    def _exited(self, rc):
        name = self.current
        self.job = None
        self.log(f"[{name} exited with {rc}]")
        if rc != 0:
            self.queue = []
            self._idle(f"{name} failed (exit {rc})")
            return
        if name in ("score", "heatmap"):
            self.win.results.reload()
        self._next()

    def _idle(self, msg):
        self.status.set_text(msg)
        for b in self.buttons:
            b.set_sensitive(True)
        self.stop_b.set_sensitive(False)
        self.win.toast(msg)

    def stop(self):
        self.queue = []
        if self.job:
            self.job.stop()


# --------------------------------------------------------------------------------------------- results

COLS = (("#", "rank", "{}"), ("score", "score", "{:.3f}"), ("S1", "S1", "{:.3f}"), ("S2", "S2", "{:.3f}"),
        ("lat", "lat", "{:.5f}"), ("lon", "lon", "{:.5f}"), ("km to truth", "dist_km", "{:.2f}"),
        ("event-1 flight", "f1", "{}"), ("event-2 flight", "f2", "{}"))


class ResultsPage(Gtk.Box):
    def __init__(self, win: "Window"):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin_top=8, margin_bottom=8,
                         margin_start=10, margin_end=10)
        self.win = win
        self.rows: list[dict] = []
        self.truth = None
        self.summary = Gtk.Label(label="no results loaded", xalign=0, hexpand=True, wrap=True)
        self.append(row(button("Reload", self.reload), self.summary,
                        button("Open output folder", self.open_folder)))
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL, position=640, vexpand=True)
        self.append(paned)
        self.cv = Gtk.ColumnView(show_row_separators=True, reorderable=False)
        self.cv.add_css_class("data-table")
        sw = Gtk.ScrolledWindow(hexpand=True)
        sw.set_child(self.cv)
        paned.set_start_child(sw)

        self.tabs = Gtk.Notebook()
        paned.set_end_child(self.tabs)
        self.curves = Gtk.Picture(can_shrink=True, content_fit=Gtk.ContentFit.CONTAIN, hexpand=True, vexpand=True)
        self.curves_label = Gtk.Label(label="select a row", xalign=0)
        cbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        cbox.append(self.curves_label)
        cbox.append(self.curves)
        self.tabs.append_page(cbox, Gtk.Label(label="Curves"))
        if Shumate is not None:
            self.map = Shumate.SimpleMap(hexpand=True, vexpand=True)
            reg = Shumate.MapSourceRegistry.new_with_defaults()
            self.map.set_map_source(reg.get_by_id(Shumate.MAP_SOURCE_OSM_MAPNIK))
            self.markers = Shumate.MarkerLayer.new(self.map.get_viewport())
            self.map.add_overlay_layer(self.markers)
            self.sel_marker = None
            self.tabs.append_page(self.map, Gtk.Label(label="Map"))
        else:
            self.map = None
            self.tabs.append_page(Gtk.Label(label="libshumate (GObject introspection) not installed"),
                                  Gtk.Label(label="Map"))
        self.scatter = ScatterView()
        self.tabs.append_page(self.scatter.canvas, Gtk.Label(label="Scores"))
        self.heat = Gtk.Picture(can_shrink=True, content_fit=Gtk.ContentFit.CONTAIN)
        self.heat_label = Gtk.Label(label="", xalign=0)
        hbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        hbox.append(self.heat_label)
        hbox.append(self.heat)
        self.tabs.append_page(hbox, Gtk.Label(label="Heat map"))
        self.text = Gtk.TextView(editable=False, monospace=True, cursor_visible=False)
        tsw = Gtk.ScrolledWindow()
        tsw.set_child(self.text)
        self.tabs.append_page(tsw, Gtk.Label(label="Summary"))
        self.sel = None

    def open_folder(self):
        out = self.win.out_dir()
        if out.is_dir():
            Gtk.FileLauncher.new(Gio.File.new_for_path(str(out))).launch(self.win, None, None)

    def reload(self):
        import polars as pl
        out = self.win.out_dir()
        csv = out / "best_per_pair.csv"
        for c in list(self.cv.get_columns()):
            self.cv.remove_column(c)
        self.rows = []
        if not csv.exists():
            self.summary.set_text(f"no results in {out}")
            self.cv.set_model(None)
            return
        best = pl.read_csv(csv, infer_schema_length=10000)
        tp = out / "truth.json"
        self.truth = json.loads(tp.read_text()) if tp.exists() else None
        if self.truth:
            from ..report import haversine
            d = haversine(best["lat"].to_numpy(), best["lon"].to_numpy(), self.truth["lat"], self.truth["lon"])
            best = best.with_columns(pl.Series("dist_km", d / 1000.0))
        self.rows = [dict(r, rank=i + 1) for i, r in enumerate(best.head(500).iter_rows(named=True))]
        msg = f"{len(best):,} flight pairs in {out}"
        if self.truth:
            within = np.flatnonzero(best["dist_km"].to_numpy() < 0.3)
            msg += (f" · truth: best pair within 300 m ranks {int(within[0]) + 1}" if len(within)
                    else " · no pair within 300 m of truth")
            msg += f" · top-1 is {best['dist_km'][0] * 1000:.0f} m from truth"
        self.summary.set_text(msg)
        for title, key, fmt in COLS:
            if key == "dist_km" and not self.truth or key == "f2" and "f2" not in best.columns:
                continue
            fac = Gtk.SignalListItemFactory()
            fac.connect("setup", lambda _f, item: item.set_child(Gtk.Label(xalign=0)))
            fac.connect("bind", lambda _f, item, key=key, fmt=fmt: item.get_child().set_text(
                self._fmt(self.rows[int(item.get_item().get_string())].get(key), fmt)))
            col = Gtk.ColumnViewColumn(title=title, factory=fac)
            col.set_resizable(True)
            if key in ("f1", "f2"):
                col.set_expand(True)
            self.cv.append_column(col)
        model = Gtk.SingleSelection(model=Gtk.StringList.new([str(i) for i in range(len(self.rows))]))
        model.connect("notify::selected", lambda m, _p: self.select(m.get_selected()))
        self.cv.set_model(model)
        s2 = best["S2"].to_numpy() if "S2" in best.columns else np.zeros(len(best))
        near = np.flatnonzero(best["dist_km"].to_numpy() < 0.3) if self.truth else None
        self.scatter.scatter_scores(best["S1"].to_numpy(), s2, None, near)
        self._markers()
        summ = out / "summary.md"
        self.text.get_buffer().set_text(summ.read_text() if summ.exists() else "")
        if self.rows:
            self.select(0)  # a new SingleSelection starts on row 0 without notifying

    @staticmethod
    def _fmt(v, fmt):
        if v is None:
            return ""
        try:
            return fmt.format(v)
        except (ValueError, TypeError):
            return str(v)

    def _pin(self, lat, lon, text, cls=None):
        m = Shumate.Marker()
        lab = Gtk.Label(label=text)
        lab.add_css_class("pin")
        if cls:
            lab.add_css_class(cls)
        m.set_child(lab)
        m.set_location(lat, lon)
        self.markers.add_marker(m)
        return m

    def _markers(self):
        if self.map is None:
            return
        self.markers.remove_all()
        self.sel_marker = None
        for r in self.rows[:25][::-1]:
            self._pin(r["lat"], r["lon"], str(r["rank"]))
        if self.truth:
            self._pin(self.truth["lat"], self.truth["lon"], "truth", "truth")

    def select(self, idx):
        """Show row ``idx`` (0-based); keeps the table selection in step."""
        if idx is None or idx < 0 or idx >= len(self.rows):
            return
        model = self.cv.get_model()
        if model is not None and model.get_selected() != idx:
            model.set_selected(idx)  # notify::selected calls back here
            self.cv.scroll_to(idx, None, Gtk.ListScrollFlags.NONE, None)
            return
        r = self.rows[idx]
        self.sel = idx
        if self.map is not None:
            if self.sel_marker is not None:
                self.markers.remove_marker(self.sel_marker)
            self.sel_marker = self._pin(r["lat"], r["lon"], f"#{r['rank']}", "sel")
            self.map.get_viewport().set_zoom_level(13)
            self.map.get_map().center_on(r["lat"], r["lon"])
        self._curves(r)
        out = self.win.out_dir()
        hm = out / f"heatmap_rank{r['rank']}.png"
        if hm.exists():
            self.heat.set_filename(str(hm))
            self.heat_label.set_text(hm.name)
        else:
            self.heat.set_paintable(None)
            self.heat_label.set_text(f"no heat map for rank {r['rank']} yet (Run page: Heat map, rank "
                                     f"{r['rank']})")

    def _curves(self, r):
        out = self.win.out_dir()
        csv = out / "best_per_pair.csv"
        rank = r["rank"]
        for p in (out / f"rank{rank}.png", out / "gui" / f"rank{rank}.png"):
            if p.exists() and p.stat().st_mtime >= csv.stat().st_mtime:
                self.curves.set_filename(str(p))
                self.curves_label.set_text(f"rank {rank}: modelled curves (cyan) over the event spectrograms")
                return
        if not self.win.case_path:
            return
        self.curves.set_paintable(None)
        self.curves_label.set_text(f"rendering rank {rank}…")
        from .. import config, report
        case_path = self.win.case_path
        p = out / "gui" / f"rank{rank}.png"
        row_ = {k: v for k, v in r.items() if k not in ("rank", "dist_km")}

        def work():
            p.parent.mkdir(parents=True, exist_ok=True)
            report.overlay(config.load(case_path), row_, p, rank)
            return p

        def done(path):
            if self.sel is not None and self.rows[self.sel]["rank"] == rank:
                self.curves.set_filename(str(path))
                self.curves_label.set_text(f"rank {rank}: modelled curves (cyan) over the event spectrograms")

        in_thread(work, done, self.win.error)


# --------------------------------------------------------------------------------------------- window

class Window(Adw.ApplicationWindow):
    def __init__(self, app, case_path: Path | None = None, page: str | None = None):
        super().__init__(application=app, title="dopploc", default_width=1360, default_height=940)
        self.doc = casefile.new_case()
        self.case_path: Path | None = None
        self.dirty = False
        self._busy = 0
        self.toasts = Adw.ToastOverlay()
        self.stack = Adw.ViewStack()
        self.toasts.set_child(self.stack)
        header = Adw.HeaderBar()
        header.set_title_widget(Adw.ViewSwitcher(stack=self.stack, policy=Adw.ViewSwitcherPolicy.WIDE))
        for icon, tip, cb in (("document-open-symbolic", "Open case (Ctrl+O)", self.open_dialog),
                              ("document-new-symbolic", "New case", self.new_case)):
            b = Gtk.Button(icon_name=icon, tooltip_text=tip)
            b.connect("clicked", lambda *_, cb=cb: cb())
            header.pack_start(b)
        for icon, tip, cb in (("document-save-as-symbolic", "Save case as…", self.save_as),
                              ("document-save-symbolic", "Save case (Ctrl+S)", self.save)):
            b = Gtk.Button(icon_name=icon, tooltip_text=tip)
            b.connect("clicked", lambda *_, cb=cb: cb())
            header.pack_end(b)
        self.spinner = Adw.Spinner(visible=False)
        header.pack_end(self.spinner)
        tv = Adw.ToolbarView()
        tv.add_top_bar(header)
        tv.set_content(self.toasts)
        self.set_content(tv)

        self.audio = AudioPage(self)
        self.case = CasePage(self)
        self.run = RunPage(self)
        self.results = ResultsPage(self)
        self.stack.add_titled_with_icon(self.audio, "audio", "Audio", "audio-x-generic-symbolic")
        self.stack.add_titled_with_icon(self.case, "case", "Case", "document-properties-symbolic")
        self.stack.add_titled_with_icon(self.run, "run", "Run", "media-playback-start-symbolic")
        self.stack.add_titled_with_icon(self.results, "results", "Results", "find-location-symbolic")

        for name, cb, accel in (("open", self.open_dialog, "<Ctrl>o"), ("save", self.save, "<Ctrl>s")):
            a = Gio.SimpleAction.new(name, None)
            a.connect("activate", lambda *_, cb=cb: cb())
            self.add_action(a)
            app.set_accels_for_action(f"win.{name}", [accel])
        self.connect("close-request", self._close_request)
        self.case.rebuild()
        if case_path:
            self.open_case(case_path)
        if page:
            self.stack.set_visible_child_name(page)

    # paths
    def root(self) -> Path:
        r = casefile.get(self.doc, None, "root")
        if r:
            return Path(str(r)).expanduser().resolve()
        if self.case_path:
            return self.case_path.parent.parent
        return Path.cwd()

    def resolve(self, key, default="") -> Path:
        v = Path(str(casefile.get(self.doc, None, key) or default)).expanduser()
        return v if v.is_absolute() else self.root() / v

    def rel(self, p: Path) -> str:
        try:
            return str(Path(p).resolve().relative_to(self.root()))
        except ValueError:
            return str(p)

    def out_dir(self) -> Path:
        stem = self.case_path.stem if self.case_path else "untitled"
        return self.resolve("out", f"out/{stem}")

    def c(self) -> float:
        try:
            return float(casefile.get(self.doc, None, "c") or 343.0)
        except (TypeError, ValueError):
            return 343.0

    # feedback
    def toast(self, msg: str):
        self.toasts.add_toast(Adw.Toast(title=GLib.markup_escape_text(msg), timeout=4))

    def error(self, e: Exception):
        self.busy(False)
        self.toast(f"error: {e}")
        print(f"dopploc-gui: {type(e).__name__}: {e}", file=sys.stderr)

    def busy(self, on: bool, what: str = ""):
        self._busy = max(0, self._busy + (1 if on else -1))
        self.spinner.set_visible(self._busy > 0)
        self.spinner.set_tooltip_text(what)

    # case file
    def set_dirty(self, d: bool):
        self.dirty = d
        name = self.case_path.name if self.case_path else "unsaved case"
        self.set_title(f"dopploc — {name}{' •' if d else ''}")

    def open_dialog(self):
        if self.dirty:
            self.toast("unsaved changes: save first (Ctrl+S)")
            return
        d = file_dialog("Open case", [("Case (*.toml)", ["*.toml"])], self.root() / "cases")
        d.open(self, None, self._opened)

    def _opened(self, dlg, res):
        try:
            f = dlg.open_finish(res)
        except GLib.Error:
            return
        self.open_case(Path(f.get_path()))

    def open_case(self, path: Path):
        path = Path(path).resolve()
        try:
            self.doc = casefile.load(path)
        except Exception as e:
            self.error(e)
            return
        self.case_path = path
        self.set_dirty(False)
        self.case.rebuild()
        self.results.reload()
        a = casefile.get(self.doc, None, "audio")
        if a and self.resolve("audio").exists():
            focus = None
            evs = casefile.events(self.doc)
            if evs:
                e = evs[0]
                hw = float(e.get("half_width_s", 60.0))
                hs = list(e.get("harmonics", [1]))
                f0 = float(e["f0"])
                focus = (float(e["audio_cpa"]) - hw, float(e["audio_cpa"]) + hw,
                         max(15.0, 0.6 * f0 * min(hs)), 1.6 * f0 * max(hs))
            self.audio.set_audio(self.resolve("audio"), focus)

    def new_case(self):
        if self.dirty:
            self.toast("unsaved changes: save first (Ctrl+S)")
            return
        self.doc = casefile.new_case(audio=self.rel(self.audio.path) if self.audio.path else "")
        self.case_path = None
        self.set_dirty(True)
        self.case.rebuild()
        self.results.reload()
        self.stack.set_visible_child_name("case")

    def save(self, then=None):
        if self.case_path is None:
            self.save_as(then)
            return
        try:
            casefile.save(self.doc, self.case_path)
        except Exception as e:
            self.error(e)
            return
        self.set_dirty(False)
        self.toast(f"saved {self.case_path.name}")
        if then:
            then()

    def save_as(self, then=None):
        name = f"{casefile.get(self.doc, None, 'name') or 'case'}.toml"
        d = file_dialog("Save case", [("Case (*.toml)", ["*.toml"])], self.root() / "cases", name)

        def done(dlg, res):
            try:
                f = dlg.save_finish(res)
            except GLib.Error:
                return
            self.case_path = Path(f.get_path()).resolve()
            self.case.rebuild()
            self.save(then)

        d.save(self, None, done)

    def ensure_saved(self, then) -> bool:
        """True when the case is saved; otherwise saves (asking for a name if needed), then calls ``then``."""
        if self.case_path and not self.dirty:
            return True
        self.save(then)
        return False

    def set_value(self, section, key, value):
        casefile.put(self.doc, section, key, value)
        self.set_dirty(True)
        self.case.rebuild()

    def add_event(self, ev: dict):
        casefile.add_event(self.doc, ev)
        self.set_dirty(True)
        self.case.rebuild()
        self.toast(f"event {ev['name']} added to the case (not saved yet)")

    def _close_request(self, *_):
        if not self.dirty:
            return False
        dlg = Adw.AlertDialog(heading="Save changes?", body="The case has unsaved changes.")
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("discard", "Discard")
        dlg.add_response("save", "Save")
        dlg.set_response_appearance("discard", Adw.ResponseAppearance.DESTRUCTIVE)
        dlg.set_response_appearance("save", Adw.ResponseAppearance.SUGGESTED)
        dlg.set_default_response("save")

        def resp(d, r):
            if r == "discard":
                self.dirty = False
                self.close()
            elif r == "save":
                self.save(self.close)

        dlg.connect("response", resp)
        dlg.present(self)
        return True


class App(Adw.Application):
    def __init__(self, case: Path | None, page: str | None):
        super().__init__(application_id="io.github.digitalis_p.dopploc", flags=Gio.ApplicationFlags.NON_UNIQUE)
        self.case = case
        self.page = page
        self.win = None

    def do_startup(self):
        Adw.Application.do_startup(self)
        css = Gtk.CssProvider()
        css.load_from_data(CSS, -1)
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), css,
                                                  Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

    def do_activate(self):
        if self.win is None:
            self.win = Window(self, self.case, self.page)
        self.win.present()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dopploc-gui", description="GTK front end for dopploc")
    ap.add_argument("case", nargs="?", type=Path, help="case TOML to open")
    ap.add_argument("--page", choices=("audio", "case", "run", "results"))
    a = ap.parse_args(argv)
    return App(a.case, a.page).run([sys.argv[0]])


if __name__ == "__main__":
    sys.exit(main())
