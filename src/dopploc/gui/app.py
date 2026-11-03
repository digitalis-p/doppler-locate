"""dopploc-gui: GTK4 / libadwaita front end for the dopploc pipeline.

Pages
  Audio    recording overview, automatic flyover detection, zoomed spectrogram, ridge tracing
           (automatic or by clicking), straight-line flyby fit with live model sliders, "add as
           event". Recordings open from a file or a web video link (yt-dlp).
  Case     every case setting as typed rows; the TOML file is edited in place (comments kept,
           .bak copy on the first save over a hand-written file).
  Run      the pipeline steps as memory-capped child processes with status and a live log.
  Results  ranked flight pairs, modelled curves over both events, OpenStreetMap view, score
           scatter, heat map, summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Adw", "1")
try:
    gi.require_version("Shumate", "1.0")
    from gi.repository import Shumate
except (ValueError, ImportError):
    Shumate = None
import gi.repository.GObject  # noqa: F401  (BindingFlags)
from gi.repository import Adw, Gdk, Gio, GLib, Gtk, Pango

import numpy as np

from . import casefile, plots
from .plots import ACCENT, PairView, ScatterView, SpecView
from .runner import CAPPED, PKG_ROOT, Job, capped, dopploc_argv

APP_ID = "io.github.digitalis_p.dopploc"
STATE = Path(GLib.get_user_state_dir()) / "dopploc" / "gui.json"

CSS = b"""
.pane { padding: 12px; }
.plot-card { padding: 6px 8px 8px 8px; }
.card-title { font-weight: 700; }
.pill { border-radius: 999px; padding: 2px 10px; font-weight: 600; font-size: 9pt;
        background: alpha(@accent_bg_color, 0.18); color: @accent_color; }
.pill.good { background: alpha(@success_bg_color, 0.20); color: @success_color; }
.pill.warn { background: alpha(@warning_bg_color, 0.20); color: @warning_color; }
.dot { font-size: 14pt; }
.dot.high { color: #57e389; }
.dot.medium { color: #f8e45c; }
.dot.low { color: #9a9996; }
.pin { background: #c01c28; color: white; border-radius: 9px; padding: 0 6px; font-size: 8pt; font-weight: bold; }
.pin.truth { background: #26a269; }
.pin.sel { background: #1c71d8; font-size: 10pt; }
.log { font-family: monospace; font-size: 9pt; padding: 8px; }
.mono { font-family: monospace; }
"""

LABELS = {
    "name": "Case name", "day": "UTC day of the ADS-B data", "audio": "Recording", "adsb": "ADS-B data folder",
    "cache": "Cache folder", "out": "Output folder", "c": "Speed of sound (m/s)", "atmosphere": "Atmosphere",
    "wind_ms": "Fixed wind [east, north] (m/s)", "altitude": "Altitude datum", "root": "Project root",
    "bbox": "Search box [lon0, lat0, lon1, lat1]", "utc_window": "Event-1 UTC window [from, to]",
    "grid_m": "Grid spacing (m)", "tile_km": "Tile size (km)", "sample_dt": "Track sampling (s)",
    "offset_tol_s": "Offset tolerance (s)", "exclude_engines": "Excluded engine classes",
    "max_alt_ft": "Maximum altitude (ft)", "speed_margin_ms": "Speed margin (m/s)", "workers": "CPU workers",
    "elevation_lattice": "Cached elevation lattice", "gpu": "GPU search", "enabled": "Silence rule",
    "mode": "Mode", "ratio": "Distance ratio", "radius_m": "Radius (m)", "margin_s": "Margin (s)",
    "jets_below_ft": "Jets below (ft)", "jet_radius_m": "Jet radius (m)", "win_s": "STFT window (s)",
    "hop_s": "STFT hop (s)", "dt_sweep": "Clock-shift sweep [from, to, step] (s)",
    "f0_sweep_frac": "f0 sweep (fraction)", "f0_step_hz": "f0 step (Hz)", "refine_m": "Refinement spacing (m)",
    "model_dt": "Model time step (s)", "freq_tol_bins": "Frequency tolerance (bins)",
    "line_suppress_s": "Steady-tone suppression (s)", "trace_sigma_hz": "Trace tolerance (Hz)",
    "dem_res": "DEM resolution (m)", "audio_cpa": "CPA time in the recording (s)", "f0": "Fundamental f0 (Hz)",
    "speed_ms": "Speed range [min, max] (m/s)", "slant_m": "CPA distance range [min, max] (m)",
    "harmonics": "Harmonics", "half_width_s": "Scoring half width (s)", "fband": "Scoring band [lo, hi] (Hz)",
    "trace": "Trace CSV (faint curve)",
}
ENUMS = {(None, "atmosphere"): ["gfs", "asos", "fixed", "none"], (None, "altitude"): ["auto", "geom", "baro"],
         ("silence", "mode"): ["cpa", "samples"], ("score", "mode"): ["joint", "locked"],
         ("score", "dem_res"): [90, 30]}
SECTION_TITLES = {"search": ("Search", "Candidate grid, filters and tolerances"),
                  "silence": ("Silence rule", "Reject places that would have heard other aircraft in between"),
                  "score": ("Scoring", "Spectrogram, sweeps and refinement")}


def in_thread(fn, done, fail):
    """Run fn() off the main loop, then done(result) or fail(exception) back on it."""
    def run():
        try:
            r = fn()
        except Exception as e:  # shown to the user by fail(); bind e, the except block unbinds it
            GLib.idle_add(lambda e=e: fail(e) and False)
            return
        GLib.idle_add(lambda: done(r) and False)
    threading.Thread(target=run, daemon=True).start()


def button(label_=None, cb=None, style=None, tip=None, icon=None):
    b = Gtk.Button(valign=Gtk.Align.CENTER)
    if icon and label_:
        b.set_child(Adw.ButtonContent(icon_name=icon, label=label_))
    elif icon:
        b.set_icon_name(icon)
    else:
        b.set_label(label_)
    if cb:
        b.connect("clicked", lambda *_: cb())
    for s in ([style] if isinstance(style, str) else style or []):
        b.add_css_class(s)
    if tip:
        b.set_tooltip_text(tip)
    return b


def hbox(*widgets, spacing=6, css=None):
    b = Gtk.Box(spacing=spacing)
    for w in widgets:
        b.append(w)
    if css:
        b.add_css_class(css)
    return b


def vbox(*widgets, spacing=12, css=None, **kw):
    b = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=spacing, **kw)
    for w in widgets:
        b.append(w)
    for c in ([css] if isinstance(css, str) else css or []):
        b.add_css_class(c)
    return b


def label(text="", css=None, xalign=0.0, **kw):
    lab = Gtk.Label(label=text, xalign=xalign, **kw)
    for c in ([css] if isinstance(css, str) else css or []):
        lab.add_css_class(c)
    return lab


def pill(text, kind=None):
    return label(text, ["pill"] + ([kind] if kind else []))


def spin(lo, hi, step, digits, value, width=6):
    s = Gtk.SpinButton.new_with_range(lo, hi, step)
    s.set_digits(digits)
    s.set_value(value)
    s.set_width_chars(width)
    s.set_valign(Gtk.Align.CENTER)
    return s


def spin_row(title, lo, hi, step, digits, value, subtitle=None):
    r = Adw.SpinRow.new_with_range(lo, hi, step)
    r.set_title(title)
    r.set_digits(digits)
    r.set_value(value)
    if subtitle:
        r.set_subtitle(subtitle)
    return r


class Card(Gtk.Box):
    """A rounded card: title / subtitle / suffix widgets on top, content below."""

    def __init__(self, *children, title=None, subtitle=None, suffix=()):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.add_css_class("card")
        self.add_css_class("plot-card")
        head = Gtk.Box(spacing=8, margin_start=6, margin_top=2)
        tb = vbox(spacing=0, hexpand=True, valign=Gtk.Align.CENTER)
        tb.append(label(title or "", "card-title"))
        self.subtitle = label(subtitle or "", ["dim-label", "caption"], ellipsize=Pango.EllipsizeMode.MIDDLE)
        tb.append(self.subtitle)
        head.append(tb)
        for w in suffix:
            head.append(w)
        self.append(head)
        for c in children:
            self.append(c)


def file_dialog(title, patterns=None, folder: Path | None = None, name: str | None = None):
    d = Gtk.FileDialog(title=title, modal=True)
    if patterns:
        store = Gio.ListStore.new(Gtk.FileFilter)
        for lab, pats in patterns:
            f = Gtk.FileFilter()
            f.set_name(lab)
            for p in pats:
                f.add_pattern(p)
            store.append(f)
        d.set_filters(store)
    if folder and folder.is_dir():
        d.set_initial_folder(Gio.File.new_for_path(str(folder)))
    if name:
        d.set_initial_name(name)
    return d


def fmt_t(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m)}:{s:04.1f}"


def fmt_path(p) -> str:
    """A path for display, with the home directory shown as ~ (keeps user names off screenshots)."""
    p, home = Path(p), Path.home()
    return f"~/{p.relative_to(home)}" if p.is_relative_to(home) else str(p)


# --------------------------------------------------------------------------------------------- audio

class AudioPage(Gtk.Box):
    def __init__(self, win: "Window"):
        super().__init__()
        self.win = win
        self.path: Path | None = None
        self.tr_t: list[float] = []
        self.tr_f: list[float] = []
        self.tr_z: list[float] = []
        self.fit = None
        self.ranges = None
        self.manual = False
        self._setting = False
        self.detections = []

        # plots column
        self.overview = SpecView(height=150)
        self.overview.on_click(self.on_overview_click)
        self.split = Adw.OverlaySplitView(sidebar_position=Gtk.PackType.END, min_sidebar_width=330,
                                          max_sidebar_width=420, sidebar_width_fraction=0.3, hexpand=True)
        self.append(self.split)
        side_toggle = Gtk.ToggleButton(icon_name="view-dual-symbolic", valign=Gtk.Align.CENTER,
                                       tooltip_text="Show or hide the inspector", active=True)
        side_toggle.add_css_class("flat")
        side_toggle.bind_property("active", self.split, "show-sidebar",
                                  gi.repository.GObject.BindingFlags.BIDIRECTIONAL)
        self.ov_card = Card(self.overview.canvas, title="Recording", subtitle="no recording open",
                            suffix=[button(icon="document-open-symbolic", cb=self.on_open, style="flat",
                                           tip="Open an audio or video file"),
                                    button(icon="insert-link-symbolic", cb=self.on_link, style="flat",
                                           tip="Open a web video link (YouTube, ...): audio via yt-dlp"),
                                    button("Detect flyovers", self.on_detect, style="suggested-action",
                                           tip="Find aircraft Doppler curves in the whole recording"),
                                    side_toggle])
        self.t0 = spin(0, 1e6, 1, 1, 0, 8)
        self.t1 = spin(0, 1e6, 1, 1, 120, 8)
        self.fmin = spin(1, 4000, 5, 0, 30, 8)
        self.fmax = spin(1, 4000, 5, 0, 600, 8)
        self.win_s = spin(0.1, 4, 0.1, 2, 1.0, 8)
        grid = Gtk.Grid(row_spacing=6, column_spacing=10, margin_top=8, margin_bottom=8, margin_start=8,
                        margin_end=8)
        for i, (lab, w) in enumerate((("From (s)", self.t0), ("To (s)", self.t1), ("Lowest (Hz)", self.fmin),
                                      ("Highest (Hz)", self.fmax), ("STFT window (s)", self.win_s))):
            grid.attach(label(lab), 0, i, 1, 1)
            grid.attach(w, 1, i, 1, 1)
        grid.attach(button("Show", self.load_zoom, style="suggested-action"), 0, 5, 2, 1)
        view_menu = Gtk.MenuButton(icon_name="document-properties-symbolic", tooltip_text="Zoom window",
                                   popover=Gtk.Popover(child=grid), valign=Gtk.Align.CENTER)
        view_menu.add_css_class("flat")
        self.zoom = SpecView(height=330)
        self.zoom.on_click(self.on_zoom_click)
        self.zoom_card = Card(self.zoom.canvas, title="Zoom", subtitle="click the overview to zoom there",
                              suffix=[view_menu,
                                      button(icon="view-refresh-symbolic", cb=self.load_zoom, style="flat",
                                             tip="Redraw the zoomed spectrogram")])
        self.zoom_card.set_vexpand(True)
        self.split.set_content(vbox(self.ov_card, self.zoom_card, css="pane"))

        # inspector column
        side = vbox(spacing=18, css="pane")
        self.split.set_sidebar(Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER, child=side))

        self.det_group = Adw.PreferencesGroup(title="Detected flyovers",
                                              description="Detect flyovers lists the aircraft Doppler curves")
        self.det_list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        self.det_list.add_css_class("boxed-list")
        self.det_list.set_placeholder(label("nothing detected yet", ["dim-label"], xalign=0.5, margin_top=12,
                                            margin_bottom=12))
        self.det_list.connect("row-activated", lambda _l, r: self.use_detection(r.idx))
        self.det_group.add(self.det_list)
        self.show_weak = Gtk.Switch(valign=Gtk.Align.CENTER, tooltip_text="Also list weak detections")
        self.show_weak.connect("notify::active", lambda *_: (self._fill_detections(), self._draw_detections()))
        self.det_group.set_header_suffix(hbox(label("weak", ["dim-label", "caption"]), self.show_weak))
        side.append(self.det_group)

        g = Adw.PreferencesGroup(title="Trace", description="Follow one curve automatically, or click points "
                                                            "(right click removes one)")
        self.harm = spin_row("Harmonic of the traced curve", 1, 12, 1, 0, 1)
        self.fs = spin_row("Start frequency (Hz)", 0, 4000, 1, 1, 0, "pins the left end; 0 = free")
        self.fe = spin_row("End frequency (Hz)", 0, 4000, 1, 1, 0, "pins the right end; 0 = free")
        for r in (self.harm, self.fs, self.fe):
            g.add(r)
        self.click_mode = Gtk.ToggleButton(icon_name="input-mouse-symbolic", tooltip_text="Click to add points")
        acts = hbox(button("Trace ridge", self.on_trace), self.click_mode,
                    button(icon="edit-clear-symbolic", cb=self.on_clear, tip="Clear the trace"),
                    button(icon="document-save-symbolic", cb=self.on_save_trace, tip="Save the trace as CSV"),
                    spacing=0, css="linked")
        acts.set_halign(Gtk.Align.START)
        acts.set_margin_top(10)
        g.add(acts)
        side.append(g)

        g = Adw.PreferencesGroup(title="Flyby model", description="Straight, constant-speed pass")
        g.set_header_suffix(button("Fit", self.on_fit, style="suggested-action", tip="Fit the traced curve"))
        self.sl = {}
        for key, title, lo, hi, step, val, dig in (
                ("f0", "f0 (Hz)", 5, 2000, 0.05, 100, 2), ("v", "Speed (m/s)", 10, 260, 0.5, 80, 1),
                ("d", "CPA distance (m)", 50, 20000, 10, 2000, 0), ("t", "CPA time (s)", 0, 1e6, 0.05, 60, 2)):
            adj = Gtk.Adjustment(value=val, lower=lo, upper=hi, step_increment=step, page_increment=step * 20)
            sc = Gtk.Scale(adjustment=adj, hexpand=True, digits=dig, draw_value=True,
                           value_pos=Gtk.PositionType.LEFT, width_request=200)
            sc.connect("value-changed", self.on_slider)
            row = Adw.ActionRow(title=title)
            row.add_suffix(sc)
            g.add(row)
            self.sl[key] = sc
        self.harms = Adw.EntryRow(title="Harmonics drawn and stored", text="1")
        self.harms.connect("changed", lambda *_: self.draw_model())
        self.show_model = Adw.SwitchRow(title="Show model curve")
        self.show_model.connect("notify::active", lambda *_: self.draw_model())
        self.fit_row = Adw.ActionRow(title="Fit", subtitle="trace a curve, then Fit")
        self.fit_row.add_css_class("property")
        self.fit_row.set_subtitle_selectable(True)
        for r in (self.harms, self.show_model, self.fit_row):
            g.add(r)
        side.append(g)

        g = Adw.PreferencesGroup(title="Add to case")
        self.ev_name = Adw.EntryRow(title="Event name", text="E1")
        self.half = spin_row("Scoring half width (s)", 10, 300, 5, 0, 60)
        self.use_trace = Adw.SwitchRow(title="Score against the trace", subtitle="for faint curves")
        add = Adw.ButtonRow(title="Add event to case", start_icon_name="list-add-symbolic")
        add.add_css_class("suggested-action")
        add.connect("activated", lambda *_: self.on_add())
        for r in (self.ev_name, self.half, self.use_trace, add):
            g.add(r)
        side.append(g)
        self.overview.blank("open a recording or a web video link")
        self.zoom.blank("")

    # loading
    def on_open(self):
        d = file_dialog("Open recording", [("Audio / video", ["*.wav", "*.flac", "*.mp3", "*.ogg", "*.opus",
                                                              "*.m4a", "*.webm", "*.mkv", "*.mp4", "*.mov"]),
                                           ("All files", ["*"])], self.win.root())
        d.open(self.win, None, self._opened)

    def _opened(self, dlg, res):
        try:
            f = dlg.open_finish(res)
        except GLib.Error:
            return
        self.set_audio(Path(f.get_path()))
        if self.win.case_path and casefile.get(self.win.doc, None, "audio") in (None, ""):
            self.win.set_value(None, "audio", self.win.rel(self.path))

    def on_link(self):
        dlg = Adw.AlertDialog(heading="Open a web video",
                              body="The audio is downloaded with yt-dlp into data/downloads/.")
        entry = Gtk.Entry(placeholder_text="https://www.youtube.com/watch?v=…", activates_default=True,
                          width_chars=40)
        dlg.set_extra_child(entry)
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("open", "Download")
        dlg.set_response_appearance("open", Adw.ResponseAppearance.SUGGESTED)
        dlg.set_default_response("open")
        dlg.set_close_response("cancel")

        def resp(_d, r):
            url = entry.get_text().strip()
            if r != "open" or not url:
                return
            from .. import detect
            self.win.busy(True, "downloading")
            self.win.toast("downloading audio…")
            root = self.win.root() / "data" / "downloads"
            in_thread(lambda: detect.fetch(url, root, log=lambda *a: None), self._fetched, self.win.error)

        dlg.connect("response", resp)
        dlg.present(self.win)

    def _fetched(self, res):
        path, meta = res
        self.win.busy(False)
        self.set_audio(path)
        self.ov_card.subtitle.set_text(f"{meta.get('title') or path.name} · {fmt_path(path)}")

    def set_audio(self, path: Path, focus=None):
        self.path = path
        self.ov_card.subtitle.set_text(fmt_path(path))
        self.detections = []
        self._fill_detections()
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
        self._draw_detections()

    def on_overview_click(self, t, f, button_):
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
            self.win.toast("the zoom window must be at least 5 s")
            return
        w = self.win_s.get_value()
        kw = dict(win_s=w, hop_s=w / 4, fmin=self.fmin.get_value(), fmax=self.fmax.get_value())
        path = self.path
        self.win.busy(True, "spectrogram")
        in_thread(lambda: audio.brightness(path, t0, t1, **kw), self._zoom_done, self.win.error)

    def _zoom_done(self, sp):
        self.win.busy(False)
        self.zoom.show(sp)
        self.zoom_card.subtitle.set_text(f"{fmt_t(sp.t[0])} – {fmt_t(sp.t[-1])} · whitened")
        self.overview.vspan("win", sp.t[0], sp.t[-1])
        adj = self.sl["t"].get_adjustment()
        self._setting = True
        adj.set_lower(sp.t[0])
        adj.set_upper(sp.t[-1])
        self._setting = False
        self.draw_trace()
        self.draw_model()

    # detection
    def on_detect(self):
        if not self.path:
            self.win.toast("open a recording first")
            return
        from .. import detect
        path = self.path
        self.det_group.set_description("detecting…")
        self.win.busy(True, "detecting flyovers")

        def log(msg):
            GLib.idle_add(lambda: self.det_group.set_description(f"detecting… {msg.strip()}") and False)

        in_thread(lambda: detect.detect(path, log=log), self._detected, self._detect_failed)

    def _detect_failed(self, e):
        self.det_group.set_description("detection failed")
        self.win.error(e)

    def _detected(self, res):
        self.win.busy(False)
        self.detections = [f for f in res.flyovers if f.kind == "aircraft"]
        strong = sum(f.confidence != "low" for f in self.detections)
        close = len(res.flyovers) - len(self.detections)
        self.det_group.set_description(
            f"{'Aircraft Doppler curves found' if res.found else 'No clear aircraft Doppler curves'}: "
            f"{strong} flyovers, {len(self.detections) - strong} weak; {close} close-range sources "
            f"(road vehicles etc.) not listed. Click a flyover to inspect it.")
        self._fill_detections()
        self._draw_detections()
        self.split.set_show_sidebar(True)
        self.win.toast(f"{strong} flyovers detected")

    def _shown(self):
        weak = self.show_weak.get_active()
        return [(i, f) for i, f in enumerate(self.detections) if weak or f.confidence != "low"]

    def _fill_detections(self):
        while (r := self.det_list.get_row_at_index(0)) is not None:
            self.det_list.remove(r)
        for n, (i, f) in enumerate(self._shown(), 1):
            row = Adw.ActionRow(title=f"{n} · {fmt_t(f.t_cpa)}", activatable=True,
                                subtitle=f"{f.f0:.1f} Hz × {','.join(map(str, f.harmonics))} · {f.v:.0f} m/s · "
                                         f"{f.d / 1000:.1f} km · score {f.z:.0f}")
            row.add_prefix(label("●", ["dot", f.confidence], tooltip_text=f"{f.confidence} confidence"))
            row.add_suffix(Gtk.Image(icon_name="go-next-symbolic"))
            row.idx = i
            self.det_list.append(row)

    def _draw_detections(self):
        from .. import physics
        lines, marks = [], []
        for n, (_, f) in enumerate(self._shown(), 1):
            col = ACCENT[f.confidence]
            half = float(np.clip(3.0 * f.d / max(f.v, 1), 20.0, 90.0))
            tt = np.linspace(f.t_cpa - half, f.t_cpa + half, 120)
            base = physics.linear_flyby(tt, f.f0, f.v, f.d, f.t_cpa, self.win.c())
            lines += [(tt, h * base, {"color": col, "lw": 1.0, "alpha": 0.85}) for h in f.harmonics]
            marks.append((f.t_cpa, str(n), col))
        self.overview.layer("det", lines)
        self.overview.markers("detm", marks)

    def use_detection(self, i):
        from .. import measure
        f = self.detections[i]
        half = float(np.clip(3.0 * f.d / max(f.v, 1), 25.0, 100.0))
        self.t0.set_value(max(0.0, f.t_cpa - half))
        self.t1.set_value(f.t_cpa + half)
        self.fmin.set_value(max(15.0, f.f0 * min(f.harmonics) * 0.75))
        self.fmax.set_value(f.f0 * max(f.harmonics) * 1.3)
        self.harms.set_text(" ".join(map(str, f.harmonics)))
        self._set_sliders(f.f0, f.v, f.d, f.t_cpa)
        if f.refined and f.ranges:
            self.fit = measure.Fit(f.f0, f.v, f.d, f.t_cpa, f.rms, 1)
            self.ranges = {k: tuple(v) for k, v in f.ranges.items()}
            self.manual = False
            self._show_fit(self.fit, self.ranges, f.points, None)
        else:
            self.fit, self.ranges, self.manual = None, None, True
            self.fit_row.set_subtitle("detection only (not fitted): trace and Fit for ranges")
        self.ev_name.set_text(f"E{len(casefile.events(self.win.doc)) + 1}")
        self.show_model.set_active(True)
        self.load_zoom()

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

    def on_zoom_click(self, t, f, button_):
        if not self.click_mode.get_active():
            return
        if button_ == 1:
            i = int(np.searchsorted(self.tr_t, t))
            self.tr_t.insert(i, t)
            self.tr_f.insert(i, f)
            self.tr_z.insert(i, 0.0)
        elif button_ == 3 and self.tr_t and self.zoom.spec is not None:
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
        self.zoom.layer("trace", [(self.tr_t, self.tr_f, {"color": ACCENT["trace"], "lw": 0, "marker": ".",
                                                          "ms": 3, "alpha": 0.9})] if self.tr_t else [])

    def trace_array(self):
        h = int(self.harm.get_value())
        return np.c_[self.tr_t, self.tr_f, np.full(len(self.tr_t), h), self.tr_z]

    def on_save_trace(self):
        if not self.tr_t:
            self.win.toast("nothing traced")
            return
        d = file_dialog("Save trace", [("CSV", ["*.csv"])], self.win.out_dir(), f"{self.ev_name.get_text()}.csv")
        d.save(self.win, None, self._save_trace)

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
            self.win.toast("trace at least 8 points first")
            return
        from .. import measure
        t, f = np.array(self.tr_t), np.array(self.tr_f)
        h, c = int(self.harm.get_value()), self.win.c()
        self.fit_row.set_subtitle("fitting…")
        in_thread(lambda: measure.fit_trace(t, f, h, c=c), self._fitted, self.win.error)

    def _set_sliders(self, f0, v, d, t):
        self._setting = True
        adj = self.sl["t"].get_adjustment()
        adj.set_lower(min(adj.get_lower(), t - 1))
        adj.set_upper(max(adj.get_upper(), t + 1))
        for k, val in (("f0", f0), ("v", v), ("d", d), ("t", t)):
            self.sl[k].set_value(val)
        self._setting = False

    def _show_fit(self, fit, rng, n_in, n_all):
        sp, dd = rng["speed_ms"], rng["slant_m"]
        pts = f" · {n_in}/{n_all} points" if n_all else ""
        self.fit_row.set_subtitle(
            f"f0 {fit.f0:.2f} Hz · {fit.v:.1f} m/s ({fit.v / 0.514444:.0f} kt) · {fit.d:,.0f} m · "
            f"CPA {fit.t_cpa:.2f} s · rms {fit.rms:.2f} Hz{pts}\n"
            f"accepted: {sp[0]:.0f}–{sp[1]:.0f} m/s, {dd[0]:,.0f}–{dd[1]:,.0f} m")

    def _fitted(self, res):
        fit, rng, inl = res
        self.fit, self.ranges, self.manual = fit, rng, False
        self._set_sliders(fit.f0, fit.v, fit.d, fit.t_cpa)
        self._show_fit(fit, rng, int(inl.sum()), len(inl))
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
        base = physics.linear_flyby(sp.t, v["f0"], v["v"], v["d"], v["t"], self.win.c())
        self.zoom.layer("model", [(sp.t, h * base, {}) for h in self.harmonics()])

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
        self.ev_name.set_text(f"E{len(casefile.events(self.win.doc)) + 1}")


# --------------------------------------------------------------------------------------------- case

class CasePage(Adw.PreferencesPage):
    def __init__(self, win: "Window"):
        super().__init__()
        self.win = win
        self.groups = []

    def _add(self, g):
        self.add(g)
        self.groups.append(g)

    def rebuild(self):
        for g in self.groups:
            self.remove(g)
        self.groups = []
        doc = self.win.doc
        g = Adw.PreferencesGroup(title="Case file")
        r = Adw.ActionRow(title=self.win.case_path.name if self.win.case_path else "Unsaved case",
                          subtitle=fmt_path(self.win.case_path.parent) if self.win.case_path else "save it to run it")
        r.add_prefix(Gtk.Image(icon_name="text-x-generic-symbolic"))
        r.add_suffix(button(icon="document-open-symbolic", cb=self.win.open_dialog, style="flat", tip="Open case"))
        r.add_suffix(button(icon="document-save-as-symbolic", cb=self.win.save_as, style="flat", tip="Save as"))
        g.add(r)
        self._add(g)

        g = Adw.PreferencesGroup(title="Recording and data")
        for key in ("name", "day", "audio", "adsb", "cache", "out"):
            g.add(self._row(doc, None, key, True, ""))
        known = {k for k, _, _ in casefile.TOP} | set(casefile.SECTIONS) | {"event"}
        for key, v in doc.items():  # other top-level keys in the file (root, legacy "wind", ...)
            if key not in known and not casefile.is_table(v):
                g.add(self._row(doc, None, key, isinstance(v, str), ""))
        self._add(g)
        g = Adw.PreferencesGroup(title="Physics")
        dflt = {"c": "343.0", "atmosphere": "gfs", "altitude": "auto"}
        for key, kind, _hint in casefile.TOP:
            if key in ("c", "atmosphere", "wind_ms", "altitude"):
                g.add(self._row(doc, None, key, kind == "str", dflt.get(key, "")))
        self._add(g)

        g = Adw.PreferencesGroup(title="Advanced", description="Empty fields use the default shown")
        for sec, cls in casefile.SECTIONS.items():
            title, sub = SECTION_TITLES[sec]
            exp = Adw.ExpanderRow(title=title, subtitle=sub)
            for name, is_str, d in casefile.fields(cls):
                exp.add_row(self._row(doc.get(sec), sec, name, is_str, d))
            g.add(exp)
        self._add(g)

        g = Adw.PreferencesGroup(title="Events", description="Measured flyovers; add them from the Audio page")
        g.set_header_suffix(button(icon="list-add-symbolic", cb=self._blank_event, style="flat",
                                   tip="Add a blank event"))
        evs = casefile.events(doc)
        if not evs:
            g.add(Adw.ActionRow(title="No events yet", subtitle="Audio page → detect or trace a flyover → Add"))
        for i, ev in enumerate(evs):
            exp = Adw.ExpanderRow(title=str(ev.get("name", f"event {i + 1}")), subtitle=self._ev_summary(ev))
            exp.add_suffix(button(icon="user-trash-symbolic", cb=lambda i=i: self._remove(i), style="flat",
                                  tip="Remove event"))
            for name, is_str, d in casefile.EVENT_FIELDS:
                exp.add_row(self._row(ev, None, name, is_str, d, event=i))
            g.add(exp)
        self._add(g)

    @staticmethod
    def _ev_summary(ev):
        def f(k, fmt):
            v = ev.get(k)
            if v is None:
                return "?"
            v = v.unwrap() if hasattr(v, "unwrap") else v
            return "–".join(fmt.format(x) for x in v) if isinstance(v, list) else fmt.format(v)
        return (f"CPA {f('audio_cpa', '{:.1f}')} s · f0 {f('f0', '{:.2f}')} Hz · {f('speed_ms', '{:.0f}')} m/s"
                f" · {f('slant_m', '{:.0f}')} m")

    def _store(self, section, key, val, event):
        if event is not None:
            casefile.put(self.win.doc["event"][event], None, key, val)
        else:
            casefile.put(self.win.doc, section, key, val)
        self.win.set_dirty(True)

    def _row(self, tbl, section, key, is_str, dflt, event=None):
        title = LABELS.get(key, key)
        v = None if tbl is None else tbl.get(key)
        cur = None if v is None else (v.unwrap() if hasattr(v, "unwrap") else v)
        enum = ENUMS.get((section, key))
        if dflt in ("true", "false") and not is_str:  # boolean field
            r = Adw.SwitchRow(title=title, subtitle=f"{key} · default {dflt}",
                              active=bool(cur) if cur is not None else dflt == "true")
            r.connect("notify::active", lambda row, _p: self._store(section, key, row.get_active(), event))
            return r
        if enum:
            opts = [str(o) for o in enum]
            r = Adw.ComboRow(title=title, subtitle=key + (f" · default {dflt}" if dflt else ""),
                             model=Gtk.StringList.new(opts))
            val = cur if cur is not None else dflt
            if str(val) in opts:
                r.set_selected(opts.index(str(val)))
            r.connect("notify::selected", lambda row, _p: self._store(section, key, enum[row.get_selected()], event))
            return r
        if is_str:
            r = Adw.EntryRow(title=title)
            if cur is not None:
                r.set_text(str(cur))
            entry = r
        else:
            r = Adw.ActionRow(title=title, subtitle=key + (f" · default {dflt}" if dflt else ""))
            entry = Gtk.Entry(valign=Gtk.Align.CENTER, width_chars=16, xalign=1.0, placeholder_text=dflt or "unset")
            if cur is not None:
                entry.set_text(casefile.to_text(v, False))
            r.add_suffix(entry)

        def changed(e):
            try:
                val = casefile.parse(e.get_text(), is_str)
            except ValueError:
                e.add_css_class("error")
                return
            e.remove_css_class("error")
            self._store(section, key, val, event)

        entry.connect("changed", changed)
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

STEPS = (("fetch", "Download ADS-B", "One UTC day of adsb.lol history (gh CLI) and the aircraft type table",
          "folder-download-symbolic"),
         ("ingest", "Ingest", "Raw dump → parquet tracks for the case's day", "view-list-symbolic"),
         ("search", "Search", "Every place that could have heard both events", "system-search-symbolic"),
         ("score", "Score", "Model and score the candidates against the audio", "starred-symbolic"),
         ("heatmap", "Heat map", "Dense score map around one ranked pair", "find-location-symbolic"))


class RunPage(Adw.PreferencesPage):
    def __init__(self, win: "Window"):
        super().__init__()
        self.win = win
        self.job: Job | None = None
        self.queue: list[tuple[str, list, bool]] = []
        self.current = None

        pipe = Adw.PreferencesGroup(title="Pipeline", description="Heavy steps run one at a time")
        pipe.set_header_suffix(button("Search + score", lambda: self.start([self.step("search"),
                                                                             self.step("score")]),
                                      style="suggested-action"))
        self.rows, self.status = {}, {}
        self.rank = spin(1, 500, 1, 0, 1, 3)
        for key, title, sub, icon in STEPS:
            r = Adw.ActionRow(title=title, subtitle=sub)
            r.add_prefix(Gtk.Image(icon_name=icon))
            st = Gtk.Stack()
            st.add_named(Gtk.Box(), "idle")
            st.add_named(Adw.Spinner(), "running")
            st.add_named(Gtk.Image(icon_name="object-select-symbolic", css_classes=["success"]), "done")
            st.add_named(Gtk.Image(icon_name="dialog-error-symbolic", css_classes=["error"]), "failed")
            r.add_suffix(st)
            if key == "heatmap":
                r.add_suffix(hbox(label("rank", "dim-label"), self.rank, spacing=4))
            b = button("Run", lambda k=key: self.start([self.step(k)]))
            r.add_suffix(b)
            pipe.add(r)
            self.rows[key] = b
            self.status[key] = st
        self.add(pipe)

        g = Adw.PreferencesGroup(title="Options")
        self.cap = Adw.EntryRow(title="Memory cap for heavy steps", text="26G")
        self.use_cap = Adw.SwitchRow(title="Memory-capped scope for heavy steps",
                                     subtitle="an overrun then kills only the job (systemd-run)",
                                     active=CAPPED.exists(), sensitive=CAPPED.exists())
        self.truth = Adw.SwitchRow(title="Compare with truth.json", subtitle="when the output folder has a truth.json",
                                   active=True)
        self.noisy = Adw.SwitchRow(title="Also score places that fail the silence rule")
        for r in (self.cap, self.use_cap, self.truth, self.noisy):
            g.add(r)
        self.add(g)

        g = Adw.PreferencesGroup(title="Log")
        self.state = label("idle", ["dim-label"])
        self.stop_b = button("Stop", self.stop, style=["destructive-action"])
        self.stop_b.set_sensitive(False)
        g.set_header_suffix(hbox(self.state, self.stop_b,
                                 button(icon="edit-clear-all-symbolic", cb=lambda: self.buf.set_text(""),
                                        style="flat", tip="Clear the log"), spacing=8))
        self.view = Gtk.TextView(editable=False, monospace=True, cursor_visible=False,
                                 wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.view.add_css_class("log")
        self.buf = self.view.get_buffer()
        frame = Gtk.Frame(child=Gtk.ScrolledWindow(child=self.view, min_content_height=320))
        frame.add_css_class("card")
        g.add(frame)
        self.add(g)

    def case(self) -> str:
        return str(self.win.case_path)

    def step(self, key):
        day = str(casefile.get(self.win.doc, None, "day") or "")
        if key == "fetch":
            return ("fetch", ["fetch-adsb", day], False)
        if key == "ingest":
            return ("ingest", ["ingest", day, "--out", str(self.win.resolve("adsb", f"data/adsb/{day}"))], True)
        if key == "search":
            return ("search", ["search", self.case()], True)
        if key == "score":
            a = ["score", self.case()]
            t = self.win.out_dir() / "truth.json"
            if self.truth.get_active() and t.exists():
                a += ["--truth", str(t)]
            if self.noisy.get_active():
                a.append("--keep-noisy")
            return ("score", a, True)
        return ("heatmap", ["heatmap", self.case(), "--rank", str(int(self.rank.get_value()))], True)

    def log(self, line: str):
        self.buf.insert(self.buf.get_end_iter(), line + "\n")
        mark = self.buf.create_mark(None, self.buf.get_end_iter(), False)
        self.view.scroll_to_mark(mark, 0.0, False, 0, 0)
        self.buf.delete_mark(mark)

    def start(self, steps):
        if self.job is not None:
            self.win.toast("a job is already running")
            return
        if any(s[0] in ("fetch", "ingest") for s in steps) and not casefile.get(self.win.doc, None, "day"):
            self.win.toast("set the case's UTC day first (Case page)")
            return
        if not self.win.ensure_saved(lambda: self.start(steps)):
            return
        if any(s[0] == "fetch" for s in steps) and \
                not (self.win.root() / "data/ref/icao_aircraft_types2.js.gz").exists():
            steps = [("fetch-types", ["fetch-types"], False)] + list(steps)
        for k, st in self.status.items():
            if any(s[0] == k for s in steps):
                st.set_visible_child_name("idle")
        self.queue = list(steps)
        self._next()

    def _next(self):
        if not self.queue:
            self._idle("finished")
            return
        name, args, heavy = self.queue.pop(0)
        argv = dopploc_argv(*args)
        if heavy and self.use_cap.get_active():
            argv = capped(argv, self.cap.get_text().strip() or None)
        self.log(f"$ {' '.join(argv)}")
        self.state.set_text(f"running {name}…")
        for b in self.rows.values():
            b.set_sensitive(False)
        self.stop_b.set_sensitive(True)
        self.current = name
        if name in self.status:
            self.status[name].set_visible_child_name("running")
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
        if name in self.status:
            self.status[name].set_visible_child_name("done" if rc == 0 else "failed")
        if rc != 0:
            self.queue = []
            self._idle(f"{name} failed (exit {rc})")
            return
        if name in ("score", "heatmap"):
            self.win.results.reload()
        self._next()

    def _idle(self, msg):
        self.state.set_text(msg)
        for b in self.rows.values():
            b.set_sensitive(True)
        self.stop_b.set_sensitive(False)
        self.win.toast(msg)

    def stop(self):
        self.queue = []
        if self.job:
            self.job.stop()


# --------------------------------------------------------------------------------------------- results

COLS = (("#", "rank", "{}"), ("Score", "score", "{:.3f}"), ("Ev 1", "S1", "{:.2f}"), ("Ev 2", "S2", "{:.2f}"),
        ("Latitude", "lat", "{:.5f}"), ("Longitude", "lon", "{:.5f}"), ("km to truth", "dist_km", "{:.2f}"))


class ResultsPage(Gtk.Box):
    def __init__(self, win: "Window"):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.add_css_class("pane")
        self.win = win
        self.rows: list[dict] = []
        self.truth = None
        self.best = None
        self.sel = None
        self.pills = Adw.WrapBox(child_spacing=6, line_spacing=6, hexpand=True, valign=Gtk.Align.CENTER)
        self.append(hbox(label("Results", ["title-3"]), self.pills,
                         button(icon="view-refresh-symbolic", cb=self.reload, style="flat", tip="Reload results"),
                         button(icon="folder-open-symbolic", cb=self.open_folder, style="flat",
                                tip="Open the output folder"), spacing=12))
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL, position=470, vexpand=True,
                          shrink_start_child=False, shrink_end_child=False)
        self.append(paned)
        self.cv = Gtk.ColumnView(show_row_separators=False, reorderable=False)
        self.cv.add_css_class("data-table")
        frame = Gtk.Frame(child=Gtk.ScrolledWindow(child=self.cv, hexpand=True))
        frame.add_css_class("card")
        paned.set_start_child(frame)

        self.stack = Adw.ViewStack(vexpand=True)
        self.pair = PairView(height=300)
        self.stack.add_titled_with_icon(self.pair.canvas, "curves", "Curves", "media-playlist-shuffle-symbolic")
        if Shumate is not None:
            self.map = Shumate.SimpleMap(hexpand=True, vexpand=True)
            reg = Shumate.MapSourceRegistry.new_with_defaults()
            self.map.set_map_source(reg.get_by_id(Shumate.MAP_SOURCE_OSM_MAPNIK))
            self.markers = Shumate.MarkerLayer.new(self.map.get_viewport())
            self.map.add_overlay_layer(self.markers)
            self.sel_marker = None
            self.stack.add_titled_with_icon(self.map, "map", "Map", "mark-location-symbolic")
        else:
            self.map = None
            self.stack.add_titled_with_icon(Adw.StatusPage(title="Map unavailable",
                                                           description="libshumate typelib not installed",
                                                           icon_name="mark-location-symbolic"),
                                            "map", "Map", "mark-location-symbolic")
        self.scatter = ScatterView()
        self.stack.add_titled_with_icon(self.scatter.canvas, "scores", "Scores", "view-grid-symbolic")
        self.heat = Gtk.Picture(can_shrink=True, content_fit=Gtk.ContentFit.CONTAIN)
        self.heat_status = Adw.StatusPage(icon_name="find-location-symbolic", title="No heat map yet")
        self.heat_stack = Gtk.Stack()
        self.heat_stack.add_named(self.heat_status, "none")
        self.heat_stack.add_named(self.heat, "pic")
        self.stack.add_titled_with_icon(self.heat_stack, "heat", "Heat map", "weather-clear-symbolic")
        self.text = Gtk.TextView(editable=False, monospace=True, cursor_visible=False, left_margin=10,
                                 top_margin=10)
        self.stack.add_titled_with_icon(Gtk.ScrolledWindow(child=self.text), "summary", "Summary",
                                        "text-x-generic-symbolic")
        right = vbox(Adw.InlineViewSwitcher(stack=self.stack, halign=Gtk.Align.CENTER), self.stack, spacing=8,
                     css=["card", "plot-card"])
        paned.set_end_child(right)

    def open_folder(self):
        out = self.win.out_dir()
        if out.is_dir():
            Gtk.FileLauncher.new(Gio.File.new_for_path(str(out))).launch(self.win, None, None)

    def _set_pills(self, items):
        while (c := self.pills.get_first_child()) is not None:
            self.pills.remove(c)
        for text, kind in items:
            self.pills.append(pill(text, kind))

    def reload(self):
        import polars as pl
        out = self.win.out_dir()
        csv = out / "best_per_pair.csv"
        for c in list(self.cv.get_columns()):
            self.cv.remove_column(c)
        self.rows = []
        self.sel = None
        if not csv.exists():
            self._set_pills([("no results yet", None)])
            self.cv.set_model(None)
            self.pair.blank("no results yet: Run → Search + score")
            return
        best = pl.read_csv(csv, infer_schema_length=10000)
        tp = out / "truth.json"
        self.truth = json.loads(tp.read_text()) if tp.exists() else None
        if self.truth:
            from ..report import haversine
            d = haversine(best["lat"].to_numpy(), best["lon"].to_numpy(), self.truth["lat"], self.truth["lon"])
            best = best.with_columns(pl.Series("dist_km", d / 1000.0))
        self.best = best
        self.rows = [dict(r, rank=i + 1) for i, r in enumerate(best.head(500).iter_rows(named=True))]
        items = [(f"{len(best):,} flight pairs", None)]
        if len(best) > 1:
            items.append((f"1st − 2nd = {best['score'][0] - best['score'][1]:.3f}", None))
        if self.truth:
            within = np.flatnonzero(best["dist_km"].to_numpy() < 0.3)
            items.append((f"truth ranks {int(within[0]) + 1}" if len(within) else "truth not in the pairs",
                          "good" if len(within) and within[0] == 0 else "warn"))
            items.append((f"top-1 is {best['dist_km'][0] * 1000:,.0f} m from truth",
                          "good" if best["dist_km"][0] < 0.3 else "warn"))
        self._set_pills(items)
        for title, key, fmt in COLS:
            if key == "dist_km" and not self.truth or key == "f2" and "f2" not in best.columns:
                continue
            fac = Gtk.SignalListItemFactory()
            fac.connect("setup", lambda _f, item: item.set_child(
                label("", ["numeric"], xalign=1.0)))
            fac.connect("bind", lambda _f, item, key=key, fmt=fmt: item.get_child().set_text(
                self._fmt(self.rows[int(item.get_item().get_string())].get(key), fmt)))
            col = Gtk.ColumnViewColumn(title=title, factory=fac, resizable=True, expand=key in ("lat", "lon"))
            self.cv.append_column(col)
        model = Gtk.SingleSelection(model=Gtk.StringList.new([str(i) for i in range(len(self.rows))]))
        model.connect("notify::selected", lambda m, _p: self.select(m.get_selected()))
        self.cv.set_model(model)
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
        m.set_child(label(text, ["pin"] + ([cls] if cls else [])))
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
        near = np.flatnonzero(self.best["dist_km"].to_numpy() < 0.3) if self.truth else None
        s2 = self.best["S2"].to_numpy() if "S2" in self.best.columns else np.zeros(len(self.best))
        self.scatter.scatter_scores(self.best["S1"].to_numpy(), s2, idx, near)
        self._curves(r)
        hm = self.win.out_dir() / f"heatmap_rank{r['rank']}.png"
        if hm.exists():
            self.heat.set_filename(str(hm))
            self.heat_stack.set_visible_child_name("pic")
        else:
            self.heat_status.set_description(f"Run → Heat map with rank {r['rank']}")
            self.heat_stack.set_visible_child_name("none")

    def _curves(self, r):
        if not self.win.case_path:
            return
        rank = r["rank"]
        self.pair.blank(f"rendering rank {rank}…")
        from .. import config, report
        case_path = self.win.case_path
        row_ = {k: v for k, v in r.items() if k not in ("rank", "dist_km")}

        def done(data):
            if self.sel is not None and self.rows[self.sel]["rank"] == rank:
                for k, d in enumerate(data):
                    d["title"] = f"#{rank} · event {k + 1} · {r['f' + str(k + 1)]}"
                self.pair.show(data)

        in_thread(lambda: report.overlay_data(config.load(case_path), row_), done, self.win.error)


# --------------------------------------------------------------------------------------------- window

class Window(Adw.ApplicationWindow):
    def __init__(self, app, case_path: Path | None = None, page: str | None = None):
        super().__init__(application=app, title="Doppler Locate", default_width=1440, default_height=940)
        self.doc = casefile.new_case()
        self.case_path: Path | None = None
        self.dirty = False
        self._busy = 0
        self.toasts = Adw.ToastOverlay()
        self.stack = Adw.ViewStack()
        self.toasts.set_child(self.stack)

        header = Adw.HeaderBar()
        header.set_title_widget(Adw.ViewSwitcher(stack=self.stack, policy=Adw.ViewSwitcherPolicy.WIDE))
        header.pack_start(button("Open", self.open_dialog, icon="document-open-symbolic", style="flat",
                                 tip="Open a case (Ctrl+O)"))
        self.title = Adw.WindowTitle(title="Unsaved case", subtitle="")
        header.pack_start(self.title)
        menu = Gio.Menu()
        sec = Gio.Menu()
        sec.append("New case", "win.new")
        sec.append("Save", "win.save")
        sec.append("Save as…", "win.save-as")
        menu.append_section(None, sec)
        sec = Gio.Menu()
        sec.append("About Doppler Locate", "win.about")
        menu.append_section(None, sec)
        header.pack_end(Gtk.MenuButton(icon_name="open-menu-symbolic", menu_model=menu, tooltip_text="Menu"))
        header.pack_end(button(icon="document-save-symbolic", cb=self.save, style="flat", tip="Save case (Ctrl+S)"))
        self.spinner = Adw.Spinner(visible=False)
        header.pack_end(self.spinner)
        tv = Adw.ToolbarView(top_bar_style=Adw.ToolbarStyle.RAISED)
        tv.add_top_bar(header)
        tv.set_content(self.toasts)
        self.set_content(tv)

        self.audio = AudioPage(self)
        self.case = CasePage(self)
        self.run = RunPage(self)
        self.results = ResultsPage(self)
        self.stack.add_titled_with_icon(self.audio, "audio", "Audio", "audio-volume-high-symbolic")
        self.stack.add_titled_with_icon(self.case, "case", "Case", "document-properties-symbolic")
        self.stack.add_titled_with_icon(self.run, "run", "Run", "media-playback-start-symbolic")
        self.stack.add_titled_with_icon(self.results, "results", "Results", "mark-location-symbolic")

        for name, cb, accel in (("open", self.open_dialog, "<Ctrl>o"), ("save", self.save, "<Ctrl>s"),
                                ("save-as", self.save_as, "<Ctrl><Shift>s"), ("new", self.new_case, None),
                                ("about", self.about, None)):
            a = Gio.SimpleAction.new(name, None)
            a.connect("activate", lambda *_, cb=cb: cb())
            self.add_action(a)
            if accel:
                app.set_accels_for_action(f"win.{name}", [accel])
        self.connect("close-request", self._close_request)
        bp = Adw.Breakpoint.new(Adw.BreakpointCondition.parse("max-width: 1100sp"))
        bp.add_setter(self.audio.split, "collapsed", True)
        self.add_breakpoint(bp)
        sm = Adw.StyleManager.get_default()
        plots.set_dark(sm.get_dark())
        sm.connect("notify::dark", lambda m, _p: plots.set_dark(m.get_dark()))
        self.case.rebuild()
        self.results.reload()
        case_path = case_path or _last_case()
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

    def about(self):
        Adw.AboutDialog(application_name="Doppler Locate", application_icon=APP_ID, version="0.1.0",
                        comments="Find where a recording was made from aircraft Doppler curves and public "
                                 "ADS-B tracks.",
                        website="https://github.com/digitalis-p/doppler-locate").present(self)

    # case file
    def set_dirty(self, d: bool):
        self.dirty = d
        name = self.case_path.name if self.case_path else "Unsaved case"
        self.title.set_title(f"{name}{' •' if d else ''}")
        self.title.set_subtitle(str(casefile.get(self.doc, None, "name") or ""))
        self.set_title(f"Doppler Locate — {name}")

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
        _remember(path)
        self.set_dirty(False)
        self.case.rebuild()
        self.results.reload()
        if casefile.get(self.doc, None, "audio") and self.resolve("audio").exists():
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
        _remember(self.case_path)
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

        def resp(_d, r):
            if r == "discard":
                self.dirty = False
                self.close()
            elif r == "save":
                self.save(self.close)

        dlg.connect("response", resp)
        dlg.present(self)
        return True


def _last_case() -> Path | None:
    """The case opened or saved last (so a start from the app launcher resumes it)."""
    try:
        p = Path(json.loads(STATE.read_text())["last_case"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return p if p.is_file() else None


def _remember(path: Path):
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps({"last_case": str(path)}))
    except OSError:
        pass


class App(Adw.Application):
    def __init__(self, case: Path | None, page: str | None):
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.NON_UNIQUE)
        self.case = case
        self.page = page
        self.win = None

    def do_startup(self):
        Adw.Application.do_startup(self)
        css = Gtk.CssProvider()
        css.load_from_data(CSS, -1)
        display = Gdk.Display.get_default()
        Gtk.StyleContext.add_provider_for_display(display, css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        # the icon also resolves from a plain checkout, before scripts/install-desktop.sh has run
        Gtk.IconTheme.get_for_display(display).add_search_path(str(PKG_ROOT / "share" / "icons"))
        Gtk.Window.set_default_icon_name(APP_ID)

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
