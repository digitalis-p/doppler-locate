"""Matplotlib canvases embedded in GTK4 (FigureCanvasGTK4Agg; pyplot is never used here)."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_gtk4agg import FigureCanvasGTK4Agg
from matplotlib.figure import Figure

from ..audio import Spectrogram


class SpecView:
    """A whitened spectrogram with overlay layers that can be replaced without redrawing the image."""

    def __init__(self, height=320, title=""):
        self.fig = Figure(figsize=(9, 3.4), dpi=96, layout="constrained")
        self.ax = self.fig.add_subplot()
        self.canvas = FigureCanvasGTK4Agg(self.fig)
        self.canvas.set_size_request(400, height)
        self.canvas.set_hexpand(True)
        self.canvas.set_vexpand(True)
        self.title = title
        self.spec: Spectrogram | None = None
        self.layers: dict[str, list] = {}
        self._blank()

    def _blank(self):
        self.ax.clear()
        self.ax.set_xlabel("audio time (s)")
        self.ax.set_ylabel("Hz")
        self.ax.set_title(self.title, fontsize=9)
        self.canvas.draw_idle()

    def show(self, spec: Spectrogram, vmin=0.0, vmax=5.0):
        self.spec = spec
        self.layers = {}
        self.ax.clear()
        self.ax.pcolormesh(spec.t, spec.f, spec.S, vmin=vmin, vmax=vmax, cmap="magma", shading="auto",
                           rasterized=True)
        self.ax.set_xlim(spec.t[0], spec.t[-1])
        self.ax.set_ylim(spec.f[0], spec.f[-1])
        self.ax.set_xlabel("audio time (s)")
        self.ax.set_ylabel("Hz")
        self.ax.set_title(self.title, fontsize=9)
        self.canvas.draw_idle()

    def layer(self, name: str, lines):
        """Replace overlay ``name`` with ``lines``: iterable of (t, f, style_dict)."""
        for a in self.layers.pop(name, []):
            a.remove()
        arts = []
        for t, f, style in lines:
            arts += self.ax.plot(t, f, **({"lw": 1.0, "color": "cyan", "alpha": 0.8} | dict(style)))
        self.layers[name] = arts
        if self.spec is not None:
            self.ax.set_xlim(self.spec.t[0], self.spec.t[-1])
            self.ax.set_ylim(self.spec.f[0], self.spec.f[-1])
        self.canvas.draw_idle()

    def vspan(self, name: str, t0: float, t1: float):
        for a in self.layers.pop(name, []):
            a.remove()
        self.layers[name] = [self.ax.axvspan(t0, t1, color="white", alpha=0.15, lw=0)]
        self.canvas.draw_idle()

    def on_click(self, fn):
        """fn(t, f, button, modifiers) for clicks inside the axes (not while zoom/pan is active)."""
        def cb(ev):
            if ev.inaxes is self.ax and ev.xdata is not None:
                fn(float(ev.xdata), float(ev.ydata), ev.button, ev.modifiers or set())
        self.canvas.mpl_connect("button_press_event", cb)


class ScatterView:
    """Generic x/y figure (score scatter, heat map)."""

    def __init__(self, height=320):
        self.fig = Figure(figsize=(6, 4), dpi=96, layout="constrained")
        self.ax = self.fig.add_subplot()
        self.canvas = FigureCanvasGTK4Agg(self.fig)
        self.canvas.set_size_request(300, height)
        self.canvas.set_hexpand(True)
        self.canvas.set_vexpand(True)

    def scatter_scores(self, s1, s2, best_idx=None, truth_idx=None):
        self.ax.clear()
        self.ax.scatter(s1, s2, s=4, c="0.6", rasterized=True)
        if best_idx is not None:
            self.ax.scatter(np.asarray(s1)[best_idx], np.asarray(s2)[best_idx], s=40, facecolors="none",
                            edgecolors="tab:red", label="selected")
        if truth_idx is not None and len(truth_idx):
            self.ax.scatter(np.asarray(s1)[truth_idx], np.asarray(s2)[truth_idx], s=30, marker="*",
                            c="tab:green", label="within 300 m of truth")
        self.ax.set_xlabel("event 1 score")
        self.ax.set_ylabel("event 2 score")
        if best_idx is not None or (truth_idx is not None and len(truth_idx)):
            self.ax.legend(loc="lower right", fontsize=8)
        self.canvas.draw_idle()
