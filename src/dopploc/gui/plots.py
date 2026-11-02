"""Matplotlib canvases embedded in GTK4 (FigureCanvasGTK4Agg; pyplot is never used here),
styled to sit inside a libadwaita window: theme background, muted axes, light/dark aware."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_gtk4agg import FigureCanvasGTK4Agg
from matplotlib.figure import Figure

from ..audio import Spectrogram

THEMES = {
    # window background, card background, foreground, dim foreground, grid
    "dark": dict(bg="#242424", card="#2e2e32", fg="#ffffff", dim="#9a9996", grid="#3d3d42"),
    "light": dict(bg="#fafafb", card="#ffffff", fg="#1e1e1e", dim="#6e6e73", grid="#e3e3e6"),
}
ACCENT = {"model": "#62a0ea", "trace": "#ffffff", "high": "#57e389", "medium": "#f8e45c", "low": "#9a9996",
          "truth": "#57e389", "pick": "#ff7800"}
FONT = {"family": ["Adwaita Sans", "Cantarell", "Inter", "DejaVu Sans"], "size": 9}

_canvases: list["_Base"] = []


def set_dark(dark: bool):
    """Restyle every live canvas (called when the desktop switches light/dark)."""
    for c in list(_canvases):
        c.style("dark" if dark else "light")


class _Base:
    theme = "dark"

    def __init__(self, height, figsize):
        self.fig = Figure(figsize=figsize, dpi=96, layout="constrained")
        self.canvas = FigureCanvasGTK4Agg(self.fig)
        self.canvas.set_size_request(300, height)
        self.canvas.set_hexpand(True)
        self.canvas.set_vexpand(True)
        _canvases.append(self)

    def _style_ax(self, ax):
        t = THEMES[self.theme]
        ax.set_facecolor(t["card"])
        for side, sp in ax.spines.items():
            sp.set_visible(side in ("left", "bottom"))
            sp.set_color(t["grid"])
        ax.tick_params(colors=t["dim"], labelsize=FONT["size"], length=3)
        ax.xaxis.label.set_color(t["dim"])
        ax.yaxis.label.set_color(t["dim"])
        ax.title.set_color(t["fg"])
        ax.xaxis.label.set_fontsize(FONT["size"])
        ax.yaxis.label.set_fontsize(FONT["size"])

    def style(self, theme: str):
        self.theme = theme
        self.fig.set_facecolor(THEMES[theme]["card"])
        for ax in self.fig.axes:
            self._style_ax(ax)
        self.canvas.draw_idle()


class SpecView(_Base):
    """A whitened spectrogram with named overlay layers that can be replaced cheaply."""

    def __init__(self, height=320, title=""):
        super().__init__(height, (9, 3.4))
        self.ax = self.fig.add_subplot()
        self.title = title
        self.spec: Spectrogram | None = None
        self.layers: dict[str, list] = {}
        self.blank()

    def blank(self, msg=""):
        self.spec = None
        self.layers = {}
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        if msg:
            self.ax.text(0.5, 0.5, msg, transform=self.ax.transAxes, ha="center", va="center",
                         color=THEMES[self.theme]["dim"], fontsize=10)
        self.style(self.theme)

    def show(self, spec: Spectrogram, vmin=0.0, vmax=5.0, title=None):
        self.spec = spec
        self.layers = {}
        self.ax.clear()
        self.ax.imshow(spec.S, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax,
                       interpolation="nearest", extent=[spec.t[0], spec.t[-1], spec.f[0], spec.f[-1]])
        self.ax.set_xlim(spec.t[0], spec.t[-1])
        self.ax.set_ylim(spec.f[0], spec.f[-1])
        self.ax.set_xlabel("audio time (s)")
        self.ax.set_ylabel("Hz")
        if title is not None:
            self.title = title
        self.ax.set_title(self.title, fontsize=FONT["size"] + 1, loc="left")
        self.style(self.theme)

    def layer(self, name: str, lines):
        """Replace overlay ``name`` with ``lines``: iterable of (t, f, style_dict)."""
        for a in self.layers.pop(name, []):
            a.remove()
        arts = []
        for t, f, style in lines:
            arts += self.ax.plot(t, f, **({"lw": 1.2, "color": ACCENT["model"], "alpha": 0.9} | dict(style)))
        self.layers[name] = arts
        if self.spec is not None:
            self.ax.set_xlim(self.spec.t[0], self.spec.t[-1])
            self.ax.set_ylim(self.spec.f[0], self.spec.f[-1])
        self.canvas.draw_idle()

    def markers(self, name: str, items):
        """Numbered markers along the top edge: iterable of (t, label, colour)."""
        for a in self.layers.pop(name, []):
            a.remove()
        arts = []
        if self.spec is not None:
            ytop = self.spec.f[-1]
            for t, label, colour in items:
                arts.append(self.ax.axvline(t, color=colour, lw=1.0, alpha=0.55))
                arts.append(self.ax.text(t, ytop, f" {label}", color=colour, va="top", ha="left",
                                         fontsize=FONT["size"], weight="bold"))
        self.layers[name] = arts
        self.canvas.draw_idle()

    def vspan(self, name: str, t0: float, t1: float):
        for a in self.layers.pop(name, []):
            a.remove()
        self.layers[name] = [self.ax.axvspan(t0, t1, color=ACCENT["pick"], alpha=0.18, lw=0)]
        self.canvas.draw_idle()

    def on_click(self, fn):
        """fn(t, f, button) for clicks inside the axes."""
        def cb(ev):
            if ev.inaxes is self.ax and ev.xdata is not None:
                fn(float(ev.xdata), float(ev.ydata), ev.button)
        self.canvas.mpl_connect("button_press_event", cb)


class PairView(_Base):
    """Side-by-side event spectrograms with modelled curves (Results page)."""

    def __init__(self, height=320):
        super().__init__(height, (9, 3.6))
        self.blank("select a candidate")

    def blank(self, msg=""):
        self.fig.clear()
        ax = self.fig.add_subplot()
        ax.set_xticks([])
        ax.set_yticks([])
        if msg:
            ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
                    color=THEMES[self.theme]["dim"], fontsize=10)
        self.style(self.theme)

    def show(self, data: list[dict]):
        self.fig.clear()
        axes = self.fig.subplots(1, len(data), squeeze=False)[0]
        for ax, d in zip(axes, data):
            sp = d["spec"]
            ax.imshow(sp.S, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=5, interpolation="nearest",
                      extent=[sp.t[0], sp.t[-1], sp.f[0], sp.f[-1]])
            for t, f in d["curves"]:
                ax.plot(t, f, color=ACCENT["model"], lw=1.0, alpha=0.75)
            ax.set_xlim(sp.t[0], sp.t[-1])
            ax.set_ylim(sp.f[0], sp.f[-1])
            ax.set_title(d["title"], fontsize=FONT["size"] - 1, loc="left")
            ax.set_xlabel("audio time (s)")
            ax.set_ylabel("Hz")
        self.style(self.theme)


class ScatterView(_Base):
    """Event-1 vs event-2 score of every flight pair."""

    def __init__(self, height=320):
        super().__init__(height, (6, 4))
        self.ax = self.fig.add_subplot()
        self.style(self.theme)

    def scatter_scores(self, s1, s2, sel=None, truth_idx=None):
        t = THEMES[self.theme]
        self.ax.clear()
        self.ax.scatter(s1, s2, s=5, c=t["dim"], alpha=0.6, lw=0, rasterized=True)
        if truth_idx is not None and len(truth_idx):
            self.ax.scatter(np.asarray(s1)[truth_idx], np.asarray(s2)[truth_idx], s=60, marker="*",
                            c=ACCENT["truth"], label="within 300 m of truth", zorder=3)
        if sel is not None:
            self.ax.scatter([np.asarray(s1)[sel]], [np.asarray(s2)[sel]], s=90, facecolors="none",
                            edgecolors=ACCENT["pick"], lw=1.5, label="selected", zorder=4)
        self.ax.set_xlabel("event 1 score")
        self.ax.set_ylabel("event 2 score")
        if sel is not None or (truth_idx is not None and len(truth_idx)):
            leg = self.ax.legend(loc="lower right", fontsize=FONT["size"], frameon=False)
            for txt in leg.get_texts():
                txt.set_color(t["dim"])
        self.style(self.theme)
