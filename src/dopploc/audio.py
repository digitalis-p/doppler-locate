"""Audio loading and high-resolution low-frequency spectrograms.

Prop Doppler curves sit at roughly 50-400 Hz and are usually inaudible, so the useful view
is a narrow band with long FFT windows (sub-Hz resolution), whitened per frequency bin so
stationary hum and slow background changes drop out.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal


def load_audio(path: str | Path, sr: int = 8000, start: float | None = None,
               duration: float | None = None) -> np.ndarray:
    """Decode any ffmpeg-readable file to mono float32 at ``sr`` Hz."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start is not None:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += ["-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


@dataclass
class Spectrogram:
    f: np.ndarray        # bin centre frequencies (Hz)
    t: np.ndarray        # frame centre times (s, audio clock)
    S: np.ndarray        # (len(f), len(t)) power in dB, or whitened score after .whiten()
    df: float
    dt: float

    def band(self, fmin: float, fmax: float) -> "Spectrogram":
        sel = (self.f >= fmin) & (self.f <= fmax)
        return Spectrogram(self.f[sel], self.t, self.S[sel], self.df, self.dt)

    def window(self, tmin: float, tmax: float) -> "Spectrogram":
        sel = (self.t >= tmin) & (self.t <= tmax)
        return Spectrogram(self.f, self.t[sel], self.S[:, sel], self.df, self.dt)

    def whiten(self, clip: tuple[float, float] = (0.0, 6.0)) -> "Spectrogram":
        """Per-bin robust z-score (median / MAD over time), clipped. Brightness for scoring."""
        med = np.median(self.S, axis=1, keepdims=True)
        mad = np.median(np.abs(self.S - med), axis=1, keepdims=True) * 1.4826 + 1e-6
        Z = np.clip((self.S - med) / mad, clip[0], clip[1])
        return Spectrogram(self.f, self.t, Z.astype(np.float32), self.df, self.dt)

    def sample(self, t: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Bilinear lookup of S at arbitrary (t, f); NaN outside."""
        x = (t - self.t[0]) / self.dt
        y = (f - self.f[0]) / self.df
        ok = (x >= 0) & (x <= len(self.t) - 1) & (y >= 0) & (y <= len(self.f) - 1)
        out = np.full(np.shape(t), np.nan)
        x, y = x[ok], y[ok]
        i = np.minimum(x.astype(int), len(self.t) - 2)
        j = np.minimum(y.astype(int), len(self.f) - 2)
        wx, wy = x - i, y - j
        S = self.S
        out[ok] = ((1 - wx) * (1 - wy) * S[j, i] + wx * (1 - wy) * S[j, i + 1]
                   + (1 - wx) * wy * S[j + 1, i] + wx * wy * S[j + 1, i + 1])
        return out


def spectrogram(x: np.ndarray, sr: int, win_s: float = 1.0, hop_s: float = 0.1,
                fmin: float = 20.0, fmax: float = 600.0, zero_pad: int = 2,
                t_offset: float = 0.0) -> Spectrogram:
    """Hann STFT in dB limited to [fmin, fmax]. ``t_offset`` shifts the time axis (clip start)."""
    nper = int(round(win_s * sr))
    hop = max(1, int(round(hop_s * sr)))
    nfft = int(2 ** np.ceil(np.log2(nper * zero_pad)))
    f, t, Z = signal.stft(x, fs=sr, window="hann", nperseg=nper, noverlap=nper - hop, nfft=nfft,
                          boundary=None, padded=False, detrend="constant")
    P = np.abs(Z) ** 2
    sel = (f >= fmin) & (f <= fmax)
    S = 10 * np.log10(P[sel] + 1e-20).astype(np.float32)
    return Spectrogram(f[sel], t + t_offset, S, f[1] - f[0], hop / sr)


def clip_spectrogram(path, t_center: float, half_width: float = 90.0, sr: int = 4000,
                     **kw) -> Spectrogram:
    """Spectrogram of the clip [t_center - half_width, t_center + half_width] (audio clock)."""
    start = max(0.0, t_center - half_width)
    x = load_audio(path, sr=sr, start=start, duration=2 * half_width)
    return spectrogram(x, sr, t_offset=start, **kw)


def render(spec: Spectrogram, out: str | Path, title: str = "", overlays=(), vmin=None,
           vmax=None, figsize=(16, 7), whiten: bool = True):
    """Save a PNG of a spectrogram. overlays: iterable of (t, f, style_dict)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sp = spec.whiten() if whiten else spec
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    lo = vmin if vmin is not None else (0 if whiten else np.percentile(sp.S, 5))
    hi = vmax if vmax is not None else (5 if whiten else np.percentile(sp.S, 99.5))
    ax.pcolormesh(sp.t, sp.f, sp.S, vmin=lo, vmax=hi, cmap="magma", shading="auto")
    for t, f, style in overlays:
        ax.plot(t, f, **({"lw": 1.0, "color": "cyan", "alpha": 0.8} | dict(style)))
    ax.set_xlim(sp.t[0], sp.t[-1])
    ax.set_ylim(sp.f[0], sp.f[-1])
    ax.set_xlabel("audio time (s)")
    ax.set_ylabel("Hz")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
