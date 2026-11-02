"""ctypes bindings for native/dopploc_native.cu (CUDA + OpenMP). Falls back to numba when the
library is not built (``make -C native`` or ``make -C native cpu``).

DOPPLOC_DEVICE: "cuda" (default when a GPU is present), "cpu" (OpenMP), "numba" (disable native).
DOPPLOC_NATIVE: explicit path to libdopploc_native.so.
"""

from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import numpy as np

_lib = None
_dev = None


def _load():
    global _lib, _dev
    if _lib is not None or _dev == "numba":
        return _lib
    mode = os.environ.get("DOPPLOC_DEVICE", "").lower()
    if mode == "numba":
        _dev = "numba"
        return None
    path = os.environ.get("DOPPLOC_NATIVE") or str(Path(__file__).parent / "_native" / "libdopploc_native.so")
    if not Path(path).exists():
        _dev = "numba"
        return None
    lib = C.CDLL(path)
    P = np.ctypeslib.ndpointer
    d, f, i32, i64 = (P(np.float64, flags="C"), P(np.float32, flags="C"), P(np.int32, flags="C"),
                      P(np.int64, flags="C"))
    lib.dl_cuda_device_count.restype = C.c_int
    lib.dl_model.argtypes = [d, d, i32, C.c_int64, d, d, i64, i32, d, d, C.c_int32, C.c_int64,
                             C.c_double, C.c_double, C.c_int32, C.c_double, C.c_double, d, f, d, d, C.c_int]
    lib.dl_score_brightness.argtypes = [f, C.c_int64, C.c_int32, C.c_double, C.c_double, d, d, C.c_int32, d,
                                        C.c_int32, d, C.c_int32, d, C.c_int32, f, C.c_int32, C.c_double,
                                        C.c_double, f, i32, C.c_int]
    lib.dl_score_trace.argtypes = [f, C.c_int64, C.c_int32, C.c_double, C.c_double, d, d, C.c_int32, d,
                                   C.c_int32, d, d, d, C.c_int32, C.c_double, f, i32, C.c_int]
    for fn in (lib.dl_model, lib.dl_score_brightness, lib.dl_score_trace):
        fn.restype = C.c_int
    _lib = lib
    _dev = "cpu" if mode == "cpu" or lib.dl_cuda_device_count() == 0 else "cuda"
    return lib


def backend() -> str:
    _load()
    return _dev or "numba"


def _device():
    return 0 if backend() == "cuda" else -1


def _c(a, dt):
    return np.ascontiguousarray(a, dtype=dt)


class TrackPack:
    """Many resampled tracks concatenated for the native model."""

    def __init__(self, sampled: dict):
        self.ids = list(sampled)
        self.index = {k: i for i, k in enumerate(self.ids)}
        lens = np.array([len(sampled[k].gs) for k in self.ids], np.int32)
        self.start = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        self.len = lens
        self.t0 = np.array([sampled[k].t0 for k in self.ids], np.float64)
        self.dt = np.array([sampled[k].dt for k in self.ids], np.float64)
        self.P = _c(np.concatenate([sampled[k].P for k in self.ids]), np.float64)
        self.V = _c(np.concatenate([sampled[k].V for k in self.ids]), np.float64)


def model(pack: TrackPack, fids, obs, t_expect, u_grid, c, win=90.0, atm=None):
    lib = _load()
    n = len(t_expect)
    fidx = _c([pack.index[f] for f in fids], np.int32)
    R = np.empty((n, len(u_grid)), np.float32)
    t_cpa = np.empty(n)
    d_cpa = np.empty(n)
    rc = lib.dl_model(_c(obs, np.float64), _c(t_expect, np.float64), fidx, n, pack.P, pack.V, pack.start,
                      pack.len, pack.t0, pack.dt, len(pack.ids), len(pack.P), float(u_grid[0]),
                      float(u_grid[1] - u_grid[0]), len(u_grid), float(c), float(win),
                      _c(np.tile([c, 0, 0, 0, c, 0, 0, 0], (n, 1)) if atm is None else atm, np.float64), R, t_cpa,
                      d_cpa, _device())
    if rc:
        raise RuntimeError(f"dl_model failed ({rc})")
    return t_cpa, d_cpa, R


def score_brightness(R, u0, du, base, deltas, f0s, harms, tau, B, fb0, dfb):
    lib = _load()
    n, nu = R.shape
    nd = len(deltas)
    S = np.empty((n, nd), np.float32)
    A = np.empty((n, nd), np.int32)
    rc = lib.dl_score_brightness(_c(R, np.float32), n, nu, u0, du, _c(base, np.float64), _c(deltas, np.float64), nd,
                                 _c(f0s, np.float64), len(f0s), _c(harms, np.float64), len(harms),
                                 _c(tau, np.float64), len(tau), _c(B, np.float32), B.shape[0], fb0, dfb, S, A,
                                 _device())
    if rc:
        raise RuntimeError(f"dl_score_brightness failed ({rc})")
    return S, A


def score_trace(R, u0, du, base, deltas, f0s, tt, tf, th, sigma):
    lib = _load()
    n, nu = R.shape
    nd = len(deltas)
    S = np.empty((n, nd), np.float32)
    A = np.empty((n, nd), np.int32)
    rc = lib.dl_score_trace(_c(R, np.float32), n, nu, u0, du, _c(base, np.float64), _c(deltas, np.float64), nd,
                            _c(f0s, np.float64), len(f0s), _c(tt, np.float64), _c(tf, np.float64),
                            _c(th, np.float64), len(tt), sigma, S, A, _device())
    if rc:
        raise RuntimeError(f"dl_score_trace failed ({rc})")
    return S, A
