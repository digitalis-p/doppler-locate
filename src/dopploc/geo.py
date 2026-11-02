"""Geodesy helpers: WGS84 <-> ECEF, ENU velocity rotation, the CONUS Albers grid, geoid."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import requests

A_WGS84 = 6378137.0
F_WGS84 = 1 / 298.257223563
E2_WGS84 = F_WGS84 * (2 - F_WGS84)

GRID_CRS = "EPSG:5070"  # NAD83 / Conus Albers, metres; candidate grids live on this lattice
GEOID_URL = "https://cdn.proj.org/us_nga_egm96_15.tif"


def lla_to_ecef(lat, lon, h):
    """Geodetic (deg, deg, m) to ECEF metres. Arrays broadcast."""
    lat = np.radians(lat)
    lon = np.radians(lon)
    s, c = np.sin(lat), np.cos(lat)
    N = A_WGS84 / np.sqrt(1 - E2_WGS84 * s * s)
    x = (N + h) * c * np.cos(lon)
    y = (N + h) * c * np.sin(lon)
    z = (N * (1 - E2_WGS84) + h) * s
    return np.stack([x, y, z], axis=-1)


def enu_to_ecef_vec(lat, lon, e, n, u):
    """Rotate local east/north/up vectors at (lat, lon) into ECEF components."""
    lat = np.radians(lat)
    lon = np.radians(lon)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    x = -so * e - sl * co * n + cl * co * u
    y = co * e - sl * so * n + cl * so * u
    z = cl * n + sl * u
    return np.stack([x, y, z], axis=-1)


@lru_cache(maxsize=None)
def _transformers():
    from pyproj import Transformer
    fwd = Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True)
    inv = Transformer.from_crs(GRID_CRS, "EPSG:4326", always_xy=True)
    return fwd, inv


def to_grid(lon, lat):
    """lon/lat (deg) -> Albers x/y (m)."""
    fwd, _ = _transformers()
    return fwd.transform(np.asarray(lon, float), np.asarray(lat, float))


def from_grid(x, y):
    """Albers x/y (m) -> lon/lat (deg)."""
    _, inv = _transformers()
    return inv.transform(np.asarray(x, float), np.asarray(y, float))


def cell_centers(ix, iy, spacing):
    """Centre coordinates of lattice cells (integer indices) at ``spacing`` metres."""
    return (np.asarray(ix) + 0.5) * spacing, (np.asarray(iy) + 0.5) * spacing


class Geoid:
    """EGM96 undulation N (m): orthometric height = ellipsoidal height - N."""

    def __init__(self, cache_dir: Path):
        self.path = Path(cache_dir) / "us_nga_egm96_15.tif"
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(GEOID_URL, timeout=120)
            r.raise_for_status()
            self.path.write_bytes(r.content)
        import rasterio
        with rasterio.open(self.path) as ds:
            self.grid = ds.read(1).astype(np.float64)
            self.tr = ds.transform

    def __call__(self, lat, lon):
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        col = (lon - self.tr.c) / self.tr.a - 0.5
        row = (lat - self.tr.f) / self.tr.e - 0.5
        g = self.grid
        c0 = np.clip(np.floor(col).astype(int), 0, g.shape[1] - 2)
        r0 = np.clip(np.floor(row).astype(int), 0, g.shape[0] - 2)
        wc = np.clip(col - c0, 0, 1)
        wr = np.clip(row - r0, 0, 1)
        return ((1 - wr) * (1 - wc) * g[r0, c0] + (1 - wr) * wc * g[r0, c0 + 1]
                + wr * (1 - wc) * g[r0 + 1, c0] + wr * wc * g[r0 + 1, c0 + 1])
