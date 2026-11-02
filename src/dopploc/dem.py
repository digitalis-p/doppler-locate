"""Copernicus DEM (GLO-30 / GLO-90) tiles from the public AWS bucket, cached on disk.

Heights are EGM2008 orthometric metres (~ mean sea level). Ocean tiles do not exist in
the bucket; those read as 0 m.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

_SRC = {
    30: ("https://copernicus-dem-30m.s3.amazonaws.com", "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM"),
    90: ("https://copernicus-dem-90m.s3.amazonaws.com", "Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM"),
}


def tile_name(lat_floor: int, lon_floor: int, res: int) -> str:
    _, pat = _SRC[res]
    return pat.format(ns="N" if lat_floor >= 0 else "S", lat=abs(lat_floor),
                      ew="E" if lon_floor >= 0 else "W", lon=abs(lon_floor))


class DEM:
    def __init__(self, cache_dir: Path, res: int = 90, max_tiles_in_memory: int = 8):
        self.dir = Path(cache_dir) / f"glo{res}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.res = res
        self.mem: OrderedDict = OrderedDict()
        self.max = max_tiles_in_memory
        self.lock = threading.Lock()

    def _paths(self, la, lo):
        name = tile_name(la, lo, self.res)
        return self.dir / f"{name}.tif", self.dir / f"{name}.missing"

    def fetch(self, la: int, lo: int) -> Path | None:
        tif, missing = self._paths(la, lo)
        if tif.exists():
            return tif
        if missing.exists():
            return None
        base, _ = _SRC[self.res]
        name = tile_name(la, lo, self.res)
        for attempt in range(6):
            try:
                r = requests.get(f"{base}/{name}/{name}.tif", timeout=300)
                if r.status_code in (403, 404):
                    missing.touch()
                    return None
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        tmp = tif.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.rename(tif)
        return tif

    def prefetch(self, tiles, workers: int = 16):
        """Download (lat_floor, lon_floor) tiles in parallel."""
        tiles = sorted(set(tiles))
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda t: self.fetch(*t), tiles))

    def _tile(self, la, lo):
        key = (la, lo)
        with self.lock:
            if key in self.mem:
                self.mem.move_to_end(key)
                return self.mem[key]
        path = self.fetch(la, lo)
        if path is None:
            val = None
        else:
            import rasterio
            with rasterio.Env(GDAL_CACHEMAX=32), rasterio.open(path) as ds:
                arr = ds.read(1).astype(np.float32)
                tr = ds.transform
            arr[~np.isfinite(arr) | (arr < -1000)] = 0.0
            val = (arr, tr)
        with self.lock:
            self.mem[key] = val
            while len(self.mem) > self.max:
                self.mem.popitem(last=False)
        return val

    def elevation(self, lat, lon) -> np.ndarray:
        """Bilinear elevation (m) at points; 0 over ocean / missing tiles."""
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        out = np.zeros(lat.shape, dtype=np.float64)
        la = np.floor(lat).astype(int)
        lo = np.floor(lon).astype(int)
        key = la * 1000 + lo
        need = {(int(a), int(b)) for a, b in np.unique(np.stack([la.ravel(), lo.ravel()], 1), axis=0)}
        if len(need) > 1:
            self.prefetch(need)
        for k in np.unique(key):
            sel = key == k
            t = self._tile(int(la[sel][0]), int(lo[sel][0]))
            if t is None:
                continue
            arr, tr = t
            col = (lon[sel] - tr.c) / tr.a - 0.5
            row = (lat[sel] - tr.f) / tr.e - 0.5
            c0 = np.clip(np.floor(col).astype(int), 0, arr.shape[1] - 2)
            r0 = np.clip(np.floor(row).astype(int), 0, arr.shape[0] - 2)
            wc = np.clip(col - c0, 0, 1)
            wr = np.clip(row - r0, 0, 1)
            out[sel] = ((1 - wr) * (1 - wc) * arr[r0, c0] + (1 - wr) * wc * arr[r0, c0 + 1]
                        + wr * (1 - wc) * arr[r0 + 1, c0] + wr * wc * arr[r0 + 1, c0 + 1])
        return out


class ElevationLattice:
    """Terrain height on the search's Albers lattice (cell centres at (i + .5) g), cached as an
    int16 memmap. Building it once (~3 min from GLO-90) replaces per-tile DEM sampling."""

    def __init__(self, path: Path):
        meta = np.load(Path(path).with_suffix(".meta.npy"))
        self.gx0, self.gy0, self.g = int(meta[0]), int(meta[1]), float(meta[2])
        self.a = np.load(path, mmap_mode="r")  # (nx, ny): [x index, y index]

    @staticmethod
    def path_for(cache: Path, g: float) -> Path:
        return Path(cache) / "elev" / f"conus_albers_{int(g)}m.npy"

    @classmethod
    def build(cls, cache: Path, g: float = 200.0, bbox=(-126.0, 23.5, -65.5, 50.5), log=print) -> "ElevationLattice":
        from . import geo
        path = cls.path_for(cache, g)
        if path.exists():
            return cls(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lon = np.r_[np.linspace(bbox[0], bbox[2], 400), np.linspace(bbox[0], bbox[2], 400),
                    np.full(200, bbox[0]), np.full(200, bbox[2])]
        lat = np.r_[np.full(400, bbox[1]), np.full(400, bbox[3]), np.linspace(bbox[1], bbox[3], 200),
                    np.linspace(bbox[1], bbox[3], 200)]
        x, y = geo.to_grid(lon, lat)
        gx0, gx1 = int(np.floor(np.min(x) / g)), int(np.ceil(np.max(x) / g))
        gy0, gy1 = int(np.floor(np.min(y) / g)), int(np.ceil(np.max(y) / g))
        nx, ny = gx1 - gx0, gy1 - gy0
        log(f"  building elevation lattice {nx} x {ny} at {g:g} m -> {path}")
        dem = DEM(Path(cache) / "dem", res=90, max_tiles_in_memory=160)
        tmp = path.with_suffix(".part.npy")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.int16, shape=(nx, ny))
        ys = (gy0 + np.arange(ny) + 0.5) * g
        step = 200
        for i0 in range(0, nx, step):
            i1 = min(nx, i0 + step)
            X, Y = np.meshgrid((gx0 + np.arange(i0, i1) + 0.5) * g, ys, indexing="ij")
            lo, la = geo.from_grid(X.ravel(), Y.ravel())
            inside = (la >= bbox[1] - 0.5) & (la <= bbox[3] + 0.5) & (lo >= bbox[0] - 0.5) & (lo <= bbox[2] + 0.5)
            e = np.zeros(X.size)
            e[inside] = dem.elevation(la[inside], lo[inside])
            out[i0:i1] = np.clip(np.round(e), -32000, 32000).astype(np.int16).reshape(i1 - i0, ny)
            if (i0 // step) % 20 == 0:
                log(f"    columns {i0}/{nx}")
        out.flush()
        del out
        np.save(path.with_suffix(".meta.npy"), np.array([gx0, gy0, g]))
        tmp.rename(path)
        return cls(path)

    def block(self, gx, gy, n):
        """(n, n) float array for lattice cells gx..gx+n-1, gy..gy+n-1 (0 outside the lattice)."""
        out = np.zeros((n, n), np.float64)
        i0, j0 = gx - self.gx0, gy - self.gy0
        a0, b0 = max(0, i0), max(0, j0)
        a1, b1 = min(self.a.shape[0], i0 + n), min(self.a.shape[1], j0 + n)
        if a1 > a0 and b1 > b0:
            out[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = self.a[a0:a1, b0:b1]
        return out
