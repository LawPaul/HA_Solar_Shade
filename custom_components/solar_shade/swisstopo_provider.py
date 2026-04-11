"""Switzerland swisstopo swissSURFACE3D elevation data provider.

Uses the swisstopo STAC API to discover and download swissSURFACE3D
LAS point cloud tiles (DSM — includes buildings, trees, etc.).

Tile discovery via STAC:
  https://data.geo.admin.ch/api/stac/v0.9/collections/
    ch.swisstopo.swisssurface3d/items?bbox={lon_min},{lat_min},{lon_max},{lat_max}

Each feature's assets contain `.las.zip` files — compressed LAS point clouds
at ~15–30 pts/m² in 1 km × 1 km tiles.  No API key required.

Native CRS: EPSG:2056 (CH1903+ / LV95).
"""

from __future__ import annotations

import logging

import aiohttp

from .elevation_provider import ElevationProvider
from .geo import latlon_to_epsg

_LOGGER = logging.getLogger(__name__)

_STAC_URL = (
    "https://data.geo.admin.ch/api/stac/v0.9/collections/"
    "ch.swisstopo.swisssurface3d/items"
)

# Search box half-side in degrees (~500 m)
_BBOX_PAD = 0.005


class SwisstopoProvider(ElevationProvider):
    """swisstopo swissSURFACE3D (Switzerland) elevation data provider."""

    PROVIDER_ID = "swisstopo"
    PROVIDER_NAME = "swisstopo swissSURFACE3D (Switzerland)"
    NATIVE_EPSG = 2056  # CH1903+ / LV95
    COUNTRY_CODES = ("CH", "LI")

    def latlon_to_native(
        self, latitude: float, longitude: float,
    ) -> tuple[float, float]:
        return latlon_to_epsg(latitude, longitude, self.NATIVE_EPSG)

    async def find_tiles(
        self,
        latitude: float,
        longitude: float,
        session: aiohttp.ClientSession,
    ) -> list[dict]:
        """Query swisstopo STAC for swissSURFACE3D tiles."""
        lon_min = longitude - _BBOX_PAD
        lat_min = latitude - _BBOX_PAD
        lon_max = longitude + _BBOX_PAD
        lat_max = latitude + _BBOX_PAD
        url = f"{_STAC_URL}?bbox={lon_min},{lat_min},{lon_max},{lat_max}&limit=10"
        _LOGGER.debug("swisstopo STAC query: %s", url)

        async with session.get(url) as resp:
            if resp.status != 200:
                _LOGGER.warning("swisstopo STAC returned HTTP %s", resp.status)
                return []
            data = await resp.json(content_type=None)

        features = data.get("features", [])
        if not features:
            _LOGGER.info(
                "swisstopo: no tiles at %.4f, %.4f", latitude, longitude,
            )
            return []

        tiles: list[dict] = []
        for feat in features:
            assets = feat.get("assets", {})
            # Pick the .las.zip asset (there's usually exactly one)
            for asset_key, asset_val in assets.items():
                href = asset_val.get("href", "")
                if href.endswith(".las.zip"):
                    tiles.append({
                        "url": href,
                        "title": feat.get("id", ""),
                    })
                    break

        _LOGGER.info(
            "swisstopo: found %d tile(s) at %.4f, %.4f",
            len(tiles), latitude, longitude,
        )
        return tiles

    async def download_elevation(
        self,
        latitude: float,
        longitude: float,
        radius_m: float = 150.0,
        min_cell_size: float = 0.5,
        dsm_gap_fill: bool = False,
    ) -> tuple | None:
        """Download and rasterize — handles .las.zip extraction."""
        center_e, center_n = self.latlon_to_native(latitude, longitude)

        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tiles = await self.find_tiles(latitude, longitude, session)
            if not tiles:
                _LOGGER.warning(
                    "%s: no tiles found at %.4f, %.4f",
                    self.PROVIDER_NAME, latitude, longitude,
                )
                return None

            for tile_info in tiles:
                url = tile_info["url"]
                _LOGGER.info(
                    "%s: attempting download: %s",
                    self.PROVIDER_NAME, url.rsplit("/", 1)[-1],
                )
                result = await self._download_and_process(
                    session, url, center_e, center_n,
                    radius_m, min_cell_size, dsm_gap_fill,
                )
                if result is not None:
                    return result

            _LOGGER.warning(
                "%s: all tile download attempts failed", self.PROVIDER_NAME,
            )
            return None

    async def _download_and_process(
        self,
        session: aiohttp.ClientSession,
        url: str,
        center_e: float,
        center_n: float,
        radius_m: float,
        min_cell_size: float,
        dsm_gap_fill: bool,
    ) -> tuple | None:
        """Download a .las.zip, extract, and rasterize."""
        import asyncio
        import os
        import tempfile
        import zipfile

        from .usgs_downloader import _rasterize_laz_file

        filename = url.rsplit("/", 1)[-1]
        _LOGGER.info("swisstopo: downloading %s", filename)

        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=600),
            ) as resp:
                if resp.status == 404:
                    _LOGGER.info("swisstopo: tile URL returned 404: %s", url)
                    return None
                resp.raise_for_status()

                tmp_fd, tmp_zip = tempfile.mkstemp(suffix=".las.zip")
                try:
                    total_bytes = 0
                    with os.fdopen(tmp_fd, "wb") as f:
                        async for chunk in resp.content.iter_chunked(1024 * 1024):
                            f.write(chunk)
                            total_bytes += len(chunk)
                    _LOGGER.info(
                        "swisstopo: downloaded %.1f MB", total_bytes / 1024 / 1024,
                    )
                except (OSError, aiohttp.ClientError):
                    os.unlink(tmp_zip)
                    raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as err:
            _LOGGER.warning("swisstopo download failed: %s", err)
            return None

        # Extract LAS from ZIP
        tmp_las = None
        try:
            with zipfile.ZipFile(tmp_zip, "r") as zf:
                las_names = [n for n in zf.namelist() if n.lower().endswith(".las")]
                if not las_names:
                    _LOGGER.warning("swisstopo: no .las file in ZIP")
                    return None
                tmp_las = tempfile.mktemp(suffix=".las")
                with zf.open(las_names[0]) as src, open(tmp_las, "wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
            _LOGGER.info("swisstopo: extracted %s from ZIP", las_names[0])
        except (OSError, zipfile.BadZipFile, KeyError) as err:
            _LOGGER.warning("swisstopo: ZIP extraction failed: %s", err)
            return None
        finally:
            try:
                os.unlink(tmp_zip)
            except OSError:
                pass

        # Rasterize in executor (CPU-bound)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                _rasterize_laz_file,
                tmp_las,
                center_e,
                center_n,
                radius_m,
                min_cell_size,
                dsm_gap_fill,
                self.NATIVE_EPSG,
            )
        finally:
            try:
                os.unlink(tmp_las)
            except OSError:
                pass
        return result
