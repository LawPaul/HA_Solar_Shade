"""Germany NRW 3D-Messdaten (Open.NRW) elevation data provider.

Uses the Open.NRW open-data portal to download 3D-Messdaten LAZ
point cloud tiles covering the state of North Rhine-Westphalia.

Tile URL pattern (1 km × 1 km tiles):
  https://www.opengeodata.nrw.de/produkte/geobasis/hm/3dm_l_las/3dm_l_las/
    3dm_32_{easting_km}_{northing_km}_1_nw.laz

Tile coordinates are in EPSG:25832 (ETRS89 / UTM zone 32N),
truncated to whole kilometres.

No API key required.  Anonymous HTTP download.
"""

from __future__ import annotations

import logging

import aiohttp

from .elevation_provider import ElevationProvider
from .geo import latlon_to_epsg

_LOGGER = logging.getLogger(__name__)

_BASE_URL = (
    "https://www.opengeodata.nrw.de/produkte/geobasis/hm/"
    "3dm_l_las/3dm_l_las"
)


class NRWProvider(ElevationProvider):
    """NRW 3D-Messdaten (Germany) elevation data provider."""

    PROVIDER_ID = "nrw"
    PROVIDER_NAME = "NRW 3D-Messdaten (Germany)"
    NATIVE_EPSG = 25832  # ETRS89 / UTM zone 32N
    COUNTRY_CODES = ("DE",)

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
        """Compute NRW tile URL from coordinates and verify it exists."""
        easting, northing = self.latlon_to_native(latitude, longitude)
        e_km = int(easting / 1000)
        n_km = int(northing / 1000)

        tile_name = f"3dm_32_{e_km}_{n_km}_1_nw.laz"
        tile_url = f"{_BASE_URL}/{tile_name}"

        _LOGGER.debug("NRW tile URL: %s", tile_url)

        # Verify the tile exists (HEAD request) — NRW only covers
        # North Rhine-Westphalia, not all of Germany.
        try:
            async with session.head(tile_url) as resp:
                if resp.status == 200:
                    _LOGGER.info(
                        "NRW: tile %s exists (%.4f, %.4f)",
                        tile_name, latitude, longitude,
                    )
                    return [{"url": tile_url, "title": tile_name}]
                _LOGGER.info(
                    "NRW: tile %s not found (HTTP %d) — location may be "
                    "outside NRW coverage",
                    tile_name, resp.status,
                )
        except aiohttp.ClientError as err:
            _LOGGER.warning("NRW: HEAD request failed: %s", err)

        return []
