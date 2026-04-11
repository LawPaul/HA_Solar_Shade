"""France IGN LiDAR HD elevation data provider.

Uses the French national LiDAR HD program (Géoplateforme) to discover
and download COPC LAZ tiles.  Coverage is growing as IGN scans all of
France at 10 pulses/m² in 1 km × 1 km tiles.

Tile discovery via WFS:
  https://data.geopf.fr/wfs/ows?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature
    &TYPENAMES=IGNF_NUAGES-DE-POINTS-LIDAR-HD:dalle
    &OUTPUTFORMAT=json&BBOX={lat_min},{lon_min},{lat_max},{lon_max}

Each feature's ``properties.url`` is a direct anonymous URL to a COPC LAZ
file (~200 MB per tile).  No API key required.

Native CRS: EPSG:2154 (RGF93 v1 / Lambert-93).
"""

from __future__ import annotations

import logging

import aiohttp

from .elevation_provider import ElevationProvider
from .geo import latlon_to_epsg

_LOGGER = logging.getLogger(__name__)

_WFS_BASE = (
    "https://data.geopf.fr/wfs/ows"
    "?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
    "&TYPENAMES=IGNF_NUAGES-DE-POINTS-LIDAR-HD:dalle"
    "&OUTPUTFORMAT=json"
)

# Search box half-side in degrees (~500 m at 45° lat)
_BBOX_PAD = 0.005


class IGNProvider(ElevationProvider):
    """IGN LiDAR HD (France) elevation data provider."""

    PROVIDER_ID = "ign"
    PROVIDER_NAME = "IGN LiDAR HD (France)"
    NATIVE_EPSG = 2154  # RGF93 v1 / Lambert-93
    COUNTRY_CODES = ("FR", "GP", "MQ", "GF", "RE", "YT", "PM")

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
        """Query IGN WFS for LiDAR HD tiles covering the location."""
        bbox = (
            f"{latitude - _BBOX_PAD},{longitude - _BBOX_PAD},"
            f"{latitude + _BBOX_PAD},{longitude + _BBOX_PAD}"
        )
        url = f"{_WFS_BASE}&BBOX={bbox}"
        _LOGGER.debug("IGN WFS query: %s", url)

        async with session.get(url) as resp:
            if resp.status != 200:
                _LOGGER.warning("IGN WFS returned HTTP %s", resp.status)
                return []
            data = await resp.json(content_type=None)

        features = data.get("features", [])
        if not features:
            _LOGGER.info("IGN: no LiDAR HD tiles at %.4f, %.4f", latitude, longitude)
            return []

        tiles: list[dict] = []
        for feat in features:
            props = feat.get("properties", {})
            tile_url = props.get("url")
            if not tile_url:
                continue
            tiles.append({
                "url": tile_url,
                "title": props.get("nom_pkk", ""),
                "date": props.get("date_vol", ""),
            })

        _LOGGER.info(
            "IGN: found %d tile(s) at %.4f, %.4f", len(tiles), latitude, longitude,
        )
        return tiles
