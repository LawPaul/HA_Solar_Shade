"""Netherlands PDOK 3D Basisvoorziening elevation data provider.

Uses the Dutch PDOK (Publieke Dienstverlening Op de Kaart) OGC API
to discover and download DSM LAZ tiles from the
3D Basisvoorziening / AHN dataset.

Tile discovery via OGC API Features:
  https://api.pdok.nl/kadaster/3d-basisvoorziening/ogc/v1_0/
    collections/digitaaloppervlaktemodel_20cm/items?bbox=...

Each feature's ``properties.download_link`` is a direct URL to a LAZ
file (~3 MB) containing point cloud DSM data at 20 cm resolution.

No API key required.  Anonymous HTTP download.

Native CRS: EPSG:28992 (Amersfoort / RD New).
"""

from __future__ import annotations

import logging

import aiohttp

from .elevation_provider import ElevationProvider
from .geo import latlon_to_epsg

_LOGGER = logging.getLogger(__name__)

_OGC_API_URL = (
    "https://api.pdok.nl/kadaster/3d-basisvoorziening/ogc/v1_0/"
    "collections/digitaaloppervlaktemodel_20cm/items"
)

# Search box half-side in degrees (~500 m at 52° lat)
_BBOX_PAD = 0.005


class PDOKProvider(ElevationProvider):
    """PDOK 3D Basisvoorziening (Netherlands) elevation data provider."""

    PROVIDER_ID = "pdok"
    PROVIDER_NAME = "PDOK 3D Basisvoorziening (Netherlands)"
    NATIVE_EPSG = 28992  # Amersfoort / RD New
    COUNTRY_CODES = ("NL",)

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
        """Query PDOK OGC API for DSM tiles covering the location."""
        bbox = (
            f"{longitude - _BBOX_PAD},{latitude - _BBOX_PAD},"
            f"{longitude + _BBOX_PAD},{latitude + _BBOX_PAD}"
        )
        url = f"{_OGC_API_URL}?bbox={bbox}&limit=10"
        _LOGGER.debug("PDOK OGC API query: %s", url)

        async with session.get(url) as resp:
            if resp.status != 200:
                _LOGGER.warning("PDOK API returned HTTP %s", resp.status)
                return []
            data = await resp.json(content_type=None)

        features = data.get("features", [])
        if not features:
            _LOGGER.info(
                "PDOK: no DSM tiles at %.4f, %.4f", latitude, longitude,
            )
            return []

        tiles: list[dict] = []
        for feat in features:
            props = feat.get("properties", {})
            dl_link = props.get("download_link")
            if not dl_link:
                continue
            tiles.append({
                "url": dl_link,
                "title": props.get("bladnr", ""),
                "size": props.get("download_size_bytes", 0),
            })

        _LOGGER.info(
            "PDOK: found %d tile(s) at %.4f, %.4f",
            len(tiles), latitude, longitude,
        )
        return tiles
