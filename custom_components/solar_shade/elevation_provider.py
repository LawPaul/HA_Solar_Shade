"""Elevation data provider abstraction for Solar Shade.

Each provider implements tile discovery and coordinate transforms for a
specific national LiDAR data source.  The LAZ rasterization pipeline is
shared across all providers (lives in usgs_downloader.py).

Supported providers:
- usgs       — USGS 3DEP (United States)
- ign        — IGN LiDAR HD (France)
- swisstopo  — swisstopo swissSURFACE3D (Switzerland)
- nrw        — NRW 3D-Messdaten (Germany)
- pdok       — PDOK 3D Basisvoorziening (Netherlands)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import aiohttp

_LOGGER = logging.getLogger(__name__)

# ── Provider registry (populated by __init_subclass__) ───────────────────

_PROVIDERS: dict[str, type] = {}


class ElevationProvider(ABC):
    """Base class for national elevation data providers."""

    PROVIDER_ID: str = ""
    PROVIDER_NAME: str = ""
    NATIVE_EPSG: int = 0

    # ISO 3166-1 alpha-2 country codes this provider covers.
    # Used by auto-detection via hass.config.country.
    # Empty tuple means this provider won't be auto-selected
    # (e.g. USGS is the global fallback).
    COUNTRY_CODES: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.PROVIDER_ID:
            _PROVIDERS[cls.PROVIDER_ID] = cls

    @abstractmethod
    def latlon_to_native(
        self, latitude: float, longitude: float,
    ) -> tuple[float, float]:
        """Convert WGS84 lat/lon to this provider's native CRS.

        Returns (easting, northing) in the native coordinate system.
        """

    @abstractmethod
    async def find_tiles(
        self,
        latitude: float,
        longitude: float,
        session: aiohttp.ClientSession,
    ) -> list[dict]:
        """Find elevation tile URLs covering a lat/lon.

        Returns a list of dicts with at least 'url' key, plus optional
        'title', 'date', 'size'.  Sorted by preference (best first).
        """

    async def download_elevation(
        self,
        latitude: float,
        longitude: float,
        radius_m: float = 150.0,
        min_cell_size: float = 0.5,
        dsm_gap_fill: bool = False,
    ) -> tuple | None:
        """Download and rasterize elevation data for a location.

        Returns the same tuple as usgs_downloader.download_usgs_dsm:
        (dsm, dtm, cls_grid, canopy_base,
         x_min, y_min, x_max, y_max, resolution)
        or None if no data is available.
        """
        from .usgs_downloader import download_and_rasterize_laz

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
                result = await download_and_rasterize_laz(
                    center_easting=center_e,
                    center_northing=center_n,
                    radius_m=radius_m,
                    session=session,
                    laz_url=url,
                    min_cell_size=min_cell_size,
                    dsm_gap_fill=dsm_gap_fill,
                    expected_epsg=self.NATIVE_EPSG,
                )
                if result is not None:
                    return result

            _LOGGER.warning(
                "%s: all tile download attempts failed", self.PROVIDER_NAME,
            )
            return None


def get_provider(provider_id: str) -> ElevationProvider:
    """Instantiate a provider by ID.  Raises KeyError if unknown."""
    # Ensure subclasses are imported so they self-register
    _ensure_providers_loaded()
    cls = _PROVIDERS[provider_id]
    return cls()


def list_providers() -> list[tuple[str, str]]:
    """Return [(provider_id, display_name), ...] for all registered providers."""
    _ensure_providers_loaded()
    return [(pid, cls.PROVIDER_NAME) for pid, cls in sorted(_PROVIDERS.items())]


def detect_provider(country: str | None) -> str:
    """Auto-detect the best provider for a country.

    Uses the ISO 3166-1 alpha-2 country code (e.g. from
    ``hass.config.country``) to find a matching provider.
    Falls back to 'usgs' if no provider covers the country.
    """
    _ensure_providers_loaded()

    if country:
        cc = country.upper()
        for pid, cls in _PROVIDERS.items():
            if cc in cls.COUNTRY_CODES:
                return pid

    return "usgs"


_PROVIDERS_LOADED = False


def _ensure_providers_loaded():
    """Import all provider modules so their subclasses self-register.

    Scans for *_provider.py files in this package directory so that adding
    a new provider doesn't require editing this function.
    """
    global _PROVIDERS_LOADED
    if _PROVIDERS_LOADED:
        return
    _PROVIDERS_LOADED = True

    import importlib
    import pathlib

    pkg_dir = pathlib.Path(__file__).parent
    for mod_path in sorted(pkg_dir.glob("*_provider.py")):
        module_name = mod_path.stem
        try:
            importlib.import_module(f".{module_name}", __package__)
        except (ImportError, ModuleNotFoundError):
            _LOGGER.debug("Could not load provider module %s", module_name)

    # Also load USGSProvider from usgs_downloader
    try:
        from . import usgs_downloader  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        pass
