"""Integration tests for international elevation data providers.

These tests hit real download servers and verify the full end-to-end path
from lat/lon → tile discovery → LAZ download → rasterization → SiteModel.

Run with: pytest tests/test_provider_integration.py -m integration -v -s

Skipped by default (addopts = -m "not integration" in pytest.ini).

WARNING: These download real LiDAR tiles (50–500 MB each) and take minutes
to run.  Only run when you need to verify provider connectivity and the
full rasterization pipeline.
"""

import asyncio
import json
import os
import tempfile
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _restore_real_aiohttp():
    """Un-mock aiohttp so integration tests can make real HTTP requests.

    conftest.py replaces sys.modules["aiohttp"] with a MagicMock so that
    code importing homeassistant can load without HA installed.  For
    integration tests we need the real library.
    """
    import importlib
    import sys

    # Drop the mock and any cached submodules
    mocked_keys = [k for k in sys.modules if k == "aiohttp" or k.startswith("aiohttp.")]
    saved = {k: sys.modules.pop(k) for k in mocked_keys}

    # Import the real aiohttp
    real_aiohttp = importlib.import_module("aiohttp")
    sys.modules["aiohttp"] = real_aiohttp

    # Also reload the provider modules so they pick up real aiohttp
    for mod_name in list(sys.modules):
        if "elevation_provider" in mod_name or "ign_provider" in mod_name or "swisstopo_provider" in mod_name or "nrw_provider" in mod_name or "pdok_provider" in mod_name or "usgs_downloader" in mod_name or "shadow_engine" in mod_name:
            try:
                importlib.reload(sys.modules[mod_name])
            except Exception:
                pass

    yield

    # Restore mocks for other test modules
    for k, v in saved.items():
        sys.modules[k] = v


def _head_url(url: str) -> int:
    """HTTP HEAD request — returns status code."""
    req = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "SolarShade/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def _download_url(url: str, dest: str, timeout: int = 600) -> int:
    """Download a file from a URL, return bytes written."""
    req = urllib.request.Request(url, headers={"User-Agent": "SolarShade/1.0"})
    total = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
    return total


def _validate_site_model(site, expected_epsg: int, lat: float, lon: float):
    """Common assertions for a SiteModel produced by any provider."""
    assert site is not None, "SiteModel is None — rasterization failed"
    assert site.dsm.shape[0] > 5, f"DSM too small: {site.dsm.shape}"
    assert site.dsm.shape[1] > 5, f"DSM too small: {site.dsm.shape}"
    assert site.dtm is not None, "DTM is None"
    assert site.classification is not None, "Classification grid is None"
    assert site.resolution > 0, f"Bad resolution: {site.resolution}"
    assert not np.all(np.isnan(site.dsm)), "DSM is all NaN"
    assert not np.all(np.isnan(site.dtm)), "DTM is all NaN"
    # DTM (ground) should generally be <= DSM (surface)
    valid = ~np.isnan(site.dsm) & ~np.isnan(site.dtm)
    if valid.any():
        assert float(np.mean(site.dtm[valid])) <= float(np.mean(site.dsm[valid])), \
            "DTM mean > DSM mean — ground above surface?"
    if expected_epsg:
        assert site.native_epsg == expected_epsg, \
            f"Expected EPSG:{expected_epsg}, got EPSG:{site.native_epsg}"
    assert site.latitude == pytest.approx(lat, abs=0.01)
    assert site.longitude == pytest.approx(lon, abs=0.01)


# ── France IGN integration tests ────────────────────────────────────────

FRANCE_LOCATIONS = {
    "Lyon": (45.76, 4.83),
    "Paris": (48.86, 2.35),
}


class TestIGNIntegration:
    """Integration tests for France IGN LiDAR HD — hit real servers."""

    def test_ign_wfs_lyon(self):
        """IGN WFS should return at least one LiDAR HD tile for Lyon."""
        lat, lon = FRANCE_LOCATIONS["Lyon"]
        pad = 0.005
        url = (
            "https://data.geopf.fr/wfs/ows"
            "?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
            "&TYPENAMES=IGNF_NUAGES-DE-POINTS-LIDAR-HD:dalle"
            "&OUTPUTFORMAT=json&COUNT=1"
            f"&BBOX={lat - pad},{lon - pad},{lat + pad},{lon + pad}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "SolarShade/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == 200
            data = json.loads(resp.read())
            features = data.get("features", [])
            assert len(features) >= 1
            assert "url" in features[0].get("properties", {})

    def test_ign_tile_head(self):
        """HEAD request on a known IGN tile URL should return 200."""
        lat, lon = FRANCE_LOCATIONS["Lyon"]
        pad = 0.005
        url = (
            "https://data.geopf.fr/wfs/ows"
            "?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
            "&TYPENAMES=IGNF_NUAGES-DE-POINTS-LIDAR-HD:dalle"
            "&OUTPUTFORMAT=json&COUNT=1"
            f"&BBOX={lat - pad},{lon - pad},{lat + pad},{lon + pad}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "SolarShade/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            tile_url = data["features"][0]["properties"]["url"]

        status = _head_url(tile_url)
        assert status == 200

    def test_ign_full_pipeline(self):
        """Download a real IGN tile, rasterize, and validate the SiteModel."""
        from custom_components.solar_shade.ign_provider import IGNProvider

        lat, lon = FRANCE_LOCATIONS["Lyon"]
        provider = IGNProvider()
        result = asyncio.run(
            provider.download_elevation(lat, lon, radius_m=50.0, min_cell_size=1.0)
        )
        assert result is not None, "IGN download_elevation returned None"
        dsm, dtm, cls_grid, canopy_base, x_min, y_min, x_max, y_max, resolution = result
        assert dsm.shape[0] > 5
        assert dtm.shape[0] > 5
        assert resolution > 0
        assert not np.all(np.isnan(dsm))
        # Ground should be below surface on average
        valid = ~np.isnan(dsm) & ~np.isnan(dtm)
        if valid.any():
            assert float(np.mean(dtm[valid])) <= float(np.mean(dsm[valid]))

    def test_ign_manual_file(self):
        """Simulate a user manually placing a French LAZ file.

        Downloads a real IGN tile, saves it to a temp file, and runs
        process_lidar_file() — the same path used when a user drops a
        .laz file into the data directory.
        """
        from custom_components.solar_shade.shadow_engine import process_lidar_file

        lat, lon = FRANCE_LOCATIONS["Lyon"]
        pad = 0.005
        # Discover tile URL
        wfs_url = (
            "https://data.geopf.fr/wfs/ows"
            "?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
            "&TYPENAMES=IGNF_NUAGES-DE-POINTS-LIDAR-HD:dalle"
            "&OUTPUTFORMAT=json&COUNT=1"
            f"&BBOX={lat - pad},{lon - pad},{lat + pad},{lon + pad}"
        )
        req = urllib.request.Request(wfs_url, headers={"User-Agent": "SolarShade/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        tile_url = data["features"][0]["properties"]["url"]

        # Download the tile to a temp file
        with tempfile.NamedTemporaryFile(suffix=".copc.laz", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            nbytes = _download_url(tile_url, tmp_path)
            assert nbytes > 1_000_000, f"Downloaded only {nbytes} bytes"

            # Process through the manual-file path
            site = process_lidar_file(
                tmp_path, latitude=lat, longitude=lon, min_cell_size=1.0,
            )
            _validate_site_model(site, expected_epsg=2154, lat=lat, lon=lon)
        finally:
            os.unlink(tmp_path)


# ── Switzerland swisstopo integration tests ─────────────────────────────

SWITZERLAND_LOCATIONS = {
    "Zurich": (47.37, 8.54),
    "Bern": (46.95, 7.45),
}


class TestSwisstopoIntegration:
    """Integration tests for swisstopo swissSURFACE3D — hit real servers."""

    def test_swisstopo_stac_zurich(self):
        """STAC should return tiles for Zurich."""
        lat, lon = SWITZERLAND_LOCATIONS["Zurich"]
        pad = 0.005
        url = (
            "https://data.geo.admin.ch/api/stac/v0.9/collections/"
            "ch.swisstopo.swisssurface3d/items"
            f"?bbox={lon - pad},{lat - pad},{lon + pad},{lat + pad}"
            "&limit=3"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "SolarShade/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == 200
            data = json.loads(resp.read())
            features = data.get("features", [])
            assert len(features) >= 1
            # Should have a .las.zip asset
            assets = features[0].get("assets", {})
            las_urls = [
                v["href"]
                for v in assets.values()
                if v.get("href", "").endswith(".las.zip")
            ]
            assert len(las_urls) >= 1

    def test_swisstopo_tile_head(self):
        """HEAD request on a swisstopo .las.zip tile should return 200."""
        lat, lon = SWITZERLAND_LOCATIONS["Zurich"]
        pad = 0.005
        url = (
            "https://data.geo.admin.ch/api/stac/v0.9/collections/"
            "ch.swisstopo.swisssurface3d/items"
            f"?bbox={lon - pad},{lat - pad},{lon + pad},{lat + pad}"
            "&limit=1"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "SolarShade/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            assets = data["features"][0]["assets"]
            tile_url = next(
                v["href"]
                for v in assets.values()
                if v.get("href", "").endswith(".las.zip")
            )

        status = _head_url(tile_url)
        assert status == 200

    def test_swisstopo_full_pipeline(self):
        """Download a real swisstopo tile, rasterize, validate SiteModel."""
        from custom_components.solar_shade.swisstopo_provider import SwisstopoProvider

        lat, lon = SWITZERLAND_LOCATIONS["Zurich"]
        provider = SwisstopoProvider()
        result = asyncio.run(
            provider.download_elevation(lat, lon, radius_m=50.0, min_cell_size=1.0)
        )
        assert result is not None, "swisstopo download_elevation returned None"
        dsm, dtm, cls_grid, canopy_base, x_min, y_min, x_max, y_max, resolution = result
        assert dsm.shape[0] > 5
        assert dtm.shape[0] > 5
        assert resolution > 0
        assert not np.all(np.isnan(dsm))

    def test_swisstopo_manual_file(self):
        """Simulate a user manually placing a Swiss LAS file.

        Downloads a real .las.zip, extracts the LAS, and runs
        process_lidar_file() — the manual import path.
        """
        from custom_components.solar_shade.shadow_engine import process_lidar_file

        lat, lon = SWITZERLAND_LOCATIONS["Zurich"]
        pad = 0.005
        # Discover tile URL via STAC
        stac_url = (
            "https://data.geo.admin.ch/api/stac/v0.9/collections/"
            "ch.swisstopo.swisssurface3d/items"
            f"?bbox={lon - pad},{lat - pad},{lon + pad},{lat + pad}"
            "&limit=1"
        )
        req = urllib.request.Request(stac_url, headers={"User-Agent": "SolarShade/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        assets = data["features"][0]["assets"]
        zip_url = next(
            v["href"]
            for v in assets.values()
            if v.get("href", "").endswith(".las.zip")
        )

        # Download the .las.zip
        with tempfile.NamedTemporaryFile(suffix=".las.zip", delete=False) as tmp:
            zip_path = tmp.name
        try:
            nbytes = _download_url(zip_url, zip_path)
            assert nbytes > 1_000_000, f"Downloaded only {nbytes} bytes"

            # Extract the .las file (simulating what a user might do)
            with zipfile.ZipFile(zip_path, "r") as zf:
                las_names = [n for n in zf.namelist() if n.lower().endswith(".las")]
                assert len(las_names) >= 1, "No .las file in ZIP"
                las_path = os.path.join(tempfile.gettempdir(), las_names[0])
                zf.extract(las_names[0], tempfile.gettempdir())

            # Process through the manual-file path
            site = process_lidar_file(
                las_path, latitude=lat, longitude=lon, min_cell_size=1.0,
            )
            _validate_site_model(site, expected_epsg=2056, lat=lat, lon=lon)
        finally:
            for p in [zip_path, las_path]:
                try:
                    os.unlink(p)
                except (OSError, UnboundLocalError):
                    pass


# ── USGS integration tests ──────────────────────────────────────────────

US_LOCATIONS = {
    "Tyler_TX": (32.2873, -95.2934),
}


class TestUSGSIntegration:
    """Integration tests for USGS 3DEP — hit real servers."""

    def test_usgs_full_pipeline(self):
        """Download a real USGS LAZ tile, rasterize, validate."""
        from custom_components.solar_shade.usgs_downloader import download_usgs_dsm

        lat, lon = US_LOCATIONS["Tyler_TX"]
        result = asyncio.run(
            download_usgs_dsm(lat, lon, radius_m=50.0, min_cell_size=1.0)
        )
        assert result is not None, "USGS download_usgs_dsm returned None"
        dsm, dtm, cls_grid, canopy_base, x_min, y_min, x_max, y_max, resolution = result
        assert dsm.shape[0] > 5
        assert dtm.shape[0] > 5
        assert resolution > 0
        assert not np.all(np.isnan(dsm))

    def test_usgs_manual_file(self):
        """Simulate a user manually placing a US LAZ file.

        Discovers and downloads a real USGS tile, then runs it through
        process_lidar_file() — the manual import path.
        """
        import aiohttp

        from custom_components.solar_shade.shadow_engine import process_lidar_file
        from custom_components.solar_shade.usgs_downloader import find_laz_urls

        lat, lon = US_LOCATIONS["Tyler_TX"]

        # Discover tile URL
        async def _get_url():
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                tiles = await find_laz_urls(lat, lon, session)
                return tiles[0]["url"] if tiles else None

        tile_url = asyncio.run(_get_url())
        assert tile_url is not None, "No USGS tiles found"

        # Download the tile
        with tempfile.NamedTemporaryFile(suffix=".laz", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            nbytes = _download_url(tile_url, tmp_path)
            assert nbytes > 100_000, f"Downloaded only {nbytes} bytes"

            # Process through the manual-file path
            site = process_lidar_file(
                tmp_path, latitude=lat, longitude=lon, min_cell_size=1.0,
            )
            assert site is not None
            assert site.dsm.shape[0] > 5
            assert site.dtm is not None
            assert site.resolution > 0
            assert not np.all(np.isnan(site.dsm))
        finally:
            os.unlink(tmp_path)


# ── Germany NRW integration tests ───────────────────────────────────────

GERMANY_LOCATIONS = {
    "Cologne": (50.94, 6.96),
}


class TestNRWIntegration:
    """Integration tests for NRW 3D-Messdaten — hit real servers."""

    def test_nrw_tile_head(self):
        """HEAD request on a computed NRW tile URL should return 200."""
        from custom_components.solar_shade.nrw_provider import NRWProvider

        lat, lon = GERMANY_LOCATIONS["Cologne"]
        provider = NRWProvider()
        e, n = provider.latlon_to_native(lat, lon)
        e_km = int(e / 1000)
        n_km = int(n / 1000)
        url = (
            f"https://www.opengeodata.nrw.de/produkte/geobasis/hm/"
            f"3dm_l_las/3dm_l_las/3dm_32_{e_km}_{n_km}_1_nw.laz"
        )
        status = _head_url(url)
        assert status == 200, f"NRW tile HEAD returned {status}"

    def test_nrw_find_tiles(self):
        """find_tiles() should discover a tile for Cologne."""
        import aiohttp

        from custom_components.solar_shade.nrw_provider import NRWProvider

        lat, lon = GERMANY_LOCATIONS["Cologne"]
        provider = NRWProvider()

        async def _find():
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                return await provider.find_tiles(lat, lon, session)

        tiles = asyncio.run(_find())
        assert len(tiles) >= 1, "No NRW tiles found for Cologne"
        assert tiles[0]["url"].endswith(".laz")

    def test_nrw_full_pipeline(self):
        """Download a real NRW tile via download_elevation(), validate."""
        from custom_components.solar_shade.nrw_provider import NRWProvider

        lat, lon = GERMANY_LOCATIONS["Cologne"]
        provider = NRWProvider()
        result = asyncio.run(
            provider.download_elevation(lat, lon, radius_m=50.0, min_cell_size=1.0)
        )
        assert result is not None, "NRW download_elevation returned None"
        dsm, dtm, cls_grid, canopy_base, x_min, y_min, x_max, y_max, resolution = result
        assert dsm.shape[0] > 5
        assert dtm.shape[0] > 5
        assert resolution > 0
        assert not np.all(np.isnan(dsm))
        valid = ~np.isnan(dsm) & ~np.isnan(dtm)
        if valid.any():
            assert float(np.mean(dtm[valid])) <= float(np.mean(dsm[valid]))

    def test_nrw_manual_file(self):
        """Simulate a user manually placing a German NRW LAZ file.

        Downloads a real NRW tile, saves it, and runs process_lidar_file().
        """
        from custom_components.solar_shade.nrw_provider import NRWProvider
        from custom_components.solar_shade.shadow_engine import process_lidar_file

        lat, lon = GERMANY_LOCATIONS["Cologne"]
        provider = NRWProvider()
        e, n = provider.latlon_to_native(lat, lon)
        e_km = int(e / 1000)
        n_km = int(n / 1000)
        tile_url = (
            f"https://www.opengeodata.nrw.de/produkte/geobasis/hm/"
            f"3dm_l_las/3dm_l_las/3dm_32_{e_km}_{n_km}_1_nw.laz"
        )

        with tempfile.NamedTemporaryFile(suffix=".laz", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            nbytes = _download_url(tile_url, tmp_path)
            assert nbytes > 1_000_000, f"Downloaded only {nbytes} bytes"

            site = process_lidar_file(
                tmp_path, latitude=lat, longitude=lon, min_cell_size=2.0,
            )
            _validate_site_model(site, expected_epsg=25832, lat=lat, lon=lon)
        finally:
            os.unlink(tmp_path)


# ── Netherlands PDOK integration tests ──────────────────────────────────

NETHERLANDS_LOCATIONS = {
    "Amsterdam": (52.37, 4.89),
}


class TestPDOKIntegration:
    """Integration tests for PDOK 3D Basisvoorziening — hit real servers."""

    def test_pdok_api_query(self):
        """OGC API Features should return DSM tiles for Amsterdam."""
        lat, lon = NETHERLANDS_LOCATIONS["Amsterdam"]
        pad = 0.005
        url = (
            "https://api.pdok.nl/kadaster/3d-basisvoorziening/ogc/v1_0/"
            "collections/digitaaloppervlaktemodel_20cm/items"
            f"?bbox={lon - pad},{lat - pad},{lon + pad},{lat + pad}"
            "&limit=3"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "SolarShade/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == 200
            data = json.loads(resp.read())
            features = data.get("features", [])
            assert len(features) >= 1, "No PDOK DSM tiles for Amsterdam"
            dl = features[0].get("properties", {}).get("download_link")
            assert dl, "Feature missing download_link"

    def test_pdok_find_tiles(self):
        """find_tiles() should discover tiles for Amsterdam."""
        import aiohttp

        from custom_components.solar_shade.pdok_provider import PDOKProvider

        lat, lon = NETHERLANDS_LOCATIONS["Amsterdam"]
        provider = PDOKProvider()

        async def _find():
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                return await provider.find_tiles(lat, lon, session)

        tiles = asyncio.run(_find())
        assert len(tiles) >= 1, "No PDOK tiles found for Amsterdam"
        assert tiles[0]["url"].endswith(".laz")

    def test_pdok_full_pipeline(self):
        """Download a real PDOK tile via download_elevation(), validate."""
        from custom_components.solar_shade.pdok_provider import PDOKProvider

        lat, lon = NETHERLANDS_LOCATIONS["Amsterdam"]
        provider = PDOKProvider()
        result = asyncio.run(
            provider.download_elevation(lat, lon, radius_m=50.0, min_cell_size=0.5)
        )
        assert result is not None, "PDOK download_elevation returned None"
        dsm, dtm, cls_grid, canopy_base, x_min, y_min, x_max, y_max, resolution = result
        assert dsm.shape[0] > 5
        assert dtm.shape[0] > 5
        assert resolution > 0
        assert not np.all(np.isnan(dsm))
        valid = ~np.isnan(dsm) & ~np.isnan(dtm)
        if valid.any():
            assert float(np.mean(dtm[valid])) <= float(np.mean(dsm[valid]))

    def test_pdok_manual_file(self):
        """Simulate a user manually placing a Dutch LAZ file.

        Downloads a real PDOK DSM tile, saves it, and runs
        process_lidar_file().
        """
        from custom_components.solar_shade.shadow_engine import process_lidar_file

        lat, lon = NETHERLANDS_LOCATIONS["Amsterdam"]
        pad = 0.005
        # Discover tile URL
        url = (
            "https://api.pdok.nl/kadaster/3d-basisvoorziening/ogc/v1_0/"
            "collections/digitaaloppervlaktemodel_20cm/items"
            f"?bbox={lon - pad},{lat - pad},{lon + pad},{lat + pad}"
            "&limit=1"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "SolarShade/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        tile_url = data["features"][0]["properties"]["download_link"]

        with tempfile.NamedTemporaryFile(suffix=".laz", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            nbytes = _download_url(tile_url, tmp_path)
            assert nbytes > 100_000, f"Downloaded only {nbytes} bytes"

            site = process_lidar_file(
                tmp_path, latitude=lat, longitude=lon, min_cell_size=0.5,
            )
            _validate_site_model(site, expected_epsg=28992, lat=lat, lon=lon)
        finally:
            os.unlink(tmp_path)
