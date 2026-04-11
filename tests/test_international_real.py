"""Download real LiDAR files from international portals and test CRS detection.

These tests download actual files from public LiDAR portals, so they are
marked as integration tests.  Run with:
    pytest tests/test_international_real.py -m integration -v -s
"""
import os
import re
import tempfile
import urllib.request
import json

import laspy
import pytest

UA = {"User-Agent": "SolarShade/1.0"}
TIMEOUT = 30
DOWNLOAD_TIMEOUT = 600


def _download(url: str, suffix: str = ".laz") -> str:
    """Download a file to a temp path, return the path."""
    req = urllib.request.Request(url, headers=UA)
    tmp = tempfile.mktemp(suffix=suffix)
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    return tmp


def _inspect_las(path: str):
    """Print LAS file metadata for debugging."""
    with laspy.open(path) as reader:
        hdr = reader.header
        print(f"  LAS {hdr.version}, format {hdr.point_format.id}, "
              f"{hdr.point_count} points")
        print(f"  Software: {hdr.generating_software}")
        print(f"  Mins: {hdr.mins}")
        print(f"  Maxs: {hdr.maxs}")
        print(f"  VLRs: {len(hdr.vlrs)}")
        for i, vlr in enumerate(hdr.vlrs):
            print(f"    [{i}] {vlr.user_id!r} record_id={vlr.record_id}")
            if hasattr(vlr, "parse_crs"):
                try:
                    crs = vlr.parse_crs()
                    if crs:
                        print(f"         parse_crs -> EPSG:{crs.to_epsg()}")
                except Exception as e:
                    print(f"         parse_crs error: {e}")


# ── Germany NRW (EPSG:25832) ────────────────────────────────────────────

@pytest.mark.integration
class TestGermanyNRW:
    """Test with real NRW LAZ data — EPSG:25832 (ETRS89/UTM 32N)."""

    NRW_BASE = ("https://www.opengeodata.nrw.de/produkte/geobasis/"
                "hm/3dm_l_las/3dm_l_las/")

    def _find_smallest_tile(self) -> tuple[str, str]:
        """Query the NRW XML index for the smallest LAZ tile."""
        req = urllib.request.Request(self.NRW_BASE, headers=UA)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = resp.read().decode("utf-8", errors="replace")
        files = re.findall(
            r'<file name="([^"]+\.laz)" size="(\d+)"', data,
        )
        assert files, "NRW XML index returned no LAZ files"
        files.sort(key=lambda x: int(x[1]))
        name = files[0][0]
        return name, self.NRW_BASE + name

    def test_nrw_crs_detection(self):
        """Download smallest NRW tile and verify CRS detection."""
        from custom_components.solar_shade.usgs_downloader import (
            _read_laz_epsg,
        )
        name, url = self._find_smallest_tile()
        print(f"\nDownloading NRW tile: {name}")

        path = _download(url)
        try:
            _inspect_las(path)
            epsg = _read_laz_epsg(path, laspy)
            print(f"  Detected EPSG: {epsg}")
            assert epsg == 25832, f"Expected 25832, got {epsg}"
        finally:
            os.unlink(path)

    def test_nrw_full_pipeline(self):
        """Full process_lidar_file() with NRW data."""
        from custom_components.solar_shade.shadow_engine import (
            process_lidar_file,
        )
        name, url = self._find_smallest_tile()
        print(f"\nDownloading NRW tile: {name}")

        # NRW tiles are in EPSG:25832; pick coords in the tile area
        # Filename pattern: 3dm_32_XXX_YYYY_1_nw.laz
        # where XXX = easting/1000, YYYY = northing/1000
        m = re.match(r"3dm_32_(\d+)_(\d+)", name)
        if m:
            e_km, n_km = int(m.group(1)), int(m.group(2))
            center_e = e_km * 1000 + 500
            center_n = n_km * 1000 + 500
            # Convert to lat/lon
            from custom_components.solar_shade.geo import epsg_to_latlon
            lat, lon = epsg_to_latlon(center_e, center_n, 25832)
        else:
            # Cologne fallback
            lat, lon = 50.94, 6.96

        path = _download(url)
        try:
            print(f"  Processing at lat={lat:.4f}, lon={lon:.4f}")
            site = process_lidar_file(path, lat, lon, min_cell_size=2.0)
            assert site is not None
            assert site.native_epsg == 25832
            print(f"  native_epsg={site.native_epsg}")
            print(f"  DSM shape: {site.dsm.shape}")
            print(f"  lat={site.latitude:.4f}, lon={site.longitude:.4f}")
        finally:
            os.unlink(path)


# ── Norway (GeoNorge / Kartverket) ──────────────────────────────────────
# Norway uses EPSG:25832 or 25833 (UTM 32N/33N) depending on region.
# No easy anonymous download discovered. Skip for now.


# ── UK Environment Agency ───────────────────────────────────────────────
# EA WFS appears to be down (404). Their download portal requires
# interactive map selection. Skip direct download test.


# ── Countries tested via synthetic files only ───────────────────────────
# These countries' LiDAR files contain proper VLR metadata, so our
# existing synthetic tests cover them:
# - Sweden (EPSG:3006) — Lantmäteriet
# - Slovenia (EPSG:3794) — ARSO/GURS
# - France (EPSG:2154) — IGN (also has live integration test)
# - Switzerland (EPSG:2056) — swisstopo (also has live integration test)
# - US — USGS 3DEP (has live integration test)
#
# The following are additional countries where users have LiDAR access
# but no API is available for automated testing:
# - UK (EPSG:27700) — OSGB 1936 / British National Grid
# - Netherlands (EPSG:28992) — Amersfoort / RD New
# - Norway (EPSG:25832/25833) — ETRS89/UTM 32N or 33N
# - Spain (EPSG:25830) — ETRS89/UTM 30N
# - Finland (EPSG:3067) — ETRS89/TM35FIN
# - Denmark (EPSG:25832) — ETRS89/UTM 32N
# - Austria (EPSG:31287) — MGI / Austria Lambert
# - Australia (EPSG:28354-28356) — GDA94/MGA zones
# - New Zealand (EPSG:2193) — NZGD2000/NZTM2000
# - Belgium (EPSG:31370) — Belge 1972/Belgian Lambert 72
# - Italy (EPSG:6706/32632) — RDN2008 or UTM zones
# - Canada (various) — NAD83 CSRS


# ── Expanded synthetic tests for more countries ─────────────────────────

class TestMoreCountriesSynthetic:
    """Synthetic CRS detection tests for countries not yet covered.

    These use the same _create_synthetic_las helper to create minimal LAS
    files with GeoKey VLRs, proving CRS detection works for any EPSG code.
    """

    @pytest.mark.parametrize("country, epsg, lat, lon", [
        # UK
        ("UK", 27700, 51.50, -0.12),         # London
        # Netherlands
        ("Netherlands", 28992, 52.37, 4.90),  # Amsterdam
        # Norway
        ("Norway", 25832, 59.91, 10.75),      # Oslo (UTM 32N)
        # Spain
        ("Spain", 25830, 40.42, -3.70),       # Madrid (UTM 30N)
        # Finland
        ("Finland", 3067, 60.17, 24.94),      # Helsinki
        # Denmark
        ("Denmark", 25832, 55.68, 12.57),     # Copenhagen
        # Austria
        ("Austria", 31287, 48.21, 16.37),     # Vienna
        # Australia
        ("Australia", 28355, -33.87, 151.21), # Sydney (MGA zone 55)
        # New Zealand
        ("New Zealand", 2193, -41.29, 174.78),# Wellington (NZTM2000)
        # Belgium
        ("Belgium", 31370, 50.85, 4.35),      # Brussels
        # Italy
        ("Italy", 32632, 41.90, 12.50),       # Rome (UTM 32N)
        # Canada
        ("Canada", 2960, 45.50, -73.57),      # Montreal (NAD83/MTM 8)
    ], ids=lambda x: x if isinstance(x, str) else "")

    def test_crs_detection_vlr(self, country, epsg, lat, lon):
        """GeoKey VLR detection works for {country} EPSG:{epsg}."""
        from custom_components.solar_shade.usgs_downloader import (
            _read_laz_epsg,
        )
        from custom_components.solar_shade.geo import latlon_to_epsg

        e, n = latlon_to_epsg(lat, lon, epsg)
        print(f"\n  {country}: EPSG:{epsg}, center E={e:.1f} N={n:.1f}")

        from tests.test_synthetic_manual import _create_synthetic_las
        path = _create_synthetic_las(epsg, e, n)
        try:
            result = _read_laz_epsg(path, laspy)
            assert result == epsg, \
                f"{country}: Expected EPSG:{epsg}, got {result}"
        finally:
            os.unlink(path)

    @pytest.mark.parametrize("country, epsg, lat, lon", [
        ("UK", 27700, 51.50, -0.12),
        ("Netherlands", 28992, 52.37, 4.90),
        ("Norway", 25832, 59.91, 10.75),
        ("Spain", 25830, 40.42, -3.70),
        ("Finland", 3067, 60.17, 24.94),
        ("Denmark", 25832, 55.68, 12.57),
        ("Austria", 31287, 48.21, 16.37),
        ("Australia", 28355, -33.87, 151.21),
        ("New Zealand", 2193, -41.29, 174.78),
        ("Belgium", 31370, 50.85, 4.35),
        ("Italy", 32632, 41.90, 12.50),
        ("Canada", 2960, 45.50, -73.57),
    ], ids=lambda x: x if isinstance(x, str) else "")

    def test_full_pipeline(self, country, epsg, lat, lon):
        """Full process_lidar_file() works for {country} EPSG:{epsg}."""
        from custom_components.solar_shade.shadow_engine import (
            process_lidar_file,
        )
        from custom_components.solar_shade.geo import latlon_to_epsg
        from tests.test_synthetic_manual import _create_synthetic_las

        e, n = latlon_to_epsg(lat, lon, epsg)
        path = _create_synthetic_las(
            epsg, e, n, n_points=5000, spread=200.0,
        )
        try:
            site = process_lidar_file(
                path, latitude=lat, longitude=lon, min_cell_size=1.0,
            )
            assert site is not None, f"{country}: site is None"
            assert site.native_epsg == epsg, \
                f"{country}: Expected EPSG:{epsg}, got {site.native_epsg}"
            assert site.dsm.shape[0] > 0
            assert site.dsm.shape[1] > 0
            assert site.latitude == pytest.approx(lat, abs=0.01)
            assert site.longitude == pytest.approx(lon, abs=0.01)
            print(f"\n  {country}: OK — EPSG:{epsg}, "
                  f"DSM {site.dsm.shape}, "
                  f"lat={site.latitude:.4f}, lon={site.longitude:.4f}")
        finally:
            os.unlink(path)

    @pytest.mark.parametrize("country, epsg, lat, lon", [
        ("UK", 27700, 51.50, -0.12),
        ("Netherlands", 28992, 52.37, 4.90),
        ("Australia", 28355, -33.87, 151.21),
        ("New Zealand", 2193, -41.29, 174.78),
        ("Belgium", 31370, 50.85, 4.35),
    ], ids=lambda x: x if isinstance(x, str) else "")

    def test_wkt_detection(self, country, epsg, lat, lon):
        """WKT VLR detection works for {country} EPSG:{epsg}."""
        from custom_components.solar_shade.usgs_downloader import (
            _read_laz_epsg,
        )
        from custom_components.solar_shade.geo import latlon_to_epsg
        from pyproj import CRS

        e, n = latlon_to_epsg(lat, lon, epsg)
        wkt = CRS.from_epsg(epsg).to_wkt()

        from tests.test_synthetic_manual import _create_las_with_wkt
        path = _create_las_with_wkt(wkt, e, n)
        try:
            result = _read_laz_epsg(path, laspy)
            assert result == epsg, \
                f"{country}: WKT detection expected {epsg}, got {result}"
        finally:
            os.unlink(path)
