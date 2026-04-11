"""Test CRS detection with synthetic LAS files for Sweden and Slovenia.

These tests create minimal LAS files with proper GeoKey VLRs and verify
that the full manual import pipeline works — no network access needed.
Also tests filename-based EPSG fallback when VLR metadata is absent.
"""
import os
import struct
import tempfile

import laspy
import numpy as np
import pytest


def _create_synthetic_las(
    epsg: int, x_center: float, y_center: float,
    z_center: float = 100.0, n_points: int = 1000,
    spread: float = 50.0, include_vlr: bool = True,
    filename: str | None = None,
) -> str:
    """Create a minimal LAS file with optional GeoKey VLR."""
    rng = np.random.default_rng(42)
    x = x_center + rng.uniform(-spread, spread, n_points)
    y = y_center + rng.uniform(-spread, spread, n_points)
    z = z_center + rng.uniform(-5, 15, n_points)

    header = laspy.LasHeader(point_format=1, version="1.2")
    header.offsets = [x_center, y_center, 0.0]
    header.scales = [0.01, 0.01, 0.01]

    if include_vlr:
        geokey_data = struct.pack(
            "<HHHH" "HHHH",
            1, 1, 0, 1,
            3072, 0, 1, epsg,
        )
        vlr = laspy.VLR(
            user_id="LASF_Projection",
            record_id=34735,
            record_data=geokey_data,
            description="GeoTIFF GeoKeyDirectoryTag",
        )
        header.vlrs.append(vlr)

    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z

    cls = np.full(n_points, 1, dtype=np.uint8)
    cls[::3] = 2
    las.classification = cls

    if filename:
        tmp = os.path.join(tempfile.gettempdir(), filename)
    else:
        tmp = tempfile.mktemp(suffix=".las")
    las.write(tmp)
    return tmp


class TestSyntheticSweden:
    """Sweden SWEREF 99 TM (EPSG:3006) manual file import."""

    EPSG = 3006
    LAT, LON = 59.33, 18.07  # Stockholm
    # Exact projected coordinates from pyproj (4326→3006)
    X_CENTER, Y_CENTER = 674_647.9, 6_580_824.6

    def test_crs_detection_with_vlr(self):
        """GeoKey VLR with EPSG:3006 should be detected."""
        from custom_components.solar_shade.usgs_downloader import _read_laz_epsg

        path = _create_synthetic_las(self.EPSG, self.X_CENTER, self.Y_CENTER)
        try:
            result = _read_laz_epsg(path, laspy)
            assert result == self.EPSG, f"Expected {self.EPSG}, got {result}"
        finally:
            os.unlink(path)

    def test_crs_detection_without_vlr(self):
        """Without VLRs and no EPSG in filename, detection returns None."""
        from custom_components.solar_shade.usgs_downloader import _read_laz_epsg

        path = _create_synthetic_las(
            self.EPSG, self.X_CENTER, self.Y_CENTER, include_vlr=False,
        )
        try:
            result = _read_laz_epsg(path, laspy)
            # Generic temp filename has no EPSG info — should return None
            assert result is None, \
                f"Expected None for file without VLR or EPSG filename, got {result}"
        finally:
            os.unlink(path)

    def test_crs_detection_filename_fallback(self):
        """EPSG in filename should be detected when VLRs are absent."""
        from custom_components.solar_shade.usgs_downloader import _read_laz_epsg

        path = _create_synthetic_las(
            self.EPSG, self.X_CENTER, self.Y_CENTER, include_vlr=False,
            filename="sweden_data_EPSG3006.las",
        )
        try:
            result = _read_laz_epsg(path, laspy)
            assert result == self.EPSG, f"Expected {self.EPSG}, got {result}"
        finally:
            os.unlink(path)

    def test_process_lidar_file(self):
        """Full process_lidar_file() pipeline with Swedish data."""
        from custom_components.solar_shade.shadow_engine import process_lidar_file

        path = _create_synthetic_las(
            self.EPSG, self.X_CENTER, self.Y_CENTER,
            n_points=5000, spread=200.0,
        )
        try:
            site = process_lidar_file(
                path, latitude=self.LAT, longitude=self.LON, min_cell_size=1.0,
            )
            assert site is not None
            assert site.native_epsg == self.EPSG
            assert site.dsm.shape[0] > 0
            assert site.dsm.shape[1] > 0
            assert site.dtm is not None
            assert site.latitude == pytest.approx(self.LAT, abs=0.01)
            assert site.longitude == pytest.approx(self.LON, abs=0.01)
        finally:
            os.unlink(path)


class TestSyntheticSlovenia:
    """Slovenia D96/TM (EPSG:3794) manual file import."""

    EPSG = 3794
    LAT, LON = 46.05, 14.51  # Ljubljana
    # Exact projected coordinates from pyproj (4326→3794)
    # Note: false northing = -5,000,000 so Y is ~101k, not ~5,101k
    X_CENTER, Y_CENTER = 462_081.0, 101_250.1

    def test_crs_detection_with_vlr(self):
        """GeoKey VLR with EPSG:3794 should be detected."""
        from custom_components.solar_shade.usgs_downloader import _read_laz_epsg

        path = _create_synthetic_las(self.EPSG, self.X_CENTER, self.Y_CENTER)
        try:
            result = _read_laz_epsg(path, laspy)
            assert result == self.EPSG, f"Expected {self.EPSG}, got {result}"
        finally:
            os.unlink(path)

    def test_crs_detection_without_vlr(self):
        """Without VLRs and no EPSG in filename, detection returns None."""
        from custom_components.solar_shade.usgs_downloader import _read_laz_epsg

        path = _create_synthetic_las(
            self.EPSG, self.X_CENTER, self.Y_CENTER, include_vlr=False,
        )
        try:
            result = _read_laz_epsg(path, laspy)
            assert result is None, \
                f"Expected None for file without VLR or EPSG filename, got {result}"
        finally:
            os.unlink(path)

    def test_crs_detection_filename_fallback(self):
        """EPSG in filename should be detected when VLRs are absent."""
        from custom_components.solar_shade.usgs_downloader import _read_laz_epsg

        path = _create_synthetic_las(
            self.EPSG, self.X_CENTER, self.Y_CENTER, include_vlr=False,
            filename="slovenija_3794_5776.las",
        )
        try:
            result = _read_laz_epsg(path, laspy)
            assert result == self.EPSG, f"Expected {self.EPSG}, got {result}"
        finally:
            os.unlink(path)

    def test_process_lidar_file(self):
        """Full process_lidar_file() pipeline with Slovenian data."""
        from custom_components.solar_shade.shadow_engine import process_lidar_file

        path = _create_synthetic_las(
            self.EPSG, self.X_CENTER, self.Y_CENTER,
            n_points=5000, spread=200.0,
        )
        try:
            site = process_lidar_file(
                path, latitude=self.LAT, longitude=self.LON, min_cell_size=1.0,
            )
            assert site is not None
            assert site.native_epsg == self.EPSG
            assert site.dsm.shape[0] > 0
            assert site.dsm.shape[1] > 0
            assert site.dtm is not None
            assert site.latitude == pytest.approx(self.LAT, abs=0.01)
            assert site.longitude == pytest.approx(self.LON, abs=0.01)
        finally:
            os.unlink(path)


class TestSyntheticWKT:
    """Test CRS detection via WKT VLR (record_id=2112) — as IGN uses."""

    def test_wkt_sweden(self):
        """WKT VLR containing EPSG:3006 should be detected."""
        from custom_components.solar_shade.usgs_downloader import _read_laz_epsg

        wkt = (
            'PROJCS["SWEREF99 TM",'
            'GEOGCS["SWEREF99",'
            'DATUM["SWEREF99",'
            'SPHEROID["GRS 1980",6378137,298.257222101]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["degree",0.0174532925199433]],'
            'PROJECTION["Transverse_Mercator"],'
            'PARAMETER["latitude_of_origin",0],'
            'PARAMETER["central_meridian",15],'
            'PARAMETER["scale_factor",0.9996],'
            'PARAMETER["false_easting",500000],'
            'PARAMETER["false_northing",0],'
            'UNIT["metre",1],'
            'AUTHORITY["EPSG","3006"]]'
        )
        path = _create_las_with_wkt(wkt, 674_647.9, 6_580_824.6)
        try:
            result = _read_laz_epsg(path, laspy)
            assert result == 3006, f"Expected 3006, got {result}"
        finally:
            os.unlink(path)

    def test_wkt_slovenia(self):
        """WKT VLR containing EPSG:3794 should be detected."""
        from custom_components.solar_shade.usgs_downloader import _read_laz_epsg

        wkt = (
            'PROJCS["Slovenia 1996 / Slovene National Grid",'
            'GEOGCS["Slovenia 1996",'
            'DATUM["Slovenia_Geodetic_Datum_1996",'
            'SPHEROID["GRS 1980",6378137,298.257222101]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["degree",0.0174532925199433]],'
            'PROJECTION["Transverse_Mercator"],'
            'PARAMETER["latitude_of_origin",0],'
            'PARAMETER["central_meridian",15],'
            'PARAMETER["scale_factor",0.9999],'
            'PARAMETER["false_easting",500000],'
            'PARAMETER["false_northing",-5000000],'
            'UNIT["metre",1],'
            'AUTHORITY["EPSG","3794"]]'
        )
        path = _create_las_with_wkt(wkt, 462_081.0, 101_250.1)
        try:
            result = _read_laz_epsg(path, laspy)
            assert result == 3794, f"Expected 3794, got {result}"
        finally:
            os.unlink(path)


def _create_las_with_wkt(wkt: str, x_center: float, y_center: float) -> str:
    """Create a LAS file with a WKT CRS VLR (record_id 2112)."""
    rng = np.random.default_rng(42)
    n = 500
    x = x_center + rng.uniform(-50, 50, n)
    y = y_center + rng.uniform(-50, 50, n)
    z = 100.0 + rng.uniform(-5, 15, n)

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = [x_center, y_center, 0.0]
    header.scales = [0.01, 0.01, 0.01]

    # Set WKT bit in global encoding
    header.global_encoding.wkt = True

    wkt_bytes = wkt.encode("utf-8") + b"\x00"
    vlr = laspy.VLR(
        user_id="LASF_Projection",
        record_id=2112,
        record_data=wkt_bytes,
        description="OGC Coordinate System WKT",
    )
    header.vlrs.append(vlr)

    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    cls = np.full(n, 1, dtype=np.uint8)
    cls[::3] = 2
    las.classification = cls

    tmp = tempfile.mktemp(suffix=".las")
    las.write(tmp)
    return tmp


class TestFilenameFallback:
    """Test EPSG extraction from filename when VLR metadata is absent."""

    X, Y = 500_000.0, 200_000.0  # arbitrary — VLR-based tests cover real coords

    def _make(self, filename: str) -> str:
        return _create_synthetic_las(
            epsg=2056, x_center=self.X, y_center=self.Y,
            include_vlr=False, filename=filename,
        )

    def test_swisstopo_pattern(self):
        """swisstopo naming: *_2056_5728.las → EPSG:2056."""
        from custom_components.solar_shade.usgs_downloader import _read_laz_epsg

        path = self._make("swisssurface3d_2018_2682-1246_2056_5728.las")
        try:
            assert _read_laz_epsg(path, laspy) == 2056
        finally:
            os.unlink(path)

    def test_epsg_prefix_uppercase(self):
        """Generic _EPSG3006 pattern."""
        from custom_components.solar_shade.usgs_downloader import _read_laz_epsg

        path = self._make("some_tile_EPSG3006.las")
        try:
            assert _read_laz_epsg(path, laspy) == 3006
        finally:
            os.unlink(path)

    def test_epsg_prefix_lowercase(self):
        """Generic _epsg2154 pattern."""
        from custom_components.solar_shade.usgs_downloader import _read_laz_epsg

        path = self._make("tile_epsg2154_data.las")
        try:
            assert _read_laz_epsg(path, laspy) == 2154
        finally:
            os.unlink(path)

    def test_epsg_with_dash(self):
        """Dash-separated -epsg25832 pattern."""
        from custom_components.solar_shade.usgs_downloader import _read_laz_epsg

        path = self._make("data-epsg25832.las")
        try:
            assert _read_laz_epsg(path, laspy) == 25832
        finally:
            os.unlink(path)

    def test_year_not_matched(self):
        """A bare 4-digit year like 2018 should not be treated as EPSG."""
        from custom_components.solar_shade.usgs_downloader import _read_laz_epsg

        path = self._make("tile_2018_data.las")
        try:
            result = _read_laz_epsg(path, laspy)
            # 2018 is not a valid projected CRS → should be None
            assert result is None, f"Year 2018 misinterpreted as EPSG:{result}"
        finally:
            os.unlink(path)

    def test_no_epsg_in_name(self):
        """Plain filename with no EPSG info returns None."""
        from custom_components.solar_shade.usgs_downloader import _read_laz_epsg

        path = self._make("my_lidar_scan.las")
        try:
            assert _read_laz_epsg(path, laspy) is None
        finally:
            os.unlink(path)

    def test_vlr_takes_precedence_over_filename(self):
        """When both VLR and filename have EPSG, VLR wins."""
        from custom_components.solar_shade.usgs_downloader import _read_laz_epsg

        # VLR says 3006, filename says 2056
        path = _create_synthetic_las(
            epsg=3006, x_center=self.X, y_center=self.Y,
            include_vlr=True, filename="tile_2056_5728.las",
        )
        try:
            assert _read_laz_epsg(path, laspy) == 3006
        finally:
            os.unlink(path)
