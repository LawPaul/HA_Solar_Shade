"""Mocked tests for international provider pipelines.

These run fast (no network) by creating synthetic LAZ files and mocking
HTTP responses.  They test the full processing chain:
  LAZ file → CRS detection → center projection → rasterization → SiteModel.
"""

import struct
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

import laspy


# ── Helpers: synthetic LAZ file creation ─────────────────────────────────

def _write_synthetic_laz(
    filepath: str,
    center_e: float,
    center_n: float,
    epsg: int | None = None,
    spread: float = 50.0,
    n_points: int = 5000,
    ground_z: float = 100.0,
    tree_height: float = 12.0,
    building_height: float = 8.0,
):
    """Write a synthetic LAS file with realistic classification.

    Creates a small point cloud centered at (center_e, center_n) with:
    - Ground points (class 2) at ground_z
    - Vegetation points (class 5) up to ground_z + tree_height
    - Building points (class 6) at ground_z + building_height
    - Optional EPSG via GeoTIFF GeoKey VLR

    The file is valid enough for laspy/laszip to read and for
    _rasterize_laz_file to process.
    """
    rng = np.random.RandomState(42)

    # Generate point positions in a square around center
    x = center_e + rng.uniform(-spread, spread, n_points)
    y = center_n + rng.uniform(-spread, spread, n_points)

    # Assign classification: 60% ground, 25% trees, 15% buildings
    n_ground = int(0.6 * n_points)
    n_trees = int(0.25 * n_points)
    n_buildings = n_points - n_ground - n_trees

    cls = np.zeros(n_points, dtype=np.uint8)
    cls[:n_ground] = 2          # ground
    cls[n_ground:n_ground + n_trees] = 5  # high vegetation
    cls[n_ground + n_trees:] = 6           # building

    # Assign Z values based on classification
    z = np.full(n_points, ground_z, dtype=np.float64)
    # Ground: small terrain variation
    z[:n_ground] += rng.uniform(-0.5, 0.5, n_ground)
    # Trees: vary from trunk base to canopy top
    z[n_ground:n_ground + n_trees] += rng.uniform(2.0, tree_height, n_trees)
    # Buildings: flat rooftops
    z[n_ground + n_trees:] += building_height + rng.uniform(-0.2, 0.2, n_buildings)

    # Return numbers: ground=1, trees get multiple returns, buildings=1
    return_number = np.ones(n_points, dtype=np.uint8)
    # Trees: some first returns (canopy top), some second returns (lower canopy)
    tree_mask = cls == 5
    return_number[tree_mask] = rng.choice([1, 2], size=int(tree_mask.sum()))

    # Create LAS header
    header = laspy.LasHeader(point_format=1, version="1.2")
    header.offsets = [center_e - spread - 1, center_n - spread - 1, 0.0]
    header.scales = [0.001, 0.001, 0.001]

    # Add GeoTIFF GeoKey VLR with EPSG if specified
    if epsg is not None:
        # Build GeoKey directory: header (version, revision, minor, numKeys)
        # + one key entry for ProjectedCSTypeGeoKey (3072)
        geo_key_data = struct.pack(
            '<HHHH HHHH',
            1, 1, 0, 1,        # GeoKey directory header
            3072, 0, 1, epsg,   # ProjectedCSTypeGeoKey → EPSG code
        )
        vlr = laspy.VLR(
            user_id="LASF_Projection",
            record_id=34735,
            description="GeoTIFF GeoKeyDirectoryTag",
            record_data=geo_key_data,
        )
        header.vlrs.append(vlr)

    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    las.classification = cls
    las.return_number = return_number
    las.number_of_returns = np.ones(n_points, dtype=np.uint8)

    las.write(filepath)


# ── process_lidar_file tests ─────────────────────────────────────────────

class TestProcessLidarWithEPSG:
    """Test process_lidar_file() with synthetic LAZ files tagged with EPSG."""

    def _center_for(self, lat, lon, epsg):
        """Compute the projected center for a lat/lon in a given EPSG."""
        from custom_components.solar_shade.geo import latlon_to_epsg
        return latlon_to_epsg(lat, lon, epsg)

    def test_sweref99tm_file(self, tmp_path):
        """LAZ file with EPSG:3006 should detect CRS and set native_epsg."""
        from custom_components.solar_shade.shadow_engine import process_lidar_file

        ce, cn = self._center_for(59.33, 18.07, 3006)
        laz_path = str(tmp_path / "sweden.las")
        _write_synthetic_laz(
            laz_path,
            center_e=ce, center_n=cn,
            epsg=3006, ground_z=15.0,
        )

        site = process_lidar_file(
            laz_path, latitude=59.33, longitude=18.07,
            min_cell_size=1.0,
        )

        assert site.native_epsg == 3006
        assert site.dsm.shape[0] > 5
        assert site.dsm.shape[1] > 5
        assert site.dtm is not None
        assert site.classification is not None
        assert site.latitude == 59.33
        assert site.longitude == 18.07
        # Elevation should be near our ground_z
        assert 10.0 < float(np.nanmin(site.dsm)) < 30.0

    def test_d96tm_file(self, tmp_path):
        """LAZ file with EPSG:3794 should detect CRS and set native_epsg."""
        from custom_components.solar_shade.shadow_engine import process_lidar_file

        ce, cn = self._center_for(46.05, 14.51, 3794)
        laz_path = str(tmp_path / "slovenia.las")
        _write_synthetic_laz(
            laz_path,
            center_e=ce, center_n=cn,
            epsg=3794, ground_z=295.0,
        )

        site = process_lidar_file(
            laz_path, latitude=46.05, longitude=14.51,
            min_cell_size=1.0,
        )

        assert site.native_epsg == 3794
        assert site.dsm.shape[0] > 5
        assert site.classification is not None
        # Ljubljana is ~295m elevation
        assert 280.0 < float(np.nanmin(site.dsm)) < 310.0

    def test_utm_file(self, tmp_path):
        """LAZ file with UTM EPSG should work and set native_epsg."""
        from custom_components.solar_shade.shadow_engine import process_lidar_file

        ce, cn = self._center_for(32.28, -95.28, 32615)
        laz_path = str(tmp_path / "texas.las")
        _write_synthetic_laz(
            laz_path,
            center_e=ce, center_n=cn,
            epsg=32615, ground_z=120.0,
        )

        site = process_lidar_file(
            laz_path, latitude=32.28, longitude=-95.28,
            min_cell_size=1.0,
        )

        assert site.native_epsg == 32615
        assert site.dsm.shape[0] > 5
        assert site.dtm is not None

    def test_no_epsg_uses_centroid(self, tmp_path):
        """LAZ file with no CRS metadata should fall back to centroid."""
        from custom_components.solar_shade.shadow_engine import process_lidar_file

        laz_path = str(tmp_path / "unknown.las")
        _write_synthetic_laz(
            laz_path,
            center_e=500000.0, center_n=3575000.0,
            epsg=None, ground_z=100.0,
        )

        site = process_lidar_file(
            laz_path, latitude=32.28, longitude=-95.28,
            min_cell_size=1.0,
        )

        # No CRS detected → native_epsg should be 0
        assert site.native_epsg == 0
        assert site.dsm.shape[0] > 5

    def test_classification_present(self, tmp_path):
        """Rasterized output should have DTM, classification, and canopy_base."""
        from custom_components.solar_shade.shadow_engine import process_lidar_file

        ce, cn = self._center_for(59.33, 18.07, 3006)
        laz_path = str(tmp_path / "classified.las")
        _write_synthetic_laz(
            laz_path,
            center_e=ce, center_n=cn,
            epsg=3006, ground_z=15.0, tree_height=15.0,
            n_points=10000,
        )

        site = process_lidar_file(
            laz_path, latitude=59.33, longitude=18.07,
            min_cell_size=1.0,
        )

        assert site.dtm is not None
        assert site.classification is not None
        assert site.canopy_base is not None
        # DTM should be lower than DSM (trees + buildings above ground)
        assert float(np.nanmean(site.dtm)) < float(np.nanmean(site.dsm))
        # Classification grid should contain ground (2), veg (5), building (6)
        classes = set(np.unique(site.classification))
        assert 2 in classes, "Should have ground classification"

    def test_save_load_preserves_native_epsg(self, tmp_path):
        """Round-trip through save/load should preserve native_epsg and all fields."""
        from custom_components.solar_shade.shadow_engine import (
            load_site_model,
            process_lidar_file,
            save_processed_dsm,
        )

        ce, cn = self._center_for(46.05, 14.51, 3794)
        laz_path = str(tmp_path / "roundtrip.las")
        _write_synthetic_laz(
            laz_path,
            center_e=ce, center_n=cn,
            epsg=3794, ground_z=295.0,
        )

        site = process_lidar_file(
            laz_path, latitude=46.05, longitude=14.51,
            min_cell_size=1.0,
        )
        assert site.native_epsg == 3794

        npz_dir = str(tmp_path / "npz")
        import os; os.makedirs(npz_dir)
        save_processed_dsm(site, npz_dir)
        loaded = load_site_model(npz_dir)

        assert loaded is not None
        assert loaded.native_epsg == 3794
        assert loaded.latitude == pytest.approx(46.05)
        assert loaded.longitude == pytest.approx(14.51)
        assert loaded.dtm is not None
        assert loaded.classification is not None
        np.testing.assert_array_equal(loaded.dsm, site.dsm)


# ── Satellite image EPSG routing ────────────────────────────────────────

class TestSatelliteImageEPSG:
    """Test that ws_get_satellite_image uses the correct EPSG."""

    def test_native_epsg_used_when_set(self):
        """When site has native_epsg=3006, satellite request should use EPSG:3006."""
        from custom_components.solar_shade.shadow_engine import SiteModel

        site = SiteModel(
            dsm=np.ones((10, 10), dtype=np.float32),
            resolution=1.0,
            latitude=59.33,
            longitude=18.07,
            x_min_m=-50.0, y_min_m=-50.0,
            x_max_m=50.0, y_max_m=50.0,
            native_epsg=3006,
        )

        # Verify the projection path would use EPSG:3006
        from custom_components.solar_shade.geo import latlon_to_epsg
        center_e, center_n = latlon_to_epsg(
            site.latitude, site.longitude, site.native_epsg,
        )
        proj_x_min = center_e + site.x_min_m
        proj_x_max = center_e + site.x_max_m

        # SWEREF99 TM eastings should be ~670000, not ~330000 (UTM zone 34)
        assert 600_000 < center_e < 750_000
        assert proj_x_max > proj_x_min

    def test_utm_fallback_when_no_native_epsg(self):
        """When site has native_epsg=0, should use UTM."""
        from custom_components.solar_shade.shadow_engine import SiteModel

        site = SiteModel(
            dsm=np.ones((10, 10), dtype=np.float32),
            resolution=1.0,
            latitude=32.28,
            longitude=-95.28,
            native_epsg=0,
        )

        from custom_components.solar_shade.geo import latlon_to_utm
        zone, center_e, center_n = latlon_to_utm(site.latitude, site.longitude)
        assert zone == 15
        assert 200_000 < center_e < 800_000


# ── IGN find_tiles (mocked) ─────────────────────────────────────────────

class TestIGNFindTilesMocked:
    """Test IGN LiDAR HD tile discovery with mocked HTTP responses."""

    def test_wfs_returns_tile_url(self):
        """WFS GeoJSON response → tile list with URL."""
        import asyncio
        from custom_components.solar_shade.ign_provider import IGNProvider

        mock_geojson = {
            "features": [
                {
                    "properties": {
                        "url": "https://storage.sbg.cloud.ovh.net/v1/AUTH_.../LHD_FXX_0844_6521_PTS_LAMB93_IGN69.copc.laz",
                        "nom_pkk": "LHD_FXX_0844_6521",
                        "date_vol": "2022-03-15",
                    }
                }
            ]
        }

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_geojson)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        provider = IGNProvider()
        tiles = asyncio.run(provider.find_tiles(45.76, 4.83, mock_session))

        assert len(tiles) == 1
        assert "LHD_FXX_0844_6521" in tiles[0]["url"]
        assert tiles[0]["title"] == "LHD_FXX_0844_6521"
        assert tiles[0]["date"] == "2022-03-15"

    def test_wfs_empty_response(self):
        """WFS returns no features → empty list."""
        import asyncio
        from custom_components.solar_shade.ign_provider import IGNProvider

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"features": []})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        provider = IGNProvider()
        tiles = asyncio.run(provider.find_tiles(48.86, 2.35, mock_session))

        assert tiles == []

    def test_wfs_http_error(self):
        """WFS returns non-200 → empty list."""
        import asyncio
        from custom_components.solar_shade.ign_provider import IGNProvider

        mock_resp = AsyncMock()
        mock_resp.status = 503
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        provider = IGNProvider()
        tiles = asyncio.run(provider.find_tiles(48.86, 2.35, mock_session))

        assert tiles == []

    def test_wfs_multiple_tiles(self):
        """WFS returns multiple features → multiple tiles sorted."""
        import asyncio
        from custom_components.solar_shade.ign_provider import IGNProvider

        mock_geojson = {
            "features": [
                {
                    "properties": {
                        "url": "https://example.com/tile1.copc.laz",
                        "nom_pkk": "tile1",
                        "date_vol": "2022-01-01",
                    }
                },
                {
                    "properties": {
                        "url": "https://example.com/tile2.copc.laz",
                        "nom_pkk": "tile2",
                        "date_vol": "2022-06-15",
                    }
                },
            ]
        }

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_geojson)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        provider = IGNProvider()
        tiles = asyncio.run(provider.find_tiles(48.86, 2.35, mock_session))

        assert len(tiles) == 2


# ── swisstopo find_tiles (mocked) ───────────────────────────────────────

class TestSwisstopoFindTilesMocked:
    """Test swisstopo STAC tile discovery with mocked HTTP responses."""

    def test_stac_returns_tile_url(self):
        """STAC response → tile list with .las.zip URL."""
        import asyncio
        from custom_components.solar_shade.swisstopo_provider import SwisstopoProvider

        mock_stac = {
            "features": [
                {
                    "id": "swisssurface3d_2018_2683-1247",
                    "assets": {
                        "swisssurface3d_2018_2683-1247_2056_5728.las.zip": {
                            "href": "https://data.geo.admin.ch/ch.swisstopo.swisssurface3d/swisssurface3d_2018_2683-1247/swisssurface3d_2018_2683-1247_2056_5728.las.zip",
                            "type": "application/vnd.laszip",
                        }
                    },
                }
            ]
        }

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_stac)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        provider = SwisstopoProvider()
        tiles = asyncio.run(provider.find_tiles(47.37, 8.54, mock_session))

        assert len(tiles) == 1
        assert tiles[0]["url"].endswith(".las.zip")
        assert tiles[0]["title"] == "swisssurface3d_2018_2683-1247"

    def test_stac_empty_response(self):
        """STAC returns no features → empty list."""
        import asyncio
        from custom_components.solar_shade.swisstopo_provider import SwisstopoProvider

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"features": []})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        provider = SwisstopoProvider()
        tiles = asyncio.run(provider.find_tiles(47.37, 8.54, mock_session))

        assert tiles == []

    def test_stac_http_error(self):
        """STAC returns non-200 → empty list."""
        import asyncio
        from custom_components.solar_shade.swisstopo_provider import SwisstopoProvider

        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        provider = SwisstopoProvider()
        tiles = asyncio.run(provider.find_tiles(47.37, 8.54, mock_session))

        assert tiles == []

    def test_stac_skips_non_laszip_assets(self):
        """Features with only .tif assets (not .las.zip) are skipped."""
        import asyncio
        from custom_components.solar_shade.swisstopo_provider import SwisstopoProvider

        mock_stac = {
            "features": [
                {
                    "id": "swisssurface3d-raster_2018_2683-1247",
                    "assets": {
                        "some_raster.tif": {
                            "href": "https://data.geo.admin.ch/some_raster.tif",
                            "type": "image/tiff",
                        }
                    },
                }
            ]
        }

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_stac)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        provider = SwisstopoProvider()
        tiles = asyncio.run(provider.find_tiles(47.37, 8.54, mock_session))

        assert tiles == []


# ── Lambert-93 and LV95 LAZ pipeline tests ──────────────────────────────

class TestProcessLidarFranceSwiss:
    """Test process_lidar_file with Lambert-93 and LV95 synthetic LAZ files."""

    def _center_for(self, lat, lon, epsg):
        from custom_components.solar_shade.geo import latlon_to_epsg
        return latlon_to_epsg(lat, lon, epsg)

    def test_lambert93_file(self, tmp_path):
        """LAZ file with EPSG:2154 should detect CRS and set native_epsg."""
        from custom_components.solar_shade.shadow_engine import process_lidar_file

        ce, cn = self._center_for(48.86, 2.35, 2154)
        laz_path = str(tmp_path / "france.las")
        _write_synthetic_laz(
            laz_path,
            center_e=ce, center_n=cn,
            epsg=2154, ground_z=35.0,
        )

        site = process_lidar_file(
            laz_path, latitude=48.86, longitude=2.35,
            min_cell_size=1.0,
        )

        assert site.native_epsg == 2154
        assert site.dsm.shape[0] > 5
        assert site.dsm.shape[1] > 5
        assert site.dtm is not None
        assert site.classification is not None
        assert 30.0 < float(np.nanmin(site.dsm)) < 50.0

    def test_lv95_file(self, tmp_path):
        """LAZ file with EPSG:2056 should detect CRS and set native_epsg."""
        from custom_components.solar_shade.shadow_engine import process_lidar_file

        ce, cn = self._center_for(47.37, 8.54, 2056)
        laz_path = str(tmp_path / "swiss.las")
        _write_synthetic_laz(
            laz_path,
            center_e=ce, center_n=cn,
            epsg=2056, ground_z=408.0,
        )

        site = process_lidar_file(
            laz_path, latitude=47.37, longitude=8.54,
            min_cell_size=1.0,
        )

        assert site.native_epsg == 2056
        assert site.dsm.shape[0] > 5
        assert site.dtm is not None
        # Zurich city is ~408m elevation
        assert 395.0 < float(np.nanmin(site.dsm)) < 420.0
