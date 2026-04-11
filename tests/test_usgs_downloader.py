"""Tests for the USGS downloader module."""

import numpy as np
import pytest

from custom_components.solar_shade.usgs_downloader import (
    _build_dsm,
    _build_dtm,
    _latlon_to_utm,
)


class TestLatLonToUTM:
    """Test lat/lon to UTM conversion for tile matching."""

    def test_east_texas(self):
        """User's location in East Texas should be UTM zone 15."""
        zone, easting, northing, is_northern = _latlon_to_utm(32.28, -95.28)
        assert zone == 15
        assert is_northern is True
        # Easting should be near 300000-400000 for western part of zone 15
        assert 200000 < easting < 800000
        # Northing should be around 3.5 million for ~32° latitude
        assert 3500000 < northing < 3700000

    def test_southern_hemisphere(self):
        zone, easting, northing, is_northern = _latlon_to_utm(-33.86, 151.21)
        assert is_northern is False
        assert northing > 0  # includes 10M offset

    def test_equator(self):
        zone, easting, northing, is_northern = _latlon_to_utm(0.0, -80.0)
        assert is_northern is True
        assert -1 < northing < 100000  # At equator, northing ≈ 0


class TestDTMTileNaming:
    """Test DTM download parameters."""

    def test_bbox_computation(self):
        """Verify bbox is computed correctly for ImageServer request."""
        import math

        lat, lon = 32.28, -95.28
        radius_m = 150.0
        m_per_deg_lat = 111320.0
        m_per_deg_lng = 111320.0 * math.cos(math.radians(lat))
        margin = radius_m * 1.2
        dlat = margin / m_per_deg_lat
        dlng = margin / m_per_deg_lng

        bbox_str = (
            f"{lon - dlng:.6f},{lat - dlat:.6f},"
            f"{lon + dlng:.6f},{lat + dlat:.6f}"
        )

        parts = [float(x) for x in bbox_str.split(",")]
        # West < East, South < North
        assert parts[0] < parts[2]
        assert parts[1] < parts[3]
        # Extent should be about 360m in each direction (150 * 1.2 * 2)
        extent_lat_m = (parts[3] - parts[1]) * m_per_deg_lat
        extent_lon_m = (parts[2] - parts[0]) * m_per_deg_lng
        assert 300 < extent_lat_m < 400
        assert 300 < extent_lon_m < 400


class TestDSMGapFilling:
    """Test that DSM cells with no LiDAR points get filled correctly."""

    def _make_grid_inputs(self, rows, cols, ground_z=10.0, tree_z=20.0):
        """Create synthetic point-cloud arrays already mapped to grid cells.

        Returns cell_idx, z, classification, return_number for a grid where:
        - Most cells have a ground point and a first-return canopy point
        - A few cells have NO points at all (the gap)
        """
        points = []
        gap_cells = set()
        for r in range(rows):
            for c in range(cols):
                cell = r * cols + c
                # Leave a 2x2 hole in the center
                if rows // 2 <= r <= rows // 2 + 1 and cols // 2 <= c <= cols // 2 + 1:
                    gap_cells.add(cell)
                    continue
                # Ground point (class 2)
                points.append((cell, ground_z + r * 0.1, 2, 1))
                # Tree canopy first-return (class 5)
                if (r + c) % 3 == 0:
                    points.append((cell, tree_z + r * 0.1, 5, 1))

        cell_idx = np.array([p[0] for p in points], dtype=np.int32)
        z = np.array([p[1] for p in points], dtype=np.float32)
        classification = np.array([p[2] for p in points], dtype=np.uint8)
        return_number = np.array([p[3] for p in points], dtype=np.uint8)
        return cell_idx, z, classification, return_number, gap_cells

    def test_build_dsm_has_nan_for_empty_cells(self):
        """_build_dsm must return NaN for cells with no points."""
        rows, cols = 10, 10
        cell_idx, z, cls, ret, gap_cells = self._make_grid_inputs(rows, cols)
        dsm = _build_dsm(cell_idx, z, ret, rows, cols)

        for cell in gap_cells:
            r, c = divmod(cell, cols)
            assert np.isnan(dsm[r, c]), f"Expected NaN at gap cell ({r},{c})"

    def test_build_dtm_has_no_nan(self):
        """_build_dtm must fill all NaN via _fill_nan_nearest."""
        rows, cols = 10, 10
        cell_idx, z, cls, ret, gap_cells = self._make_grid_inputs(rows, cols)
        n_cells = rows * cols
        dtm = _build_dtm(cell_idx, z, cls, n_cells, rows, cols)

        assert not np.any(np.isnan(dtm)), "DTM should have no NaN after gap-fill"

    def test_fmax_fills_dsm_nan_with_dtm(self):
        """np.fmax(dsm, dtm) must replace DSM NaN with DTM values."""
        rows, cols = 10, 10
        cell_idx, z, cls, ret, gap_cells = self._make_grid_inputs(rows, cols)
        n_cells = rows * cols

        dsm = _build_dsm(cell_idx, z, ret, rows, cols)
        dtm = _build_dtm(cell_idx, z, cls, n_cells, rows, cols)

        # Before fix: np.maximum would propagate NaN
        bad = np.maximum(dsm, dtm)
        for cell in gap_cells:
            r, c = divmod(cell, cols)
            assert np.isnan(bad[r, c]), "np.maximum should propagate NaN"

        # After fix: np.fmax ignores NaN
        good = np.fmax(dsm, dtm)
        assert not np.any(np.isnan(good)), "np.fmax should eliminate all NaN"

        # Gap cells should get the DTM value
        for cell in gap_cells:
            r, c = divmod(cell, cols)
            assert good[r, c] == pytest.approx(dtm[r, c], abs=0.01)

    def test_shadow_engine_survives_nan_dsm(self):
        """compute_shadow_map must not crash if DSM contains NaN."""
        from custom_components.solar_shade.shadow_engine import compute_shadow_map

        dsm = np.array([
            [10, 10, 10, 10],
            [10, np.nan, 20, 10],
            [10, 10, 10, 10],
            [10, 10, 10, 10],
        ], dtype=np.float32)
        ground = np.full_like(dsm, 10.0)

        shadow = compute_shadow_map(
            dsm, sun_azimuth_deg=180.0, sun_elevation_deg=30.0,
            pixel_size_m=1.0, ground=ground,
        )
        assert shadow.shape == dsm.shape
        assert not np.any(np.isnan(shadow)), "Shadow map should not contain NaN"
