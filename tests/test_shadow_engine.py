"""Tests for the Solar Shade shadow engine."""

import numpy as np
import pytest

from custom_components.solar_shade.shadow_engine import (
    SiteModel,
    ZoneDef,
    compute_adjusted_radiation,
    compute_shadow_map,
    compute_zone_shade_fractions,
    save_processed_dsm,
    load_site_model,
)


class TestComputeShadowMap:
    """Test the scanline shadow computation."""

    def _flat_dsm(self, rows=10, cols=10, height=0.0):
        return np.full((rows, cols), height, dtype=np.float32)

    def test_flat_ground_no_shadows(self):
        """Flat ground at any sun angle should produce no shadows."""
        dsm = self._flat_dsm(20, 20, height=100.0)
        shadow = compute_shadow_map(dsm, sun_azimuth_deg=180, sun_elevation_deg=45, pixel_size_m=1.0)
        assert shadow.shape == (20, 20)
        assert not shadow.any(), "Flat ground should have no shadows"

    def test_sun_below_horizon(self):
        """Sun below horizon should produce no shadows."""
        dsm = self._flat_dsm(10, 10)
        dsm[5, 5] = 20.0  # tall object
        shadow = compute_shadow_map(dsm, sun_azimuth_deg=180, sun_elevation_deg=-5, pixel_size_m=1.0)
        assert not shadow.any()

    def test_sun_at_minimum_elevation(self):
        """Sun at exactly MIN_SUN_ELEVATION should produce no shadows."""
        dsm = self._flat_dsm(10, 10)
        dsm[5, 5] = 20.0
        shadow = compute_shadow_map(dsm, sun_azimuth_deg=180, sun_elevation_deg=2.0, pixel_size_m=1.0)
        assert not shadow.any()

    def test_tall_object_casts_shadow_south(self):
        """A tall object with sun from the south should cast shadow to the north."""
        dsm = self._flat_dsm(20, 20, height=0.0)
        dsm[10, 10] = 10.0  # 10m tall object in center

        # Sun from south (azimuth 180), moderate elevation
        shadow = compute_shadow_map(dsm, sun_azimuth_deg=180, sun_elevation_deg=30, pixel_size_m=1.0)

        # Pixels north of the object (lower row indices) should be shaded
        assert shadow[9, 10] or shadow[8, 10], "Pixels north of tall object should be shaded"
        # Pixels south of the object should NOT be shaded
        assert not shadow[11, 10], "Pixels south of tall object should not be shaded"

    def test_tall_object_casts_shadow_east(self):
        """Sun from east (azimuth 90) should cast shadow to the west."""
        dsm = self._flat_dsm(20, 20, height=0.0)
        dsm[10, 10] = 10.0

        shadow = compute_shadow_map(dsm, sun_azimuth_deg=90, sun_elevation_deg=30, pixel_size_m=1.0)

        # Pixels west (lower col) should be shaded
        assert shadow[10, 9] or shadow[10, 8], "Pixels west should be shaded"
        # Pixels east should not
        assert not shadow[10, 11], "Pixels east should not be shaded"

    def test_higher_sun_shorter_shadow(self):
        """Higher sun elevation should produce shorter shadows."""
        dsm = self._flat_dsm(30, 30, height=0.0)
        dsm[15, 15] = 10.0

        shadow_low = compute_shadow_map(dsm, sun_azimuth_deg=180, sun_elevation_deg=15, pixel_size_m=1.0)
        shadow_high = compute_shadow_map(dsm, sun_azimuth_deg=180, sun_elevation_deg=60, pixel_size_m=1.0)

        assert shadow_low.sum() > shadow_high.sum(), "Lower sun should cast longer shadows"

    def test_ground_surface_separate_from_dsm(self):
        """When ground surface is provided, shadows should check ground level."""
        dsm = self._flat_dsm(20, 20, height=0.0)
        dsm[10, 10] = 15.0  # tree canopy at 15m

        # Without ground: pixel under tree is at DSM height (15m) — hard to shade from itself
        shadow_no_ground = compute_shadow_map(
            dsm, sun_azimuth_deg=180, sun_elevation_deg=45, pixel_size_m=1.0
        )

        # With ground: pixel under tree is at 0m — easily shaded by the 15m canopy
        ground = self._flat_dsm(20, 20, height=0.0)
        shadow_with_ground = compute_shadow_map(
            dsm, sun_azimuth_deg=180, sun_elevation_deg=45, pixel_size_m=1.0,
            ground=ground,
        )

        # With ground surface, more area should be detected as shaded
        assert shadow_with_ground.sum() >= shadow_no_ground.sum()

    def test_min_shadow_height_filters_noise(self):
        """Objects below min_shadow_height should not cast shadows."""
        ground = self._flat_dsm(20, 20, height=100.0)
        dsm = ground.copy()
        dsm[10, 10] = 101.0  # 1m above ground — below default 1.5m threshold

        shadow = compute_shadow_map(
            dsm, sun_azimuth_deg=180, sun_elevation_deg=30, pixel_size_m=1.0,
            ground=ground, min_shadow_height=1.5,
        )
        assert not shadow.any(), "1m object should not cast shadow with 1.5m threshold"

        # Same object but lower threshold
        shadow2 = compute_shadow_map(
            dsm, sun_azimuth_deg=180, sun_elevation_deg=30, pixel_size_m=1.0,
            ground=ground, min_shadow_height=0.5,
        )
        assert shadow2.any(), "1m object should cast shadow with 0.5m threshold"

    def test_pixel_size_affects_shadow_length(self):
        """Larger pixels should result in shorter shadow extent (in pixels)."""
        dsm_1m = self._flat_dsm(30, 30, height=0.0)
        dsm_1m[15, 15] = 10.0

        shadow_1m = compute_shadow_map(dsm_1m, sun_azimuth_deg=180, sun_elevation_deg=30, pixel_size_m=1.0)
        shadow_2m = compute_shadow_map(dsm_1m, sun_azimuth_deg=180, sun_elevation_deg=30, pixel_size_m=2.0)

        # With 2m pixels, shadow covers fewer pixels (same real-world length)
        assert shadow_1m.sum() >= shadow_2m.sum()


class TestComputeAdjustedRadiation:
    """Test the radiation adjustment formula."""

    def test_full_sun(self):
        assert compute_adjusted_radiation(1000.0, shade_fraction=0.0) == 1000.0

    def test_full_shade(self):
        result = compute_adjusted_radiation(1000.0, shade_fraction=1.0, diffuse_fraction=0.15)
        assert result == 150.0

    def test_half_shade(self):
        result = compute_adjusted_radiation(1000.0, shade_fraction=0.5, diffuse_fraction=0.15)
        assert result == pytest.approx(575.0, abs=1)

    def test_zero_radiation(self):
        assert compute_adjusted_radiation(0.0, shade_fraction=0.5) == 0.0

    def test_negative_radiation(self):
        assert compute_adjusted_radiation(-10.0, shade_fraction=0.5) == 0.0


class TestZoneShade:
    """Test zone-level shade fraction computation."""

    def test_fully_sunlit_zone(self):
        dsm = np.zeros((20, 20), dtype=np.float32)
        zone = ZoneDef(zone_id="test", zone_name="Test", row_start=5, row_end=15, col_start=5, col_end=15)
        site = SiteModel(dsm=dsm, resolution=1.0, zones=[zone])

        fracs = compute_zone_shade_fractions(site, sun_azimuth_deg=180, sun_elevation_deg=45)
        assert fracs["test"]["average"] == 0.0

    def test_partially_shaded_zone(self):
        dsm = np.zeros((20, 20), dtype=np.float32)
        # Wall on the south edge of the zone
        dsm[14, 5:15] = 8.0

        # Ground is flat at 0, wall rises above it
        dtm = np.zeros((20, 20), dtype=np.float32)

        zone = ZoneDef(zone_id="test", zone_name="Test", row_start=5, row_end=15, col_start=5, col_end=15)
        site = SiteModel(dsm=dsm, resolution=1.0, zones=[zone], dtm=dtm)

        fracs = compute_zone_shade_fractions(site, sun_azimuth_deg=180, sun_elevation_deg=30)
        assert 0.0 < fracs["test"]["average"] < 1.0, "Zone should be partially shaded"
        # Sunniest should be <= average, shadiest >= average
        assert fracs["test"]["sunniest"] <= fracs["test"]["average"]
        assert fracs["test"]["shadiest"] >= fracs["test"]["average"]

    def test_zone_with_mask(self):
        dsm = np.zeros((20, 20), dtype=np.float32)
        dsm[8, 10] = 15.0  # tall object

        mask = np.zeros((5, 5), dtype=bool)
        mask[2:4, 2:4] = True  # only 4 pixels in the mask

        zone = ZoneDef(
            zone_id="masked", zone_name="Masked",
            row_start=5, row_end=10, col_start=8, col_end=13,
            mask=mask,
        )
        site = SiteModel(dsm=dsm, resolution=1.0, zones=[zone])
        fracs = compute_zone_shade_fractions(site, sun_azimuth_deg=180, sun_elevation_deg=45)
        assert "masked" in fracs

    def test_sun_below_horizon_returns_full_shade(self):
        dsm = np.zeros((10, 10), dtype=np.float32)
        dsm[5, 5] = 20.0
        zone = ZoneDef(zone_id="z", zone_name="z", row_start=0, row_end=10, col_start=0, col_end=10)
        site = SiteModel(dsm=dsm, resolution=1.0, zones=[zone])

        fracs = compute_zone_shade_fractions(site, sun_azimuth_deg=180, sun_elevation_deg=-10)
        assert fracs["z"]["average"] == 1.0


class TestRaisedCanopy:
    """Test raised canopy model — rays passing under tree canopy."""

    def test_solid_model_blocks_under_canopy(self):
        """Without canopy_base, a tall tree blocks all rays (solid column)."""
        dsm = np.zeros((20, 20), dtype=np.float32)
        ground = np.zeros((20, 20), dtype=np.float32)
        # Tree at row 10, cols 8-12 (10m tall)
        dsm[10, 8:12] = 10.0
        # Sun from south (az=180) → shadow falls north (lower rows)
        shadow = compute_shadow_map(
            dsm, sun_azimuth_deg=180, sun_elevation_deg=30, pixel_size_m=1.0,
            ground=ground, canopy_base=None,
        )
        # Row 5 (north of tree) should be shadowed
        assert shadow[5, 10] > 0.5

    def test_raised_model_allows_under_canopy(self):
        """With canopy_base high enough, rays pass under the canopy."""
        dsm = np.zeros((20, 20), dtype=np.float32)
        ground = np.zeros((20, 20), dtype=np.float32)
        canopy_base = np.zeros((20, 20), dtype=np.float32)
        # Tree at row 10, cols 8-12: DSM=10m, canopy starts at 7m
        dsm[10, 8:12] = 10.0
        canopy_base[10, 8:12] = 7.0  # trunk is 7m tall, open below
        # Sun from south at 30 deg → shadow falls north
        # Ray from row 5 to tree at row 10 (5 pixels): height = 5*tan(30°) ≈ 2.9m
        # 2.9m < canopy_base of 7m → ray passes under
        shadow = compute_shadow_map(
            dsm, sun_azimuth_deg=180, sun_elevation_deg=30, pixel_size_m=1.0,
            ground=ground, canopy_base=canopy_base,
        )
        # Row 5 should NOT be shadowed (ray passes under canopy)
        assert shadow[5, 10] < 0.1

    def test_raised_canopy_still_blocks_high_rays(self):
        """Rays that hit the canopy zone (above canopy_base) are still blocked."""
        dsm = np.zeros((20, 20), dtype=np.float32)
        ground = np.zeros((20, 20), dtype=np.float32)
        canopy_base = np.zeros((20, 20), dtype=np.float32)
        # Tree at row 10: DSM=10m, canopy starts at 3m (low canopy)
        dsm[10, 8:12] = 10.0
        canopy_base[10, 8:12] = 3.0
        # Sun from south at 45 deg
        # Row 8 (2 pixels from tree): ray height = 2*tan(45°) = 2.0m < canopy 3m → passes under
        # Row 3 (7 pixels from tree): ray height = 7*tan(45°) = 7.0m > canopy 3m → blocked
        shadow = compute_shadow_map(
            dsm, sun_azimuth_deg=180, sun_elevation_deg=45, pixel_size_m=1.0,
            ground=ground, canopy_base=canopy_base,
        )
        # Close receiver (row 8): ray goes under canopy
        assert shadow[8, 10] < 0.1
        # Far receiver (row 3): ray hits canopy zone
        assert shadow[3, 10] > 0.5

    def test_building_unaffected_by_canopy_base(self):
        """Buildings should block regardless of canopy_base (canopy_base = DSM for buildings)."""
        dsm = np.zeros((30, 30), dtype=np.float32)
        ground = np.zeros((30, 30), dtype=np.float32)
        canopy_base = np.zeros((30, 30), dtype=np.float32)
        # Building at row 15: DSM=10m, canopy_base=10m (no trunk clearance)
        dsm[15, 12:18] = 10.0
        canopy_base[15, 12:18] = 10.0  # solid — base equals top
        shadow = compute_shadow_map(
            dsm, sun_azimuth_deg=180, sun_elevation_deg=30, pixel_size_m=1.0,
            ground=ground, canopy_base=canopy_base,
        )
        # Building shadow should still block north of building
        assert shadow[8, 15] > 0.5

    def test_canopy_base_at_ground_equals_solid(self):
        """canopy_base = 0 (ground level) should behave same as solid column."""
        dsm = np.zeros((20, 20), dtype=np.float32)
        ground = np.zeros((20, 20), dtype=np.float32)
        canopy_base_zero = np.zeros((20, 20), dtype=np.float32)
        dsm[10, 8:12] = 10.0
        # canopy_base at ground = no clearance
        shadow_raised = compute_shadow_map(
            dsm, sun_azimuth_deg=180, sun_elevation_deg=30, pixel_size_m=1.0,
            ground=ground, canopy_base=canopy_base_zero,
        )
        shadow_solid = compute_shadow_map(
            dsm, sun_azimuth_deg=180, sun_elevation_deg=30, pixel_size_m=1.0,
            ground=ground, canopy_base=None,
        )
        np.testing.assert_array_equal(shadow_raised, shadow_solid)

    def test_mixed_scene_building_and_tree(self):
        """In a mixed scene, building stays solid, tree gets raised canopy."""
        dsm = np.zeros((30, 30), dtype=np.float32)
        ground = np.zeros((30, 30), dtype=np.float32)
        canopy_base = np.zeros((30, 30), dtype=np.float32)
        # Building at row 15, cols 5-8: solid
        dsm[15, 5:8] = 10.0
        canopy_base[15, 5:8] = 10.0  # building: base = top
        # Tree at row 15, cols 20-23: raised canopy
        dsm[15, 20:23] = 10.0
        canopy_base[15, 20:23] = 7.0  # trunk clearance
        shadow = compute_shadow_map(
            dsm, sun_azimuth_deg=180, sun_elevation_deg=30, pixel_size_m=1.0,
            ground=ground, canopy_base=canopy_base,
        )
        # Building shadow: blocked (north of building)
        assert shadow[8, 6] > 0.5
        # Tree shadow: passes under (north of tree)
        assert shadow[10, 21] < 0.1

    def test_zone_shade_fractions_with_canopy_model(self):
        """compute_zone_shade_fractions respects canopy_model parameter."""
        dsm = np.zeros((30, 30), dtype=np.float32)
        ground = np.zeros((30, 30), dtype=np.float32)
        canopy_base = np.zeros((30, 30), dtype=np.float32)
        dsm[15, 12:18] = 10.0
        canopy_base[15, 12:18] = 7.0
        site = SiteModel(
            dsm=dsm, resolution=1.0, latitude=32.0, longitude=-95.0,
            dtm=ground, canopy_base=canopy_base,
        )
        # Use full-grid mask with bounding box matching the zone area
        zone_mask = np.zeros((30, 30), dtype=bool)
        zone_mask[8:12, 12:18] = True
        site.zones = [type('Z', (), {
            'zone_id': 'z', 'zone_name': 'test', 'polygon_utm': [],
            'mask': zone_mask[8:12, 12:18],  # cropped to bounding box
            'surface': 'ground',
            'row_start': 8, 'row_end': 12,
            'col_start': 12, 'col_end': 18,
        })()]
        fracs_solid = compute_zone_shade_fractions(
            site, 180, 30, canopy_model="solid",
        )
        fracs_raised = compute_zone_shade_fractions(
            site, 180, 30, canopy_model="raised",
        )
        assert fracs_solid["z"]["average"] > 0.3
        assert fracs_raised["z"]["average"] < fracs_solid["z"]["average"]


class TestSaveLoadNativeEPSG:
    """Test that native_epsg persists through save/load roundtrip."""

    def test_roundtrip_with_native_epsg(self, tmp_path):
        """native_epsg should survive NPZ roundtrip."""
        site = SiteModel(
            dsm=np.ones((5, 5), dtype=np.float32),
            resolution=1.0,
            latitude=59.0,
            longitude=18.0,
            native_epsg=3006,
            x_min_m=-10.0,
            y_min_m=-10.0,
            x_max_m=10.0,
            y_max_m=10.0,
        )
        save_processed_dsm(site, str(tmp_path))
        loaded = load_site_model(str(tmp_path))
        assert loaded is not None
        assert loaded.native_epsg == 3006
        assert loaded.latitude == 59.0
        assert loaded.longitude == 18.0

    def test_roundtrip_without_native_epsg(self, tmp_path):
        """Legacy NPZ files (no native_epsg) should default to 0."""
        site = SiteModel(
            dsm=np.ones((5, 5), dtype=np.float32),
            resolution=1.0,
            latitude=32.0,
            longitude=-95.0,
        )
        save_processed_dsm(site, str(tmp_path))
        loaded = load_site_model(str(tmp_path))
        assert loaded is not None
        assert loaded.native_epsg == 0


