"""Tests for shadow scan direction discontinuity at 45° boundaries."""

import math

import numpy as np
import pytest

from custom_components.solar_shade.shadow_engine import (
    compute_shadow_map,
)


def _test_scene():
    """Create a 50x50 test scene with a building and tree."""
    ground = np.full((50, 50), 100.0, dtype=np.float32)
    dsm = ground.copy()
    # Building: 5x5 block, 8m tall
    dsm[20:25, 20:25] = 108.0
    # Tree: circular canopy, ~12m peak
    for r in range(15, 22):
        for c in range(30, 37):
            d = math.sqrt((r - 18) ** 2 + (c - 33) ** 2)
            if d < 4:
                dsm[r, c] = 112.0 - d
    return dsm, ground


class TestScanDirectionConsistency:
    """Verify shadow computation produces smooth results across all azimuths."""

    def test_no_large_jump_across_225_boundary(self):
        """Shadow should change smoothly across the old 225° scan-switch boundary."""
        dsm, ground = _test_scene()
        el = 35.0

        sums = []
        azimuths = np.arange(220, 231, 0.5)
        for az in azimuths:
            s = compute_shadow_map(dsm, az, el, 1.0, ground=ground, min_shadow_height=1.5)
            sums.append(float(s.sum()))

        # Check that no adjacent pair has a jump > 10% of the mean
        mean_sum = np.mean(sums)
        for i in range(1, len(sums)):
            jump = abs(sums[i] - sums[i - 1])
            pct = jump / mean_sum * 100 if mean_sum > 0 else 0
            assert pct < 10, (
                f"Shadow jump of {pct:.1f}% between az={azimuths[i-1]:.1f}° and "
                f"az={azimuths[i]:.1f}° (sums: {sums[i-1]:.1f} → {sums[i]:.1f})"
            )

    def test_all_cardinal_boundaries_smooth(self):
        """Test all four old 45° boundaries: 45, 135, 225, 315."""
        dsm, ground = _test_scene()
        el = 35.0

        for center_az in [45, 135, 225, 315]:
            azimuths = np.arange(center_az - 5, center_az + 5.5, 0.5)
            sums = []
            for az in azimuths:
                s = compute_shadow_map(dsm, az, el, 1.0, ground=ground, min_shadow_height=1.5)
                sums.append(float(s.sum()))

            mean_sum = np.mean(sums)
            if mean_sum == 0:
                continue

            max_jump = 0
            for i in range(1, len(sums)):
                jump = abs(sums[i] - sums[i - 1]) / mean_sum * 100
                max_jump = max(max_jump, jump)

            assert max_jump < 10, (
                f"Shadow jump of {max_jump:.1f}% near az={center_az}°"
            )

    def test_shadow_direction_correct_south(self):
        """Sun from south (180°): shadow goes north (decreasing row)."""
        dsm, ground = _test_scene()
        s = compute_shadow_map(dsm, 180, 35, 1.0, ground=ground, min_shadow_height=1.5)
        # Building at rows 20-25. Shadow should be at rows < 20.
        assert s[:20, 20:25].sum() > 0, "Shadow should be north of building"
        assert s[25:, 20:25].sum() == 0, "No shadow south of building"

    def test_shadow_direction_correct_southwest(self):
        """Sun from SW (225°): shadow goes NE (decreasing row, increasing col)."""
        ground = np.full((40, 40), 100.0, dtype=np.float32)
        dsm = ground.copy()
        dsm[20, 20] = 110.0  # single tall pixel

        s = compute_shadow_map(dsm, 225, 35, 1.0, ground=ground, min_shadow_height=1.5)
        # Shadow should be NE of the object: rows < 20, cols > 20
        ne_shadow = s[:20, 21:].sum()
        sw_shadow = s[21:, :20].sum()
        assert ne_shadow > 0, "Shadow should be NE of object for SW sun"
        assert sw_shadow == 0, "No shadow SW of object for SW sun"

    def test_full_azimuth_sweep_smooth(self):
        """A full 360° azimuth sweep should produce smoothly varying shadows."""
        dsm, ground = _test_scene()
        el = 40.0

        azimuths = np.arange(0, 360, 2.0)
        sums = []
        for az in azimuths:
            s = compute_shadow_map(dsm, az, el, 1.0, ground=ground, min_shadow_height=1.5)
            sums.append(float(s.sum()))

        # Find the maximum step-to-step jump as percentage of local mean
        for i in range(1, len(sums)):
            if sums[i] == 0 and sums[i-1] == 0:
                continue
            local_mean = (sums[i] + sums[i-1]) / 2
            if local_mean < 1:
                continue
            pct = abs(sums[i] - sums[i-1]) / local_mean * 100
            assert pct < 25, (
                f"Shadow jump of {pct:.1f}% at az={azimuths[i]:.0f}°"
            )

    def test_transmittance_works_with_ray_march(self):
        """Partial transmittance should produce partial shadow values."""
        ground = np.full((30, 30), 0.0, dtype=np.float32)
        dsm = ground.copy()
        dsm[15, 15] = 10.0  # tall tree

        trans = np.zeros((30, 30), dtype=np.float32)
        trans[15, 15] = 0.25  # tree canopy — 75% opaque

        s = compute_shadow_map(
            dsm, 180, 35, 1.0, ground=ground,
            min_shadow_height=1.5, transmittance=trans,
        )
        # Shadow pixels should have partial opacity (0.75), not full 1.0
        shaded_values = s[s > 0]
        assert len(shaded_values) > 0, "Should have shadow"
        assert all(v <= 0.76 for v in shaded_values), (
            f"Shadow should be ~0.75 opacity, got max {shaded_values.max():.2f}"
        )


class TestTerrainShadowing:
    """Verify that terrain (hills) correctly casts shadows on ground and DSM surfaces."""

    def test_hill_shadows_ground(self):
        """A hill should cast shadow on lower ground behind it."""
        # Terrain: flat at 100m with a 10m hill in the middle
        ground = np.full((40, 40), 100.0, dtype=np.float32)
        ground[18:22, 18:22] = 110.0  # 10m hill
        dsm = ground.copy()  # no structures, DSM == ground

        # Sun from south (az=180), low angle (20°) — hill should shadow north side
        s = compute_shadow_map(dsm, 180, 20, 1.0, ground=ground, min_shadow_height=0.0)

        # North of the hill (rows < 18) should have shadow
        north_shadow = s[:18, 18:22].sum()
        south_shadow = s[22:, 18:22].sum()
        assert north_shadow > 0, "Hill should cast shadow to the north"
        assert south_shadow == 0, "No shadow south of hill"

    def test_hill_shadows_ground_at_steep_angle(self):
        """At high sun elevation, hill shadow should be shorter."""
        ground = np.full((40, 40), 100.0, dtype=np.float32)
        ground[18:22, 18:22] = 110.0
        dsm = ground.copy()

        s_low = compute_shadow_map(dsm, 180, 15, 1.0, ground=ground, min_shadow_height=0.0)
        s_high = compute_shadow_map(dsm, 180, 60, 1.0, ground=ground, min_shadow_height=0.0)

        assert s_low.sum() > s_high.sum(), "Lower sun should produce longer shadow from hill"

    def test_valley_between_hills(self):
        """A valley between two hills — one hill shadows the valley floor."""
        ground = np.full((50, 50), 100.0, dtype=np.float32)
        # South hill
        ground[30:35, 20:30] = 115.0
        # North hill
        ground[10:15, 20:30] = 112.0
        # Valley between them at row 20-25 is at 100m
        dsm = ground.copy()

        # Sun from south — south hill shadows the valley
        s = compute_shadow_map(dsm, 180, 25, 1.0, ground=ground, min_shadow_height=0.0)
        valley_shadow = s[20:30, 20:30].sum()
        assert valley_shadow > 0, "South hill should shadow the valley"

    def test_dsm_receive_surface_roof_not_self_shadowed(self):
        """When using DSM as receive surface, a building shouldn't shadow itself."""
        ground = np.full((30, 30), 100.0, dtype=np.float32)
        dsm = ground.copy()
        # Building: 5x5 block, 8m tall
        dsm[12:17, 12:17] = 108.0

        # Compute shadow with DSM as receiver (roof zone scenario)
        s = compute_shadow_map(dsm, 180, 45, 1.0, ground=dsm, min_shadow_height=0.0)

        # The building's own roof pixels should NOT be shaded
        roof_shadow = s[12:17, 12:17].sum()
        assert roof_shadow == 0, "Building should not shadow its own roof"

    def test_tree_shadows_roof(self):
        """A tall tree should shadow a nearby shorter building's roof."""
        ground = np.full((40, 40), 100.0, dtype=np.float32)
        dsm = ground.copy()
        # Building: 5m tall roof at row 15
        dsm[14:18, 20:25] = 105.0
        # Tall tree: 15m, south of the building
        dsm[22, 22] = 115.0

        # Sun from south (az=180), moderate angle — tree shadows the roof
        s = compute_shadow_map(dsm, 180, 30, 1.0, ground=dsm, min_shadow_height=0.0)

        # Some roof pixels should be shaded by the tree
        roof_shadow = s[14:18, 20:25].sum()
        assert roof_shadow > 0, "Tree should cast shadow on the building's roof"

    def test_hill_shadows_roof(self):
        """A large hill should shadow a building's roof behind it."""
        ground = np.full((50, 50), 100.0, dtype=np.float32)
        # Hill south of building
        ground[30:35, 20:30] = 120.0
        dsm = ground.copy()
        # Building north of hill, 5m tall roof
        dsm[15:20, 20:25] = 105.0

        # Sun from south, low angle — hill shadows the building
        s = compute_shadow_map(dsm, 180, 15, 1.0, ground=dsm, min_shadow_height=0.0)

        roof_shadow = s[15:20, 20:25].sum()
        assert roof_shadow > 0, "Hill should shadow the building's roof"

    def test_ground_zone_vs_dsm_zone_different_results(self):
        """Same polygon, different receive surfaces should give different shade values."""
        ground = np.full((40, 40), 100.0, dtype=np.float32)
        dsm = ground.copy()
        # Building: 8m tall
        dsm[18:22, 18:22] = 108.0
        # Tree south of building: 12m
        dsm[28, 20] = 112.0

        az, el = 180, 30

        # Shadow on ground (ground zones)
        s_ground = compute_shadow_map(dsm, az, el, 1.0, ground=ground, min_shadow_height=0.0)
        # Shadow on DSM (roof zones)
        s_dsm = compute_shadow_map(dsm, az, el, 1.0, ground=dsm, min_shadow_height=0.0)

        # Ground should show more shadow (lower surface = easier to shade)
        assert s_ground.sum() >= s_dsm.sum(), (
            "Ground receive surface should have >= shadow than DSM receive surface"
        )
