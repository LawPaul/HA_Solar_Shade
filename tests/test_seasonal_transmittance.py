"""Tests for seasonal transmittance variation."""

import pytest
from custom_components.solar_shade.shadow_engine import (
    seasonal_high_veg_transmittance,
    TRANSMITTANCE_LEAF_ON,
    TRANSMITTANCE_LEAF_OFF,
)


class TestSeasonalHighVegTransmittance:
    """Test seasonal transmittance curve for high vegetation."""

    # ── Tropical (no seasonality) ─────────────────────────────

    def test_tropical_returns_leaf_on_summer(self):
        assert seasonal_high_veg_transmittance(172, 10.0) == TRANSMITTANCE_LEAF_ON

    def test_tropical_returns_leaf_on_winter(self):
        assert seasonal_high_veg_transmittance(355, 10.0) == TRANSMITTANCE_LEAF_ON

    def test_tropical_south_returns_leaf_on(self):
        assert seasonal_high_veg_transmittance(172, -5.0) == TRANSMITTANCE_LEAF_ON

    # ── Temperate northern (full seasonality) ─────────────────

    def test_north_summer_solstice_leaf_on(self):
        t = seasonal_high_veg_transmittance(172, 45.0)
        assert t == pytest.approx(TRANSMITTANCE_LEAF_ON, abs=0.01)

    def test_north_winter_solstice_leaf_off(self):
        t = seasonal_high_veg_transmittance(355, 45.0)
        assert t == pytest.approx(TRANSMITTANCE_LEAF_OFF, abs=0.01)

    def test_north_spring_equinox_midpoint(self):
        t = seasonal_high_veg_transmittance(80, 45.0)
        midpoint = (TRANSMITTANCE_LEAF_ON + TRANSMITTANCE_LEAF_OFF) / 2
        assert abs(t - midpoint) < 0.05

    def test_north_autumn_equinox_midpoint(self):
        t = seasonal_high_veg_transmittance(266, 45.0)
        midpoint = (TRANSMITTANCE_LEAF_ON + TRANSMITTANCE_LEAF_OFF) / 2
        assert abs(t - midpoint) < 0.05

    # ── Temperate southern (flipped) ──────────────────────────

    def test_south_summer_solstice_leaf_on(self):
        # Southern summer solstice is around day 355
        t = seasonal_high_veg_transmittance(355, -40.0)
        assert t == pytest.approx(TRANSMITTANCE_LEAF_ON, abs=0.01)

    def test_south_winter_solstice_leaf_off(self):
        # Southern winter is around day 172
        t = seasonal_high_veg_transmittance(172, -40.0)
        assert t == pytest.approx(TRANSMITTANCE_LEAF_OFF, abs=0.01)

    # ── Subtropical blend zone (23-35) ────────────────────────

    def test_subtropical_23_equals_tropical(self):
        t = seasonal_high_veg_transmittance(355, 23.0)
        assert t == TRANSMITTANCE_LEAF_ON

    def test_subtropical_35_equals_temperate(self):
        t35 = seasonal_high_veg_transmittance(172, 35.0)
        t45 = seasonal_high_veg_transmittance(172, 45.0)
        assert t35 == pytest.approx(t45, abs=0.01)

    def test_subtropical_30_blended(self):
        t_winter = seasonal_high_veg_transmittance(355, 30.0)
        # Should be between tropical (0.15) and full winter (0.65)
        assert TRANSMITTANCE_LEAF_ON < t_winter < TRANSMITTANCE_LEAF_OFF

    # ── Range checks ──────────────────────────────────────────

    def test_always_between_bounds(self):
        for day in range(1, 366):
            for lat in [-60, -40, -25, 0, 25, 40, 60]:
                t = seasonal_high_veg_transmittance(day, lat)
                assert TRANSMITTANCE_LEAF_ON <= t <= TRANSMITTANCE_LEAF_OFF, \
                    f"Out of bounds: day={day}, lat={lat}, t={t}"

    def test_summer_always_lower_than_winter(self):
        """For temperate locations, summer transmittance should always be lower."""
        for lat in [40, 45, 50, 55]:
            t_summer = seasonal_high_veg_transmittance(172, lat)
            t_winter = seasonal_high_veg_transmittance(355, lat)
            assert t_summer < t_winter
