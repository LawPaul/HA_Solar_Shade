"""Tests for the fixed sunniest/shadiest patch selection.

Covers ``compute_zone_spot_windows``, which picks a stable patch per zone by
integrating shade over a day's sun path, and the websocket handler that feeds
those patches to the map panel.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from custom_components.solar_shade.shadow_engine import (
    SiteModel,
    ZoneDef,
    compute_zone_spot_windows,
)


def _day_of_sun_from_west(n=6, elevation=30.0):
    """Sun samples all from the west, so shadows fall to the east."""
    return [(270.0, elevation, math.sin(math.radians(elevation)))] * n


class TestComputeZoneSpotWindows:
    """Patch selection from an integrated day of sun."""

    def _site_with_wall(self, spot_area=1.0, resolution=1.0):
        """A 5m wall at col 5; zone spans cols 6-19, east of the wall.

        The wall is deliberately short: a tall one casts a shadow longer than
        the zone, shading every pixel equally and leaving no gradient for the
        sunniest/shadiest search to find.
        """
        dsm = np.zeros((30, 30), dtype=np.float32)
        dsm[:, 5] = 5.0
        zone = ZoneDef(
            zone_id="z1", zone_name="Zone 1",
            row_start=10, row_end=20, col_start=6, col_end=20,
            spot_area=spot_area,
        )
        return SiteModel(
            dsm=dsm, resolution=resolution, zones=[zone],
            dtm=np.zeros((30, 30), dtype=np.float32),
        )

    def test_no_zones_returns_empty(self):
        site = SiteModel(dsm=np.zeros((10, 10), dtype=np.float32), resolution=1.0)
        assert compute_zone_spot_windows(site, _day_of_sun_from_west()) == {}

    def test_sun_below_horizon_returns_empty(self):
        site = self._site_with_wall()
        samples = [(270.0, -10.0, 1.0), (90.0, 0.5, 1.0)]
        assert compute_zone_spot_windows(site, samples) == {}

    def test_zero_weight_samples_return_empty(self):
        site = self._site_with_wall()
        samples = [(270.0, 30.0, 0.0), (180.0, 45.0, 0.0)]
        assert compute_zone_spot_windows(site, samples) == {}

    def test_no_samples_at_all_returns_empty(self):
        site = self._site_with_wall()
        assert compute_zone_spot_windows(site, []) == {}

    def test_returns_both_patches_per_zone(self):
        site = self._site_with_wall()
        out = compute_zone_spot_windows(site, _day_of_sun_from_west())
        assert set(out) == {"z1"}
        assert set(out["z1"]) == {"sunniest", "shadiest"}

    def test_shadiest_patch_sits_nearer_the_wall_than_sunniest(self):
        """With the sun always in the west, shade piles up beside the wall.

        A 5m wall at 45 degrees throws a 5m shadow across a 14m zone, so the
        near columns are shaded and the far ones are not.
        """
        site = self._site_with_wall()
        out = compute_zone_spot_windows(site, _day_of_sun_from_west(elevation=45.0))
        _, sunniest_col, _ = out["z1"]["sunniest"]
        _, shadiest_col, _ = out["z1"]["shadiest"]
        assert shadiest_col < sunniest_col

    def test_patches_lie_within_the_zone(self):
        """Guards the lat/lng conversion, which assumes in-bounds offsets."""
        site = self._site_with_wall(spot_area=4.0)
        zone = site.zones[0]
        rows = zone.row_end - zone.row_start
        cols = zone.col_end - zone.col_start
        out = compute_zone_spot_windows(site, _day_of_sun_from_west())
        for r0, c0, w in out["z1"].values():
            assert 0 <= r0 and r0 + w <= rows
            assert 0 <= c0 and c0 + w <= cols


class TestWindowSizing:
    """``spot_area`` is a square patch in m², converted via the resolution."""

    def _site(self, spot_area, resolution=1.0, zone_span=14):
        dsm = np.zeros((40, 40), dtype=np.float32)
        dsm[:, 2] = 20.0
        zone = ZoneDef(
            zone_id="z", zone_name="z",
            row_start=5, row_end=5 + zone_span,
            col_start=5, col_end=5 + zone_span,
            spot_area=spot_area,
        )
        return SiteModel(dsm=dsm, resolution=resolution, zones=[zone])

    @pytest.mark.parametrize(
        "spot_area,resolution,expected",
        [
            (1.0, 1.0, 1),    # 1m² at 1m/px -> 1px
            (9.0, 1.0, 3),    # 3m x 3m
            (16.0, 1.0, 4),
            (9.0, 0.5, 6),    # 3m side at 0.5m/px -> 6px
            (4.0, 2.0, 1),    # 2m side at 2m/px -> 1px
        ],
    )
    def test_window_side_length(self, spot_area, resolution, expected):
        site = self._site(spot_area, resolution)
        out = compute_zone_spot_windows(site, _day_of_sun_from_west())
        assert out["z"]["sunniest"][2] == expected

    def test_window_is_clamped_to_the_zone(self):
        """A patch larger than the zone collapses to the whole zone."""
        site = self._site(spot_area=10000.0, zone_span=8)
        out = compute_zone_spot_windows(site, _day_of_sun_from_west())
        assert out["z"]["sunniest"][2] == 8

    def test_spot_area_is_per_zone(self):
        dsm = np.zeros((40, 40), dtype=np.float32)
        dsm[:, 2] = 20.0
        site = SiteModel(
            dsm=dsm, resolution=1.0,
            zones=[
                ZoneDef(zone_id="small", zone_name="small",
                        row_start=5, row_end=19, col_start=5, col_end=19,
                        spot_area=1.0),
                ZoneDef(zone_id="big", zone_name="big",
                        row_start=20, row_end=34, col_start=5, col_end=19,
                        spot_area=25.0),
            ],
        )
        out = compute_zone_spot_windows(site, _day_of_sun_from_west())
        assert out["small"]["sunniest"][2] == 1
        assert out["big"]["sunniest"][2] == 5


class TestMaskedZones:
    """Only masked pixels may be considered."""

    def test_patch_stays_inside_the_mask(self):
        dsm = np.zeros((30, 30), dtype=np.float32)
        dsm[:, 4] = 20.0
        mask = np.zeros((10, 10), dtype=bool)
        mask[6:9, 6:9] = True
        zone = ZoneDef(
            zone_id="m", zone_name="m",
            row_start=10, row_end=20, col_start=10, col_end=20,
            mask=mask, spot_area=1.0,
        )
        site = SiteModel(dsm=dsm, resolution=1.0, zones=[zone])
        out = compute_zone_spot_windows(site, _day_of_sun_from_west())
        for r0, c0, _ in out["m"].values():
            assert mask[r0, c0], "patch must land on a masked pixel"


class TestSpotWindowParameters:
    """The map overlay must be computed the same way as the sensor value.

    Regression test: the websocket handler used to hardcode
    ``min_shadow_height=1.5`` and leave ``canopy_model`` at its default, so the
    rectangle drawn on the map was not the patch the watering figure came from.

    The handler cannot be called directly here because conftest replaces the
    Home Assistant websocket decorators with mocks, so this reads the source.
    """

    def _handler_source(self):
        path = (
            Path(__file__).parent.parent
            / "custom_components" / "solar_shade" / "websocket_api.py"
        )
        src = path.read_text(encoding="utf-8")
        start = src.index("async def ws_get_spot_windows")
        end = src.index("@websocket_api.websocket_command", start)
        return src[start:end]

    def test_handler_uses_the_configured_min_shadow_height(self):
        src = self._handler_source()
        assert "CONF_MIN_SHADOW_HEIGHT" in src
        assert "samples, 1.5," not in src, "hardcoded min_shadow_height is back"

    def test_handler_uses_the_configured_canopy_model(self):
        src = self._handler_source()
        assert "CONF_CANOPY_MODEL" in src
        assert "canopy_model" in src

    def test_handler_passes_both_to_the_engine(self):
        src = self._handler_source()
        call = src[src.index("compute_zone_spot_windows, site"):]
        call = call[:call.index(")")]
        assert "min_shadow_height" in call
        assert "canopy_model" in call
