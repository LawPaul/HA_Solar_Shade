"""Tests for Solar Shade config flow coordinate validation.

Tests the lat/long validation logic used in the options flow settings step.
Since we can't easily instantiate the full HA options flow in a test without
the HA test harness, we extract and test the validation logic directly.
"""

import pytest

from custom_components.solar_shade.const import CONF_LATITUDE, CONF_LONGITUDE


def _validate_coordinates(user_input: dict) -> dict:
    """Reproduce the coordinate validation from async_step_settings.

    Returns a dict of errors (empty if valid).
    Modifies user_input in place (converts valid strings to float, pops blanks).
    """
    errors = {}
    for key, lo, hi in [
        (CONF_LATITUDE, -90, 90),
        (CONF_LONGITUDE, -180, 180),
    ]:
        val = user_input.get(key, "")
        if isinstance(val, str):
            val = val.strip()
        if val:
            try:
                fval = float(val)
                if not lo <= fval <= hi:
                    errors[key] = "invalid_coordinates"
                else:
                    user_input[key] = fval
            except ValueError:
                errors[key] = "invalid_coordinates"
        else:
            user_input.pop(key, None)
    return errors


class TestCoordinateValidation:
    """Test lat/long coordinate validation logic."""

    def test_valid_latitude(self):
        inp = {CONF_LATITUDE: "35.5"}
        errors = _validate_coordinates(inp)
        assert not errors
        assert inp[CONF_LATITUDE] == 35.5

    def test_valid_longitude(self):
        inp = {CONF_LONGITUDE: "-105.2"}
        errors = _validate_coordinates(inp)
        assert not errors
        assert inp[CONF_LONGITUDE] == -105.2

    def test_valid_both(self):
        inp = {CONF_LATITUDE: "40.0", CONF_LONGITUDE: "-105.0"}
        errors = _validate_coordinates(inp)
        assert not errors
        assert inp[CONF_LATITUDE] == 40.0
        assert inp[CONF_LONGITUDE] == -105.0

    def test_latitude_boundary_90(self):
        inp = {CONF_LATITUDE: "90"}
        errors = _validate_coordinates(inp)
        assert not errors
        assert inp[CONF_LATITUDE] == 90.0

    def test_latitude_boundary_negative_90(self):
        inp = {CONF_LATITUDE: "-90"}
        errors = _validate_coordinates(inp)
        assert not errors
        assert inp[CONF_LATITUDE] == -90.0

    def test_longitude_boundary_180(self):
        inp = {CONF_LONGITUDE: "180"}
        errors = _validate_coordinates(inp)
        assert not errors
        assert inp[CONF_LONGITUDE] == 180.0

    def test_longitude_boundary_negative_180(self):
        inp = {CONF_LONGITUDE: "-180"}
        errors = _validate_coordinates(inp)
        assert not errors
        assert inp[CONF_LONGITUDE] == -180.0

    def test_latitude_out_of_range_high(self):
        inp = {CONF_LATITUDE: "91"}
        errors = _validate_coordinates(inp)
        assert CONF_LATITUDE in errors
        assert errors[CONF_LATITUDE] == "invalid_coordinates"

    def test_latitude_out_of_range_low(self):
        inp = {CONF_LATITUDE: "-91"}
        errors = _validate_coordinates(inp)
        assert CONF_LATITUDE in errors

    def test_longitude_out_of_range_high(self):
        inp = {CONF_LONGITUDE: "181"}
        errors = _validate_coordinates(inp)
        assert CONF_LONGITUDE in errors

    def test_longitude_out_of_range_low(self):
        inp = {CONF_LONGITUDE: "-181"}
        errors = _validate_coordinates(inp)
        assert CONF_LONGITUDE in errors

    def test_non_numeric_latitude(self):
        inp = {CONF_LATITUDE: "abc"}
        errors = _validate_coordinates(inp)
        assert CONF_LATITUDE in errors
        assert errors[CONF_LATITUDE] == "invalid_coordinates"

    def test_non_numeric_longitude(self):
        inp = {CONF_LONGITUDE: "not_a_number"}
        errors = _validate_coordinates(inp)
        assert CONF_LONGITUDE in errors

    def test_empty_string_removed(self):
        inp = {CONF_LATITUDE: "", CONF_LONGITUDE: ""}
        errors = _validate_coordinates(inp)
        assert not errors
        assert CONF_LATITUDE not in inp
        assert CONF_LONGITUDE not in inp

    def test_whitespace_only_removed(self):
        inp = {CONF_LATITUDE: "   "}
        errors = _validate_coordinates(inp)
        assert not errors
        assert CONF_LATITUDE not in inp

    def test_whitespace_trimmed(self):
        inp = {CONF_LATITUDE: "  35.5  "}
        errors = _validate_coordinates(inp)
        assert not errors
        assert inp[CONF_LATITUDE] == 35.5

    def test_no_coordinate_keys_is_valid(self):
        inp = {"some_other_key": "value"}
        errors = _validate_coordinates(inp)
        assert not errors
        assert inp == {"some_other_key": "value"}

    def test_mixed_valid_and_invalid(self):
        inp = {CONF_LATITUDE: "35.5", CONF_LONGITUDE: "999"}
        errors = _validate_coordinates(inp)
        assert CONF_LATITUDE not in errors
        assert CONF_LONGITUDE in errors
        assert inp[CONF_LATITUDE] == 35.5

    def test_zero_latitude(self):
        inp = {CONF_LATITUDE: "0"}
        errors = _validate_coordinates(inp)
        assert not errors
        assert inp[CONF_LATITUDE] == 0.0

    def test_zero_longitude(self):
        inp = {CONF_LONGITUDE: "0"}
        errors = _validate_coordinates(inp)
        assert not errors
        assert inp[CONF_LONGITUDE] == 0.0
