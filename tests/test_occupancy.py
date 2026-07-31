"""Tier 1: grid geometry, ray rasterisation, and the log-odds inverse sensor model."""

from __future__ import annotations

import numpy as np
import pytest

from diffdrive_slam.algorithm.occupancy import OccupancyGridMapper
from diffdrive_slam.model.grid import (
    GridSpec,
    LogOddsParams,
    log_odds_to_probability,
    probability_to_log_odds,
    raster_line,
)

SPEC = GridSpec(origin_x=-5.0, origin_y=-5.0, resolution=0.1, width=100, height=100)


def test_world_to_cell_round_trips_through_the_cell_centre() -> None:
    for column, row in ((0, 0), (17, 42), (99, 99)):
        x, y = SPEC.cell_to_world(column, row)
        assert SPEC.world_to_cell(x, y) == (column, row)


def test_grid_extent_covers_the_requested_area() -> None:
    x_min, x_max, y_min, y_max = SPEC.extent
    assert x_min == pytest.approx(-5.0)
    assert x_max == pytest.approx(5.0)
    assert y_min == pytest.approx(-5.0)
    assert y_max == pytest.approx(5.0)


def test_contains_rejects_out_of_bounds_cells() -> None:
    assert SPEC.contains(0, 0)
    assert not SPEC.contains(-1, 0)
    assert not SPEC.contains(0, SPEC.height)


def test_invalid_grid_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="resolution"):
        GridSpec(resolution=0.0)
    with pytest.raises(ValueError, match="positive extent"):
        GridSpec(width=0)


def test_log_odds_and_probability_are_inverse() -> None:
    for probability in (0.05, 0.25, 0.5, 0.8, 0.99):
        value = probability_to_log_odds(probability)
        recovered = float(log_odds_to_probability(np.array([value]))[0])
        assert recovered == pytest.approx(probability, abs=1e-12)


def test_zero_log_odds_is_one_half() -> None:
    assert float(log_odds_to_probability(np.zeros(1))[0]) == pytest.approx(0.5)


def test_raster_line_endpoints_and_connectivity() -> None:
    columns, rows = raster_line((3, 4), (17, 9))
    assert (int(columns[0]), int(rows[0])) == (3, 4)
    assert (int(columns[-1]), int(rows[-1])) == (17, 9)
    assert columns.size == max(abs(17 - 3), abs(9 - 4)) + 1
    steps = np.stack([np.diff(columns), np.diff(rows)], axis=1)
    assert int(np.abs(steps).max()) <= 1


def test_raster_line_of_a_single_cell() -> None:
    columns, rows = raster_line((5, 5), (5, 5))
    assert columns.tolist() == [5]
    assert rows.tolist() == [5]


def test_raster_line_is_symmetric_under_reversal() -> None:
    forward = raster_line((2, 2), (9, 5))
    backward = raster_line((9, 5), (2, 2))
    assert forward[0].tolist() == backward[0][::-1].tolist()
    assert forward[1].tolist() == backward[1][::-1].tolist()


def test_unobserved_cells_keep_the_prior() -> None:
    params = LogOddsParams(prior=0.0)
    mapper = OccupancyGridMapper(SPEC, params)
    mapper.integrate_beam(np.array([0.0, 0.0, 0.0]), 1.0, 0.0, max_range=5.0)
    grid = mapper.snapshot()
    assert float(grid[0, 0]) == pytest.approx(params.prior)
    assert float(grid[-1, -1]) == pytest.approx(params.prior)


def test_a_hit_raises_the_endpoint_and_lowers_the_free_space() -> None:
    params = LogOddsParams(occupied=0.85, free=-0.4)
    mapper = OccupancyGridMapper(SPEC, params)
    pose = np.array([0.0, 0.0, 0.0])
    mapper.integrate_beam(pose, 2.0, 0.0, max_range=5.0)
    grid = mapper.snapshot()

    endpoint = SPEC.world_to_cell(2.0, 0.0)
    midpoint = SPEC.world_to_cell(1.0, 0.0)
    assert float(grid[endpoint[1], endpoint[0]]) == pytest.approx(params.occupied)
    assert float(grid[midpoint[1], midpoint[0]]) == pytest.approx(params.free)


def test_a_maximum_range_return_marks_the_whole_beam_free() -> None:
    params = LogOddsParams()
    mapper = OccupancyGridMapper(SPEC, params)
    pose = np.array([0.0, 0.0, 0.0])
    mapper.integrate_beam(pose, 3.0, 0.0, max_range=3.0)
    grid = mapper.snapshot()
    assert float(grid.max()) <= 0.0
    assert float(grid.min()) == pytest.approx(params.free)


def test_log_odds_saturate_at_the_configured_bounds() -> None:
    params = LogOddsParams(occupied=0.85, free=-0.4, minimum=-2.0, maximum=2.0)
    mapper = OccupancyGridMapper(SPEC, params)
    pose = np.array([0.0, 0.0, 0.0])
    for _ in range(200):
        mapper.integrate_beam(pose, 2.0, 0.0, max_range=5.0)
    grid = mapper.snapshot()
    assert float(grid.max()) == pytest.approx(params.maximum)
    assert float(grid.min()) == pytest.approx(params.minimum)


def test_probabilities_stay_inside_the_unit_interval() -> None:
    mapper = OccupancyGridMapper(SPEC, LogOddsParams())
    pose = np.array([0.0, 0.0, 0.0])
    for bearing in np.linspace(-np.pi, np.pi, 24, endpoint=False):
        mapper.integrate_beam(pose, 2.5, float(bearing), max_range=5.0)
    probabilities = mapper.probabilities()
    assert float(probabilities.min()) > 0.0
    assert float(probabilities.max()) < 1.0


def test_beams_leaving_the_grid_are_clipped_not_wrapped() -> None:
    mapper = OccupancyGridMapper(SPEC, LogOddsParams())
    pose = np.array([4.9, 4.9, 0.0])
    mapper.integrate_beam(pose, 4.0, 0.0, max_range=5.0)
    grid = mapper.snapshot()
    assert float(grid[0, 0]) == pytest.approx(0.0)


def test_log_odds_view_is_read_only() -> None:
    mapper = OccupancyGridMapper(SPEC, LogOddsParams())
    with pytest.raises(ValueError, match="read-only"):
        mapper.log_odds[0, 0] = 1.0


def test_scan_shapes_are_validated() -> None:
    mapper = OccupancyGridMapper(SPEC, LogOddsParams())
    with pytest.raises(ValueError, match="must match"):
        mapper.integrate_scan(np.zeros(3), np.zeros(4), np.zeros(3), 5.0)
    with pytest.raises(ValueError, match="max_range"):
        mapper.integrate_beam(np.zeros(3), 1.0, 0.0, max_range=0.0)


def test_invalid_log_odds_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="occupied increment"):
        LogOddsParams(occupied=-0.1)
    with pytest.raises(ValueError, match="free increment"):
        LogOddsParams(free=0.1)
    with pytest.raises(ValueError, match="minimum bound"):
        LogOddsParams(minimum=1.0, maximum=-1.0)
    with pytest.raises(ValueError, match="prior"):
        LogOddsParams(prior=9.0)
