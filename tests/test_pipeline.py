"""Tier 1: the simulated environment, the control sequences, and the trace."""

from __future__ import annotations

import numpy as np
import pytest

from diffdrive_slam.model.grid import GridSpec
from diffdrive_slam.model.motion import Control
from diffdrive_slam.model.sensor import RangeBearingParams, observe
from diffdrive_slam.pipeline.environment import (
    Environment,
    arena_environment,
    rectangle_segments,
    sample_landmarks,
)
from diffdrive_slam.pipeline.simulate import SimulationConfig, run_simulation
from diffdrive_slam.pipeline.trace import Trace
from diffdrive_slam.pipeline.trajectory import (
    figure_eight_controls,
    repeat_to_length,
    square_loop_controls,
    square_loop_start,
)


def test_rectangle_has_four_closed_segments() -> None:
    walls = rectangle_segments(-1.0, -1.0, 1.0, 1.0)
    assert walls.shape == (4, 4)
    for index in range(4):
        np.testing.assert_allclose(walls[index, 2:], walls[(index + 1) % 4, :2])


def test_range_scan_finds_the_nearest_wall() -> None:
    environment = Environment(
        landmarks=np.zeros((0, 2)), walls=rectangle_segments(-2.0, -2.0, 2.0, 2.0)
    )
    pose = np.array([0.0, 0.0, 0.0])
    bearings = np.array([0.0, np.pi / 2.0, np.pi, -np.pi / 2.0])
    ranges = environment.range_scan(pose, bearings, max_range=10.0)
    np.testing.assert_allclose(ranges, np.full(4, 2.0), atol=1e-12)


def test_range_scan_saturates_at_the_maximum_range() -> None:
    environment = Environment(
        landmarks=np.zeros((0, 2)), walls=rectangle_segments(-20.0, -20.0, 20.0, 20.0)
    )
    ranges = environment.range_scan(np.zeros(3), np.array([0.0]), max_range=5.0)
    assert float(ranges[0]) == pytest.approx(5.0)


def test_range_scan_respects_the_heading() -> None:
    environment = Environment(
        landmarks=np.zeros((0, 2)), walls=np.array([[1.0, -5.0, 1.0, 5.0]])
    )
    forward = environment.range_scan(np.array([0.0, 0.0, 0.0]), np.array([0.0]), 10.0)
    backward = environment.range_scan(np.array([0.0, 0.0, np.pi]), np.array([0.0]), 10.0)
    assert float(forward[0]) == pytest.approx(1.0)
    assert float(backward[0]) == pytest.approx(10.0)


def test_visible_landmarks_match_the_sensor_model() -> None:
    environment = arena_environment(seed=3, landmark_count=12)
    params = RangeBearingParams(max_range=3.0, field_of_view=np.pi)
    pose = np.array([0.0, -3.0, 0.4])
    identities, measurements = environment.visible_landmarks(pose, params)
    assert identities.shape[0] == measurements.shape[0]
    for identity, measurement in zip(identities, measurements, strict=True):
        np.testing.assert_allclose(
            measurement, observe(pose, environment.landmarks[identity]), atol=1e-12
        )
        assert float(measurement[0]) <= params.max_range
        assert abs(float(measurement[1])) <= 0.5 * params.field_of_view


def test_sampled_landmarks_respect_the_minimum_separation() -> None:
    rng = np.random.default_rng(1)
    landmarks = sample_landmarks(
        rng, count=15, half_extent=4.0, exclusion_half_extent=1.0, minimum_separation=1.2
    )
    assert landmarks.shape == (15, 2)
    distances = np.linalg.norm(landmarks[:, None, :] - landmarks[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    assert float(distances.min()) >= 1.2
    assert float(np.abs(landmarks).max(axis=1).min()) >= 1.0


def test_sampling_reports_failure_rather_than_looping_forever() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="only placed"):
        sample_landmarks(
            rng,
            count=200,
            half_extent=1.0,
            exclusion_half_extent=0.0,
            minimum_separation=1.0,
            max_attempts=300,
        )


def test_environment_shapes_are_validated() -> None:
    with pytest.raises(ValueError, match="landmarks"):
        Environment(landmarks=np.zeros(4), walls=np.zeros((1, 4)))
    with pytest.raises(ValueError, match="walls"):
        Environment(landmarks=np.zeros((1, 2)), walls=np.zeros((1, 2)))


def test_rasterised_walls_mark_the_boundary() -> None:
    environment = Environment(
        landmarks=np.zeros((0, 2)), walls=rectangle_segments(-1.0, -1.0, 1.0, 1.0)
    )
    spec = GridSpec(origin_x=-2.0, origin_y=-2.0, resolution=0.1, width=40, height=40)
    truth = environment.rasterise_walls(spec)
    corner = spec.world_to_cell(-1.0, -1.0)
    centre = spec.world_to_cell(0.0, 0.0)
    assert bool(truth[corner[1], corner[0]])
    assert not bool(truth[centre[1], centre[0]])


def test_square_loop_closes_on_itself() -> None:
    from diffdrive_slam.model.motion import predict_pose

    controls = square_loop_controls(dt=0.1, side=6.0, speed=1.0, laps=1)
    pose = np.array([-3.0, -3.0, 0.0])
    start = pose.copy()
    for control in controls:
        pose = predict_pose(pose, control, 0.1)
    assert float(np.linalg.norm(pose[:2] - start[:2])) < 1e-9


def test_square_loop_start_centres_the_loop() -> None:
    from diffdrive_slam.model.motion import predict_pose

    side, speed, turn_rate = 5.0, 1.0, float(np.pi) / 4.0
    pose = np.asarray(square_loop_start(side, speed, turn_rate), dtype=np.float64)
    visited = [pose[:2].copy()]
    for control in square_loop_controls(0.1, side, speed, turn_rate, laps=1):
        pose = predict_pose(pose, control, 0.1)
        visited.append(pose[:2].copy())
    points = np.asarray(visited)
    half_extent = 0.5 * side + speed / turn_rate
    np.testing.assert_allclose(points.min(axis=0), -half_extent, atol=1e-9)
    np.testing.assert_allclose(points.max(axis=0), half_extent, atol=1e-9)


def test_square_loop_start_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="positive"):
        square_loop_start(side=0.0)


def test_the_default_trajectory_stays_inside_the_arena() -> None:
    trace = run_simulation(SimulationConfig(steps=640, seed=13, build_grid=False))
    wall_half_extent = 5.5
    assert float(np.abs(trace.true_poses[:, :2]).max()) < wall_half_extent


def test_the_default_grid_covers_the_arena() -> None:
    spec = GridSpec()
    x_min, x_max, y_min, y_max = spec.extent
    assert x_min <= -5.5 and x_max >= 5.5
    assert y_min <= -5.5 and y_max >= 5.5


def test_control_sequences_have_the_expected_length() -> None:
    controls = square_loop_controls(dt=0.1, side=6.0, speed=1.0, laps=2)
    assert len(controls) == 2 * 4 * (60 + 20)
    assert len(figure_eight_controls(dt=0.1, lobes=1)) == 2 * 80


def test_repeat_to_length_cycles_the_sequence() -> None:
    controls = (Control(1.0, 0.0), Control(0.0, 1.0))
    repeated = repeat_to_length(controls, 5)
    assert len(repeated) == 5
    assert repeated[4] == controls[0]


def test_trajectory_generators_validate_their_arguments() -> None:
    with pytest.raises(ValueError, match="positive"):
        square_loop_controls(dt=0.0)
    with pytest.raises(ValueError, match="laps"):
        square_loop_controls(laps=0)
    with pytest.raises(ValueError, match="lobes"):
        figure_eight_controls(lobes=0)
    with pytest.raises(ValueError, match="must not be empty"):
        repeat_to_length((), 4)


def test_trace_arrays_have_consistent_shapes(short_trace: Trace) -> None:
    length = len(short_trace)
    assert short_trace.times.shape == (length,)
    assert short_trace.true_poses.shape == (length, 3)
    assert short_trace.estimated_poses.shape == (length, 3)
    assert short_trace.pose_covariances.shape == (length, 3, 3)
    assert short_trace.dead_reckoned_poses.shape == (length, 3)
    assert short_trace.landmark_counts.shape == (length,)


def test_trace_records_the_grid_when_requested(short_trace: Trace) -> None:
    assert short_trace.grid is not None
    assert short_trace.occupancy_log_odds is not None
    assert short_trace.occupancy_log_odds.shape == short_trace.grid.shape


def test_landmark_count_never_decreases(short_trace: Trace) -> None:
    counts = short_trace.landmark_counts
    assert bool(np.all(np.diff(counts) >= 0))
    assert int(counts[-1]) == short_trace.final_state.num_landmarks


def test_every_estimated_landmark_carries_an_identity(short_trace: Trace) -> None:
    assert len(short_trace.slot_to_identity) == short_trace.final_state.num_landmarks
    assert all(identity >= 0 for identity in short_trace.slot_to_identity)


def test_simulation_is_reproducible() -> None:
    config = SimulationConfig(steps=40, seed=77, build_grid=False)
    first = run_simulation(config)
    second = run_simulation(config)
    np.testing.assert_allclose(first.true_poses, second.true_poses, atol=0.0)
    np.testing.assert_allclose(first.estimated_poses, second.estimated_poses, atol=0.0)


def test_grid_mapping_does_not_perturb_the_trajectory() -> None:
    without = run_simulation(SimulationConfig(steps=60, seed=31, build_grid=False))
    with_grid = run_simulation(SimulationConfig(steps=60, seed=31, build_grid=True))
    np.testing.assert_allclose(without.true_poses, with_grid.true_poses, atol=0.0)
    np.testing.assert_allclose(
        without.estimated_poses, with_grid.estimated_poses, atol=0.0
    )


def test_different_seeds_give_different_runs() -> None:
    first = run_simulation(SimulationConfig(steps=40, seed=1, build_grid=False))
    second = run_simulation(SimulationConfig(steps=40, seed=2, build_grid=False))
    assert not np.allclose(first.true_poses, second.true_poses)


def test_known_correspondence_mode_creates_no_spurious_landmarks() -> None:
    trace = run_simulation(
        SimulationConfig(steps=200, seed=4, build_grid=False, known_correspondence=True)
    )
    assert all(identity >= 0 for identity in trace.slot_to_identity)
    assert len(set(trace.slot_to_identity)) == len(trace.slot_to_identity)


def test_simulation_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="steps"):
        SimulationConfig(steps=0)
    with pytest.raises(ValueError, match="dt"):
        SimulationConfig(dt=0.0)
    with pytest.raises(ValueError, match="scan_interval"):
        SimulationConfig(scan_interval=0)
    with pytest.raises(ValueError, match="scan_beams"):
        SimulationConfig(scan_beams=0)
