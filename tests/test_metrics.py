"""Tier 1: the accuracy and consistency metrics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from diffdrive_slam.analysis.metrics import (
    absolute_trajectory_error,
    association_summary,
    consistency_summary,
    evaluate,
    grid_summary,
    landmark_error,
    nees_bounds,
    pose_errors,
    pose_nees,
)
from diffdrive_slam.pipeline.trace import Trace


def test_zero_error_trajectory() -> None:
    poses = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.5]])
    result = absolute_trajectory_error(poses, poses)
    assert result.position_rmse == pytest.approx(0.0)
    assert result.heading_rmse == pytest.approx(0.0)
    assert result.samples == 2


def test_trajectory_error_is_the_root_mean_square() -> None:
    truth = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    estimate = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    result = absolute_trajectory_error(truth, estimate)
    assert result.position_max == pytest.approx(5.0)
    assert result.position_mean == pytest.approx(2.5)
    assert result.position_rmse == pytest.approx(np.sqrt(12.5))


def test_heading_error_is_wrapped() -> None:
    truth = np.array([[0.0, 0.0, np.pi - 0.05]])
    estimate = np.array([[0.0, 0.0, -np.pi + 0.05]])
    errors = pose_errors(truth, estimate)
    assert abs(float(errors[0, 2])) == pytest.approx(0.1, abs=1e-12)


def test_pose_error_shapes_are_validated() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        pose_errors(np.zeros((2, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="shape"):
        pose_errors(np.zeros((2, 4)), np.zeros((2, 4)))


def test_landmark_error_scores_only_identified_slots() -> None:
    truth = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    estimate = np.array([[0.0, 0.3], [1.0, -0.4], [9.0, 9.0]])
    result = landmark_error(truth, estimate, (0, 1, -1))
    assert result.estimated == 3
    assert result.matched == 2
    assert result.rmse == pytest.approx(np.sqrt((0.09 + 0.16) / 2.0))
    assert result.maximum == pytest.approx(0.4)


def test_landmark_error_requires_one_identity_per_slot() -> None:
    with pytest.raises(ValueError, match="one entry per estimated landmark"):
        landmark_error(np.zeros((2, 2)), np.zeros((2, 2)), (0,))


def test_nees_of_a_unit_covariance_is_the_squared_error() -> None:
    truth = np.array([[1.0, 2.0, 0.3]])
    estimate = np.zeros((1, 3))
    covariances = np.eye(3)[None, :, :]
    values = pose_nees(truth, estimate, covariances)
    assert float(values[0]) == pytest.approx(1.0 + 4.0 + 0.09)


def test_nees_shapes_are_validated() -> None:
    with pytest.raises(ValueError, match="shape"):
        pose_nees(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((3, 3, 3)))


def test_nees_bounds_bracket_the_expected_value() -> None:
    lower, upper = nees_bounds(3, 1, 0.95)
    assert lower == pytest.approx(float(chi2.ppf(0.025, 3)))
    assert upper == pytest.approx(float(chi2.ppf(0.975, 3)))
    assert lower < 3.0 < upper


def test_nees_bounds_tighten_as_samples_grow() -> None:
    narrow = nees_bounds(3, 100, 0.95)
    wide = nees_bounds(3, 1, 0.95)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_nees_bounds_reject_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        nees_bounds(0, 5)
    with pytest.raises(ValueError, match="confidence"):
        nees_bounds(3, 5, confidence=0.0)


def test_consistency_verdicts() -> None:
    rng = np.random.default_rng(0)
    consistent = consistency_summary(
        np.asarray(rng.chisquare(3, size=4000), dtype=np.float64)
    )
    assert consistent.verdict == "consistent"
    assert consistent.consistent

    optimistic = consistency_summary(np.full(500, 9.0))
    assert optimistic.verdict == "optimistic"
    assert not optimistic.consistent

    conservative = consistency_summary(np.full(500, 0.5))
    assert conservative.verdict == "conservative"


def test_consistency_requires_a_non_empty_sequence() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        consistency_summary(np.zeros(0))


def test_consistency_carries_both_intervals() -> None:
    summary = consistency_summary(np.full(400, 3.0), samples_per_value=20)
    assert (summary.per_step_lower, summary.per_step_upper) == nees_bounds(3, 20)
    assert (summary.lower_bound, summary.upper_bound) == nees_bounds(3, 400 * 20)
    assert summary.per_step_lower < summary.lower_bound
    assert summary.upper_bound < summary.per_step_upper


def test_inside_fraction_uses_the_per_step_interval() -> None:
    values = np.array([3.0, 3.0, 3.0, 100.0])
    summary = consistency_summary(values)
    assert summary.inside_fraction == pytest.approx(0.75)


def test_grid_summary_on_a_perfect_map() -> None:
    truth = np.zeros((5, 5), dtype=np.bool_)
    truth[2, 2] = True
    log_odds = np.full((5, 5), -3.0)
    log_odds[2, 2] = 3.0
    summary = grid_summary(log_odds, truth, tolerance_cells=0)
    assert summary.occupied == 1
    assert summary.free == 24
    assert summary.unknown == 0
    assert summary.occupied_agreement == pytest.approx(1.0)
    assert summary.free_agreement == pytest.approx(1.0)
    assert summary.overall_agreement == pytest.approx(1.0)
    assert summary.decided_fraction == pytest.approx(1.0)


def test_grid_summary_counts_unknown_cells() -> None:
    truth = np.zeros((4, 4), dtype=np.bool_)
    summary = grid_summary(np.zeros((4, 4)), truth)
    assert summary.unknown == 16
    assert summary.decided_fraction == pytest.approx(0.0)
    assert np.isnan(summary.overall_agreement)


def test_grid_summary_tolerance_forgives_a_one_cell_offset() -> None:
    truth = np.zeros((5, 5), dtype=np.bool_)
    truth[2, 2] = True
    log_odds = np.full((5, 5), -3.0)
    log_odds[2, 3] = 3.0
    strict = grid_summary(log_odds, truth, tolerance_cells=0)
    lenient = grid_summary(log_odds, truth, tolerance_cells=1)
    assert strict.occupied_agreement == pytest.approx(0.0)
    assert lenient.occupied_agreement == pytest.approx(1.0)


def test_grid_summary_validates_its_arguments() -> None:
    truth = np.zeros((3, 3), dtype=np.bool_)
    with pytest.raises(ValueError, match="shapes must match"):
        grid_summary(np.zeros((2, 2)), truth)
    with pytest.raises(ValueError, match="thresholds"):
        grid_summary(np.zeros((3, 3)), truth, occupied_threshold=0.2, free_threshold=0.7)
    with pytest.raises(ValueError, match="tolerance_cells"):
        grid_summary(np.zeros((3, 3)), truth, tolerance_cells=-1)


def test_evaluate_reports_every_family_of_metric(medium_trace: Trace) -> None:
    result = evaluate(medium_trace)
    assert result.trajectory.samples == len(medium_trace)
    assert result.landmarks.matched > 0
    assert result.consistency.degrees_of_freedom == 3
    assert result.associations.measurements > 0


def test_slam_beats_dead_reckoning_once_the_map_is_built(medium_trace: Trace) -> None:
    result = evaluate(medium_trace)
    assert result.dead_reckoning.position_rmse > result.trajectory.position_rmse


def test_association_summary_counts_every_decision(short_trace: Trace) -> None:
    summary = association_summary(short_trace)
    total = summary.matched + summary.initialised + summary.rejected
    assert summary.measurements == total
    assert summary.incorrect == 0
    assert summary.accuracy == pytest.approx(1.0)
