"""Tier 1: the figure helpers, checked on their content rather than on their pixels.

Matplotlib output is not byte reproducible across platforms, so nothing here
compares images. What is asserted instead is that each figure carries the series it
claims to carry, that the covariance ellipse has the geometry the chi-square scaling
implies, and that saving honours the resolution it was given.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2

from diffdrive_slam.analysis.figures import (
    covariance_ellipse,
    plot_error_history,
    plot_nees,
    plot_occupancy_grid,
    plot_trajectory,
    save_figure,
)
from diffdrive_slam.analysis.metrics import consistency_summary, pose_nees
from diffdrive_slam.model.sensor import RangeBearingParams
from diffdrive_slam.pipeline.environment import arena_environment
from diffdrive_slam.pipeline.simulate import SimulationConfig, run_simulation
from diffdrive_slam.pipeline.trace import Trace


def test_covariance_ellipse_scales_with_the_chi_square_quantile() -> None:
    covariance = np.diag([4.0, 1.0])
    ellipse = covariance_ellipse(np.array([1.0, -2.0]), covariance, confidence=0.95)
    scale = float(np.sqrt(chi2.ppf(0.95, 2)))
    assert ellipse.get_center() == pytest.approx((1.0, -2.0))
    assert ellipse.width == pytest.approx(2.0 * 2.0 * scale)
    assert ellipse.height == pytest.approx(2.0 * 1.0 * scale)
    assert float(ellipse.angle) == pytest.approx(0.0)


def test_covariance_ellipse_follows_the_principal_axis() -> None:
    rotation = np.array([[np.cos(0.6), -np.sin(0.6)], [np.sin(0.6), np.cos(0.6)]])
    covariance = rotation @ np.diag([9.0, 1.0]) @ rotation.T
    ellipse = covariance_ellipse(np.zeros(2), covariance)
    # An eigenvector is defined up to sign, so the axis is only fixed modulo pi.
    angle = float(np.deg2rad(float(ellipse.angle))) % np.pi
    assert angle == pytest.approx(0.6, abs=1e-9)
    assert ellipse.width > ellipse.height


def test_covariance_ellipse_tolerates_a_semidefinite_block() -> None:
    """A rank deficient block can carry a small negative eigenvalue after rounding."""
    ellipse = covariance_ellipse(np.zeros(2), np.diag([1.0, -1e-18]))
    assert ellipse.height == pytest.approx(0.0, abs=1e-6)


def test_trajectory_figure_draws_all_three_paths(short_trace: Trace) -> None:
    figure = plot_trajectory(short_trace)
    axes = figure.axes[0]
    labels = [line.get_label() for line in axes.get_lines()]
    assert labels == ["ground truth", "dead reckoning", "EKF-SLAM"]
    for line in axes.get_lines():
        assert len(line.get_xdata()) == len(short_trace)
    assert len(axes.patches) == short_trace.final_state.num_landmarks
    assert axes.get_xlabel() == "x [m]"


def test_trajectory_figure_survives_an_empty_map() -> None:
    blind = RangeBearingParams(max_range=0.01)
    trace = run_simulation(SimulationConfig(steps=4, seed=2, build_grid=False, sensor=blind))
    assert trace.final_state.num_landmarks == 0
    assert list(plot_trajectory(trace).axes[0].patches) == []


def test_error_history_plots_the_two_error_curves(short_trace: Trace) -> None:
    figure = plot_error_history(short_trace)
    upper, lower = figure.axes
    odometry, slam = upper.get_lines()
    truth = short_trace.true_poses[:, :2]
    expected = np.linalg.norm(truth - short_trace.estimated_poses[:, :2], axis=1)
    np.testing.assert_allclose(np.asarray(slam.get_ydata()), expected)
    assert float(np.mean(np.asarray(odometry.get_ydata()))) > float(np.mean(expected))
    np.testing.assert_allclose(
        np.asarray(lower.get_lines()[0].get_ydata()), short_trace.landmark_counts
    )


def test_nees_figure_bands_match_the_summary(short_trace: Trace) -> None:
    nees = pose_nees(
        short_trace.true_poses, short_trace.estimated_poses, short_trace.pose_covariances
    )
    summary = consistency_summary(nees)
    figure = plot_nees(short_trace.times, nees, summary)
    axes = figure.axes[0]
    levels = sorted(float(line.get_ydata()[0]) for line in axes.get_lines()[1:])
    assert levels == pytest.approx(
        sorted([summary.per_step_lower, float(summary.degrees_of_freedom), summary.per_step_upper])
    )
    assert "95 percent" in axes.get_title()


def test_occupancy_figure_overlays_the_true_walls(short_trace: Trace) -> None:
    walls = arena_environment().walls
    figure = plot_occupancy_grid(short_trace, walls)
    axes = figure.axes[0]
    assert len(axes.images) == 1
    assert len(axes.get_lines()) == walls.shape[0] + 2
    labelled = [
        line.get_label() for line in axes.get_lines() if not str(line.get_label()).startswith("_")
    ]
    assert labelled == ["true walls", "ground truth", "EKF-SLAM"]


def test_occupancy_figure_without_walls_draws_only_the_paths(short_trace: Trace) -> None:
    axes = plot_occupancy_grid(short_trace).axes[0]
    assert len(axes.get_lines()) == 2


def test_occupancy_figure_requires_a_grid() -> None:
    trace = run_simulation(SimulationConfig(steps=4, seed=8, build_grid=False))
    with pytest.raises(ValueError, match="does not carry an occupancy grid"):
        plot_occupancy_grid(trace)


def test_save_figure_creates_the_directory_and_writes_a_png(
    short_trace: Trace, tmp_path: Path
) -> None:
    target = tmp_path / "nested" / "trajectory.png"
    written = save_figure(plot_trajectory(short_trace), target)
    assert written == target
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_lower_resolution_produces_a_smaller_file(short_trace: Trace, tmp_path: Path) -> None:
    coarse = save_figure(plot_trajectory(short_trace), tmp_path / "coarse.png", dpi=60)
    fine = save_figure(plot_trajectory(short_trace), tmp_path / "fine.png", dpi=140)
    assert coarse.stat().st_size < fine.stat().st_size


def test_save_figure_rejects_a_non_positive_resolution(
    short_trace: Trace, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="dpi must be positive"):
        save_figure(plot_trajectory(short_trace), tmp_path / "bad.png", dpi=0)
