"""Tier 1: invariants and analytic Jacobians of the differential drive motion model."""

from __future__ import annotations

import numpy as np
import pytest

from diffdrive_slam.model.motion import (
    Control,
    DifferentialDriveParams,
    MotionNoise,
    control_noise_covariance,
    motion_jacobian_control,
    motion_jacobian_state,
    predict_pose,
    process_noise_covariance,
)
from tests.conftest import central_difference

POSES = [
    np.array([0.0, 0.0, 0.0]),
    np.array([1.3, -2.7, 0.6]),
    np.array([-4.0, 3.5, -2.1]),
]
CONTROLS = [
    Control(1.0, 0.0),
    Control(1.0, 0.7),
    Control(-0.6, -1.3),
    Control(0.0, 0.9),
    Control(0.0, 0.0),
]


def test_straight_line_is_integrated_exactly() -> None:
    pose = np.array([2.0, -1.0, np.pi / 6.0])
    control = Control(linear_velocity=1.5, angular_velocity=0.0)
    result = predict_pose(pose, control, 2.0)
    distance = 1.5 * 2.0
    expected = np.array(
        [
            pose[0] + distance * np.cos(pose[2]),
            pose[1] + distance * np.sin(pose[2]),
            pose[2],
        ]
    )
    np.testing.assert_allclose(result, expected, atol=1e-15)


def test_straight_line_accumulates_over_substeps() -> None:
    pose = np.array([0.0, 0.0, 0.3])
    control = Control(linear_velocity=2.0, angular_velocity=0.0)
    single = predict_pose(pose, control, 1.0)
    stepped = pose.copy()
    for _ in range(10):
        stepped = predict_pose(stepped, control, 0.1)
    np.testing.assert_allclose(single, stepped, atol=1e-12)


def test_pure_rotation_leaves_position_unchanged() -> None:
    pose = np.array([-1.25, 4.0, 0.2])
    control = Control(linear_velocity=0.0, angular_velocity=0.75)
    result = predict_pose(pose, control, 1.6)
    np.testing.assert_allclose(result[:2], pose[:2], atol=1e-15)
    assert result[2] == pytest.approx(0.2 + 0.75 * 1.6, abs=1e-15)


def test_full_circle_returns_to_start() -> None:
    pose = np.array([1.0, 1.0, 0.0])
    control = Control(linear_velocity=1.0, angular_velocity=0.5)
    period = 2.0 * np.pi / 0.5
    result = predict_pose(pose, control, period)
    np.testing.assert_allclose(result[:2], pose[:2], atol=1e-12)


def test_arc_radius_matches_the_kinematics() -> None:
    pose = np.array([0.0, 0.0, 0.0])
    control = Control(linear_velocity=2.0, angular_velocity=0.5)
    centre = np.array([0.0, control.linear_velocity / control.angular_velocity])
    for dt in (0.3, 1.1, 2.9):
        result = predict_pose(pose, control, dt)
        radius = float(np.linalg.norm(result[:2] - centre))
        assert radius == pytest.approx(4.0, abs=1e-12)


def test_heading_output_is_wrapped() -> None:
    pose = np.array([0.0, 0.0, 3.0])
    control = Control(linear_velocity=0.0, angular_velocity=1.0)
    result = predict_pose(pose, control, 1.0)
    assert -np.pi <= result[2] < np.pi


def test_arc_branch_agrees_with_the_straight_line_limit() -> None:
    pose = np.array([0.5, -0.5, 0.4])
    dt = 0.1
    straight = predict_pose(pose, Control(1.0, 0.0), dt)
    nearly_straight = predict_pose(pose, Control(1.0, 1e-7), dt)
    np.testing.assert_allclose(straight, nearly_straight, atol=1e-8)


@pytest.mark.parametrize("pose", POSES)
@pytest.mark.parametrize("control", CONTROLS)
def test_state_jacobian_matches_finite_differences(pose: np.ndarray, control: Control) -> None:
    dt = 0.4
    numeric = central_difference(lambda value: predict_pose(value, control, dt), pose)
    analytic = motion_jacobian_state(pose, control, dt)
    np.testing.assert_allclose(analytic, numeric, atol=1e-7)


@pytest.mark.parametrize("pose", POSES)
@pytest.mark.parametrize("control", CONTROLS)
def test_control_jacobian_matches_finite_differences(pose: np.ndarray, control: Control) -> None:
    dt = 0.4

    def evaluate(values: np.ndarray) -> np.ndarray:
        return predict_pose(pose, Control.from_array(values), dt)

    numeric = central_difference(evaluate, control.as_array(), step=1e-4)
    analytic = motion_jacobian_control(pose, control, dt)
    np.testing.assert_allclose(analytic, numeric, atol=1e-6)


def test_control_noise_grows_with_velocity() -> None:
    noise = MotionNoise()
    slow = control_noise_covariance(Control(0.5, 0.1), noise)
    fast = control_noise_covariance(Control(2.0, 0.1), noise)
    assert fast[0, 0] > slow[0, 0]
    assert np.allclose(slow, np.diag(np.diag(slow)))


@pytest.mark.parametrize("control", CONTROLS)
def test_process_noise_is_symmetric_positive_semidefinite(control: Control) -> None:
    pose = np.array([0.2, -0.3, 0.9])
    covariance = process_noise_covariance(pose, control, 0.1, MotionNoise())
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-15)
    assert float(np.linalg.eigvalsh(covariance).min()) > 0.0


def test_wheel_rates_round_trip() -> None:
    params = DifferentialDriveParams(wheel_radius=0.06, wheel_base=0.35)
    control = Control(linear_velocity=0.8, angular_velocity=-0.4)
    left, right = params.wheel_rates(control)
    recovered = params.body_velocity(left, right)
    assert recovered.linear_velocity == pytest.approx(control.linear_velocity)
    assert recovered.angular_velocity == pytest.approx(control.angular_velocity)


def test_equal_wheel_rates_drive_straight() -> None:
    params = DifferentialDriveParams(wheel_radius=0.06, wheel_base=0.35)
    control = params.body_velocity(3.0, 3.0)
    assert control.angular_velocity == pytest.approx(0.0)
    assert control.linear_velocity == pytest.approx(0.18)


def test_opposite_wheel_rates_spin_in_place() -> None:
    params = DifferentialDriveParams(wheel_radius=0.06, wheel_base=0.35)
    control = params.body_velocity(-2.0, 2.0)
    assert control.linear_velocity == pytest.approx(0.0)
    assert control.angular_velocity == pytest.approx(2.0 * 0.06 * 2.0 / 0.35)


def test_invalid_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="wheel_radius"):
        DifferentialDriveParams(wheel_radius=0.0, wheel_base=0.3)
    with pytest.raises(ValueError, match="wheel_base"):
        DifferentialDriveParams(wheel_radius=0.1, wheel_base=-1.0)


def test_negative_noise_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MotionNoise(alpha_1=-0.1)
