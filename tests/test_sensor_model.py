"""Tier 1: invariants and analytic Jacobians of the range and bearing sensor model."""

from __future__ import annotations

import numpy as np
import pytest

from diffdrive_slam.model.arrays import wrap_angle
from diffdrive_slam.model.sensor import (
    RangeBearingParams,
    inverse_observation,
    inverse_observation_jacobians,
    is_visible,
    observation_jacobians,
    observe,
)
from tests.conftest import central_difference

POSES = [
    np.array([0.0, 0.0, 0.0]),
    np.array([1.5, -0.5, 1.1]),
    np.array([-2.0, 3.0, -2.4]),
]
LANDMARKS = [
    np.array([3.0, 0.0]),
    np.array([-1.0, 2.5]),
    np.array([0.4, -3.2]),
]


def test_measurement_of_a_landmark_straight_ahead() -> None:
    pose = np.array([0.0, 0.0, 0.0])
    measurement = observe(pose, np.array([2.0, 0.0]))
    assert measurement[0] == pytest.approx(2.0)
    assert measurement[1] == pytest.approx(0.0)


def test_bearing_is_relative_to_the_heading() -> None:
    landmark = np.array([1.0, 0.0])
    forward = observe(np.array([0.0, 0.0, 0.0]), landmark)
    turned = observe(np.array([0.0, 0.0, np.pi / 2.0]), landmark)
    assert forward[1] == pytest.approx(0.0)
    assert turned[1] == pytest.approx(-np.pi / 2.0)


def test_bearing_is_wrapped() -> None:
    pose = np.array([0.0, 0.0, 3.0])
    measurement = observe(pose, np.array([1.0, 0.0]))
    assert -np.pi <= measurement[1] < np.pi


def test_forward_and_inverse_models_are_consistent() -> None:
    for pose in POSES:
        for landmark in LANDMARKS:
            measurement = observe(pose, landmark)
            recovered = inverse_observation(pose, measurement)
            np.testing.assert_allclose(recovered, landmark, atol=1e-12)


def _relative(measurement: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Shift the bearing by a constant reference so the wrap point cannot be crossed.

    The constant offset differentiates away, so the finite difference of the shifted
    measurement equals the finite difference of the measurement itself.
    """
    shifted = measurement.copy()
    shifted[1] = wrap_angle(float(measurement[1]) - float(reference[1]))
    return shifted


@pytest.mark.parametrize("pose", POSES)
@pytest.mark.parametrize("landmark", LANDMARKS)
def test_observation_jacobians_match_finite_differences(
    pose: np.ndarray, landmark: np.ndarray
) -> None:
    reference = observe(pose, landmark)
    analytic_pose, analytic_landmark = observation_jacobians(pose, landmark)
    numeric_pose = central_difference(
        lambda value: _relative(observe(value, landmark), reference), pose
    )
    numeric_landmark = central_difference(
        lambda value: _relative(observe(pose, value), reference), landmark
    )
    np.testing.assert_allclose(analytic_pose, numeric_pose, atol=1e-7)
    np.testing.assert_allclose(analytic_landmark, numeric_landmark, atol=1e-7)


@pytest.mark.parametrize("pose", POSES)
@pytest.mark.parametrize("landmark", LANDMARKS)
def test_inverse_observation_jacobians_match_finite_differences(
    pose: np.ndarray, landmark: np.ndarray
) -> None:
    measurement = observe(pose, landmark)
    analytic_pose, analytic_measurement = inverse_observation_jacobians(pose, measurement)
    numeric_pose = central_difference(lambda value: inverse_observation(value, measurement), pose)
    numeric_measurement = central_difference(
        lambda value: inverse_observation(pose, value), measurement
    )
    np.testing.assert_allclose(analytic_pose, numeric_pose, atol=1e-7)
    np.testing.assert_allclose(analytic_measurement, numeric_measurement, atol=1e-7)


def test_visibility_respects_range_and_field_of_view() -> None:
    params = RangeBearingParams(max_range=2.0, field_of_view=np.pi / 2.0)
    pose = np.array([0.0, 0.0, 0.0])
    assert is_visible(pose, np.array([1.5, 0.0]), params)
    assert not is_visible(pose, np.array([3.0, 0.0]), params)
    assert not is_visible(pose, np.array([0.0, 1.5]), params)


def test_covariance_is_diagonal_and_positive() -> None:
    params = RangeBearingParams(sigma_range=0.2, sigma_bearing=0.05)
    covariance = params.covariance()
    assert covariance[0, 0] == pytest.approx(0.04)
    assert covariance[1, 1] == pytest.approx(0.0025)
    assert covariance[0, 1] == pytest.approx(0.0)


def test_invalid_sensor_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="standard deviations"):
        RangeBearingParams(sigma_range=0.0)
    with pytest.raises(ValueError, match="max_range"):
        RangeBearingParams(max_range=-1.0)
    with pytest.raises(ValueError, match="field_of_view"):
        RangeBearingParams(field_of_view=0.0)
