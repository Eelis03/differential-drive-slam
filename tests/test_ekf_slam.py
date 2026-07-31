"""Tier 1: filter invariants, uncertainty reduction, and convergence to ground truth."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from diffdrive_slam.algorithm.association import AssociationKind
from diffdrive_slam.algorithm.ekf_slam import EkfSlam, EkfSlamConfig
from diffdrive_slam.model.arrays import is_positive_semidefinite, is_symmetric, wrap_angle
from diffdrive_slam.model.motion import Control, MotionNoise
from diffdrive_slam.model.sensor import RangeBearingParams, observe
from diffdrive_slam.model.state import SlamState

LANDMARKS = np.array(
    [
        [2.0, 1.0],
        [-1.5, 2.5],
        [0.5, -2.0],
        [3.0, -1.0],
    ]
)


def build_filter(sigma_range: float = 0.05, sigma_bearing: float = 0.01) -> EkfSlam:
    """Return a filter at the origin with a small initial covariance."""
    config = EkfSlamConfig(
        motion_noise=MotionNoise(),
        sensor=RangeBearingParams(
            sigma_range=sigma_range, sigma_bearing=sigma_bearing, max_range=10.0
        ),
    )
    return EkfSlam(
        np.zeros(3), np.diag([1e-4, 1e-4, 1e-5]), config
    )


def test_initial_state_holds_only_the_pose() -> None:
    slam = build_filter()
    assert slam.state.dimension == 3
    assert slam.num_landmarks == 0


def test_prediction_grows_the_pose_covariance() -> None:
    slam = build_filter()
    before = float(np.trace(slam.state.robot_covariance))
    slam.predict(Control(1.0, 0.2), 0.1)
    after = float(np.trace(slam.state.robot_covariance))
    assert after > before


def test_covariance_stays_symmetric_and_positive_semidefinite() -> None:
    slam = build_filter()
    pose = np.zeros(3)
    for step in range(60):
        control = Control(1.0, 0.3 if step % 2 else -0.2)
        slam.predict(control, 0.1)
        pose = _integrate(pose, control, 0.1)
        measurements = np.array([observe(pose, landmark) for landmark in LANDMARKS])
        slam.integrate(measurements, [0, 1, 2, 3])
        assert is_symmetric(slam.state.covariance, tolerance=1e-12)
        assert is_positive_semidefinite(slam.state.covariance, tolerance=1e-9)


def test_augmentation_appends_two_entries_per_landmark() -> None:
    slam = build_filter()
    measurement = observe(np.zeros(3), LANDMARKS[0])
    index = slam.augment(measurement)
    assert index == 0
    assert slam.num_landmarks == 1
    assert slam.state.dimension == 5
    np.testing.assert_allclose(slam.state.landmark(0), LANDMARKS[0], atol=1e-12)


def test_augmented_landmark_inherits_the_pose_uncertainty() -> None:
    slam = build_filter()
    slam.predict(Control(1.0, 0.0), 1.0)
    measurement = observe(slam.state.robot_pose, LANDMARKS[0])
    slam.augment(measurement)
    cross = slam.state.covariance[:3, 3:5]
    assert float(np.abs(cross).max()) > 0.0


def test_observing_a_landmark_reduces_its_marginal_uncertainty() -> None:
    slam = build_filter()
    pose = np.zeros(3)
    measurement = observe(pose, LANDMARKS[0])
    slam.augment(measurement)
    initial = float(np.linalg.det(slam.state.landmark_covariance(0)))
    for _ in range(10):
        slam.update(measurement, 0)
    reduced = float(np.linalg.det(slam.state.landmark_covariance(0)))
    assert reduced < initial


def test_update_leaves_the_state_unchanged_for_a_perfect_measurement() -> None:
    slam = build_filter()
    pose = np.zeros(3)
    measurement = observe(pose, LANDMARKS[0])
    slam.augment(measurement)
    before = slam.state.mean.copy()
    slam.update(measurement, 0)
    np.testing.assert_allclose(slam.state.mean, before, atol=1e-9)


def _integrate(pose: np.ndarray, control: Control, dt: float) -> np.ndarray:
    from diffdrive_slam.model.motion import predict_pose

    return predict_pose(pose, control, dt)


def test_filter_converges_to_ground_truth_with_known_correspondences() -> None:
    rng = np.random.default_rng(3)
    sigma_range = 0.05
    sigma_bearing = 0.01
    slam = build_filter(sigma_range, sigma_bearing)

    pose = np.zeros(3)
    controls = [Control(0.8, 0.0)] * 20 + [Control(0.8, 0.4)] * 20
    for step in range(400):
        control = controls[step % len(controls)]
        pose = _integrate(pose, control, 0.1)
        slam.predict(control, 0.1)
        clean = np.array([observe(pose, landmark) for landmark in LANDMARKS])
        noisy = clean + rng.normal(0.0, [sigma_range, sigma_bearing], size=clean.shape)
        slam.integrate(noisy, [0, 1, 2, 3])

    assert slam.num_landmarks == len(LANDMARKS)
    estimated = slam.state.landmarks()
    errors = np.linalg.norm(estimated - LANDMARKS, axis=1)
    assert float(errors.max()) < 0.05

    position_error = float(np.linalg.norm(slam.state.robot_pose[:2] - pose[:2]))
    assert position_error < 0.1
    assert abs(wrap_angle(float(slam.state.robot_pose[2]) - float(pose[2]))) < 0.05


def test_landmark_uncertainty_decreases_monotonically_under_repeated_views() -> None:
    slam = build_filter()
    pose = np.zeros(3)
    measurement = observe(pose, LANDMARKS[1])
    slam.augment(measurement)
    determinants = []
    for _ in range(8):
        slam.update(measurement, 0)
        determinants.append(float(np.linalg.det(slam.state.landmark_covariance(0))))
    assert all(later <= earlier + 1e-15 for earlier, later in pairwise(determinants))


def test_unknown_correspondence_recovers_the_landmark_count() -> None:
    rng = np.random.default_rng(9)
    slam = build_filter()
    pose = np.zeros(3)
    for step in range(200):
        control = Control(0.5, 0.25 if step % 40 < 20 else -0.25)
        pose = _integrate(pose, control, 0.1)
        slam.predict(control, 0.1)
        clean = np.array([observe(pose, landmark) for landmark in LANDMARKS])
        noisy = clean + rng.normal(0.0, [0.05, 0.01], size=clean.shape)
        slam.integrate(noisy)
    assert slam.num_landmarks == len(LANDMARKS)


def test_integrate_reports_new_then_matched() -> None:
    slam = build_filter()
    measurements = np.array([observe(np.zeros(3), LANDMARKS[0])])
    first = slam.integrate(measurements)
    second = slam.integrate(measurements)
    assert first[0].kind is AssociationKind.NEW
    assert second[0].kind is AssociationKind.MATCHED
    assert second[0].landmark_index == 0


def test_measurement_batch_shape_is_validated() -> None:
    slam = build_filter()
    with pytest.raises(ValueError, match="shape"):
        slam.integrate(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="one entry per measurement"):
        slam.integrate(np.zeros((2, 2)), [0])


def test_state_rejects_inconsistent_shapes() -> None:
    with pytest.raises(ValueError, match="3 \\+ 2 N"):
        SlamState(mean=np.zeros(4), covariance=np.eye(4))
    with pytest.raises(ValueError, match="covariance must have shape"):
        SlamState(mean=np.zeros(5), covariance=np.eye(4))


def test_landmark_index_is_bounds_checked() -> None:
    slam = build_filter()
    with pytest.raises(IndexError, match="out of range"):
        slam.state.landmark(0)


def test_configuration_rejects_inverted_gates() -> None:
    with pytest.raises(ValueError, match="new_landmark_confidence"):
        EkfSlamConfig(acceptance_confidence=0.99, new_landmark_confidence=0.5)
    assert EkfSlamConfig().new_landmark_gate > EkfSlamConfig().acceptance_gate
