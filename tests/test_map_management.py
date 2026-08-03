"""Tier 1: deletion of landmarks the association policy created in error.

Two things have to hold for a delete operation to be safe. The belief left behind
must be the exact marginal over the surviving state, and the rule that fires it must
not remove landmarks that are real. The first is checked against a hand-built
Gaussian, the second against runs in which a duplicate is known to appear.
"""

from __future__ import annotations

import numpy as np
import pytest

from diffdrive_slam.algorithm.ekf_slam import EkfSlam, EkfSlamConfig, MapManagement
from diffdrive_slam.analysis.metrics import (
    absolute_trajectory_error,
    association_summary,
    landmark_error,
)
from diffdrive_slam.model.arrays import is_positive_semidefinite, is_symmetric
from diffdrive_slam.model.motion import Control, MotionNoise
from diffdrive_slam.model.sensor import RangeBearingParams, observe
from diffdrive_slam.model.state import SlamState
from diffdrive_slam.pipeline.simulate import SimulationConfig, run_simulation
from diffdrive_slam.pipeline.trace import Trace

LANDMARKS = np.array([[2.0, 1.0], [-1.5, 2.5], [0.5, -2.0]])
#: A seed on which the maximum likelihood policy is known to create a duplicate.
DUPLICATE_SEED = 2000


@pytest.fixture(scope="module")
def managed() -> Trace:
    """A full run on the duplicate seed, with map management enabled."""
    return run_simulation(SimulationConfig(steps=640, seed=DUPLICATE_SEED, build_grid=False))


@pytest.fixture(scope="module")
def unmanaged() -> Trace:
    """The same run with map management switched off."""
    return run_simulation(
        SimulationConfig(
            steps=640,
            seed=DUPLICATE_SEED,
            build_grid=False,
            map_management=MapManagement(enabled=False),
        )
    )


def build_filter(policy: MapManagement | None = None) -> EkfSlam:
    """Return a filter at the origin whose sensor reaches every test landmark."""
    config = EkfSlamConfig(
        motion_noise=MotionNoise(),
        sensor=RangeBearingParams(sigma_range=0.05, sigma_bearing=0.01, max_range=10.0),
        map_management=policy if policy is not None else MapManagement(),
    )
    return EkfSlam(np.zeros(3), np.diag([1e-4, 1e-4, 1e-5]), config)


def random_state(landmarks: int, seed: int = 0) -> SlamState:
    """Return a valid joint Gaussian with a dense, definite covariance."""
    rng = np.random.default_rng(seed)
    dimension = 3 + 2 * landmarks
    factor = rng.normal(size=(dimension, dimension))
    covariance = factor @ factor.T + dimension * np.eye(dimension)
    return SlamState(mean=rng.normal(size=dimension), covariance=covariance)


def test_marginalising_a_landmark_keeps_the_surviving_block_exactly() -> None:
    state = random_state(landmarks=3, seed=1)
    reduced = state.without_landmark(1)

    keep = [0, 1, 2, 3, 4, 7, 8]
    assert reduced.num_landmarks == 2
    np.testing.assert_array_equal(reduced.mean, state.mean[keep])
    np.testing.assert_array_equal(reduced.covariance, state.covariance[np.ix_(keep, keep)])


def test_marginalising_preserves_the_cross_correlations_of_the_survivors() -> None:
    state = random_state(landmarks=3, seed=2)
    reduced = state.without_landmark(0)
    np.testing.assert_array_equal(
        reduced.covariance[:3, 3:5], state.covariance[:3, 5:7]
    )
    np.testing.assert_array_equal(reduced.landmark(1), state.landmark(2))


def test_marginalising_leaves_a_valid_gaussian() -> None:
    state = random_state(landmarks=4, seed=3)
    reduced = state.without_landmark(2)
    assert is_symmetric(reduced.covariance, tolerance=1e-12)
    assert is_positive_semidefinite(reduced.covariance, tolerance=1e-9)


def test_marginalising_does_not_mutate_the_original() -> None:
    state = random_state(landmarks=2, seed=4)
    before = state.covariance.copy()
    reduced = state.without_landmark(0)
    reduced.covariance[0, 0] = 1e6
    np.testing.assert_array_equal(state.covariance, before)


def test_removal_index_is_bounds_checked() -> None:
    with pytest.raises(IndexError, match="out of range"):
        random_state(landmarks=1).without_landmark(1)


def test_map_management_validates_its_parameters() -> None:
    with pytest.raises(ValueError, match="confirm_after"):
        MapManagement(confirm_after=0)
    with pytest.raises(ValueError, match="misses_allowed"):
        MapManagement(misses_allowed=0)
    with pytest.raises(ValueError, match="range_margin"):
        MapManagement(range_margin=0.0)
    with pytest.raises(ValueError, match="range_margin"):
        MapManagement(range_margin=1.5)


def _observe_all(slam: EkfSlam, pose: np.ndarray) -> None:
    slam.integrate(np.array([observe(pose, landmark) for landmark in LANDMARKS]))


def test_a_provisional_landmark_is_deleted_once_its_misses_run_out() -> None:
    slam = build_filter(MapManagement(confirm_after=5, misses_allowed=2))
    pose = np.zeros(3)
    _observe_all(slam, pose)
    assert slam.num_landmarks == 3

    ghost = slam.augment(np.array([3.0, 2.4]))
    assert slam.num_landmarks == 4

    removals: list[tuple[int, ...]] = []
    for _ in range(4):
        _observe_all(slam, pose)
        removals.append(slam.last_removals)

    assert (ghost,) in removals
    assert slam.num_landmarks == 3
    np.testing.assert_allclose(slam.state.landmarks(), LANDMARKS, atol=1e-6)


def test_deletion_renumbers_the_slots_above_it() -> None:
    slam = build_filter(MapManagement(confirm_after=5, misses_allowed=1))
    pose = np.zeros(3)
    slam.integrate(np.array([observe(pose, LANDMARKS[0])]))
    ghost = slam.augment(np.array([3.0, 2.4]))
    slam.integrate(np.array([observe(pose, landmark) for landmark in LANDMARKS[:2]]))

    for _ in range(3):
        slam.integrate(np.array([observe(pose, landmark) for landmark in LANDMARKS[:2]]))

    assert ghost == 1
    assert slam.num_landmarks == 2
    np.testing.assert_allclose(slam.state.landmarks(), LANDMARKS[:2], atol=1e-6)


def test_a_confirmed_landmark_survives_a_long_run_of_missed_detections() -> None:
    slam = build_filter(MapManagement(confirm_after=3, misses_allowed=1))
    pose = np.zeros(3)
    for _ in range(4):
        _observe_all(slam, pose)
    assert slam.num_landmarks == 3
    assert all(hits >= 3 for hits, _ in slam.observation_counts())

    for _ in range(20):
        slam.integrate(np.zeros((0, 2)))
    assert slam.num_landmarks == 3


def test_a_landmark_outside_the_footprint_is_not_charged_a_miss() -> None:
    """The margin protects a landmark sitting on the edge of the sensor range."""
    config = EkfSlamConfig(
        sensor=RangeBearingParams(sigma_range=0.05, sigma_bearing=0.01, max_range=1.0),
        map_management=MapManagement(confirm_after=5, misses_allowed=1, range_margin=0.9),
    )
    slam = EkfSlam(np.zeros(3), np.diag([1e-4, 1e-4, 1e-5]), config)
    slam.augment(np.array([0.95, 0.0]))
    for _ in range(10):
        slam.integrate(np.zeros((0, 2)))
    assert slam.num_landmarks == 1
    assert slam.observation_counts() == ((0, 0),)


def test_a_landmark_well_inside_the_footprint_is_charged_a_miss() -> None:
    config = EkfSlamConfig(
        sensor=RangeBearingParams(sigma_range=0.05, sigma_bearing=0.01, max_range=1.0),
        map_management=MapManagement(confirm_after=5, misses_allowed=1, range_margin=0.9),
    )
    slam = EkfSlam(np.zeros(3), np.diag([1e-4, 1e-4, 1e-5]), config)
    slam.augment(np.array([0.5, 0.0]))
    slam.integrate(np.zeros((0, 2)))
    assert slam.observation_counts() == ((0, 1),)
    slam.integrate(np.zeros((0, 2)))
    slam.integrate(np.zeros((0, 2)))
    assert slam.num_landmarks == 0


def test_disabling_map_management_keeps_every_landmark() -> None:
    slam = build_filter(MapManagement(enabled=False))
    pose = np.zeros(3)
    _observe_all(slam, pose)
    slam.augment(np.array([3.0, 2.4]))
    for _ in range(10):
        _observe_all(slam, pose)
    assert slam.num_landmarks == 4
    assert slam.last_removals == ()


def test_the_known_correspondence_mode_never_deletes() -> None:
    slam = build_filter(MapManagement(confirm_after=5, misses_allowed=1))
    pose = np.zeros(3)
    slam.integrate(np.array([observe(pose, landmark) for landmark in LANDMARKS]), [0, 1, 2])
    for _ in range(10):
        slam.integrate(np.array([observe(pose, LANDMARKS[0])]), [0])
        assert slam.last_removals == ()
    assert slam.num_landmarks == 3


def test_map_management_removes_the_duplicate_the_policy_creates(
    managed: Trace, unmanaged: Trace
) -> None:
    truth = unmanaged.true_landmarks.shape[0]
    assert unmanaged.final_state.num_landmarks == truth + 1
    assert len(set(unmanaged.slot_to_identity)) == truth
    assert managed.final_state.num_landmarks == truth
    assert managed.removed_landmarks == 1


def test_the_recorded_identities_are_renumbered_with_the_slots(managed: Trace) -> None:
    assert managed.removed_landmarks > 0
    assert sorted(managed.slot_to_identity) == list(range(managed.true_landmarks.shape[0]))
    for step in managed.steps:
        assert len(step.slot_identities) == step.num_landmarks
        assert all(identity >= 0 for identity in step.slot_identities)


def test_deletion_does_not_corrupt_the_association_record(managed: Trace) -> None:
    summary = association_summary(managed)
    assert managed.removed_landmarks > 0
    assert summary.incorrect == 0
    assert summary.accuracy == pytest.approx(1.0)


def test_the_state_stays_a_valid_gaussian_across_a_run_with_deletions(
    managed: Trace,
) -> None:
    covariance = managed.final_state.covariance
    assert is_symmetric(covariance, tolerance=1e-12)
    assert is_positive_semidefinite(covariance, tolerance=1e-9)


def test_deletion_does_not_degrade_the_trajectory_estimate(
    managed: Trace, unmanaged: Trace
) -> None:
    with_deletion = absolute_trajectory_error(
        managed.true_poses, managed.estimated_poses
    ).position_rmse
    without = absolute_trajectory_error(
        unmanaged.true_poses, unmanaged.estimated_poses
    ).position_rmse
    assert with_deletion <= without + 0.01


def test_deletion_improves_the_map(managed: Trace, unmanaged: Trace) -> None:
    """The duplicate is scored as a spurious slot, so removing it sharpens the map."""
    with_deletion = landmark_error(
        managed.true_landmarks, managed.estimated_landmarks, managed.slot_to_identity
    )
    without = landmark_error(
        unmanaged.true_landmarks, unmanaged.estimated_landmarks, unmanaged.slot_to_identity
    )
    assert with_deletion.estimated < without.estimated
    assert with_deletion.rmse <= without.rmse


def test_a_run_with_a_prediction_between_batches_still_deletes() -> None:
    """Deletion must survive the covariance growth that prediction introduces."""
    slam = build_filter(MapManagement(confirm_after=5, misses_allowed=2))
    pose = np.zeros(3)
    _observe_all(slam, pose)
    slam.augment(np.array([3.0, 2.4]))
    for _ in range(6):
        slam.predict(Control(0.0, 0.0), 0.1)
        _observe_all(slam, pose)
    assert slam.num_landmarks == 3
