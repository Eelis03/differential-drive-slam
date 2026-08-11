"""Extended Kalman filter SLAM over a range and bearing landmark map.

The filter maintains a single Gaussian over the joint robot and map state. Three
operations act on that state.

Prediction propagates the robot block through the differential drive motion model
and mixes the added process noise into the robot to map cross covariances. Only
the robot rows and columns change, which is why the block form below is used in
place of the identity-padded Jacobian of the textbook derivation. The two are
algebraically identical.

Correction applies one range and bearing measurement at a time. The Joseph form
is used for the covariance so that the result stays symmetric and positive
semidefinite even when the Kalman gain is computed inexactly.

Augmentation appends a landmark on its first observation. The new block is the
inverse sensor model evaluated at the current pose estimate, and its covariance
inherits the pose uncertainty through the inverse model Jacobians, which is what
keeps the new landmark correlated with the robot and with the rest of the map.

A fourth operation, removal, deletes a landmark that the association policy created
in error. It is the marginalisation of a jointly Gaussian variable, which in moment
form is the deletion of two rows and two columns, so the belief left over the
survivors is exact. See :class:`MapManagement` for the rule that decides when it
fires.

Reference: Thrun, Burgard, and Fox, Probabilistic Robotics, chapter 10.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from diffdrive_slam.algorithm.association import (
    Association,
    AssociationKind,
    Candidate,
    associate,
    chi_square_gate,
    mahalanobis_squared,
    negative_log_likelihood,
)
from diffdrive_slam.model.arrays import FloatArray, symmetrise, wrap_angle
from diffdrive_slam.model.motion import (
    POSE_DIM,
    Control,
    MotionNoise,
    motion_jacobian_state,
    predict_pose,
    process_noise_covariance,
)
from diffdrive_slam.model.sensor import (
    LANDMARK_DIM,
    MEASUREMENT_DIM,
    RangeBearingParams,
    inverse_observation,
    inverse_observation_jacobians,
    is_visible,
    observation_jacobians,
    observe,
)
from diffdrive_slam.model.state import SlamState

__all__ = ["EkfSlam", "EkfSlamConfig", "Innovation", "MapManagement"]


@dataclass(frozen=True, slots=True)
class MapManagement:
    """Rule that deletes landmarks the association policy created in error.

    A landmark is provisional until it has been matched ``confirm_after`` times.
    While it is provisional, every batch in which the filter predicts it to lie
    inside the sensor footprint and no measurement is assigned to it counts as a
    miss, and exceeding ``misses_allowed`` misses deletes it. A confirmed landmark
    is never deleted, because a landmark seen that often is supported by evidence
    that a run of missed detections should not overturn.

    The asymmetry is deliberate. A spurious landmark is created by a single outlying
    measurement and is then contradicted at every subsequent step, because the real
    landmark that produced the outlier explains the following measurements better.
    A real landmark is contradicted only when its measurements are being lost, which
    the two thresholds together make unlikely to happen ``misses_allowed`` times in a
    row before ``confirm_after`` matches have accumulated.

    ``range_margin`` scales the sensor maximum range down before the visibility test.
    A landmark sitting on the range boundary is expected to be detected
    intermittently, so counting those steps as misses would delete correct landmarks
    at the edge of the footprint.
    """

    enabled: bool = True
    confirm_after: int = 5
    misses_allowed: int = 3
    range_margin: float = 0.9

    def __post_init__(self) -> None:
        if self.confirm_after < 1:
            raise ValueError(f"confirm_after must be at least 1, got {self.confirm_after}")
        if self.misses_allowed < 1:
            raise ValueError(f"misses_allowed must be at least 1, got {self.misses_allowed}")
        if not 0.0 < self.range_margin <= 1.0:
            raise ValueError(f"range_margin must lie in (0, 1], got {self.range_margin}")


@dataclass(frozen=True, slots=True)
class EkfSlamConfig:
    """Tuning of the filter and of its data association policy."""

    motion_noise: MotionNoise = field(default_factory=MotionNoise)
    sensor: RangeBearingParams = field(default_factory=RangeBearingParams)
    #: Chi-square confidence below which a measurement may be matched to a landmark.
    acceptance_confidence: float = 0.99
    #: Chi-square confidence above which a measurement initialises a new landmark.
    new_landmark_confidence: float = 0.9999
    #: Rule that removes unsupported landmarks. Applied only when correspondences
    #: are recovered from the measurements, since the known correspondence mode
    #: bypasses association and therefore has nothing to correct.
    map_management: MapManagement = field(default_factory=MapManagement)

    def __post_init__(self) -> None:
        if not 0.0 < self.acceptance_confidence < 1.0:
            raise ValueError("acceptance_confidence must lie in (0, 1)")
        if not self.acceptance_confidence <= self.new_landmark_confidence < 1.0:
            raise ValueError("new_landmark_confidence must lie in [acceptance_confidence, 1)")

    @property
    def acceptance_gate(self) -> float:
        """Squared Mahalanobis threshold for accepting a match."""
        return chi_square_gate(MEASUREMENT_DIM, self.acceptance_confidence)

    @property
    def new_landmark_gate(self) -> float:
        """Squared Mahalanobis threshold above which a landmark is initialised."""
        return chi_square_gate(MEASUREMENT_DIM, self.new_landmark_confidence)


@dataclass(frozen=True, slots=True)
class Innovation:
    """The quantities that a single landmark correction depends on."""

    residual: FloatArray
    jacobian: FloatArray
    covariance: FloatArray


class EkfSlam:
    """An EKF-SLAM filter over a growing range and bearing landmark map."""

    def __init__(
        self,
        initial_pose: FloatArray,
        initial_covariance: FloatArray,
        config: EkfSlamConfig | None = None,
    ) -> None:
        self._config = config if config is not None else EkfSlamConfig()
        self._state = SlamState.initial(initial_pose, initial_covariance)
        self._measurement_covariance = self._config.sensor.covariance()
        self._identity_to_index: dict[int, int] = {}
        self._hits: list[int] = []
        self._misses: list[int] = []
        self._removals: tuple[int, ...] = ()
        self._footprint = replace(
            self._config.sensor,
            max_range=self._config.map_management.range_margin * self._config.sensor.max_range,
        )

    @property
    def config(self) -> EkfSlamConfig:
        """The configuration this filter was built with."""
        return self._config

    @property
    def last_removals(self) -> tuple[int, ...]:
        """Landmark slots deleted during the most recent :meth:`integrate` call.

        The indices are ascending and refer to the numbering in force before the
        deletion, so a caller holding its own per-slot bookkeeping can replay them
        to renumber it. Deleting a slot shifts every higher slot down by one.
        """
        return self._removals

    def observation_counts(self) -> tuple[tuple[int, int], ...]:
        """Return ``(hits, misses)`` per landmark slot, in slot order."""
        return tuple(zip(self._hits, self._misses, strict=True))

    @property
    def state(self) -> SlamState:
        """The live joint state. Mutating it mutates the filter."""
        return self._state

    @property
    def num_landmarks(self) -> int:
        """Number of landmarks currently in the state."""
        return self._state.num_landmarks

    def predict(self, control: Control, dt: float) -> None:
        """Propagate the belief through the motion model for one interval."""
        pose = self._state.robot_pose
        jacobian = motion_jacobian_state(pose, control, dt)
        process_noise = process_noise_covariance(pose, control, dt, self._config.motion_noise)

        self._state.mean[:POSE_DIM] = predict_pose(pose, control, dt)

        covariance = self._state.covariance
        robot_block = covariance[:POSE_DIM, :POSE_DIM]
        covariance[:POSE_DIM, :POSE_DIM] = jacobian @ robot_block @ jacobian.T + process_noise
        if self._state.num_landmarks:
            cross = jacobian @ covariance[:POSE_DIM, POSE_DIM:]
            covariance[:POSE_DIM, POSE_DIM:] = cross
            covariance[POSE_DIM:, :POSE_DIM] = cross.T
        self._state.covariance = symmetrise(covariance)

    def innovation(self, measurement: FloatArray, landmark_index: int) -> Innovation:
        """Return the residual, measurement Jacobian, and innovation covariance."""
        state = self._state
        pose = state.robot_pose
        landmark = state.landmark(landmark_index)

        residual = measurement - observe(pose, landmark)
        residual[1] = wrap_angle(float(residual[1]))

        pose_jacobian, landmark_jacobian = observation_jacobians(pose, landmark)
        jacobian = np.zeros((MEASUREMENT_DIM, state.dimension), dtype=np.float64)
        jacobian[:, :POSE_DIM] = pose_jacobian
        jacobian[:, state.landmark_slice(landmark_index)] = landmark_jacobian

        covariance = jacobian @ state.covariance @ jacobian.T + self._measurement_covariance
        return Innovation(residual=residual, jacobian=jacobian, covariance=symmetrise(covariance))

    def update(self, measurement: FloatArray, landmark_index: int) -> Innovation:
        """Correct the belief with one measurement of a known landmark."""
        innovation = self.innovation(measurement, landmark_index)
        state = self._state
        covariance = state.covariance
        jacobian = innovation.jacobian

        # gain = P H^T S^-1, obtained from a solve rather than an explicit inverse.
        gain = np.linalg.solve(innovation.covariance, jacobian @ covariance).T

        state.mean += gain @ innovation.residual
        state.mean[2] = wrap_angle(float(state.mean[2]))

        identity = np.eye(state.dimension, dtype=np.float64)
        factor = identity - gain @ jacobian
        joseph = factor @ covariance @ factor.T + gain @ self._measurement_covariance @ gain.T
        state.covariance = symmetrise(joseph)
        return innovation

    def augment(self, measurement: FloatArray) -> int:
        """Append a landmark initialised from ``measurement`` and return its index."""
        state = self._state
        pose = state.robot_pose
        dimension = state.dimension

        position = inverse_observation(pose, measurement)
        pose_jacobian, measurement_jacobian = inverse_observation_jacobians(pose, measurement)

        mean = np.concatenate([state.mean, position])
        covariance = np.zeros(
            (dimension + LANDMARK_DIM, dimension + LANDMARK_DIM), dtype=np.float64
        )
        covariance[:dimension, :dimension] = state.covariance

        cross = pose_jacobian @ state.covariance[:POSE_DIM, :]
        covariance[dimension:, :dimension] = cross
        covariance[:dimension, dimension:] = cross.T
        covariance[dimension:, dimension:] = (
            pose_jacobian @ state.robot_covariance @ pose_jacobian.T
            + measurement_jacobian @ self._measurement_covariance @ measurement_jacobian.T
        )

        self._state = SlamState(mean=mean, covariance=symmetrise(covariance))
        self._hits.append(0)
        self._misses.append(0)
        return self._state.num_landmarks - 1

    def remove_landmark(self, index: int) -> None:
        """Marginalise landmark ``index`` out of the belief.

        Every slot above ``index`` moves down by one, and any identity mapping the
        caller keeps must be renumbered to match.
        """
        self._state = self._state.without_landmark(index)
        del self._hits[index]
        del self._misses[index]
        self._identity_to_index = {
            identity: slot - 1 if slot > index else slot
            for identity, slot in self._identity_to_index.items()
            if slot != index
        }

    def evaluate_candidates(self, measurement: FloatArray) -> list[Candidate]:
        """Score ``measurement`` against every landmark currently in the state."""
        candidates: list[Candidate] = []
        for index in range(self._state.num_landmarks):
            innovation = self.innovation(measurement, index)
            candidates.append(
                Candidate(
                    landmark_index=index,
                    mahalanobis=mahalanobis_squared(innovation.residual, innovation.covariance),
                    log_likelihood_cost=negative_log_likelihood(
                        innovation.residual, innovation.covariance
                    ),
                )
            )
        return candidates

    def integrate(
        self,
        measurements: FloatArray,
        correspondences: Sequence[int] | None = None,
    ) -> tuple[Association, ...]:
        """Fold a batch of measurements into the belief.

        ``measurements`` has shape (K, 2) holding range and bearing pairs. When
        ``correspondences`` is given it supplies the true landmark identity of every
        measurement and data association is bypassed, which is the reference mode
        used to separate filter error from association error.

        The landmark indices carried by the returned associations refer to the state
        as it stands when the call returns. Map management runs before the batch is
        processed, so any deletion is reported by :attr:`last_removals` and the
        indices are already expressed in the renumbered state.
        """
        if measurements.ndim != 2 or measurements.shape[1] != MEASUREMENT_DIM:
            raise ValueError(f"measurements must have shape (K, 2), got {measurements.shape}")
        if correspondences is not None and len(correspondences) != measurements.shape[0]:
            raise ValueError("correspondences must have one entry per measurement")

        self._removals = ()
        if correspondences is not None:
            return self._integrate_known(measurements, correspondences)
        return self._integrate_unknown(measurements)

    def _prune(self) -> tuple[int, ...]:
        """Delete every provisional landmark that has run out of misses."""
        policy = self._config.map_management
        if not policy.enabled:
            return ()
        doomed = tuple(
            index
            for index in range(self._state.num_landmarks)
            if self._hits[index] < policy.confirm_after
            and self._misses[index] > policy.misses_allowed
        )
        for index in reversed(doomed):
            self.remove_landmark(index)
        return doomed

    def _record_observations(self, observed: set[int]) -> None:
        """Credit a hit to every landmark seen and a miss to every one expected.

        Expectation is evaluated at the corrected pose, which is the best estimate of
        where the robot was when the batch was taken. The deletion it may trigger is
        applied at the start of the next batch rather than here, so that the indices
        the caller has just been handed stay valid.
        """
        if not self._config.map_management.enabled:
            return
        pose = self._state.robot_pose
        for index in range(self._state.num_landmarks):
            if index in observed:
                self._hits[index] += 1
            elif is_visible(pose, self._state.landmark(index), self._footprint):
                self._misses[index] += 1

    def _integrate_known(
        self, measurements: FloatArray, correspondences: Sequence[int]
    ) -> tuple[Association, ...]:
        results: list[Association] = []
        for row, identity in enumerate(correspondences):
            measurement = np.asarray(measurements[row], dtype=np.float64)
            known = self._identity_to_index.get(int(identity))
            if known is None:
                index = self.augment(measurement)
                self._identity_to_index[int(identity)] = index
                kind = AssociationKind.NEW
                distance = float("inf")
            else:
                index = known
                innovation = self.update(measurement, index)
                kind = AssociationKind.MATCHED
                distance = mahalanobis_squared(innovation.residual, innovation.covariance)
            results.append(
                Association(
                    measurement_index=row,
                    kind=kind,
                    landmark_index=index,
                    mahalanobis=distance,
                )
            )
        return tuple(results)

    def _integrate_unknown(self, measurements: FloatArray) -> tuple[Association, ...]:
        self._removals = self._prune()
        acceptance = self._config.acceptance_gate
        initialisation = self._config.new_landmark_gate
        results: list[Association] = []
        observed: set[int] = set()
        for row in range(measurements.shape[0]):
            measurement = np.asarray(measurements[row], dtype=np.float64)
            decision = associate(
                measurement_index=row,
                candidates=self.evaluate_candidates(measurement),
                acceptance_gate=acceptance,
                new_landmark_gate=initialisation,
            )
            if decision.kind is AssociationKind.MATCHED and decision.landmark_index is not None:
                self.update(measurement, decision.landmark_index)
                observed.add(decision.landmark_index)
                results.append(decision)
            elif decision.kind is AssociationKind.NEW:
                index = self.augment(measurement)
                observed.add(index)
                results.append(
                    Association(
                        measurement_index=row,
                        kind=AssociationKind.NEW,
                        landmark_index=index,
                        mahalanobis=decision.mahalanobis,
                    )
                )
            else:
                results.append(decision)
        self._record_observations(observed)
        return tuple(results)
