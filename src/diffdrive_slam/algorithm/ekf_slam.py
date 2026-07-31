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

Reference: Thrun, Burgard, and Fox, Probabilistic Robotics, chapter 10.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

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
    observation_jacobians,
    observe,
)
from diffdrive_slam.model.state import SlamState

__all__ = ["EkfSlam", "EkfSlamConfig", "Innovation"]


@dataclass(frozen=True, slots=True)
class EkfSlamConfig:
    """Tuning of the filter and of its data association policy."""

    motion_noise: MotionNoise = field(default_factory=MotionNoise)
    sensor: RangeBearingParams = field(default_factory=RangeBearingParams)
    #: Chi-square confidence below which a measurement may be matched to a landmark.
    acceptance_confidence: float = 0.99
    #: Chi-square confidence above which a measurement initialises a new landmark.
    new_landmark_confidence: float = 0.9999

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

    @property
    def config(self) -> EkfSlamConfig:
        """The configuration this filter was built with."""
        return self._config

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
        process_noise = process_noise_covariance(
            pose, control, dt, self._config.motion_noise
        )

        self._state.mean[:POSE_DIM] = predict_pose(pose, control, dt)

        covariance = self._state.covariance
        robot_block = covariance[:POSE_DIM, :POSE_DIM]
        covariance[:POSE_DIM, :POSE_DIM] = (
            jacobian @ robot_block @ jacobian.T + process_noise
        )
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
        return Innovation(
            residual=residual, jacobian=jacobian, covariance=symmetrise(covariance)
        )

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
        joseph = (
            factor @ covariance @ factor.T
            + gain @ self._measurement_covariance @ gain.T
        )
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
        return self._state.num_landmarks - 1

    def evaluate_candidates(self, measurement: FloatArray) -> list[Candidate]:
        """Score ``measurement`` against every landmark currently in the state."""
        candidates: list[Candidate] = []
        for index in range(self._state.num_landmarks):
            innovation = self.innovation(measurement, index)
            candidates.append(
                Candidate(
                    landmark_index=index,
                    mahalanobis=mahalanobis_squared(
                        innovation.residual, innovation.covariance
                    ),
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
        """
        if measurements.ndim != 2 or measurements.shape[1] != MEASUREMENT_DIM:
            raise ValueError(f"measurements must have shape (K, 2), got {measurements.shape}")
        if correspondences is not None and len(correspondences) != measurements.shape[0]:
            raise ValueError("correspondences must have one entry per measurement")

        if correspondences is not None:
            return self._integrate_known(measurements, correspondences)
        return self._integrate_unknown(measurements)

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
        acceptance = self._config.acceptance_gate
        initialisation = self._config.new_landmark_gate
        results: list[Association] = []
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
                results.append(decision)
            elif decision.kind is AssociationKind.NEW:
                index = self.augment(measurement)
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
        return tuple(results)
