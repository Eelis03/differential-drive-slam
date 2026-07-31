"""The structured record produced by one simulated run.

A trace holds, for every time step, the commanded control, the true pose, the
filtered pose and its covariance, the dead reckoned pose, and the association
decision taken for every measurement. It also holds the ground truth map and the
slot to identity mapping needed to score the estimated map. The analysis layer
consumes traces and nothing else, so a run and its evaluation are fully separable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffdrive_slam.algorithm.association import Association
from diffdrive_slam.model.arrays import FloatArray, IntArray
from diffdrive_slam.model.grid import GridSpec
from diffdrive_slam.model.motion import Control
from diffdrive_slam.model.state import SlamState

__all__ = ["StepRecord", "Trace"]


@dataclass(frozen=True, slots=True)
class StepRecord:
    """Everything observed and estimated at one time step."""

    time: float
    control: Control
    true_pose: FloatArray
    estimated_pose: FloatArray
    pose_covariance: FloatArray
    dead_reckoned_pose: FloatArray
    num_landmarks: int
    associations: tuple[Association, ...]
    measurement_identities: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Trace:
    """A complete simulated run together with its ground truth."""

    steps: tuple[StepRecord, ...]
    true_landmarks: FloatArray
    final_state: SlamState
    slot_to_identity: tuple[int, ...]
    grid: GridSpec | None = None
    occupancy_log_odds: FloatArray | None = None

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("a trace must contain at least one step")
        if len(self.slot_to_identity) != self.final_state.num_landmarks:
            raise ValueError("slot_to_identity must have one entry per estimated landmark")

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def times(self) -> FloatArray:
        """Time stamp of every step, shaped (T,)."""
        return np.array([step.time for step in self.steps], dtype=np.float64)

    @property
    def true_poses(self) -> FloatArray:
        """Ground truth pose at every step, shaped (T, 3)."""
        return np.array([step.true_pose for step in self.steps], dtype=np.float64)

    @property
    def estimated_poses(self) -> FloatArray:
        """Filtered pose at every step, shaped (T, 3)."""
        return np.array([step.estimated_pose for step in self.steps], dtype=np.float64)

    @property
    def dead_reckoned_poses(self) -> FloatArray:
        """Pose obtained by integrating the commanded controls alone, shaped (T, 3)."""
        return np.array([step.dead_reckoned_pose for step in self.steps], dtype=np.float64)

    @property
    def pose_covariances(self) -> FloatArray:
        """Robot marginal covariance at every step, shaped (T, 3, 3)."""
        return np.array([step.pose_covariance for step in self.steps], dtype=np.float64)

    @property
    def landmark_counts(self) -> IntArray:
        """Number of landmarks in the state at every step, shaped (T,)."""
        return np.array([step.num_landmarks for step in self.steps], dtype=np.int64)

    @property
    def estimated_landmarks(self) -> FloatArray:
        """Final estimated landmark positions, shaped (N, 2)."""
        return self.final_state.landmarks()

    @property
    def estimated_landmark_covariances(self) -> FloatArray:
        """Final landmark marginal covariances, shaped (N, 2, 2)."""
        return self.final_state.landmark_covariances()

    def associations(self) -> tuple[tuple[int, Association], ...]:
        """Return every association paired with the true identity of its measurement."""
        pairs: list[tuple[int, Association]] = []
        for step in self.steps:
            for association in step.associations:
                identity = step.measurement_identities[association.measurement_index]
                pairs.append((identity, association))
        return tuple(pairs)
