"""The joint robot and map state carried by the filter.

The state vector stacks the robot pose and every landmark that has been
initialised so far::

    mu = [x, y, theta, m1x, m1y, m2x, m2y, ...]

The covariance is the full dense matrix over that vector. Keeping the robot and
map correlations is what distinguishes SLAM from separate localisation and
mapping, and it is also the reason the memory cost grows with the square of the
landmark count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffdrive_slam.model.arrays import FloatArray, symmetrise
from diffdrive_slam.model.motion import POSE_DIM
from diffdrive_slam.model.sensor import LANDMARK_DIM

__all__ = ["SlamState"]


@dataclass(slots=True)
class SlamState:
    """A Gaussian belief over the robot pose and the landmark map."""

    mean: FloatArray
    covariance: FloatArray

    def __post_init__(self) -> None:
        if self.mean.ndim != 1:
            raise ValueError(f"mean must be one-dimensional, got shape {self.mean.shape}")
        dimension = int(self.mean.size)
        if dimension < POSE_DIM or (dimension - POSE_DIM) % LANDMARK_DIM != 0:
            raise ValueError(f"mean length {dimension} is not 3 + 2 N for an integer N")
        if self.covariance.shape != (dimension, dimension):
            raise ValueError(
                f"covariance must have shape ({dimension}, {dimension}), "
                f"got {self.covariance.shape}"
            )

    @classmethod
    def initial(cls, pose: FloatArray, covariance: FloatArray) -> SlamState:
        """Build a state holding only the robot pose."""
        if pose.shape != (POSE_DIM,):
            raise ValueError(f"pose must have shape (3,), got {pose.shape}")
        return cls(
            mean=np.asarray(pose, dtype=np.float64).copy(),
            covariance=symmetrise(np.asarray(covariance, dtype=np.float64)),
        )

    @property
    def dimension(self) -> int:
        """Length of the state vector."""
        return int(self.mean.size)

    @property
    def num_landmarks(self) -> int:
        """Number of landmarks currently held in the state."""
        return (self.dimension - POSE_DIM) // LANDMARK_DIM

    @property
    def robot_pose(self) -> FloatArray:
        """Copy of the estimated robot pose."""
        return np.asarray(self.mean[:POSE_DIM], dtype=np.float64).copy()

    @property
    def robot_covariance(self) -> FloatArray:
        """Copy of the 3 by 3 marginal covariance of the robot pose."""
        return np.asarray(self.covariance[:POSE_DIM, :POSE_DIM], dtype=np.float64).copy()

    def landmark_slice(self, index: int) -> slice:
        """Return the slice of the state vector occupied by landmark ``index``."""
        if not 0 <= index < self.num_landmarks:
            raise IndexError(f"landmark index {index} out of range for {self.num_landmarks}")
        start = POSE_DIM + LANDMARK_DIM * index
        return slice(start, start + LANDMARK_DIM)

    def landmark(self, index: int) -> FloatArray:
        """Copy of the estimated position of landmark ``index``."""
        return np.asarray(self.mean[self.landmark_slice(index)], dtype=np.float64).copy()

    def landmark_covariance(self, index: int) -> FloatArray:
        """Copy of the 2 by 2 marginal covariance of landmark ``index``."""
        block = self.landmark_slice(index)
        return np.asarray(self.covariance[block, block], dtype=np.float64).copy()

    def landmarks(self) -> FloatArray:
        """Copy of every landmark position, shaped (N, 2)."""
        return np.asarray(
            self.mean[POSE_DIM:].reshape(self.num_landmarks, LANDMARK_DIM), dtype=np.float64
        ).copy()

    def landmark_covariances(self) -> FloatArray:
        """Copy of every landmark marginal covariance, shaped (N, 2, 2)."""
        count = self.num_landmarks
        blocks = np.zeros((count, LANDMARK_DIM, LANDMARK_DIM), dtype=np.float64)
        for index in range(count):
            blocks[index] = self.landmark_covariance(index)
        return blocks

    def without_landmark(self, index: int) -> SlamState:
        """Return the state with landmark ``index`` marginalised out.

        Marginalising a jointly Gaussian variable out of a moment-form belief is
        exactly the deletion of its rows and columns: the remaining block is already
        the covariance of the marginal. No approximation is introduced and no
        information about the surviving landmarks is lost, which is what makes a
        delete operation safe in this parameterisation and expensive in the
        information form, where it would require a Schur complement.
        """
        block = self.landmark_slice(index)
        keep = np.ones(self.dimension, dtype=np.bool_)
        keep[block] = False
        return SlamState(
            mean=np.asarray(self.mean[keep], dtype=np.float64).copy(),
            covariance=np.asarray(self.covariance[np.ix_(keep, keep)], dtype=np.float64).copy(),
        )

    def copy(self) -> SlamState:
        """Return a deep copy of the state."""
        return SlamState(mean=self.mean.copy(), covariance=self.covariance.copy())
