"""Range and bearing landmark sensor model with analytic Jacobians.

A measurement of landmark ``m = [mx, my]`` from pose ``x = [x, y, theta]`` is
``z = [sqrt((mx - x)^2 + (my - y)^2), atan2(my - y, mx - x) - theta]`` with the
bearing wrapped into [-pi, pi). The inverse model and its Jacobians are used to
augment the filter state when a landmark is observed for the first time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from diffdrive_slam.model.arrays import FloatArray, wrap_angle

__all__ = [
    "LANDMARK_DIM",
    "MEASUREMENT_DIM",
    "RangeBearingParams",
    "inverse_observation",
    "inverse_observation_jacobians",
    "is_visible",
    "observation_jacobians",
    "observe",
]

MEASUREMENT_DIM: Final[int] = 2
LANDMARK_DIM: Final[int] = 2

_MINIMUM_RANGE: Final[float] = 1e-9


@dataclass(frozen=True, slots=True)
class RangeBearingParams:
    """Noise and visibility parameters of the landmark detector."""

    sigma_range: float = 0.15
    sigma_bearing: float = 0.026
    max_range: float = 4.0
    field_of_view: float = 2.0 * float(np.pi)

    def __post_init__(self) -> None:
        if self.sigma_range <= 0.0 or self.sigma_bearing <= 0.0:
            raise ValueError("measurement standard deviations must be positive")
        if self.max_range <= 0.0:
            raise ValueError(f"max_range must be positive, got {self.max_range}")
        if not 0.0 < self.field_of_view <= 2.0 * np.pi:
            raise ValueError(f"field_of_view must lie in (0, 2 pi], got {self.field_of_view}")

    def covariance(self) -> FloatArray:
        """Return the 2 by 2 measurement covariance ``diag(sigma_r^2, sigma_b^2)``."""
        return np.diag([self.sigma_range**2, self.sigma_bearing**2]).astype(np.float64)


def observe(pose: FloatArray, landmark: FloatArray) -> FloatArray:
    """Return the noiseless range and bearing of ``landmark`` seen from ``pose``."""
    delta_x = float(landmark[0]) - float(pose[0])
    delta_y = float(landmark[1]) - float(pose[1])
    range_to_landmark = float(np.hypot(delta_x, delta_y))
    bearing = wrap_angle(float(np.arctan2(delta_y, delta_x)) - float(pose[2]))
    return np.array([range_to_landmark, bearing], dtype=np.float64)


def observation_jacobians(pose: FloatArray, landmark: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return the Jacobians of :func:`observe` with respect to pose and landmark.

    The first element has shape (2, 3) and the second has shape (2, 2). The model is
    singular when the landmark coincides with the sensor origin, so the squared range
    is floored at a small positive value.
    """
    delta_x = float(landmark[0]) - float(pose[0])
    delta_y = float(landmark[1]) - float(pose[1])
    squared_range = max(delta_x * delta_x + delta_y * delta_y, _MINIMUM_RANGE)
    range_to_landmark = float(np.sqrt(squared_range))

    pose_jacobian = np.array(
        [
            [-delta_x / range_to_landmark, -delta_y / range_to_landmark, 0.0],
            [delta_y / squared_range, -delta_x / squared_range, -1.0],
        ],
        dtype=np.float64,
    )
    landmark_jacobian = np.array(
        [
            [delta_x / range_to_landmark, delta_y / range_to_landmark],
            [-delta_y / squared_range, delta_x / squared_range],
        ],
        dtype=np.float64,
    )
    return pose_jacobian, landmark_jacobian


def is_visible(pose: FloatArray, landmark: FloatArray, params: RangeBearingParams) -> bool:
    """Return whether ``landmark`` falls inside the range and angular limits."""
    measurement = observe(pose, landmark)
    within_range = float(measurement[0]) <= params.max_range
    within_view = abs(float(measurement[1])) <= 0.5 * params.field_of_view
    return bool(within_range and within_view)


def inverse_observation(pose: FloatArray, measurement: FloatArray) -> FloatArray:
    """Return the world position implied by ``measurement`` taken from ``pose``."""
    range_to_landmark = float(measurement[0])
    global_bearing = float(pose[2]) + float(measurement[1])
    return np.array(
        [
            float(pose[0]) + range_to_landmark * np.cos(global_bearing),
            float(pose[1]) + range_to_landmark * np.sin(global_bearing),
        ],
        dtype=np.float64,
    )


def inverse_observation_jacobians(
    pose: FloatArray, measurement: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return the Jacobians of :func:`inverse_observation`.

    The first element has shape (2, 3) and differentiates with respect to the pose.
    The second has shape (2, 2) and differentiates with respect to the measurement.
    """
    range_to_landmark = float(measurement[0])
    global_bearing = float(pose[2]) + float(measurement[1])
    cos_bearing = float(np.cos(global_bearing))
    sin_bearing = float(np.sin(global_bearing))

    pose_jacobian = np.array(
        [
            [1.0, 0.0, -range_to_landmark * sin_bearing],
            [0.0, 1.0, range_to_landmark * cos_bearing],
        ],
        dtype=np.float64,
    )
    measurement_jacobian = np.array(
        [
            [cos_bearing, -range_to_landmark * sin_bearing],
            [sin_bearing, range_to_landmark * cos_bearing],
        ],
        dtype=np.float64,
    )
    return pose_jacobian, measurement_jacobian
