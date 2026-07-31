"""Differential drive kinematic motion model, noise model, and analytic Jacobians.

The pose is ``[x, y, theta]`` in a fixed world frame and the control is the body
twist ``[v, omega]`` held constant across one interval of length ``dt``. The
integration is the exact arc solution of the unicycle equations, which reduces to
a straight line when ``omega`` is zero and to a pure rotation when ``v`` is zero.
The noise model is the velocity motion model of Thrun, Burgard, and Fox, in which
the control covariance grows with the square of the commanded velocities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from diffdrive_slam.model.arrays import FloatArray, wrap_angle

__all__ = [
    "CONTROL_DIM",
    "POSE_DIM",
    "STRAIGHT_LINE_TOLERANCE",
    "Control",
    "DifferentialDriveParams",
    "MotionNoise",
    "control_noise_covariance",
    "motion_jacobian_control",
    "motion_jacobian_state",
    "predict_pose",
    "process_noise_covariance",
]

POSE_DIM: Final[int] = 3
CONTROL_DIM: Final[int] = 2

#: Below this angular rate the arc solution is replaced by its straight-line limit,
#: which avoids the removable singularity at ``omega == 0`` in ``v / omega``.
STRAIGHT_LINE_TOLERANCE: Final[float] = 1e-9


@dataclass(frozen=True, slots=True)
class Control:
    """A body twist held constant over one integration interval."""

    linear_velocity: float
    angular_velocity: float

    def as_array(self) -> FloatArray:
        """Return the control as a length-2 array ``[v, omega]``."""
        return np.array([self.linear_velocity, self.angular_velocity], dtype=np.float64)

    @classmethod
    def from_array(cls, values: FloatArray) -> Control:
        """Build a control from a length-2 array ``[v, omega]``."""
        if values.shape != (CONTROL_DIM,):
            raise ValueError(f"control array must have shape (2,), got {values.shape}")
        return cls(linear_velocity=float(values[0]), angular_velocity=float(values[1]))


@dataclass(frozen=True, slots=True)
class DifferentialDriveParams:
    """Geometry of a two-wheeled differential drive base."""

    wheel_radius: float
    wheel_base: float

    def __post_init__(self) -> None:
        if self.wheel_radius <= 0.0:
            raise ValueError(f"wheel_radius must be positive, got {self.wheel_radius}")
        if self.wheel_base <= 0.0:
            raise ValueError(f"wheel_base must be positive, got {self.wheel_base}")

    def body_velocity(self, left_rate: float, right_rate: float) -> Control:
        """Convert wheel angular rates in rad/s into the equivalent body twist."""
        linear = 0.5 * self.wheel_radius * (right_rate + left_rate)
        angular = self.wheel_radius * (right_rate - left_rate) / self.wheel_base
        return Control(linear_velocity=linear, angular_velocity=angular)

    def wheel_rates(self, control: Control) -> tuple[float, float]:
        """Convert a body twist into the left and right wheel angular rates in rad/s."""
        half_track = 0.5 * self.wheel_base * control.angular_velocity
        left = (control.linear_velocity - half_track) / self.wheel_radius
        right = (control.linear_velocity + half_track) / self.wheel_radius
        return left, right


@dataclass(frozen=True, slots=True)
class MotionNoise:
    """Velocity-dependent control noise coefficients.

    The control covariance is ``diag(a1 v^2 + a2 w^2, a3 v^2 + a4 w^2)``. The
    coefficients are dimensionless. ``floor`` adds an isotropic term to the pose
    process noise so that the covariance stays strictly positive definite when the
    commanded velocities are zero.
    """

    alpha_1: float = 0.010
    alpha_2: float = 0.002
    alpha_3: float = 0.002
    alpha_4: float = 0.010
    floor: float = 1e-9

    def __post_init__(self) -> None:
        values = (self.alpha_1, self.alpha_2, self.alpha_3, self.alpha_4, self.floor)
        if any(value < 0.0 for value in values):
            raise ValueError("motion noise coefficients must be non-negative")


def predict_pose(pose: FloatArray, control: Control, dt: float) -> FloatArray:
    """Integrate ``pose`` forward by ``dt`` under a constant body twist.

    The result is exact for a constant twist: a straight line when the angular rate
    vanishes, a pure rotation in place when the linear velocity vanishes, and an arc
    of radius ``v / omega`` otherwise.
    """
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    linear = control.linear_velocity
    angular = control.angular_velocity

    if abs(angular) < STRAIGHT_LINE_TOLERANCE:
        return np.array(
            [
                x + linear * dt * np.cos(theta),
                y + linear * dt * np.sin(theta),
                wrap_angle(theta),
            ],
            dtype=np.float64,
        )

    radius = linear / angular
    theta_next = theta + angular * dt
    return np.array(
        [
            x + radius * (np.sin(theta_next) - np.sin(theta)),
            y + radius * (np.cos(theta) - np.cos(theta_next)),
            wrap_angle(theta_next),
        ],
        dtype=np.float64,
    )


def motion_jacobian_state(pose: FloatArray, control: Control, dt: float) -> FloatArray:
    """Return the 3 by 3 Jacobian of :func:`predict_pose` with respect to the pose."""
    theta = float(pose[2])
    linear = control.linear_velocity
    angular = control.angular_velocity

    jacobian = np.eye(POSE_DIM, dtype=np.float64)
    if abs(angular) < STRAIGHT_LINE_TOLERANCE:
        jacobian[0, 2] = -linear * dt * np.sin(theta)
        jacobian[1, 2] = linear * dt * np.cos(theta)
        return jacobian

    radius = linear / angular
    theta_next = theta + angular * dt
    jacobian[0, 2] = radius * (np.cos(theta_next) - np.cos(theta))
    jacobian[1, 2] = radius * (np.sin(theta_next) - np.sin(theta))
    return jacobian


def motion_jacobian_control(pose: FloatArray, control: Control, dt: float) -> FloatArray:
    """Return the 3 by 2 Jacobian of :func:`predict_pose` with respect to the control.

    The straight-line branch uses the second-order limit of the arc solution as the
    angular rate tends to zero, so the two branches agree to first order in ``omega``.
    """
    theta = float(pose[2])
    linear = control.linear_velocity
    angular = control.angular_velocity
    jacobian = np.zeros((POSE_DIM, CONTROL_DIM), dtype=np.float64)
    jacobian[2, 1] = dt

    sin_theta = float(np.sin(theta))
    cos_theta = float(np.cos(theta))

    if abs(angular) < STRAIGHT_LINE_TOLERANCE:
        jacobian[0, 0] = dt * cos_theta
        jacobian[1, 0] = dt * sin_theta
        jacobian[0, 1] = -0.5 * linear * dt * dt * sin_theta
        jacobian[1, 1] = 0.5 * linear * dt * dt * cos_theta
        return jacobian

    theta_next = theta + angular * dt
    sin_next = float(np.sin(theta_next))
    cos_next = float(np.cos(theta_next))

    jacobian[0, 0] = (sin_next - sin_theta) / angular
    jacobian[1, 0] = (cos_theta - cos_next) / angular
    jacobian[0, 1] = (
        linear * (sin_theta - sin_next) / angular**2 + linear * dt * cos_next / angular
    )
    jacobian[1, 1] = (
        linear * (cos_next - cos_theta) / angular**2 + linear * dt * sin_next / angular
    )
    return jacobian


def control_noise_covariance(control: Control, noise: MotionNoise) -> FloatArray:
    """Return the 2 by 2 control covariance for ``control`` under ``noise``."""
    linear_squared = control.linear_velocity**2
    angular_squared = control.angular_velocity**2
    return np.diag(
        [
            noise.alpha_1 * linear_squared + noise.alpha_2 * angular_squared,
            noise.alpha_3 * linear_squared + noise.alpha_4 * angular_squared,
        ]
    ).astype(np.float64)


def process_noise_covariance(
    pose: FloatArray, control: Control, dt: float, noise: MotionNoise
) -> FloatArray:
    """Map the control covariance into pose space as ``V M V^T`` plus a floor term."""
    control_jacobian = motion_jacobian_control(pose, control, dt)
    control_covariance = control_noise_covariance(control, noise)
    mapped = control_jacobian @ control_covariance @ control_jacobian.T
    return np.asarray(mapped + noise.floor * np.eye(POSE_DIM), dtype=np.float64)
