"""Model layer: dataclasses and pure functions with no I/O and no state.

Everything in this package is a total function of its arguments. The motion model,
the sensor model, the joint state container, and the grid geometry live here so
that the filter in :mod:`diffdrive_slam.algorithm` contains only estimation logic.
"""

from __future__ import annotations

from diffdrive_slam.model.arrays import (
    FloatArray,
    IntArray,
    is_positive_semidefinite,
    is_symmetric,
    symmetrise,
    wrap_angle,
    wrap_angles,
)
from diffdrive_slam.model.grid import (
    GridSpec,
    LogOddsParams,
    log_odds_to_probability,
    probability_to_log_odds,
    raster_line,
)
from diffdrive_slam.model.motion import (
    CONTROL_DIM,
    POSE_DIM,
    Control,
    DifferentialDriveParams,
    MotionNoise,
    control_noise_covariance,
    motion_jacobian_control,
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

__all__ = [
    "CONTROL_DIM",
    "LANDMARK_DIM",
    "MEASUREMENT_DIM",
    "POSE_DIM",
    "Control",
    "DifferentialDriveParams",
    "FloatArray",
    "GridSpec",
    "IntArray",
    "LogOddsParams",
    "MotionNoise",
    "RangeBearingParams",
    "SlamState",
    "control_noise_covariance",
    "inverse_observation",
    "inverse_observation_jacobians",
    "is_positive_semidefinite",
    "is_symmetric",
    "is_visible",
    "log_odds_to_probability",
    "motion_jacobian_control",
    "motion_jacobian_state",
    "observation_jacobians",
    "observe",
    "predict_pose",
    "probability_to_log_odds",
    "process_noise_covariance",
    "raster_line",
    "symmetrise",
    "wrap_angle",
    "wrap_angles",
]
