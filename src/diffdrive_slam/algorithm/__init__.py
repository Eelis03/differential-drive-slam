"""Algorithm layer: estimation only, no simulation and no plotting.

This package holds the EKF-SLAM filter, the maximum likelihood data association
policy it uses, and the log-odds occupancy grid mapper. Nothing here reads a file,
draws a figure, or samples a random number.
"""

from __future__ import annotations

from diffdrive_slam.algorithm.association import (
    Association,
    AssociationKind,
    Candidate,
    associate,
    chi_square_gate,
    mahalanobis_squared,
    negative_log_likelihood,
)
from diffdrive_slam.algorithm.ekf_slam import EkfSlam, EkfSlamConfig, Innovation
from diffdrive_slam.algorithm.occupancy import OccupancyGridMapper

__all__ = [
    "Association",
    "AssociationKind",
    "Candidate",
    "EkfSlam",
    "EkfSlamConfig",
    "Innovation",
    "OccupancyGridMapper",
    "associate",
    "chi_square_gate",
    "mahalanobis_squared",
    "negative_log_likelihood",
]
