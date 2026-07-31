"""Analysis layer: turns a trace into metrics and figures.

Nothing here runs the simulator or the filter. Every function takes a trace, or
arrays extracted from one, and returns numbers or a matplotlib figure.
"""

from __future__ import annotations

from diffdrive_slam.analysis.metrics import (
    AssociationSummary,
    ConsistencySummary,
    Evaluation,
    GridSummary,
    LandmarkError,
    TrajectoryError,
    absolute_trajectory_error,
    association_summary,
    consistency_summary,
    evaluate,
    grid_summary,
    landmark_error,
    nees_bounds,
    pose_errors,
    pose_nees,
)

__all__ = [
    "AssociationSummary",
    "ConsistencySummary",
    "Evaluation",
    "GridSummary",
    "LandmarkError",
    "TrajectoryError",
    "absolute_trajectory_error",
    "association_summary",
    "consistency_summary",
    "evaluate",
    "grid_summary",
    "landmark_error",
    "nees_bounds",
    "pose_errors",
    "pose_nees",
]
