"""EKF-SLAM with landmark association and occupancy grid mapping.

The package is organised in four library layers plus the example scripts:

``diffdrive_slam.model``
    Dataclasses and pure functions: the differential drive motion model, the range
    and bearing sensor model, the joint state container, and the grid geometry.
``diffdrive_slam.algorithm``
    The EKF-SLAM filter, maximum likelihood data association, and the log-odds
    occupancy grid mapper.
``diffdrive_slam.pipeline``
    The simulated environment and the loop that drives the filter and records a
    trace with ground truth.
``diffdrive_slam.analysis``
    Absolute trajectory error, landmark RMSE, NEES consistency, and figures.

The figure helpers live in :mod:`diffdrive_slam.analysis.figures` and are imported
on demand so that importing this package does not pull in matplotlib.
"""

from __future__ import annotations

from diffdrive_slam.algorithm import (
    Association,
    AssociationKind,
    EkfSlam,
    EkfSlamConfig,
    OccupancyGridMapper,
)
from diffdrive_slam.analysis import Evaluation, evaluate
from diffdrive_slam.model import (
    Control,
    DifferentialDriveParams,
    GridSpec,
    LogOddsParams,
    MotionNoise,
    RangeBearingParams,
    SlamState,
)
from diffdrive_slam.pipeline import Environment, SimulationConfig, Trace, run_simulation

__all__ = [
    "Association",
    "AssociationKind",
    "Control",
    "DifferentialDriveParams",
    "EkfSlam",
    "EkfSlamConfig",
    "Environment",
    "Evaluation",
    "GridSpec",
    "LogOddsParams",
    "MotionNoise",
    "OccupancyGridMapper",
    "RangeBearingParams",
    "SimulationConfig",
    "SlamState",
    "Trace",
    "__version__",
    "evaluate",
    "run_simulation",
]

__version__ = "0.1.0"
