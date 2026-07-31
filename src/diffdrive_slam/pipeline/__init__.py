"""Pipeline layer: the simulator that exercises the filter and records a trace.

This is the only package that samples random numbers. It owns the ground truth,
generates noisy controls and measurements from it, drives the estimator, and
returns a :class:`~diffdrive_slam.pipeline.trace.Trace` for the analysis layer.
"""

from __future__ import annotations

from diffdrive_slam.pipeline.environment import (
    Environment,
    arena_environment,
    rectangle_segments,
    sample_landmarks,
)
from diffdrive_slam.pipeline.simulate import SimulationConfig, run_simulation
from diffdrive_slam.pipeline.trace import StepRecord, Trace
from diffdrive_slam.pipeline.trajectory import (
    figure_eight_controls,
    repeat_to_length,
    square_loop_controls,
    square_loop_start,
)

__all__ = [
    "Environment",
    "SimulationConfig",
    "StepRecord",
    "Trace",
    "arena_environment",
    "figure_eight_controls",
    "rectangle_segments",
    "repeat_to_length",
    "run_simulation",
    "sample_landmarks",
    "square_loop_controls",
    "square_loop_start",
]
