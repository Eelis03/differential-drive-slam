"""Shared fixtures and finite difference helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from diffdrive_slam.model.arrays import FloatArray
from diffdrive_slam.pipeline.simulate import SimulationConfig, run_simulation
from diffdrive_slam.pipeline.trace import Trace

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"


def central_difference(
    function: Callable[[FloatArray], FloatArray], point: FloatArray, step: float = 1e-5
) -> FloatArray:
    """Return the Jacobian of ``function`` at ``point`` by central differences.

    The result has shape ``(len(function(point)), len(point))``.
    """
    base = function(point)
    jacobian = np.zeros((base.size, point.size), dtype=np.float64)
    for column in range(point.size):
        offset = np.zeros_like(point)
        offset[column] = step
        forward = function(point + offset)
        backward = function(point - offset)
        jacobian[:, column] = (forward - backward) / (2.0 * step)
    return jacobian


@pytest.fixture(scope="session")
def short_trace() -> Trace:
    """A short run with the occupancy grid enabled, shared across tests."""
    return run_simulation(SimulationConfig(steps=120, seed=11, build_grid=True))


@pytest.fixture(scope="session")
def medium_trace() -> Trace:
    """A longer run without the grid, used where convergence matters."""
    return run_simulation(SimulationConfig(steps=320, seed=5, build_grid=False))
