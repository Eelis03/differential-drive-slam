"""Occupancy grid geometry and the log-odds parameterisation.

The grid is an axis-aligned array of square cells stored row major, with rows
indexing the world ``y`` axis and columns indexing the world ``x`` axis. Cell
occupancy is stored as a log odds ratio ``l = log(p / (1 - p))`` so that repeated
observations combine by addition, following the recursive occupancy grid update
of Moravec and Elfes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from diffdrive_slam.model.arrays import FloatArray, IntArray

__all__ = [
    "GridSpec",
    "LogOddsParams",
    "log_odds_to_probability",
    "probability_to_log_odds",
    "raster_line",
]

_PROBABILITY_EPSILON: Final[float] = 1e-12


@dataclass(frozen=True, slots=True)
class GridSpec:
    """Geometry of a rectangular occupancy grid."""

    origin_x: float = -6.0
    origin_y: float = -6.0
    resolution: float = 0.10
    width: int = 120
    height: int = 120

    def __post_init__(self) -> None:
        if self.resolution <= 0.0:
            raise ValueError(f"resolution must be positive, got {self.resolution}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"grid must have positive extent, got {self.width}x{self.height}")

    @property
    def shape(self) -> tuple[int, int]:
        """Array shape of the grid as ``(rows, columns)``."""
        return self.height, self.width

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """World bounds as ``(x_min, x_max, y_min, y_max)``."""
        return (
            self.origin_x,
            self.origin_x + self.width * self.resolution,
            self.origin_y,
            self.origin_y + self.height * self.resolution,
        )

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """Return the ``(column, row)`` containing the world point ``(x, y)``."""
        column = int(np.floor((x - self.origin_x) / self.resolution))
        row = int(np.floor((y - self.origin_y) / self.resolution))
        return column, row

    def cell_to_world(self, column: int, row: int) -> tuple[float, float]:
        """Return the world coordinates of the centre of ``(column, row)``."""
        x = self.origin_x + (column + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return x, y

    def contains(self, column: int, row: int) -> bool:
        """Return whether ``(column, row)`` lies inside the grid."""
        return 0 <= column < self.width and 0 <= row < self.height

    def mask_inside(self, columns: IntArray, rows: IntArray) -> IntArray:
        """Return the indices of ``(columns, rows)`` pairs that lie inside the grid."""
        inside = (
            (columns >= 0) & (columns < self.width) & (rows >= 0) & (rows < self.height)
        )
        return np.asarray(np.flatnonzero(inside), dtype=np.int64)


@dataclass(frozen=True, slots=True)
class LogOddsParams:
    """Increments and saturation bounds of the inverse sensor model.

    ``occupied`` and ``free`` are the log odds contributions of a single beam that
    terminates in a cell or passes through it. ``minimum`` and ``maximum`` clamp the
    accumulated value so that a cell never becomes so certain that later evidence
    cannot revise it, which is the standard remedy for a dynamic environment.
    """

    prior: float = 0.0
    occupied: float = 0.85
    free: float = -0.40
    minimum: float = -4.0
    maximum: float = 4.0

    def __post_init__(self) -> None:
        if self.occupied <= 0.0:
            raise ValueError(f"occupied increment must be positive, got {self.occupied}")
        if self.free >= 0.0:
            raise ValueError(f"free increment must be negative, got {self.free}")
        if self.minimum >= self.maximum:
            raise ValueError("minimum bound must be below maximum bound")
        if not self.minimum <= self.prior <= self.maximum:
            raise ValueError("prior must lie between the clamping bounds")


def log_odds_to_probability(log_odds: FloatArray) -> FloatArray:
    """Convert log odds to occupancy probability elementwise."""
    return np.asarray(1.0 - 1.0 / (1.0 + np.exp(log_odds)), dtype=np.float64)


def probability_to_log_odds(probability: float) -> float:
    """Convert a single occupancy probability to log odds."""
    clipped = min(max(probability, _PROBABILITY_EPSILON), 1.0 - _PROBABILITY_EPSILON)
    return float(np.log(clipped / (1.0 - clipped)))


def raster_line(start: tuple[int, int], end: tuple[int, int]) -> tuple[IntArray, IntArray]:
    """Return the eight-connected cell chain from ``start`` to ``end`` inclusive.

    This is the integer digital differential analyser form of Bresenham's line
    algorithm: sampling ``max(|dx|, |dy|) + 1`` evenly spaced points and rounding
    reproduces the same chain for integer endpoints while remaining vectorised.
    Both arguments and both results are ordered ``(column, row)``.
    """
    column_start, row_start = start
    column_end, row_end = end
    steps = int(max(abs(column_end - column_start), abs(row_end - row_start)))
    if steps == 0:
        return (
            np.array([column_start], dtype=np.int64),
            np.array([row_start], dtype=np.int64),
        )
    columns = np.rint(np.linspace(column_start, column_end, steps + 1)).astype(np.int64)
    rows = np.rint(np.linspace(row_start, row_end, steps + 1)).astype(np.int64)
    return columns, rows
