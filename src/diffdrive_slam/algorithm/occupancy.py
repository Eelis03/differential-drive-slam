"""Occupancy grid mapping from range scans with a log-odds inverse sensor model.

Every beam contributes evidence to the cells it crosses. Cells strictly between
the sensor and the returned range receive the free increment because the beam
passed through them. The cell holding the endpoint receives the occupied
increment because the beam stopped there. A beam that reaches the maximum range
carries no endpoint evidence, so its final cell is treated as free as well.

The accumulated log odds is clamped after every beam. Without clamping a cell that
has been observed many times becomes numerically certain and can no longer be
revised, which is the standard failure of the unclamped recursive update.

References: Moravec and Elfes, High resolution maps from wide angle sonar, ICRA
1985; Thrun, Burgard, and Fox, Probabilistic Robotics, chapter 9.
"""

from __future__ import annotations

import numpy as np

from diffdrive_slam.model.arrays import FloatArray, IntArray
from diffdrive_slam.model.grid import (
    GridSpec,
    LogOddsParams,
    log_odds_to_probability,
    raster_line,
)

__all__ = ["OccupancyGridMapper"]


class OccupancyGridMapper:
    """Accumulates a log-odds occupancy grid from range scans."""

    def __init__(self, spec: GridSpec | None = None, params: LogOddsParams | None = None) -> None:
        self._spec = spec if spec is not None else GridSpec()
        self._params = params if params is not None else LogOddsParams()
        self._log_odds = np.full(self._spec.shape, self._params.prior, dtype=np.float64)

    @property
    def spec(self) -> GridSpec:
        """Geometry of the grid."""
        return self._spec

    @property
    def params(self) -> LogOddsParams:
        """Inverse sensor model parameters."""
        return self._params

    @property
    def log_odds(self) -> FloatArray:
        """Read-only view of the accumulated log odds, shaped ``(rows, columns)``."""
        view = self._log_odds.view()
        view.flags.writeable = False
        return view

    def probabilities(self) -> FloatArray:
        """Occupancy probability of every cell, shaped ``(rows, columns)``."""
        return log_odds_to_probability(self._log_odds)

    def snapshot(self) -> FloatArray:
        """Return an independent copy of the accumulated log odds."""
        return self._log_odds.copy()

    def integrate_beam(
        self, pose: FloatArray, beam_range: float, bearing: float, max_range: float
    ) -> None:
        """Fold a single range return into the grid."""
        if max_range <= 0.0:
            raise ValueError(f"max_range must be positive, got {max_range}")

        hit = beam_range < max_range
        travelled = min(beam_range, max_range)
        global_bearing = float(pose[2]) + bearing
        endpoint_x = float(pose[0]) + travelled * float(np.cos(global_bearing))
        endpoint_y = float(pose[1]) + travelled * float(np.sin(global_bearing))

        start = self._spec.world_to_cell(float(pose[0]), float(pose[1]))
        end = self._spec.world_to_cell(endpoint_x, endpoint_y)
        columns, rows = raster_line(start, end)

        if columns.size > 1:
            self._apply(columns[:-1], rows[:-1], self._params.free)
        increment = self._params.occupied if hit else self._params.free
        self._apply(columns[-1:], rows[-1:], increment)

    def integrate_scan(
        self,
        pose: FloatArray,
        ranges: FloatArray,
        bearings: FloatArray,
        max_range: float,
    ) -> None:
        """Fold a full scan into the grid, one beam at a time."""
        if ranges.shape != bearings.shape:
            raise ValueError(
                f"ranges and bearings must match, got {ranges.shape} and {bearings.shape}"
            )
        for beam_range, bearing in zip(ranges, bearings, strict=True):
            self.integrate_beam(pose, float(beam_range), float(bearing), max_range)

    def _apply(self, columns: IntArray, rows: IntArray, increment: float) -> None:
        keep = self._spec.mask_inside(columns, rows)
        if keep.size == 0:
            return
        selected_rows = rows[keep]
        selected_columns = columns[keep]
        updated = self._log_odds[selected_rows, selected_columns] + increment
        self._log_odds[selected_rows, selected_columns] = np.clip(
            updated, self._params.minimum, self._params.maximum
        )
