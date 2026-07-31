"""The simulated world: point landmarks and wall segments, with ground truth.

The environment is the only source of truth in the pipeline. It answers two
questions: which landmarks a pose can see and at what noiseless range and bearing,
and how far a laser beam travels before it meets a wall. Measurement noise is added
by the simulator, not here, so that the same environment can be replayed under
different noise seeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffdrive_slam.model.arrays import BoolArray, FloatArray, IntArray, wrap_angles
from diffdrive_slam.model.grid import GridSpec
from diffdrive_slam.model.sensor import RangeBearingParams

__all__ = ["Environment", "arena_environment", "rectangle_segments", "sample_landmarks"]


@dataclass(frozen=True, slots=True)
class Environment:
    """Point landmarks and wall segments in a fixed world frame."""

    landmarks: FloatArray
    walls: FloatArray

    def __post_init__(self) -> None:
        if self.landmarks.ndim != 2 or self.landmarks.shape[1] != 2:
            raise ValueError(f"landmarks must have shape (M, 2), got {self.landmarks.shape}")
        if self.walls.ndim != 2 or self.walls.shape[1] != 4:
            raise ValueError(f"walls must have shape (K, 4), got {self.walls.shape}")

    @property
    def num_landmarks(self) -> int:
        """Number of point landmarks in the world."""
        return int(self.landmarks.shape[0])

    def visible_landmarks(
        self, pose: FloatArray, params: RangeBearingParams
    ) -> tuple[IntArray, FloatArray]:
        """Return the identities and noiseless measurements visible from ``pose``.

        The result is a pair of arrays: landmark identities of shape (K,) and range
        and bearing pairs of shape (K, 2), ordered by identity.
        """
        deltas = self.landmarks - np.asarray(pose[:2], dtype=np.float64)
        ranges = np.hypot(deltas[:, 0], deltas[:, 1])
        bearings = wrap_angles(np.arctan2(deltas[:, 1], deltas[:, 0]) - float(pose[2]))
        visible = (ranges <= params.max_range) & (
            np.abs(bearings) <= 0.5 * params.field_of_view
        )
        identities = np.asarray(np.flatnonzero(visible), dtype=np.int64)
        measurements = np.stack([ranges[visible], bearings[visible]], axis=1)
        return identities, np.asarray(measurements, dtype=np.float64)

    def range_scan(self, pose: FloatArray, bearings: FloatArray, max_range: float) -> FloatArray:
        """Return the distance to the nearest wall along each beam of a scan.

        Beams that meet no wall inside ``max_range`` return ``max_range``. The
        computation is a vectorised ray against line segment intersection over every
        beam and every wall at once.
        """
        origin = np.asarray(pose[:2], dtype=np.float64)
        angles = np.asarray(pose[2] + bearings, dtype=np.float64)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        starts = self.walls[:, :2]
        spans = self.walls[:, 2:] - starts
        relative = starts - origin

        denominator = (
            directions[:, None, 0] * spans[None, :, 1]
            - directions[:, None, 1] * spans[None, :, 0]
        )
        parallel = np.abs(denominator) < 1e-12
        safe = np.where(parallel, 1.0, denominator)

        along_ray = (
            relative[None, :, 0] * spans[None, :, 1]
            - relative[None, :, 1] * spans[None, :, 0]
        ) / safe
        along_wall = (
            relative[None, :, 0] * directions[:, None, 1]
            - relative[None, :, 1] * directions[:, None, 0]
        ) / safe

        valid = (~parallel) & (along_ray >= 0.0) & (along_wall >= 0.0) & (along_wall <= 1.0)
        distances = np.where(valid, along_ray, np.inf)
        nearest = distances.min(axis=1)
        return np.asarray(np.minimum(nearest, max_range), dtype=np.float64)

    def rasterise_walls(self, spec: GridSpec, samples_per_cell: int = 4) -> BoolArray:
        """Return the ground truth occupied mask of ``spec``, shaped ``(rows, columns)``.

        Each wall segment is sampled at ``samples_per_cell`` points per cell width and
        the containing cells are marked. This is the reference the estimated grid is
        scored against.
        """
        if samples_per_cell <= 0:
            raise ValueError(f"samples_per_cell must be positive, got {samples_per_cell}")
        occupied = np.zeros(spec.shape, dtype=np.bool_)
        for segment in self.walls:
            start = segment[:2]
            end = segment[2:]
            length = float(np.linalg.norm(end - start))
            count = max(int(np.ceil(length / spec.resolution)) * samples_per_cell, 2)
            fractions = np.linspace(0.0, 1.0, count)
            points = start[None, :] + fractions[:, None] * (end - start)[None, :]
            columns = np.floor((points[:, 0] - spec.origin_x) / spec.resolution).astype(np.int64)
            rows = np.floor((points[:, 1] - spec.origin_y) / spec.resolution).astype(np.int64)
            keep = spec.mask_inside(columns, rows)
            occupied[rows[keep], columns[keep]] = True
        return occupied


def rectangle_segments(
    x_min: float, y_min: float, x_max: float, y_max: float
) -> FloatArray:
    """Return the four wall segments of an axis-aligned rectangle."""
    return np.array(
        [
            [x_min, y_min, x_max, y_min],
            [x_max, y_min, x_max, y_max],
            [x_max, y_max, x_min, y_max],
            [x_min, y_max, x_min, y_min],
        ],
        dtype=np.float64,
    )


def sample_landmarks(
    rng: np.random.Generator,
    count: int,
    half_extent: float,
    exclusion_half_extent: float,
    minimum_separation: float,
    max_attempts: int = 10_000,
) -> FloatArray:
    """Draw ``count`` landmarks by rejection sampling inside a square annulus.

    Landmarks are rejected inside the central obstacle and within
    ``minimum_separation`` of an already accepted landmark. The separation keeps the
    data association problem well posed: landmarks packed closer than the sensor can
    resolve produce ambiguous measurements by construction.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    accepted: list[FloatArray] = []
    for _ in range(max_attempts):
        if len(accepted) == count:
            break
        point = rng.uniform(-half_extent, half_extent, size=2)
        if float(np.max(np.abs(point))) < exclusion_half_extent:
            continue
        if accepted:
            distances = np.linalg.norm(np.asarray(accepted) - point, axis=1)
            if float(distances.min()) < minimum_separation:
                continue
        accepted.append(np.asarray(point, dtype=np.float64))
    if len(accepted) != count:
        raise RuntimeError(
            f"only placed {len(accepted)} of {count} landmarks within {max_attempts} attempts"
        )
    return np.asarray(accepted, dtype=np.float64)


def arena_environment(
    seed: int = 7,
    landmark_count: int = 20,
    landmark_half_extent: float = 5.0,
    wall_half_extent: float = 5.5,
    obstacle_half_extent: float = 1.2,
    minimum_separation: float = 1.0,
) -> Environment:
    """Build the default arena: a square room with a central block obstacle.

    The robot drives a closed loop between the obstacle and the outer wall, so both
    surfaces are scanned from several directions and every landmark is revisited on
    the second lap.
    """
    rng = np.random.default_rng(seed)
    landmarks = sample_landmarks(
        rng=rng,
        count=landmark_count,
        half_extent=landmark_half_extent,
        exclusion_half_extent=obstacle_half_extent + 0.4,
        minimum_separation=minimum_separation,
    )
    walls = np.concatenate(
        [
            rectangle_segments(
                -wall_half_extent, -wall_half_extent, wall_half_extent, wall_half_extent
            ),
            rectangle_segments(
                -obstacle_half_extent,
                -obstacle_half_extent,
                obstacle_half_extent,
                obstacle_half_extent,
            ),
        ],
        axis=0,
    )
    return Environment(landmarks=landmarks, walls=np.asarray(walls, dtype=np.float64))
