"""Accuracy and consistency metrics computed from a trace.

Three families of number are produced.

Absolute trajectory error compares the filtered pose against ground truth in the
same frame. Because the simulator fixes the world frame, no trajectory alignment
step is needed, unlike the Umeyama-aligned ATE used for real datasets.

Landmark position error compares each estimated landmark against the ground truth
landmark it was initialised from, which is recorded by the simulator and never
shown to the filter.

The normalised estimation error squared measures whether the reported covariance
matches the error actually made. For a consistent filter the NEES of a
three-dimensional pose error is chi-square distributed with three degrees of
freedom, so its expectation is three. Averaging ``n`` independent samples gives a
scaled chi-square with ``3 n`` degrees of freedom, which yields the two-sided
confidence interval used here. Values above the interval mean the filter is
optimistic, that is, it reports less uncertainty than it has.

Reference: Bar-Shalom, Li, and Kirubarajan, Estimation with Applications to
Tracking and Navigation, chapter 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.stats import chi2

from diffdrive_slam.algorithm.association import AssociationKind
from diffdrive_slam.model.arrays import BoolArray, FloatArray, wrap_angles
from diffdrive_slam.model.grid import log_odds_to_probability
from diffdrive_slam.model.motion import POSE_DIM
from diffdrive_slam.pipeline.trace import Trace

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


@dataclass(frozen=True, slots=True)
class TrajectoryError:
    """Position and heading error of an estimated trajectory."""

    samples: int
    position_rmse: float
    position_mean: float
    position_max: float
    heading_rmse: float


@dataclass(frozen=True, slots=True)
class LandmarkError:
    """Position error of the estimated map."""

    estimated: int
    matched: int
    rmse: float
    maximum: float


@dataclass(frozen=True, slots=True)
class ConsistencySummary:
    """Average NEES against its two-sided chi-square confidence intervals.

    ``per_step_lower`` and ``per_step_upper`` bound one entry of the NEES sequence and
    are the interval the per-step test uses. ``lower_bound`` and ``upper_bound`` bound
    the average of the whole sequence, which is a valid interval only if the entries
    are independent.
    """

    degrees_of_freedom: int
    samples: int
    average: float
    lower_bound: float
    upper_bound: float
    per_step_lower: float
    per_step_upper: float
    confidence: float
    inside_fraction: float

    @property
    def consistent(self) -> bool:
        """Whether the average NEES lies inside the confidence interval."""
        return bool(self.lower_bound <= self.average <= self.upper_bound)

    @property
    def verdict(self) -> str:
        """One word describing where the average NEES falls."""
        if self.average > self.upper_bound:
            return "optimistic"
        if self.average < self.lower_bound:
            return "conservative"
        return "consistent"


@dataclass(frozen=True, slots=True)
class AssociationSummary:
    """Counts of the decisions taken by the data association policy."""

    measurements: int
    matched: int
    initialised: int
    rejected: int
    incorrect: int

    @property
    def accuracy(self) -> float:
        """Fraction of matched measurements assigned to the correct landmark."""
        if self.matched == 0:
            return float("nan")
        return (self.matched - self.incorrect) / self.matched


@dataclass(frozen=True, slots=True)
class GridSummary:
    """How much of the occupancy grid was decided and how well it agrees with truth."""

    cells: int
    occupied: int
    free: int
    unknown: int
    occupied_agreement: float
    free_agreement: float
    tolerance_cells: int

    @property
    def decided_fraction(self) -> float:
        """Fraction of cells classified as either occupied or free."""
        return (self.occupied + self.free) / self.cells

    @property
    def overall_agreement(self) -> float:
        """Agreement over every decided cell."""
        decided = self.occupied + self.free
        if decided == 0:
            return float("nan")
        correct = self.occupied_agreement * self.occupied + self.free_agreement * self.free
        return correct / decided


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Every metric computed for one trace."""

    trajectory: TrajectoryError
    dead_reckoning: TrajectoryError
    landmarks: LandmarkError
    consistency: ConsistencySummary
    associations: AssociationSummary


def pose_errors(true_poses: FloatArray, estimated_poses: FloatArray) -> FloatArray:
    """Return the pose error ``true - estimated`` with the heading wrapped."""
    if true_poses.shape != estimated_poses.shape:
        raise ValueError(f"shapes must match, got {true_poses.shape} and {estimated_poses.shape}")
    if true_poses.ndim != 2 or true_poses.shape[1] != POSE_DIM:
        raise ValueError(f"poses must have shape (T, 3), got {true_poses.shape}")
    errors = np.asarray(true_poses - estimated_poses, dtype=np.float64).copy()
    errors[:, 2] = wrap_angles(errors[:, 2])
    return errors


def absolute_trajectory_error(
    true_poses: FloatArray, estimated_poses: FloatArray
) -> TrajectoryError:
    """Return the absolute trajectory error of ``estimated_poses``."""
    errors = pose_errors(true_poses, estimated_poses)
    distances = np.linalg.norm(errors[:, :2], axis=1)
    return TrajectoryError(
        samples=int(distances.size),
        position_rmse=float(np.sqrt(np.mean(np.square(distances)))),
        position_mean=float(np.mean(distances)),
        position_max=float(np.max(distances)),
        heading_rmse=float(np.sqrt(np.mean(np.square(errors[:, 2])))),
    )


def landmark_error(
    true_landmarks: FloatArray,
    estimated_landmarks: FloatArray,
    slot_to_identity: tuple[int, ...],
) -> LandmarkError:
    """Return the position error of the estimated map.

    Only landmarks whose ground truth identity was recorded at initialisation are
    scored. A slot with identity ``-1`` is a spurious landmark that the association
    policy created and is counted in ``estimated`` but not in ``matched``.
    """
    estimated = int(estimated_landmarks.shape[0])
    if len(slot_to_identity) != estimated:
        raise ValueError("slot_to_identity must have one entry per estimated landmark")

    squared: list[float] = []
    for slot, identity in enumerate(slot_to_identity):
        if identity < 0:
            continue
        offset = estimated_landmarks[slot] - true_landmarks[identity]
        squared.append(float(offset @ offset))

    if not squared:
        return LandmarkError(
            estimated=estimated, matched=0, rmse=float("nan"), maximum=float("nan")
        )
    values = np.asarray(squared, dtype=np.float64)
    return LandmarkError(
        estimated=estimated,
        matched=int(values.size),
        rmse=float(np.sqrt(values.mean())),
        maximum=float(np.sqrt(values.max())),
    )


def pose_nees(
    true_poses: FloatArray, estimated_poses: FloatArray, covariances: FloatArray
) -> FloatArray:
    """Return the NEES of the pose error at every step, shaped (T,)."""
    errors = pose_errors(true_poses, estimated_poses)
    if covariances.shape != (errors.shape[0], POSE_DIM, POSE_DIM):
        raise ValueError(f"covariances must have shape (T, 3, 3), got {covariances.shape}")
    values = np.empty(errors.shape[0], dtype=np.float64)
    for index, (error, covariance) in enumerate(zip(errors, covariances, strict=True)):
        values[index] = float(error @ np.linalg.solve(covariance, error))
    return values


def nees_bounds(
    degrees_of_freedom: int, samples: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Return the two-sided interval on the average NEES of ``samples`` draws."""
    if degrees_of_freedom <= 0 or samples <= 0:
        raise ValueError("degrees_of_freedom and samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    tail = 0.5 * (1.0 - confidence)
    total = degrees_of_freedom * samples
    lower = float(chi2.ppf(tail, total)) / samples
    upper = float(chi2.ppf(1.0 - tail, total)) / samples
    return lower, upper


def consistency_summary(
    nees: FloatArray,
    degrees_of_freedom: int = POSE_DIM,
    confidence: float = 0.95,
    samples_per_value: int = 1,
) -> ConsistencySummary:
    """Summarise a sequence of NEES values against the chi-square interval.

    ``samples_per_value`` is the number of independent runs that were averaged to
    form each entry of ``nees``. It is one for a single run and equal to the Monte
    Carlo count for an ensemble average.
    """
    if nees.ndim != 1 or nees.size == 0:
        raise ValueError(f"nees must be a non-empty one-dimensional array, got {nees.shape}")
    per_value_lower, per_value_upper = nees_bounds(
        degrees_of_freedom, samples_per_value, confidence
    )
    inside = np.count_nonzero((nees >= per_value_lower) & (nees <= per_value_upper))
    total_samples = samples_per_value * int(nees.size)
    lower, upper = nees_bounds(degrees_of_freedom, total_samples, confidence)
    return ConsistencySummary(
        degrees_of_freedom=degrees_of_freedom,
        samples=total_samples,
        average=float(np.mean(nees)),
        lower_bound=lower,
        upper_bound=upper,
        per_step_lower=per_value_lower,
        per_step_upper=per_value_upper,
        confidence=confidence,
        inside_fraction=float(inside) / float(nees.size),
    )


def association_summary(trace: Trace) -> AssociationSummary:
    """Count the association decisions taken across a trace."""
    matched = 0
    initialised = 0
    rejected = 0
    incorrect = 0
    for identity, assigned, association in trace.associations():
        if association.kind is AssociationKind.MATCHED:
            matched += 1
            if assigned != identity:
                incorrect += 1
        elif association.kind is AssociationKind.NEW:
            initialised += 1
        else:
            rejected += 1
    return AssociationSummary(
        measurements=matched + initialised + rejected,
        matched=matched,
        initialised=initialised,
        rejected=rejected,
        incorrect=incorrect,
    )


def grid_summary(
    log_odds: FloatArray,
    truth_occupied: BoolArray,
    occupied_threshold: float = 0.65,
    free_threshold: float = 0.35,
    tolerance_cells: int = 1,
) -> GridSummary:
    """Score an estimated occupancy grid against a ground truth occupancy mask.

    A cell counts as occupied when its probability exceeds ``occupied_threshold``,
    free when it falls below ``free_threshold``, and unknown in between. Cells that
    were never observed keep the prior and are therefore unknown by construction.

    Walls are infinitely thin in the simulated world while the grid quantises them
    to whole cells, and the map is built at the filtered pose rather than the true
    one. A predicted occupied cell is therefore counted as correct when a true wall
    lies within ``tolerance_cells`` of it. The tolerance is deliberately not applied
    to the free class: a predicted free cell counts as wrong only when it coincides
    with a wall cell, because cells beside a wall genuinely are free and must not be
    penalised.
    """
    if log_odds.shape != truth_occupied.shape:
        raise ValueError(f"shapes must match, got {log_odds.shape} and {truth_occupied.shape}")
    if not 0.0 < free_threshold < occupied_threshold < 1.0:
        raise ValueError("thresholds must satisfy 0 < free < occupied < 1")
    if tolerance_cells < 0:
        raise ValueError(f"tolerance_cells must be non-negative, got {tolerance_cells}")

    probabilities = log_odds_to_probability(log_odds)
    predicted_occupied = probabilities > occupied_threshold
    predicted_free = probabilities < free_threshold

    if tolerance_cells == 0:
        near_wall = truth_occupied
    else:
        near_wall = np.asarray(
            binary_dilation(truth_occupied, iterations=tolerance_cells), dtype=np.bool_
        )

    occupied_count = int(np.count_nonzero(predicted_occupied))
    free_count = int(np.count_nonzero(predicted_free))
    occupied_correct = int(np.count_nonzero(predicted_occupied & near_wall))
    free_correct = int(np.count_nonzero(predicted_free & ~truth_occupied))

    return GridSummary(
        cells=int(log_odds.size),
        occupied=occupied_count,
        free=free_count,
        unknown=int(log_odds.size) - occupied_count - free_count,
        occupied_agreement=(occupied_correct / occupied_count if occupied_count else float("nan")),
        free_agreement=free_correct / free_count if free_count else float("nan"),
        tolerance_cells=tolerance_cells,
    )


def evaluate(trace: Trace, confidence: float = 0.95) -> Evaluation:
    """Compute every metric for ``trace``."""
    true_poses = trace.true_poses
    nees = pose_nees(true_poses, trace.estimated_poses, trace.pose_covariances)
    return Evaluation(
        trajectory=absolute_trajectory_error(true_poses, trace.estimated_poses),
        dead_reckoning=absolute_trajectory_error(true_poses, trace.dead_reckoned_poses),
        landmarks=landmark_error(
            trace.true_landmarks, trace.estimated_landmarks, trace.slot_to_identity
        ),
        consistency=consistency_summary(nees, POSE_DIM, confidence),
        associations=association_summary(trace),
    )
