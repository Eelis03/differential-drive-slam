"""Maximum likelihood data association with a chi-square gate.

Each measurement is compared against every landmark already in the state. The
squared Mahalanobis distance of the innovation is chi-square distributed with two
degrees of freedom when the association is correct, which gives a principled
acceptance gate. Among the candidates that pass the gate the association with the
highest Gaussian likelihood is selected, which is the maximum likelihood rule of
Probabilistic Robotics table 10.1 with the innovation covariance normaliser
retained rather than dropped.

Three outcomes are possible. A measurement close to exactly one landmark is
matched. A measurement far from every landmark, beyond a second and looser
threshold, initialises a new landmark. A measurement in between is ambiguous and
is discarded, because accepting it risks a false association and initialising it
risks duplicating an existing landmark.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
from scipy.stats import chi2

from diffdrive_slam.model.arrays import FloatArray

__all__ = [
    "Association",
    "AssociationKind",
    "Candidate",
    "associate",
    "chi_square_gate",
    "mahalanobis_squared",
    "negative_log_likelihood",
]

_LOG_TWO_PI: Final[float] = float(np.log(2.0 * np.pi))


class AssociationKind(StrEnum):
    """Outcome of associating one measurement."""

    MATCHED = "matched"
    NEW = "new"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Candidate:
    """One landmark evaluated as an explanation of a measurement."""

    landmark_index: int
    mahalanobis: float
    log_likelihood_cost: float


@dataclass(frozen=True, slots=True)
class Association:
    """The decision taken for one measurement."""

    measurement_index: int
    kind: AssociationKind
    landmark_index: int | None
    mahalanobis: float

    def __post_init__(self) -> None:
        if self.kind is AssociationKind.REJECTED and self.landmark_index is not None:
            raise ValueError("a rejected association must not carry a landmark index")


def mahalanobis_squared(innovation: FloatArray, innovation_covariance: FloatArray) -> float:
    """Return ``nu^T S^-1 nu`` for innovation ``nu`` and covariance ``S``."""
    solution = np.linalg.solve(innovation_covariance, innovation)
    return float(innovation @ solution)


def negative_log_likelihood(innovation: FloatArray, innovation_covariance: FloatArray) -> float:
    """Return twice the negative log likelihood of ``innovation`` under ``S``.

    The value is ``nu^T S^-1 nu + log det(2 pi S)``. The normaliser matters because
    candidate landmarks generally have different innovation covariances, so ranking
    by the Mahalanobis distance alone is not the maximum likelihood rule.
    """
    dimension = int(innovation.size)
    sign, log_determinant = np.linalg.slogdet(innovation_covariance)
    if sign <= 0.0:
        raise ValueError("innovation covariance must be positive definite")
    return mahalanobis_squared(innovation, innovation_covariance) + float(
        log_determinant + dimension * _LOG_TWO_PI
    )


def chi_square_gate(degrees_of_freedom: int, confidence: float) -> float:
    """Return the chi-square quantile used as an acceptance threshold."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    return float(chi2.ppf(confidence, degrees_of_freedom))


def associate(
    measurement_index: int,
    candidates: Sequence[Candidate],
    acceptance_gate: float,
    new_landmark_gate: float,
) -> Association:
    """Decide the fate of one measurement given its evaluated ``candidates``.

    ``acceptance_gate`` is the chi-square threshold below which a candidate may be
    matched. ``new_landmark_gate`` is the larger threshold above which the nearest
    candidate is considered unrelated, so the measurement initialises a landmark.
    """
    if new_landmark_gate < acceptance_gate:
        raise ValueError("new_landmark_gate must not be below acceptance_gate")

    if not candidates:
        return Association(
            measurement_index=measurement_index,
            kind=AssociationKind.NEW,
            landmark_index=None,
            mahalanobis=float("inf"),
        )

    nearest = min(candidates, key=lambda candidate: candidate.mahalanobis)
    accepted = [candidate for candidate in candidates if candidate.mahalanobis <= acceptance_gate]
    if accepted:
        best = min(accepted, key=lambda candidate: candidate.log_likelihood_cost)
        return Association(
            measurement_index=measurement_index,
            kind=AssociationKind.MATCHED,
            landmark_index=best.landmark_index,
            mahalanobis=best.mahalanobis,
        )

    if nearest.mahalanobis > new_landmark_gate:
        return Association(
            measurement_index=measurement_index,
            kind=AssociationKind.NEW,
            landmark_index=None,
            mahalanobis=nearest.mahalanobis,
        )

    return Association(
        measurement_index=measurement_index,
        kind=AssociationKind.REJECTED,
        landmark_index=None,
        mahalanobis=nearest.mahalanobis,
    )
