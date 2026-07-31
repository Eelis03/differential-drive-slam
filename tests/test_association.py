"""Tier 1: the Mahalanobis gate and the maximum likelihood selection rule."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from diffdrive_slam.algorithm.association import (
    AssociationKind,
    Candidate,
    associate,
    chi_square_gate,
    mahalanobis_squared,
    negative_log_likelihood,
)


def test_mahalanobis_of_a_zero_innovation_is_zero() -> None:
    assert mahalanobis_squared(np.zeros(2), np.eye(2)) == pytest.approx(0.0)


def test_mahalanobis_is_scale_invariant_under_the_covariance() -> None:
    innovation = np.array([0.3, -0.4])
    covariance = np.diag([0.09, 0.16])
    assert mahalanobis_squared(innovation, covariance) == pytest.approx(2.0)


def test_mahalanobis_accounts_for_correlation() -> None:
    innovation = np.array([1.0, 1.0])
    independent = mahalanobis_squared(innovation, np.eye(2))
    correlated = mahalanobis_squared(innovation, np.array([[1.0, 0.9], [0.9, 1.0]]))
    assert correlated < independent


def test_negative_log_likelihood_penalises_a_wide_covariance() -> None:
    innovation = np.zeros(2)
    tight = negative_log_likelihood(innovation, np.eye(2) * 0.01)
    wide = negative_log_likelihood(innovation, np.eye(2))
    assert tight < wide


def test_negative_log_likelihood_rejects_a_singular_covariance() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        negative_log_likelihood(np.zeros(2), np.zeros((2, 2)))


def test_chi_square_gate_matches_scipy() -> None:
    assert chi_square_gate(2, 0.99) == pytest.approx(float(chi2.ppf(0.99, 2)))
    assert chi_square_gate(3, 0.95) == pytest.approx(float(chi2.ppf(0.95, 3)))


def test_chi_square_gate_rejects_an_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        chi_square_gate(2, 1.5)


def test_no_candidates_initialises_a_landmark() -> None:
    decision = associate(0, [], acceptance_gate=9.21, new_landmark_gate=18.4)
    assert decision.kind is AssociationKind.NEW
    assert decision.landmark_index is None


def test_a_close_candidate_is_matched() -> None:
    candidates = [Candidate(landmark_index=4, mahalanobis=1.2, log_likelihood_cost=3.0)]
    decision = associate(7, candidates, acceptance_gate=9.21, new_landmark_gate=18.4)
    assert decision.kind is AssociationKind.MATCHED
    assert decision.landmark_index == 4
    assert decision.measurement_index == 7


def test_a_distant_candidate_initialises_a_landmark() -> None:
    candidates = [Candidate(landmark_index=0, mahalanobis=40.0, log_likelihood_cost=44.0)]
    decision = associate(0, candidates, acceptance_gate=9.21, new_landmark_gate=18.4)
    assert decision.kind is AssociationKind.NEW


def test_an_intermediate_candidate_is_rejected_as_ambiguous() -> None:
    candidates = [Candidate(landmark_index=0, mahalanobis=12.0, log_likelihood_cost=16.0)]
    decision = associate(0, candidates, acceptance_gate=9.21, new_landmark_gate=18.4)
    assert decision.kind is AssociationKind.REJECTED
    assert decision.landmark_index is None


def test_selection_uses_the_likelihood_not_the_distance() -> None:
    candidates = [
        Candidate(landmark_index=0, mahalanobis=2.0, log_likelihood_cost=9.0),
        Candidate(landmark_index=1, mahalanobis=3.0, log_likelihood_cost=4.0),
    ]
    decision = associate(0, candidates, acceptance_gate=9.21, new_landmark_gate=18.4)
    assert decision.landmark_index == 1


def test_gates_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="new_landmark_gate"):
        associate(0, [], acceptance_gate=18.4, new_landmark_gate=9.21)


def test_a_rejected_association_cannot_carry_a_landmark() -> None:
    from diffdrive_slam.algorithm.association import Association

    with pytest.raises(ValueError, match="rejected association"):
        Association(
            measurement_index=0,
            kind=AssociationKind.REJECTED,
            landmark_index=2,
            mahalanobis=12.0,
        )
