"""Tier 2: a recorded reference run compared against the current implementation.

The reference is regenerated with ``uv run python tests/generate_reference.py``.
Numeric tolerances are loose enough to absorb platform differences in the last
bits of the floating point results, and tight enough that any change in the
estimator, the association policy, or the mapper shows up here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from diffdrive_slam.analysis.metrics import evaluate, grid_summary
from diffdrive_slam.pipeline.environment import arena_environment
from diffdrive_slam.pipeline.simulate import run_simulation
from tests.conftest import DATA_DIR
from tests.generate_reference import REFERENCE_CONFIG

TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def reference() -> dict[str, object]:
    """Load the recorded reference run."""
    return json.loads((DATA_DIR / "reference_run.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def replayed() -> dict[str, object]:
    """Re-run the reference configuration."""
    environment = arena_environment(
        seed=REFERENCE_CONFIG.environment_seed,
        landmark_count=REFERENCE_CONFIG.landmark_count,
    )
    trace = run_simulation(REFERENCE_CONFIG, environment)
    if trace.occupancy_log_odds is None or trace.grid is None:
        raise RuntimeError("the reference configuration must build an occupancy grid")
    return {
        "trace": trace,
        "evaluation": evaluate(trace),
        "grid": grid_summary(trace.occupancy_log_odds, environment.rasterise_walls(trace.grid)),
    }


def test_reference_configuration_is_unchanged(reference: dict[str, object]) -> None:
    assert reference["steps"] == REFERENCE_CONFIG.steps
    assert reference["seed"] == REFERENCE_CONFIG.seed


def test_final_pose_matches_the_reference(
    reference: dict[str, object], replayed: dict[str, object]
) -> None:
    trace = replayed["trace"]
    np.testing.assert_allclose(
        trace.estimated_poses[-1], np.asarray(reference["final_pose"]), atol=TOLERANCE
    )
    np.testing.assert_allclose(
        trace.true_poses[-1], np.asarray(reference["final_true_pose"]), atol=TOLERANCE
    )


def test_final_covariance_matches_the_reference(
    reference: dict[str, object], replayed: dict[str, object]
) -> None:
    trace = replayed["trace"]
    np.testing.assert_allclose(
        trace.pose_covariances[-1],
        np.asarray(reference["final_pose_covariance"]),
        atol=TOLERANCE,
    )


def test_map_matches_the_reference(
    reference: dict[str, object], replayed: dict[str, object]
) -> None:
    trace = replayed["trace"]
    assert trace.final_state.num_landmarks == reference["num_landmarks"]
    assert list(trace.slot_to_identity) == list(reference["slot_to_identity"])
    np.testing.assert_allclose(
        trace.estimated_landmarks,
        np.asarray(reference["estimated_landmarks"]),
        atol=TOLERANCE,
    )


def test_metrics_match_the_reference(
    reference: dict[str, object], replayed: dict[str, object]
) -> None:
    result = replayed["evaluation"]
    assert result.trajectory.position_rmse == pytest.approx(
        reference["ate_position_rmse"], abs=TOLERANCE
    )
    assert result.trajectory.heading_rmse == pytest.approx(
        reference["ate_heading_rmse"], abs=TOLERANCE
    )
    assert result.dead_reckoning.position_rmse == pytest.approx(
        reference["dead_reckoning_rmse"], abs=TOLERANCE
    )
    assert result.landmarks.rmse == pytest.approx(reference["landmark_rmse"], abs=TOLERANCE)
    assert result.consistency.average == pytest.approx(reference["average_nees"], abs=TOLERANCE)


def test_association_counts_match_the_reference(
    reference: dict[str, object], replayed: dict[str, object]
) -> None:
    expected = reference["associations"]
    summary = replayed["evaluation"].associations
    assert summary.measurements == expected["measurements"]
    assert summary.matched == expected["matched"]
    assert summary.initialised == expected["initialised"]
    assert summary.rejected == expected["rejected"]
    assert summary.incorrect == expected["incorrect"]


def test_occupancy_grid_matches_the_reference(
    reference: dict[str, object], replayed: dict[str, object]
) -> None:
    expected = reference["grid"]
    grid = replayed["grid"]
    trace = replayed["trace"]
    assert grid.occupied == expected["occupied"]
    assert grid.free == expected["free"]
    assert grid.unknown == expected["unknown"]
    assert float(np.sum(trace.occupancy_log_odds)) == pytest.approx(
        expected["log_odds_sum"], abs=1e-6
    )
