"""Regenerate the recorded reference run used by the regression tier.

Run this only when a change to the algorithm is intended, and review the diff::

    uv run python tests/generate_reference.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diffdrive_slam.analysis.metrics import evaluate, grid_summary
from diffdrive_slam.pipeline.environment import arena_environment
from diffdrive_slam.pipeline.simulate import SimulationConfig, run_simulation

REFERENCE_PATH = Path(__file__).resolve().parent / "data" / "reference_run.json"
REFERENCE_CONFIG = SimulationConfig(steps=150, seed=424242, build_grid=True)


def build_reference() -> dict[str, object]:
    """Run the reference configuration and collect the recorded quantities."""
    environment = arena_environment(
        seed=REFERENCE_CONFIG.environment_seed,
        landmark_count=REFERENCE_CONFIG.landmark_count,
    )
    trace = run_simulation(REFERENCE_CONFIG, environment)
    if trace.occupancy_log_odds is None or trace.grid is None:
        raise RuntimeError("the reference configuration must build an occupancy grid")

    result = evaluate(trace)
    grid = grid_summary(
        trace.occupancy_log_odds, environment.rasterise_walls(trace.grid)
    )
    return {
        "steps": REFERENCE_CONFIG.steps,
        "seed": REFERENCE_CONFIG.seed,
        "final_pose": trace.estimated_poses[-1].tolist(),
        "final_true_pose": trace.true_poses[-1].tolist(),
        "final_pose_covariance": trace.pose_covariances[-1].tolist(),
        "num_landmarks": trace.final_state.num_landmarks,
        "slot_to_identity": list(trace.slot_to_identity),
        "estimated_landmarks": trace.estimated_landmarks.tolist(),
        "ate_position_rmse": result.trajectory.position_rmse,
        "ate_heading_rmse": result.trajectory.heading_rmse,
        "dead_reckoning_rmse": result.dead_reckoning.position_rmse,
        "landmark_rmse": result.landmarks.rmse,
        "average_nees": result.consistency.average,
        "associations": {
            "measurements": result.associations.measurements,
            "matched": result.associations.matched,
            "initialised": result.associations.initialised,
            "rejected": result.associations.rejected,
            "incorrect": result.associations.incorrect,
        },
        "grid": {
            "occupied": grid.occupied,
            "free": grid.free,
            "unknown": grid.unknown,
            "log_odds_sum": float(np.sum(trace.occupancy_log_odds)),
        },
    }


def main() -> int:
    """Write the reference file."""
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_PATH.write_text(
        json.dumps(build_reference(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {REFERENCE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
