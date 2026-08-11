"""Monte Carlo NEES study over independent noise realisations of the same map.

Averaging the NEES across independent runs at each time step removes the temporal
correlation that makes a single run's time average hard to interpret, and gives a
confidence interval whose width is set by the number of runs.

Usage::

    uv run python examples/run_consistency_study.py
    uv run python examples/run_consistency_study.py --runs 4 --steps 120 --no-figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from diffdrive_slam.analysis.metrics import (
    absolute_trajectory_error,
    consistency_summary,
    landmark_error,
    pose_nees,
)
from diffdrive_slam.pipeline.simulate import SimulationConfig, run_simulation

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "figures"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20, help="number of Monte Carlo runs")
    parser.add_argument("--steps", type=int, default=640, help="steps per run")
    parser.add_argument("--seed", type=int, default=1000, help="first noise seed")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="figure directory")
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)

    nees_rows: list[np.ndarray] = []
    trajectory_rmse: list[float] = []
    landmark_rmse: list[float] = []
    deleted = 0
    surplus = 0
    times = np.zeros(0, dtype=np.float64)

    for index in range(args.runs):
        config = SimulationConfig(steps=args.steps, seed=args.seed + index, build_grid=False)
        trace = run_simulation(config)
        times = trace.times
        nees_rows.append(pose_nees(trace.true_poses, trace.estimated_poses, trace.pose_covariances))
        trajectory_rmse.append(
            absolute_trajectory_error(trace.true_poses, trace.estimated_poses).position_rmse
        )
        landmark_rmse.append(
            landmark_error(
                trace.true_landmarks, trace.estimated_landmarks, trace.slot_to_identity
            ).rmse
        )
        deleted += trace.removed_landmarks
        surplus += trace.final_state.num_landmarks - trace.true_landmarks.shape[0]

    ensemble = np.mean(np.asarray(nees_rows, dtype=np.float64), axis=0)
    summary = consistency_summary(ensemble, samples_per_value=args.runs)

    print(f"runs                         {args.runs}")
    print(f"steps per run                {args.steps}")
    print(f"ATE position RMSE mean [m]   {float(np.mean(trajectory_rmse)):.4f}")
    print(f"ATE position RMSE std [m]    {float(np.std(trajectory_rmse)):.4f}")
    print(f"landmark RMSE mean [m]       {float(np.mean(landmark_rmse)):.4f}")
    print(f"landmark RMSE std [m]        {float(np.std(landmark_rmse)):.4f}")
    print(f"landmarks deleted            {deleted}")
    print(f"surplus landmarks remaining  {surplus}")
    print(f"ensemble average NEES        {summary.average:.4f}")
    print(f"expected value               {summary.degrees_of_freedom}")
    print(
        f"per step bounds (95 percent) [{summary.per_step_lower:.4f}, {summary.per_step_upper:.4f}]"
    )
    print(f"steps inside per step bounds {summary.inside_fraction:.4f}")
    print(f"nominal inside fraction      {summary.confidence:.4f}")
    print(f"pooled bounds (95 percent)   [{summary.lower_bound:.4f}, {summary.upper_bound:.4f}]")
    print(f"verdict on pooled average    {summary.verdict}")
    print("note: the per step test is the primary evidence; the pooled bounds assume")
    print("      independence across time steps, which inflates the effective count")

    if not args.no_figures:
        from diffdrive_slam.analysis.figures import plot_nees, save_figure

        save_figure(plot_nees(times, ensemble, summary), args.output / "consistency_nees.png")
        print(f"figures written to           {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
