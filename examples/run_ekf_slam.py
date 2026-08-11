"""Run EKF-SLAM on the default arena and report accuracy and consistency.

Usage::

    uv run python examples/run_ekf_slam.py
    uv run python examples/run_ekf_slam.py --steps 200 --no-figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

from diffdrive_slam.analysis.metrics import consistency_summary, evaluate, pose_nees
from diffdrive_slam.pipeline.simulate import SimulationConfig, run_simulation

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "figures"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=640, help="number of simulation steps")
    parser.add_argument("--seed", type=int, default=20260731, help="noise seed")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="figure directory")
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    config = SimulationConfig(steps=args.steps, seed=args.seed, build_grid=False)
    trace = run_simulation(config)
    result = evaluate(trace)

    nees = pose_nees(trace.true_poses, trace.estimated_poses, trace.pose_covariances)
    summary = consistency_summary(nees)

    print(f"steps                        {len(trace) - 1}")
    print(f"true landmarks               {trace.true_landmarks.shape[0]}")
    print(f"estimated landmarks          {result.landmarks.estimated}")
    print(f"ATE position RMSE [m]        {result.trajectory.position_rmse:.4f}")
    print(f"ATE position max [m]         {result.trajectory.position_max:.4f}")
    print(f"ATE heading RMSE [rad]       {result.trajectory.heading_rmse:.4f}")
    print(f"dead reckoning RMSE [m]      {result.dead_reckoning.position_rmse:.4f}")
    print(f"dead reckoning max [m]       {result.dead_reckoning.position_max:.4f}")
    print(f"landmark position RMSE [m]   {result.landmarks.rmse:.4f}")
    print(f"landmark position max [m]    {result.landmarks.maximum:.4f}")
    print(f"measurements                 {result.associations.measurements}")
    print(f"matched                      {result.associations.matched}")
    print(f"initialised                  {result.associations.initialised}")
    print(f"rejected as ambiguous        {result.associations.rejected}")
    print(f"deleted by map management    {trace.removed_landmarks}")
    print(f"incorrect matches            {result.associations.incorrect}")
    print(f"association accuracy         {result.associations.accuracy:.4f}")
    print(f"time averaged NEES           {summary.average:.4f}")
    print(f"expected value               {summary.degrees_of_freedom}")
    print(
        f"per step bounds (95 percent) [{summary.per_step_lower:.4f}, {summary.per_step_upper:.4f}]"
    )
    print(f"steps inside per step bounds {summary.inside_fraction:.4f}")
    print(f"nominal inside fraction      {summary.confidence:.4f}")
    print(f"pooled bounds (95 percent)   [{summary.lower_bound:.4f}, {summary.upper_bound:.4f}]")
    print(f"verdict on pooled average    {summary.verdict}")
    if result.map_consistency is not None:
        map_summary = result.map_consistency
        print(f"map NEES average             {map_summary.average:.4f}")
        print(f"map expected value           {map_summary.degrees_of_freedom}")
        print(
            f"map per landmark bounds      "
            f"[{map_summary.per_step_lower:.4f}, {map_summary.per_step_upper:.4f}]"
        )
        print(f"landmarks inside map bounds  {map_summary.inside_fraction:.4f}")
        print(f"verdict on the map           {map_summary.verdict}")
    print("note: the pooled bounds treat the time samples as independent, which they")
    print("      are not; run_consistency_study.py gives the Monte Carlo verdict")

    if not args.no_figures:
        from diffdrive_slam.analysis.figures import (
            plot_error_history,
            plot_nees,
            plot_trajectory,
            save_figure,
        )

        save_figure(plot_trajectory(trace), args.output / "ekf_slam_trajectory.png")
        save_figure(plot_error_history(trace), args.output / "ekf_slam_error.png")
        save_figure(plot_nees(trace.times, nees, summary), args.output / "ekf_slam_nees.png")
        print(f"figures written to           {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
