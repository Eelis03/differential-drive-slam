"""Regenerate the three figures embedded in the README.

The README carries three figures and no more, because three is the number that
still earns its place: the trajectory against ground truth, the ensemble NEES
against its confidence band, and the occupancy grid against the true walls. Each
shows something the tables of numbers cannot.

They share a byte budget, so this script writes them at a lower resolution than the
example scripts use for their own working output. Resolution is the only lever
pulled: no image is post-processed and no compression dependency is involved.

Usage::

    uv run python examples/make_readme_figures.py
    uv run python examples/make_readme_figures.py --runs 4 --steps 120 --output /tmp/f
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from diffdrive_slam.analysis.figures import (
    plot_nees,
    plot_occupancy_grid,
    plot_trajectory,
    save_figure,
)
from diffdrive_slam.analysis.metrics import consistency_summary, pose_nees
from diffdrive_slam.pipeline.environment import arena_environment
from diffdrive_slam.pipeline.simulate import SimulationConfig, run_simulation

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
#: Resolution of the published figures. Chosen so that the three together stay
#: inside the 250 KB budget the portfolio applies to tracked images, with enough
#: headroom that a matplotlib version bump cannot push them over it.
FIGURE_DPI = 90


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=640, help="steps per run")
    parser.add_argument("--seed", type=int, default=20260731, help="seed of the single run")
    parser.add_argument("--runs", type=int, default=20, help="Monte Carlo runs for the NEES")
    parser.add_argument(
        "--study-seed", type=int, default=1000, help="first seed of the Monte Carlo study"
    )
    parser.add_argument("--dpi", type=int, default=FIGURE_DPI, help="output resolution")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="figure directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    environment = arena_environment()

    mapped = run_simulation(
        SimulationConfig(steps=args.steps, seed=args.seed, build_grid=True), environment
    )
    save_figure(plot_trajectory(mapped), args.output / "trajectory.png", dpi=args.dpi)
    save_figure(
        plot_occupancy_grid(mapped, environment.walls),
        args.output / "occupancy_grid.png",
        dpi=args.dpi,
    )

    rows: list[np.ndarray] = []
    times = np.zeros(0, dtype=np.float64)
    for index in range(args.runs):
        trace = run_simulation(
            SimulationConfig(steps=args.steps, seed=args.study_seed + index, build_grid=False),
            environment,
        )
        times = trace.times
        rows.append(pose_nees(trace.true_poses, trace.estimated_poses, trace.pose_covariances))
    ensemble = np.mean(np.asarray(rows, dtype=np.float64), axis=0)
    summary = consistency_summary(ensemble, samples_per_value=args.runs)
    save_figure(plot_nees(times, ensemble, summary), args.output / "consistency.png", dpi=args.dpi)

    total = sum(path.stat().st_size for path in sorted(args.output.glob("*.png")))
    for path in sorted(args.output.glob("*.png")):
        print(f"{path.name:<24} {path.stat().st_size:>8} bytes")
    print(f"{'total':<24} {total:>8} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
