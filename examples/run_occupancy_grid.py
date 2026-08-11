"""Build an occupancy grid while running EKF-SLAM and score it against ground truth.

Usage::

    uv run python examples/run_occupancy_grid.py
    uv run python examples/run_occupancy_grid.py --steps 200 --no-figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

from diffdrive_slam.analysis.metrics import evaluate, grid_summary
from diffdrive_slam.pipeline.environment import arena_environment
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
    config = SimulationConfig(steps=args.steps, seed=args.seed, build_grid=True)
    environment = arena_environment(
        seed=config.environment_seed, landmark_count=config.landmark_count
    )
    trace = run_simulation(config, environment)
    if trace.occupancy_log_odds is None or trace.grid is None:
        raise RuntimeError("the simulation did not produce an occupancy grid")

    result = evaluate(trace)
    truth = environment.rasterise_walls(trace.grid)
    grid = grid_summary(trace.occupancy_log_odds, truth)
    params = config.log_odds

    print(f"steps                        {len(trace) - 1}")
    print(
        f"grid                         {trace.grid.width} x {trace.grid.height} cells "
        f"at {trace.grid.resolution:.2f} m"
    )
    print(f"beams per scan               {config.scan_beams}")
    print(f"scan interval [steps]        {config.scan_interval}")
    print(f"log odds bounds              [{params.minimum:.1f}, {params.maximum:.1f}]")
    print(f"ATE position RMSE [m]        {result.trajectory.position_rmse:.4f}")
    print(f"cells                        {grid.cells}")
    print(f"classified occupied          {grid.occupied}")
    print(f"classified free              {grid.free}")
    print(f"unknown (prior retained)     {grid.unknown}")
    print(f"decided fraction             {grid.decided_fraction:.4f}")
    print(f"free agreement               {grid.free_agreement:.4f}")
    print("occupied agreement against the wall tolerance:")
    for tolerance in (0, 1, 2, 3):
        swept = grid_summary(trace.occupancy_log_odds, truth, tolerance_cells=tolerance)
        print(
            f"  tolerance {tolerance} cells ({tolerance * trace.grid.resolution:.2f} m)  "
            f"{swept.occupied_agreement:.4f}"
        )

    if not args.no_figures:
        from diffdrive_slam.analysis.figures import plot_occupancy_grid, save_figure

        save_figure(
            plot_occupancy_grid(trace, environment.walls), args.output / "occupancy_grid.png"
        )
        print(f"figures written to           {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
