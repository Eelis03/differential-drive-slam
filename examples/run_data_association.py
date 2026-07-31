"""Compare maximum likelihood data association against known correspondences.

The known correspondence run is the reference: it isolates the error made by the
filter itself. The difference between the two runs is the price of solving the
association problem from the measurements alone.

Usage::

    uv run python examples/run_data_association.py
    uv run python examples/run_data_association.py --steps 200 --seeds 2
"""

from __future__ import annotations

import argparse

import numpy as np

from diffdrive_slam.analysis.metrics import Evaluation, evaluate
from diffdrive_slam.pipeline.simulate import SimulationConfig, run_simulation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=640, help="steps per run")
    parser.add_argument("--seeds", type=int, default=5, help="number of noise seeds")
    parser.add_argument("--seed", type=int, default=2000, help="first noise seed")
    return parser.parse_args(argv)


def _run(steps: int, seed: int, known: bool) -> Evaluation:
    config = SimulationConfig(
        steps=steps, seed=seed, build_grid=False, known_correspondence=known
    )
    return evaluate(run_simulation(config))


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)

    rows: dict[str, list[Evaluation]] = {"known": [], "maximum likelihood": []}
    for index in range(args.seeds):
        seed = args.seed + index
        rows["known"].append(_run(args.steps, seed, known=True))
        rows["maximum likelihood"].append(_run(args.steps, seed, known=False))

    print(f"steps per run                {args.steps}")
    print(f"seeds                        {args.seeds}")
    header = f"{'association':<20}{'ATE RMSE [m]':>14}{'landmark RMSE [m]':>20}"
    header += f"{'landmarks':>12}{'incorrect':>11}"
    print(header)
    for label, results in rows.items():
        trajectory = float(np.mean([item.trajectory.position_rmse for item in results]))
        landmarks = float(np.mean([item.landmarks.rmse for item in results]))
        count = float(np.mean([item.landmarks.estimated for item in results]))
        incorrect = int(sum(item.associations.incorrect for item in results))
        print(
            f"{label:<20}{trajectory:>14.4f}{landmarks:>20.4f}{count:>12.1f}{incorrect:>11d}"
        )

    rejected = int(
        sum(item.associations.rejected for item in rows["maximum likelihood"])
    )
    measurements = int(
        sum(item.associations.measurements for item in rows["maximum likelihood"])
    )
    print(f"measurements (ML runs)       {measurements}")
    print(f"rejected as ambiguous        {rejected}")
    print(f"rejection rate               {rejected / measurements:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
