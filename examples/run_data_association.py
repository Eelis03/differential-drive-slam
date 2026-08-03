"""Compare maximum likelihood data association against known correspondences.

Three modes are run over the same noise seeds. The known correspondence run is the
reference: it isolates the error made by the filter itself. The maximum likelihood
run recovers the correspondences from the measurements alone, and the difference
against the reference is the price of solving the association problem. The third
mode repeats the second with map management switched off, which isolates what
deleting unsupported landmarks is worth.

Usage::

    uv run python examples/run_data_association.py
    uv run python examples/run_data_association.py --steps 200 --seeds 2
"""

from __future__ import annotations

import argparse

import numpy as np

from diffdrive_slam.algorithm.ekf_slam import MapManagement
from diffdrive_slam.analysis.metrics import Evaluation, evaluate
from diffdrive_slam.pipeline.simulate import SimulationConfig, run_simulation
from diffdrive_slam.pipeline.trace import Trace

KNOWN = "known correspondence"
MAXIMUM_LIKELIHOOD = "maximum likelihood"
NO_DELETION = "the same, no deletion"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=640, help="steps per run")
    parser.add_argument("--seeds", type=int, default=5, help="number of noise seeds")
    parser.add_argument("--seed", type=int, default=2000, help="first noise seed")
    return parser.parse_args(argv)


def _run(steps: int, seed: int, known: bool, manage: bool) -> tuple[Trace, Evaluation]:
    config = SimulationConfig(
        steps=steps,
        seed=seed,
        build_grid=False,
        known_correspondence=known,
        map_management=MapManagement(enabled=manage),
    )
    trace = run_simulation(config)
    return trace, evaluate(trace)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)

    rows: dict[str, list[tuple[Trace, Evaluation]]] = {
        KNOWN: [],
        MAXIMUM_LIKELIHOOD: [],
        NO_DELETION: [],
    }
    for index in range(args.seeds):
        seed = args.seed + index
        rows[KNOWN].append(_run(args.steps, seed, known=True, manage=True))
        rows[MAXIMUM_LIKELIHOOD].append(_run(args.steps, seed, known=False, manage=True))
        rows[NO_DELETION].append(_run(args.steps, seed, known=False, manage=False))

    print(f"steps per run                {args.steps}")
    print(f"seeds                        {args.seeds}")
    header = f"{'association':<22}{'ATE RMSE [m]':>14}{'landmark RMSE [m]':>20}"
    header += f"{'landmarks':>12}{'incorrect':>11}"
    print(header)
    for label, results in rows.items():
        evaluations = [item for _, item in results]
        trajectory = float(np.mean([item.trajectory.position_rmse for item in evaluations]))
        landmarks = float(np.mean([item.landmarks.rmse for item in evaluations]))
        count = float(np.mean([item.landmarks.estimated for item in evaluations]))
        incorrect = int(sum(item.associations.incorrect for item in evaluations))
        print(
            f"{label:<22}{trajectory:>14.4f}{landmarks:>20.4f}{count:>12.1f}{incorrect:>11d}"
        )

    managed = rows[MAXIMUM_LIKELIHOOD]
    rejected = int(sum(item.associations.rejected for _, item in managed))
    measurements = int(sum(item.associations.measurements for _, item in managed))
    print(f"measurements (ML runs)       {measurements}")
    print(f"rejected as ambiguous        {rejected}")
    print(f"rejection rate               {rejected / measurements:.4f}")
    deleted = sum(trace.removed_landmarks for trace, _ in managed)
    surplus = sum(
        trace.final_state.num_landmarks - trace.true_landmarks.shape[0]
        for trace, _ in rows[NO_DELETION]
    )
    print(f"landmarks deleted            {deleted}")
    print(f"surplus landmarks without it {surplus}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
