"""Open-loop control sequences that drive the simulated robot.

The controls are commanded body twists. The simulator perturbs them with the
motion noise model to obtain the true motion, while the filter predicts with the
commanded values, which is the usual arrangement for evaluating a filter against
ground truth.
"""

from __future__ import annotations

import numpy as np

from diffdrive_slam.model.motion import Control

__all__ = [
    "figure_eight_controls",
    "repeat_to_length",
    "square_loop_controls",
    "square_loop_start",
]


def square_loop_controls(
    dt: float = 0.1,
    side: float = 6.0,
    speed: float = 1.0,
    turn_rate: float = float(np.pi) / 4.0,
    laps: int = 2,
) -> tuple[Control, ...]:
    """Return controls that drive a closed loop of four straight legs and four turns.

    The turns are executed at constant angular rate, so the corners are arcs of
    radius ``speed / turn_rate`` rather than exact right angles. The path closes on
    itself, which lets the second lap revisit every landmark of the first.
    """
    if dt <= 0.0 or side <= 0.0 or speed <= 0.0 or turn_rate <= 0.0:
        raise ValueError("dt, side, speed, and turn_rate must all be positive")
    if laps <= 0:
        raise ValueError(f"laps must be positive, got {laps}")

    straight_steps = max(round(side / (speed * dt)), 1)
    turn_steps = max(round(0.5 * float(np.pi) / (turn_rate * dt)), 1)

    leg = [Control(linear_velocity=speed, angular_velocity=0.0)] * straight_steps
    corner = [Control(linear_velocity=speed, angular_velocity=turn_rate)] * turn_steps
    lap = (leg + corner) * 4
    return tuple(lap * laps)


def square_loop_start(
    side: float = 5.0, speed: float = 1.0, turn_rate: float = float(np.pi) / 4.0
) -> tuple[float, float, float]:
    """Return the start pose that centres :func:`square_loop_controls` on the origin.

    The traversed path is a rounded square whose straight legs have length ``side``
    and whose corners are quarter arcs of radius ``speed / turn_rate``, so its
    bounding box is ``side + 2 * radius`` on each axis. Starting anywhere else leaves
    the loop off centre, which in a bounded room can drive the robot through a wall.
    """
    if side <= 0.0 or speed <= 0.0 or turn_rate <= 0.0:
        raise ValueError("side, speed, and turn_rate must all be positive")
    radius = speed / turn_rate
    return (-0.5 * side, -(0.5 * side + radius), 0.0)


def figure_eight_controls(
    dt: float = 0.1,
    speed: float = 1.0,
    turn_rate: float = float(np.pi) / 4.0,
    lobes: int = 2,
) -> tuple[Control, ...]:
    """Return controls tracing a figure eight of two counter-rotating circles."""
    if dt <= 0.0 or speed <= 0.0 or turn_rate <= 0.0:
        raise ValueError("dt, speed, and turn_rate must all be positive")
    if lobes <= 0:
        raise ValueError(f"lobes must be positive, got {lobes}")

    circle_steps = max(round(2.0 * float(np.pi) / (turn_rate * dt)), 1)
    left = [Control(linear_velocity=speed, angular_velocity=turn_rate)] * circle_steps
    right = [Control(linear_velocity=speed, angular_velocity=-turn_rate)] * circle_steps
    return tuple((left + right) * lobes)


def repeat_to_length(controls: tuple[Control, ...], length: int) -> tuple[Control, ...]:
    """Cycle ``controls`` until exactly ``length`` entries are available."""
    if not controls:
        raise ValueError("controls must not be empty")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    repeats = -(-length // len(controls))
    return tuple((controls * repeats)[:length])
