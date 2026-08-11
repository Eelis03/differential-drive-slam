"""Drives the filter over a simulated trajectory and records a trace.

One step of the loop does the following. The commanded control is perturbed by a
sample from the motion noise model to obtain the true motion, while the filter
predicts with the unperturbed command. The environment is then queried for the
landmarks visible from the true pose, the measurements are corrupted with sensor
noise, and the filter folds them in. Finally a laser scan is simulated and mapped
into the occupancy grid at the filtered pose, so the map inherits the localisation
error rather than being handed ground truth.

The true initial pose is drawn from the initial belief. Starting with a zero error
but a non-zero initial covariance would make the filter look conservative at the
first steps and would bias the consistency study.

Motion noise, landmark measurement noise, and laser noise are drawn from three
independent generators spawned from one seed sequence. A single shared stream
would make the trajectory depend on whether the occupancy grid is switched on,
which would prevent the two example scripts from reporting the same run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from diffdrive_slam.algorithm.association import AssociationKind
from diffdrive_slam.algorithm.ekf_slam import EkfSlam, EkfSlamConfig, MapManagement
from diffdrive_slam.algorithm.occupancy import OccupancyGridMapper
from diffdrive_slam.model.arrays import FloatArray, wrap_angles
from diffdrive_slam.model.grid import GridSpec, LogOddsParams
from diffdrive_slam.model.motion import (
    Control,
    MotionNoise,
    control_noise_covariance,
    predict_pose,
)
from diffdrive_slam.model.sensor import RangeBearingParams
from diffdrive_slam.pipeline.environment import Environment, arena_environment
from diffdrive_slam.pipeline.trace import StepRecord, Trace
from diffdrive_slam.pipeline.trajectory import (
    repeat_to_length,
    square_loop_controls,
    square_loop_start,
)

__all__ = ["SimulationConfig", "run_simulation"]

_DEFAULT_INITIAL_STD = (0.05, 0.05, 0.017)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Everything needed to reproduce one run."""

    steps: int = 640
    dt: float = 0.1
    #: Seed of the noise stream. Vary this and hold ``environment_seed`` to obtain
    #: independent Monte Carlo runs over the same map.
    seed: int = 20260731
    environment_seed: int = 7
    landmark_count: int = 20

    side: float = 5.0
    speed: float = 1.0
    turn_rate: float = float(np.pi) / 4.0

    motion_noise: MotionNoise = field(default_factory=MotionNoise)
    sensor: RangeBearingParams = field(default_factory=RangeBearingParams)
    known_correspondence: bool = False
    map_management: MapManagement = field(default_factory=MapManagement)

    #: Leave as ``None`` to start where the closed loop is centred on the origin.
    initial_pose: tuple[float, float, float] | None = None
    initial_std: tuple[float, float, float] = _DEFAULT_INITIAL_STD

    build_grid: bool = True
    grid: GridSpec = field(default_factory=GridSpec)
    log_odds: LogOddsParams = field(default_factory=LogOddsParams)
    scan_beams: int = 45
    scan_max_range: float = 6.0
    scan_sigma: float = 0.05
    scan_interval: int = 2

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError(f"steps must be positive, got {self.steps}")
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.scan_interval <= 0:
            raise ValueError(f"scan_interval must be positive, got {self.scan_interval}")
        if self.scan_beams <= 0:
            raise ValueError(f"scan_beams must be positive, got {self.scan_beams}")

    def initial_covariance(self) -> FloatArray:
        """Initial pose covariance implied by ``initial_std``."""
        return np.diag(np.square(np.asarray(self.initial_std, dtype=np.float64)))

    def resolved_initial_pose(self) -> FloatArray:
        """Return the start pose, defaulting to the one that centres the loop."""
        pose = (
            self.initial_pose
            if self.initial_pose is not None
            else square_loop_start(self.side, self.speed, self.turn_rate)
        )
        return np.asarray(pose, dtype=np.float64)


def _sample_control(rng: np.random.Generator, control: Control, noise: MotionNoise) -> Control:
    covariance = control_noise_covariance(control, noise)
    perturbation = rng.normal(0.0, np.sqrt(np.diag(covariance)))
    return Control(
        linear_velocity=control.linear_velocity + float(perturbation[0]),
        angular_velocity=control.angular_velocity + float(perturbation[1]),
    )


def _renumber(mapping: dict[int, int], removed: Sequence[int]) -> dict[int, int]:
    """Apply the slot deletions in ``removed`` to a slot to identity mapping.

    Deleting a slot shifts every higher slot down by one, so a slot that survives
    loses one position for each deleted slot below it.
    """
    dropped = set(removed)
    return {
        slot - sum(1 for gone in dropped if gone < slot): identity
        for slot, identity in mapping.items()
        if slot not in dropped
    }


def run_simulation(
    config: SimulationConfig | None = None, environment: Environment | None = None
) -> Trace:
    """Run the simulator and return the resulting :class:`Trace`."""
    settings = config if config is not None else SimulationConfig()
    world = (
        environment
        if environment is not None
        else arena_environment(
            seed=settings.environment_seed, landmark_count=settings.landmark_count
        )
    )

    motion_rng, landmark_rng, scan_rng = (
        np.random.default_rng(child) for child in np.random.SeedSequence(settings.seed).spawn(3)
    )
    controls = repeat_to_length(
        square_loop_controls(
            dt=settings.dt,
            side=settings.side,
            speed=settings.speed,
            turn_rate=settings.turn_rate,
            laps=1,
        ),
        settings.steps,
    )

    nominal_pose = settings.resolved_initial_pose()
    initial_covariance = settings.initial_covariance()
    true_pose = nominal_pose + motion_rng.normal(0.0, np.asarray(settings.initial_std))
    true_pose[2] = float(wrap_angles(np.asarray([true_pose[2]]))[0])
    dead_reckoned = nominal_pose.copy()

    filter_config = EkfSlamConfig(
        motion_noise=settings.motion_noise,
        sensor=settings.sensor,
        map_management=settings.map_management,
    )
    slam = EkfSlam(nominal_pose, initial_covariance, filter_config)

    mapper = OccupancyGridMapper(settings.grid, settings.log_odds) if settings.build_grid else None
    scan_bearings = np.linspace(
        -np.pi, np.pi, settings.scan_beams, endpoint=False, dtype=np.float64
    )

    slot_to_identity: dict[int, int] = {}
    records: list[StepRecord] = [
        StepRecord(
            time=0.0,
            control=Control(0.0, 0.0),
            true_pose=true_pose.copy(),
            estimated_pose=slam.state.robot_pose,
            pose_covariance=slam.state.robot_covariance,
            dead_reckoned_pose=dead_reckoned.copy(),
            num_landmarks=0,
            associations=(),
            measurement_identities=(),
        )
    ]

    for step in range(1, settings.steps + 1):
        command = controls[step - 1]

        realised = _sample_control(motion_rng, command, settings.motion_noise)
        true_pose = predict_pose(true_pose, realised, settings.dt)
        dead_reckoned = predict_pose(dead_reckoned, command, settings.dt)
        slam.predict(command, settings.dt)

        identities, clean = world.visible_landmarks(true_pose, settings.sensor)
        if clean.shape[0]:
            noise = landmark_rng.normal(
                0.0,
                [settings.sensor.sigma_range, settings.sensor.sigma_bearing],
                size=clean.shape,
            )
            measurements = clean + noise
        else:
            measurements = np.zeros((0, 2), dtype=np.float64)

        correspondences = (
            [int(value) for value in identities] if settings.known_correspondence else None
        )
        associations = slam.integrate(measurements, correspondences)

        removed = slam.last_removals
        if removed:
            slot_to_identity = _renumber(slot_to_identity, removed)
        for association in associations:
            if association.kind is AssociationKind.NEW and association.landmark_index is not None:
                identity = int(identities[association.measurement_index])
                slot_to_identity[association.landmark_index] = identity

        estimated_pose = slam.state.robot_pose
        if mapper is not None and step % settings.scan_interval == 0:
            ranges = world.range_scan(true_pose, scan_bearings, settings.scan_max_range)
            noisy = ranges + scan_rng.normal(0.0, settings.scan_sigma, size=ranges.shape)
            mapper.integrate_scan(
                estimated_pose,
                np.clip(noisy, 0.0, settings.scan_max_range),
                scan_bearings,
                settings.scan_max_range,
            )

        records.append(
            StepRecord(
                time=step * settings.dt,
                control=command,
                true_pose=true_pose.copy(),
                estimated_pose=estimated_pose,
                pose_covariance=slam.state.robot_covariance,
                dead_reckoned_pose=dead_reckoned.copy(),
                num_landmarks=slam.num_landmarks,
                associations=associations,
                measurement_identities=tuple(int(value) for value in identities),
                slot_identities=tuple(
                    slot_to_identity.get(slot, -1) for slot in range(slam.num_landmarks)
                ),
                removed_landmarks=removed,
            )
        )

    ordered_identities = tuple(slot_to_identity.get(slot, -1) for slot in range(slam.num_landmarks))
    return Trace(
        steps=tuple(records),
        true_landmarks=world.landmarks.copy(),
        final_state=slam.state.copy(),
        slot_to_identity=ordered_identities,
        grid=settings.grid if mapper is not None else None,
        occupancy_log_odds=mapper.snapshot() if mapper is not None else None,
    )
