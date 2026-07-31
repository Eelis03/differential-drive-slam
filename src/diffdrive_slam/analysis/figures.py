"""Figures drawn from a trace. Plotting lives here and nowhere else.

The Agg backend is selected on import so that the example scripts and the test
suite run without a display server.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from diffdrive_slam.analysis.metrics import ConsistencySummary
from diffdrive_slam.model.arrays import FloatArray
from diffdrive_slam.model.grid import log_odds_to_probability
from diffdrive_slam.pipeline.trace import Trace

matplotlib.use("Agg")

__all__ = [
    "covariance_ellipse",
    "plot_error_history",
    "plot_nees",
    "plot_occupancy_grid",
    "plot_trajectory",
    "save_figure",
]


def covariance_ellipse(
    mean: FloatArray,
    covariance: FloatArray,
    confidence: float = 0.95,
    edgecolor: str = "tab:red",
    linewidth: float = 0.8,
    alpha: float = 0.7,
) -> Ellipse:
    """Return the confidence ellipse of a two-dimensional Gaussian.

    The axes are scaled by the square root of the chi-square quantile with two
    degrees of freedom, so the ellipse encloses ``confidence`` of the probability
    mass.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    scaled = np.clip(eigenvalues[order], 0.0, None) * float(chi2.ppf(confidence, 2))
    major = eigenvectors[:, order[0]]
    angle = float(np.degrees(np.arctan2(float(major[1]), float(major[0]))))
    axes_lengths = 2.0 * np.sqrt(scaled)
    return Ellipse(
        xy=(float(mean[0]), float(mean[1])),
        width=float(axes_lengths[0]),
        height=float(axes_lengths[1]),
        angle=angle,
        edgecolor=edgecolor,
        facecolor="none",
        linewidth=linewidth,
        alpha=alpha,
    )


def plot_trajectory(trace: Trace, confidence: float = 0.95) -> Figure:
    """Plot ground truth, dead reckoning, and the filtered trajectory with the map."""
    figure, axes = plt.subplots(figsize=(7.0, 7.0))
    true_poses = trace.true_poses
    estimated = trace.estimated_poses
    dead_reckoned = trace.dead_reckoned_poses

    axes.plot(true_poses[:, 0], true_poses[:, 1], color="black", lw=1.6, label="ground truth")
    axes.plot(
        dead_reckoned[:, 0],
        dead_reckoned[:, 1],
        color="tab:orange",
        lw=1.2,
        ls="--",
        label="dead reckoning",
    )
    axes.plot(estimated[:, 0], estimated[:, 1], color="tab:blue", lw=1.2, label="EKF-SLAM")

    axes.scatter(
        trace.true_landmarks[:, 0],
        trace.true_landmarks[:, 1],
        marker="+",
        s=70,
        color="black",
        label="true landmarks",
    )
    landmarks = trace.estimated_landmarks
    if landmarks.size:
        axes.scatter(
            landmarks[:, 0],
            landmarks[:, 1],
            marker="o",
            s=22,
            facecolors="none",
            edgecolors="tab:red",
            label="estimated landmarks",
        )
        for position, covariance in zip(
            landmarks, trace.estimated_landmark_covariances, strict=True
        ):
            axes.add_patch(covariance_ellipse(position, covariance, confidence))

    axes.set_aspect("equal")
    axes.set_xlabel("x [m]")
    axes.set_ylabel("y [m]")
    axes.set_title("EKF-SLAM trajectory and map")
    axes.legend(loc="upper left", fontsize=8)
    axes.grid(True, lw=0.3, alpha=0.5)
    figure.tight_layout()
    return figure


def plot_error_history(trace: Trace) -> Figure:
    """Plot position error against time for the filter and for dead reckoning."""
    times = trace.times
    true_poses = trace.true_poses
    slam_error = np.linalg.norm(true_poses[:, :2] - trace.estimated_poses[:, :2], axis=1)
    odometry_error = np.linalg.norm(
        true_poses[:, :2] - trace.dead_reckoned_poses[:, :2], axis=1
    )

    figure, axes = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True)
    axes[0].plot(times, odometry_error, color="tab:orange", lw=1.0, label="dead reckoning")
    axes[0].plot(times, slam_error, color="tab:blue", lw=1.0, label="EKF-SLAM")
    axes[0].set_ylabel("position error [m]")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, lw=0.3, alpha=0.5)

    axes[1].plot(times, trace.landmark_counts, color="tab:green", lw=1.0)
    axes[1].set_ylabel("landmarks in state")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(True, lw=0.3, alpha=0.5)

    figure.suptitle("Localisation error and map growth")
    figure.tight_layout()
    return figure


def plot_nees(times: FloatArray, nees: FloatArray, summary: ConsistencySummary) -> Figure:
    """Plot the NEES history against the per-step confidence bounds.

    The bands are the interval a single entry of ``nees`` should fall inside, not the
    tighter interval that applies to the average of the whole sequence.
    """
    percent = round(100.0 * summary.confidence)
    figure, axes = plt.subplots(figsize=(7.0, 3.6))
    axes.plot(times, nees, color="tab:blue", lw=0.9, label="NEES")
    axes.axhline(
        summary.degrees_of_freedom, color="black", lw=1.0, ls="-", label="expected value"
    )
    axes.axhline(
        summary.per_step_lower,
        color="tab:red",
        lw=1.0,
        ls="--",
        label=f"{percent} percent per step bounds",
    )
    axes.axhline(summary.per_step_upper, color="tab:red", lw=1.0, ls="--")
    axes.set_xlabel("time [s]")
    axes.set_ylabel("NEES")
    axes.set_title(
        f"Filter consistency: {summary.inside_fraction:.3f} of steps inside "
        f"the {percent} percent band"
    )
    axes.legend(fontsize=8)
    axes.grid(True, lw=0.3, alpha=0.5)
    figure.tight_layout()
    return figure


def plot_occupancy_grid(trace: Trace) -> Figure:
    """Plot the accumulated occupancy grid with the trajectory drawn over it."""
    if trace.grid is None or trace.occupancy_log_odds is None:
        raise ValueError("the trace does not carry an occupancy grid")

    probabilities = log_odds_to_probability(trace.occupancy_log_odds)
    figure, axes = plt.subplots(figsize=(7.0, 6.4))
    image = axes.imshow(
        probabilities,
        origin="lower",
        extent=trace.grid.extent,
        cmap="bone_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    true_poses = trace.true_poses
    axes.plot(true_poses[:, 0], true_poses[:, 1], color="tab:green", lw=1.2, label="ground truth")
    estimated = trace.estimated_poses
    axes.plot(estimated[:, 0], estimated[:, 1], color="tab:blue", lw=1.0, label="EKF-SLAM")
    axes.set_aspect("equal")
    axes.set_xlabel("x [m]")
    axes.set_ylabel("y [m]")
    axes.set_title("Occupancy grid from log-odds inverse sensor model")
    axes.legend(loc="upper left", fontsize=8)
    figure.colorbar(image, ax=axes, label="occupancy probability")
    figure.tight_layout()
    return figure


def save_figure(figure: Figure, path: Path) -> Path:
    """Write ``figure`` to ``path``, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path
