"""Array aliases and small numeric helpers shared by the model layer.

The helpers here are deliberately trivial. They exist so that angle wrapping and
covariance symmetrisation are defined once and used identically by the motion
model, the sensor model, the filter, and the analysis code.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = [
    "BoolArray",
    "FloatArray",
    "IntArray",
    "is_positive_semidefinite",
    "is_symmetric",
    "symmetrise",
    "wrap_angle",
    "wrap_angles",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

_TWO_PI: Final[float] = 2.0 * float(np.pi)


def wrap_angle(angle: float) -> float:
    """Wrap ``angle`` in radians into the half-open interval [-pi, pi)."""
    return float((angle + np.pi) % _TWO_PI - np.pi)


def wrap_angles(angles: FloatArray) -> FloatArray:
    """Wrap every element of ``angles`` into the half-open interval [-pi, pi)."""
    return np.asarray((angles + np.pi) % _TWO_PI - np.pi, dtype=np.float64)


def symmetrise(matrix: FloatArray) -> FloatArray:
    """Return the symmetric part of ``matrix``.

    Covariance updates accumulate asymmetry of the order of machine epsilon. The
    filter applies this after every prediction and correction so that the stored
    covariance remains exactly symmetric.
    """
    return np.asarray(0.5 * (matrix + matrix.T), dtype=np.float64)


def is_symmetric(matrix: FloatArray, tolerance: float = 1e-9) -> bool:
    """Return whether ``matrix`` equals its transpose within ``tolerance``."""
    return bool(np.allclose(matrix, matrix.T, atol=tolerance, rtol=0.0))


def is_positive_semidefinite(matrix: FloatArray, tolerance: float = 1e-9) -> bool:
    """Return whether ``matrix`` is symmetric with no eigenvalue below ``-tolerance``."""
    if not is_symmetric(matrix, tolerance):
        return False
    eigenvalues = np.linalg.eigvalsh(symmetrise(matrix))
    return bool(float(eigenvalues.min()) >= -tolerance)
