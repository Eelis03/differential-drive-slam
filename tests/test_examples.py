"""Tier 3: every script in examples/ runs to completion under a reduced step count."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

EXAMPLES = REPO_ROOT / "examples"

INVOCATIONS = [
    ("run_ekf_slam.py", ["--steps", "60", "--no-figures"]),
    ("run_occupancy_grid.py", ["--steps", "60", "--no-figures"]),
    ("run_consistency_study.py", ["--runs", "3", "--steps", "40", "--no-figures"]),
    ("run_data_association.py", ["--steps", "60", "--seeds", "2"]),
]


def test_every_example_is_covered() -> None:
    scripts = {path.name for path in EXAMPLES.glob("*.py")}
    assert scripts == {name for name, _ in INVOCATIONS}


@pytest.mark.parametrize(("script", "arguments"), INVOCATIONS)
def test_example_runs_to_completion(script: str, arguments: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / script), *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_figure_generation_runs(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / "run_ekf_slam.py"),
            "--steps",
            "40",
            "--output",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    written = sorted(path.name for path in tmp_path.glob("*.png"))
    assert written == [
        "ekf_slam_error.png",
        "ekf_slam_nees.png",
        "ekf_slam_trajectory.png",
    ]


def test_occupancy_example_writes_its_figure(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / "run_occupancy_grid.py"),
            "--steps",
            "40",
            "--output",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "occupancy_grid.png").exists()
