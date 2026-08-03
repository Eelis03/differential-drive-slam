"""Tier 1: what the installed distribution promises to the code that imports it."""

from __future__ import annotations

import diffdrive_slam
from tests.conftest import REPO_ROOT

PACKAGE_ROOT = REPO_ROOT / "src" / "diffdrive_slam"


def test_py_typed_marker_sits_inside_the_package() -> None:
    """PEP 561: without this file a strictly typed package ships no types at all."""
    marker = PACKAGE_ROOT / "py.typed"
    assert marker.is_file()
    assert marker.parent.name == "diffdrive_slam"
    assert (marker.parent / "__init__.py").is_file()


def test_the_marker_ships_beside_the_imported_module() -> None:
    installed = diffdrive_slam.__file__
    assert installed is not None
    from pathlib import Path

    assert (Path(installed).parent / "py.typed").is_file()


def test_the_marker_is_empty() -> None:
    """PEP 561 gives no meaning to the contents, so anything in it would be noise."""
    assert (PACKAGE_ROOT / "py.typed").read_bytes() == b""


def test_published_figures_fit_the_byte_budget() -> None:
    """The tracked figures share a 250 KB budget across the whole repository."""
    figures = sorted((REPO_ROOT / "docs" / "figures").glob("*.png"))
    assert len(figures) == 3
    total = sum(path.stat().st_size for path in figures)
    assert total <= 250 * 1024, f"tracked figures total {total} bytes"


def test_every_published_figure_is_embedded_in_the_readme() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for path in sorted((REPO_ROOT / "docs" / "figures").glob("*.png")):
        assert f"docs/figures/{path.name}" in readme
