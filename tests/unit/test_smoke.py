"""Baseline smoke tests to verify test runner and package importability."""

import nestor


def test_package_metadata(package_version: str) -> None:
    """Verify that the package version is defined and matches the fixture."""
    assert nestor.__version__ == "0.1.0"
    assert nestor.__version__ == package_version


def test_package_exports() -> None:
    """Verify that __all__ contains canonical package exports."""
    assert "__version__" in nestor.__all__
