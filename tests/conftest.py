"""Global test configuration and shared fixtures."""

import pytest

import nestor


@pytest.fixture
def package_version() -> str:
    """Fixture returning the canonical package version string."""
    return nestor.__version__
