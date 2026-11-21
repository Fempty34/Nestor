"""Configuration settings for TeX compilation engines."""

import os
from typing import Self

from pydantic import BaseModel, ConfigDict, Field


class TectonicConfig(BaseModel):
    """Configuration settings for Tectonic TeX compiler runner."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    binary_path: str = Field(
        default="tectonic",
        description="Path or binary name for the Tectonic compiler executable.",
    )
    timeout: float = Field(
        default=30.0,
        gt=0.0,
        description="Maximum execution timeout in seconds per compilation run.",
    )
    untrusted: bool = Field(
        default=True,
        description="Enforce Tectonic --untrusted sandbox mode to restrict filesystem access.",
    )
    keep_logs: bool = Field(
        default=True,
        description="Ensure engine writes and preserves detailed diagnostic logs.",
    )
    chatter: str = Field(
        default="minimal",
        description="Console output verbosity ('default', 'minimal', or 'error').",
    )

    @classmethod
    def from_env(cls) -> Self:
        """Create TectonicConfig loading overrides from environment variables."""
        binary_path = os.environ.get("TECTONIC_PATH", "tectonic")
        timeout_env = os.environ.get("TECTONIC_TIMEOUT")
        timeout = float(timeout_env) if timeout_env else 30.0
        return cls(binary_path=binary_path, timeout=timeout)
