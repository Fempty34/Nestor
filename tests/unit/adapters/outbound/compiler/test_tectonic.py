"""Unit tests for subprocess-based Tectonic LaTeX compiler adapter."""

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nestor.adapters.outbound.compiler.tectonic import TectonicCompiler
from nestor.core.config.compiler import TectonicConfig
from nestor.ports.compiler import (
    LatexErrorKind,
    TexCompiler,
    TexCompilerError,
    TexNotFoundError,
    TexTimeoutError,
)


class TestTectonicCompilerInitialization:
    """Verifies initialization invariants and binary resolution."""

    def test_protocol_conformance(self) -> None:
        """Verify TectonicCompiler satisfies abstract TexCompiler protocol."""
        compiler = TectonicCompiler()
        assert isinstance(compiler, TexCompiler)

    def test_missing_binary_raises_not_found_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify TexNotFoundError is raised when binary is not in PATH and not on disk."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: None)
        compiler = TectonicCompiler(config=TectonicConfig(binary_path="nonexistent_tectonic_bin"))

        with pytest.raises(TexNotFoundError, match="Tectonic binary not found"):
            compiler._validate_binary_availability()


class TestTectonicCompilerSubprocessExecution:
    """Tests compilation runner behavior using mocked subprocess runners."""

    @pytest.mark.asyncio
    async def test_compile_successful_minimal_document(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify successful compilation invokes runner with security flags and returns PDF bytes."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/tectonic")

        captured_cmd: list[str] = []
        captured_cwd: str = ""

        def mock_runner(
            cmd: list[str], cwd: str, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            nonlocal captured_cmd, captured_cwd
            captured_cmd = list(cmd)
            captured_cwd = cwd

            out_pdf = Path(cwd) / "document.pdf"
            out_pdf.write_bytes(b"%PDF-1.4 simulated pdf document")

            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="Running TeX...\nWriting output...",
                stderr="",
            )

        compiler = TectonicCompiler(runner=mock_runner)
        latex_source = r"\documentclass{article}\begin{document}Hello World\end{document}"

        result = await compiler.compile(latex_source)

        assert result.success is True
        assert result.pdf_bytes == b"%PDF-1.4 simulated pdf document"
        assert result.errors == []

        assert "--untrusted" in captured_cmd
        assert "--keep-logs" in captured_cmd
        assert "--chatter" in captured_cmd
        assert "minimal" in captured_cmd
        assert "--outdir" in captured_cmd
        assert "document.tex" in captured_cmd

    @pytest.mark.asyncio
    async def test_compile_with_custom_output_filename(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify custom output filename compiles with corresponding stem and returns correct name."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/tectonic")

        captured_cmd: list[str] = []

        def mock_runner(
            cmd: list[str], cwd: str, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            nonlocal captured_cmd
            captured_cmd = list(cmd)
            out_pdf = Path(cwd) / "lecture_01.pdf"
            out_pdf.write_bytes(b"%PDF-1.4 lecture notes")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        compiler = TectonicCompiler(runner=mock_runner)
        result = await compiler.compile(
            r"\documentclass{article}",
            output_filename="lecture_01.pdf",
        )

        assert result.success is True
        assert result.output_filename == "lecture_01.pdf"
        assert result.pdf_bytes == b"%PDF-1.4 lecture notes"
        assert "lecture_01.tex" in captured_cmd

    @pytest.mark.asyncio
    async def test_compile_rejects_insecure_output_filename(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify output filename containing directory paths is rejected."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/tectonic")
        compiler = TectonicCompiler()

        with pytest.raises(TexCompilerError, match="Insecure output filename"):
            await compiler.compile(
                r"\documentclass{article}",
                output_filename="../evil.pdf",
            )

    @pytest.mark.asyncio
    async def test_compile_syntax_error_trapping_undefined_control_sequence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify syntax error in LaTeX is parsed and returned in CompilationResult without raising."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/tectonic")

        error_log = (
            "! Undefined control sequence.\n"
            "l.15 \\nonexistentcommand\n"
            "                         {value}\n"
        )

        def mock_failing_runner(
            cmd: list[str], cwd: str, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout=error_log,
                stderr="error: compilation aborted due to syntax errors",
            )

        compiler = TectonicCompiler(runner=mock_failing_runner)
        result = await compiler.compile(r"\nonexistentcommand{value}")

        assert result.success is False
        assert result.pdf_bytes is None
        assert len(result.errors) == 1

        err = result.errors[0]
        assert err.line == 15
        assert err.kind == LatexErrorKind.UNDEFINED_CONTROL_SEQUENCE
        assert "Undefined control sequence." in err.message

    @pytest.mark.asyncio
    async def test_compile_syntax_error_trapping_missing_delimiter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify missing delimiter error is correctly trapped and structured."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/tectonic")

        error_log = "! Missing $ inserted.\nl.28 x = \\int_0^1 f(t dt\n"

        def mock_failing_runner(
            cmd: list[str], cwd: str, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout=error_log,
                stderr="",
            )

        compiler = TectonicCompiler(runner=mock_failing_runner)
        result = await compiler.compile(r"x = \int_0^1 f(t dt")

        assert result.success is False
        assert len(result.errors) == 1
        assert result.errors[0].line == 28
        assert result.errors[0].kind == LatexErrorKind.MISSING_DELIMITER

    @pytest.mark.asyncio
    async def test_compile_fallback_error_when_no_diagnostic_parsed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify fallback error is generated when engine fails without recognizable TeX log."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/tectonic")

        def mock_silent_failure(
            cmd: list[str], cwd: str, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=137,
                stdout="",
                stderr="Fatal signal received",
            )

        compiler = TectonicCompiler(runner=mock_silent_failure)
        result = await compiler.compile("source")

        assert result.success is False
        assert len(result.errors) == 1
        assert result.errors[0].kind == LatexErrorKind.UNKNOWN
        assert "137" in result.errors[0].message

    @pytest.mark.asyncio
    async def test_compile_timeout_raises_tex_timeout_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify Subprocess.TimeoutExpired is translated into TexTimeoutError."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/tectonic")

        def mock_timeout_runner(
            cmd: list[str], cwd: str, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=5.0)

        compiler = TectonicCompiler(runner=mock_timeout_runner)

        with pytest.raises(TexTimeoutError, match=r"timed out after 5\.0 seconds"):
            await compiler.compile(r"\loop\repeat", timeout=5.0)

    @pytest.mark.asyncio
    async def test_compile_file_not_found_raises_tex_not_found_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify FileNotFoundError during execution raises TexNotFoundError."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/tectonic")

        def mock_missing_runner(
            cmd: list[str], cwd: str, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            raise FileNotFoundError("Executable not found")

        compiler = TectonicCompiler(runner=mock_missing_runner)

        with pytest.raises(TexNotFoundError, match="Tectonic executable not found"):
            await compiler.compile("source")


class TestTectonicCompilerAssetsAndSecurity:
    """Tests asset handling and path traversal protection."""

    @pytest.mark.asyncio
    async def test_compile_writes_valid_relative_assets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify valid relative assets are saved to temporary directory prior to execution."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/tectonic")

        asset_verified = False

        def mock_runner(
            cmd: list[str], cwd: str, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            nonlocal asset_verified
            workdir = Path(cwd)
            asset_path = workdir / "assets" / "diagram.png"
            if asset_path.is_file() and asset_path.read_bytes() == b"png-data":
                asset_verified = True
            (workdir / "document.pdf").write_bytes(b"%PDF-success")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        compiler = TectonicCompiler(runner=mock_runner)
        result = await compiler.compile(
            r"\includegraphics{assets/diagram.png}",
            assets={"assets/diagram.png": b"png-data"},
        )

        assert result.success is True
        assert asset_verified is True

    @pytest.mark.asyncio
    async def test_compile_rejects_parent_directory_traversal_asset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify assets attempting path traversal with '..' are rejected with TexCompilerError."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/tectonic")
        compiler = TectonicCompiler(runner=MagicMock())

        with pytest.raises(TexCompilerError, match="Path traversal is prohibited"):
            await compiler.compile(
                "source",
                assets={"../forbidden.txt": b"content"},
            )

    @pytest.mark.asyncio
    async def test_compile_rejects_absolute_path_asset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify assets attempting absolute path escape are rejected with TexCompilerError."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/tectonic")
        compiler = TectonicCompiler(runner=MagicMock())

        with pytest.raises(TexCompilerError, match="Path traversal is prohibited"):
            await compiler.compile(
                "source",
                assets={"/etc/shadow": b"content"},
            )
