"""Tectonic TeX compiler outbound adapter."""

import asyncio
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from nestor.adapters.outbound.compiler.log_parser import parse_latex_log
from nestor.core.config.compiler import TectonicConfig
from nestor.ports.compiler import (
    CompilationResult,
    LatexError,
    LatexErrorKind,
    TexCompilerError,
    TexNotFoundError,
    TexTimeoutError,
)

type SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]


class TectonicCompiler:
    """Outbound adapter for compiling LaTeX into PDF using the Tectonic engine."""

    def __init__(
        self,
        config: TectonicConfig | None = None,
        *,
        runner: SubprocessRunner | None = None,
    ) -> None:
        self.config = config or TectonicConfig()
        self._runner = runner or subprocess.run

    def _validate_binary_availability(self) -> str:
        """Verify compiler binary exists and return resolved executable path."""
        binary = self.config.binary_path
        resolved = shutil.which(binary)
        if resolved is None and not Path(binary).is_file():
            msg = (
                f"Tectonic binary not found at '{binary}'. "
                "Ensure tectonic is installed and available in PATH, "
                "or specify an explicit path via TectonicConfig."
            )
            raise TexNotFoundError(msg)
        return resolved or binary

    def _execute_compilation(
        self,
        temp_dir: Path,
        source: str,
        stem: str,
        target_pdf_name: str,
        assets: dict[str, bytes] | None,
        timeout: float,
    ) -> CompilationResult:
        """Synchronously execute Tectonic within an isolated temporary directory."""
        tex_file = temp_dir / f"{stem}.tex"
        tex_file.write_text(source, encoding="utf-8")

        if assets:
            for rel_path_str, content in assets.items():
                rel_path = Path(rel_path_str)
                if (
                    rel_path.is_absolute()
                    or bool(rel_path.anchor)
                    or rel_path_str.startswith(("/", "\\"))
                    or ".." in rel_path.parts
                ):
                    msg = f"Insecure asset path '{rel_path_str}'. Path traversal is prohibited."
                    raise TexCompilerError(msg)
                asset_file = temp_dir / rel_path
                asset_file.parent.mkdir(parents=True, exist_ok=True)
                asset_file.write_bytes(content)

        cmd: list[str] = [self.config.binary_path]
        if self.config.untrusted:
            cmd.append("--untrusted")
        if self.config.keep_logs:
            cmd.append("--keep-logs")
        if self.config.chatter:
            cmd.extend(["--chatter", self.config.chatter])
        cmd.extend(["--outdir", str(temp_dir), str(tex_file.name)])

        stdout = ""
        stderr = ""
        returncode = -1

        try:
            result = self._runner(
                cmd,
                cwd=str(temp_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            returncode = result.returncode
        except FileNotFoundError as err:
            msg = f"Tectonic executable not found: '{self.config.binary_path}'"
            raise TexNotFoundError(msg) from err
        except subprocess.TimeoutExpired as err:
            msg = f"TeX compilation timed out after {timeout:.1f} seconds."
            raise TexTimeoutError(msg) from err

        log_file = temp_dir / f"{stem}.log"
        file_log_content = ""
        if log_file.is_file():
            try:
                file_log_content = log_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                file_log_content = ""

        combined_log = "\n".join(
            part for part in (stdout, stderr, file_log_content) if part.strip()
        )

        pdf_file = temp_dir / target_pdf_name
        if returncode == 0 and pdf_file.is_file():
            pdf_bytes = pdf_file.read_bytes()
            return CompilationResult(
                success=True,
                pdf_bytes=pdf_bytes,
                errors=[],
                log=combined_log,
                output_filename=target_pdf_name,
            )

        parsed_errors = parse_latex_log(combined_log)
        if not parsed_errors:
            fallback_msg = f"Tectonic exited with code {returncode}. Check engine log for details."
            parsed_errors = [
                LatexError(
                    message=fallback_msg,
                    line=None,
                    kind=LatexErrorKind.UNKNOWN,
                    context=None,
                )
            ]

        return CompilationResult(
            success=False,
            pdf_bytes=None,
            errors=parsed_errors,
            log=combined_log,
            output_filename=target_pdf_name,
        )

    async def compile(
        self,
        source: str,
        *,
        output_filename: str = "document.pdf",
        assets: dict[str, bytes] | None = None,
        timeout: float | None = None,
    ) -> CompilationResult:
        """Asynchronously compile LaTeX source into a PDF document."""
        self._validate_binary_availability()
        effective_timeout = timeout if timeout is not None else self.config.timeout

        clean_name = output_filename.strip()
        if not clean_name:
            clean_name = "document.pdf"
        elif "/" in clean_name or "\\" in clean_name:
            msg = f"Insecure output filename '{output_filename}'. Directory paths are prohibited."
            raise TexCompilerError(msg)

        if not clean_name.endswith(".pdf"):
            stem = clean_name
            target_pdf_name = f"{clean_name}.pdf"
        else:
            stem = clean_name[:-4]
            target_pdf_name = clean_name

        if not stem:
            stem = "document"
            target_pdf_name = "document.pdf"

        with tempfile.TemporaryDirectory(prefix="nestor_tex_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            return await asyncio.to_thread(
                self._execute_compilation,
                temp_dir,
                source,
                stem,
                target_pdf_name,
                assets,
                effective_timeout,
            )
