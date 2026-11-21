"""LaTeX compilation diagnostic and error log parser.

Extracts structured LatexError instances containing line numbers, classification,
and root-cause messages from engine stdout, stderr, and .log files.
"""

import re
from typing import Final

from nestor.ports.compiler import LatexError, LatexErrorKind

# Matches classical TeX error headers: "! Some error message"
_TEX_ERROR_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"^!\s+(.+)$")

# Matches classical TeX line indicator: "l.42 \invalidcommand"
_TEX_LINE_INDICATOR_RE: Final[re.Pattern[str]] = re.compile(r"^l\.(\d+)\s*(.*)$")

# Matches Tectonic modern diagnostic line: "error: input.tex:14: message" or "error: message"
_TECTONIC_ERROR_RE: Final[re.Pattern[str]] = re.compile(
    r"^error(?::\s*(?:(?P<file>[^:\n\r]+):(?P<line>\d+):)?\s*(?P<msg>.+))?$",
    re.IGNORECASE,
)

# Noise lines commonly emitted by TeX interactive prompt
_IGNORED_PROMPT_PATTERNS: Final[tuple[str, ...]] = (
    "Type  H <return>  for immediate help",
    "Type <return> to proceed",
    "Type X to quit",
    "See the ",
    "Enter file name:",
    "==> Fatal error occurred",
)


def _classify_error(message: str) -> LatexErrorKind:
    """Classify error string into a standardized LatexErrorKind."""
    lowered = message.lower()
    if "undefined control sequence" in lowered:
        return LatexErrorKind.UNDEFINED_CONTROL_SEQUENCE
    if "missing $" in lowered or "missing }" in lowered or "missing delimiter" in lowered:
        return LatexErrorKind.MISSING_DELIMITER
    if "not found" in lowered and ("file " in lowered or "package" in lowered):
        return LatexErrorKind.PACKAGE_NOT_FOUND
    if "emergency stop" in lowered or "fatal error" in lowered:
        return LatexErrorKind.FATAL
    return LatexErrorKind.SYNTAX


def _is_noise_line(line: str) -> bool:
    """Return True if the line contains standard interactive TeX prompt boilerplate."""
    stripped = line.strip()
    return any(stripped.startswith(pattern) for pattern in _IGNORED_PROMPT_PATTERNS)


def parse_latex_log(log_content: str) -> list[LatexError]:
    """Parse raw TeX or Tectonic compiler output into structured errors."""
    if not log_content or not log_content.strip():
        return []

    errors: list[LatexError] = []
    lines = log_content.splitlines()
    total_lines = len(lines)
    i = 0

    while i < total_lines:
        line = lines[i].strip()

        tex_match = _TEX_ERROR_HEADER_RE.match(line)
        if tex_match:
            raw_msg = tex_match.group(1).strip()
            if "fatal error occurred" in raw_msg.lower():
                i += 1
                continue

            msg_parts = [raw_msg]
            line_number: int | None = None
            context_snippet: str | None = None

            i += 1
            while i < total_lines:
                sub_line = lines[i]
                stripped_sub = sub_line.strip()

                if not stripped_sub:
                    i += 1
                    break

                if _TEX_ERROR_HEADER_RE.match(stripped_sub) or stripped_sub.startswith("error:"):
                    break

                loc_match = _TEX_LINE_INDICATOR_RE.match(stripped_sub)
                if loc_match:
                    line_number = int(loc_match.group(1))
                    loc_context = loc_match.group(2).strip()
                    if (
                        i + 1 < total_lines
                        and lines[i + 1].strip()
                        and not lines[i + 1].startswith("!")
                    ):
                        next_part = lines[i + 1].strip()
                        loc_context = f"{loc_context} {next_part}".strip()
                        i += 1
                    context_snippet = loc_context if loc_context else None
                    i += 1
                    break

                if not _is_noise_line(stripped_sub):
                    cleaned = re.sub(r"^\([^)]+\)\s*", "", stripped_sub)
                    if cleaned:
                        msg_parts.append(cleaned)

                i += 1

            full_msg = " ".join(msg_parts).strip()
            errors.append(
                LatexError(
                    message=full_msg,
                    line=line_number,
                    kind=_classify_error(full_msg),
                    context=context_snippet,
                )
            )
            continue

        tectonic_match = _TECTONIC_ERROR_RE.match(line)
        if tectonic_match:
            msg = (tectonic_match.group("msg") or "").strip()
            lowered_msg = msg.lower()
            is_summary = (
                "halted on fatal error" in lowered_msg or "compilation aborted" in lowered_msg
            )
            if msg and not is_summary:
                line_str = tectonic_match.group("line")
                line_num = int(line_str) if line_str else None
                errors.append(
                    LatexError(
                        message=msg,
                        line=line_num,
                        kind=_classify_error(msg),
                        context=None,
                    )
                )

        i += 1

    return errors
