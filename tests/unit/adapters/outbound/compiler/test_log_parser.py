"""Unit tests for LaTeX compiler error log parser."""

from nestor.adapters.outbound.compiler.log_parser import parse_latex_log
from nestor.ports.compiler import LatexErrorKind


class TestParseLatexLogEmptyAndValid:
    """Tests handling of empty, whitespace, and clean logs."""

    def test_empty_string_returns_empty_list(self) -> None:
        """Verify empty log produces no errors."""
        assert parse_latex_log("") == []

    def test_whitespace_string_returns_empty_list(self) -> None:
        """Verify whitespace-only log produces no errors."""
        assert parse_latex_log("   \n\t  \n  ") == []

    def test_clean_successful_compilation_log_returns_empty_list(self) -> None:
        """Verify standard successful compilation output produces no errors."""
        clean_log = (
            "This is XeTeX, Version 3.141592653-2.6-0.999995 (TeX Live 2023/Debian)\n"
            "(./document.tex\n"
            "LaTeX2e <2023-11-01> pre-release-1\n"
            "Output written on document.pdf (1 page).\n"
            "Transcript written on document.log.\n"
        )
        assert parse_latex_log(clean_log) == []


class TestParseLatexLogClassicErrors:
    """Tests parsing classical TeX '!' diagnostic messages."""

    def test_undefined_control_sequence_with_line_and_context(self) -> None:
        """Verify undefined control sequence extracts line number, kind, and context."""
        raw_log = (
            "! Undefined control sequence.\n"
            "l.14 \\invalidmacro\n"
            "                  {arg}\n"
            "The control sequence at the end of the top line\n"
            "of your error message was never \\def'ed.\n"
        )
        errors = parse_latex_log(raw_log)
        assert len(errors) == 1
        err = errors[0]
        assert err.line == 14
        assert err.kind == LatexErrorKind.UNDEFINED_CONTROL_SEQUENCE
        assert "Undefined control sequence." in err.message
        assert err.context is not None
        assert "\\invalidmacro" in err.context

    def test_missing_delimiter_inserted(self) -> None:
        """Verify 'Missing $ inserted' is classified as MISSING_DELIMITER with line."""
        raw_log = (
            "! Missing $ inserted.\n"
            "<inserted text> \n"
            "                $\n"
            "l.22 y = f(x\n"
            "            ) + 1\n"
        )
        errors = parse_latex_log(raw_log)
        assert len(errors) == 1
        err = errors[0]
        assert err.line == 22
        assert err.kind == LatexErrorKind.MISSING_DELIMITER
        assert "Missing $ inserted." in err.message

    def test_multiline_package_error_with_noise_suppression(self) -> None:
        """Verify package error details are captured while interactive prompt noise is stripped."""
        raw_log = (
            "! Package amsmath Error: Erroneous nesting of equation structures;\n"
            "(amsmath)                trying to recover with `aligned'.\n"
            "See the amsmath package documentation for explanation.\n"
            "Type  H <return>  for immediate help.\n"
            " ...                                              \n"
            "l.30 \\end{equation}\n"
        )
        errors = parse_latex_log(raw_log)
        assert len(errors) == 1
        err = errors[0]
        assert err.line == 30
        assert err.kind == LatexErrorKind.SYNTAX
        assert "Erroneous nesting of equation structures" in err.message
        assert "trying to recover with `aligned'" in err.message
        assert "Type  H <return>" not in err.message

    def test_package_file_not_found(self) -> None:
        """Verify missing package file is classified as PACKAGE_NOT_FOUND."""
        raw_log = (
            "! LaTeX Error: File `nonexistent_pkg.sty' not found.\n"
            "Type X to quit or <RETURN> to proceed,\n"
            "or enter new name. (Default extension: sty)\n"
            "Enter file name: \n"
            "l.5 \\usepackage\n"
            "               {nonexistent_pkg}\n"
        )
        errors = parse_latex_log(raw_log)
        assert len(errors) == 1
        err = errors[0]
        assert err.line == 5
        assert err.kind == LatexErrorKind.PACKAGE_NOT_FOUND
        assert "File `nonexistent_pkg.sty' not found" in err.message

    def test_emergency_stop_without_line(self) -> None:
        """Verify emergency stop without line indicator is parsed as FATAL with line=None."""
        raw_log = (
            "! Emergency stop.\n"
            "<*> document.tex\n"
            "                \n"
            "!  ==> Fatal error occurred, no output PDF file produced!\n"
        )
        errors = parse_latex_log(raw_log)
        assert len(errors) == 1
        err = errors[0]
        assert err.line is None
        assert err.kind == LatexErrorKind.FATAL
        assert "Emergency stop." in err.message

    def test_multiple_sequential_errors(self) -> None:
        """Verify multiple errors in a single log are all extracted in order."""
        raw_log = (
            "! Undefined control sequence.\n"
            "l.10 \\foo\n"
            "          \n"
            "! Missing $ inserted.\n"
            "l.25 x = 1\n"
        )
        errors = parse_latex_log(raw_log)
        assert len(errors) == 2
        assert errors[0].line == 10
        assert errors[0].kind == LatexErrorKind.UNDEFINED_CONTROL_SEQUENCE
        assert errors[1].line == 25
        assert errors[1].kind == LatexErrorKind.MISSING_DELIMITER


class TestParseLatexLogTectonicFormat:
    """Tests parsing modern Tectonic CLI diagnostic error messages."""

    def test_tectonic_file_line_error(self) -> None:
        """Verify 'error: file.tex:line: message' is correctly extracted."""
        raw_log = "error: document.tex:42: unrecognized command `\\customdef`"
        errors = parse_latex_log(raw_log)
        assert len(errors) == 1
        err = errors[0]
        assert err.line == 42
        assert err.message == "unrecognized command `\\customdef`"
        assert err.kind == LatexErrorKind.SYNTAX

    def test_tectonic_error_without_line(self) -> None:
        """Verify general Tectonic error without line is extracted with line=None."""
        raw_log = "error: could not read input file `bad.tex`"
        errors = parse_latex_log(raw_log)
        assert len(errors) == 1
        err = errors[0]
        assert err.line is None
        assert err.message == "could not read input file `bad.tex`"

    def test_tectonic_fatal_summary_is_ignored(self) -> None:
        """Verify Tectonic's redundant 'halted on fatal error' is not treated as a separate error."""
        raw_log = (
            "error: input.tex:15: unexpected token\n"
            "error: halted on fatal error: compilation failed\n"
        )
        errors = parse_latex_log(raw_log)
        assert len(errors) == 1
        assert errors[0].line == 15
        assert errors[0].message == "unexpected token"
