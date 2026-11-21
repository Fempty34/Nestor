"""Unit tests for FakeLlmClient and LLM port contracts."""

import pytest
from pydantic import BaseModel, Field

from nestor.adapters.outbound.llm.fake import FakeLlmClient, LlmCallRecord
from nestor.ports.llm import (
    LlmClient,
    LlmError,
    LlmProviderError,
    LlmResponse,
    LlmSchemaValidationError,
)


class SampleReport(BaseModel):
    """Sample structured model for testing structured generation."""

    summary: str = Field(description="Summary of the notes.")
    formula_count: int = Field(ge=0, description="Count of validated formulas.")


class AlternateReport(BaseModel):
    """Incompatible model for schema validation tests."""

    different_field: str


class TestLlmPortContracts:
    """Verifies interface invariants, types, and exceptions of the LLM port."""

    def test_protocol_conformance(self) -> None:
        """Verify FakeLlmClient conforms to LlmClient protocol via runtime check."""
        client = FakeLlmClient()
        assert isinstance(client, LlmClient)

    def test_arbitrary_object_does_not_conform_to_protocol(self) -> None:
        """Verify non-conforming objects fail runtime protocol check."""
        assert not isinstance(object(), LlmClient)

    def test_exception_hierarchy(self) -> None:
        """Verify domain exception inheritance relationships."""
        assert issubclass(LlmSchemaValidationError, LlmError)
        assert issubclass(LlmProviderError, LlmError)
        assert issubclass(LlmError, Exception)

    def test_llm_response_frozen_immutability(self) -> None:
        """Verify LlmResponse is immutable and raises error on attribute modification."""
        resp = LlmResponse[str](
            content="Sample text",
            model="test-model",
            finish_reason="stop",
        )
        assert resp.content == "Sample text"
        assert resp.model == "test-model"
        assert resp.finish_reason == "stop"
        assert resp.parsed is None

        with pytest.raises(Exception):  # noqa: B017
            resp.content = "Modified text"  # type: ignore[misc]


class TestFakeLlmClientGenerate:
    """Tests for unstructured text generation via FakeLlmClient.generate()."""

    @pytest.mark.asyncio
    async def test_generate_returns_default_response(self) -> None:
        """Verify default response string is returned when queue is empty."""
        client = FakeLlmClient()
        resp = await client.generate(prompt="Explain integration by parts")

        assert resp.content == "Mocked LLM completion"
        assert resp.parsed is None
        assert resp.model == "fake-llm-model"
        assert resp.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_respects_custom_default_response(self) -> None:
        """Verify custom default response is returned when provided."""
        client = FakeLlmClient(default_response="Custom fallback response")
        resp = await client.generate(prompt="Test prompt")
        assert resp.content == "Custom fallback response"

    @pytest.mark.asyncio
    async def test_generate_records_call_arguments(self) -> None:
        """Verify all call parameters are accurately logged in call history."""
        client = FakeLlmClient()
        await client.generate(
            prompt="Prove the squeeze theorem",
            system_instruction="You are a mathematics professor.",
            model="gpt-astra-latest",
            temperature=0.3,
            max_tokens=500,
        )

        assert client.call_count == 1
        last = client.last_call
        assert last is not None
        assert last.prompt == "Prove the squeeze theorem"
        assert last.system_instruction == "You are a mathematics professor."
        assert last.model == "gpt-astra-latest"
        assert last.temperature == 0.3
        assert last.max_tokens == 500
        assert last.response_model is None

    @pytest.mark.asyncio
    async def test_generate_fifo_queue(self) -> None:
        """Verify pre-queued responses are returned sequentially in FIFO order."""
        client = FakeLlmClient(
            responses=[
                "First response",
                "Second response",
            ]
        )

        r1 = await client.generate("Prompt 1")
        r2 = await client.generate("Prompt 2")
        r3 = await client.generate("Prompt 3")

        assert r1.content == "First response"
        assert r2.content == "Second response"
        assert r3.content == "Mocked LLM completion"  # falls back to default

    @pytest.mark.asyncio
    async def test_generate_serializes_pydantic_model_in_queue(self) -> None:
        """Verify BaseModel in queue is serialized to JSON string in generate()."""
        report = SampleReport(summary="Limits proof", formula_count=3)
        client = FakeLlmClient(responses=[report])

        resp = await client.generate("Summarize block")
        assert resp.content == report.model_dump_json()
        assert resp.parsed is None

    @pytest.mark.asyncio
    async def test_generate_returns_llm_response_directly_from_queue(self) -> None:
        """Verify pre-constructed LlmResponse in queue is returned verbatim."""
        preset = LlmResponse[str](
            content="Direct response",
            model="custom-direct-model",
            finish_reason="length",
        )
        client = FakeLlmClient(responses=[preset])

        resp = await client.generate("Prompt")
        assert resp == preset


class TestFakeLlmClientStructured:
    """Tests for structured data generation via FakeLlmClient.generate_structured()."""

    @pytest.mark.asyncio
    async def test_generate_structured_with_queued_model_instance(self) -> None:
        """Verify queued Pydantic instance is returned directly in parsed attribute."""
        expected = SampleReport(summary="Topology basics", formula_count=5)
        client = FakeLlmClient(responses=[expected])

        resp = await client.generate_structured(
            prompt="Extract metadata",
            response_model=SampleReport,
        )

        assert resp.parsed == expected
        assert resp.content == expected.model_dump_json()
        assert resp.model == "fake-llm-model"

    @pytest.mark.asyncio
    async def test_generate_structured_parses_valid_json_string(self) -> None:
        """Verify JSON string from queue is parsed into requested schema."""
        raw_json = '{"summary": "Convergence tests", "formula_count": 2}'
        client = FakeLlmClient(responses=[raw_json])

        resp = await client.generate_structured(
            prompt="Analyze lecture",
            response_model=SampleReport,
        )

        assert isinstance(resp.parsed, SampleReport)
        assert resp.parsed.summary == "Convergence tests"
        assert resp.parsed.formula_count == 2
        assert resp.content == raw_json

    @pytest.mark.asyncio
    async def test_generate_structured_with_default_model_response(self) -> None:
        """Verify default_response as BaseModel instance works when queue is empty."""
        default_report = SampleReport(summary="Default summary", formula_count=0)
        client = FakeLlmClient(default_response=default_report)

        resp = await client.generate_structured(
            prompt="Any prompt",
            response_model=SampleReport,
        )

        assert resp.parsed == default_report
        assert resp.content == default_report.model_dump_json()

    @pytest.mark.asyncio
    async def test_generate_structured_raises_on_invalid_json(self) -> None:
        """Verify LlmSchemaValidationError is raised when queue contains malformed JSON."""
        client = FakeLlmClient(responses=["Not a valid JSON string"])

        with pytest.raises(LlmSchemaValidationError, match="Failed to parse string response"):
            await client.generate_structured(
                prompt="Extract",
                response_model=SampleReport,
            )

    @pytest.mark.asyncio
    async def test_generate_structured_raises_on_schema_mismatch(self) -> None:
        """Verify LlmSchemaValidationError is raised when JSON violates Pydantic constraints."""
        # formula_count is negative, violating ge=0
        invalid_schema_json = '{"summary": "Test", "formula_count": -5}'
        client = FakeLlmClient(responses=[invalid_schema_json])

        with pytest.raises(LlmSchemaValidationError, match="Failed to parse string response"):
            await client.generate_structured(
                prompt="Extract",
                response_model=SampleReport,
            )

    @pytest.mark.asyncio
    async def test_generate_structured_raises_on_incompatible_model_in_queue(self) -> None:
        """Verify LlmSchemaValidationError is raised when incompatible model cannot coerce."""
        incompatible = AlternateReport(different_field="random text")
        client = FakeLlmClient(responses=[incompatible])

        with pytest.raises(LlmSchemaValidationError, match="Incompatible model"):
            await client.generate_structured(
                prompt="Extract",
                response_model=SampleReport,
            )

    @pytest.mark.asyncio
    async def test_generate_structured_with_queued_llm_response(self) -> None:
        """Verify pre-constructed LlmResponse is handled correctly in structured call."""
        report = SampleReport(summary="Pre-packaged", formula_count=1)
        preset = LlmResponse[SampleReport](
            content=report.model_dump_json(),
            parsed=report,
            model="gpt-test",
            finish_reason="stop",
        )
        client = FakeLlmClient(responses=[preset])

        resp = await client.generate_structured(
            prompt="Extract",
            response_model=SampleReport,
        )

        assert resp == preset
        assert resp.parsed == report

    @pytest.mark.asyncio
    async def test_generate_structured_records_response_model(self) -> None:
        """Verify response_model type is stored in call history."""
        report = SampleReport(summary="Record test", formula_count=1)
        client = FakeLlmClient(responses=[report])

        await client.generate_structured(
            prompt="Extract",
            response_model=SampleReport,
        )

        assert client.last_call is not None
        assert client.last_call.response_model is SampleReport


class TestFakeLlmClientRoutingAndState:
    """Tests for prompt pattern routing, history spying, and state reset."""

    @pytest.mark.asyncio
    async def test_prompt_pattern_mapping(self) -> None:
        """Verify keyword routing matches substring in prompt when queue is empty."""
        client = FakeLlmClient(
            default_response="Default answer",
            prompt_mappings={
                "taylor": "Taylor series expansion response",
                "cauchy": "Cauchy-Schwarz inequality response",
            },
        )

        r_taylor = await client.generate("Please explain taylor polynomial")
        r_cauchy = await client.generate("Show cauchy theorem")
        r_other = await client.generate("Explain matrices")

        assert r_taylor.content == "Taylor series expansion response"
        assert r_cauchy.content == "Cauchy-Schwarz inequality response"
        assert r_other.content == "Default answer"

    @pytest.mark.asyncio
    async def test_queue_takes_precedence_over_pattern_mappings(self) -> None:
        """Verify FIFO queue has higher priority than prompt keyword mappings."""
        client = FakeLlmClient(
            responses=["Priority from queue"],
            prompt_mappings={"keyword": "Mapped response"},
        )

        resp = await client.generate("Contains keyword")
        assert resp.content == "Priority from queue"

        resp2 = await client.generate("Contains keyword")
        assert resp2.content == "Mapped response"

    @pytest.mark.asyncio
    async def test_enqueue_response_appends_to_queue(self) -> None:
        """Verify dynamic enqueue_response appends to response queue."""
        client = FakeLlmClient()
        client.enqueue_response("Dynamic 1")
        client.enqueue_response("Dynamic 2")

        r1 = await client.generate("P1")
        r2 = await client.generate("P2")

        assert r1.content == "Dynamic 1"
        assert r2.content == "Dynamic 2"

    @pytest.mark.asyncio
    async def test_reset_clears_history_and_queue(self) -> None:
        """Verify reset clears both call audit history and queued responses."""
        client = FakeLlmClient(responses=["Queued 1"])
        await client.generate("P1")

        assert client.call_count == 1
        assert client.last_call is not None

        client.reset()

        assert client.call_count == 0
        assert client.last_call is None
        assert client.calls == []

        resp = await client.generate("P2")
        assert resp.content == "Mocked LLM completion"

    def test_calls_property_returns_shallow_copy(self) -> None:
        """Verify mutating client.calls does not affect internal history."""
        client = FakeLlmClient()
        external_calls = client.calls
        assert external_calls == []

        external_calls.append(
            LlmCallRecord(
                prompt="fake",
                system_instruction=None,
                model=None,
                temperature=0.0,
                max_tokens=None,
            )
        )
        assert client.call_count == 0


class TestFakeLlmClientErrorSimulation:
    """Tests for exception injection and failure simulation."""

    @pytest.mark.asyncio
    async def test_generate_raises_configured_exception(self) -> None:
        """Verify generate() raises error_to_raise and records invocation."""
        error = LlmProviderError("Rate limit exceeded (429)")
        client = FakeLlmClient(error_to_raise=error)

        with pytest.raises(LlmProviderError, match="Rate limit exceeded"):
            await client.generate("Any prompt")

        assert client.call_count == 1
        assert client.last_call is not None
        assert client.last_call.prompt == "Any prompt"

    @pytest.mark.asyncio
    async def test_generate_structured_raises_configured_exception(self) -> None:
        """Verify generate_structured() raises error_to_raise and records invocation."""
        error = TimeoutError("LLM upstream timeout")
        client = FakeLlmClient(error_to_raise=error)

        with pytest.raises(TimeoutError, match="timeout"):
            await client.generate_structured(
                prompt="Any prompt",
                response_model=SampleReport,
            )

        assert client.call_count == 1
        assert client.last_call is not None
        assert client.last_call.response_model is SampleReport
