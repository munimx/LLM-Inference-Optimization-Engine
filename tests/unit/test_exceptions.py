"""Unit tests for the custom exception hierarchy."""

import pytest

from llm_inference_engine.exceptions import (
    BatchFormationError,
    CacheError,
    ConfigurationError,
    InferenceEngineError,
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaTimeoutError,
    OutOfMemoryError,
    SchedulingError,
    ValidationError,
)


class TestInferenceEngineError:
    """Tests for the base InferenceEngineError."""

    def test_message_stored(self) -> None:
        err = InferenceEngineError("test message")
        assert err.message == "test message"
        assert str(err) == "test message"

    def test_details_default_empty_dict(self) -> None:
        err = InferenceEngineError("msg")
        assert err.details == {}

    def test_details_stored_when_provided(self) -> None:
        err = InferenceEngineError("msg", details={"key": "value"})
        assert err.details == {"key": "value"}

    def test_details_not_shared_between_instances(self) -> None:
        err1 = InferenceEngineError("a")
        err2 = InferenceEngineError("b")
        err1.details["x"] = 1
        assert "x" not in err2.details

    def test_is_exception_subclass(self) -> None:
        with pytest.raises(InferenceEngineError):
            raise InferenceEngineError("boom")


class TestSubclassHierarchy:
    """All subclasses inherit from InferenceEngineError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            OllamaConnectionError,
            OllamaTimeoutError,
            ModelNotFoundError,
            ConfigurationError,
            ValidationError,
            OutOfMemoryError,
            SchedulingError,
            BatchFormationError,
            CacheError,
        ],
    )
    def test_is_inference_engine_error(self, exc_class: type) -> None:
        err = exc_class("test")
        assert isinstance(err, InferenceEngineError)

    @pytest.mark.parametrize(
        "exc_class",
        [
            OllamaConnectionError,
            OllamaTimeoutError,
            ModelNotFoundError,
            ConfigurationError,
            ValidationError,
            OutOfMemoryError,
            SchedulingError,
            BatchFormationError,
            CacheError,
        ],
    )
    def test_message_preserved(self, exc_class: type) -> None:
        err = exc_class("specific error text")
        assert err.message == "specific error text"
        assert str(err) == "specific error text"

    @pytest.mark.parametrize(
        "exc_class",
        [
            OllamaConnectionError,
            OllamaTimeoutError,
            ModelNotFoundError,
            ConfigurationError,
            ValidationError,
            OutOfMemoryError,
            SchedulingError,
            BatchFormationError,
            CacheError,
        ],
    )
    def test_details_accepted(self, exc_class: type) -> None:
        err = exc_class("msg", details={"model": "llama3:8b"})
        assert err.details["model"] == "llama3:8b"


class TestSpecificExceptions:
    """Targeted tests for each concrete exception type."""

    def test_ollama_connection_error_catchable_as_base(self) -> None:
        with pytest.raises(InferenceEngineError):
            raise OllamaConnectionError("connection refused")

    def test_ollama_timeout_error_catchable_as_base(self) -> None:
        with pytest.raises(InferenceEngineError):
            raise OllamaTimeoutError("timed out after 30s")

    def test_model_not_found_error(self) -> None:
        err = ModelNotFoundError("llama4:8b not found", details={"available": ["llama3:8b"]})
        assert "llama4:8b" in str(err)
        assert err.details["available"] == ["llama3:8b"]

    def test_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="invalid yaml"):
            raise ConfigurationError("invalid yaml")

    def test_validation_error(self) -> None:
        err = ValidationError("prompt is empty")
        assert isinstance(err, InferenceEngineError)

    def test_out_of_memory_error(self) -> None:
        err = OutOfMemoryError("OOM", details={"requested_gb": 20.0, "available_gb": 14.0})
        assert err.details["requested_gb"] == 20.0

    def test_scheduling_error(self) -> None:
        with pytest.raises(SchedulingError):
            raise SchedulingError("queue full")

    def test_batch_formation_error(self) -> None:
        err = BatchFormationError("no eligible requests")
        assert "no eligible" in str(err)

    def test_cache_error(self) -> None:
        err = CacheError("cache corrupted")
        assert isinstance(err, InferenceEngineError)
