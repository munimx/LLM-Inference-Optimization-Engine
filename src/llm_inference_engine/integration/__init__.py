"""vLLM integration package."""

from llm_inference_engine.integration.backend import BackendResult, InferenceBackend
from llm_inference_engine.integration.backend_pool import BackendPool
from llm_inference_engine.integration.vllm_backend import VLLMBackend

__all__ = ["InferenceBackend", "BackendResult", "VLLMBackend", "BackendPool"]
