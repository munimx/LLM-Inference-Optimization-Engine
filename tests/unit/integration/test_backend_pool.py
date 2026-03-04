"""Unit tests for BackendPool."""

import pytest

from llm_inference_engine.api.circuit_breaker import CircuitState
from llm_inference_engine.integration.backend_pool import BackendPool
from llm_inference_engine.integration.vllm_backend import VLLMBackend


def _make_pool(n: int = 2, failure_threshold: int = 3) -> BackendPool:
    backends = [VLLMBackend(f"http://vllm{i}:8080") for i in range(n)]
    return BackendPool(backends, failure_threshold=failure_threshold)


class TestBackendPool:
    def test_from_urls(self) -> None:
        pool = BackendPool.from_urls(["http://vllm1:8080", "http://vllm2:8080"])
        assert len(pool._backends) == 2

    def test_requires_at_least_one_backend(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            BackendPool([])

    def test_get_healthy_backend_returns_backend(self) -> None:
        pool = _make_pool(2)
        backend = pool.get_healthy_backend()
        assert isinstance(backend, VLLMBackend)

    def test_round_robin_distribution(self) -> None:
        pool = _make_pool(2)
        b1 = pool.get_healthy_backend()
        b2 = pool.get_healthy_backend()
        assert b1 is not b2

    def test_skips_open_circuit(self) -> None:
        pool = _make_pool(2, failure_threshold=1)
        first_backend = pool._backends[0]
        pool.record_failure(first_backend)
        assert pool._breakers[0].state == CircuitState.OPEN

        # Should skip the first and return the second
        for _ in range(4):
            backend = pool.get_healthy_backend()
            assert backend is not first_backend

    def test_returns_none_when_all_open(self) -> None:
        pool = _make_pool(2, failure_threshold=1)
        for backend in pool._backends:
            pool.record_failure(backend)
        assert pool.get_healthy_backend() is None

    def test_healthy_count_all_healthy(self) -> None:
        pool = _make_pool(3)
        assert pool.healthy_count() == 3

    def test_healthy_count_excludes_open_circuits(self) -> None:
        pool = _make_pool(2, failure_threshold=1)
        pool.record_failure(pool._backends[0])
        assert pool.healthy_count() == 1

    def test_record_failure_updates_breaker(self) -> None:
        pool = _make_pool(1)
        pool.record_failure(pool._backends[0])
        assert pool._breakers[0]._consecutive_failures == 1

    def test_record_success_resets_breaker(self) -> None:
        pool = _make_pool(1)
        pool.record_failure(pool._backends[0])
        pool.record_success(pool._backends[0])
        assert pool._breakers[0]._consecutive_failures == 0
