"""Health-aware backend pool for vLLM instances.

Maintains a list of :class:`~llm_inference_engine.integration.vllm_backend.VLLMBackend`
instances, each paired with a :class:`~llm_inference_engine.api.circuit_breaker.CircuitBreaker`.
Incoming requests are distributed via round-robin; backends with an open circuit are skipped.

Usage::

    pool = BackendPool.from_urls(["http://vllm-1:8080", "http://vllm-2:8080"])
    backend = pool.get_healthy_backend()
    if backend is None:
        raise RuntimeError("no healthy backends")
    try:
        result = await backend.generate(...)
        pool.record_success(backend)
    except Exception:
        pool.record_failure(backend)
        raise
"""

from __future__ import annotations

import structlog

from llm_inference_engine.api.circuit_breaker import CircuitBreaker
from llm_inference_engine.integration.vllm_backend import VLLMBackend

logger = structlog.get_logger(__name__)


class BackendPool:
    """Round-robin pool of vLLM backends with per-backend circuit breakers.

    Args:
        backends: List of :class:`VLLMBackend` instances.
        failure_threshold: Consecutive failures before opening a circuit.
        cooldown_seconds: Seconds a circuit stays open before probing.
    """

    def __init__(
        self,
        backends: list[VLLMBackend],
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
    ) -> None:
        if not backends:
            raise ValueError("BackendPool requires at least one backend")
        self._backends = backends
        self._breakers = [
            CircuitBreaker(
                failure_threshold=failure_threshold,
                cooldown_seconds=cooldown_seconds,
                name=b._base_url,
            )
            for b in backends
        ]
        self._index = 0  # round-robin cursor

    @classmethod
    def from_urls(
        cls,
        urls: list[str],
        timeout: float = 120.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
    ) -> BackendPool:
        """Construct a pool from a list of vLLM base URLs."""
        backends = [
            VLLMBackend(
                url,
                timeout=timeout,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            for url in urls
        ]
        return cls(
            backends,
            failure_threshold=failure_threshold,
            cooldown_seconds=cooldown_seconds,
        )

    def get_healthy_backend(self) -> VLLMBackend | None:
        """Return the next healthy backend, or ``None`` if all circuits are open.

        Iterates through backends starting from the current round-robin position,
        skipping any whose circuit breaker is open.
        """
        n = len(self._backends)
        for i in range(n):
            idx = (self._index + i) % n
            if self._breakers[idx].is_available:
                self._index = (idx + 1) % n
                return self._backends[idx]
        logger.warning("backend_pool_all_circuits_open", count=n)
        return None

    def record_success(self, backend: VLLMBackend) -> None:
        """Record a successful call, resetting the backend's circuit breaker."""
        breaker = self._breaker_for(backend)
        if breaker is not None:
            breaker.record_success()

    def record_failure(self, backend: VLLMBackend) -> None:
        """Record a failed call, potentially opening the backend's circuit breaker."""
        breaker = self._breaker_for(backend)
        if breaker is not None:
            breaker.record_failure()

    def healthy_count(self) -> int:
        """Return the number of backends with a closed or half-open circuit."""
        return sum(1 for b in self._breakers if b.is_available)

    def _breaker_for(self, backend: VLLMBackend) -> CircuitBreaker | None:
        for b, cb in zip(self._backends, self._breakers, strict=False):
            if b is backend:
                return cb
        return None

    async def close(self) -> None:
        """Close all underlying httpx clients."""
        for backend in self._backends:
            await backend.close()


__all__ = ["BackendPool"]
