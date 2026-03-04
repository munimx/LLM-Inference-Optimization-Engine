"""Adaptive throttler for vLLM-metric-based admission control.

The throttler polls vLLM's ``/metrics`` Prometheus endpoint on a configurable
interval and reads the ``vllm:kv_cache_usage_perc`` gauge.  New requests are
admitted, queued, or rejected based on that real signal:

- Below ``soft_limit``:            **ACCEPT**
- Between ``soft_limit`` and ``hard_limit``:  **QUEUE**
- At or above ``hard_limit``:      **REJECT**

Usage::

    throttler = AdaptiveThrottler(
        backend_url="http://vllm:8080",
        soft_limit=0.70,
        hard_limit=0.90,
        poll_interval_seconds=5.0,
    )
    await throttler.start()
    decision = throttler.check()
    # ... later ...
    await throttler.stop()
"""

import asyncio
import re
from dataclasses import dataclass
from enum import StrEnum

import httpx
import structlog

logger = structlog.get_logger(__name__)

# Pattern to extract the kv cache usage gauge from Prometheus text format
_KV_CACHE_RE = re.compile(
    r'^vllm:kv_cache_usage_perc\{[^}]*\}\s+([\d.eE+\-]+)',
    re.MULTILINE,
)


class AdmissionDecision(StrEnum):
    """Outcome of a throttler admission check.

    Attributes:
        ACCEPT: Request can be dispatched immediately.
        QUEUE: System is near capacity; request should wait.
        REJECT: System is at/over capacity; request should be dropped.
    """

    ACCEPT = "accept"
    QUEUE = "queue"
    REJECT = "reject"


@dataclass
class ThrottlerStats:
    """Snapshot of throttler state."""

    kv_cache_usage: float   # 0.0 – 1.0 as reported by vLLM
    soft_limit: float
    hard_limit: float
    active_requests: int


class AdaptiveThrottler:
    """Admission control based on vLLM KV cache pressure.

    Args:
        backend_url: Base URL of the vLLM instance to poll.
        soft_limit: KV cache fraction above which requests are queued.
        hard_limit: KV cache fraction above which requests are rejected.
        poll_interval_seconds: How often to fetch /metrics from vLLM.
    """

    def __init__(
        self,
        backend_url: str,
        soft_limit: float = 0.70,
        hard_limit: float = 0.90,
        poll_interval_seconds: float = 5.0,
    ) -> None:
        if not (0.0 < soft_limit < hard_limit <= 1.0):
            raise ValueError(
                "soft_limit must be in (0, hard_limit] and hard_limit <= 1.0"
            )
        self._backend_url = backend_url.rstrip("/")
        self._soft_limit = soft_limit
        self._hard_limit = hard_limit
        self._poll_interval = poll_interval_seconds
        self._kv_usage: float = 0.0
        self._active_requests: int = 0
        self._poll_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background polling task."""
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(
                self._poll_loop(), name="throttler_poll"
            )
            logger.info(
                "throttler_started",
                backend=self._backend_url,
                poll_interval=self._poll_interval,
            )

    async def stop(self) -> None:
        """Stop the background polling task."""
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._poll_task = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> AdmissionDecision:
        """Evaluate whether a new request can be admitted.

        Uses the last observed KV cache usage from the polling loop.
        Thread-safe for reading (single float write is atomic in CPython).
        """
        usage = self._kv_usage
        if usage >= self._hard_limit:
            return AdmissionDecision.REJECT
        if usage >= self._soft_limit:
            return AdmissionDecision.QUEUE
        return AdmissionDecision.ACCEPT

    def increment_active(self) -> None:
        """Record that a new request has been dispatched."""
        self._active_requests += 1

    def decrement_active(self) -> None:
        """Record that a request has completed."""
        self._active_requests = max(0, self._active_requests - 1)

    @property
    def stats(self) -> ThrottlerStats:
        """Return a snapshot of current throttler state."""
        return ThrottlerStats(
            kv_cache_usage=self._kv_usage,
            soft_limit=self._soft_limit,
            hard_limit=self._hard_limit,
            active_requests=self._active_requests,
        )

    @property
    def kv_cache_usage(self) -> float:
        """Last observed vLLM KV cache usage (0.0 – 1.0)."""
        return self._kv_usage

    # ------------------------------------------------------------------
    # Internal polling
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Repeatedly fetch vLLM /metrics and update _kv_usage."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                try:
                    response = await client.get(
                        f"{self._backend_url}/metrics"
                    )
                    if response.status_code == 200:
                        self._kv_usage = _parse_kv_cache_usage(response.text)
                        logger.debug(
                            "throttler_polled",
                            kv_usage=round(self._kv_usage, 4),
                        )
                except Exception as exc:
                    logger.debug(
                        "throttler_poll_error",
                        backend=self._backend_url,
                        error=str(exc),
                    )
                await asyncio.sleep(self._poll_interval)


def _parse_kv_cache_usage(metrics_text: str) -> float:
    """Parse ``vllm:kv_cache_usage_perc`` from Prometheus text format.

    Returns 0.0 if the metric is absent (safe default: allow requests through).
    """
    match = _KV_CACHE_RE.search(metrics_text)
    if match is None:
        return 0.0
    try:
        return float(match.group(1))
    except ValueError:
        return 0.0


__all__ = ["AdaptiveThrottler", "AdmissionDecision", "ThrottlerStats"]

