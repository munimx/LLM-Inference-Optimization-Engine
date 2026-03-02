"""Circuit breaker for Ollama backend connectivity.

Implements the standard closed → open → half-open pattern:

* **Closed** — requests flow through normally.
* **Open** — after *failure_threshold* consecutive failures, all requests
  are immediately rejected for *cooldown_seconds*.
* **Half-open** — after the cooldown expires, one probe request is allowed
  through.  If it succeeds, the breaker resets to *closed*.  If it fails,
  it goes back to *open*.
"""

import time
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Lightweight circuit breaker for backend connectivity.

    Args:
        failure_threshold: Number of consecutive failures before opening.
        cooldown_seconds: How long the circuit stays open before probing.
        name: Human-readable name for logging.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        name: str = "ollama",
    ) -> None:
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._name = name
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._last_failure_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state, transitioning to half-open if
        the cooldown has elapsed while in the open state."""
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self._cooldown:
                self._state = CircuitState.HALF_OPEN
                logger.info("circuit_half_open", breaker=self._name)
        return self._state

    @property
    def is_available(self) -> bool:
        """Whether the breaker allows requests through."""
        return self.state != CircuitState.OPEN

    def record_success(self) -> None:
        """Record a successful call.  Resets the breaker to closed."""
        if self._state != CircuitState.CLOSED:
            logger.info("circuit_closed", breaker=self._name)
        self._consecutive_failures = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed call.  If the threshold is reached, open the
        circuit."""
        self._consecutive_failures += 1
        self._last_failure_time = time.monotonic()
        if self._consecutive_failures >= self._threshold:
            if self._state != CircuitState.OPEN:
                logger.warning(
                    "circuit_opened",
                    breaker=self._name,
                    failures=self._consecutive_failures,
                    cooldown=self._cooldown,
                )
            self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset to closed."""
        self._consecutive_failures = 0
        self._state = CircuitState.CLOSED


__all__ = ["CircuitBreaker", "CircuitState"]
