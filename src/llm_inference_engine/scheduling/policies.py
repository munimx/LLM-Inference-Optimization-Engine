"""Scheduling policies that determine batch formation order."""

from enum import Enum
from typing import Protocol

import structlog

from llm_inference_engine.core.types import Request
from llm_inference_engine.scheduling.batch import Batch

logger = structlog.get_logger(__name__)


class SchedulingPolicy(str, Enum):  # noqa: UP042
    """Available scheduling policies.

    Attributes:
        FCFS: First-Come-First-Served — requests are scheduled in the
            order they arrived (lowest sequence number first).
        SJF: Shortest-Job-First — requests with the smallest token
            budget (``max_tokens``) are scheduled first.
        PRIORITY: Priority scheduling — requests are sorted by their
            ``priority`` field (higher value = scheduled first).  FIFO
            tie-breaking within the same priority level.
        TOKEN_BUDGET: Token-budget-aware scheduling — greedily packs
            requests that fit within the batch token budget, preferring
            higher-priority requests.
    """

    FCFS = "fcfs"
    SJF = "sjf"
    PRIORITY = "priority"
    TOKEN_BUDGET = "token_budget"


class BatchFormationPolicy(Protocol):
    """Protocol satisfied by all concrete scheduling policies."""

    def form_batch(
        self,
        requests: list[Request],
        batch_id: str,
        model: str,
        max_requests: int,
        max_tokens: int,
    ) -> Batch:
        """Form a :class:`~llm_inference_engine.scheduling.batch.Batch`.

        Args:
            requests: Candidate requests (already filtered to *model*).
            batch_id: Identifier for the new batch.
            model: Ollama model tag.
            max_requests: Maximum number of requests in the batch.
            max_tokens: Maximum token budget (``0`` = unlimited).

        Returns:
            A :class:`~llm_inference_engine.scheduling.batch.Batch`
            populated according to the policy.
        """
        ...  # pragma: no cover


class FCFSPolicy:
    """First-Come-First-Served batch formation."""

    def form_batch(
        self,
        requests: list[Request],
        batch_id: str,
        model: str,
        max_requests: int,
        max_tokens: int,
    ) -> Batch:
        """Form a batch by taking requests in arrival order."""
        batch = Batch(
            batch_id=batch_id,
            model=model,
            max_requests=max_requests,
            max_tokens=max_tokens,
        )
        for request in requests:
            if not batch.can_add(request):
                break
            batch.add(request)
        logger.debug("fcfs_batch_formed", batch_id=batch_id, size=batch.size)
        return batch


class SJFPolicy:
    """Shortest-Job-First batch formation.

    Sorts requests by ``generation_config.max_tokens`` ascending so that
    requests with smaller output budgets are processed first, reducing
    average waiting time.
    """

    def form_batch(
        self,
        requests: list[Request],
        batch_id: str,
        model: str,
        max_requests: int,
        max_tokens: int,
    ) -> Batch:
        """Form a batch with shortest-job-first ordering."""
        sorted_requests = sorted(requests, key=lambda r: r.generation_config.max_tokens)
        batch = Batch(
            batch_id=batch_id,
            model=model,
            max_requests=max_requests,
            max_tokens=max_tokens,
        )
        for request in sorted_requests:
            if not batch.can_add(request):
                continue  # SJF skips over jobs that don't fit
            batch.add(request)
        logger.debug("sjf_batch_formed", batch_id=batch_id, size=batch.size)
        return batch


class PriorityPolicy:
    """Priority-based batch formation.

    Requests are sorted by ``priority`` descending; within the same
    priority, FIFO ordering is preserved via ``timestamp``.
    """

    def form_batch(
        self,
        requests: list[Request],
        batch_id: str,
        model: str,
        max_requests: int,
        max_tokens: int,
    ) -> Batch:
        """Form a batch with priority ordering."""
        sorted_requests = sorted(
            requests,
            key=lambda r: (-r.priority, r.timestamp),
        )
        batch = Batch(
            batch_id=batch_id,
            model=model,
            max_requests=max_requests,
            max_tokens=max_tokens,
        )
        for request in sorted_requests:
            if not batch.can_add(request):
                break
            batch.add(request)
        logger.debug("priority_batch_formed", batch_id=batch_id, size=batch.size)
        return batch


class TokenBudgetPolicy:
    """Token-budget-aware greedy bin-packing.

    Greedily selects requests in priority order, skipping those that
    exceed the remaining token budget.  This maximises token utilisation
    within each batch.
    """

    def form_batch(
        self,
        requests: list[Request],
        batch_id: str,
        model: str,
        max_requests: int,
        max_tokens: int,
    ) -> Batch:
        """Form a batch by greedily packing within token budget."""
        sorted_requests = sorted(
            requests,
            key=lambda r: (-r.priority, r.timestamp),
        )
        batch = Batch(
            batch_id=batch_id,
            model=model,
            max_requests=max_requests,
            max_tokens=max_tokens,
        )
        for request in sorted_requests:
            if batch.can_add(request):
                batch.add(request)
        logger.debug("token_budget_batch_formed", batch_id=batch_id, size=batch.size)
        return batch


def get_policy(policy: SchedulingPolicy) -> BatchFormationPolicy:
    """Return the concrete policy implementation for *policy*.

    Args:
        policy: The desired :class:`SchedulingPolicy`.

    Returns:
        A policy instance implementing :class:`BatchFormationPolicy`.

    Raises:
        ValueError: If the policy is not recognised.
    """
    mapping: dict[SchedulingPolicy, BatchFormationPolicy] = {
        SchedulingPolicy.FCFS: FCFSPolicy(),
        SchedulingPolicy.SJF: SJFPolicy(),
        SchedulingPolicy.PRIORITY: PriorityPolicy(),
        SchedulingPolicy.TOKEN_BUDGET: TokenBudgetPolicy(),
    }
    if policy not in mapping:
        raise ValueError(f"Unknown scheduling policy: {policy!r}")
    return mapping[policy]


__all__ = [
    "SchedulingPolicy",
    "BatchFormationPolicy",
    "FCFSPolicy",
    "SJFPolicy",
    "PriorityPolicy",
    "TokenBudgetPolicy",
    "get_policy",
]
