"""Utilities for benchmark timing, memory, and statistics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import cast

import numpy as np
import psutil


@dataclass
class Timer:
    """Simple precision timer utility."""

    _start: float | None = None
    _end: float | None = None

    def __enter__(self) -> Timer:
        """Start timer on context entry."""
        self._start = time.perf_counter()
        self._end = None
        return self

    def __exit__(self, *args: object) -> None:
        """Stop timer on context exit."""
        self._end = time.perf_counter()

    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds."""
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else time.perf_counter()
        return (end - self._start) * 1000.0


class MemoryProfiler:
    """Memory profiling helper for benchmark execution."""

    def __init__(self, process_pid: int | None = None) -> None:
        """Initialize with target process."""
        self._process = psutil.Process(process_pid)
        self._peak_mb = 0.0
        self._baseline_mb = self.get_current_memory_mb()

    def get_current_memory_mb(self) -> float:
        """Get current resident memory in MB."""
        rss_bytes = cast(float, self._process.memory_info().rss)
        current = rss_bytes / (1024 * 1024)
        if current > self._peak_mb:
            self._peak_mb = current
        return current

    def get_baseline(self) -> float:
        """Get baseline memory measurement."""
        return self._baseline_mb

    def get_peak(self) -> float:
        """Get peak memory measured so far."""
        self.get_current_memory_mb()
        return self._peak_mb

    def reset(self) -> None:
        """Reset baseline and peak to current memory."""
        current = self.get_current_memory_mb()
        self._baseline_mb = current
        self._peak_mb = current


class StatisticsCalculator:
    """Statistics helper functions for benchmark aggregations."""

    @staticmethod
    def calculate_statistics(values: list[float]) -> dict[str, float]:
        """Calculate summary statistics for values."""
        if not values:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        arr = np.array(values, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    @staticmethod
    def remove_outliers(values: list[float], std_threshold: float = 2.0) -> list[float]:
        """Remove points outside N standard deviations."""
        if len(values) < 3:
            return values
        arr = np.array(values, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std == 0:
            return values
        filtered = [value for value in values if abs(value - mean) <= std_threshold * std]
        return filtered or values

    @staticmethod
    def calculate_percentiles(values: list[float], percentiles: list[float]) -> dict[str, float]:
        """Calculate selected percentiles."""
        if not values:
            return {f"p{int(p)}": 0.0 for p in percentiles}
        arr = np.array(values, dtype=np.float64)
        return {f"p{int(p)}": float(np.percentile(arr, p)) for p in percentiles}
