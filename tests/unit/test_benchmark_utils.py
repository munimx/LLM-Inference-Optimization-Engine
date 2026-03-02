import time
from unittest.mock import MagicMock, patch

from llm_inference_engine.utils.benchmark_utils import MemoryProfiler, StatisticsCalculator, Timer


def test_timer_elapsed_ms() -> None:
    with Timer() as timer:
        time.sleep(0.001)
    assert timer.elapsed_ms() > 0.0


def test_memory_profiler_mocked_psutil() -> None:
    with patch("llm_inference_engine.utils.benchmark_utils.psutil.Process") as mock_process_cls:
        process = MagicMock()
        process.memory_info.return_value.rss = 200 * 1024 * 1024
        mock_process_cls.return_value = process
        profiler = MemoryProfiler()
        assert profiler.get_baseline() == 200.0
        assert profiler.get_peak() == 200.0


def test_statistics_calculator() -> None:
    values = [1.0, 2.0, 3.0, 100.0]
    stats = StatisticsCalculator.calculate_statistics(values)
    filtered = StatisticsCalculator.remove_outliers(values)
    percentiles = StatisticsCalculator.calculate_percentiles(values, [50, 95])
    assert stats["mean"] > 0
    assert len(filtered) >= 1
    assert "p50" in percentiles


def test_timer_not_started_returns_zero() -> None:
    """elapsed_ms before entering context should return 0.0."""
    timer = Timer()
    assert timer.elapsed_ms() == 0.0


def test_timer_running_returns_positive() -> None:
    """elapsed_ms called inside context should return a positive value."""
    with Timer() as timer:
        elapsed = timer.elapsed_ms()
    assert elapsed >= 0.0


def test_timer_finished_elapsed_stable() -> None:
    """After context exits, repeated elapsed_ms calls should return same value."""
    with Timer() as timer:
        pass
    first = timer.elapsed_ms()
    second = timer.elapsed_ms()
    assert first == second


def test_memory_profiler_peak_increases() -> None:
    """get_peak should update when current memory exceeds previous peak."""
    with patch("llm_inference_engine.utils.benchmark_utils.psutil.Process") as mock_cls:
        process = MagicMock()
        # First call (baseline): 100 MB, then 200 MB
        process.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_cls.return_value = process
        profiler = MemoryProfiler()

        process.memory_info.return_value.rss = 200 * 1024 * 1024
        peak = profiler.get_peak()
        assert peak == 200.0


def test_memory_profiler_reset() -> None:
    """reset should set peak = baseline = current memory."""
    with patch("llm_inference_engine.utils.benchmark_utils.psutil.Process") as mock_cls:
        process = MagicMock()
        process.memory_info.return_value.rss = 150 * 1024 * 1024
        mock_cls.return_value = process
        profiler = MemoryProfiler()
        profiler.reset()
        assert profiler.get_baseline() == profiler.get_peak()


def test_statistics_calculator_empty_values() -> None:
    stats = StatisticsCalculator.calculate_statistics([])
    assert stats["mean"] == 0.0
    assert stats["std"] == 0.0


def test_statistics_calculator_single_value() -> None:
    stats = StatisticsCalculator.calculate_statistics([42.0])
    assert stats["mean"] == 42.0
    assert stats["min"] == 42.0
    assert stats["max"] == 42.0


def test_remove_outliers_fewer_than_3_returns_same() -> None:
    vals = [1.0, 2.0]
    assert StatisticsCalculator.remove_outliers(vals) is vals


def test_remove_outliers_zero_std_returns_original() -> None:
    """All identical values → std=0 → return original."""
    vals = [5.0, 5.0, 5.0]
    result = StatisticsCalculator.remove_outliers(vals)
    assert result == vals


def test_remove_outliers_removes_extreme_value() -> None:
    # Use a value far enough from the mean to exceed 2 std deviations
    vals = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10000.0]
    result = StatisticsCalculator.remove_outliers(vals)
    assert 10000.0 not in result


def test_calculate_percentiles_empty_returns_zeros() -> None:
    result = StatisticsCalculator.calculate_percentiles([], [50, 95])
    assert result["p50"] == 0.0
    assert result["p95"] == 0.0


def test_calculate_percentiles_p50_is_median() -> None:
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = StatisticsCalculator.calculate_percentiles(vals, [50])
    assert result["p50"] == 3.0


def test_calculate_percentiles_p100_is_max() -> None:
    vals = [1.0, 2.0, 3.0, 10.0]
    result = StatisticsCalculator.calculate_percentiles(vals, [100])
    assert result["p100"] == 10.0
