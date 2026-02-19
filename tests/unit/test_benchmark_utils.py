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
