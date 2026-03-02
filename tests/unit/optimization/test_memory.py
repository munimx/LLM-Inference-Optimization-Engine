"""Unit tests for MemoryEstimator."""

import pytest

from llm_inference_engine.optimization.memory import MemoryEstimator
from llm_inference_engine.quantization.types import QuantizationLevel


class TestMemoryEstimator:
    """Tests for MemoryEstimator."""

    def test_default_safety_margin(self) -> None:
        """Default safety margin should be 1.1."""
        est = MemoryEstimator()
        assert est._safety_margin == 1.1  # noqa: SLF001

    def test_invalid_safety_margin(self) -> None:
        """Safety margin below 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="safety_margin"):
            MemoryEstimator(safety_margin=0.5)

    def test_model_weights_fp16_7b(self) -> None:
        """FP16 7B model should require ~14.6 GB (with safety margin)."""
        est = MemoryEstimator()
        gb = est.estimate_model_weights_gb(7_000_000_000, QuantizationLevel.FP16)
        # 7B * 2 bytes * 1.1 / (1024^3) ≈ 14.3 GB
        assert 13.0 < gb < 16.0

    def test_model_weights_q4_km_7b(self) -> None:
        """Q4_K_M 7B model should need roughly half of FP16."""
        est = MemoryEstimator()
        q4_gb = est.estimate_model_weights_gb(7_000_000_000, QuantizationLevel.Q4_K_M)
        fp16_gb = est.estimate_model_weights_gb(7_000_000_000, QuantizationLevel.FP16)
        assert q4_gb < fp16_gb

    def test_model_weights_invalid_params(self) -> None:
        """Non-positive num_parameters should raise ValueError."""
        est = MemoryEstimator()
        with pytest.raises(ValueError, match="num_parameters"):
            est.estimate_model_weights_gb(0, QuantizationLevel.Q4_K_M)

    def test_kv_cache_basic(self) -> None:
        """KV cache estimate should be positive and scale with tokens."""
        est = MemoryEstimator()
        gb_1k = est.estimate_kv_cache_gb(1_000, 32)
        gb_2k = est.estimate_kv_cache_gb(2_000, 32)
        assert gb_1k > 0
        assert gb_2k == pytest.approx(gb_1k * 2, rel=0.01)

    def test_kv_cache_invalid_tokens(self) -> None:
        """Non-positive num_tokens should raise ValueError."""
        est = MemoryEstimator()
        with pytest.raises(ValueError, match="num_tokens"):
            est.estimate_kv_cache_gb(0, 32)

    def test_kv_cache_invalid_layers(self) -> None:
        """Non-positive num_layers should raise ValueError."""
        est = MemoryEstimator()
        with pytest.raises(ValueError, match="num_layers"):
            est.estimate_kv_cache_gb(1024, 0)

    def test_estimate_total_is_sum(self) -> None:
        """Total estimate should equal weights + kv_cache."""
        est = MemoryEstimator()
        weights = est.estimate_model_weights_gb(7_000_000_000, QuantizationLevel.Q4_K_M)
        kv = est.estimate_kv_cache_gb(2048, 32)
        total = est.estimate_total_gb(7_000_000_000, QuantizationLevel.Q4_K_M, 2048, 32)
        assert total == pytest.approx(weights + kv, rel=0.001)

    def test_infer_num_layers(self) -> None:
        """infer_num_layers should return sensible values for common model sizes."""
        assert MemoryEstimator.infer_num_layers(500_000_000) == 12
        assert MemoryEstimator.infer_num_layers(3_000_000_000) == 28
        assert MemoryEstimator.infer_num_layers(7_000_000_000) == 32
        assert MemoryEstimator.infer_num_layers(13_000_000_000) == 40
        assert MemoryEstimator.infer_num_layers(33_000_000_000) == 48
        assert MemoryEstimator.infer_num_layers(70_000_000_000) == 60

    def test_bytes_per_param_known_levels(self) -> None:
        """bytes_per_param should return correct values for all known quant levels."""
        assert MemoryEstimator.bytes_per_param(QuantizationLevel.FP16) == 2.0
        assert MemoryEstimator.bytes_per_param(QuantizationLevel.Q8_0) == 1.0
        assert MemoryEstimator.bytes_per_param(QuantizationLevel.Q4_K_M) == 0.5625

    def test_smaller_safety_margin_gives_smaller_estimate(self) -> None:
        """A smaller safety margin should yield a smaller estimate."""
        est_tight = MemoryEstimator(safety_margin=1.0)
        est_safe = MemoryEstimator(safety_margin=1.2)
        tight = est_tight.estimate_model_weights_gb(7_000_000_000, QuantizationLevel.Q8_0)
        safe = est_safe.estimate_model_weights_gb(7_000_000_000, QuantizationLevel.Q8_0)
        assert tight < safe

    def test_all_quantization_levels_produce_positive_estimate(self) -> None:
        """All known quantization levels should produce a positive weight estimate."""
        est = MemoryEstimator()
        for level in QuantizationLevel:
            gb = est.estimate_model_weights_gb(7_000_000_000, level)
            assert gb > 0, f"Expected positive estimate for {level}"
