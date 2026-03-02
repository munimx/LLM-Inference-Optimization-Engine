"""Memory estimation for LLM inference workloads.

Predicts peak memory usage for a given model and request configuration
by combining model weight size with KV-cache overhead estimates.  The
estimates are intentionally conservative so that the
:class:`~llm_inference_engine.optimization.throttler.AdaptiveThrottler`
can safely gate requests before OOM conditions occur.
"""


import structlog

from llm_inference_engine.quantization.types import QuantizationLevel

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Bytes-per-parameter constants by quantization level.
# These are empirically derived from llama.cpp / Ollama model data.
# ---------------------------------------------------------------------------
_BYTES_PER_PARAM: dict[QuantizationLevel, float] = {
    QuantizationLevel.Q2_K: 0.313,
    QuantizationLevel.Q3_K: 0.375,
    QuantizationLevel.Q4_0: 0.5,
    QuantizationLevel.Q4_1: 0.5625,
    QuantizationLevel.Q4_K_S: 0.5,
    QuantizationLevel.Q4_K_M: 0.5625,
    QuantizationLevel.Q5_K_S: 0.625,
    QuantizationLevel.Q5_K_M: 0.6875,
    QuantizationLevel.Q6_K: 0.75,
    QuantizationLevel.Q8_0: 1.0,
    QuantizationLevel.FP16: 2.0,
    QuantizationLevel.UNKNOWN: 1.0,  # conservative default
}

# Safety margin applied on top of raw estimates (10 %).
_SAFETY_MARGIN: float = 1.1

# KV-cache bytes per token per layer (two half-precision floats for K and V,
# each multiplied by the number of heads × head dimension — but we use a
# simplified per-token-per-layer figure that matches observed Ollama behaviour).
_KV_CACHE_BYTES_PER_TOKEN_PER_LAYER: float = 512.0


class MemoryEstimator:
    """Estimates peak memory usage for LLM inference requests.

    Usage::

        estimator = MemoryEstimator()
        weights_gb = estimator.estimate_model_weights_gb(
            num_parameters=7_000_000_000,
            quantization=QuantizationLevel.Q4_K_M,
        )
        kv_gb = estimator.estimate_kv_cache_gb(
            num_tokens=2048,
            num_layers=32,
        )
        total_gb = estimator.estimate_total_gb(
            num_parameters=7_000_000_000,
            quantization=QuantizationLevel.Q4_K_M,
            num_tokens=2048,
            num_layers=32,
        )
    """

    def __init__(self, safety_margin: float = _SAFETY_MARGIN) -> None:
        """Initialise the estimator.

        Args:
            safety_margin: Multiplicative safety margin applied to all
                estimates (default 1.1 = +10 %).
        """
        if safety_margin < 1.0:
            raise ValueError("safety_margin must be >= 1.0")
        self._safety_margin = safety_margin

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_model_weights_gb(
        self,
        num_parameters: int,
        quantization: QuantizationLevel = QuantizationLevel.Q4_K_M,
    ) -> float:
        """Estimate the GPU/unified-memory cost of loading model weights.

        Args:
            num_parameters: Total number of model parameters (e.g.
                ``7_000_000_000`` for a 7B model).
            quantization: The quantization level of the model.

        Returns:
            Estimated memory in gigabytes (including safety margin).
        """
        if num_parameters <= 0:
            raise ValueError("num_parameters must be positive")
        bytes_per_param = _BYTES_PER_PARAM.get(quantization, 1.0)
        raw_bytes = num_parameters * bytes_per_param
        gb = (raw_bytes * self._safety_margin) / (1024**3)
        logger.debug(
            "model_weights_estimated",
            num_parameters=num_parameters,
            quantization=quantization,
            estimated_gb=round(gb, 3),
        )
        return gb

    def estimate_kv_cache_gb(
        self,
        num_tokens: int,
        num_layers: int,
        bytes_per_token_per_layer: float = _KV_CACHE_BYTES_PER_TOKEN_PER_LAYER,
    ) -> float:
        """Estimate KV-cache memory for a given context length.

        Args:
            num_tokens: Number of tokens in the context (input + output).
            num_layers: Number of transformer layers in the model.
            bytes_per_token_per_layer: Bytes consumed per token per layer
                for the KV-cache (default 512 B, a conservative estimate
                for fp16 KV heads).

        Returns:
            Estimated KV-cache size in gigabytes (including safety margin).
        """
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        raw_bytes = num_tokens * num_layers * bytes_per_token_per_layer
        gb = (raw_bytes * self._safety_margin) / (1024**3)
        logger.debug(
            "kv_cache_estimated",
            num_tokens=num_tokens,
            num_layers=num_layers,
            estimated_gb=round(gb, 3),
        )
        return gb

    def estimate_total_gb(
        self,
        num_parameters: int,
        quantization: QuantizationLevel,
        num_tokens: int,
        num_layers: int,
    ) -> float:
        """Estimate total peak memory (weights + KV-cache).

        Args:
            num_parameters: Total model parameter count.
            quantization: Quantization level.
            num_tokens: Context length in tokens.
            num_layers: Number of transformer layers.

        Returns:
            Total estimated memory in gigabytes.
        """
        weights_gb = self.estimate_model_weights_gb(num_parameters, quantization)
        kv_gb = self.estimate_kv_cache_gb(num_tokens, num_layers)
        total = weights_gb + kv_gb
        logger.debug(
            "total_memory_estimated",
            weights_gb=round(weights_gb, 3),
            kv_cache_gb=round(kv_gb, 3),
            total_gb=round(total, 3),
        )
        return total

    @staticmethod
    def infer_num_layers(num_parameters: int) -> int:
        """Infer a reasonable transformer depth from parameter count.

        Uses a heuristic based on standard open-source model configs:

        - <1B parameters  → 12 layers
        - <4B parameters  → 28 layers
        - <8B parameters  → 32 layers
        - <14B parameters → 40 layers
        - <34B parameters → 48 layers
        - ≥34B parameters → 60 layers

        Args:
            num_parameters: Total parameter count.

        Returns:
            Estimated number of transformer layers.
        """
        thresholds: list[tuple[int, int]] = [
            (1_000_000_000, 12),
            (4_000_000_000, 28),
            (8_000_000_000, 32),
            (14_000_000_000, 40),
            (34_000_000_000, 48),
        ]
        for threshold, layers in thresholds:
            if num_parameters < threshold:
                return layers
        return 60

    @staticmethod
    def bytes_per_param(quantization: QuantizationLevel) -> float:
        """Return the bytes-per-parameter constant for *quantization*.

        Args:
            quantization: The quantization level to query.

        Returns:
            Bytes per model parameter.
        """
        return _BYTES_PER_PARAM.get(quantization, 1.0)


__all__ = ["MemoryEstimator"]
