from pathlib import Path

import pytest

from llm_inference_engine.config import InferenceConfig
from llm_inference_engine.exceptions import ConfigurationError


def test_invalid_yaml(tmp_path: Path) -> None:
    p = tmp_path / "invalid.yaml"
    p.write_text("invalid: [ yaml: {")

    with pytest.raises(ConfigurationError, match="Invalid YAML"):
        InferenceConfig.from_yaml(p)

def test_not_a_dict(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- item1\n- item2")

    with pytest.raises(ConfigurationError, match="Configuration must be a dictionary"):
        InferenceConfig.from_yaml(p)

def test_type_error(tmp_path: Path) -> None:
    p = tmp_path / "bad_type.yaml"
    # unexpected argument for vllm config
    p.write_text("vllm:\n  unknown_field: 123")

    with pytest.raises(ConfigurationError, match="Invalid vllm configuration"):
        InferenceConfig.from_yaml(p)
