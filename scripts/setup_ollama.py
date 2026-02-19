#!/usr/bin/env python3
"""Setup verification script for Ollama integration."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_inference_engine.config import load_config
from llm_inference_engine.integration import OllamaClient, OllamaModelManager


async def verify_ollama_setup() -> bool:
    """Verify Ollama is properly set up.

    Returns:
        True if setup is valid, False otherwise
    """
    print("=" * 60)
    print("LLM Inference Engine - Ollama Setup Verification")
    print("=" * 60)
    print()

    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    try:
        config = load_config(config_path)
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False

    # Create Ollama client
    client = OllamaClient(
        host=config.ollama.host,
        port=config.ollama.port,
        timeout=config.ollama.timeout_seconds,
        max_retries=config.ollama.retry_count,
    )

    print(f"✓ Ollama client initialized (URL: http://{config.ollama.host}:{config.ollama.port})")
    print()

    async with client:
        # Check if Ollama is available
        print("Checking Ollama service availability...")
        is_available = await client.is_available()

        if not is_available:
            print("✗ Ollama service is not reachable")
            print()
            print("Please ensure Ollama is running:")
            print("  1. Download Ollama from https://ollama.ai/")
            print("  2. Start Ollama service")
            print("  3. Run: ollama serve")
            return False

        print("✓ Ollama service is reachable")
        print()

        # List available models
        print("Fetching available models...")
        try:
            models = await client.list_models()
            print(f"✓ Found {len(models)} model(s)")
            print()

            if len(models) == 0:
                print("⚠ No models available. Please pull some models:")
                print("  ollama pull mistral")
                print("  ollama pull llama2")
                print("  ollama pull phi3")
                print()
            else:
                print("Available models:")
                for model in models:
                    name = model.get("name", "unknown")
                    size = model.get("size", 0)
                    size_gb = size / (1024**3)
                    print(f"  • {name} ({size_gb:.2f} GB)")
                print()

        except Exception as e:
            print(f"✗ Failed to list models: {e}")
            return False

        # Verify model manager
        print("Verifying model manager...")
        try:
            model_manager = OllamaModelManager(client)
            await model_manager.refresh_models()
            available_models = await model_manager.get_available_models()
            print(f"✓ Model manager working ({len(available_models)} models)")
            print()

        except Exception as e:
            print(f"✗ Model manager failed: {e}")
            return False

        # Check configured models
        print("Checking configured models...")
        required_models = ["mistral", "llama2", "phi3"]
        missing_models = []

        for model_name in required_models:
            model_config = config.get_model_config(model_name)
            if model_config:
                actual_name = model_config.name
                is_available_check = await model_manager.verify_model_available(actual_name)

                if is_available_check:
                    print(f"✓ {model_name}: {actual_name}")
                else:
                    print(f"✗ {model_name}: {actual_name} (not available)")
                    missing_models.append(actual_name)
            else:
                print(f"⚠ {model_name}: not configured")

        print()

        if missing_models:
            print("⚠ Some configured models are missing. To install:")
            for model in missing_models:
                print(f"  ollama pull {model}")
            print()

        # Health check
        print("Performing health check...")
        health = await client.health_check()
        status = health.get("status")

        if status == "healthy":
            print("✓ Health check passed")
            print(f"  Base URL: {health.get('base_url')}")
            print(f"  Models available: {health.get('models_available')}")
            print()
        else:
            print(f"✗ Health check failed: {health.get('message')}")
            return False

    print("=" * 60)
    print("Setup verification complete!")
    print("=" * 60)
    print()

    if len(models) == 0:
        print("⚠ Warning: No models available")
        print("  Please pull models before proceeding")
        return False

    print("✓ Ready to proceed with Phase 1 implementation")
    return True


async def main() -> None:
    """Main entry point."""
    try:
        success = await verify_ollama_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nVerification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
