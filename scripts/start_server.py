#!/usr/bin/env python3
"""CLI script to start the LLM Inference Optimization Engine API server."""

import argparse
import sys

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the LLM Inference Optimization Engine API server"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Starting server on {args.host}:{args.port} ...")
    uvicorn.run(
        "llm_inference_engine.api.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
