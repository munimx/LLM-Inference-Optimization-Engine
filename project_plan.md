up # LLM Inference Optimization Engine - Strategic Project Plan

## 1. Project Overview & Objectives

**Goal:** Build a production-grade inference orchestration layer on top of
Ollama, optimizing for throughput and latency on Apple Silicon (M2 Air). **Core
Philosophy:** Leverage Ollama for the "heavy lifting" (model serving,
quantization, hardware acceleration) while building a sophisticated
orchestration layer (batching, scheduling, caching) to maximize system
performance.

### Architectural Strategy

- **Decoupled Architecture:** The optimization engine acts as a proxy/gateway in
  front of Ollama.
- **Orchestration over Implementation:** Focus on _when_ and _how_ to send
  requests (scheduling), rather than _how_ to execute them (inference).
- **Platform Specific:** Tuning batch sizes and concurrency specifically for the
  M2 Air's unified memory architecture.

---

## 2. Phased Execution Plan

### Phase 1: Foundation & Integration (Days 1-4)

- **Objective:** Establish project structure and robust communication with
  Ollama.
- **Key Deliverables:**
  - `OllamaClient`: Robust HTTP client with connection pooling, retries, and
    error handling.
  - `OllamaModelManager`: Service to discover and validate models.
  - Configuration system (`configs/ollama.yaml`, `configs/models.yaml`).
  - Project scaffolding (poetry, linting, logging).
- **Validation:** Successful end-to-end "Hello World" request through the client
  to a local Ollama instance.

### Phase 2: Quantization Analysis (Days 5-8)

- **Objective:** Empirically map Ollama's quantization levels to performance
  metrics to inform scheduling decisions.
- **Key Deliverables:**
  - `QuantizationInfoCollector`: Tool to scrape model metadata.
  - `BenchmarkSuite`: Framework to measure speed (t/s), memory, and quality
    (perplexity/BLEU) across `q4_K_M`, `q8_0`, `fp16`, etc.
  - `QuantizationMapper`: Logic to map high-level user preferences (e.g.,
    "fast", "precise") to specific Ollama model tags.
- **Validation:** A comprehensive report/table correlating quantization levels
  with memory usage and throughput on the target M2 hardware.

### Phase 3: Batching & Scheduling Engine (Days 9-14)

- **Objective:** Implement the core value-add—intelligent request batching to
  maximize throughput.
- **Key Deliverables:**
  - `RequestQueue`: Thread-safe, priority-aware queue.
  - `Scheduler`: Logic to form batches based on policies (FCFS, SJF, Priority,
    Token-Budget).
  - `Batch` abstraction: Tracking token counts and estimated memory footprint.
- **Validation:** Unit tests proving batch formation logic adheres to policies;
  benchmarks showing throughput gain from parallel Ollama calls (simulating
  batching via concurrent requests if native batching isn't exposed, or using
  Ollama's concurrency features).

### Phase 4: Memory & Capacity Planning (Days 15-19)

- **Objective:** Prevent OOMs and optimize concurrency by estimating memory
  usage _before_ execution.
- **Key Deliverables:**
  - `MemoryEstimator`: Service that predicts KV-cache and model weight memory
    usage based on Phase 2 data.
  - Adaptive Throttling: Reject or queue requests when estimated memory exceeds
    M2 limits (e.g., 16GB).
  - Context Awareness: Dynamic calculation of available context window.
- **Validation:** Stress tests that push the system to memory limits without
  crashing, verifying the estimator's accuracy.

### Phase 5: Orchestration & API Layer (Days 20-24)

- **Objective:** Expose the engine via a production-ready REST API.
- **Key Deliverables:**
  - FastAPI Server: Endpoints for `/completions`, `/chat/completions`,
    `/health`.
  - `RequestAggregator`: Logic to fan-out batched requests to Ollama and fan-in
    results.
  - `ResultMapper`: Correlating async responses back to original request IDs.
  - Caching Layer: Semantic caching for identical/similar prompts.
- **Validation:** 100% pass rate on integration tests; working Swagger UI;
  successful handling of concurrent client requests.

### Phase 6: Speculative Decoding (Optional/Advanced) (Days 25-29)

- **Objective:** Further reduce latency using a draft-verifier architecture.
- **Key Deliverables:**
  - `SpeculationEngine`: Logic to coordinate a small "draft" model and large
    "verifier" model.
  - Drafting Logic: Generating candidate tokens with a small model (e.g.,
    Phi-3).
  - Verification Logic: Scoring candidates with the main model.
- **Validation:** Measurable latency reduction (speedup > 1.2x) on standard
  benchmarks compared to baseline.

### Phase 7: Production Hardening (Days 30-33)

- **Objective:** Prepare for deployment and showcase.
- **Key Deliverables:**
  - Comprehensive Documentation (README, Architecture.md, API docs).
  - Final Benchmark Report (Throughput vs. Latency graphs).
  - CI/CD Pipelines (GitHub Actions for test/lint).
  - Code Polish (100% type coverage, strictly linted).
- **Validation:** Peer-ready repository structure and professional documentation
  artifacts.

---

## 3. Architecture & Quality Gates

### Architectural Decisions

1.  **Ollama as Runtime:**
    - _Justification:_ Avoids reinventing the wheel for low-level inference
      (Metal kernels, quantization) which is solved well by llama.cpp/Ollama.
      Allows focus on high-level orchestration.
2.  **Python/FastAPI:**
    - _Justification:_ Standard for ML engineering; async support is crucial for
      handling concurrent I/O-bound requests to Ollama.
3.  **Client-Side Batching (Logical Batching):**
    - _Justification:_ Since we sit _in front_ of Ollama, "batching" primarily
      means managing concurrency and grouping requests to maximize Ollama's
      throughput (Ollama processes requests in parallel if configured).
4.  **Stateless Orchestration:**
    - _Justification:_ Keeps the orchestration layer lightweight; state is
      either in the request (context) or managed by Ollama (model loading).

### Quality Gates (Entry/Exit Criteria)

- **Phase Entry:** Defined by completion of previous phase's critical
  deliverables.
- **Phase Exit:**
  - **Unit Tests:** >80% coverage for new code.
  - **Linting:** Zero errors (mypy strict, ruff).
  - **Benchmarking:** No regression in baseline throughput.
  - **Documentation:** APIs and internal modules documented.

---

## 4. Testing & Validation Strategy

- **Unit Testing:** `pytest` for all logic (Scheduling policies, Memory
  estimation, Request validation).
- **Integration Testing:**
  - Mocked Ollama: Verify orchestration logic without running heavy models.
  - Live Ollama: End-to-end tests ensuring real connectivity and payload
    correctness.
- **Load Testing:** `locust` or custom scripts to simulate concurrent users and
  verify queue stability and scheduler fairness.
- **Memory Profiling:** Monitoring RSS and swap usage during stress tests to
  validate memory estimation logic.

---

## 5. Performance Benchmarking Approach

- **Metrics:** Tokens/sec (Throughput), Time-to-First-Token (Latency), Memory
  Usage (GB), Request Success Rate.
- **Baselines:**
  1.  Direct Ollama API (serial requests).
  2.  Direct Ollama API (naive parallel requests).
- **Scenarios:**
  - Short input / Long output (Generation heavy).
  - Long input / Short output (Prompt processing heavy).
  - Mixed workload (Production simulation).
- **Hardware:** All benchmarks normalized to MacBook Air M2 (16GB).

---

## 6. Documentation Requirements

- **Code:** Google-style docstrings for all functions/classes.
- **API:** OpenAPI (Swagger) auto-generated docs.
- **Architecture:** `docs/ARCHITECTURE.md` with system diagrams.
- **User Guide:** `README.md` with "Quick Start", "Configuration", and
  "Performance Tuning" sections.
- **Dev Guide:** `CONTRIBUTING.md` for setup and testing workflows.
