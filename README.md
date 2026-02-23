# Beyond Vibes

This project provides a framework to evaluate local models and compare them in latency, and quality across real-world engineering tasks. While applicable to API providers, the primary focus is benchmarking local model performance under constrained hardware.

## Archetypes

Our use cases naturally fall into four 'archetypes'. These represent the categories of tasks we leverage local models for, serving as the primary lenses for our qualitative evals.

| Archetype | Description | Primary Goal | Examples |
| :--- | :--- | :--- | :--- |
| **Architectural Planning** | High-level design & research | Feasibility & Clarity | Auth plan, Protein folding architecture |
| **Repo Maintenance** | Refactoring & chores | Stability & Cleanliness | Migrating to `uv`, adding unit tests |
| **Feature Implementation** | New functionality code | Correctness & Integration | Implementing Lidarr/Readarr, nvim config |
| **Comparative Research** | Vendor/Tool Analysis | Accuracy & Decision Support | LiteLLM vs Kong, Observability comparisons |

## Methodology

We employ a scientific methodology to benchmark these archetypes. We target specific repositories and define a "Golden Trajectory"—a set of instructions with a known, optimal outcome (e.g., a passing test suite or a verified architectural decision).

### Stratification (The "Local" Variables)

Unlike API-based benchmarks, local inference is highly sensitive to runtime configuration. We stratify simulations across:

*   **Engine Config**: `llama.cpp` container variants (ROCm, Vulkan, CUDA) and CLI args (batch size, threads, kv-cache type).
*   **Model Weights**: Different HuggingFace model families (GLM, Qwen, Mistral).
*   **Quantization**: Impact of precision loss (Q4_K_M vs Q8_0 vs FP16) on reasoning capabilities.

### Simulations

To create our **Golden Dataset**, we first execute tasks manually with a Human-in-the-Loop (HITL) approach using a high-intelligence model (e.g., Claude 4.5 Opus, Kimi K2.5). Once a clean execution path is verified, we replay these tasks autonomously using **OpenCode** against our local models.

**Current Simulation Tasks:**

*   **Architectural Planning**
    *   Research and architect a login/auth plan for [lighthearted](https://github.com/blake-hamm/lighthearted)
    *   Create a high-level architecture/plan for overnight protein folding in [bhamm-lab](https://github.com/blake-hamm/bhamm-lab)
    *   Migration plan: Ceph CSI vs. Rook Ceph in [bhamm-lab](https://github.com/blake-hamm/bhamm-lab)

*   **Repo Maintenance**
    *   Switch from poetry to `uv` for [lighthearted](https://github.com/blake-hamm/lighthearted)
    *   Setup tests in [kube-ai-stack](https://github.com/blake-hamm/kube-ai-stack)
    *   Write unit tests for [lighthearted](https://github.com/blake-hamm/lighthearted)

*   **Feature Implementation**
    *   Create FastAPI endpoints and decouple from fronted in [lighthearted](https://github.com/blake-hamm/lighthearted)
    *   Implement Lidarr and Readarr in [bhamm-lab](https://github.com/blake-hamm/bhamm-lab)
    *   Implement nvim with nvf in [bhamm-lab](https://github.com/blake-hamm/bhamm-lab)

*   **Comparative Research**
    *   Compare LiteLLM and Kong AI Gateway and provide a recommendation
    *   Compare Arize Phoenix, MLflow, and LangFuse for GenAI observability

### Evals

We leverage **DSPy** to compile and optimize "LLM Judges." By using our Golden Dataset as a training set, we optimize the judge's prompts to ensure they align with human expert preference before running them at scale.

The evaluation framework has two tiers:

#### 1. Universal Evals
*Baseline health metrics applied to every run regardless of archetype.*

*   **Instruction Adherence (The "Vibe" Check)**
    *   **Sycophancy Score (1-5):** Did the model apologize excessively or agree with a bad premise?
    *   **Refusal Rate:** Did it falsely refuse a safe coding task?
    *   **Efficiency:** Total turns/steps to solve (crucial for slow local inference).
*   **Tool Use Quality**
    *   **Hallucination Rate:** Frequency of non-existent tool calls.
    *   **Schema Adherence:** Frequency of JSON formatting errors.
    *   **Loop Detection:** Did the agent get stuck in a read/list loop?
*   **System Performance**
    *   **Throughput:** Tokens Per Second (TPS) generation.
    *   **Latency:** Time to First Token (TTFT) and Prompt Processing.
    *   **Resource Cost:** Peak VRAM usage and model load time.

#### 2. Category-Specific Evals

*   **A. Architectural Planning (The "Design" Judge)**
    *   **Specificity:** Does the plan cite specific files/APIs, or generic concepts?
    *   **Security:** Checks for hardcoded secrets or insecure defaults.
    *   **Constraints:** Adherence to "local-only" or "no-cost" requirements.

*   **B. Repo Maintenance (The "Janitor" Judge)**
    *   **Determinisim:** Binary Pass/Fail on `uv sync`, `pytest`, or build commands.
    *   **Diff Hygiene:** Penalties for modifying unrelated files or formatting changes.
    *   **Idempotency:** Consistency of output across multiple runs.

*   **C. Feature Implementation (The "Engineer" Judge)**
    *   **Idiomatic Check:** Code style alignment (logger usage, error patterns).
    *   **Completeness:** Implementation of all functional requirements (not just the "happy path").

*   **D. Comparative Research (The "Analyst" Judge)**
    *   **Citation Grounding:** Verification that compared features actually exist.
    *   **Decisiveness:** Did it provide a clear recommendation vs. a vague "it depends"?
    *   **Structure:** Adherence to requested formats (e.g., tables, pros/cons lists).
