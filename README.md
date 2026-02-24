# Beyond Vibes

*LLM evaluation framework for testing self-hosted, open source models running on llama.cpp.*

This project provides a framework to evaluate local models and compare them in latency and quality across real-world engineering tasks. While applicable to API providers, the primary focus is benchmarking local model performance under constrained hardware.

## Archetypes

Our use cases naturally fall into four 'archetypes'. These represent the categories of tasks we leverage local models for, serving as the primary lenses for our qualitative evals.

| Archetype | Description | Primary Goal | Examples |
| :--- | :--- | :--- | :--- |
| **Architectural Planning** | High-level design & research | Feasibility & Clarity | Auth plan, Protein folding architecture |
| **Repo Maintenance** | Refactoring & chores | Stability & Cleanliness | Migrating to `uv`, adding unit tests |
| **Feature Implementation** | New functionality code | Correctness & Integration | Implementing Lidarr/Readarr, nvim config |
| **Comparative Research** | Vendor/Tool Analysis | Accuracy & Decision Support | LiteLLM vs Kong, Observability comparisons |

## Methodology

We employ a scientific methodology to benchmark these archetypes. We target specific repositories and define clear success criteria (e.g., a passing test suite, a verified architectural decision).

### Stratification (The "Local" Variables)

Unlike API-based benchmarks, local inference is highly sensitive to runtime configuration. We stratify simulations across:

*   **Engine Config**: `llama.cpp` container variants (ROCm, Vulkan, CUDA) and CLI args (batch size, threads, kv-cache type).
*   **Base Models**: Different HuggingFace model families (GLM, Qwen, Mistral).
*   **Quantization**: Impact of precision loss (Q4_K_M vs Q8_0 vs FP16) on reasoning capabilities.

### Simulations

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

We leverage **DSPy** to compile LLM Judges that score outputs against quality rubrics.

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
    *   **Reproducibility:** Consistency of output across multiple runs.

*   **C. Feature Implementation (The "Engineer" Judge)**
    *   **Idiomatic Check:** Code style alignment (logger usage, error patterns).
    *   **Completeness:** Implementation of all functional requirements (not just the "happy path").

*   **D. Comparative Research (The "Analyst" Judge)**
    *   **Citation Grounding:** Verification that compared features actually exist.
    *   **Decisiveness:** Did it provide a clear recommendation vs. a vague "it depends"?
    *   **Structure:** Adherence to requested formats (e.g., tables, pros/cons lists).


#### Helpful commands:
```bash
# To get into a nix-based development environment with python and uv
nix develop

# To create and activate a virtual environment
uv venv
source .venv/bin/activate

# To install dependencies
uv sync --all-extras
```

## CLI - Model Download

Download models from HuggingFace to S3.

### Prerequisites

- S3 bucket must exist before running
- Valid HuggingFace model repo

### Setup

1. **Create `.env` file:**
```bash
S3_BUCKET=your-bucket
S3_ENDPOINT=https://s3.example.com
S3_ACCESS_KEY=your-access-key
S3_SECRET_KEY=your-secret-key
```

2. **Create `models.yaml` config:**
```yaml
bucket: your-bucket
models:
  - name: model-name
    repo_id: namespace/model-repo
    quant_tags: ["Q4_K_M", "Q8_0"]
```

### Run

```bash
# Dry run (preview only)
uv run beyond-vibes download --config-path models.yaml --dry-run

# Actual download
uv run beyond-vibes download
```

## CLI - Simulations

Run simulations by cloning a repo and executing a prompt via OpenCode.

### Prerequisites

- OpenCode server running (default: http://127.0.0.1:4096)
- MLflow tracking server configured (optional, for logging)
- Model defined in `models.yaml`

### Setup

1. **Create `.env` file (if not already done):**
```bash
MLFLOW_TRACKING_URI=https://mlflow.example.com
```

2. **Ensure `models.yaml` has your model:**
```yaml
bucket: beyond-vibes
models:
  - name: minimax-m2.5-free
    repo_id: opencode/minimax-m2.5-free
    quant_tags: []
```

### Run

```bash
# Run simulation with a model from models.yaml
uv run beyond-vibes simulate --task poetry_to_uv --model minimax-m2.5-free

# With custom config
uv run beyond-vibes simulate --task poetry_to_uv --model qwen3-0.6B --config-path mymodels.yaml

# With custom prompt variables
uv run beyond-vibes simulate --task auth_plan --model minimax-m2.5-free --prompt-vars '{"requirements": "OAuth2"}'
```

### Options

| Option | Description |
|--------|-------------|
| `--task` | Task name (without .yaml) - required |
| `--model` | Model name from models.yaml - required |
| `--config-path` | Path to models.yaml (default: models.yaml) |
| `--prompt-vars` | JSON dict of variables for prompt templating (default: {}) |
