# Beyond Vibes

This project provides a framework to evaluate local models and compare them in latency and quality across various tasks. Of course, this can also be used to test model api vendors, but the focus is local models with unique requirements.


## Archetypes

Our use cases naturally into four 'archetypes'. These represent the categories of tasks we would leverage local models for. These categories are also going to be the primary focus of our qualitative evals.

| Archetype | Description | Primary Goal | Examples |
|-|-|-|-|
| Architectural Planning | High-level design & research | Feasibility & Clarity | Auth plan for lighthearted, Protein folding plan, Rook Ceph migration |
| Repo Maintenance | Refactoring & chores | Stability & Cleanliness | Poetry to uv, Add tests to kube-ai-stack, Orphan script tests |
| Feature Implementation | New functionality code | Correctness & Integration | Lidarr/Readarr implementation, nvim with nvf |
| Comparative Research | Vendor/Tool Analysis | Accuracy & Decision Support | LiteLLM vs Kong, Observability tools comparison |

## Methodology

In order to test these use cases, we plan a 'scientific' methodology. First off, we setup our experiment by targeting specific repos and a task for each use case. We use OpenCode and provide instructions with a desired outcome in mind. This establishes a 'golden dataset' with pre-defined inputs and desired outputs.

### Stratification

Each simulation is run across multiple configurations to understand how different factors affect model performance. This stratification covers:

- **Backend**: llama.cpp container images (ROCm, Vulkan, etc.)
- **Server Args**: Various server configurations (kv quant, batch size)
- **Models**: Different HuggingFace models
- **Quantizations**: Different quantization levels for each model

### Simulations

To form the golden outputs, we manually walk through the tasks (inputs) with a human-in-the-loop and a more powerful model like Opus 4.6. Once we are satisfied with the desired outcome of our tasks after some hand-holding, we ensure the same tasks can be executed autonomously using OpenCode. Here are the tasks we execute:

- Write unit tests for [lighthearted](https://github.com/blake-hamm/lighthearted)
- Switch from poetry to uv for [lighthearted](https://github.com/blake-hamm/lighthearted)
- Research and architect a login/auth plan for for [lighthearted](https://github.com/blake-hamm/lighthearted)
- Research and compare the current implementation of ceph and usage of the csi plugin to using rook ceph in [bhamm-lab](https://github.com/blake-hamm/bhamm-lab)
- Setup unit tests for ceph orphan python script in [bhamm-lab](https://github.com/blake-hamm/bhamm-lab)
- Implement Lidarr and Redarr in [bhamm-lab](https://github.com/blake-hamm/bhamm-lab)
- Implement nvim with nvf in [bhamm-lab](https://github.com/blake-hamm/bhamm-lab)
- Create a high level architecture and plan to enable overnight protein folding in [bhamm-lab](https://github.com/blake-hamm/bhamm-lab)
- Setup tests in [kube-ai-stack](https://github.com/blake-hamm/kube-ai-stack)
- Compare LiteLLM and Kong AI Gateway and provide a recommendation
- Compare Arize Phoenix, MLflow and LangFuse for GenAI observability and experimentation


### Evals

Now that we have our dataset, we will need a way to evaluate different outcomes. To do this, we use DSPy and create an LLM judge. We can use the golden dataset to optimize the prompt for this judge to ensure it correctly judges our outcomes.

The evaluation framework has two tiers: **Universal Evals** (apply to all archetypes) and **Category-Specific Evals** (tailored to each archetype).

#### Universal Evals

These are baseline health metrics. Every run is scored on these regardless of task type.

**Instruction Adherence (The "Vibe" Check):**
- Sycophancy Score (1-5): Did the model apologize excessively or agree with a bad premise?
- Refusal Rate: Did it falsely refuse a safe coding task?
- Efficiency: Number of turns/steps to solve (crucial for local models where inference is slow/expensive)

**Tool Use Quality:**
- Hallucinated Tool Calls: Did it try to call a tool that doesn't exist?
- Argument Formatting: How often did it fail JSON schema validation for tool arguments?
- Loop Detection: Did it get stuck calling ls or read_file on the same file 3+ times?

**Performance Metrics:**
- Time to First Token (TTFT): How quickly did generation start?
- Prompt Processing Speed: Time to process the input prompt
- Tokens Per Second (TPS): Generation throughput
- Total Completion Time: Start to finish for entire task
- Model Load Time: Time to load model into memory (relevant for zero-scale architectures)

#### Category-Specific Evals

**A. Architectural Planning (The "Design" Judge)**
- Specificity Score: Does the plan mention specific filenames, libraries, or API endpoints?
- Security Check: Does the plan suggest insecure defaults (e.g., storing secrets in plain text)?
- Constraint Satisfaction: Did it respect constraints like "local-only" or "no paid APIs"?

**B. Repo Maintenance (The "Janitor" Judge)**
- Deterministic - Build/Test Pass: Did `uv sync` or `pytest` pass after the change? (Binary 0/1)
- Diff Hygiene: Did it change unrelated files?
- Idempotency: If you run the maintenance task twice, does it result in the same state?

**C. Feature Implementation (The "Engineer" Judge)**
- Idiomatic Check: Does the new code match existing style (logger, error handling patterns)?
- Implementation Completeness: Did it implement all requirements?

**D. Comparative Research (The "Analyst" Judge)**
- Citation Grounding: Are the features it compares actually real?
- Decisiveness: Did it make a recommendation as requested, or just say "it depends"?
- Structure: Did it follow the requested format (e.g., "Pros/Cons table")?
