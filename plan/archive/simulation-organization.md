# Simulation Run Organization

## Overview

This document describes the MLflow run hierarchy and organization strategy for Beyond Vibes simulations.

## Architecture

### Hierarchical Run Structure

We use MLflow's native parent-child relationship to organize simulation data:

```
Experiment: "beyond-vibes"
├── Parent Run: "{model_name}-{quant_tag}-{container}"
│   ├── Parameters:
│   │   - model.name
│   │   - model.quant
│   │   - runtime.container
│   ├── Tags:
│   │   - eval.batch_id
│   │   - model.config_id
│   ├── Child Run 1: "{task_name}"
│   │   ├── Parameters:
│   │   │   - model.name
│   │   │   - model.quant
│   │   │   - runtime.container
│   │   ├── Tags:
│   │   │   - task.name
│   │   │   - task.archetype
│   │   ├── Metrics:
│   │   │   - quality_score (from LLM judge)
│   │   │   - latency_seconds
│   │   │   - tokens_per_second
│   │   │   - total_messages
│   │   └── Artifacts:
│   │       - git_diff.patch
│   │       - conversation.json
│   ├── Child Run 2: "{task_name}"
│   └── ...
```

### Why This Structure?

**Granular Retry Capability**
- Each eval task runs as an independent child run
- Failed tasks can be retried individually without restarting the entire batch
- Enables efficient debugging of task-specific issues

**Comprehensive Comparison**
- Parent runs serve as the comparison unit across model configurations
- Compare "Qwen3-8b-Q4_K_M-rocm" vs "Mistral-7b-Q8_0-cuda" in MLflow UI
- Parameters enable correlation analysis between configuration and performance

**Parallel Execution**
- Child runs are independent, enabling parallel execution on multiple GPUs
- Argo Workflows can launch all tasks simultaneously
- No contention for shared run state

**Clean Debugging**
- Each task's MLflow traces are isolated in separate runs
- Easier to analyze conversation flow and failures per-task

## Execution Workflow

### Batch Orchestration

1. **Batch Initialization**
   - Generate unique `eval.batch_id` (timestamp-based, e.g., "eval-2024-01-15-143052")
   - Create parent run for each unique model configuration

2. **Parallel Task Execution**
   - Launch all eval tasks across all model configurations in parallel
   - Each task creates a child run under its parent
   - Task isolation prevents cross-contamination of logs and traces

3. **Post-Processing**
   - Run LLM-as-judge on each child run's artifacts
   - Log quality scores back to respective child runs
   - Compute aggregate metrics on parent run

### Argo Workflows Integration

```yaml
# Simplified workflow structure
spec:
  templates:
    - name: run-eval-batch
      steps:
        - - name: parent-run-creation
            template: create-parent-runs
        - - name: run-tasks
            template: run-task
            withItems:
              - {task: poetry_to_uv, model: qwen3-8b, quant: Q4_K_M, container: rocm}
              - {task: unit_tests, model: qwen3-8b, quant: Q4_K_M, container: rocm}
              # ... all combinations
```

## Analysis Capabilities

### High-Level Model Comparison

Compare parent runs to answer:
- Which model configuration has the best overall quality?
- Which has the lowest average latency?
- What's the quality/latency trade-off per configuration?

### Task-Specific Analysis

Query child runs to generate:
- Task-specific leaderboards (which model wins at unit_tests?)
- Heatmaps of model × task performance
- Archetype-based aggregation (which models excel at repo_maintenance?)

### Dashboard Queries

```python
# Overall model ranking
parent_runs = mlflow.search_runs(
    filter_string="tags.mlflow.parentRunId = ''",
    order_by=["metrics.avg_quality_score DESC"]
)

# Task heatmap
child_runs = mlflow.search_runs(
    filter_string=f"tags.eval.batch_id = '{batch_id}'"
)
# Pivot: rows=task_name, cols=model_config, values=quality_score
```

## Key Design Decisions

1. **Task-level granularity**: Despite the overhead of multiple runs, per-task retry and debugging capabilities are essential

2. **Parent as comparison unit**: Model configurations (not tasks) are the primary comparison dimension

3. **Batch tagging**: All runs in a single eval sweep share a `eval.batch_id` tag for cross-referencing

4. **Container as parameter**: Runtime configuration is tracked as parameters to enable performance correlation analysis

## Future Considerations

- **Cross-batch trending**: Compare quality scores across multiple eval batches to track model improvements over time
- **A/B testing**: Use parent runs to compare specific model versions (e.g., quantized vs FP16)
- **Automatic aggregation**: Compute parent-level metrics (avg quality, success rate) after all children complete
