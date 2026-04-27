# Dataset Curation Plan

## Overview

This document outlines the architecture for the dataset curation phase of the Beyond Vibes evaluation pipeline. The curation step transforms raw MLflow traces from simulation runs into structured datasets optimized for LLM judge evaluation.

## Architecture Goals

- Extract meaningful evaluation signals from multiturn simulation traces without storing full conversation history
- Support both universal metrics (applicable to all archetypes) and archetype-specific analysis
- Leverage MLflow's native Dataset format for seamless integration with MLflow evaluation features
- Enable filtering and versioning of curated datasets
- Tie curated datasets directly to their source simulation runs as artifacts

## Data Flow

```
┌─────────────────┐
│  MLflow Traces  │ (Existing simulation runs with spans)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Query & Filter │ (By archetype, status, date range, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Extract &       │ (Summary stats from traces, input/output content)
│ Transform       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Build Dataset  │ (MLflow Dataset with schema)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Log Artifact   │ (Attached to source simulation run)
└─────────────────┘
```

## Dataset Schema Design

### Universal Schema (All Archetypes)

All curated datasets include these baseline fields:

**Identifiers**
- Run ID, model name, quantization tag
- Task name, archetype classification
- Dataset version timestamp

**Content**
- Input prompt (reconstructed from task configuration)
- Final assistant output (last message text content)
- Git diff (when available, null for non-code tasks)

**Summary Statistics** (derived from trace metrics)
- Total message count (conversation length)
- Total tool calls and tool error count
- Token usage and cost
- Wall clock duration

**Status**
- Run completion status (success, error, aborted)
- Flags for data availability (has_git_diff, has_output)

### Archetype-Specific Extensions

Beyond the universal schema, each archetype may define extended fields:

- **Repository Maintenance**: Diff statistics (file count, line changes), test results
- **Feature Implementation**: Implementation completeness flags, dependency analysis
- **Architectural Planning**: Recommendation confidence, cited sources count
- **Comparative Research**: Comparison structure adherence, decisiveness score

## CLI Interface

```bash
beyond-vibes curate \
  --experiment beyond-vibes \
  --dataset-name my-evaluation-set \
  [--archetype repo_maintenance] \
  [--status success] \
  [--date-from YYYY-MM-DD] \
  [--date-to YYYY-MM-DD] \
  [--min-tool-calls N]
```

### Filtering Options

The curate command supports filtering simulation runs by:
- Archetype (optional, creates archetype-specific dataset)
- Run status (success, error, or both)
- Date range (simulation execution date)
- Minimum tool call threshold (for loop detection analysis)
- Git diff availability (for code-focused tasks)

### Dataset Naming

Datasets are versioned with timestamps:
- Base name provided via `--dataset-name`
- Auto-append timestamp: `{name}_YYYYMMDD_HHMMSS`
- Logged as MLflow artifact with version metadata

## Implementation Phases

### Phase 1: Core Curation Pipeline

**Goal**: Implement the `curate` CLI command with universal schema support.

**Action Items**:
1. Create curation module structure
2. Implement MLflow run querying with filter support
3. Build trace extractor to pull summary stats from spans
4. Reconstruct input prompts from task configurations
5. Create MLflow Dataset builder with universal schema
6. Implement artifact logging to simulation runs
7. Add CLI integration to main entry point

**Success Criteria**:
- Can execute `beyond-vibes curate` with various filters
- Produces MLflow Dataset artifacts attached to runs
- Dataset includes all universal schema fields
- Versioning works with timestamps

### Phase 2: Archetype-Specific Curation

**Goal**: Extend curation to support archetype-specific dataset schemas and filtering.

**Action Items**:
1. Define per-archetype schema extensions
2. Implement archetype-specific extractors (e.g., git diff parsing for repo maintenance)
3. Add archetype validation and routing logic
4. Create archetype-specific filtering options
5. Document archetype schemas

**Success Criteria**:
- Can curate datasets for specific archetypes with extended fields
- Archetype-specific metrics are computed and stored
- Universal schema remains consistent across archetypes

### Phase 3: Integration & Documentation

**Goal**: Connect curated datasets to evaluators and document usage.

**Action Items**:
1. Document dataset access patterns for evaluators
2. Create example queries for common analysis tasks
3. Add dataset validation and quality checks
4. Write user documentation for curation workflow
5. Define golden dataset creation process

**Success Criteria**:
- Evaluators can load and use curated datasets
- Documentation covers filtering, versioning, and schema
- Quality checks prevent invalid dataset creation

## Design Decisions

### Why Summary Stats Over Full Conversations

The dataset only stores aggregated metrics rather than full conversation traces because:
- LLM judges evaluate final output quality, not intermediate reasoning steps
- Universal metrics (loop detection, efficiency) can be inferred from summary statistics
- Reduces dataset size and complexity
- Full traces remain accessible in MLflow for deep-dive debugging

### Why MLflow Native Format

Using MLflow's native Dataset format provides:
- Native integration with MLflow evaluation and model registry
- Built-in versioning and lineage tracking
- UI support for dataset browsing
- Consistency with existing MLflow tracing infrastructure

### Why Per-Archetype Datasets

Creating separate datasets per archetype (in addition to universal) allows:
- Archetype-specific schemas without polluting universal dataset
- Targeted evaluation pipelines per task type
- Cleaner data models for LLM judges
- Flexibility to run archetype-specific analyses

## Open Questions

1. **Dataset Size**: Should there be limits on dataset size (number of runs) to prevent performance issues?
2. **Incremental Curation**: Should we support appending to existing datasets or only create new versions?
3. **Golden Dataset Integration**: How will human-validated golden datasets integrate with automated curation?
4. **Cross-Experiment Curation**: Should curation support combining runs across multiple MLflow experiments?

## Success Metrics

The curation implementation will be considered successful when:
- Can curate datasets from 100+ simulation runs in under 60 seconds
- Dataset artifacts are accessible via MLflow UI
- All universal metrics are accurate and complete
- Archetype-specific fields are populated correctly
- Filtering produces expected subsets
- Versioning prevents accidental overwrites

## Future Considerations

- **Real-time Curation**: Trigger dataset updates automatically when new simulations complete
- **Dataset Validation**: Automated checks for data quality and completeness
- **Comparative Datasets**: Create datasets specifically for A/B testing model configurations
- **Export Formats**: Support exporting to Parquet or JSONL for external tools
- **Dataset Lineage**: Track which curated datasets were used for which evaluation runs
