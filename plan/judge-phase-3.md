# Judge Phase 3: Evaluations Module Structure

## Overview

Create the evaluations module directory structure and initialize the module. This phase is minimal but important - it establishes the namespace for all evaluation code.

## Prerequisites

- ✅ Phase 2: Core Data Models (SimulationConfig with JudgeMapping, judges.yaml, Settings)

## Changes Required

### 1. Create Evaluations Module Directory

**Structure**:
```
src/beyond_vibes/evaluations/
├── __init__.py              # Public API exports
└── judges.yaml              # Registry (created in Phase 2)
```

**Implementation**:

```bash
mkdir -p src/beyond_vibes/evaluations
```

### 2. Create __init__.py

**File**: `src/beyond_vibes/evaluations/__init__.py`

**Upstream Dependencies**:
- Phase 2: Judges registry YAML file must exist
- Phase 4: Extractor module (will be imported here)
- Phase 5: Judge factory module (will be imported here)
- Phase 6: Runner module (will be imported here)

**Downstream Impact**:
- Other modules import from this package: `from beyond_vibes.evaluations import ...`
- CLI imports runner from here

**Implementation**:

```python
"""Evaluations module for judging simulation runs.

This module provides infrastructure for evaluating simulation runs
using configurable judges from the judges registry.

Example:
    from beyond_vibes.evaluations import EvaluationRunner, extract_run_data
    
    runner = EvaluationRunner()
    results = runner.evaluate_run("run-id-123")
"""

# Public API exports
# These will be populated as we implement each phase:
# - Phase 4: extract_run_data, query_simulation_runs
# - Phase 5: create_judge, build_judges_for_task
# - Phase 6: EvaluationRunner

__all__ = [
    # Phase 4
    # "extract_run_data",
    # "query_simulation_runs",
    # "JudgeInput",
    
    # Phase 5
    # "create_judge",
    # "build_judges_for_task",
    
    # Phase 6
    # "EvaluationRunner",
]
```

**Notes**:
- Start with empty `__all__` and uncomment as each phase is implemented
- This prevents ImportError during development
- Alternatively, implement all imports with try/except for development

**Alternative approach for incremental development**:

```python
"""Evaluations module for judging simulation runs."""

try:
    from .extractor import extract_run_data, query_simulation_runs, JudgeInput
except ImportError:
    # Phase 4 not yet implemented
    extract_run_data = None
    query_simulation_runs = None
    JudgeInput = None

try:
    from .judge_factory import create_judge, build_judges_for_task
except ImportError:
    # Phase 5 not yet implemented
    create_judge = None
    build_judges_for_task = None

try:
    from .runner import EvaluationRunner
except ImportError:
    # Phase 6 not yet implemented
    EvaluationRunner = None

__all__ = [
    "extract_run_data",
    "query_simulation_runs",
    "JudgeInput",
    "create_judge",
    "build_judges_for_task",
    "EvaluationRunner",
]
```

---

## Dependencies

### Phase Dependencies

| Phase | Depends On | Reason |
|-------|-----------|--------|
| Phase 3 | Phase 2 | Needs models and registry to exist |
| Phase 4 | Phase 3 | Extends evaluations module with extractor |
| Phase 5 | Phase 3 | Extends evaluations module with judge factory |
| Phase 6 | Phase 3 | Extends evaluations module with runner |

### File Dependencies

```
Phase 2 outputs:
  ├── src/beyond_vibes/simulations/models.py (JudgeMapping)
  ├── src/beyond_vibes/settings.py (judge settings)
  └── src/beyond_vibes/evaluations/judges.yaml (registry)
        ↓
Phase 3 creates:
  └── src/beyond_vibes/evaluations/__init__.py
```

---

## Success Criteria

- [ ] Directory `src/beyond_vibes/evaluations/` exists
- [ ] `__init__.py` exists (can be empty initially)
- [ ] Module can be imported: `from beyond_vibes import evaluations`
- [ ] No ImportError when importing the module

---

## Next Phase

**Phase 4**: Extractor Module
- Depends on: Phase 3 (module structure)
- Creates: `extractor.py` with `JudgeInput` and MLflow integration
