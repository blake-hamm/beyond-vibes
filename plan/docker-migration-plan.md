# Docker Migration Plan

## Problem

The current `PiDevClient._build_sandbox_env` tries to build an isolated environment by:
- Setting `HOME` to the temp workspace
- Filtering `PATH` to only 4 tool directories (`git`, `python3`, `bash`, `node`)
- Blocking virtual-env and conda paths

This breaks on NixOS (and any non-FHS system) because basic tools like `ls`, `find`, `cat` live in separate store paths that get stripped out. It also breaks provider discovery for `pi` when `HOME` is moved away from `~/.pi`.

We temporarily removed the sandbox logic. For production (Argo Workflows), Docker is the right isolation layer.

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Argo Workflow                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Init container                                     │    │
│  │  • git clone <repo> → /workspace/repo               │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Main container: beyond-vibes-simulator:latest      │    │
│  │  • pi CLI pre-installed                             │    │
│  │  • git, python3, node, bash, standard unix tools    │    │
│  │  • beyond-vibes Python package + deps               │    │
│  │                                                     │    │
│  │  Entrypoint:                                        │    │
│  │    beyond-vibes simulate                            │    │
│  │      --task <task>                                  │    │
│  │      --model <model>                                │    │
│  │      --repo-path /workspace/repo                    │    │
│  │      (reads JSONL from pi stdout)                   │    │
│  │                                                     │    │
│  │  Outputs:                                           │    │
│  │    • MLflow run (metrics, turns, git diff)          │    │
│  │    • Exit code 0 = success, 1 = failure             │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Dockerfile

```dockerfile
FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install pi CLI (adjust to your actual install method)
RUN npm install -g @mariozechner/pi-coding-agent

# Install beyond-vibes
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --no-dev

COPY src/ ./src/
RUN uv pip install -e .

# Pre-configure pi providers if needed (litellm config, etc.)
# COPY .pi/ /root/.pi/

ENTRYPOINT ["beyond-vibes"]
```

---

## Argo Workflow Template (sketch)

```yaml
apiVersion: argoproj.io/v1alpha1
kind: WorkflowTemplate
metadata:
  name: beyond-vibes-simulation
spec:
  templates:
    - name: simulate
      inputs:
        parameters:
          - name: task
          - name: model
          - name: repo-url
          - name: repo-branch
            value: main
      volumes:
        - name: workspace
          emptyDir: {}
      initContainers:
        - name: clone
          image: alpine/git
          command: [sh, -c]
          args:
            - |
              git clone --depth 1 -b {{inputs.parameters.repo-branch}} \
                {{inputs.parameters.repo-url}} /workspace/repo
          volumeMounts:
            - name: workspace
              mountPath: /workspace
      container:
        image: beyond-vibes-simulator:latest
        command: [beyond-vibes]
        args:
          - simulate
          - --task
          - "{{inputs.parameters.task}}"
          - --model
          - "{{inputs.parameters.model}}"
          - --repo-path
          - /workspace/repo
        env:
          - name: MLFLOW_TRACKING_URI
            valueFrom:
              secretKeyRef:
                name: mlflow-secrets
                key: tracking-uri
          - name: OPENAI_API_KEY
            valueFrom:
              secretKeyRef:
                name: api-keys
                key: openai
        volumeMounts:
          - name: workspace
            mountPath: /workspace
```

---

## CLI Changes Needed

1. **Add `--repo-path` flag** to `simulate`
   - If provided, skip `SandboxManager.create()` + `clone_repo()`
   - Use the given path as `working_dir` directly

2. **Remove `SandboxManager` from `simulate` command** (or make it optional)
   - In Docker, the repo is already cloned by the init container
   - No temp dir creation needed

3. **Keep `PiDevClient` simple**
   - Already done: `_build_sandbox_env` returns `None` (inherits host env)
   - `subprocess.Popen` with `env=None` uses full container environment
   - All tools (`ls`, `find`, `cat`, `git`, `python3`, `node`, `bash`) available

---

## Migration Steps

| Step | Action | Owner |
|------|--------|-------|
| 1 | Merge stripped sandbox code | You |
| 2 | Add `--repo-path` to CLI, skip clone when given | You |
| 3 | Build Dockerfile, test locally | You |
| 4 | Push image to registry | CI |
| 5 | Write Argo WorkflowTemplate | You |
| 6 | Test end-to-end in local k8s (kind/minikube) | You |
| 7 | Deploy to production Argo | You |

---

## Benefits

- **No PATH scavenging**: Docker image has everything in standard FHS paths
- **Reproducible**: Same image everywhere (local, CI, Argo)
- **Secure**: Argo secrets for API keys, no host filesystem access
- **Scalable**: Argo can run many simulations in parallel
