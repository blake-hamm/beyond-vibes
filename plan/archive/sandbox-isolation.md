# Sandbox Isolation Plan — Beyond Vibes Simulation Security

**Status:** Proposed (pending approval)
**Council round:** 2 (converged)
**Date:** 2026-04-30

---

## Problem Statement

AI agent models spawned via `pi.dev` CLI run as the same OS user with full filesystem and environment access. A qwen-36-a3b model spent **28 of 40 turns** fighting NixOS/venv contamination instead of writing tests:

- `poetry run` resolved to the parent project venv (`/home/bhamm/repos/beyond-vibes/.venv`)
- `pip install --break-system-packages` tried writing to the read-only Nix store
- `poetry install` mutated the parent venv (overwrote urllib3, tzdata, etc.)
- `poetry env remove` attempted to delete the parent venv (permission denied)

The user stated: **"Safety is the primary concern. Ensure this sandbox has extremely limited access."**

---

## Threat Model

| Vector | Likelihood | Impact | Current defense |
|--------|-----------|--------|-----------------|
| Env/PATH confusion → wrong Python/venv | **High** (observed) | High (wasted turns, corrupted parent venv) | Partial blocklist |
| Absolute path traversal to parent project | Medium | High (reads/writes outside sandbox) | None |
| Credential exfiltration via env vars | Low | High (API keys, SSH keys leaked to model) | Partial blocklist |
| `~/.bashrc`, `~/.gitconfig`, `~/.ssh` pollution | Low-Medium | Medium | None |

**Scope:** Accidental contamination by confused models. Not adversarial sandbox escape.

---

## Options Evaluated

### Option A — Env blocklist only (current, rejected)
Grow `strip_prefixes` tuple in `_build_sandbox_env`. Whack-a-mole. Always misses something (`S3_`, `MINIO_`, `DATABASE_URL`, `PYTHONSTARTUP`).

### Option B — Bubblewrap mount namespaces (deferred)
Wrap `pi` invocation in `bwrap --unshare-all --share-net --die-with-parent ...`. Real filesystem isolation. Linux-only. Fragile on Ubuntu AppArmor, Debian `userns=0`, macOS, nested CI containers.

**Verdict:** Correct architecture, but premature. Adds 2–3 days implementation + 1 week CI debugging for a threat not yet observed in traces.

### Option C — Env allowlist + explicit PATH (converged recommendation)
Start with empty `env = {}`. Add only what `pi` needs. Build PATH from resolved system binaries. Fail-closed. Cross-platform. Testable today.

**Verdict:** Ship now. Fixes observed failure. Closes credential leak vector. Bubblewrap can be added later as a feature-flag if filesystem-based contamination appears in logs.

---

## Converged Recommendation

### Phase 1 — Env Allowlist (ship immediately)

Replace `_build_sandbox_env` blocklist with allowlist:

```python
@staticmethod
def _build_sandbox_env(working_dir: Path | None) -> dict[str, str]:
    """Create an isolated environment for the pi subprocess."""
    env: dict[str, str] = {}

    # Working directory as home prevents writes to ~/.ssh, ~/.config, etc.
    env["HOME"] = str(working_dir) if working_dir else os.environ.get("HOME", "")
    env["TMPDIR"] = str(working_dir / "tmp") if working_dir else "/tmp"
    env["PYTHONNOUSERSITE"] = "1"
    env["POETRY_VIRTUALENVS_CREATE"] = "true"
    env["POETRY_VIRTUALENVS_IN_PROJECT"] = "true"

    # Build PATH explicitly from resolved system paths — no inherited host PATH
    path_dirs: list[str] = []
    for cmd in ("git", "python3", "bash", "node"):
        p = shutil.which(cmd)
        if p:
            path_dirs.append(str(Path(p).parent))
    # Deduplicate while preserving order
    seen: set[str] = set()
    env["PATH"] = ":".join(d for d in path_dirs if not (d in seen or seen.add(d)))

    # Add only safe locale/terminal vars if they exist on host
    for key in ("LANG", "LC_ALL", "TERM"):
        if key in os.environ:
            env[key] = os.environ[key]

    return env
```

Key properties:
- **Fail-closed**: New env var formats (e.g. `GROQ_API_KEY`) cannot leak because they are not explicitly added
- **No PATH pollution**: `VIRTUAL_ENV`, `POETRY_HOME`, `PYENV_ROOT`, `CONDA_PREFIX` are absent by construction
- **No parent project paths**: `beyond-vibes/.venv` cannot appear in PATH because we build it from scratch
- **Cross-platform**: Works on Linux, macOS, NixOS, CI today

### Phase 2 — Prompt Hardening (already applied)

All task prompts updated to include:

```yaml
system_prompt: |
  You are running on NixOS inside a temporary sandbox.
  Never ask for human feedback or confirmation.
  Just proceed with the best approach and explain what you did afterward.
```

Task-specific `CRITICAL — Environment` sections added to:
- `unit_tests.yaml`
- `poetry_to_uv.yaml`
- `e2e_test.yaml`

### Phase 3 — Tests (add before merge)

| Test | What it verifies |
|------|-----------------|
| `test_env_allowlist_blocks_leaked_vars` | `AWS_SECRET_ACCESS_KEY`, `GITHUB_TOKEN`, `VIRTUAL_ENV` absent from `Popen` env |
| `test_env_path_has_no_parent_venv` | No `.venv`, `conda`, `pyenv`, `beyond-vibes` segments in PATH |
| `test_env_path_is_constructed` | PATH contains only directories of resolved `git`, `python3`, `bash`, `node` |
| `test_home_is_working_dir` | `HOME` env var equals `working_dir` |
| `test_pi_runs_in_sandbox` | `pi --version` succeeds with allowlisted env |

Update existing tests:
- `test_run_command_construction`: assert `cmd[0]` ends with `"pi"` (absolute path from `shutil.which`)

### Phase 4 — Bubblewrap Feature Flag (backlog, deferred)

**Trigger condition:** Demonstrate one simulation trace where a model accidentally escapes the temp workspace via absolute path traversal (e.g. `cd /home/bhamm/repos/beyond-vibes && rm -rf .venv`).

If triggered, implement behind a flag:

```python
def _has_bwrap() -> bool:
    return sys.platform == "linux" and shutil.which("bwrap") is not None
```

Tested invocation:
```bash
bwrap \
  --unshare-all \
  --share-net \
  --die-with-parent \
  --proc /proc \
  --dev /dev \
  --tmpfs /tmp \
  --bind $WORK_DIR $WORK_DIR \
  --ro-bind /nix/store /nix/store \
  --ro-bind /run/current-system/sw/bin /run/current-system/sw/bin \
  --ro-bind /etc/profiles/per-user/bhamm/bin /etc/profiles/per-user/bhamm/bin \
  --ro-bind /etc/ssl/certs /etc/ssl/certs \
  --ro-bind /etc/resolv.conf /etc/resolv.conf \
  --chdir $WORK_DIR \
  --setenv HOME $WORK_DIR \
  -- pi --version
```

Fallback: env allowlist on macOS or missing bwrap.

---

## Files to Change

| File | Change |
|------|--------|
| `src/beyond_vibes/simulations/pi_dev.py` | Replace `_build_sandbox_env` blocklist with allowlist; resolve `pi` to absolute path via `shutil.which` |
| `tests/test_pi_dev.py` | Add allowlist isolation tests; update command construction test |
| `src/beyond_vibes/simulations/prompts/tasks/*.yaml` | Already updated (system_prompt + env guardrails) |

---

## Residual Tensions

| Architect wants | Product wants | Resolution |
|-----------------|---------------|------------|
| Filesystem containment via bwrap now | Defer bwrap until evidence of filesystem escape | **Deferred.** Log ticket. Enable if traces show absolute-path contamination. |
| Integration tests with real subprocesses | Fast unit tests only | **Hybrid.** Unit tests for env allowlist. Integration tests deferred with bwrap ticket. |

---

## Action Items

| # | Action | Owner | Priority |
|---|--------|-------|----------|
| 1 | Implement env allowlist in `_build_sandbox_env` | TBD | **P0** |
| 2 | Add unit tests for env isolation | TBD | **P0** |
| 3 | Run existing tests, verify no regressions | TBD | **P0** |
| 4 | Create backlog ticket: "Bubblewrap filesystem sandboxing (Linux-only, feature-flag)" | TBD | P2 |

---

*Plan produced by council review (architect + product perspectives). Both agree: ship env allowlist now, defer bubblewrap until filesystem contamination is demonstrated in traces.*
