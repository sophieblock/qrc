# Graphviz setup — drop-in snippets

Pair these with `graphviz_check.py`. Copy whichever apply to your project.

---

## 1. Wire the check into your package

In your top-level viz module (e.g. `qrew/visualization_tools.py`):

```python
from .graphviz_check import require_graphviz, check_graphviz

# Option A — fail fast at import time of the viz module
require_graphviz()

# Option B — defer until first render call (better for libraries; lets
# users import the package on a machine with no Graphviz)
def render_dag(dag, path):
    require_graphviz()
    ...

# Option C — soft degrade
def render_dag(dag, path):
    if not check_graphviz()["ok"]:
        warnings.warn("Graphviz unavailable, skipping render")
        return None
    ...
```

Run the smoke test anytime:

```bash
python -m graphviz_check
```

---

## 2. Python deps

### `pyproject.toml` (modern PEP 621 — recommended)

```toml
[project]
dependencies = [
    "graphviz>=0.20",         # Python wrapper. The SYSTEM dot binary is NOT pip-installable.
]

[project.optional-dependencies]
viz = [
    "graphviz>=0.20",
    "pydot>=2.0",             # only if you use it
    "networkx>=3.0",          # only if you use it
]
```

### `requirements.txt` (legacy / pip-tools)

```text
graphviz>=0.20
```

### `environment.yml` (conda / mamba / miniforge — installs both binary AND wrapper)

```yaml
name: qrew
channels:
  - conda-forge
dependencies:
  - python>=3.10
  - graphviz            # the C binary
  - python-graphviz     # the Python wrapper
```

This conda variant is the **single least-painful path** on Apple Silicon and on locked-down work boxes — no Homebrew, no admin rights, no PATH editing.

---

## 3. GitHub Actions (personal repo)

```yaml
# .github/workflows/test.yml
name: test
on: [push, pull_request]

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: ["3.11", "3.12"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}

      # Install Graphviz per platform
      - name: Install Graphviz (Linux)
        if: runner.os == 'Linux'
        run: sudo apt-get update && sudo apt-get install -y graphviz

      - name: Install Graphviz (macOS)
        if: runner.os == 'macOS'
        run: brew install graphviz

      - name: Install Graphviz (Windows)
        if: runner.os == 'Windows'
        run: choco install graphviz --no-progress -y

      - name: Install Python deps
        run: pip install -e ".[viz]"

      - name: Verify Graphviz
        run: python -m graphviz_check

      - name: Run tests
        run: pytest
```

---

## 4. GitLab CI (work repo with GitLab Runner)

The trick on a locked-down GitLab Runner is to pick an image that already has Graphviz, or install it once and cache. Two patterns:

### Pattern A — apt-based image (Debian/Ubuntu runner)

```yaml
# .gitlab-ci.yml
image: python:3.12-slim

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

cache:
  paths:
    - .cache/pip
    - .venv

before_script:
  - apt-get update -qq && apt-get install -y --no-install-recommends graphviz
  - python -m venv .venv
  - source .venv/bin/activate
  - pip install -e ".[viz]"
  - python -m graphviz_check    # fail the job early if anything is off

stages: [test]

unit_tests:
  stage: test
  script:
    - pytest -q
```

### Pattern B — conda runner (closer to your local Miniforge setup)

```yaml
image: condaforge/miniforge3:latest

before_script:
  - conda env create -f environment.yml -n ci || conda env update -f environment.yml -n ci
  - source activate ci
  - python -m graphviz_check

unit_tests:
  script:
    - pytest -q
```

This is the better match for your work setup — same `environment.yml` runs locally and in CI, and you sidestep the Boeing `apt` repo question entirely.

---

## 5. Why `graphviz` (the wrapper) and not `pygraphviz`

| | `graphviz` (pure Python) | `pygraphviz` |
|---|---|---|
| Install | `pip install graphviz` — wheels everywhere | Builds C extension against `libgraphviz-dev`. Frequent breakage on macOS + Windows. |
| Apple Silicon | Fine | Historically painful; needs `--config-settings` flags pointing at brew prefix |
| Locked-down Windows | Fine | Needs MSVC build tools |
| API | Builds DOT source, shells out to `dot` | Direct C bindings — faster, but the install cost rarely pays back |

Unless you specifically need the C-binding speed (you don't, for circuit DAGs of any reasonable size), stick with `graphviz`. `pydot` is the other reasonable pure-Python option — same binary dependency, slightly different API.

---

## 6. Common failure modes the check catches

| Symptom | What `check_graphviz()` reports |
|---|---|
| `dot` not on PATH | `ok=False`, `dot_path=None`, `reason="not found"` |
| Windows installer skipped PATH checkbox | Same as above — fix by setting `GRAPHVIZ_DOT=C:\Program Files\Graphviz\bin\dot.exe` |
| Wrong-arch binary on Apple Silicon (rare with Homebrew, common with old `.dmg`) | `ok=False`, `reason` shows the OSError from subprocess |
| Sandboxed runner blocks subprocess | Same — `reason` shows the OSError |
| Python `graphviz` package missing but binary present | `ok=True` from `check_graphviz()`; `require_graphviz()` raises with pip hint |
