# VS Code setup (optional)

An example configuration for working on this project in VS Code. It is a
template: your own `.vscode/` folder is ignored by Git, so you can change it
freely.

## Install

From the repository root, after `uv sync --locked`:

```sh
cp -r .vscode.example .vscode          # Linux, macOS, Git Bash
```

```powershell
Copy-Item -Recurse .vscode.example .vscode   # PowerShell
```

Open the **repository root** as the VS Code folder, accept the recommended
extensions (Python, Python Debugger, Pylance, Ruff), then run
**Developer: Reload Window**. If the Testing panel stays empty, run
**Python: Select Interpreter** and choose `.venv`.

## What you get

| File | Purpose |
|---|---|
| `settings.json` | pytest in the Testing panel; Ruff as formatter on save; LF line endings |
| `launch.json` | Run and Debug entries for every script, grouped by TP, with breakpoints |
| `tasks.json` | Setup, tests, coverage, Ruff, pre-commit and MLflow server tasks |
| `start_mlflow.py` | Starts the local tracking server, or does nothing if one already answers |
| `extensions.json` | Recommended extensions |

## Everyday use

- **Tests**: the Testing panel (flask icon), or `Ctrl+Shift+P` →
  *Tasks: Run Test Task*.
- **All checks before a commit**: `Ctrl+Shift+B` runs Ruff check, Ruff
  format check and pytest.
- **Everything at once**: Run and Debug (`Ctrl+Shift+D`) →
  **▶ Launch everything** → `F5`. It verifies the data, starts the MLflow
  server, opens its UI, then runs the tests and the two TP03 inference
  checks.
- **MLflow server**: tasks *MLflow: start server and open the UI* and
  *MLflow: stop server*. The address comes from `tracking.tracking_uri` in
  `configs/train.json`.

## Things to know

- **Launch everything does not retrain.** *TP03: tracked training* is a
  separate entry on purpose: each run creates new MLflow runs and a new
  registered version, and moves the alias `candidate` to it.
- *TP02: train* and *TP01: inspect data* rewrite tracked report files, so
  Git shows changes after running them.
- The two "verify" entries write their report to `.vscode/.out/`, not to
  `reports/`, so committed evidence is left alone.
- The inference entries need a model in the local MLflow store. On a fresh
  clone, run *TP03: tracked training* once first.
