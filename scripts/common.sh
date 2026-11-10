#!/usr/bin/env bash
# Shared helpers for the EvoState reproduction scripts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="$REPO_ROOT/configs"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Shared filesystems with several Python versions can serve stale bytecode.
export PYTHONDONTWRITEBYTECODE=1

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

list_tasks() {
  local f
  for f in "$CONFIG_DIR"/*.conf; do basename "$f" .conf; done
}

# Load configs/<task>.conf as plain KEY=VALUE pairs (the file is parsed, not executed).
# Variables already present in the environment take precedence, so any setting
# can be overridden on the command line, e.g. `FOLDS=2 PRETRAIN_EPOCHS=5 bash ...`.
load_task_config() {
  local task="$1"
  local cfg="$CONFIG_DIR/$task.conf"
  [[ "$task" =~ ^[A-Za-z0-9_]+$ ]] || die "invalid task name: $task"
  [[ -f "$cfg" ]] || die "unknown task '$task'. Available: $(list_tasks | tr '\n' ' ')"
  local key value
  while IFS='=' read -r key value || [[ -n "$key" ]]; do
    [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
    if [[ -z "${!key+x}" ]]; then
      printf -v "$key" '%s' "$value"
      export "${key?}"
    fi
  done < "$cfg"
  TASK="$task"
}

abs_path() {
  local p="$1"
  [[ -z "$p" ]] && { printf ''; return; }
  [[ "$p" = /* ]] && printf '%s' "$p" || printf '%s/%s' "$REPO_ROOT" "$p"
}

check_python() {
  "$PYTHON_BIN" -c "import numpy, pandas, torch, sklearn" >/dev/null 2>&1 \
    || die "python ($PYTHON_BIN) lacks numpy/pandas/torch/scikit-learn; activate the evostate environment or set PYTHON_BIN."
}
