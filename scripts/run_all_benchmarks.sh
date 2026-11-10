#!/usr/bin/env bash
# Run the cross-validation benchmark for all seven manuscript tasks sequentially and
# print a summary table (Supplementary Table S1 layout).
#
# Usage:
#   bash scripts/run_all_benchmarks.sh                 # outputs/benchmark/<task>/
#   TASKS="tuberculosis_genes19 ecoli_nit" bash scripts/run_all_benchmarks.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TASKS="${TASKS:-$(list_tasks | tr '\n' ' ')}"
OUT_ROOT="${OUT_ROOT:-outputs/benchmark}"
for task in $TASKS; do
  log "==== $task"
  bash "$REPO_ROOT/scripts/run_benchmark.sh" "$task" "$OUT_ROOT/$task"
done
"$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_benchmark.py" --root "$(abs_path "$OUT_ROOT")"
