#!/usr/bin/env bash
# Grouped five-fold cross-validation for one prediction task.
#
# Within every fold: Stage 1 state pretraining on training-partition states ->
# Stage 2 conditional event prediction (EvoState); the same architecture trained
# from random initialisation (EvoState w/o pretraining); Uniform and Prevalence
# baselines; and, where configured and available, HyperTraPS.
#
# Usage:
#   bash scripts/run_benchmark.sh <task> [outdir]
#   bash scripts/run_benchmark.sh tuberculosis_genes19
#   WITH_HT=0 bash scripts/run_benchmark.sh klebsiella_kleborate22        # skip HyperTraPS
#   HT_BIN=/path/to/RUN_MCMC_SAMPLER bash scripts/run_benchmark.sh tuberculosis_genes19
#   FOLDS=2 PRETRAIN_EPOCHS=3 FINETUNE_EPOCHS=3 WITH_HT=0 \
#       bash scripts/run_benchmark.sh tuberculosis_genes19 outputs/smoke   # quick smoke test
#
# Outputs: <outdir>/eval_summary.json, fold_metrics.json, method_runtime.csv, fold_runs/
# Method keys: B0_uniform=Uniform, B1_prevalence=Prevalence, TS_random_init=EvoState w/o
# pretraining, TS_pretrained=EvoState, HT_second-order=HyperTraPS.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[[ $# -ge 1 ]] || die "usage: $0 <task> [outdir]   tasks: $(list_tasks | tr '\n' ' ')"
load_task_config "$1"
OUTDIR="$(abs_path "${2:-outputs/benchmark/$TASK}")"
check_python

TRANSITIONS_ABS="$(abs_path "$TRANSITIONS")"
HIERARCHY_ABS="$(abs_path "${HIERARCHY:-}")"
[[ -f "$TRANSITIONS_ABS" ]] || die "missing transitions: $TRANSITIONS_ABS"

cmd=(
  "$PYTHON_BIN" "$REPO_ROOT/src/eval/run_two_stage_eval.py"
  --transitions "$TRANSITIONS_ABS"
  --outdir "$OUTDIR"
  --split-mode "$SPLIT_MODE" --folds "$FOLDS" --seed "$SEED"
  --arch "$ARCH" --d-model "$D_MODEL" --n-layers "$N_LAYERS" --n-heads "$N_HEADS"
  --dropout "$DROPOUT" --weight-decay "$WEIGHT_DECAY" --lr "$LR"
  --batch-size "$BATCH_SIZE" --val-frac "$VAL_FRAC" --patience "$PATIENCE"
  --pretrain-epochs "$PRETRAIN_EPOCHS" --finetune-epochs "$FINETUNE_EPOCHS"
  --mask-prob "$MASK_PROB" --mask-prob-pos "$MASK_PROB_POS" --mask-prob-neg "$MASK_PROB_NEG"
  --contrastive-weight "$CONTRASTIVE_WEIGHT"
  --contrastive-temperature "$CONTRASTIVE_TEMPERATURE"
  --contrastive-warmup-epochs "$CONTRASTIVE_WARMUP_EPOCHS"
  --similar-state-neg-threshold "$SIMILAR_STATE_NEG_THRESHOLD"
  --pretrain-checkpoint-metric "$PRETRAIN_CHECKPOINT_METRIC"
  --state-source "$STATE_SOURCE"
  --no-meta
  --eval-sides auto --ts-head-mode auto --loss-weight 1.0 --stratify-by auto
  # Stage-2 adaptation settings used in the manuscript (full fine-tuning, no alignment stage)
  --freeze-backbone-epochs 0 --backbone-lr-scale 1.0 --head-lr-scale 1.0 --unfreeze-last-n-layers 0
)
[[ -n "$HIERARCHY_ABS" ]] && cmd+=(--hierarchy "$HIERARCHY_ABS")

if [[ "${WITH_HT:-0}" == "1" ]]; then
  cmd+=(--with-hypertraps --ht-model "$HT_MODEL" --ht-N "$HT_N" --ht-burn "$HT_BURN"
        --ht-r "$HT_R" --ht-timeout "$HT_TIMEOUT" --ht-threads "$HT_THREADS")
  [[ -n "${HT_BIN:-}" ]] && cmd+=(--ht-bin "$HT_BIN")
else
  cmd+=(--no-hypertraps)
fi
[[ "${DUMP_PREDICTIONS:-0}" == "1" ]] && cmd+=(--dump-predictions)
[[ "${DUMP_FOLD_MODELS:-0}" == "1" ]] && cmd+=(--dump-fold-models)

mkdir -p "$OUTDIR"
log "task=$TASK ($TASK_NAME) d_model=$D_MODEL n_layers=$N_LAYERS lambda_con=$CONTRASTIVE_WEIGHT with_ht=${WITH_HT:-0}"
log "outdir=$OUTDIR"
"${cmd[@]}" 2>&1 | tee "$OUTDIR/run.log"
exit "${PIPESTATUS[0]}"
