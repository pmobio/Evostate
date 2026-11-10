#!/usr/bin/env bash
# Fit EvoState on all transitions of one task (no held-out data).
#
# This produces the full-data models used for the descriptive, dataset-specific
# analyses (forward simulation, dependency estimates, E. coli prioritisation and
# partial-state completion). Do NOT use these models for held-out performance;
# use scripts/run_benchmark.sh for cross-validated metrics.
#
# Usage:
#   bash scripts/train_full.sh <task> [outdir]
# Outputs:
#   <outdir>/stage1/stage1_model.pt   self-supervised state encoder
#   <outdir>/stage2/                  conditional event predictor (initialised from stage1)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[[ $# -ge 1 ]] || die "usage: $0 <task> [outdir]   tasks: $(list_tasks | tr '\n' ' ')"
load_task_config "$1"
OUTDIR="$(abs_path "${2:-outputs/full_fit/$TASK}")"
check_python

TRANSITIONS_ABS="$(abs_path "$TRANSITIONS")"
HIERARCHY_ABS="$(abs_path "${HIERARCHY:-}")"
hier_args=()
[[ -n "$HIERARCHY_ABS" ]] && hier_args=(--hierarchy "$HIERARCHY_ABS")

shared=(
  --transitions "$TRANSITIONS_ABS"
  --arch "$ARCH" --d-model "$D_MODEL" --n-layers "$N_LAYERS" --n-heads "$N_HEADS"
  --dropout "$DROPOUT" --weight-decay "$WEIGHT_DECAY" --lr "$LR"
  --batch-size "$BATCH_SIZE" --val-frac "$VAL_FRAC" --patience "$PATIENCE" --seed "$SEED"
  --no-meta
)

mkdir -p "$OUTDIR/stage1" "$OUTDIR/stage2"
log "[$TASK] Stage 1: self-supervised state pretraining"
"$PYTHON_BIN" "$REPO_ROOT/src/models/train_state_pretrain.py" \
  "${shared[@]}" "${hier_args[@]}" \
  --outdir "$OUTDIR/stage1" --epochs "$PRETRAIN_EPOCHS" \
  --mask-prob "$MASK_PROB" --mask-prob-pos "$MASK_PROB_POS" --mask-prob-neg "$MASK_PROB_NEG" \
  --contrastive-weight "$CONTRASTIVE_WEIGHT" \
  --contrastive-temperature "$CONTRASTIVE_TEMPERATURE" \
  --contrastive-warmup-epochs "$CONTRASTIVE_WARMUP_EPOCHS" \
  --similar-state-neg-threshold "$SIMILAR_STATE_NEG_THRESHOLD" \
  --pretrain-checkpoint-metric "$PRETRAIN_CHECKPOINT_METRIC" \
  --state-source "$STATE_SOURCE" 2>&1 | tee "$OUTDIR/stage1/run.log"

log "[$TASK] Stage 2: conditional event prediction"
"$PYTHON_BIN" "$REPO_ROOT/src/models/train_transition_two_stage.py" \
  "${shared[@]}" "${hier_args[@]}" \
  --outdir "$OUTDIR/stage2" --epochs "$FINETUNE_EPOCHS" \
  --pretrained "$OUTDIR/stage1/stage1_model.pt" \
  --freeze-backbone-epochs 0 --backbone-lr-scale 1.0 --head-lr-scale 1.0 \
  --unfreeze-last-n-layers 0 2>&1 | tee "$OUTDIR/stage2/run.log"
log "[$TASK] done -> $OUTDIR"
