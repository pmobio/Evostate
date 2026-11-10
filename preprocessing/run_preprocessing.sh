#!/usr/bin/env bash
# Rebuild the processed datasets in data/processed/ from the original public sources.
#
# The processed datasets are already shipped with the repository; this script is only
# needed to regenerate them. See preprocessing/README.md for where to obtain raw data.
#
# Usage:
#   bash preprocessing/run_preprocessing.sh tuberculosis   # Genes19 / Antibiotics10 / Combined29
#   bash preprocessing/run_preprocessing.sh klebsiella     # Kleborate22 (+ country / lineage subsets)
#   bash preprocessing/run_preprocessing.sh ecoli          # NIT / MEC / TMP
#   bash preprocessing/run_preprocessing.sh all
#
# Environment overrides:
#   RAW_DIR       raw-data root (default: data/raw)
#   BUILD_DIR     intermediate output root (default: outputs/preprocessing)
#   INSTALL=1     copy rebuilt transitions into data/processed/ (default: 0, compare only)
set -euo pipefail

PRE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PRE_DIR/../scripts/common.sh"

RAW_DIR="$(abs_path "${RAW_DIR:-data/raw}")"
BUILD_DIR="$(abs_path "${BUILD_DIR:-outputs/preprocessing}")"
INSTALL="${INSTALL:-0}"
ECOLI_DIR="$REPO_ROOT/data/ecoli2026"   # fixed location expected by preprocessing/ecoli/*.py

require() { [[ -e "$1" ]] || die "missing input: $1 (see preprocessing/README.md)"; }

# compare <rebuilt_file> <task> <relative_path_in_processed>
compare() {
  local rebuilt="$1" task="$2" rel="$3"
  local shipped="$REPO_ROOT/data/processed/$task/$rel"
  if cmp -s "$rebuilt" "$shipped"; then
    log "  [identical] $task/$rel"
  else
    log "  [DIFFERENT] $task/$rel  (rebuilt: $rebuilt)"
  fi
  if [[ "$INSTALL" == "1" ]]; then
    mkdir -p "$(dirname "$shipped")"
    cp "$rebuilt" "$shipped"
  fi
}

prepare_tuberculosis() {
  local tb_raw="$RAW_DIR/tuberculosis"
  require "$tb_raw/ng.2878-S2.txt"
  local -A matrices=(
    [tuberculosis_genes19]="tuberculosis-v5-header-29-limit-19.csv"
    [tuberculosis_antibiotics10]="tuberculosis-v5-header-19-29.csv"
    [tuberculosis_combined29]="tuberculosis-v5-header-29.csv"
  )
  local task ds
  for task in tuberculosis_genes19 tuberculosis_antibiotics10 tuberculosis_combined29; do
    require "$tb_raw/${matrices[$task]}"
    ds="$BUILD_DIR/$task"
    log "[tuberculosis] $task: binarise features, prune tree, define lineage groups"
    "$PYTHON_BIN" "$PRE_DIR/tuberculosis/prep_tuberculosis_ht.py" \
      --matrix "$tb_raw/${matrices[$task]}" --tree "$tb_raw/ng.2878-S2.txt" \
      --outdir "$ds" --max-group-size 40
    log "[tuberculosis] $task: irreversible parsimony ancestral reconstruction -> transitions"
    "$PYTHON_BIN" "$PRE_DIR/build_transitions.py" \
      --barcode "$ds/category_barcode.csv" --id-col sample_id \
      --tree "$ds/tree_pruned.nwk" --tree-format newick \
      --leaf-group-file "$ds/samples.csv" --leaf-group-id-col sample_id \
      --leaf-group-col lineage_group \
      --outdir "$ds/transitions"
    compare "$ds/transitions/transitions.csv" "$task" transitions/transitions.csv
    compare "$ds/feature_hierarchy.csv" "$task" feature_hierarchy.csv
  done
}

prepare_klebsiella() {
  local kp_raw="$RAW_DIR/klebsiella"
  local task=klebsiella_kleborate22 ds="$BUILD_DIR/klebsiella_kleborate22"
  require "$kp_raw/2021-Klebsiella.csv"
  require "$kp_raw/2021-Klebsiella_tree.nwk"
  log "[klebsiella] filter K. pneumoniae sensu stricto and binarise 22 Kleborate classes"
  "$PYTHON_BIN" "$PRE_DIR/klebsiella/prep_microreact_kp.py" \
    --csv "$kp_raw/2021-Klebsiella.csv" --outdir "$ds" \
    --species "Klebsiella pneumoniae" --min-prevalence 0.0
  log "[klebsiella] ancestral reconstruction on the Mash-NJ tree, grouped by ST"
  "$PYTHON_BIN" "$PRE_DIR/build_transitions.py" \
    --barcode "$ds/category_barcode.csv" --id-col sample_id \
    --tree "$kp_raw/2021-Klebsiella_tree.nwk" --tree-format newick \
    --leaf-group-file "$ds/samples.csv" --leaf-group-id-col sample_id --leaf-group-col ST \
    --outdir "$ds/transitions"
  compare "$ds/transitions/transitions.csv" "$task" transitions/transitions.csv

  # Subsets used by the country- and lineage-specific checkpoints (Figure 4)
  log "[klebsiella] country subsets: independent reconstruction per country (>=100 isolates, >=30 transitions)"
  "$PYTHON_BIN" "$PRE_DIR/stratify_transitions_tree.py" \
    --barcode "$ds/category_barcode.csv" --tree "$kp_raw/2021-Klebsiella_tree.nwk" \
    --samples "$ds/samples.csv" --stratify-by Country --leaf-group-col ST \
    --min-samples 100 --min-transitions 30 --outdir "$ds/transitions_by_country"
  local c st
  for c in Brazil China India Italy Thailand UK USA; do
    compare "$ds/transitions_by_country/$c/transitions.csv" "$task" "subsets/country_$c/transitions.csv"
  done
  log "[klebsiella] lineage subsets: global transitions of each sequence type"
  "$PYTHON_BIN" "$PRE_DIR/klebsiella/extract_lineage_subsets.py" \
    --transitions "$ds/transitions/transitions.csv" \
    --sts ST11,ST147,ST15,ST258,ST307,ST512 --outdir "$ds/lineages"
  for st in ST11 ST147 ST15 ST258 ST307 ST512; do
    compare "$ds/lineages/lineage_$st/transitions.csv" "$task" "subsets/lineage_$st/transitions.csv"
  done
}

prepare_ecoli() {
  require "$ECOLI_DIR/SourceData_MainFigures.xlsx"
  require "$ECOLI_DIR/SourceData_SupplementaryFigures.xlsx"
  require "$ECOLI_DIR/fsepi_data"
  log "[ecoli] extract endpoint mutation matrices from supplementary data"
  "$PYTHON_BIN" "$PRE_DIR/ecoli/extract_supplementary_matrices.py"
  log "[ecoli] build phenotype (MIC) trajectories"
  "$PYTHON_BIN" "$PRE_DIR/ecoli/build_phenotype_trajectories.py"
  log "[ecoli] build per-drug endpoint datasets (features carried by >=2 populations)"
  "$PYTHON_BIN" "$PRE_DIR/ecoli/build_endpoint_datasets.py" \
    --drugs MEC,NIT,TMP --min-carriers 2 --mutator-mode off \
    --mutator-backgrounds mutL,mutT,uvrD,recA,hfq,uvrA,uvrC,holC
  log "[ecoli] construct single-gain state sequences (frequency order, background markers)"
  "$PYTHON_BIN" "$PRE_DIR/ecoli/build_two_stage_inputs.py" \
    --drugs MEC,NIT,TMP --background-mode marker --env-mode none --temporal inferred \
    --order-by frequency --out-root "$ECOLI_DIR/derived/two_stage"
  local drug task src="$ECOLI_DIR/derived/two_stage_temporal_inferred"
  for drug in NIT MEC TMP; do
    task="ecoli_$(printf '%s' "$drug" | tr '[:upper:]' '[:lower:]')"
    compare "$src/$drug/transitions/transitions.csv" "$task" transitions/transitions.csv
    compare "$src/$drug/feature_hierarchy.csv" "$task" feature_hierarchy.csv
  done
}

[[ $# -ge 1 ]] || die "usage: $0 {tuberculosis|klebsiella|ecoli|all}"
check_python
mkdir -p "$BUILD_DIR"
case "$1" in
  tuberculosis) prepare_tuberculosis ;;
  klebsiella)   prepare_klebsiella ;;
  ecoli)        prepare_ecoli ;;
  all)          prepare_tuberculosis; prepare_klebsiella; prepare_ecoli ;;
  *) die "unknown dataset '$1'" ;;
esac
log "done"
