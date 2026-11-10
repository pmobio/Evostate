#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prep_tuberculosis_ht.py —— 把 HyperTraPS 论文附带的 tuberculosis dataset
适配成 EvoAMR 建模输入。

数据源:
  - refs/HyperTraPS/datasets/tuberculosis/tuberculosis-v5-header-29.csv
  - refs/HyperTraPS/datasets/tuberculosis/ng.2878-S2.txt

与 step1_dataset.sh / work.sh 的 AMRGlobal-cgMLST 主链路不同, 这里走一条
“外部数据集 -> 直接生成 barcode/samples/tree -> build_transitions --tree”的独立支线,
对齐 stepK_microreact.sh 的扩展模式。

默认流程:
  1. 读取论文附带的二值化 TB 特征矩阵 (默认 29-feature 版本);
  2. 读取 Casali et al. 论文树(Newick), 去引号并裁剪到矩阵覆盖的 isolates;
  3. 生成 EvoAMR 所需的 category_barcode.csv / samples.csv / evam_matrix.csv;
  4. 生成一份简单的 feature_hierarchy.csv, 供 build_transitions / train_dl 自动定位;
  5. 依据剪枝后的系统树做 clade 递归切分, 产出 grouped CV 可用的 lineage_group;
  6. 写出 tree_pruned.nwk 与 prep_report.json。

产物 (写到 --outdir):
  category_barcode.csv : sample_id + 0/1 特征
  samples.csv          : sample_id + lineage_group + 元信息
  phylo_groups.csv     : sample_id -> lineage_group 映射
  evam_matrix.csv      : 仅特征矩阵 (无 id)
  evam_rownames.csv    : evam_matrix 对应样本顺序
  feature_hierarchy.csv: 特征元数据 / reversible 先验
  tree_pruned.nwk      : 仅保留 matched isolates 的系统树
  prep_report.json     : 样本匹配、group 切分、特征统计摘要
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
from Bio import Phylo


FEATURE_META: Dict[str, Dict[str, object]] = {
    # mutation / region markers
    "katG": {"layer": "mutation_marker", "drug_class": "isoniazid", "mechanism": "resistance_mutation", "location": "chromosome", "reversible": 0},
    "PinhA": {"layer": "mutation_marker", "drug_class": "isoniazid", "mechanism": "resistance_mutation", "location": "chromosome", "reversible": 0},
    "RRDR": {"layer": "mutation_marker", "drug_class": "rifampicin", "mechanism": "resistance_mutation", "location": "chromosome", "reversible": 0},
    "PpncA": {"layer": "mutation_marker", "drug_class": "pyrazinamide", "mechanism": "resistance_mutation", "location": "chromosome", "reversible": 0},
    "pncA": {"layer": "mutation_marker", "drug_class": "pyrazinamide", "mechanism": "resistance_mutation", "location": "chromosome", "reversible": 0},
    "PembA": {"layer": "mutation_marker", "drug_class": "ethambutol", "mechanism": "resistance_mutation", "location": "chromosome", "reversible": 0},
    "embB": {"layer": "mutation_marker", "drug_class": "ethambutol", "mechanism": "resistance_mutation", "location": "chromosome", "reversible": 0},
    "rpsL": {"layer": "mutation_marker", "drug_class": "streptomycin", "mechanism": "resistance_mutation", "location": "chromosome", "reversible": 0},
    "rrs_str": {"layer": "mutation_marker", "drug_class": "streptomycin", "mechanism": "resistance_mutation", "location": "chromosome", "reversible": 0},
    "gidB": {"layer": "mutation_marker", "drug_class": "streptomycin", "mechanism": "resistance_mutation", "location": "chromosome", "reversible": 0},
    "rrs_inj": {"layer": "mutation_marker", "drug_class": "second_line_injectable", "mechanism": "resistance_mutation", "location": "chromosome", "reversible": 0},
    "Peis": {"layer": "mutation_marker", "drug_class": "second_line_injectable", "mechanism": "resistance_mutation", "location": "chromosome", "reversible": 0},
    "gyrA": {"layer": "mutation_marker", "drug_class": "fluoroquinolone", "mechanism": "resistance_mutation", "location": "chromosome", "reversible": 0},
    "gyrB": {"layer": "mutation_marker", "drug_class": "fluoroquinolone", "mechanism": "resistance_mutation", "location": "chromosome", "reversible": 0},
    "PethA": {"layer": "mutation_marker", "drug_class": "thioamide", "mechanism": "resistance_mutation", "location": "chromosome", "reversible": 0},
    "ethA": {"layer": "mutation_marker", "drug_class": "thioamide", "mechanism": "resistance_mutation", "location": "chromosome", "reversible": 0},
    "rpoA": {"layer": "mutation_marker", "drug_class": "rifampicin", "mechanism": "compensatory_mutation", "location": "chromosome", "reversible": 0},
    "rpoC": {"layer": "mutation_marker", "drug_class": "rifampicin", "mechanism": "compensatory_mutation", "location": "chromosome", "reversible": 0},
    "rpoB": {"layer": "mutation_marker", "drug_class": "rifampicin", "mechanism": "resistance_mutation", "location": "chromosome", "reversible": 0},
    # phenotype features
    "INH": {"layer": "drug_phenotype", "drug_class": "isoniazid", "mechanism": "resistance_phenotype", "location": "chromosome", "reversible": 0},
    "RIF": {"layer": "drug_phenotype", "drug_class": "rifampicin", "mechanism": "resistance_phenotype", "location": "chromosome", "reversible": 0},
    "PZA": {"layer": "drug_phenotype", "drug_class": "pyrazinamide", "mechanism": "resistance_phenotype", "location": "chromosome", "reversible": 0},
    "EMB": {"layer": "drug_phenotype", "drug_class": "ethambutol", "mechanism": "resistance_phenotype", "location": "chromosome", "reversible": 0},
    "STR": {"layer": "drug_phenotype", "drug_class": "streptomycin", "mechanism": "resistance_phenotype", "location": "chromosome", "reversible": 0},
    "AMI": {"layer": "drug_phenotype", "drug_class": "amikacin", "mechanism": "resistance_phenotype", "location": "chromosome", "reversible": 0},
    "CAP": {"layer": "drug_phenotype", "drug_class": "capreomycin", "mechanism": "resistance_phenotype", "location": "chromosome", "reversible": 0},
    "MOX": {"layer": "drug_phenotype", "drug_class": "moxifloxacin", "mechanism": "resistance_phenotype", "location": "chromosome", "reversible": 0},
    "OFL": {"layer": "drug_phenotype", "drug_class": "ofloxacin", "mechanism": "resistance_phenotype", "location": "chromosome", "reversible": 0},
    "PRO": {"layer": "drug_phenotype", "drug_class": "prothionamide", "mechanism": "resistance_phenotype", "location": "chromosome", "reversible": 0},
}


def normalize_leaf_name(name: str | None) -> str:
    nm = (name or "").strip()
    if len(nm) >= 2 and nm[0] in ("'", '"') and nm[-1] == nm[0]:
        nm = nm[1:-1]
    return nm


def load_binary_matrix(path: str, id_col: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if id_col not in df.columns:
        raise SystemExit(f"ERROR: 找不到 id 列 '{id_col}'")

    feat_cols = [c for c in df.columns if c != id_col]
    if not feat_cols:
        raise SystemExit("ERROR: 矩阵里没有特征列")

    if df[id_col].astype(str).duplicated().any():
        dup = df.loc[df[id_col].astype(str).duplicated(), id_col].astype(str).tolist()[:10]
        raise SystemExit(f"ERROR: id 列存在重复样本: {dup}")

    out = df.copy()
    out[id_col] = out[id_col].astype(str)
    for col in feat_cols:
        vals = pd.to_numeric(out[col], errors="raise")
        bad = sorted(set(vals.dropna().astype(int).tolist()) - {0, 1})
        if bad:
            raise SystemExit(f"ERROR: 特征列 {col} 出现非二值取值: {bad}")
        out[col] = vals.astype(int)
    return out, feat_cols


def load_tree(tree_path: str, tree_format: str):
    tree = Phylo.read(tree_path, tree_format)
    raw_leaves = []
    for term in tree.get_terminals():
        term.name = normalize_leaf_name(term.name)
        raw_leaves.append(term.name)
    if len(raw_leaves) != len(set(raw_leaves)):
        dup = [k for k, v in Counter(raw_leaves).items() if v > 1][:10]
        raise SystemExit(f"ERROR: 树叶标签存在重复: {dup}")
    return tree, raw_leaves


def prune_tree_to_ids(tree, keep_ids: set[str]):
    for term in list(tree.get_terminals()):
        if term.name not in keep_ids:
            tree.prune(term)
    return tree


def collect_leaf_names(clade) -> List[str]:
    return [normalize_leaf_name(term.name) for term in clade.get_terminals()]


def build_lineage_groups(tree, max_group_size: int) -> Tuple[Dict[str, str], Dict[str, int]]:
    if max_group_size <= 0:
        raise SystemExit("ERROR: --max-group-size 必须 > 0")

    parts = []

    def visit(clade):
        n = len(clade.get_terminals())
        if n <= max_group_size or len(clade.clades) < 2:
            parts.append(clade)
            return
        for child in clade.clades:
            visit(child)

    visit(tree.root)

    leaf_to_group: Dict[str, str] = {}
    group_sizes: Dict[str, int] = {}
    for idx, clade in enumerate(parts, start=1):
        gid = f"clade_{idx:03d}"
        leaves = collect_leaf_names(clade)
        group_sizes[gid] = len(leaves)
        for leaf in leaves:
            if leaf in leaf_to_group:
                raise SystemExit(f"ERROR: 叶 {leaf} 被重复分配到多个 lineage_group")
            leaf_to_group[leaf] = gid
    return leaf_to_group, group_sizes


def build_feature_hierarchy(feat_cols: List[str]) -> pd.DataFrame:
    rows = []
    for feat in feat_cols:
        meta = FEATURE_META.get(feat, {
            "layer": "unknown",
            "drug_class": "unknown",
            "mechanism": "unknown",
            "location": "chromosome",
            "reversible": 0,
        })
        rows.append({"feature_id": feat, **meta})
    return pd.DataFrame(rows, columns=[
        "feature_id", "layer", "drug_class", "mechanism", "location", "reversible",
    ])


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    tb_root = os.path.join(root, "refs", "HyperTraPS", "datasets", "tuberculosis")

    ap = argparse.ArgumentParser(description="HyperTraPS tuberculosis dataset -> EvoAMR barcode/samples/tree")
    ap.add_argument("--matrix", default=os.path.join(tb_root, "tuberculosis-v5-header-29.csv"),
                    help="TB 二值特征矩阵 (默认 29-feature 版本)")
    ap.add_argument("--tree", default=os.path.join(tb_root, "ng.2878-S2.txt"),
                    help="Casali et al. 论文 Newick 树")
    ap.add_argument("--tree-format", default="newick")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--id-col", default="Isolate")
    ap.add_argument("--max-group-size", type=int, default=40,
                    help="按 clade 递归切分 lineage_group 时允许的最大组大小")
    ap.add_argument("--drop-all-zero", action="store_true",
                    help="仅影响 evam_matrix / evam_rownames: 丢弃全0样本; barcode/samples 不丢")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    matrix_df, feat_cols = load_binary_matrix(args.matrix, args.id_col)
    tree, raw_leaves = load_tree(args.tree, args.tree_format)

    matrix_ids = matrix_df[args.id_col].astype(str).tolist()
    matrix_id_set = set(matrix_ids)
    tree_leaf_set = set(raw_leaves)
    matched_ids = matrix_id_set & tree_leaf_set
    if not matched_ids:
        raise SystemExit("ERROR: 特征矩阵与系统树没有任何重叠样本")

    matrix_only = sorted(matrix_id_set - tree_leaf_set)
    tree_only = sorted(tree_leaf_set - matrix_id_set)

    prune_tree_to_ids(tree, matched_ids)
    pruned_order = [normalize_leaf_name(term.name) for term in tree.get_terminals()]
    pruned_order = [sid for sid in pruned_order if sid in matched_ids]

    matrix_sub = matrix_df.set_index(args.id_col).loc[pruned_order, feat_cols].copy()
    matrix_sub.index.name = "sample_id"

    leaf_to_group, group_sizes = build_lineage_groups(tree, args.max_group_size)
    if set(pruned_order) != set(leaf_to_group):
        missing = sorted(set(pruned_order) - set(leaf_to_group))[:10]
        raise SystemExit(f"ERROR: 有样本未分配 lineage_group: {missing}")

    # 1) category_barcode.csv
    barcode = matrix_sub.reset_index()
    barcode.to_csv(os.path.join(args.outdir, "category_barcode.csv"), index=False)

    # 2) samples.csv / phylo_groups.csv
    n_positive = matrix_sub.sum(axis=1).astype(int)
    samples = pd.DataFrame({
        "sample_id": pruned_order,
        "group_id": [leaf_to_group[sid] for sid in pruned_order],
        "lineage_group": [leaf_to_group[sid] for sid in pruned_order],
        "lineage_group_size": [group_sizes[leaf_to_group[sid]] for sid in pruned_order],
        "tree_leaf_order": list(range(1, len(pruned_order) + 1)),
        "n_positive_features": [int(n_positive.loc[sid]) for sid in pruned_order],
        "feature_set_name": os.path.splitext(os.path.basename(args.matrix))[0],
        "source_dataset": "HyperTraPS_tuberculosis",
    })
    samples.to_csv(os.path.join(args.outdir, "samples.csv"), index=False)
    samples[["sample_id", "group_id", "lineage_group_size"]].to_csv(
        os.path.join(args.outdir, "phylo_groups.csv"), index=False
    )

    # 3) evam matrix
    evam = matrix_sub.copy()
    if args.drop_all_zero:
        evam = evam[evam.sum(axis=1) > 0]
    evam.to_csv(os.path.join(args.outdir, "evam_matrix.csv"), index=False)
    pd.DataFrame({"sample_id": evam.index.tolist()}).to_csv(
        os.path.join(args.outdir, "evam_rownames.csv"), index=False
    )

    # 4) feature hierarchy
    hierarchy = build_feature_hierarchy(feat_cols)
    hierarchy.to_csv(os.path.join(args.outdir, "feature_hierarchy.csv"), index=False)

    # 5) pruned tree
    tree_path = os.path.join(args.outdir, "tree_pruned.nwk")
    Phylo.write(tree, tree_path, args.tree_format)

    group_size_series = pd.Series(list(group_sizes.values()), dtype=int)
    report = {
        "source_matrix": os.path.abspath(args.matrix),
        "source_tree": os.path.abspath(args.tree),
        "tree_format": args.tree_format,
        "matrix_id_col": args.id_col,
        "feature_set_name": os.path.splitext(os.path.basename(args.matrix))[0],
        "feature_columns": feat_cols,
        "n_features": len(feat_cols),
        "n_matrix_rows_raw": int(len(matrix_df)),
        "n_tree_leaves_raw": int(len(raw_leaves)),
        "n_matched": int(len(matched_ids)),
        "n_matrix_only": int(len(matrix_only)),
        "n_tree_only": int(len(tree_only)),
        "matrix_only_examples": matrix_only[:20],
        "tree_only_examples": tree_only[:20],
        "n_pruned_tree_leaves": int(len(pruned_order)),
        "lineage_group_column": "lineage_group",
        "lineage_max_group_size": int(args.max_group_size),
        "n_lineage_groups": int(len(group_sizes)),
        "lineage_group_size_summary": {
            "min": int(group_size_series.min()),
            "median": float(group_size_series.median()),
            "max": int(group_size_series.max()),
        },
        "largest_lineage_groups": [
            {"group_id": gid, "size": int(sz)}
            for gid, sz in sorted(group_sizes.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
        ],
        "n_all_zero_barcodes": int((matrix_sub.sum(axis=1) == 0).sum()),
        "n_unique_barcodes": int(matrix_sub.drop_duplicates().shape[0]),
        "feature_hierarchy": os.path.abspath(os.path.join(args.outdir, "feature_hierarchy.csv")),
        "pruned_tree": os.path.abspath(tree_path),
        "category_barcode": os.path.abspath(os.path.join(args.outdir, "category_barcode.csv")),
        "samples": os.path.abspath(os.path.join(args.outdir, "samples.csv")),
        "evam_matrix": os.path.abspath(os.path.join(args.outdir, "evam_matrix.csv")),
    }
    with open(os.path.join(args.outdir, "prep_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[tb-prep] matched {len(pruned_order)}/{len(matrix_df)} isolates to tree leaves")
    print(f"[tb-prep] features={len(feat_cols)} groups={len(group_sizes)} max_group_size={args.max_group_size}")
    print(f"[tb-prep] outputs -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
