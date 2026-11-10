#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prep_microreact_kp.py —— 把 Microreact (Lam2021 "Global Klebsiella") 的 data.csv
适配成 EvoAMR 建模输入。

流程 (对齐论文 Methods 的二值化口径):
  1. 按 species 列筛 K. pneumoniae sensu stricto (默认 "Klebsiella pneumoniae");
  2. 对 22 个 Kleborate 决定簇列做二值化: 值为 '-'/空 -> 0, 否则(存在任意基因/突变) -> 1;
  3. 产出 barcode / 元数据 / evam 矩阵 / 报告, 直接喂 build_transitions.py --tree (模式A)。

与 refs/AMRGlobal + BV-BRC cgMLST 那条链路完全独立, 不改动原 step1~4。

产物 (写到 --outdir):
  category_barcode.csv : sample_id + 22 个 0/1 特征  (喂 build_transitions --id-col sample_id)
  samples.csv          : sample_id + 元数据(Country/Year/ST/...)  (供分组CV / 国家分层)
  evam_matrix.csv      : 仅 22 个 0/1 特征(无 id, 横截面 EvAM 备用)
  prep_report.json     : 计数 / 阳性率 / 丢弃特征 / 唯一barcode 等

用法:
  python3 prep_microreact_kp.py --csv data/kleborate/2021-Klebsiella.csv \
      --outdir data/processed/microreact_kp_kleborate22 \
      --species "Klebsiella pneumoniae"
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

# 论文 / Kleborate 的 22 个耐药"字符"列 (与 kleborate-dichotomized.csv 同口径)
AMR22 = [
    "AGly_acquired", "Col_acquired", "Fcyn_acquired", "Flq_acquired", "Gly_acquired",
    "MLS_acquired", "Phe_acquired", "Rif_acquired", "Sul_acquired", "Tet_acquired",
    "Tgc_acquired", "Tmt_acquired", "Bla_acquired", "Bla_inhR_acquired", "Bla_ESBL_acquired",
    "Bla_ESBL_inhR_acquired", "Bla_Carb_acquired", "Bla_chr", "SHV_mutations",
    "Omp_mutations", "Col_mutations", "Flq_mutations",
]
# 被判为"无/敏感"的取值 (Kleborate 用 '-' 表示无)
NEG = {"", "-", "na", "nan", "none", "null", "0"}
# 想保留到 samples.csv 的元数据列 (存在才取)
META = [
    "Country", "Region", "Year", "Sample", "Classification", "Convergence_event",
    "species", "species_match", "ST", "wzi", "K_locus", "O_locus", "O_type",
    "virulence_score", "resistance_score", "latitude", "longitude",
]


def dichotomize(v) -> int:
    """'-'/空/缺失 -> 0; 存在任意基因或突变 -> 1。"""
    return 0 if str(v).strip().lower() in NEG else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Microreact Kp data.csv -> EvoAMR barcode/samples")
    ap.add_argument("--csv", required=True, help="Microreact 导出的 data.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--species", default="Klebsiella pneumoniae",
                    help="按 species 列精确筛选; 传空串 '' 则不筛(保留全部菌种)")
    ap.add_argument("--species-col", default="species")
    ap.add_argument("--min-prevalence", type=float, default=0.0,
                    help=">0 时丢弃阳性率 < 该值的稀疏特征 (默认 0 = 保留全部 22, 对齐论文 L=22)")
    ap.add_argument("--max-prevalence", type=float, default=1.0,
                    help="<1 时丢弃阳性率 > 该值的特征")
    ap.add_argument("--drop-all-zero", action="store_true",
                    help="仅影响 evam_matrix.csv: 丢弃全敏感(全0)行; barcode/samples 不丢")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    n_total = len(df)

    if args.id_col not in df.columns:
        raise SystemExit(f"ERROR: 找不到 id 列 '{args.id_col}'; 现有列前几个: {list(df.columns)[:6]}")

    # 1. 筛物种 (sensu stricto)
    if args.species:
        if args.species_col not in df.columns:
            raise SystemExit(f"ERROR: 找不到 species 列 '{args.species_col}'")
        df = df[df[args.species_col] == args.species].copy()
    n_kept = len(df)
    if n_kept == 0:
        raise SystemExit(f"ERROR: species='{args.species}' 筛完 0 条, 请检查取值")

    # 2. 二值化 22 列
    missing = [c for c in AMR22 if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: data.csv 缺少 AMR 列: {missing}")
    B = df[AMR22].apply(lambda s: s.map(dichotomize))
    B.columns = AMR22
    prevalence = {c: round(float(B[c].mean()), 6) for c in AMR22}

    # 稀疏/恒定特征过滤 (默认不过滤, 保留 22)
    keep = list(AMR22)
    dropped = []
    if args.min_prevalence > 0.0 or args.max_prevalence < 1.0:
        keep = [c for c in AMR22 if args.min_prevalence <= prevalence[c] <= args.max_prevalence]
        dropped = [c for c in AMR22 if c not in keep]
    B = B[keep]

    ids = df[args.id_col].astype(str).tolist()

    # category_barcode.csv: sample_id + features
    bar = B.copy()
    bar.insert(0, "sample_id", ids)
    bar.to_csv(os.path.join(args.outdir, "category_barcode.csv"), index=False)

    # samples.csv: sample_id + 元数据
    meta_cols = [c for c in META if c in df.columns]
    sm = df[meta_cols].copy()
    sm.insert(0, "sample_id", ids)
    sm.to_csv(os.path.join(args.outdir, "samples.csv"), index=False)

    # evam_matrix.csv: 仅特征 (可选丢全零)
    ev = B.copy()
    if args.drop_all_zero:
        ev = ev[ev.sum(axis=1) > 0]
    ev.to_csv(os.path.join(args.outdir, "evam_matrix.csv"), index=False)

    n_all_zero = int((B.sum(axis=1) == 0).sum())
    n_unique = int(B.drop_duplicates().shape[0])
    report = {
        "source_csv": os.path.abspath(args.csv),
        "species_filter": args.species,
        "id_col": args.id_col,
        "n_total": n_total,
        "n_kept": n_kept,
        "n_features_in": len(AMR22),
        "n_features_kept": len(keep),
        "dropped_features": dropped,
        "feature_columns": keep,
        "prevalence": prevalence,
        "n_all_zero_barcodes": n_all_zero,
        "n_unique_barcodes": n_unique,
        "meta_columns": meta_cols,
    }
    with open(os.path.join(args.outdir, "prep_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("[prep] species='%s': 保留 %d/%d 条" % (args.species, n_kept, n_total))
    print("[prep] 特征 %d/%d (丢弃 %d: %s)" % (len(keep), len(AMR22), len(dropped), dropped))
    print("[prep] all-zero(全敏感)=%d  唯一barcode=%d" % (n_all_zero, n_unique))
    print("[prep] 产物 -> %s (category_barcode.csv / samples.csv / evam_matrix.csv / prep_report.json)"
          % args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
