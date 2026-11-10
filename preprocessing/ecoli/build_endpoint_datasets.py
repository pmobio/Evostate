#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按抗生素构建 ecoli2026 的终点数据集 (基因型横截面 + 表型指标)。

数据形式 (已核实)
----------------
ecoli2026 的 SourceData 突变矩阵是**演化终点横截面**: 每行 = 一个 evolved population
的固定突变谱, 没有时间维度。同一菌株的 3 个样本是并行 replicate (共享 strain_group_id)。
真实时间轴在 morbidostat 的 IC50 时序里, 由 `build_phenotype_trajectories.py` 产出。

因此本脚本产出的是"终点表": 每个 population 一行, 左侧是基因型 (0/1 命中矩阵),
右侧是该 population 的表型结局 (initial/final IC50、fold increase、log10 AUC)。
这支持的核心任务是 **genetic background -> 终点突变组合 / 耐药结局**, 而不是演化顺序。

为什么按药物拆分
---------------
三种药的耐药通路差异极大 (folA 仅 TMP, nfsA/nfsB 仅 NIT, mrdA/ompC/envZ 仅 MEC),
混合建模会让药物私有基因互相成为噪声候选, 并掩盖 drug-specific 结构。

输出
----
`derived/endpoint/<DRUG>/`
  population_matrix.csv   population_id + 各基因 0/1
  populations.csv         population 级元数据 + 表型指标
  feature_stats.csv       gene, n_hit, fraction_of_populations, rank
  manifest.json           规模、过滤记录、mutator 报告、突变多样性熵
`<DRUG>` 取 ALL 与各单药。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# 复用已实现的论文版 mutator 识别 (递归 permutation test) 与抗生素过滤, 避免重复实现。
from build_genotype_encodings import DEFAULT_MUTATOR_BACKGROUNDS, drop_mutators, filter_by_antibiotic


# 论文 Fig 4a 中标注为 "mutation rate altering" (#) 的缺失背景。
KNOWN_MUTATOR_BACKGROUNDS = ["mutL", "mutT", "uvrD", "recA", "hfq", "uvrA", "uvrC", "holC"]


def drop_known_mutator_backgrounds(
    long_df: pd.DataFrame,
    meta: pd.DataFrame,
    backgrounds: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """只剔除已知突变率相关的缺失背景 (论文 Fig 4a 的 # 标注)。

    为什么不用论文的 permutation test 作默认: 论文统计的是 WGS 检出的 de novo 点/插失
    突变, 而本数据集的矩阵是**基因级命中** —— 一次大片段缺失或前噬菌体切除会把数十个
    相邻基因同时标为命中 (如 NIT 的 cus 操纵子、ent 铁摄取簇、ybd/ybc 邻域)。这使得
    每个 population 的"突变数"被严重高估且过散, 均匀多项式 null 过窄, 递归检验会把
    近三成 population 判为 mutator (实测 85-89 个, 论文为 20 个), 且无论如何调节
    min_carriers / alpha 都无法收敛到论文的 6/0/14 (NIT 始终 >=17, 论文为 0)。
    因此默认不做统计剔除, 只提供基于论文标注的保守剔除与统计法对照。
    """
    wanted = {b.strip().lower() for b in backgrounds if b.strip()}
    hit = meta["background"].astype(str).str.lower().isin(wanted)
    dropped_ids = set(meta.loc[hit, "sample_id"])
    report = {
        "mode": "known_backgrounds",
        "backgrounds": sorted(wanted),
        "n_populations_dropped": int(len(dropped_ids)),
        "dropped_by_antibiotic": (
            meta.loc[hit].groupby("antibiotic")["sample_id"].nunique().to_dict()
            if "antibiotic" in meta.columns else {}
        ),
    }
    if not dropped_ids:
        return long_df, meta, report
    return (
        long_df[~long_df["sample_id"].isin(dropped_ids)].reset_index(drop=True),
        meta[~hit].reset_index(drop=True),
        report,
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_dataset_dir() -> Path:
    return repo_root() / "data" / "ecoli2026"


def load_genotype(genotype_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_path = genotype_dir / "mutations_long.csv"
    manifest_path = genotype_dir / "sample_manifest.csv"
    for path in (long_path, manifest_path):
        if not path.exists():
            raise SystemExit(
                f"ERROR: 缺少 {path}; 请先运行 scripts/ecoli/extract_supplementary_matrices.py"
            )
    long_df = pd.read_csv(long_path, dtype=str, keep_default_na=False)
    long_df["presence"] = pd.to_numeric(long_df.get("presence", 1), errors="coerce").fillna(0).astype(int)
    meta = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    src_manifest = genotype_dir / "source_matrix_manifest.json"
    if src_manifest.exists():
        payload = json.loads(src_manifest.read_text(encoding="utf-8"))
        if payload.get("has_temporal_axis") is not False:
            raise SystemExit(
                "ERROR: source_matrix_manifest.json 未标注 has_temporal_axis=false, "
                "说明提取脚本版本不一致; 请重新运行 extract_supplementary_matrices.py。"
            )
    return long_df, meta


def load_phenotype(phenotype_dir: Path) -> pd.DataFrame | None:
    path = phenotype_dir / "phenotype_populations.csv"
    if not path.exists():
        return None
    pheno = pd.read_csv(path)
    pheno["plate_id"] = pheno["plate_id"].astype(str)
    keep = [
        "antibiotic", "plate_id", "well", "well_role", "plate_group",
        "n_timepoints", "duration_h", "initial_ic50", "final_ic50", "max_ic50",
        "fold_increase", "log10_ic50_auc", "initial_growth_rate",
        "fold_increase_paper", "final_ic50_paper",
    ]
    keep = [c for c in keep if c in pheno.columns]
    # 每 (药物, 板, 孔) 唯一; 9 板实验与 2023 批次可能重复出现同一孔, 取 9 板实验优先。
    pheno = pheno.sort_values("experiment", key=lambda s: ~s.str.contains("9plates"))
    return pheno[keep].drop_duplicates(subset=["antibiotic", "plate_id", "well"])


def build_matrix(long_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """population x gene 的 0/1 命中矩阵。"""
    hits = long_df[long_df["presence"] > 0]
    matrix = (
        hits.assign(v=1)
        .pivot_table(index="sample_id", columns="feature_id", values="v", aggfunc="max", fill_value=0)
        .astype(int)
    )
    matrix = matrix.reindex(index=meta["sample_id"].tolist(), fill_value=0)
    matrix = matrix.reindex(sorted(matrix.columns), axis=1).fillna(0).astype(int)
    matrix.index.name = "population_id"
    return matrix


def filter_features(matrix: pd.DataFrame, min_carriers: int) -> tuple[pd.DataFrame, dict]:
    """按命中 population 数过滤基因。

    单个大片段缺失或 IS 插入会把上千个相邻基因一起标为命中 (本数据集单样本最高 2405)。
    这类基因只出现在极少数 population 里, 既非收敛信号也无统计功效, 需要剔除。
    """
    n_before = int(matrix.shape[1])
    carriers = matrix.sum(axis=0)
    kept = [g for g in matrix.columns if int(carriers[g]) >= max(1, min_carriers)]
    if not kept:
        raise SystemExit(f"ERROR: min_carriers={min_carriers} 过严, 无基因保留")
    report = {
        "min_carriers": int(min_carriers),
        "n_features_before": n_before,
        "n_features_after": len(kept),
        "n_features_dropped": n_before - len(kept),
        "n_singleton_features": int((carriers == 1).sum()),
    }
    return matrix[kept], report


def feature_stats(matrix: pd.DataFrame) -> pd.DataFrame:
    n_pop = int(matrix.shape[0])
    hits = matrix.sum(axis=0).sort_values(ascending=False)
    out = pd.DataFrame({
        "gene": hits.index,
        "n_hit": hits.to_numpy(dtype=int),
        "fraction_of_populations": (hits.to_numpy(dtype=float) / n_pop) if n_pop else np.nan,
    })
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def mutation_diversity_entropy(stats: pd.DataFrame) -> dict:
    """突变在基因间分布的香农熵 (论文 Fig 3a inset 用它衡量突变多样性)。"""
    counts = stats["n_hit"].to_numpy(dtype=float)
    total = counts.sum()
    if total <= 0:
        return {"entropy_bits": None, "entropy_nats": None, "entropy_normalized": None}
    p = counts[counts > 0] / total
    h_nats = float(-(p * np.log(p)).sum())
    n_genes = int((counts > 0).sum())
    return {
        "entropy_bits": round(h_nats / math.log(2), 4),
        "entropy_nats": round(h_nats, 4),
        "entropy_normalized": round(h_nats / math.log(n_genes), 4) if n_genes > 1 else None,
    }


def write_subset(
    outdir: Path,
    label: str,
    long_df: pd.DataFrame,
    meta: pd.DataFrame,
    pheno: pd.DataFrame | None,
    min_carriers: int,
    mutator_args: dict | None,
) -> dict:
    sub_out = outdir / label
    sub_out.mkdir(parents=True, exist_ok=True)

    mutator_report = None
    if mutator_args is not None:
        mode = mutator_args["mode"]
        if mode == "known_backgrounds":
            long_df, meta, mutator_report = drop_known_mutator_backgrounds(
                long_df, meta, mutator_args["backgrounds"])
        elif mode == "permutation":
            kwargs = {k: v for k, v in mutator_args.items() if k not in ("mode", "backgrounds")}
            long_df, meta, mutator_report = drop_mutators(
                long_df, meta, manual_backgrounds=mutator_args["backgrounds"], **kwargs)
            mutator_report["mode"] = "permutation"

    matrix = build_matrix(long_df, meta)
    matrix, feat_report = filter_features(matrix, min_carriers)
    stats = feature_stats(matrix)

    pops = meta.copy()
    pops = pops[[c for c in [
        "population_id", "strain_group_id", "background", "antibiotic",
        "plate_id", "well", "replicate_index", "endpoint_tag", "n_positive_features",
    ] if c in pops.columns]]
    pops["n_hit_features"] = pops["population_id"].map(matrix.sum(axis=1)).fillna(0).astype(int)
    n_pheno_joined = 0
    if pheno is not None:
        pops = pops.merge(pheno, on=["antibiotic", "plate_id", "well"], how="left")
        n_pheno_joined = int(pops["final_ic50"].notna().sum()) if "final_ic50" in pops.columns else 0

    matrix.reset_index().to_csv(sub_out / "population_matrix.csv", index=False)
    pops.to_csv(sub_out / "populations.csv", index=False)
    stats.to_csv(sub_out / "feature_stats.csv", index=False)

    manifest = {
        "subset": label,
        "unit_of_observation": "evolved population endpoint",
        "has_temporal_axis": False,
        "n_populations": int(matrix.shape[0]),
        "n_strain_groups": int(pops["strain_group_id"].nunique()) if "strain_group_id" in pops else None,
        "n_backgrounds": int(pops["background"].nunique()) if "background" in pops else None,
        "n_features": int(matrix.shape[1]),
        "feature_filter": feat_report,
        "mutator_filter": (
            {k: v for k, v in mutator_report.items() if k != "per_experiment"}
            if mutator_report else None
        ),
        "mutation_diversity": mutation_diversity_entropy(stats),
        "top_genes": stats.head(10)[["gene", "n_hit", "fraction_of_populations"]].to_dict(orient="records"),
        "n_populations_with_phenotype": n_pheno_joined,
    }
    if mutator_report is not None:
        with open(sub_out / "mutator_filter.json", "w", encoding="utf-8") as f:
            json.dump(mutator_report, f, ensure_ascii=False, indent=2)
    with open(sub_out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(
        "[ecoli][endpoint] %-4s populations=%-4d strain_groups=%-4s backgrounds=%-4s features=%-5d "
        "(dropped %d) entropy=%s pheno_joined=%d"
        % (
            label, manifest["n_populations"], manifest["n_strain_groups"],
            manifest["n_backgrounds"], manifest["n_features"],
            feat_report["n_features_dropped"],
            manifest["mutation_diversity"]["entropy_bits"], n_pheno_joined,
        )
    )
    if mutator_report is not None:
        print("             mutator[%s] dropped=%s" % (
            mutator_report.get("mode"),
            mutator_report.get("n_populations_dropped", mutator_report.get("n_mutators_total"))))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Build per-drug endpoint datasets for ecoli2026")
    ap.add_argument("--dataset-dir", type=Path, default=default_dataset_dir())
    ap.add_argument("--genotype-dir", type=Path, default=None,
                    help="默认 <dataset-dir>/derived/genotype_route_b")
    ap.add_argument("--phenotype-dir", type=Path, default=None,
                    help="默认 <dataset-dir>/derived/phenotype")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="默认 <dataset-dir>/derived/endpoint")
    ap.add_argument("--drugs", type=str, default="MEC,NIT,TMP",
                    help="逗号分隔的抗生素列表, 每个生成一个子集 (默认 MEC,NIT,TMP)")
    ap.add_argument("--no-all", action="store_true", help="不生成合并的 ALL 子集")
    ap.add_argument("--min-carriers", type=int, default=2,
                    help="仅保留被至少 N 个 population 命中的基因 (默认 2)")
    ap.add_argument("--mutator-mode", choices=["off", "known_backgrounds", "permutation"],
                    default="off",
                    help=("mutator 处理方式: off=不剔除 (默认, 本数据集下统计法无法标定); "
                          "known_backgrounds=只剔除论文标注的突变率相关背景; "
                          "permutation=论文的递归置换检验 (会过度剔除, 仅供对照)"))
    ap.add_argument("--mutator-backgrounds", type=str, default=",".join(KNOWN_MUTATOR_BACKGROUNDS),
                    help="known_backgrounds 模式使用的背景列表; permutation 模式下作为手动补入项")
    ap.add_argument("--mutator-min-carriers", type=int, default=2)
    ap.add_argument("--mutator-surrogates", type=int, default=100000)
    ap.add_argument("--mutator-alpha", type=float, default=0.05)
    ap.add_argument("--mutator-seed", type=int, default=0)
    args = ap.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    genotype_dir = (args.genotype_dir or dataset_dir / "derived" / "genotype_route_b").resolve()
    phenotype_dir = (args.phenotype_dir or dataset_dir / "derived" / "phenotype").resolve()
    outdir = (args.outdir or dataset_dir / "derived" / "endpoint").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    long_df, meta = load_genotype(genotype_dir)
    pheno = load_phenotype(phenotype_dir)
    if pheno is None:
        print("[ecoli][endpoint] WARN: 未找到表型汇总, 将只输出基因型部分; "
              "请先运行 scripts_repr_evo/ecoli/01_prepare_phenotype_trajectories.sh")

    mutator_args = None
    if args.mutator_mode != "off":
        mutator_args = {
            "mode": args.mutator_mode,
            "backgrounds": [b for b in str(args.mutator_backgrounds or "").split(",")],
            "unit": "sample",
            "min_carriers": args.mutator_min_carriers,
            "n_surrogates": args.mutator_surrogates,
            "alpha": args.mutator_alpha,
            "seed": args.mutator_seed,
        }

    subsets: list[str] = []
    if not args.no_all:
        subsets.append("ALL")
    subsets.extend([d.strip().upper() for d in args.drugs.split(",") if d.strip()])

    manifests = []
    for label in subsets:
        if label == "ALL":
            manifests.append(write_subset(
                outdir, "ALL", long_df, meta, pheno, args.min_carriers, mutator_args))
            continue
        sub_long, sub_meta = filter_by_antibiotic(long_df, meta, [label])
        manifests.append(write_subset(
            outdir, label, sub_long, sub_meta, pheno, args.min_carriers, mutator_args))

    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "ecoli2026",
            "unit_of_observation": "evolved population endpoint",
            "has_temporal_axis": False,
            "genotype_dir": str(genotype_dir),
            "phenotype_dir": str(phenotype_dir) if pheno is not None else None,
            "subsets": manifests,
        }, f, ensure_ascii=False, indent=2)
    print("[ecoli][endpoint] done -> %s" % outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
