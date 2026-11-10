#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从标准化突变长表生成 Route B 所需的双 state encoding。

产物结构:
  outdir/
    source_registry.json
    monotonic/
      category_barcode.csv
      samples.csv
      evam_matrix.csv
      evam_rownames.csv
      feature_hierarchy.csv
      transitions.csv                # HyperTraPS-compatible gain-only
      manifest.json
    full_temporal/
      category_barcode.csv
      samples.csv
      evam_matrix.csv
      evam_rownames.csv
      feature_hierarchy.csv
      transitions.csv                # 可包含 loss, 供自建模型使用
      manifest.json

编码定义:
- monotonic: 轨迹内按时间累计 gain，loss 不回退；用于 HyperTraPS 和与我方模型平行评估；
- full_temporal: 保留每个时间点的真实状态变化，可出现 loss；用于我方时序模型。
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_REGISTRY = {
    "paper_title": "Function-specific epistasis shapes evolutionary trajectories towards antibiotic resistance",
    "journal": "Nature Communications",
    "year": 2026,
    "zotero_raw_phenotype_accession": "2008295081",
    "ena_study_accession": "PRJEB103832",
    "processed_mutational_data": "Supplementary Data 1",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_input() -> Path:
    return repo_root() / "data" / "ecoli2026" / "derived" / "genotype_route_b" / "mutations_long.csv"


def default_output() -> Path:
    return repo_root() / "data" / "ecoli2026" / "derived" / "genotype_route_b"


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: 输入长表缺少必要列: {missing}")


def load_long_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    require_columns(df, ["sample_id", "trajectory_id", "time_order", "feature_id"])
    for c in ["presence", "time_order"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    if "event_type" not in df.columns:
        df["event_type"] = "snapshot"
    df = df.sort_values(["trajectory_id", "time_order", "sample_id", "feature_id"]).reset_index(drop=True)
    return df


def load_sidecar_sample_manifest(long_path: Path) -> pd.DataFrame | None:
    sidecar = long_path.parent / "sample_manifest.csv"
    if not sidecar.exists():
        return None
    meta = pd.read_csv(sidecar, dtype=str, keep_default_na=False)
    require_columns(meta, ["sample_id", "trajectory_id", "time_order"])
    meta["time_order"] = pd.to_numeric(meta["time_order"], errors="coerce").fillna(0).astype(int)
    return meta


def parse_antibiotics(raw: str | None) -> list[str] | None:
    """解析 --antibiotic 参数, 返回大写去空的抗生素列表 (None=不过滤)。"""
    if not raw:
        return None
    values = [v.strip().upper() for v in raw.split(",") if v.strip()]
    return values or None


def filter_by_antibiotic(
    long_df: pd.DataFrame,
    sidecar_meta: pd.DataFrame | None,
    antibiotics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """按抗生素筛选长表与 sidecar manifest。

    动机: 三种药物 (MEC/NIT/TMP) 的耐药通路差异极大 (folA 只在 TMP, nfsA/nfsB 只在
    NIT, mrdA/ompC 只在 MEC)。混合建模会让药物私有特征互相成为噪声候选, 并掩盖
    drug-specific 的演化顺序结构, 因此提供按药物独立建模的入口。
    """
    source = sidecar_meta if sidecar_meta is not None else long_df
    if "antibiotic" not in source.columns:
        raise SystemExit("ERROR: 输入缺少 antibiotic 列, 无法按药物拆分")

    available = sorted(source["antibiotic"].astype(str).str.upper().unique().tolist())
    unknown = [ab for ab in antibiotics if ab not in available]
    if unknown:
        raise SystemExit(f"ERROR: 未知抗生素 {unknown}; 可选: {available}")

    keep = source["antibiotic"].astype(str).str.upper().isin(antibiotics)
    keep_samples = set(source.loc[keep, "sample_id"].astype(str))

    long_out = long_df[long_df["sample_id"].astype(str).isin(keep_samples)].reset_index(drop=True)
    meta_out = None
    if sidecar_meta is not None:
        meta_out = sidecar_meta[sidecar_meta["sample_id"].astype(str).isin(keep_samples)].reset_index(drop=True)
    if long_out.empty:
        raise SystemExit(f"ERROR: 按抗生素 {antibiotics} 过滤后长表为空")
    return long_out, meta_out


DEFAULT_MUTATOR_BACKGROUNDS = "mutL,mutT,uvrD"


def mutation_counts(long_df: pd.DataFrame, unit: str, min_carriers: int) -> pd.Series:
    """按指定单元统计突变数。

    unit="sample": 每个 sample 即论文中的一个 evolved population, 突变数 = 命中基因数。
        这是论文 Methods 的口径 (Supplementary Table 1 逐 population 列出)。
    unit="trajectory": 每条轨迹内所有时间点的并集, 仅在输入确为时序数据时有意义。

    min_carriers: 计数时只保留被至少 N 个 sample 携带的基因。单个大片段缺失或 IS
        插入会连带上千个相邻基因 (本数据集单样本最高 2405), 若不排除这类单例基因,
        它们会完全支配 mutator 检验。论文只统计映射到基因/启动子的功能性突变,
        本过滤是与之对齐的近似。
    """
    present = long_df
    if "presence" in present.columns:
        present = present[present["presence"].astype(int) > 0]
    if min_carriers > 1:
        carriers = present.groupby("feature_id")["sample_id"].nunique()
        keep = set(carriers[carriers >= min_carriers].index)
        present = present[present["feature_id"].isin(keep)]
    key = "sample_id" if unit == "sample" else "trajectory_id"
    return present.groupby(key)["feature_id"].nunique().astype(int)


def _permutation_mutator_step(
    counts: np.ndarray,
    n_surrogates: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[int | None, float]:
    """单轮 permutation test: 返回最高突变数样本的下标 (若显著) 与其 p 值。

    Null model: 把总突变数 K 均匀随机地分配到 N 个 population (multinomial),
    统计 M 个 surrogate 中 "最大突变数 >= 实测最大值" 的个数 n,
    p = (n + 1) / (M + 1)。
    """
    n_samples = int(counts.size)
    total = int(counts.sum())
    if n_samples < 2 or total <= 0:
        return None, 1.0
    observed_max = int(counts.max())
    # 分块生成 surrogate, 避免 M 很大时一次性分配巨大数组。
    chunk = max(1, min(n_surrogates, 20000))
    n_exceed = 0
    remaining = n_surrogates
    probs = np.full(n_samples, 1.0 / n_samples)
    while remaining > 0:
        size = min(chunk, remaining)
        draws = rng.multinomial(total, probs, size=size)
        n_exceed += int((draws.max(axis=1) >= observed_max).sum())
        remaining -= size
    p_value = (n_exceed + 1) / (n_surrogates + 1)
    if p_value < alpha:
        return int(np.argmax(counts)), p_value
    return None, p_value


def identify_mutators(
    unit_meta: pd.DataFrame,
    counts: pd.Series,
    unit_key: str,
    n_surrogates: int,
    alpha: float,
    seed: int,
    manual_backgrounds: list[str],
) -> tuple[list[str], dict]:
    """按论文方法逐个抗生素递归识别 mutator population。

    识别出 mutator 后, null model 中的突变率被高估, 因此剔除已识别 mutator 后
    重复检验, 直到不再有新的 mutator (论文 Methods)。
    额外支持手动补入已知错配修复缺失背景 (论文对 NIT 的 ΔuvrD 做了同样处理)。
    """
    rng = np.random.default_rng(seed)
    manual_set = {b.strip().lower() for b in manual_backgrounds if b.strip()}
    has_ab = "antibiotic" in unit_meta.columns
    report: dict = {
        "method": "recursive_permutation_test",
        "unit": unit_key,
        "n_surrogates": int(n_surrogates),
        "alpha": float(alpha),
        "seed": int(seed),
        "manual_mutator_backgrounds": sorted(manual_set),
        "per_experiment": [],
    }
    mutators: list[str] = []

    groups = unit_meta.groupby("antibiotic", sort=True) if has_ab else [("ALL", unit_meta)]
    for ab, sub in groups:
        unit_ids = sub[unit_key].tolist()
        active = {u: int(counts.get(u, 0)) for u in unit_ids}
        detected: list[dict] = []
        stop_p_value: float | None = None

        while len(active) >= 2:
            ids = list(active.keys())
            arr = np.array([active[u] for u in ids], dtype=np.int64)
            idx, p_value = _permutation_mutator_step(arr, n_surrogates, alpha, rng)
            if idx is None:
                stop_p_value = p_value
                break
            hit = ids[idx]
            detected.append({
                unit_key: hit,
                "n_mutations": int(active[hit]),
                "p_value": p_value,
                "reason": "permutation_test",
            })
            active.pop(hit)

        # 手动补入: 已知 mutator 背景但未达显著 (突变尚未积累够) 的 population。
        if manual_set:
            for _, row in sub.iterrows():
                uid = row[unit_key]
                bg = str(row.get("background", "")).strip().lower()
                if bg in manual_set and all(d[unit_key] != uid for d in detected):
                    detected.append({
                        unit_key: uid,
                        "n_mutations": int(counts.get(uid, 0)),
                        "p_value": None,
                        "reason": "known_mutator_background",
                    })

        report["per_experiment"].append({
            "antibiotic": str(ab),
            "n_units": len(unit_ids),
            "n_mutators": len(detected),
            "stop_p_value": stop_p_value,
            "mutators": detected,
        })
        mutators.extend(d[unit_key] for d in detected)

    report["n_mutators_total"] = len(mutators)
    report["mutator_units"] = sorted(mutators)
    return mutators, report


def drop_mutators(
    long_df: pd.DataFrame,
    sidecar_meta: pd.DataFrame | None,
    unit: str,
    min_carriers: int,
    n_surrogates: int,
    alpha: float,
    seed: int,
    manual_backgrounds: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """识别并剔除 mutator population。

    动机: mutator 会带来数十到数百个随机突变, 既污染特征表也严重扭曲演化顺序的
    估计。论文同样在所有下游分析前剔除了 20 个 mutator population
    (NIT 0 / MEC 6 / TMP 14)。
    """
    unit_key = "sample_id" if unit == "sample" else "trajectory_id"
    source = sidecar_meta if sidecar_meta is not None else long_df
    meta_cols = [c for c in [unit_key, "antibiotic", "background"] if c in source.columns]
    unit_meta = source[meta_cols].drop_duplicates(subset=[unit_key]).reset_index(drop=True)
    counts = mutation_counts(long_df, unit=unit, min_carriers=min_carriers)

    mutators, report = identify_mutators(
        unit_meta, counts, unit_key=unit_key, n_surrogates=n_surrogates,
        alpha=alpha, seed=seed, manual_backgrounds=manual_backgrounds,
    )
    report["min_carriers_for_counting"] = int(min_carriers)
    if not mutators:
        report["n_samples_dropped"] = 0
        return long_df, sidecar_meta, report

    mutator_set = set(mutators)
    long_before = len(long_df)
    long_out = long_df[~long_df[unit_key].isin(mutator_set)].reset_index(drop=True)
    meta_out = sidecar_meta
    n_dropped_samples = 0
    if sidecar_meta is not None:
        n_dropped_samples = int(sidecar_meta[unit_key].isin(mutator_set).sum())
        meta_out = sidecar_meta[~sidecar_meta[unit_key].isin(mutator_set)].reset_index(drop=True)
    if long_out.empty:
        raise SystemExit("ERROR: 剔除 mutator 后长表为空, 请检查 --mutator-alpha / --mutator-backgrounds")

    report["n_samples_dropped"] = n_dropped_samples
    report["n_long_rows_dropped"] = long_before - len(long_out)
    return long_out, meta_out, report


def build_sample_meta(df: pd.DataFrame, sidecar_meta: pd.DataFrame | None = None) -> pd.DataFrame:
    if sidecar_meta is not None:
        meta = sidecar_meta.copy()
    else:
        meta_cols = [c for c in [
            "sample_id", "trajectory_id", "background", "antibiotic", "replicate", "timepoint", "time_order"
        ] if c in df.columns]
        meta = df[meta_cols].drop_duplicates(subset=["sample_id"]).copy()
    meta = meta.drop_duplicates(subset=["sample_id"]).copy()
    sort_keys = [k for k in ["trajectory_id", "time_order", "sample_id"] if k in meta.columns]
    meta = meta.sort_values(sort_keys).reset_index(drop=True)
    # grouped CV / parallelism 检验的正确分组单元是"菌株"(含其 3 个平行 replicate),
    # 而不是单个 population; strain_group_id 存在时优先使用。
    if "strain_group_id" in meta.columns:
        meta["group_id"] = meta["strain_group_id"]
        meta["lineage_group"] = meta["strain_group_id"]
    else:
        meta["group_id"] = meta["trajectory_id"]
        meta["lineage_group"] = meta["trajectory_id"]
    meta["source_dataset"] = "ecoli2026"
    return meta


def base_presence_matrix(df: pd.DataFrame, sample_meta: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    snapshot = df.copy()
    if "presence" not in snapshot.columns:
        snapshot["presence"] = 1
    snapshot = snapshot.assign(_presence=snapshot["presence"].astype(int).clip(lower=0, upper=1))
    matrix = (
        snapshot.groupby(["sample_id", "feature_id"], as_index=False)["_presence"]
        .max()
        .pivot(index="sample_id", columns="feature_id", values="_presence")
        .fillna(0)
        .astype(int)
    )
    matrix = matrix.reindex(index=sample_meta["sample_id"].tolist(), columns=features, fill_value=0)
    matrix.index.name = "sample_id"
    return matrix


def monotonic_matrix(base: pd.DataFrame, sample_meta: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=base.index, columns=base.columns, data=0, dtype=int)
    for traj, sub in sample_meta.groupby("trajectory_id", sort=False):
        ordered_ids = sub.sort_values(["time_order", "sample_id"])["sample_id"].tolist()
        running = pd.Series(0, index=base.columns, dtype=int)
        for sid in ordered_ids:
            running = running.combine(base.loc[sid], max)
            out.loc[sid] = running.astype(int)
    out.index.name = "sample_id"
    return out.astype(int)


def feature_hierarchy(features: list[str], matrix: pd.DataFrame, reversible_mode: str) -> pd.DataFrame:
    reversible_set = set()
    if reversible_mode == "observed_losses" and len(matrix) > 1:
        diff = matrix.diff().fillna(0)
        reversible_set = {feat for feat in features if (diff[feat] < 0).any()}
    rows = []
    for feat in features:
        rows.append({
            "feature_id": feat,
            "layer": "mutation_marker",
            "drug_class": "unknown",
            "mechanism": "genotype_event",
            "location": "chromosome",
            "reversible": int(feat in reversible_set),
        })
    return pd.DataFrame(rows)


def transitions_from_matrix(matrix: pd.DataFrame, sample_meta: pd.DataFrame, strategy: str) -> tuple[pd.DataFrame, dict]:
    rows = []
    gain_counter = Counter()
    loss_counter = Counter()
    for traj, sub in sample_meta.groupby("trajectory_id", sort=False):
        ordered = sub.sort_values(["time_order", "sample_id"])
        sample_ids = ordered["sample_id"].tolist()
        if len(sample_ids) < 2:
            continue
        for src_id, dst_id in zip(sample_ids[:-1], sample_ids[1:]):
            src = matrix.loc[src_id].astype(int)
            dst = matrix.loc[dst_id].astype(int)
            gained = [feat for feat in matrix.columns if src[feat] == 0 and dst[feat] == 1]
            lost = [feat for feat in matrix.columns if src[feat] == 1 and dst[feat] == 0]
            if strategy == "monotonic":
                lost = []
            if not gained and not lost:
                continue
            rows.append({
                "src_sample_id": src_id,
                "dest_sample_id": dst_id,
                "trajectory_id": traj,
                "src_time_order": int(ordered.loc[ordered["sample_id"] == src_id, "time_order"].iloc[0]),
                "dest_time_order": int(ordered.loc[ordered["sample_id"] == dst_id, "time_order"].iloc[0]),
                "src_state": "".join(src.astype(str).tolist()),
                "dest_state": "".join(dst.astype(str).tolist()),
                "n_gained": len(gained),
                "gained_features": ";".join(gained),
                "n_lost": len(lost),
                "lost_features": ";".join(lost),
                "group_id": traj,
                "transition_strategy": "irreversible" if strategy == "monotonic" else "full_temporal",
            })
            for feat in gained:
                gain_counter[feat] += 1
            for feat in lost:
                loss_counter[feat] += 1
    out = pd.DataFrame(rows, columns=[
        "src_sample_id", "dest_sample_id", "trajectory_id", "src_time_order", "dest_time_order",
        "src_state", "dest_state", "n_gained", "gained_features",
        "n_lost", "lost_features", "group_id", "transition_strategy",
    ])
    summary = {
        "n_informative_transitions": int(len(out)),
        "n_multi_gain_transitions": int((out["n_gained"] > 1).sum()) if not out.empty else 0,
        "n_loss_transitions": int((out["n_lost"] > 0).sum()) if not out.empty else 0,
        "gained_feature_counts": dict(gain_counter.most_common()),
        "lost_feature_counts": dict(loss_counter.most_common()),
    }
    return out, summary


def select_features(
    base: pd.DataFrame,
    sample_meta: pd.DataFrame,
    features: list[str],
    min_carriers: int,
    min_prevalence: float,
    min_gain_events: int,
) -> tuple[list[str], dict]:
    """按携带样本数与增益事件频次过滤特征。

    动机: ecoli2026 原始编码含 3799 个特征但仅 311 个样本, 且大部分特征从未作为
    增益事件出现, 导致排序任务候选空间被极度稀释 (top-k 恒为 0)。这里提供两类
    与标签无关或弱相关的频次过滤:

    - min_carriers / min_prevalence: 基于基因型携带率的过滤 (类似 MAF 过滤);
    - min_gain_events: 要求特征在单调编码中至少发生 N 次增益, 否则该特征在
      grouped CV 下几乎不可能被学习到, 只会作为噪声候选存在。
    """
    n_samples = int(base.shape[0])
    carriers = base.sum(axis=0)

    report: dict = {
        "n_samples": n_samples,
        "n_features_input": len(features),
        "min_carriers": int(min_carriers),
        "min_prevalence": float(min_prevalence),
        "min_gain_events": int(min_gain_events),
        "stages": [],
    }

    kept = list(features)

    carrier_floor = int(min_carriers)
    if min_prevalence > 0:
        carrier_floor = max(carrier_floor, math.ceil(min_prevalence * n_samples))

    if carrier_floor > 0:
        before = len(kept)
        kept = [f for f in kept if int(carriers.get(f, 0)) >= carrier_floor]
        report["stages"].append({
            "filter": "carriers",
            "threshold": carrier_floor,
            "n_before": before,
            "n_after": len(kept),
            "n_dropped": before - len(kept),
        })

    gain_counts: dict[str, int] = {}
    if kept:
        mono_all = monotonic_matrix(base[kept], sample_meta)
        _, summary_all = transitions_from_matrix(mono_all, sample_meta, strategy="monotonic")
        gain_counts = {str(k): int(v) for k, v in summary_all.get("gained_feature_counts", {}).items()}

    report["n_features_ever_gained"] = int(sum(1 for f in kept if gain_counts.get(f, 0) > 0))
    report["n_features_never_gained"] = int(len(kept) - report["n_features_ever_gained"])
    report["n_features_singleton_gain"] = int(sum(1 for f in kept if gain_counts.get(f, 0) == 1))

    if min_gain_events > 0:
        before = len(kept)
        kept = [f for f in kept if gain_counts.get(f, 0) >= min_gain_events]
        report["stages"].append({
            "filter": "gain_events",
            "threshold": int(min_gain_events),
            "n_before": before,
            "n_after": len(kept),
            "n_dropped": before - len(kept),
        })

    if not kept:
        raise SystemExit(
            "ERROR: 特征过滤后为空, 请降低 --min-carriers/--min-prevalence/--min-gain-events 阈值"
        )

    report["n_features_kept"] = len(kept)
    report["kept_features"] = kept
    report["gain_counts_kept"] = {f: gain_counts.get(f, 0) for f in kept}
    return kept, report


def write_encoding(outdir: Path, name: str, sample_meta: pd.DataFrame, matrix: pd.DataFrame, transitions: pd.DataFrame,
                   transition_summary: dict, hierarchy: pd.DataFrame, hypertraps_compatible: bool,
                   feature_filter: dict | None = None) -> None:
    enc_dir = outdir / name
    enc_dir.mkdir(parents=True, exist_ok=True)

    barcode = matrix.reset_index()
    barcode.to_csv(enc_dir / "category_barcode.csv", index=False)
    sample_meta.to_csv(enc_dir / "samples.csv", index=False)
    matrix.to_csv(enc_dir / "evam_matrix.csv", index=False)
    pd.DataFrame({"sample_id": matrix.index.tolist()}).to_csv(enc_dir / "evam_rownames.csv", index=False)
    hierarchy.to_csv(enc_dir / "feature_hierarchy.csv", index=False)
    transitions.to_csv(enc_dir / "transitions.csv", index=False)

    manifest = {
        "encoding": name,
        "hypertraps_compatible": hypertraps_compatible,
        "n_samples": int(sample_meta["sample_id"].nunique()),
        "n_trajectories": int(sample_meta["trajectory_id"].nunique()),
        "n_features": int(matrix.shape[1]),
        "feature_columns": list(matrix.columns),
        **transition_summary,
    }
    if feature_filter is not None:
        manifest["feature_filter"] = {
            k: v for k, v in feature_filter.items() if k not in ("kept_features", "gain_counts_kept")
        }
    with open(enc_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build monotonic and full-temporal genotype encodings")
    ap.add_argument("--input", type=Path, default=default_input(), help="标准化突变长表路径")
    ap.add_argument("--outdir", type=Path, default=default_output(), help="输出目录")
    ap.add_argument("--min-carriers", type=int, default=0,
                    help="仅保留至少被 N 个样本携带的特征 (0=关闭)")
    ap.add_argument("--min-prevalence", type=float, default=0.0,
                    help="仅保留携带率不低于该比例的特征 (0=关闭), 与 --min-carriers 取更严者")
    ap.add_argument("--min-gain-events", type=int, default=3,
                    help="仅保留在单调编码中至少发生 N 次增益的特征 (0=关闭, 默认 3)")
    ap.add_argument("--antibiotic", type=str, default=None,
                    help="仅使用指定抗生素的样本, 逗号分隔 (如 MEC 或 MEC,NIT); 默认使用全部")
    ap.add_argument("--exclude-mutators", action="store_true",
                    help="按论文方法 (递归 permutation test) 识别并剔除 mutator population")
    ap.add_argument("--mutator-unit", choices=["sample", "trajectory"], default="sample",
                    help=("mutator 判定单元: sample=每行即一个 evolved population (论文口径, 默认); "
                          "trajectory=轨迹内并集 (仅当输入确为时序数据时使用)"))
    ap.add_argument("--mutator-min-carriers", type=int, default=2,
                    help=("mutator 计数时只统计被至少 N 个 sample 携带的基因 (默认 2); "
                          "用于排除大片段缺失/IS 插入连带的上千个单例邻近基因"))
    ap.add_argument("--mutator-surrogates", type=int, default=100000,
                    help="mutator permutation test 的 surrogate 数 M (默认 1e5, 同论文)")
    ap.add_argument("--mutator-alpha", type=float, default=0.05,
                    help="mutator 判定显著性阈值 (默认 0.05, 同论文)")
    ap.add_argument("--mutator-seed", type=int, default=0, help="mutator permutation test 随机种子")
    ap.add_argument("--mutator-backgrounds", type=str, default=DEFAULT_MUTATOR_BACKGROUNDS,
                    help=("手动补入的已知 mutator 背景基因, 逗号分隔 (默认 %s); "
                          "留空则关闭手动补入" % DEFAULT_MUTATOR_BACKGROUNDS))
    args = ap.parse_args()

    antibiotics = parse_antibiotics(args.antibiotic)

    long_df = load_long_table(args.input)
    sidecar_meta = load_sidecar_sample_manifest(args.input)

    # 输入若无时间轴 (每条 trajectory 只有 1 个样本), 构造转移会得到空表, 下游
    # 排序/似然评估都无意义。这里显式失败并指向真实时序数据, 避免静默产出垃圾。
    probe = sidecar_meta if sidecar_meta is not None else long_df
    if probe.groupby("trajectory_id")["sample_id"].nunique().max() < 2:
        raise SystemExit(
            "ERROR: 输入没有时间轴 (每条 trajectory 仅 1 个样本), 无法构造演化转移。\n"
            "       ecoli2026 的突变矩阵是演化终点横截面 (详见 source_matrix_manifest.json 的\n"
            "       has_temporal_axis=false)。真实时间轴在表型数据里, 请使用\n"
            "       scripts/ecoli/build_phenotype_trajectories.py。"
        )

    if antibiotics:
        long_df, sidecar_meta = filter_by_antibiotic(long_df, sidecar_meta, antibiotics)

    mutator_report: dict | None = None
    if args.exclude_mutators:
        long_df, sidecar_meta, mutator_report = drop_mutators(
            long_df,
            sidecar_meta,
            unit=args.mutator_unit,
            min_carriers=args.mutator_min_carriers,
            n_surrogates=args.mutator_surrogates,
            alpha=args.mutator_alpha,
            seed=args.mutator_seed,
            manual_backgrounds=[b for b in str(args.mutator_backgrounds or "").split(",")],
        )

    sample_meta = build_sample_meta(long_df, sidecar_meta=sidecar_meta)
    all_features = sorted(long_df["feature_id"].astype(str).unique().tolist())

    base_all = base_presence_matrix(long_df, sample_meta, all_features)
    features, filter_report = select_features(
        base_all,
        sample_meta,
        all_features,
        min_carriers=args.min_carriers,
        min_prevalence=args.min_prevalence,
        min_gain_events=args.min_gain_events,
    )

    full_matrix = base_all[features].copy()
    full_matrix.index.name = "sample_id"
    mono_matrix = monotonic_matrix(full_matrix, sample_meta)

    mono_trans, mono_summary = transitions_from_matrix(mono_matrix, sample_meta, strategy="monotonic")
    full_trans, full_summary = transitions_from_matrix(full_matrix, sample_meta, strategy="full_temporal")

    args.outdir.mkdir(parents=True, exist_ok=True)
    if mutator_report is not None:
        filter_report["mutator_filter"] = {
            k: v for k, v in mutator_report.items() if k != "per_experiment"
        }
        with open(args.outdir / "mutator_filter.json", "w", encoding="utf-8") as f:
            json.dump(mutator_report, f, ensure_ascii=False, indent=2)
    if antibiotics:
        filter_report["antibiotics"] = antibiotics
    if antibiotics or mutator_report is not None:
        # 让派生目录自成一体, 便于下游脚本/notebook 直接指向它。
        long_df.to_csv(args.outdir / "mutations_long.csv", index=False)
        if sidecar_meta is not None:
            sidecar_meta.to_csv(args.outdir / "sample_manifest.csv", index=False)
    with open(args.outdir / "source_registry.json", "w", encoding="utf-8") as f:
        json.dump(SOURCE_REGISTRY, f, ensure_ascii=False, indent=2)
    with open(args.outdir / "feature_filter.json", "w", encoding="utf-8") as f:
        json.dump(filter_report, f, ensure_ascii=False, indent=2)

    write_encoding(
        args.outdir,
        "monotonic",
        sample_meta,
        mono_matrix,
        mono_trans,
        mono_summary,
        feature_hierarchy(features, mono_matrix, reversible_mode="none"),
        hypertraps_compatible=True,
        feature_filter=filter_report,
    )
    write_encoding(
        args.outdir,
        "full_temporal",
        sample_meta,
        full_matrix,
        full_trans,
        full_summary,
        feature_hierarchy(features, full_matrix, reversible_mode="observed_losses"),
        hypertraps_compatible=False,
        feature_filter=filter_report,
    )

    for stage in filter_report["stages"]:
        print(
            "[ecoli] filter %s>=%s: %d -> %d (dropped %d)"
            % (stage["filter"], stage["threshold"], stage["n_before"], stage["n_after"], stage["n_dropped"])
        )
    if mutator_report is not None:
        unit_key = mutator_report["unit"]
        for exp in mutator_report["per_experiment"]:
            print(
                "[ecoli] mutator %s: %d/%d %ss excluded%s"
                % (
                    exp["antibiotic"], exp["n_mutators"], exp["n_units"], unit_key,
                    "" if not exp["mutators"] else " -> " + ", ".join(
                        "%s(n=%d,%s)" % (
                            m[unit_key], m["n_mutations"],
                            "p=%.1e" % m["p_value"] if m["p_value"] is not None else m["reason"],
                        )
                        for m in exp["mutators"]
                    ),
                )
            )
        print("[ecoli] mutator total excluded units=%d samples=%d"
              % (mutator_report["n_mutators_total"], mutator_report["n_samples_dropped"]))
    print(
        "[ecoli] encodings built: antibiotic=%s samples=%d trajectories=%d features=%d/%d transitions(mono)=%d -> %s"
        % (
            ",".join(antibiotics) if antibiotics else "ALL",
            sample_meta["sample_id"].nunique(),
            sample_meta["trajectory_id"].nunique(),
            len(features),
            len(all_features),
            len(mono_trans),
            args.outdir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
