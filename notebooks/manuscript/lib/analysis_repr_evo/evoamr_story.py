#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evoamr_story.py —— repr_evo 结果的"耐药演化演进故事"分析工具库

设计目标（对齐 Aga2026 / PLoS Biol 的分析范式）:
  1. 数据层  : 样本谱 (UpSet 式 profile)、特征流行率、状态复杂度分布            -> Fig 1B/1C(i)
  2. 顺序层  : 获得顺序 bubble plot、期望获得次序 (expected ordering)          -> Fig 1C(iii) / Fig 2A
  3. 驱动层  : 期望次序 vs 流行率回归, 区分"频率驱动" vs "依赖驱动"             -> Fig 4D
  4. 分歧层  : 跨谱系 (lineage/clade/ST) 的顺序一致性与分歧特征                -> Fig 3 / Fig 4C
  5. 预测层  : 方法对比、逐折分布、配对显著性                                  -> 模型评估
  6. 路径层  : 由训练好的两阶段模型前向游走得到的模型推断 bubble 与特征交互      -> Fig 1C(ii) / interaction graph

所有绘图标签使用英文 (避免缺失中文字体导致的方框), 叙述性文字在 notebook 的 markdown 中用中文。
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# --------------------------------------------------------------------------------------
# 路径与数据集登记
# --------------------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
EVOAMR_ROOT = os.path.dirname(HERE)                       # .../EvoAMR/evoamr
DATA_ROOT = os.path.join(EVOAMR_ROOT, "data")
PROCESSED_ROOT = os.path.join(DATA_ROOT, "processed")
RESULTS_ROOT = os.path.join(DATA_ROOT, "results_repr_evo")
MODELS_DIR = os.path.join(EVOAMR_ROOT, "models")

AMRGLOBAL_SUFFIX = "drug-class-gene-mge-plasmid_arg_cgmlst_hc10"
AMRGLOBAL_SPECIES = [
    "Acinetobacter_baumannii", "Clostridioides_difficile", "Corynebacterium_diphtheriae",
    "Enterococcus_faecium", "Escherichia_coli", "Klebsiella_pneumoniae",
    "Mycobacterium_tuberculosis", "Neisseria_gonorrhoeae", "Pseudomonas_aeruginosa",
    "Salmonella_enterica", "Shigella_flexneri", "Shigella_sonnei",
    "Staphylococcus_aureus", "Streptococcus_agalactiae", "Streptococcus_pneumoniae",
    "Streptococcus_suis",
]


@dataclass
class DatasetSpec:
    """一个待分析数据集的路径与语义描述。"""
    key: str
    label_cn: str
    label_en: str
    processed_dir: str
    results_dir: str
    transitions_path: str
    group_name_cn: str = "谱系分组"
    notes: str = ""
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def figures_dir(self) -> str:
        return os.path.join(self.results_dir, "figures")

    def ensure_figures_dir(self) -> str:
        os.makedirs(self.figures_dir, exist_ok=True)
        return self.figures_dir

    def eval_summaries(self) -> Dict[str, str]:
        """返回 {实验名: eval_summary.json 路径}，实验名取相对 results_dir 的目录名。"""
        found: Dict[str, str] = {}
        for root, _dirs, files in os.walk(self.results_dir):
            if "eval_summary.json" in files:
                rel = os.path.relpath(root, self.results_dir)
                name = rel.replace(os.sep + "eval", "").replace(os.sep, "/")
                found[name] = os.path.join(root, "eval_summary.json")
        return dict(sorted(found.items()))

    def stage2_model_path(self) -> Optional[str]:
        p = os.path.join(self.results_dir, "default", "stage2", "model.pt")
        return p if os.path.isfile(p) else None


def amrglobal_processed_dir(species: str) -> str:
    return os.path.join(PROCESSED_ROOT, f"amrglobal_{species}_{AMRGLOBAL_SUFFIX}")


def dataset_registry() -> Dict[str, DatasetSpec]:
    """所有可分析数据集的登记表。"""
    reg: Dict[str, DatasetSpec] = {}

    for key, label_cn, label_en in [
        ("tuberculosis_antibiotics10", "结核-抗生素类别层 (10 特征)", "M. tuberculosis / drug-class layer (L=10)"),
        ("tuberculosis_genes19", "结核-耐药基因层 (19 特征)", "M. tuberculosis / gene layer (L=19)"),
        ("tuberculosis_combined29", "结核-基因+药物混合层 (29 特征)", "M. tuberculosis / combined layer (L=29)"),
    ]:
        proc = os.path.join(PROCESSED_ROOT, key)
        reg[key] = DatasetSpec(
            key=key, label_cn=label_cn, label_en=label_en,
            processed_dir=proc,
            results_dir=os.path.join(RESULTS_ROOT, key),
            transitions_path=os.path.join(proc, "transitions", "transitions.csv"),
            group_name_cn="系统发育进化枝 (clade)",
            notes="HyperTraPS TB benchmark, 395 株, 祖先状态重建得到的不可逆获得转移。",
        )

    proc = os.path.join(PROCESSED_ROOT, "microreact_kp_kleborate22")
    reg["microreact_kp_kleborate22"] = DatasetSpec(
        key="microreact_kp_kleborate22",
        label_cn="肺炎克雷伯菌-Kleborate 22 类耐药 character",
        label_en="K. pneumoniae / Kleborate 22 characters",
        processed_dir=proc,
        results_dir=os.path.join(RESULTS_ROOT, "microreact_kp_kleborate22"),
        transitions_path=os.path.join(proc, "transitions", "transitions.csv"),
        group_name_cn="ST / LIN 谱系",
        notes="与 Aga2026 (PLoS Biol) 同口径的 22 类 character，本项目最大规模的单物种转移集。",
    )

    proc = os.path.join(DATA_ROOT, "ecoli2026", "derived", "genotype_route_b", "monotonic")
    reg["ecoli2026_route_b"] = DatasetSpec(
        key="ecoli2026_route_b",
        label_cn="大肠杆菌-泛基因组基因层 (Route-B monotonic)",
        label_en="E. coli / pan-genome gene layer (Route-B)",
        processed_dir=proc,
        results_dir=os.path.join(RESULTS_ROOT, "ecoli2026_route_b", "monotonic"),
        transitions_path=os.path.join(proc, "transitions.csv"),
        group_name_cn="PopPUNK 聚类",
        notes="超高维 (L≈3799) 低样本设定，用于界定方法的适用边界。",
    )

    for sp in AMRGLOBAL_SPECIES:
        key = f"amrglobal16/{sp}"
        proc = amrglobal_processed_dir(sp)
        reg[key] = DatasetSpec(
            key=key,
            label_cn=f"AMRGlobal16-{sp.replace('_', ' ')}",
            label_en=f"AMRGlobal16 / {sp.replace('_', ' ')}",
            processed_dir=proc,
            results_dir=os.path.join(RESULTS_ROOT, "amrglobal16", sp),
            transitions_path=os.path.join(proc, "transitions", "transitions.csv"),
            group_name_cn="cgMLST HC10 聚类",
            notes="ARG + MGE + 质粒 ARG + 药物类别的多层特征空间。",
            extra={"species": sp},
        )
    return reg


# --------------------------------------------------------------------------------------
# 1. 数据层
# --------------------------------------------------------------------------------------

def load_feature_names(processed_dir: str, expected_L: Optional[int] = None) -> List[str]:
    """特征名优先取 feature_hierarchy.csv 的 feature_id，退化到 evam_matrix.csv 表头。"""
    hier = os.path.join(processed_dir, "feature_hierarchy.csv")
    if os.path.isfile(hier):
        names = pd.read_csv(hier)["feature_id"].astype(str).tolist()
        if expected_L is None or len(names) == expected_L:
            return names
    mat = os.path.join(processed_dir, "evam_matrix.csv")
    if os.path.isfile(mat):
        cols = pd.read_csv(mat, nrows=0).columns.tolist()
        cols = [c for c in cols if c.lower() not in {"sample_id", "id", "name"}]
        if expected_L is None or len(cols) == expected_L:
            return [str(c) for c in cols]
    if expected_L is None:
        raise FileNotFoundError(f"无法解析特征名: {processed_dir}")
    return [f"f{i:03d}" for i in range(expected_L)]


def load_feature_hierarchy(processed_dir: str) -> Optional[pd.DataFrame]:
    hier = os.path.join(processed_dir, "feature_hierarchy.csv")
    return pd.read_csv(hier) if os.path.isfile(hier) else None


@dataclass
class TransitionSet:
    """解析后的转移集合。"""
    frame: pd.DataFrame          # 原始行 + 解析列
    src: np.ndarray              # (n, L) int8
    dest: np.ndarray             # (n, L) int8
    gained: List[List[int]]      # 每行的获得特征索引
    feature_names: List[str]

    @property
    def L(self) -> int:
        return self.src.shape[1]

    @property
    def n(self) -> int:
        return self.src.shape[0]


def _bits_to_array(series: pd.Series) -> np.ndarray:
    arr = np.array([[int(c) for c in str(s)] for s in series], dtype=np.int8)
    return arr


def load_transition_set(
    transitions_path: str,
    processed_dir: str,
    strategy: str = "irreversible",
) -> TransitionSet:
    df = pd.read_csv(transitions_path, dtype={"src_state": str, "dest_state": str})
    if "transition_strategy" in df.columns and strategy:
        sub = df[df["transition_strategy"] == strategy]
        if len(sub) > 0:
            df = sub
    df = df.reset_index(drop=True)
    src = _bits_to_array(df["src_state"])
    dest = _bits_to_array(df["dest_state"])
    names = load_feature_names(processed_dir, expected_L=src.shape[1])
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    gained: List[List[int]] = []
    for i, raw in enumerate(df.get("gained_features", pd.Series([""] * len(df)))):
        idx: List[int] = []
        if isinstance(raw, str) and raw.strip():
            for token in raw.split(";"):
                token = token.strip()
                if token in name_to_idx:
                    idx.append(name_to_idx[token])
        if not idx:  # 名称对不上时退回位向量差
            idx = np.flatnonzero((dest[i] == 1) & (src[i] == 0)).tolist()
        gained.append(idx)
    df["n_src_features"] = src.sum(axis=1)
    df["n_gained_parsed"] = [len(g) for g in gained]
    return TransitionSet(frame=df, src=src, dest=dest, gained=gained, feature_names=names)


def transition_overview(ts: TransitionSet) -> pd.DataFrame:
    """转移集合的基本画像。"""
    n_multi = int((ts.frame["n_gained_parsed"] > 1).sum())
    rows = [
        ("n_transitions", ts.n),
        ("n_features (L)", ts.L),
        ("n_groups", int(ts.frame["group_id"].nunique()) if "group_id" in ts.frame else np.nan),
        ("single-gain transitions", int((ts.frame["n_gained_parsed"] == 1).sum())),
        ("multi-gain transitions", n_multi),
        ("multi-gain fraction", round(n_multi / max(1, ts.n), 3)),
        ("mean |gained| per transition", round(float(ts.frame["n_gained_parsed"].mean()), 3)),
        ("mean |src| (already acquired)", round(float(ts.frame["n_src_features"].mean()), 3)),
        ("mean candidate-set size", round(float((ts.L - ts.frame["n_src_features"]).mean()), 3)),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def feature_stats(ts: TransitionSet) -> pd.DataFrame:
    """每个特征的流行率与获得事件计数。"""
    prevalence = ts.dest.mean(axis=0)
    gain_counts = np.zeros(ts.L, dtype=int)
    for g in ts.gained:
        for i in g:
            gain_counts[i] += 1
    at_risk = (ts.src == 0).sum(axis=0)
    return pd.DataFrame({
        "feature": ts.feature_names,
        "prevalence_dest": prevalence,
        "n_gain_events": gain_counts,
        "n_at_risk": at_risk,
        "gain_rate_per_opportunity": np.where(at_risk > 0, gain_counts / np.maximum(at_risk, 1), np.nan),
    }).sort_values("n_gain_events", ascending=False).reset_index(drop=True)


def plot_state_profiles(ts: TransitionSet, top_n: int = 25, save: Optional[str] = None):
    """UpSet 式：最常见的耐药 profile 计数（bar 颜色 = 携带特征数）。对齐 Aga2026 Fig 1B。"""
    profiles = ["".join(map(str, row)) for row in ts.dest]
    vc = pd.Series(profiles).value_counts().head(top_n)
    n_acq = [sum(int(c) for c in p) for p in vc.index]
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = plt.cm.viridis(np.array(n_acq) / max(1, max(n_acq)))
    ax.bar(range(len(vc)), vc.values, color=colors)
    ax.set_xticks(range(len(vc)))
    ax.set_xticklabels([f"P{i+1}" for i in range(len(vc))], rotation=90, fontsize=7)
    ax.set_xlabel(f"top-{top_n} resistance profiles (descendant states)")
    ax.set_ylabel("count")
    ax.set_title("Most common AMR profiles (color = #acquired characters)")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, max(1, max(n_acq))))
    fig.colorbar(sm, ax=ax, label="#acquired")
    if save:
        fig.savefig(save)
    return fig, vc


def plot_complexity_distribution(ts: TransitionSet, save: Optional[str] = None):
    """演化"深度"分布：源状态已获得特征数 vs 单步获得数。"""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    axes[0].hist(ts.frame["n_src_features"], bins=min(30, ts.L + 1), color="#4C72B0")
    axes[0].set_xlabel("#already acquired in source state")
    axes[0].set_ylabel("#transitions")
    axes[0].set_title("Evolutionary depth of transitions")

    vc = ts.frame["n_gained_parsed"].value_counts().sort_index()
    axes[1].bar(vc.index, vc.values, color="#DD8452")
    axes[1].set_xlabel("#features gained in one step")
    axes[1].set_ylabel("#transitions")
    axes[1].set_title("Single- vs multi-gain events")

    cand = ts.L - ts.frame["n_src_features"]
    axes[2].scatter(ts.frame["n_src_features"], cand, s=8, alpha=0.4, color="#55A868")
    axes[2].set_xlabel("#already acquired")
    axes[2].set_ylabel("candidate-set size |C(s)|")
    axes[2].set_title("Candidate-set shrinkage along evolution")
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


# --------------------------------------------------------------------------------------
# 2. 顺序层：经验 bubble plot 与期望获得次序
# --------------------------------------------------------------------------------------

def empirical_bubble_matrix(ts: TransitionSet, max_steps: Optional[int] = None) -> pd.DataFrame:
    """
    经验获得顺序矩阵 (feature × ordering step)。

    对每条转移，源状态已含 k 个特征，则本次获得事件发生在演化序位 step = k + 1
    （多获得时 |G| 个特征平分权重）。逐特征行归一化后即为
    "该特征在第 step 位被获得" 的经验概率，与 HyperTraPS bubble plot 语义一致。
    """
    L = ts.L
    steps = L if max_steps is None else min(L, max_steps)
    mat = np.zeros((L, steps), dtype=float)
    for row_i, g in enumerate(ts.gained):
        if not g:
            continue
        step = int(ts.frame["n_src_features"].iloc[row_i]) + 1
        if step > steps:
            step = steps
        w = 1.0 / len(g)
        for f in g:
            mat[f, step - 1] += w
    row_sum = mat.sum(axis=1, keepdims=True)
    norm = np.divide(mat, row_sum, out=np.zeros_like(mat), where=row_sum > 0)
    return pd.DataFrame(norm, index=ts.feature_names, columns=[f"step{i+1}" for i in range(steps)])


def expected_ordering(bubble: pd.DataFrame) -> pd.DataFrame:
    """期望获得次序 = Σ step · p(step)；同时给出分布宽度（sd）。"""
    steps = np.arange(1, bubble.shape[1] + 1, dtype=float)
    p = bubble.values
    mass = p.sum(axis=1)
    mean = np.divide((p * steps).sum(axis=1), mass, out=np.full(len(p), np.nan), where=mass > 0)
    var = np.divide((p * (steps[None, :] - mean[:, None]) ** 2).sum(axis=1), mass,
                    out=np.full(len(p), np.nan), where=mass > 0)
    out = pd.DataFrame({
        "feature": bubble.index,
        "expected_order": mean,
        "order_sd": np.sqrt(var),
        "total_mass": mass,
    })
    return out.sort_values("expected_order").reset_index(drop=True)


def plot_bubble(bubble: pd.DataFrame, title: str, top_features: Optional[int] = 30,
                order_by_expected: bool = True, save: Optional[str] = None):
    """bubble plot：气泡面积 = 该特征在该序位被获得的概率，颜色由早到晚。"""
    b = bubble
    if order_by_expected:
        eo = expected_ordering(b).dropna(subset=["expected_order"])
        b = b.loc[eo["feature"].tolist()]
    if top_features is not None and len(b) > top_features:
        keep = bubble.sum(axis=1).sort_values(ascending=False).head(top_features).index
        b = b.loc[[f for f in b.index if f in set(keep)]]
    n_f, n_s = b.shape
    fig, ax = plt.subplots(figsize=(max(5.0, 0.42 * n_s + 2.5), max(3.2, 0.28 * n_f + 1.4)))
    xs, ys, sizes, cols = [], [], [], []
    for r, feat in enumerate(b.index):
        for c in range(n_s):
            v = b.iloc[r, c]
            if v > 1e-4:
                xs.append(c + 1)
                ys.append(n_f - r)
                sizes.append(v * 620)
                cols.append(c + 1)
    sc = ax.scatter(xs, ys, s=sizes, c=cols, cmap="viridis", alpha=0.85, edgecolors="none")
    ax.set_yticks(range(1, n_f + 1))
    ax.set_yticklabels(list(b.index)[::-1], fontsize=7)
    ax.set_xlabel("acquisition ordering (step)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=ax, label="ordering")
    if save:
        fig.savefig(save)
    return fig


# --------------------------------------------------------------------------------------
# 3. 驱动层：期望次序 vs 流行率
# --------------------------------------------------------------------------------------

def _linreg(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """最小二乘 + 斜率零假设 t 检验（不依赖 scipy）。"""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3 or np.allclose(x, x[0]):
        return {"n": n, "slope": np.nan, "intercept": np.nan, "r": np.nan, "r2": np.nan, "p_value": np.nan}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r = float(np.corrcoef(x, y)[0, 1])
    if ss_res <= 0 or n <= 2:
        p = 0.0
    else:
        se = math.sqrt(ss_res / (n - 2) / float(((x - x.mean()) ** 2).sum()))
        t = slope / se if se > 0 else np.inf
        # 双侧 p 值的正态近似（n 较小时略保守，够用于定性判断）
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))
    return {"n": n, "slope": float(slope), "intercept": float(intercept), "r": r, "r2": float(r2), "p_value": float(p)}


def order_vs_prevalence(ts: TransitionSet, bubble: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    eo = expected_ordering(bubble)
    fs = feature_stats(ts)[["feature", "prevalence_dest", "n_gain_events", "gain_rate_per_opportunity"]]
    merged = eo.merge(fs, on="feature", how="left")
    stat = _linreg(merged["prevalence_dest"].values.astype(float), merged["expected_order"].values.astype(float))
    return merged, stat


def plot_order_vs_prevalence(merged: pd.DataFrame, stat: Dict[str, float], title: str,
                             annotate_top: int = 12, save: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(merged["prevalence_dest"], merged["expected_order"],
               s=np.clip(merged["n_gain_events"].fillna(0) * 1.6, 12, 200),
               c="#4C72B0", alpha=0.7, edgecolors="white")
    if np.isfinite(stat.get("slope", np.nan)):
        xs = np.linspace(merged["prevalence_dest"].min(), merged["prevalence_dest"].max(), 50)
        ax.plot(xs, stat["slope"] * xs + stat["intercept"], "--", color="#C44E52",
                label=f"R²={stat['r2']:.2f}, p={stat['p_value']:.1e}")
        ax.legend(fontsize=8)
    sub = merged.nlargest(annotate_top, "n_gain_events")
    for _, r in sub.iterrows():
        ax.annotate(str(r["feature"])[:18], (r["prevalence_dest"], r["expected_order"]), fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("prevalence in descendant states")
    ax.set_ylabel("expected acquisition ordering (earlier = smaller)")
    ax.set_title(title)
    if save:
        fig.savefig(save)
    return fig


# --------------------------------------------------------------------------------------
# 4. 分歧层：跨谱系一致 vs 分歧
# --------------------------------------------------------------------------------------

def lineage_ordering_matrix(ts: TransitionSet, min_events: int = 15, max_groups: int = 12
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对事件数足够的谱系分别计算期望获得次序，得到 (feature × lineage) 矩阵，
    并按跨谱系标准差区分"全局一致"与"全局分歧"特征（对齐 Aga2026 Fig 3）。
    """
    if "group_id" not in ts.frame.columns:
        return pd.DataFrame(), pd.DataFrame()
    counts = ts.frame.groupby("group_id")["n_gained_parsed"].sum().sort_values(ascending=False)
    groups = [g for g, c in counts.items() if c >= min_events][:max_groups]
    cols: Dict[str, pd.Series] = {}
    for g in groups:
        idx = np.flatnonzero((ts.frame["group_id"] == g).values)
        sub = TransitionSet(
            frame=ts.frame.iloc[idx].reset_index(drop=True),
            src=ts.src[idx], dest=ts.dest[idx],
            gained=[ts.gained[i] for i in idx],
            feature_names=ts.feature_names,
        )
        eo = expected_ordering(empirical_bubble_matrix(sub)).set_index("feature")["expected_order"]
        cols[str(g)] = eo
    if not cols:
        return pd.DataFrame(), pd.DataFrame()
    mat = pd.DataFrame(cols)
    summary = pd.DataFrame({
        "feature": mat.index,
        "mean_order": mat.mean(axis=1).values,
        "sd_across_lineages": mat.std(axis=1).values,
        "n_lineages_observed": mat.notna().sum(axis=1).values,
    })
    summary["consistency"] = np.where(
        summary["sd_across_lineages"] <= summary["sd_across_lineages"].median(),
        "globally consistent", "globally divergent")
    summary = summary.sort_values("sd_across_lineages", ascending=False).reset_index(drop=True)
    return mat, summary


def plot_lineage_ordering(mat: pd.DataFrame, summary: pd.DataFrame, title: str,
                          top_features: int = 25, save: Optional[str] = None):
    if mat.empty:
        return None
    keep = summary.dropna(subset=["sd_across_lineages"]).head(top_features)["feature"].tolist()
    keep = [f for f in summary.sort_values("mean_order")["feature"] if f in set(keep)]
    sub = mat.loc[keep]
    fig, axes = plt.subplots(1, 2, figsize=(12, max(3.4, 0.3 * len(sub) + 1.2)),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    im = axes[0].imshow(sub.values, aspect="auto", cmap="coolwarm")
    axes[0].set_yticks(range(len(sub)))
    axes[0].set_yticklabels(sub.index, fontsize=7)
    axes[0].set_xticks(range(sub.shape[1]))
    axes[0].set_xticklabels(sub.columns, rotation=90, fontsize=7)
    axes[0].set_title(f"{title}\nexpected ordering per lineage")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], label="expected ordering")

    s = summary.set_index("feature").loc[sub.index]
    colors = ["#C44E52" if c == "globally divergent" else "#4C72B0" for c in s["consistency"]]
    axes[1].barh(range(len(s)), s["sd_across_lineages"].values, color=colors)
    axes[1].set_yticks(range(len(s)))
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("sd of ordering across lineages")
    axes[1].set_title("consistent (blue) vs divergent (red)")
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


# --------------------------------------------------------------------------------------
# 5. 预测层：评估结果
# --------------------------------------------------------------------------------------

METRIC_COLS = ["mean_logl", "top1", "top3", "top5", "mrr", "auc"]


def load_eval_summary(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def overall_table(summary: dict) -> pd.DataFrame:
    rows = []
    for method, vals in (summary.get("overall") or {}).items():
        if not isinstance(vals, dict):
            continue
        rows.append({"method": method, **{m: vals.get(m) for m in METRIC_COLS}})
    return pd.DataFrame(rows)


def comparison_table(summary: dict) -> pd.DataFrame:
    rows = []
    for key, val in summary.items():
        if key.endswith(("_vs_prevalence", "_vs_random", "_vs_pretrained", "_vs_frozen", "_vs_hypertraps")) \
                and isinstance(val, dict) and val:
            rows.append({
                "comparison": key,
                "left": val.get("left_method"),
                "right": val.get("right_method"),
                "delta_mean_logl": val.get("delta_mean_logl"),
                "p_value": val.get("p_value"),
                "verdict": val.get("verdict"),
            })
    return pd.DataFrame(rows)


def fold_table(fold_metrics_path: str) -> pd.DataFrame:
    with open(fold_metrics_path, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    rows = []
    for fold in blob.get("folds", []):
        methods = fold.get("methods") or {}
        if not isinstance(methods, dict):
            continue
        for method, vals in methods.items():
            rows.append({"fold": fold.get("fold"), "n_train": fold.get("n_train"),
                         "n_test": fold.get("n_test"), "method": method,
                         **{m: vals.get(m) for m in METRIC_COLS}})
    return pd.DataFrame(rows)


def plot_method_comparison(overall: pd.DataFrame, title: str, save: Optional[str] = None):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    order = overall.sort_values("mean_logl")["method"].tolist()
    o = overall.set_index("method").loc[order]
    palette = {"B0_uniform": "#BBBBBB", "B1_prevalence": "#8C8C8C"}
    colors = [palette.get(m, "#C44E52" if m.startswith("HT") else "#4C72B0") for m in order]
    axes[0].barh(range(len(o)), o["mean_logl"], color=colors)
    axes[0].set_yticks(range(len(o)))
    axes[0].set_yticklabels(order, fontsize=8)
    axes[0].set_xlabel("held-out mean log-likelihood (higher better)")
    axes[0].set_title("Predictive likelihood")

    idx = np.arange(len(o))
    w = 0.26
    for k, (m, c) in enumerate(zip(["top1", "top3", "top5"], ["#4C72B0", "#DD8452", "#55A868"])):
        axes[1].bar(idx + (k - 1) * w, o[m], width=w, label=m, color=c)
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(order, rotation=30, ha="right", fontsize=7)
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Top-k hit rate")
    axes[1].legend(fontsize=7)

    axes[2].bar(idx, o["auc"], color=colors)
    axes[2].axhline(0.5, ls="--", color="grey", lw=1)
    axes[2].set_xticks(idx)
    axes[2].set_xticklabels(order, rotation=30, ha="right", fontsize=7)
    axes[2].set_ylabel("AUC")
    axes[2].set_title("Within-candidate AUC")
    fig.suptitle(title, y=1.04)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


def plot_fold_distribution(folds: pd.DataFrame, title: str, save: Optional[str] = None):
    if folds.empty:
        return None
    order = folds.groupby("method")["mean_logl"].mean().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    data = [folds.loc[folds["method"] == m, "mean_logl"].dropna().values for m in order]
    bp = axes[0].boxplot(data, vert=True, patch_artist=True, widths=0.6)
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor("#C44E52" if m.startswith("HT") else ("#BBBBBB" if m.startswith("B") else "#4C72B0"))
        patch.set_alpha(0.8)
    for i, vals in enumerate(data, start=1):
        axes[0].scatter(np.full(len(vals), i) + np.random.uniform(-0.08, 0.08, len(vals)), vals,
                        s=14, color="black", alpha=0.6, zorder=3)
    axes[0].set_xticklabels(order, rotation=30, ha="right", fontsize=7)
    axes[0].set_ylabel("mean logL per fold")
    axes[0].set_title("Fold-level variability")

    for m in order:
        sub = folds[folds["method"] == m].sort_values("fold")
        axes[1].plot(sub["fold"], sub["top1"], marker="o", label=m, lw=1.4)
    axes[1].set_xlabel("fold")
    axes[1].set_ylabel("Top-1")
    axes[1].set_title("Top-1 across folds")
    axes[1].legend(fontsize=6.5, ncol=2)
    fig.suptitle(title, y=1.05)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


# --------------------------------------------------------------------------------------
# 6. 路径层：由两阶段模型推断演化路径
# --------------------------------------------------------------------------------------

def load_two_stage_model(model_path: str, device: str = "cpu"):
    """加载 default/stage2/model.pt（需要 torch）。"""
    if MODELS_DIR not in sys.path:
        sys.path.insert(0, MODELS_DIR)
    from two_stage_backbone import TwoStageTransitionModel  # noqa: E402
    return TwoStageTransitionModel.load(model_path, device=device)


def model_bubble_matrix(model, feature_names: Sequence[str], n_walks: int = 2000,
                        max_steps: Optional[int] = None, seed: int = 0) -> pd.DataFrame:
    """
    从空状态出发按模型给出的条件概率做前向随机游走（hypercubic transition path sampling 的
    神经网络对应物），统计"特征 i 在第 step 位被获得"的后验概率。
    """
    L = len(feature_names)
    steps = L if max_steps is None else min(L, max_steps)
    rng = np.random.default_rng(seed)
    mat = np.zeros((L, steps), dtype=float)
    for _ in range(n_walks):
        state = np.zeros(L, dtype=np.int8)
        for step in range(1, steps + 1):
            scores = np.asarray(model.predict_scores(state), dtype=float)
            scores[state == 1] = 0.0
            total = scores.sum()
            if total <= 0:
                break
            pick = int(rng.choice(L, p=scores / total))
            mat[pick, step - 1] += 1.0
            state[pick] = 1
    mat /= max(1, n_walks)
    return pd.DataFrame(mat, index=list(feature_names), columns=[f"step{i+1}" for i in range(steps)])


def model_first_step_ranking(model, feature_names: Sequence[str], top_n: int = 15) -> pd.DataFrame:
    L = len(feature_names)
    scores = np.asarray(model.predict_scores(np.zeros(L, dtype=np.int8)), dtype=float)
    out = pd.DataFrame({"feature": list(feature_names), "p_first_step": scores})
    return out.sort_values("p_first_step", ascending=False).head(top_n).reset_index(drop=True)


def model_interaction_matrix(model, feature_names: Sequence[str]) -> pd.DataFrame:
    """
    特征交互（促进/抑制）矩阵：
      对每个已获得特征 j，比较"仅含 j 的状态"与"空状态"下，其余特征 i 的条件概率对数比
          M[i, j] = log2( p(i | s = {j}) / p(i | s = ∅) )
      两侧概率均在同一候选集合（排除 j）上重新归一化，故 M>0 表示 j 促进 i 的后续获得，
      M<0 表示 j 抑制 i。与 HyperTraPS 的 interaction graph 语义一致。
    """
    L = len(feature_names)
    base = np.asarray(model.predict_scores(np.zeros(L, dtype=np.int8)), dtype=float)
    M = np.full((L, L), np.nan, dtype=float)
    eps = 1e-12
    for j in range(L):
        state = np.zeros(L, dtype=np.int8)
        state[j] = 1
        cond = np.asarray(model.predict_scores(state), dtype=float)
        mask = np.ones(L, dtype=bool)
        mask[j] = False
        b = base.copy()
        b[~mask] = 0.0
        c = cond.copy()
        c[~mask] = 0.0
        if b.sum() <= 0 or c.sum() <= 0:
            continue
        b = b / b.sum()
        c = c / c.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.log2((c + eps) / (b + eps))
        ratio[~mask] = np.nan
        M[:, j] = ratio
    return pd.DataFrame(M, index=list(feature_names), columns=list(feature_names))


def plot_interaction_matrix(M: pd.DataFrame, title: str, top_features: Optional[int] = 25,
                            save: Optional[str] = None):
    sub = M
    if top_features is not None and len(M) > top_features:
        strength = M.abs().mean(axis=1).sort_values(ascending=False).head(top_features).index
        sub = M.loc[strength, strength]
    v = np.nanmax(np.abs(sub.values)) if np.isfinite(sub.values).any() else 1.0
    fig, ax = plt.subplots(figsize=(max(5.0, 0.34 * len(sub) + 2.4), max(4.2, 0.32 * len(sub) + 1.6)))
    im = ax.imshow(sub.values, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub.columns, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub.index, fontsize=6.5)
    ax.set_xlabel("already acquired feature j (context)")
    ax.set_ylabel("next-step feature i")
    ax.set_title(f"{title}\nlog2 ratio p(i|{{j}}) / p(i|∅):  red = promotes, blue = represses")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="log2 odds shift")
    if save:
        fig.savefig(save)
    return fig


def top_interactions(M: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    recs = []
    for j in M.columns:
        for i in M.index:
            v = M.loc[i, j]
            if i != j and np.isfinite(v):
                recs.append({"context_j": j, "next_i": i, "log2_shift": float(v)})
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    top_pos = df.nlargest(top_n, "log2_shift").assign(effect="promotes")
    top_neg = df.nsmallest(top_n, "log2_shift").assign(effect="represses")
    return pd.concat([top_pos, top_neg], ignore_index=True)


# --------------------------------------------------------------------------------------
# 7. 跨数据集/跨物种汇总
# --------------------------------------------------------------------------------------

def collect_overall(specs: Sequence[DatasetSpec], experiment_filter: Optional[str] = None) -> pd.DataFrame:
    """把多个数据集/物种的 overall 指标收集为长表。"""
    rows = []
    for spec in specs:
        for exp, path in spec.eval_summaries().items():
            if experiment_filter and experiment_filter not in exp:
                continue
            summary = load_eval_summary(path)
            tbl = overall_table(summary)
            if tbl.empty:
                continue
            tbl["dataset"] = spec.key
            tbl["label"] = spec.label_en
            tbl["experiment"] = exp
            tbl["L"] = summary.get("L")
            tbl["n_transitions"] = summary.get("n_transitions")
            tbl["n_groups"] = summary.get("n_groups")
            rows.append(tbl)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_comparisons(specs: Sequence[DatasetSpec], experiment_filter: Optional[str] = None) -> pd.DataFrame:
    rows = []
    for spec in specs:
        for exp, path in spec.eval_summaries().items():
            if experiment_filter and experiment_filter not in exp:
                continue
            summary = load_eval_summary(path)
            tbl = comparison_table(summary)
            if tbl.empty:
                continue
            tbl["dataset"] = spec.key
            tbl["experiment"] = exp
            tbl["L"] = summary.get("L")
            tbl["n_transitions"] = summary.get("n_transitions")
            rows.append(tbl)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
