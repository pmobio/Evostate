#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_figure_kp.py —— 肺炎克雷伯菌（Kleborate 22 character）主图的计算与绘图

设计口径对齐 Aga et al. 2026 (PLoS Biol) *Evolutionary inference reveals global natural
histories and predicted pathways of antimicrobial resistance in Klebsiella pneumoniae*：

| 论文图 | 本图 panel | 说明 |
|---|---|---|
| Fig 1B  | A | 最常见 AMR character profile 的 UpSet 式计数（颜色 = 携带 character 数） |
| Fig 2A  | B | 聚合后的获得动力学，**高亮双峰 character**（双峰 = 存在不同演化通路） |
| Fig 2Bi | C | 分组特异 bubble 矩阵的 PCA（论文按国家，本项目按 **ST 谱系**） |
| Fig 3   | D | "全局一致" vs "全局分歧" character（跨谱系序位跨度 + 与 PC2 的协变） |
| Fig 5B  | E | 真实下一步 character 的预测排名（rank ≤ k 曲线）+ **可扩展性**：HyperTraPS 全折超时 |

与 TB 主图的根本差别：**KP 上 HyperTraPS 无法运行**（L=22、9,026 条转移，
每折 1800 s 预算内 5/5 折 TimeoutExpired），因此这里没有双模型对照，
"两阶段模型可在此规模上训练、HyperTraPS 不能"本身就是结果之一。

绘图文字全部英文，叙述性中文留在 notebook 的 markdown。
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

import evoamr_story as st
import hypertraps_compare as hc
import paper_figure_tb as pf

# --------------------------------------------------------------------------------------
# 1. Kleborate 22 character 的注释（本数据集没有 feature_hierarchy.csv）
# --------------------------------------------------------------------------------------

# 论文 Fig 1 caption 的命名口径：-a = acquired gene，-m = mutation
KP_SHORT = {
    "AGly_acquired": "AGly-a", "Col_acquired": "Col-a", "Fcyn_acquired": "Fcyn-a",
    "Flq_acquired": "Flq-a", "Gly_acquired": "Gly-a", "MLS_acquired": "MLS-a",
    "Phe_acquired": "Phe-a", "Rif_acquired": "Rif-a", "Sul_acquired": "Sul-a",
    "Tet_acquired": "Tet-a", "Tgc_acquired": "Tgc-a", "Tmt_acquired": "Tmt-a",
    "Bla_acquired": "Bla-a", "Bla_inhR_acquired": "Bla_inhR-a",
    "Bla_ESBL_acquired": "Bla_ESBL-a", "Bla_ESBL_inhR_acquired": "Bla_ESBL_inhR-a",
    "Bla_Carb_acquired": "Bla_Carb-a", "Bla_chr": "Bla_chr",
    "SHV_mutations": "SHV-m", "Omp_mutations": "Omp-m",
    "Col_mutations": "Col-m", "Flq_mutations": "Flq-m",
}
KP_DRUG = {
    "AGly_acquired": "aminoglycoside", "Col_acquired": "colistin",
    "Fcyn_acquired": "fosfomycin", "Flq_acquired": "fluoroquinolone",
    "Gly_acquired": "glycopeptide", "MLS_acquired": "macrolide",
    "Phe_acquired": "phenicol", "Rif_acquired": "rifampin",
    "Sul_acquired": "sulfonamide", "Tet_acquired": "tetracycline",
    "Tgc_acquired": "tigecycline", "Tmt_acquired": "trimethoprim",
    "Bla_acquired": "beta-lactam", "Bla_inhR_acquired": "beta-lactam+inhibitor",
    "Bla_ESBL_acquired": "extended-spectrum beta-lactam",
    "Bla_ESBL_inhR_acquired": "ESBL+inhibitor", "Bla_Carb_acquired": "carbapenem",
    "Bla_chr": "beta-lactam (intrinsic)", "SHV_mutations": "beta-lactam (SHV)",
    "Omp_mutations": "carbapenem (porin loss)", "Col_mutations": "colistin",
    "Flq_mutations": "fluoroquinolone",
}
# 三类机制层：横向获得的 ARG / β-内酰胺家族（含内源）/ 染色体突变
KP_GROUPS = {
    "acquired ARG": ["AGly_acquired", "Col_acquired", "Fcyn_acquired", "Flq_acquired",
                     "Gly_acquired", "MLS_acquired", "Phe_acquired", "Rif_acquired",
                     "Sul_acquired", "Tet_acquired", "Tgc_acquired", "Tmt_acquired"],
    "beta-lactam": ["Bla_acquired", "Bla_inhR_acquired", "Bla_ESBL_acquired",
                    "Bla_ESBL_inhR_acquired", "Bla_Carb_acquired", "Bla_chr"],
    "mutation": ["SHV_mutations", "Omp_mutations", "Col_mutations", "Flq_mutations"],
}
KP_GROUP_COLORS = {
    "acquired ARG": "#1f6fb4",
    "beta-lactam": "#b7472a",
    "mutation": "#2e7d32",
    "other": "#999999",
}


def register_kp_colors() -> None:
    """把 KP 的三类机制颜色注入 `paper_figure_tb.CLASS_COLORS`，以复用其绘图函数。"""
    pf.CLASS_COLORS.update(KP_GROUP_COLORS)


def kp_annotation(feature_names: Sequence[str]) -> pd.DataFrame:
    """22 个 character 的展示注释（短名 / 药物 / 机制层）。"""
    inv = {f: g for g, fs in KP_GROUPS.items() for f in fs}
    rows = []
    for f in feature_names:
        rows.append({"feature": f, "short": KP_SHORT.get(f, f),
                     "drug": KP_DRUG.get(f, ""), "group": inv.get(f, "other")})
    return pd.DataFrame(rows)


def kp_class_map(feature_names: Sequence[str]) -> Dict[str, str]:
    inv = {f: g for g, fs in KP_GROUPS.items() for f in fs}
    return {f: inv.get(f, "other") for f in feature_names}


def kp_short_map(feature_names: Sequence[str]) -> Dict[str, str]:
    return {f: KP_SHORT.get(f, f) for f in feature_names}


def rename_index(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """把 bubble / 矩阵的行名换成论文口径的短名（不改动原对象）。"""
    return df.rename(index=mapping)


# --------------------------------------------------------------------------------------
# 2. 双峰性：论文用它标记"存在不同演化通路"的 character（Fig 2A）
# --------------------------------------------------------------------------------------

def bimodality_table(bubble: pd.DataFrame, min_peak_mass: float = 0.06,
                     min_valley_ratio: float = 0.55, smooth: int = 1,
                     observed: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    对每个 character 的获得序位分布做双峰性检测。

    两个互补判据：
      * `bimodality_coef` —— Sarle 双峰系数 (skew^2 + 1) / kurtosis，> 5/9 提示非单峰；
      * `n_peaks` / `valley_ratio` —— 在（可选平滑后的）分布上找局部极大，
        要求峰质量 ≥ `min_peak_mass`，且两峰之间的谷 ≤ `min_valley_ratio` × 次高峰，
        避免把锯齿噪声当成第二个峰。
    `is_bimodal` 取"检出 ≥2 个有效峰"，与论文"bimodal acquisition distributions"一致。

    `observed`（`hc.observed_step_range` 的输出）给出时，**先把每个 character 的分布
    截断到它自己被观测到的序位窗口并重新归一化**，再做检测。这一步对模型侧是必需的：
    前向游走会一直走到 L 步，落在"从未观测"区间的峰在数据中无法验证
    （KP 上模型给 SHV-m 的第二个峰在 step 17，而 SHV-m 最后一次被观测到是 step 14）。
    """
    obs = observed.set_index("feature") if observed is not None else None
    steps_all = np.arange(1, bubble.shape[1] + 1, dtype=float)
    rows = []
    for f in bubble.index:
        p_raw = bubble.loc[f].values.astype(float)
        lo, hi = 1, bubble.shape[1]
        if obs is not None and f in obs.index:
            v_lo = obs.loc[f, "first_observed_step"]
            v_hi = obs.loc[f, "last_observed_step"]
            if np.isfinite(v_lo):
                lo = int(v_lo)
            if np.isfinite(v_hi):
                hi = int(v_hi)
        p = np.zeros_like(p_raw)
        p[lo - 1:hi] = p_raw[lo - 1:hi]
        steps = steps_all
        mass = p.sum()
        base = {"feature": f, "window_lo": lo, "window_hi": hi,
                "mass_inside_window": float(mass / max(p_raw.sum(), 1e-12))}
        if mass <= 0:
            rows.append({**base, "bimodality_coef": np.nan, "n_peaks": 0,
                         "valley_ratio": np.nan, "is_bimodal": False,
                         "peak_steps": "", "total_mass": 0.0})
            continue
        q = p / mass
        mu = float((q * steps).sum())
        var = float((q * (steps - mu) ** 2).sum())
        sd = math.sqrt(max(var, 1e-12))
        skew = float((q * ((steps - mu) / sd) ** 3).sum())
        kurt = float((q * ((steps - mu) / sd) ** 4).sum())
        bc = (skew ** 2 + 1.0) / kurt if kurt > 0 else np.nan

        s = q.copy()
        for _ in range(max(0, int(smooth))):
            s = np.convolve(s, np.array([0.25, 0.5, 0.25]), mode="same")
        peaks = [i for i in range(lo - 1, hi)
                 if s[i] >= s[max(lo - 1, i - 1)] and s[i] >= s[min(hi - 1, i + 1)]
                 and q[i] >= min_peak_mass]
        # 合并相邻峰
        merged: List[int] = []
        for i in peaks:
            if merged and i - merged[-1] <= 1:
                if s[i] > s[merged[-1]]:
                    merged[-1] = i
            else:
                merged.append(i)
        valley_ratio, ok_peaks = np.nan, merged
        if len(merged) >= 2:
            top2 = sorted(merged, key=lambda i: s[i], reverse=True)[:2]
            a, b = sorted(top2)
            valley = float(s[a:b + 1].min())
            second = float(s[top2[1]])
            valley_ratio = valley / max(second, 1e-12)
            ok_peaks = top2 if valley_ratio <= min_valley_ratio else [top2[0]]
        rows.append({
            **base, "bimodality_coef": bc, "n_peaks": len(ok_peaks),
            "valley_ratio": valley_ratio, "is_bimodal": len(ok_peaks) >= 2,
            "peak_steps": ";".join(str(i + 1) for i in sorted(ok_peaks)),
            "total_mass": float(mass),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 3. 分组（谱系）特异动力学与其 PCA —— 论文 Fig 2B / Fig 3
# --------------------------------------------------------------------------------------

def lineage_bubbles(ts: st.TransitionSet, min_events: int = 50, max_groups: int = 60
                    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    每个谱系（ST）各自的**经验** bubble 矩阵。

    与论文的差别必须说明：论文对每个国家**分别训练** HyperTraPS，得到推断后的 bubble；
    本项目在 ST 层面做同一件事的成本过高（980 个 ST），因此这里用**经验** bubble
    作为分组特异动力学的观测量。它同样是 feature × step 的概率矩阵，PCA 的语义一致，
    但 PC1 更多反映"该谱系被观测到的深度/锐度"，而非推断精度。
    """
    counts = ts.frame.groupby("group_id")["n_gained_parsed"].sum().sort_values(ascending=False)
    groups = [g for g, c in counts.items() if c >= min_events][:max_groups]
    out: Dict[str, pd.DataFrame] = {}
    meta = []
    for g in groups:
        idx = np.flatnonzero((ts.frame["group_id"] == g).values)
        sub = pf.subset_transitions(ts, idx)
        out[str(g)] = st.empirical_bubble_matrix(sub)
        meta.append({"group_id": str(g), "n_transitions": int(len(idx)),
                     "n_gain_events": int(ts.frame["n_gained_parsed"].iloc[idx].sum()),
                     "mean_depth": float(ts.frame["n_src_features"].iloc[idx].mean())})
    return out, pd.DataFrame(meta)


def bubble_pca(bubbles: Dict[str, pd.DataFrame], n_pc: int = 3
               ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    对"分组 × (feature×step 展平)"矩阵做 PCA（中心化 + SVD，不依赖 sklearn）。

    返回 (scores, loadings, explained_variance_ratio)：
      scores   : index = group_id, columns = PC1..PCn
      loadings : index = "feature|step", columns = PC1..PCn
    """
    names = list(bubbles)
    ref = bubbles[names[0]]
    cols = [f"{f}|{c}" for f in ref.index for c in ref.columns]
    X = np.vstack([bubbles[g].loc[ref.index, ref.columns].values.ravel() for g in names])
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_pc, len(S))
    evr = (S ** 2) / max(float((S ** 2).sum()), 1e-12)
    scores = pd.DataFrame(U[:, :k] * S[:k], index=names,
                          columns=[f"PC{i+1}" for i in range(k)])
    loadings = pd.DataFrame(Vt[:k].T, index=cols,
                            columns=[f"PC{i+1}" for i in range(k)])
    return scores, loadings, evr[:k]


def lineage_expected_order(bubbles: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """feature × group 的期望获得序位矩阵。"""
    cols = {}
    for g, b in bubbles.items():
        cols[g] = st.expected_ordering(b).set_index("feature")["expected_order"]
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------------------
# 3b. 单一代表性谱系：在该谱系的转移子集上单独拟合两阶段模型
#     （论文对每个国家分别拟合 HyperTraPS，这里是同一思路的对应物）
# --------------------------------------------------------------------------------------

def write_lineage_transitions(spec: "st.DatasetSpec", group_id: str, outdir: str) -> Tuple[str, int]:
    """把某个谱系的转移行原样抽出为独立 CSV（保留原始列，供训练脚本直接读取）。"""
    os.makedirs(outdir, exist_ok=True)
    raw = pd.read_csv(spec.transitions_path, dtype={"src_state": str, "dest_state": str})
    sub = raw[raw["group_id"].astype(str) == str(group_id)].reset_index(drop=True)
    path = os.path.join(outdir, "transitions.csv")
    sub.to_csv(path, index=False)
    return path, int(len(sub))


def ensure_lineage_two_stage_model(spec: "st.DatasetSpec", group_id: str,
                                   python_exe: Optional[str] = None,
                                   epochs: int = 60, d_model: int = 128, n_layers: int = 3,
                                   n_heads: int = 4, batch_size: int = 32, dropout: float = 0.30,
                                   contrastive_weight: float = 0.05, mask_prob_pos: float = 0.5,
                                   mask_prob_neg: float = 0.1, seed: int = 0,
                                   force: bool = False) -> Dict[str, object]:
    """
    只用某个谱系（ST）的转移训练一次 Stage1+Stage2，产出**谱系特异**模型。

    输出目录：`<results_dir>/lineage_<group_id>/{transitions.csv,stage1,stage2}`。
    与全局模型的唯一差别是训练数据的范围 —— 因此二者的对照可以直接回答
    "把谱系条件引入模型后，该谱系的获得动力学是否被复现"。
    """
    base = os.path.join(spec.results_dir, "lineage_%s" % group_id)
    tr_path, n_tr = write_lineage_transitions(spec, group_id, base)
    out = train_two_stage_on(
        tr_path, base, python_exe=python_exe, epochs=epochs, d_model=d_model,
        n_layers=n_layers, n_heads=n_heads, batch_size=batch_size, dropout=dropout,
        contrastive_weight=contrastive_weight, mask_prob_pos=mask_prob_pos,
        mask_prob_neg=mask_prob_neg, seed=seed, force=force)
    out["group_id"] = str(group_id)
    out["n_transitions"] = n_tr
    return out


def lineage_dynamics_comparison(bubbles: Dict[str, pd.DataFrame],
                                keys: Sequence[str],
                                short_map: Dict[str, str]) -> pd.DataFrame:
    """
    某个谱系下几路 bubble 的期望序位对照表 + 与该谱系经验序位的 Spearman/MAE。

    `keys[0]` 必须是该谱系的**经验** bubble（作为参照）。
    """
    eo = {k: st.expected_ordering(bubbles[k]).set_index("feature")["expected_order"]
          for k in keys}
    df = pd.DataFrame(eo)
    df.insert(0, "short", [short_map.get(f, f) for f in df.index])
    ref = keys[0]
    stats = []
    for k in keys[1:]:
        a = df[ref].values.astype(float)
        b = df[k].values.astype(float)
        stats.append({"source": k,
                      "spearman_vs_lineage_empirical": hc.spearman(a, b),
                      "mae_vs_lineage_empirical": float(np.nanmean(np.abs(b - a)))})
    return df.sort_values(ref), pd.DataFrame(stats)


def _legend_handles(legend):
    """兼容不同 matplotlib 版本的图例句柄属性名。"""
    return getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", [])


def plot_lineage_dynamics(
        axes,
        bubbles: Dict[str, pd.DataFrame],
        keys: Sequence[str],
        feature_order: Sequence[str],
        class_map: Dict[str, str],
        short_map: Dict[str, str],
        titles: Sequence[str],
        empirical_key: str,
        observed: Optional[pd.DataFrame] = None,
        highlight: Optional[Sequence[str]] = None,
        show_empirical_order: bool = True,
        show_never_observed: bool = True,
):
    """
    某个谱系的多列获得动力学对照。

    兼容旧版 pf.panel_b_dynamics()：
    底层函数不接收 show_empirical_order /
    show_never_observed，因此这里先正常绘图，
    再根据开关删除相应元素。
    """

    # ------------------------------------------------------------
    # 1. 使用原始 panel_b_dynamics 绘图
    # ------------------------------------------------------------
    pf.panel_b_dynamics(
        list(axes),
        None,
        bubbles,
        feature_order,
        class_map,
        {},
        model_keys=list(keys),
        empirical_key=empirical_key,
        observed=observed,
        highlight=highlight,
        header="",
    )

    # ------------------------------------------------------------
    # 2. 根据开关删除 expected-order 红色虚线
    # ------------------------------------------------------------
    if not show_empirical_order:

        for ax in np.atleast_1d(axes):

            for line in list(ax.lines):

                linestyle = line.get_linestyle()

                if linestyle in ("--", "dashed"):
                    line.remove()

            # panel_b_dynamics() 的 expected-order 红色竖线是用 ax.vlines() 画的，
            # 它落在 ax.collections 里（LineCollection），不在 ax.lines 里；
            # 只清理 ax.lines 会漏掉这批红线，因此这里按"虚线段集合"再清一次。
            for coll in list(ax.collections):

                if not isinstance(coll, LineCollection):
                    continue

                dashes = coll.get_linestyle()

                is_dashed = bool(dashes) and any(
                    seq is not None and len(seq) > 0 for _offset, seq in dashes
                )

                if is_dashed:
                    coll.remove()

    # ------------------------------------------------------------
    # 3. 根据开关删除 never-observed 灰色背景
    # ------------------------------------------------------------
    if not show_never_observed:

        for ax in np.atleast_1d(axes):

            for patch in list(ax.patches):

                try:
                    fc = patch.get_facecolor()
                except Exception:
                    continue

                if fc is None or len(fc) < 3:
                    continue

                r, g, b = fc[:3]

                # 非常浅、接近中性的灰色 patch
                # 作为 never-observed 背景识别
                is_light_grey = (
                    r > 0.85
                    and g > 0.85
                    and b > 0.85
                    and abs(r - g) < 0.05
                    and abs(g - b) < 0.05
                )

                if is_light_grey:
                    patch.remove()

    # ------------------------------------------------------------
    # 3b. 图元被删掉后，同步清掉对应的图例条目
    # ------------------------------------------------------------
    dropped_labels = set()
    if not show_empirical_order:
        dropped_labels.add("empirical expected order")
    if not show_never_observed:
        dropped_labels.add("never observed")

    if dropped_labels:
        for ax in np.atleast_1d(axes):
            legend = ax.get_legend()
            if legend is None:
                continue
            kept = [(h, t.get_text()) for h, t in zip(_legend_handles(legend), legend.texts)
                    if t.get_text() not in dropped_labels]
            if not kept:
                legend.remove()
                continue
            ax.legend([h for h, _ in kept], [t for _, t in kept],
                      fontsize=6.6, ncol=max(1, len(kept)), loc="upper left",
                      bbox_to_anchor=(0.0, -0.095), frameon=False, handlelength=1.4,
                      columnspacing=1.1, borderpad=0.2)

    # ------------------------------------------------------------
    # 4. 第一列 Y-axis feature labels
    # ------------------------------------------------------------
    visible_features = [
        f for f in feature_order
        if f in bubbles[keys[0]].index
    ]

    axes[0].set_yticklabels(
        [
            short_map.get(f, f)
            for f in visible_features
        ],
        fontsize=7.2
    )

    for tick, f in zip(
        axes[0].get_yticklabels(),
        visible_features
    ):
        tick.set_color(
            pf.CLASS_COLORS.get(
                class_map.get(f, "other"),
                "#333333"
            )
        )

    # ------------------------------------------------------------
    # 5. Panel titles
    # ------------------------------------------------------------
    for ax, t in zip(
        axes,
        titles
    ):
        ax.set_title(
            t,
            fontsize=9
        )

def panel_lineage_distribution_only(
        ax,
        bubble: pd.DataFrame,
        feature_order: Sequence[str],
        short_map: Dict[str, str],
        class_map: Dict[str, str],
        show_ylabels: bool = True,
):
    """
    Plot lineage-specific acquisition-step distributions only.

    - no empirical expected-order line
    - no never-observed shading
    - no reference distribution
    - each row independently scaled to its own maximum
    - bar colour = resistance mechanism group
    """

    register_kp_colors()

    feats = [
        f for f in feature_order
        if f in bubble.index
    ]

    n_feat = len(feats)
    n_step = bubble.shape[1]

    for row_i, f in enumerate(feats):

        y0 = n_feat - 1 - row_i

        p = bubble.loc[f].values.astype(float)

        p = np.nan_to_num(
            p,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        pmax = np.max(p)

        if pmax > 0:
            q = p / pmax
        else:
            q = np.zeros_like(p)

        x = np.arange(
            1,
            n_step + 1
        )

        group = class_map.get(
            f,
            "other"
        )

        color = KP_GROUP_COLORS.get(
            group,
            KP_GROUP_COLORS["other"]
        )

        ax.bar(
            x,
            q * 0.72,
            width=0.78,
            bottom=y0 - 0.36,
            color=color,
            edgecolor="none",
            alpha=0.95,
            zorder=2
        )

        ax.hlines(
            y0 - 0.36,
            0.5,
            n_step + 0.5,
            color="#8c8c8c",
            linewidth=0.45,
            zorder=1
        )

    # Y axis
    y_positions = np.arange(
        n_feat
    )[::-1]

    ax.set_yticks(
        y_positions
    )

    if show_ylabels:

        ax.set_yticklabels(
            [
                short_map.get(f, f)
                for f in feats
            ],
            fontsize=6.4
        )

        for tick, f in zip(
            ax.get_yticklabels(),
            feats
        ):

            group = class_map.get(
                f,
                "other"
            )

            tick.set_color(
                KP_GROUP_COLORS.get(
                    group,
                    "#333333"
                )
            )

        ax.tick_params(
            axis="y",
            labelleft=True,
            length=2,
            width=0.6,
            pad=2
        )

    else:

        ax.set_yticklabels([])

        ax.tick_params(
            axis="y",
            labelleft=False,
            length=0
        )

    # X axis
    ax.set_xlim(
        0.5,
        n_step + 0.5
    )

    ax.set_ylim(
        -0.6,
        n_feat - 0.4
    )

    xticks = [
        1,
        4,
        7,
        10,
        13,
        16,
        19,
        22
    ]

    xticks = [
        v for v in xticks
        if v <= n_step
    ]

    ax.set_xticks(
        xticks
    )

    ax.set_xticklabels(
        [str(v) for v in xticks],
        fontsize=6.5
    )

    ax.set_xlabel(
        "acquisition step",
        fontsize=7
    )

    ax.tick_params(
        axis="x",
        labelbottom=True,
        length=3,
        width=0.6
    )

    # Appearance
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)

    ax.grid(False)

def panel_lineage_empirical_only(
        ax,
        bubble: pd.DataFrame,
        feature_order: Sequence[str],
        short_map: Dict[str, str],
        class_map: Dict[str, str],
        show_ylabels: bool = True,
):
    """
    Lineage-specific empirical acquisition dynamics.

    只展示 lineage-specific empirical acquisition-step distribution。

    不展示：
      - empirical expected-order 红色虚线
      - never-observed 灰色区域
      - empirical reference outline

    每一个 resistance character 的分布独立归一化到最大值 1。

    颜色：
      - acquired ARG : blue
      - beta-lactam  : orange/red
      - mutation     : green
    """

    register_kp_colors()

    # ============================================================
    # 1. Features
    # ============================================================

    feats = [
        f for f in feature_order
        if f in bubble.index
    ]

    n_feat = len(feats)
    n_step = bubble.shape[1]


    # ============================================================
    # 2. Draw each acquisition distribution
    # ============================================================

    for row_i, f in enumerate(feats):

        # 第一条 feature 位于最上方
        y0 = n_feat - 1 - row_i

        # --------------------------------------------------------
        # 当前 feature 的 empirical distribution
        # --------------------------------------------------------

        p = (
            bubble
            .loc[f]
            .values
            .astype(float)
        )

        p = np.nan_to_num(
            p,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # --------------------------------------------------------
        # 每一行 independently normalize
        # --------------------------------------------------------

        pmax = np.max(p)

        if pmax > 0:
            q = p / pmax
        else:
            q = np.zeros_like(p)


        # --------------------------------------------------------
        # Acquisition steps
        # --------------------------------------------------------

        x = np.arange(
            1,
            n_step + 1
        )


        # ========================================================
        # 3. Resistance mechanism class -> color
        # ========================================================

        group = class_map.get(
            f,
            "other"
        )

        color = KP_GROUP_COLORS.get(
            group,
            KP_GROUP_COLORS["other"]
        )


        # ========================================================
        # 4. Histogram
        # ========================================================

        ax.bar(
            x,
            q * 0.72,
            width=0.78,
            bottom=y0 - 0.36,
            color=color,
            edgecolor="none",
            alpha=0.95,
            zorder=2
        )


        # ========================================================
        # 5. Row baseline
        #
        # 这里只是一条细基线，用于区分各 character。
        # 它不是 never-observed 区域。
        # ========================================================

        ax.hlines(
            y0 - 0.36,
            0.5,
            n_step + 0.5,
            color="#8c8c8c",
            linewidth=0.45,
            zorder=1
        )


    # ============================================================
    # 6. Y axis
    # ============================================================

    y_positions = np.arange(
        n_feat
    )[::-1]

    ax.set_yticks(
        y_positions
    )

    if show_ylabels:

        labels = [
            short_map.get(f, f)
            for f in feats
        ]

        ax.set_yticklabels(
            labels,
            fontsize=6.4
        )

        # --------------------------------------------------------
        # 标签颜色与对应 resistance mechanism 一致
        # --------------------------------------------------------

        for tick, f in zip(
            ax.get_yticklabels(),
            feats
        ):

            group = class_map.get(
                f,
                "other"
            )

            tick.set_color(
                KP_GROUP_COLORS.get(
                    group,
                    "#333333"
                )
            )

        ax.tick_params(
            axis="y",
            labelleft=True,
            length=2,
            width=0.6,
            pad=2
        )

    else:

        # 后面的 lineage 不重复写 character 名称
        ax.set_yticklabels([])

        ax.tick_params(
            axis="y",
            labelleft=False,
            length=0
        )


    # ============================================================
    # 7. X axis
    # ============================================================

    ax.set_xlim(
        0.5,
        n_step + 0.5
    )

    ax.set_ylim(
        -0.6,
        n_feat - 0.4
    )

    # 与 22-character 状态空间保持统一
    xticks = [
        1,
        4,
        7,
        10,
        13,
        16,
        19,
        22
    ]

    xticks = [
        v for v in xticks
        if v <= n_step
    ]

    ax.set_xticks(
        xticks
    )

    ax.set_xticklabels(
        [str(v) for v in xticks],
        fontsize=6.5
    )

    ax.set_xlabel(
        "acquisition step",
        fontsize=7
    )

    ax.tick_params(
        axis="x",
        labelbottom=True,
        length=3,
        width=0.6
    )


    # ============================================================
    # 8. Appearance
    # ============================================================

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 左轴线可以去掉，避免视觉过重
    ax.spines["left"].set_visible(False)

    ax.spines["bottom"].set_linewidth(0.6)

    ax.grid(False)

def character_pc_correlation(order_mat: pd.DataFrame, scores: pd.DataFrame,
                             min_groups: int = 3) -> pd.DataFrame:
    """
    每个 character 的跨分组期望序位与各主成分的相关（论文 Fig 3 的量化版）。

    `consistent`：与 PC1 的 |r| 最大 —— 序位差异主要由"该分组的观测锐度"驱动，
                  即跨分组的**相对**次序稳定（论文的 "globally consistent"）。
    `divergent` ：与 PC2/PC3 的 |r| 超过 PC1 —— 存在独立于观测噪声的真实分组间差异。
    `insufficient data`：被观测到的分组数 < `min_groups`，任何判定都是噪声。
    """
    rows = []
    for f in order_mat.index:
        v = order_mat.loc[f].astype(float)
        finite = np.isfinite(v.values)
        n_ok = int(finite.sum())
        rec = {"feature": f, "n_lineages": n_ok}
        if n_ok == 0:
            rec.update({"order_median": np.nan, "order_min": np.nan, "order_max": np.nan,
                        "order_iqr": np.nan, "order_sd_across_lineages": np.nan})
        else:
            w = v.values[finite]
            rec.update({
                "order_median": float(np.median(w)),
                "order_min": float(w.min()), "order_max": float(w.max()),
                "order_iqr": float(np.percentile(w, 75) - np.percentile(w, 25)),
                "order_sd_across_lineages": float(w.std()),
            })
        for pc in scores.columns:
            ok = finite & np.isfinite(scores[pc].values)
            rec["r_" + pc] = (float(np.corrcoef(v.values[ok], scores[pc].values[ok])[0, 1])
                              if ok.sum() > 2 else np.nan)
        pcs = list(scores.columns)
        best = max(pcs, key=lambda pc: abs(rec.get("r_" + pc) or 0.0))
        rec["dominant_pc"] = best
        if n_ok < min_groups:
            rec["class"] = "insufficient data"
        else:
            rec["class"] = ("globally consistent" if best == "PC1"
                            else "globally divergent")
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("order_median").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 4. 预测验证（论文 Fig 5B 的可得版本）
# --------------------------------------------------------------------------------------

def rank_curve(overall: pd.DataFrame, folds: pd.DataFrame,
               methods: Sequence[str] = ("TS_pretrained", "TS_random_init",
                                         "B1_prevalence", "B0_uniform"),
               ks: Sequence[int] = (1, 3, 5)) -> pd.DataFrame:
    """
    "真实下一步 character 被排进前 k 名"的比例（= Top-k accuracy）及其跨折 sd。

    论文 Fig 5B 画的是逐条转移的**排名直方图**；本项目的评估流程只落盘了折级指标，
    因此这里给出等价的累积版本（rank ≤ k）。要复现完整直方图需要在
    `eval/run_phylo_eval.py` 中额外 dump per-transition 的 rank。
    """
    o = overall.set_index("method")
    rows = []
    for m in methods:
        if m not in o.index:
            continue
        for k in ks:
            col = f"top{k}"
            v = folds[folds["method"] == m][col].values.astype(float)
            rows.append({"method": m, "k": int(k), "value": float(o.loc[m, col]),
                         "sd": float(np.nanstd(v)) if len(v) else np.nan,
                         "n_folds": int(len(v))})
        rows.append({"method": m, "k": 0, "value": float(o.loc[m, "mrr"]),
                     "sd": float(np.nanstd(folds[folds["method"] == m]["mrr"].values.astype(float))),
                     "n_folds": int((folds["method"] == m).sum()), "metric": "mrr"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 5. 数据概览图（国家覆盖 / 年份分布 / profile 组合）
# --------------------------------------------------------------------------------------

def overview_tables(ts: st.TransitionSet, samples_path: str, top_n: int = 28,
                    max_features: int = 18, pre_year_cutoff: int = 2008
                    ) -> Dict[str, pd.DataFrame]:
    """
    为数据概览图整理三张表：国家样本数、年份分布、最常见 resistance profile 的 UpSet 矩阵。

    - `country`: 每国样本数 + 国家质心坐标；颜色映射用 `log10_n_genomes`
    - `year`: 年份直方图；缺失值单列为 `No value`
    - `profiles` / `profile_matrix`: 最常见 profile 及其 character 组成
    """
    s = pd.read_csv(samples_path, dtype=str, keep_default_na=False)
    sid_col = "sample_id" if "sample_id" in s.columns else s.columns[0]

    # ---- 国家层 ----
    for c in ("latitude", "longitude"):
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    country = s.get("Country", pd.Series([""] * len(s), index=s.index)).astype(str).str.strip()
    bad = {"", "unknown", "Unknown", "nan", "NA", "None"}
    ok_country = ~country.isin(bad)
    if {"latitude", "longitude"}.issubset(s.columns):
        ok_country &= s["latitude"].notna() & s["longitude"].notna()
    country_df = (s.loc[ok_country]
                  .assign(Country=country[ok_country])
                  .groupby("Country", as_index=False)
                  .agg(n_genomes=(sid_col, "size"),
                       latitude=("latitude", "median"),
                       longitude=("longitude", "median")))
    country_df = country_df.rename(columns={"Country": "key"})
    country_df["log10_n_genomes"] = np.log10(country_df["n_genomes"].clip(lower=1))
    country_df = country_df.sort_values("n_genomes", ascending=False).reset_index(drop=True)

    # ---- 年份层 ----
    year_raw = s.get("Year", pd.Series([""] * len(s), index=s.index)).astype(str).str.strip()
    year_num = pd.to_numeric(year_raw, errors="coerce")
    rows = []
    known = year_num.dropna().astype(int)
    pre_n = int((known < pre_year_cutoff).sum())
    if pre_n > 0:
        rows.append({"year_label": f"Pre-{pre_year_cutoff}", "n_genomes": pre_n,
                     "sort_key": pre_year_cutoff - 0.5})
    for y, n in known[known >= pre_year_cutoff].value_counts().sort_index().items():
        rows.append({"year_label": str(int(y)), "n_genomes": int(n), "sort_key": float(y)})
    no_n = int(year_raw.isin(["", "unknown", "Unknown", "NA", "None", "nan"]).sum())
    if no_n > 0:
        max_sort = max([r["sort_key"] for r in rows], default=float(pre_year_cutoff))
        rows.append({"year_label": "No value", "n_genomes": no_n, "sort_key": max_sort + 1.0})
    year_df = pd.DataFrame(rows).sort_values("sort_key").reset_index(drop=True)

    # ---- 最常见 profile ----
    states = ["".join(map(str, row)) for row in ts.dest]
    vc = pd.Series(states).value_counts().head(top_n)
    profiles = pd.DataFrame({
        "profile_rank": np.arange(1, len(vc) + 1),
        "profile": vc.index.astype(str),
        "count": vc.values.astype(int),
    })
    profiles["n_characters"] = profiles["profile"].map(lambda s_: sum(int(c) for c in str(s_)))
    mat = pd.DataFrame([[int(c) for c in str(p)] for p in profiles["profile"]],
                       columns=list(ts.feature_names), index=profiles["profile_rank"])
    feat_weight = (mat.T * profiles["count"].values).sum(axis=1).sort_values(ascending=False)
    feat_order = feat_weight[feat_weight > 0].index.tolist()[:max_features]
    profile_mat = mat.loc[:, feat_order].T
    profile_mat.columns = profiles["profile_rank"].astype(str)
    profile_mat.index.name = "feature"
    return {"country": country_df, "year": year_df,
            "profiles": profiles, "profile_matrix": profile_mat}


def panel_overview_map(ax, country_df: pd.DataFrame, annotate_top: int = 0):
    """
    全球采样覆盖的国家 choropleth map。

    国家填充颜色 = log10(#genomes)
    无数据国家 = 浅灰色
    国界 = 灰色细线

    annotate_top > 0 时，可选标注样本量最多的若干国家。
    默认不标国家名称。

    注意：
    - 显示轴应在 make_dataset_overview_figure() 中使用 Robinson projection。
    - 原始经纬度数据仍使用 PlateCarree() 进行 transform。
    """

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    import matplotlib as mpl

    # ------------------------------------------------------------------
    # 数据
    # ------------------------------------------------------------------
    d = country_df.dropna(
        subset=["latitude", "longitude", "log10_n_genomes"]
    ).copy()

    # 国家名称 -> log10(genome count)
    value_map = dict(
        zip(
            d["key"].astype(str),
            d["log10_n_genomes"].astype(float)
        )
    )

    # ------------------------------------------------------------------
    # Natural Earth 国家名称和数据中名称的常见映射
    # ------------------------------------------------------------------
    name_alias = {
        "United States of America": "USA",
        "United States": "USA",

        "United Kingdom": "UK",

        "Russian Federation": "Russia",
        "Russia": "Russia",

        "Viet Nam": "Vietnam",

        "Republic of Korea": "South Korea",
        "South Korea": "South Korea",

        "Democratic Republic of the Congo":
            "Democratic Republic of the Congo",
        "Dem. Rep. Congo":
            "Democratic Republic of the Congo",

        "Czechia": "Czech Republic",

        "Iran": "Iran",
        "Iran (Islamic Republic of)": "Iran",

        "Lao PDR": "Laos",

        "United Republic of Tanzania": "Tanzania",

        "Bolivia": "Bolivia",
        "Bolivia (Plurinational State of)": "Bolivia",

        "Venezuela": "Venezuela",
        "Venezuela (Bolivarian Republic of)": "Venezuela",

        "Côte d'Ivoire": "Cote d'Ivoire",
        "Ivory Coast": "Cote d'Ivoire",
    }

    # ------------------------------------------------------------------
    # 色阶
    # ------------------------------------------------------------------
    vals = d["log10_n_genomes"].astype(float).values

    vmin = 0.0
    vmax = max(
        3.0,
        float(np.nanmax(vals))
    )

    cmap = plt.get_cmap("inferno_r")

    norm = mpl.colors.Normalize(
        vmin=vmin,
        vmax=vmax
    )

    # ------------------------------------------------------------------
    # 世界地图
    # ------------------------------------------------------------------
    ax.set_global()

    # 海洋背景
    ax.add_feature(
        cfeature.OCEAN.with_scale("110m"),
        facecolor="white",
        edgecolor="none",
        zorder=0
    )

    # --------------------------------------------------------------
    # Natural Earth 国家 polygon
    # --------------------------------------------------------------
    shpfilename = shpreader.natural_earth(
        resolution="110m",
        category="cultural",
        name="admin_0_countries"
    )

    reader = shpreader.Reader(
        shpfilename
    )

    for record in reader.records():

        attrs = record.attributes

        ne_name = (
            attrs.get("ADMIN")
            or attrs.get("NAME_LONG")
            or attrs.get("NAME")
            or ""
        )

        data_name = name_alias.get(
            ne_name,
            ne_name
        )

        if data_name in value_map:

            value = value_map[data_name]

            facecolor = cmap(
                norm(value)
            )

        else:

            # 无样本国家
            facecolor = "#f4f4f4"

        ax.add_geometries(
            [record.geometry],
            crs=ccrs.PlateCarree(),
            facecolor=facecolor,
            edgecolor="#b8b8b8",
            linewidth=0.28,
            zorder=1
        )

    # ------------------------------------------------------------------
    # 海岸线
    # ------------------------------------------------------------------
    ax.add_feature(
        cfeature.COASTLINE.with_scale("110m"),
        edgecolor="#999999",
        linewidth=0.35,
        zorder=2
    )

    # ------------------------------------------------------------------
    # 弱化经纬网格
    # ------------------------------------------------------------------
    ax.gridlines(
        draw_labels=False,
        linewidth=0.35,
        color="#dddddd",
        alpha=0.60,
        linestyle="-",
        zorder=0
    )

    # ------------------------------------------------------------------
    # 可选国家名称标注
    # ------------------------------------------------------------------
    if annotate_top > 0:

        top = (
            d.sort_values(
                "n_genomes",
                ascending=False
            )
            .head(annotate_top)
        )

        for _, r in top.iterrows():

            ax.text(
                float(r["longitude"]),
                float(r["latitude"]),
                str(r["key"]),
                fontsize=5.8,
                ha="center",
                va="center",
                transform=ccrs.PlateCarree(),
                zorder=5
            )

    # ------------------------------------------------------------------
    # 标题
    # ------------------------------------------------------------------
    ax.set_title(
        "Global sampling coverage",
        fontsize=9
    )

    # ------------------------------------------------------------------
    # Compact publication-style colorbar
    # ------------------------------------------------------------------
    sm = mpl.cm.ScalarMappable(
        norm=norm,
        cmap=cmap
    )

    sm.set_array([])

    fig = ax.get_figure()

    # 获取地图 panel 在整张 figure 中的位置
    pos = ax.get_position()

    # 手动建立 colorbar axis
    #
    # left   = 地图右侧稍微留一点间距
    # bottom = 地图高度约 26% 的位置
    # width  = 很窄
    # height = 地图高度约 48%
    #
    # 这样不会像默认 colorbar 一样贯穿整个地图高度。
    cax = fig.add_axes([
        pos.x1 + 0.012,
        pos.y0 + pos.height * 0.26,
        0.009,
        pos.height * 0.48
    ])

    cb = fig.colorbar(
        sm,
        cax=cax,
        orientation="vertical"
    )

    cb.set_label(
        "log$_{10}$ genomes",
        fontsize=7,
        labelpad=4
    )

    cb.ax.tick_params(
        labelsize=6,
        length=2,
        width=0.6
    )

    cb.outline.set_linewidth(
        0.6
    )
def panel_overview_years(ax, year_df: pd.DataFrame):
    """概览图左下：按年份统计样本数，缺失值单独保留。"""
    d = year_df.copy()
    x = np.arange(len(d))
    colors = ["#4d4d4d" if str(v) in ("No value",) or str(v).startswith("Pre-") else "#6a6a6a"
              for v in d["year_label"]]
    ax.bar(x, d["n_genomes"].values.astype(float), color=colors, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(d["year_label"].astype(str), rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Genomes", fontsize=8)
    ax.set_xlabel("Isolation year", fontsize=8)
    ax.set_title("Sampling over isolation years", fontsize=9)
    ax.grid(axis="y", color="#ececec", lw=0.6)
    ax.set_axisbelow(True)


def panel_profile_upset(ax_bar, ax_mat, profiles: pd.DataFrame, profile_mat: pd.DataFrame,
                        short_map: Dict[str, str]):
    """
    概览图右侧：最常见 resistance profile 的 UpSet 风格主条形图 + 组成矩阵。
    """
    p = profiles.reset_index(drop=True)
    x = np.arange(len(p))
    cmap = plt.get_cmap("PuBu")
    cmin = float(p["n_characters"].min())
    cmax = float(p["n_characters"].max())
    # 抬高色标下限，避免最少 character 数的 bar 落到接近白的浅色区
    lo = cmin - 0.55 * max(1e-9, cmax - cmin)
    norm = plt.Normalize(vmin=lo, vmax=cmax)
    ax_bar.bar(x, p["count"].values.astype(float),
               color=[cmap(norm(v)) for v in p["n_characters"].values.astype(float)],
               edgecolor="none")
    ax_bar.set_xlim(-0.6, len(p) - 0.4)
    ax_bar.set_ylabel("Number of\ngenomes", fontsize=8)
    ax_bar.set_xticks([])
    ax_bar.set_title("Most common resistance character profiles", fontsize=9)
    ax_bar.grid(axis="y", color="#ececec", lw=0.6)
    ax_bar.set_axisbelow(True)

    M = profile_mat.copy()
    feats = list(M.index)
    yy = np.arange(len(feats))[::-1]
    for xi in range(M.shape[1]):
        ax_mat.scatter([xi] * len(yy), yy, s=11, color="#d9d9d9", zorder=1)
        present = np.flatnonzero(M.iloc[:, xi].values.astype(int) > 0)
        if len(present):
            yp = yy[present]
            ax_mat.plot([xi, xi], [yp.min(), yp.max()], color="black", lw=0.9, zorder=2)
            ax_mat.scatter([xi] * len(yp), yp, s=14, color="black", zorder=3)
    ax_mat.set_xlim(-0.6, M.shape[1] - 0.4)
    ax_mat.set_ylim(-0.5, len(feats) - 0.5)
    ax_mat.set_yticks(yy)
    ax_mat.set_yticklabels([short_map.get(f, f) for f in feats], fontsize=7)
    ax_mat.set_xticks(x)
    ax_mat.set_xticklabels([str(v) for v in p["profile_rank"]], fontsize=6.3)
    ax_mat.set_xlabel("Resistance character profiles (rank)", fontsize=8)
    ax_mat.grid(axis="x", color="#f0f0f0", lw=0.5)
    ax_mat.set_axisbelow(True)


def panel_overview_year_by_country(ax, samples_path: str, top_n: int = 15,
                                   pool_before: Optional[int] = 2000,
                                   key_col: str = "Country",
                                   show_global_margin: bool = True):
    """
    概览图 Aii 的升级版：`国家 × 年份` 采样热图（取样本量前 `top_n` 国，其余合并为一行）。

    相比原来的"全局年份条形图"，这里同时给出**采样在国家之间的时间错位**——
    某些国家只在个别年份被集中采样，这是解读后续国家间差异的必要背景。
    `show_global_margin=True` 时在上方加一条全局逐年样本量的边缘条形，
    因此原条形图的信息没有丢失。

    返回实际绘制的计数矩阵（含 "other countries" 行）。
    """
    counts, summary = country_year_table(samples_path, key_col=key_col)
    if counts.empty:
        ax.text(0.5, 0.5, "no collection-year information", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#888888")
        ax.set_axis_off()
        return counts

    totals = counts.sum(axis=1).sort_values(ascending=False)
    # 国家字段缺失的样本（"unknown" / 空值）不单独占一行，并入 other countries
    unknown_like = {"", "unknown", "Unknown", "UNKNOWN", "na", "NA", "nan", "None"}
    ranked = [k for k in totals.index if str(k).strip() not in unknown_like]
    top = ranked[:int(top_n)]
    med = summary.set_index("key")["median"]
    top = sorted(top, key=lambda k: (med.get(k, np.inf), k))       # 按中位采样年排序
    C = counts.loc[top].copy()
    rest = counts.drop(index=top, errors="ignore")
    if len(rest):
        C.loc["other countries (n=%d)" % len(rest)] = rest.sum(axis=0)

    if pool_before is not None:
        old = [c for c in C.columns if int(c) < int(pool_before)]
        if old:
            pooled = C[old].sum(axis=1)
            C = C.drop(columns=old)
            C.insert(0, "\u2264%d" % (int(pool_before) - 1), pooled)

    years = list(C.columns)
    M = np.log10(1.0 + C.values.astype(float))
    cmap = plt.get_cmap("YlGnBu").copy()
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=1e-9,
                   vmax=max(float(M.max()), 0.5), interpolation="nearest")
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels([str(y) for y in years], rotation=90, fontsize=6.0)
    ax.set_yticks(np.arange(C.shape[0]))
    ax.set_yticklabels(C.index.astype(str), fontsize=6.2)
    if len(rest):
        ax.get_yticklabels()[-1].set_style("italic")
    ax.set_xlabel("Isolation year", fontsize=8)
    n_undated = int(summary["n_samples"].sum() - summary["n_dated"].sum())
    ax.set_title("Sampling over isolation years, by country\n"
                 "colour = log10(1 + #genomes); %d genomes without a year are omitted"
                 % n_undated, fontsize=9, pad=34)
    cb = ax.get_figure().colorbar(im, ax=ax, pad=0.012, fraction=0.030)
    cb.set_label("log10(1 + #genomes)", fontsize=6.6)
    cb.ax.tick_params(labelsize=5.8)

    if show_global_margin:
        ax_top = ax.inset_axes([0.0, 1.02, 1.0, 0.20], sharex=ax)
        gtot = counts.copy()
        if pool_before is not None:
            old = [c for c in gtot.columns if int(c) < int(pool_before)]
            if old:
                pooled = gtot[old].sum(axis=1)
                gtot = gtot.drop(columns=old)
                gtot.insert(0, "pooled", pooled)
        ax_top.bar(np.arange(gtot.shape[1]), gtot.sum(axis=0).values.astype(float),
                   color="#6a6a6a", edgecolor="white", linewidth=0.3)
        ax_top.set_xlim(-0.5, gtot.shape[1] - 0.5)
        ax_top.tick_params(axis="x", labelbottom=False, bottom=False)
        ax_top.tick_params(axis="y", labelsize=5.8)
        ax_top.set_ylabel("all\ngenomes", fontsize=6.0)
        for side in ("top", "right"):
            ax_top.spines[side].set_visible(False)
        ax_top.grid(axis="y", color="#EFEFEF", lw=0.5)
        ax_top.set_axisbelow(True)
    return C


def make_dataset_overview_figure(ts: st.TransitionSet, samples_path: str,
                                 short_map: Dict[str, str], top_n_profiles: int = 28,
                                 max_profile_features: int = 18,
                                 year_by_country: bool = True,
                                 year_top_n: int = 15,
                                 save_prefix: Optional[str] = None):
    """
    组装数据概览图：左侧是国家分布与年份分布，右侧是最常见 resistance profile 的 UpSet 图。

    `year_by_country=True` 时 Aii 画 `国家 × 年份` 采样热图（上方带全局逐年条形），
    比单纯的全局年份条形图多出"采样在国家之间如何错位"这一层信息；
    设为 False 则回到原来的全局条形图。
    """
    tabs = overview_tables(ts, samples_path, top_n=top_n_profiles,
                           max_features=max_profile_features)
    try:
        import cartopy.crs as ccrs
        ax_map_proj = ccrs.PlateCarree()
    except Exception:
        ax_map_proj = None

    fig = plt.figure(figsize=(13.4, 8.3 if not year_by_country else 9.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.00, 1.18],
                          height_ratios=[1.0, 0.50 if not year_by_country else 0.85],
                          wspace=0.28, hspace=0.22 if not year_by_country else 0.42)
    if ax_map_proj is not None:
        ax_map = fig.add_subplot(gs[0, 0], projection=ax_map_proj)
    else:
        ax_map = fig.add_subplot(gs[0, 0])
    ax_year = fig.add_subplot(gs[1, 0])
    gs_r = gs[:, 1].subgridspec(2, 1, height_ratios=[0.32, 1.0], hspace=0.03)
    ax_bar = fig.add_subplot(gs_r[0, 0])
    ax_mat = fig.add_subplot(gs_r[1, 0], sharex=ax_bar)

    panel_overview_map(ax_map, tabs["country"])
    if year_by_country:
        tabs["year_by_country"] = panel_overview_year_by_country(
            ax_year, samples_path, top_n=year_top_n)
    else:
        panel_overview_years(ax_year, tabs["year"])
    panel_profile_upset(ax_bar, ax_mat, tabs["profiles"], tabs["profile_matrix"], short_map)
    pf.panel_label(ax_map, "Ai", dx=-0.055)
    pf.panel_label(ax_year, "Aii", dx=-0.055, dy=1.30 if year_by_country else 1.06)
    pf.panel_label(ax_bar, "B", dx=-0.070)

    if save_prefix:
        for ext in ("png", "pdf"):
            fig.savefig(f"{save_prefix}.{ext}", dpi=350, bbox_inches="tight")
    return fig, tabs


# --------------------------------------------------------------------------------------
# 6. 主图 Panel 绘制
# --------------------------------------------------------------------------------------

def panel_a_profiles(ax, ts: st.TransitionSet, short_map: Dict[str, str], top_n: int = 22):
    """A：最常见的 AMR character profile 计数（论文 Fig 1B 的 UpSet 主条形）。"""
    states = ["".join(map(str, row)) for row in ts.dest]
    vc = pd.Series(states).value_counts().head(top_n)
    n_char = np.array([sum(int(c) for c in s) for s in vc.index], dtype=float)
    x = np.arange(len(vc))
    cmap = plt.get_cmap("viridis")
    norm = (n_char - n_char.min()) / max(n_char.max() - n_char.min(), 1e-9)
    ax.bar(x, vc.values, color=[cmap(v) for v in norm], edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(len(vc))], fontsize=6.5)
    ax.set_xlabel("most common AMR character profiles (ranked; colour = #characters carried)",
                  fontsize=8)
    ax.set_ylabel("#descendant states", fontsize=8)
    ax.set_title("Most common resistance profiles", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=n_char.min(), vmax=n_char.max()))
    cb = ax.get_figure().colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("#characters", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    return pd.DataFrame({"profile": vc.index, "count": vc.values, "n_characters": n_char})


def panel_c_pca(ax, scores: pd.DataFrame, evr: np.ndarray, meta: pd.DataFrame,
                pcs: Tuple[str, str] = ("PC1", "PC2"), annotate_top: int = 8):
    """C：谱系特异动力学的 PCA（论文 Fig 2Bi）。点大小 = 该谱系的获得事件数。"""
    m = meta.set_index("group_id").reindex(scores.index)
    sizes = np.clip(m["n_gain_events"].values.astype(float) / 8.0, 12, 260)
    sc = ax.scatter(scores[pcs[0]], scores[pcs[1]], s=sizes,
                    c=m["mean_depth"].values.astype(float), cmap="magma_r",
                    alpha=0.85, edgecolors="white", linewidths=0.6)
    for g in m["n_gain_events"].sort_values(ascending=False).head(annotate_top).index:
        if g in scores.index:
            ax.annotate(str(g), (scores.loc[g, pcs[0]], scores.loc[g, pcs[1]]),
                        fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    i, j = int(pcs[0][2:]) - 1, int(pcs[1][2:]) - 1
    ax.set_xlabel("%s (%.0f%% var)" % (pcs[0], 100 * evr[i]), fontsize=8)
    ax.set_ylabel("%s (%.0f%% var)" % (pcs[1], 100 * evr[j]), fontsize=8)
    ax.set_title("Lineage-specific acquisition dynamics (PCA)\n"
                 "point size = #gain events, colour = mean resistance depth", fontsize=9)
    cb = ax.get_figure().colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("mean #characters in source state", fontsize=7)
    cb.ax.tick_params(labelsize=6)


def panel_d_consistency(ax, corr: pd.DataFrame, short_map: Dict[str, str],
                        class_map: Dict[str, str], min_lineages: int = 5):
    """
    D：全局一致 vs 全局分歧 character（论文 Fig 3 的紧凑版）。

    每个 character 一行：跨谱系期望序位的 min–max 区间 + 中位数点；
    ▲ 标记 = 分歧（与 PC2/PC3 的协变强于 PC1）；右侧标注主导 PC 与 |r|。
    """
    d = corr[(corr["n_lineages"] >= min_lineages)
             & (corr["class"] != "insufficient data")].sort_values("order_median")
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        f = str(r["feature"])
        col = pf.CLASS_COLORS.get(class_map.get(f, "other"), "#999999")
        div = r["class"] == "globally divergent"
        ax.hlines(yi, r["order_min"], r["order_max"], color=col, lw=2.2,
                  alpha=0.45 if not div else 0.9)
        ax.scatter(r["order_median"], yi, s=34, marker="^" if div else "o",
                   color=col, edgecolors="white", linewidths=0.6, zorder=3)
        rr = r.get("r_" + str(r["dominant_pc"]), np.nan)
        ax.text(1.005, yi, "%s |r|=%.2f" % (r["dominant_pc"], abs(rr) if rr == rr else float("nan")),
                transform=ax.get_yaxis_transform(), fontsize=6.0, va="center",
                color="#c0392b" if div else "#777777")
    ax.set_yticks(y)
    ax.set_yticklabels([short_map.get(str(f), str(f)) for f in d["feature"]], fontsize=7)
    for tick, f in zip(ax.get_yticklabels(), d["feature"]):
        tick.set_color(pf.CLASS_COLORS.get(class_map.get(str(f), "other"), "#333333"))
    ax.set_xlabel("expected acquisition order across lineages (min \u2013 median \u2013 max)",
                  fontsize=8)
    ax.set_title("Globally consistent vs divergent characters\n"
                 "(\u25b2 = divergent: covaries with PC2/PC3 rather than PC1)", fontsize=9)


def panel_e_rank(ax, curve: pd.DataFrame, note: str = ""):
    """E：真实下一步 character 的预测排名（rank ≤ k）。"""
    del note  # 主图右下角不再放说明框，避免遮挡 rank ≤ 5 的柱形
    ks = sorted(curve[curve["k"] > 0]["k"].unique())
    order = [m for m in ("TS_pretrained", "TS_random_init", "B1_prevalence", "B0_uniform")
             if m in set(curve["method"])]
    panel_colors = {
        "TS_pretrained": "#2C7FB8",
        "TS_random_init": "#7FB6D6",
        "B1_prevalence": "#B8B8B8",
        "B0_uniform": "#E2E2E2",
    }
    group_w = 0.62
    w = group_w / max(1, len(order))
    x = np.arange(len(ks))
    for i, m in enumerate(order):
        sub = curve[(curve["method"] == m) & (curve["k"] > 0)].set_index("k").reindex(ks)
        ax.bar(x + (i - (len(order) - 1) / 2) * w, sub["value"], width=w * 0.88,
               yerr=sub["sd"], capsize=2, error_kw=dict(lw=0.8, ecolor="#4d4d4d"),
               color=panel_colors.get(m, pf.METHOD_COLORS.get(m, "#4C72B0")),
               edgecolor="white", linewidth=0.5,
               label=pf.METHOD_LABELS.get(m, m).replace("\n", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(["rank $\\leq$ %d" % k for k in ks], fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("fraction of transitions", fontsize=8)
    ax.set_title("Prediction of the character truly acquired next\n"
                 "(lineage-blocked CV; bars = CV mean, whiskers = across-fold SD)",
                 fontsize=9)
    ax.legend(fontsize=6.4, loc="upper left", frameon=True,
              facecolor="white", edgecolor="#D8D8D8", framealpha=0.95)
def panel_b_forward_walks(
        ax,
        bubble: pd.DataFrame,
        feature_order: Sequence[str],
        class_map: Dict[str, str],
        short_map: Dict[str, str],
        bimodal_features: Optional[Sequence[str]] = None
):
    """
    KP Panel B:
    仅展示 two-stage forward walks 推演得到的 acquisition-step distributions。

    - 每一行 = 一个 resistance character
    - 柱高 = 该 character 在不同 ordering step 被获得的相对概率
    - 每一行独立归一化到最大值 1
    - 颜色 = resistance mechanism group
    - ✦ = bimodal acquisition distribution
    """

    register_kp_colors()

    hi = set(bimodal_features or [])

    # 保证 feature_order 中的特征确实存在于 bubble
    feats = [f for f in feature_order if f in bubble.index]

    n_feat = len(feats)
    n_step = bubble.shape[1]

    # 每个 character 一条横向 strip
    row_height = 1.0

    for row_i, f in enumerate(feats):

        # 从上往下排
        y0 = n_feat - 1 - row_i

        p = bubble.loc[f].values.astype(float)

        # 每一行独立缩放
        pmax = np.nanmax(p)
        if np.isfinite(pmax) and pmax > 0:
            q = p / pmax
        else:
            q = np.zeros_like(p)

        col = pf.CLASS_COLORS.get(
            class_map.get(f, "other"),
            "#777777"
        )

        # 轻微灰底，帮助区分各行
        ax.add_patch(
            Rectangle(
                (0.5, y0 - 0.42),
                n_step,
                0.84,
                facecolor="#f7f7f7",
                edgecolor="none",
                zorder=0
            )
        )

        # acquisition distribution
        x = np.arange(1, n_step + 1)

        ax.bar(
            x,
            q * 0.72,
            width=0.72,
            bottom=y0 - 0.36,
            color=col,
            edgecolor="none",
            alpha=0.95,
            zorder=2
        )

        # 每行 baseline
        ax.hlines(
            y0 - 0.36,
            0.5,
            n_step + 0.5,
            color="#777777",
            linewidth=0.45,
            zorder=1
        )

    # ------------------------------------------------------------------
    # y labels
    # ------------------------------------------------------------------
    y_positions = np.arange(n_feat)[::-1]

    labels = [
        short_map.get(f, f) + (" ✦" if f in hi else "")
        for f in feats
    ]

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7.2)

    for tick, f in zip(ax.get_yticklabels(), feats):
        tick.set_color(
            pf.CLASS_COLORS.get(
                class_map.get(f, "other"),
                "#333333"
            )
        )

    # ------------------------------------------------------------------
    # x axis
    # ------------------------------------------------------------------
    ax.set_xlim(0.5, n_step + 0.5)
    ax.set_ylim(-0.6, n_feat - 0.4)

    xticks = [1, 4, 7, 10, 13, 16, 19, n_step]
    xticks = sorted(set(x for x in xticks if x <= n_step))

    ax.set_xticks(xticks)

    ax.set_xlabel(
        "number of features already acquired (ordering step)",
        fontsize=8
    )

    # ------------------------------------------------------------------
    # title
    # ------------------------------------------------------------------
    ax.set_title(
        "Predicted acquisition dynamics from two-stage forward walks\n"
        "✦ = bimodal acquisition distribution",
        fontsize=9
    )

    # ------------------------------------------------------------------
    # legend
    # ------------------------------------------------------------------
    legend_handles = []

    for group, color in KP_GROUP_COLORS.items():
        if group == "other":
            continue

        legend_handles.append(
            plt.Line2D(
                [0], [0],
                marker="s",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=6,
                label=group
            )
        )

    ax.legend(
        handles=legend_handles,
        fontsize=6.5,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.18),
        ncol=3
    )

    # ------------------------------------------------------------------
    # aesthetics
    # ------------------------------------------------------------------
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="x",
        labelsize=7
    )

def make_kp_figure(
        ts: st.TransitionSet,
        bubbles: Dict[str, pd.DataFrame],
        bimodal: pd.DataFrame,
        scores: pd.DataFrame,
        evr: np.ndarray,
        meta: pd.DataFrame,
        corr: pd.DataFrame,
        curve: pd.DataFrame,
        short_map: Dict[str, str],
        class_map: Dict[str, str],
        observed: Optional[pd.DataFrame] = None,
        model_key: str = "two_stage",
        scalability_note: str = "",
        suptitle: str = "",
        save_prefix: Optional[str] = None
):
    """
    K. pneumoniae 主图：

    A — predicted acquisition dynamics
    B — lineage-specific PCA
    C — lineage consistency/divergence
    D — next-step prediction

    注意：
    - 删除原来的 Most common resistance profiles
    - Prediction 中删除 TS_random_init
    """

    register_kp_colors()

    # ============================================================
    # 主文 prediction 只保留三个方法
    # ============================================================
    main_methods = [
        "TS_pretrained",
        "B1_prevalence",
        "B0_uniform",
    ]

    curve_main = curve[
        curve["method"].isin(main_methods)
    ].copy()

    if curve_main.empty:
        raise ValueError(
            "No prediction data remain after filtering. "
            f"Available methods: {list(curve['method'].unique())}"
        )

    # ============================================================
    # Layout
    #
    # A：forward walks
    # B/C：lineage analysis
    # D：prediction
    # ============================================================
    fig = plt.figure(figsize=(12.0, 12.0))

    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[2.8, 1.55, 1.25],
        hspace=0.62,
        wspace=0.30
    )

    # ============================================================
    # A — forward-walk acquisition dynamics
    # ============================================================
    ax_a = fig.add_subplot(gs[0, :])

    if model_key not in bubbles:
        raise KeyError(
            f"model_key={model_key!r} not found. "
            f"Available bubbles: {list(bubbles.keys())}"
        )

    model_bubble = bubbles[model_key]

    model_order = (
        st.expected_ordering(model_bubble)
        .dropna(subset=["expected_order"])
        .sort_values("expected_order")
    )

    feature_order = (
        model_order["feature"]
        .astype(str)
        .tolist()
    )

    if (
        bimodal is not None
        and len(bimodal)
        and "is_bimodal" in bimodal.columns
    ):
        hi = set(
            bimodal.loc[
                bimodal["is_bimodal"],
                "feature"
            ].astype(str)
        )
    else:
        hi = set()

    panel_b_forward_walks(
        ax_a,
        bubble=model_bubble,
        feature_order=feature_order,
        class_map=class_map,
        short_map=short_map,
        bimodal_features=hi
    )

    pf.panel_label(
        ax_a,
        "A",
        dx=-0.055
    )

    # ============================================================
    # B — lineage PCA
    # ============================================================
    ax_b = fig.add_subplot(gs[1, 0])

    panel_c_pca(
        ax_b,
        scores,
        evr,
        meta
    )

    pf.panel_label(
        ax_b,
        "B"
    )

    # ============================================================
    # C — lineage consistency/divergence
    # ============================================================
    ax_c = fig.add_subplot(gs[1, 1])

    panel_d_consistency(
        ax_c,
        corr,
        short_map,
        class_map
    )

    ax_c.set_title(
        "Lineage consistency of resistance acquisition\n"
        "▲ = lineage-divergent character",
        fontsize=9
    )

    ax_c.set_xlabel(
        "expected acquisition order across lineages "
        "(min – median – max)",
        fontsize=8
    )

    pf.panel_label(
        ax_c,
        "C"
    )

    # ============================================================
    # D — prediction performance
    # ============================================================
    ax_d = fig.add_subplot(gs[2, :])

    panel_e_rank(
        ax_d,
        curve_main,
        note=scalability_note
    )

    ax_d.set_title(
        "Next-step resistance acquisition prediction\n"
        "(lineage-blocked cross-validation)",
        fontsize=9
    )

    pf.panel_label(
        ax_d,
        "D",
        dx=-0.055
    )

    # ============================================================
    # Title
    # ============================================================
    if suptitle:
        fig.suptitle(
            suptitle,
            y=0.995,
            fontsize=11.5,
            fontweight="bold"
        )

    # ============================================================
    # Save
    # ============================================================
    if save_prefix:
        for ext in ("png", "pdf"):
            fig.savefig(
                f"{save_prefix}.{ext}",
                dpi=350,
                bbox_inches="tight"
            )

    return fig

def load_stratified_sets(strat_dir: str, processed_dir: str, min_transitions: int = 30,
                         max_groups: int = 0
                         ) -> Tuple[Dict[str, "st.TransitionSet"], pd.DataFrame]:
    """
    读入 `stratify_transitions_tree.py` 的产物目录，返回 {分层键: TransitionSet} 与元表。

    元表列：`key / safe_name / n_samples / n_transitions / n_gain_events /
    mean_depth / multi_gain_fraction / n_cv_groups / dir`。
    """
    man_path = os.path.join(strat_dir, "manifest.json")
    if not os.path.isfile(man_path):
        raise FileNotFoundError(
            "未找到 %s —— 请先运行 pipeline/stratify_transitions_tree.py" % man_path)
    with open(man_path, encoding="utf-8") as fh:
        man = json.load(fh)
    sets: Dict[str, "st.TransitionSet"] = {}
    rows = []
    groups = sorted(man.get("groups", []),
                    key=lambda g: -int(g.get("n_informative_transitions", 0)))
    for g in groups:
        if int(g.get("n_informative_transitions", 0)) < min_transitions:
            continue
        if max_groups and len(sets) >= max_groups:
            break
        path = os.path.join(strat_dir, g["safe_name"], "transitions.csv")
        if not os.path.isfile(path):
            continue
        ts = st.load_transition_set(path, processed_dir)
        key = str(g["key"])
        sets[key] = ts
        rows.append({
            "key": key, "safe_name": g["safe_name"],
            "n_samples": int(g.get("n_samples", 0)),
            "n_transitions": int(ts.n),
            "n_gain_events": int(ts.frame["n_gained_parsed"].sum()),
            "mean_depth": float(ts.frame["n_src_features"].mean()),
            "multi_gain_fraction": float((ts.frame["n_gained_parsed"] > 1).mean()),
            "n_cv_groups": int(ts.frame["group_id"].nunique()),
            "transitions_path": path,
        })
    meta = pd.DataFrame(rows)
    meta.attrs["stratify_by"] = man.get("stratify_by", "")
    return sets, meta


def stratum_bubbles(sets: Dict[str, "st.TransitionSet"]) -> Dict[str, pd.DataFrame]:
    """每个分层组各自的经验 bubble 矩阵（feature × step）。"""
    return {k: st.empirical_bubble_matrix(ts) for k, ts in sets.items()}


def attach_geo(meta: pd.DataFrame, samples_path: str, key_col: str = "Country"
               ) -> pd.DataFrame:
    """
    给分层元表补上地理坐标与所属大区，并统计该组样本的耐药负担。

    `latitude / longitude` 在上游是**国家质心**（同国样本完全相同），因此只适合画
    "国家级"散点地图，不能当作样本坐标使用。
    """
    s = pd.read_csv(samples_path, dtype=str, keep_default_na=False)
    num = ["latitude", "longitude", "resistance_score", "virulence_score",
           "num_resistance_classes"]
    for c in num:
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    agg: Dict[str, object] = {}
    if "latitude" in s.columns:
        agg["latitude"] = ("latitude", "median")
        agg["longitude"] = ("longitude", "median")
    if "resistance_score" in s.columns:
        agg["resistance_score"] = ("resistance_score", "mean")
    if "Region" in s.columns:
        agg["region"] = ("Region", lambda v: v.value_counts().index[0] if len(v) else "")
    if "Year" in s.columns:
        yr = pd.to_numeric(s["Year"], errors="coerce")
        s = s.assign(_year=yr)
        agg["year_median"] = ("_year", "median")
    g = s.groupby(key_col).agg(**agg).reset_index().rename(columns={key_col: "key"})
    out = meta.merge(g, on="key", how="left")
    if "region" in out.columns:
        out["continent"] = out["region"].astype(str).str.split("_").str[0]
    return out


def country_year_table(samples_path: str, keys: Optional[Sequence[str]] = None,
                       key_col: str = "Country", year_col: str = "Year"
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    统计各国采样年份分布，返回 `(counts, summary)`。

    - `counts`：`国家 × 年份` 的样本计数矩阵（年份列按升序，缺失年份的样本不计入）；
    - `summary`：每国的 `n_samples / n_dated / year_min / q25 / median / q75 / year_max /
      span / n_years / undated_fraction`。

    采样年份是解读国家间差异的必要背景：不同国家的采样窗口本身就不同，
    "获得顺序偏早/偏晚"有可能只是**采样年代**而非流行病学差异造成的。
    """
    s = pd.read_csv(samples_path, dtype=str, keep_default_na=False)
    if key_col not in s.columns:
        raise KeyError("samples.csv 里没有列 %s" % key_col)
    s["_year"] = pd.to_numeric(s.get(year_col, ""), errors="coerce")
    if keys is not None:
        s = s[s[key_col].astype(str).isin({str(k) for k in keys})]

    dated = s.dropna(subset=["_year"]).copy()
    dated["_year"] = dated["_year"].astype(int)
    counts = (dated.groupby([key_col, "_year"]).size().unstack(fill_value=0)
              .sort_index(axis=1))
    counts.index.name = "key"
    counts.columns = [int(c) for c in counts.columns]

    rows = []
    for key, grp in s.groupby(key_col):
        v = grp["_year"].dropna().astype(float)
        rec = {"key": str(key), "n_samples": int(len(grp)), "n_dated": int(len(v))}
        if len(v):
            rec.update({
                "year_min": float(v.min()),
                "p05": float(np.percentile(v, 5)),
                "q25": float(np.percentile(v, 25)),
                "median": float(np.median(v)),
                "q75": float(np.percentile(v, 75)),
                "p95": float(np.percentile(v, 95)),
                "year_max": float(v.max()),
                "span": float(v.max() - v.min()),
                "n_years": int(v.nunique()),
            })
        else:
            rec.update({k: np.nan for k in ("year_min", "p05", "q25", "median", "q75",
                                            "p95", "year_max", "span")})
            rec["n_years"] = 0
        rec["undated_fraction"] = float(1.0 - rec["n_dated"] / max(rec["n_samples"], 1))
        rows.append(rec)
    summary = pd.DataFrame(rows).sort_values("median").reset_index(drop=True)
    return counts, summary


def panel_country_year_heatmap(ax, counts: pd.DataFrame, order: Sequence[str],
                               meta: Optional[pd.DataFrame] = None,
                               pool_before: Optional[int] = 2000,
                               highlight: Optional[Sequence[str]] = None,
                               title: Optional[str] = None,
                               colorbar_fraction: float = 0.03,
                               tick_fontsize: float = 6.6):
    """
    各国采样年份的计数热图（颜色 = log10(1+样本数)）。

    `pool_before` 之前的年份合并成最左一列（本数据集有个别 1920–1990 年代的历史菌株，
    不合并会让 70 多列里绝大部分是空白）。`highlight` 里的国家标签加粗。
    """
    C = counts.reindex([k for k in order if k in counts.index]).fillna(0.0)
    if pool_before is not None:
        old = [c for c in C.columns if int(c) < int(pool_before)]
        if old:
            pooled = C[old].sum(axis=1)
            C = C.drop(columns=old)
            C.insert(0, "\u2264%d" % (int(pool_before) - 1), pooled)
    years = list(C.columns)
    M = np.log10(1.0 + C.values.astype(float))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_under("#F7F7F7")
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=1e-9,
                   vmax=max(float(M.max()), 0.5), interpolation="nearest")
    step = max(1, len(years) // 14)
    ax.set_xticks(np.arange(0, len(years), step))
    ax.set_xticklabels([years[i] for i in range(0, len(years), step)],
                       rotation=90, fontsize=tick_fontsize - 0.2)
    ax.set_yticks(np.arange(C.shape[0]))
    ax.set_yticklabels(C.index.astype(str), fontsize=tick_fontsize)
    if meta is not None and "continent" in meta.columns:
        cont = meta.set_index("key")["continent"].astype(str)
        cats = [c for c in sorted(set(cont)) if c and c.lower() != "nan"]
        palette = plt.get_cmap("tab10")
        colors = {c: palette(i % 10) for i, c in enumerate(cats)}
        for tick, key in zip(ax.get_yticklabels(), C.index.astype(str)):
            tick.set_color(colors.get(str(cont.get(key)), "#333333"))
    ax.set_xlabel("collection year", fontsize=8)
    if highlight:                      # 被单独建模的国家：加粗标签（不遮挡热图 cell）
        hl = {str(h) for h in highlight}
        for tick, key in zip(ax.get_yticklabels(), C.index.astype(str)):
            if key in hl:
                tick.set_fontweight("bold")
    ax.set_title(title if title is not None else
                 ("Sampling year distribution per country\n"
                  "colour = log10(1 + #genomes); white = none that year"), fontsize=9)
    cb = ax.get_figure().colorbar(im, ax=ax, pad=0.015, fraction=colorbar_fraction)
    cb.set_label("log10(1 + #genomes)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    return C


def panel_country_year_span(ax, summary: pd.DataFrame, order: Sequence[str],
                            meta: Optional[pd.DataFrame] = None,
                            highlight: Optional[Sequence[str]] = None,
                            title: Optional[str] = None,
                            show_counts: bool = True):
    """
    各国采样窗口的 box-and-whisker 式摘要：细线 = min–max，粗条 = IQR，点 = 中位年。

    点大小 = 该国基因组数；颜色 = 大洲（与其他面板一致）。
    """
    s = summary.set_index("key")
    keys = [k for k in order if k in s.index]
    hl = {str(h) for h in (highlight or [])}

    colors = {}
    if meta is not None and "continent" in meta.columns:
        cont = meta.set_index("key")["continent"].astype(str)
        cats = [c for c in sorted(set(cont)) if c and c.lower() != "nan"]
        palette = plt.get_cmap("tab10")
        colors = {c: palette(i % 10) for i, c in enumerate(cats)}
    else:
        cont = pd.Series(dtype=str)

    for i, key in enumerate(keys):
        r = s.loc[key]
        if not np.isfinite(r.get("median", np.nan)):
            ax.text(0.02, i, "no collection year", fontsize=6.0, va="center",
                    color="#999999", transform=ax.get_yaxis_transform())
            continue
        col = colors.get(str(cont.get(key, "")), "#4C72B0")
        lo = r["p05"] if np.isfinite(r.get("p05", np.nan)) else r["year_min"]
        hi = r["p95"] if np.isfinite(r.get("p95", np.nan)) else r["year_max"]
        ax.plot([lo, hi], [i, i], color=col, lw=0.9, alpha=0.55,
                solid_capstyle="round", zorder=2)
        ax.plot([r["q25"], r["q75"]], [i, i], color=col, lw=4.2, alpha=0.85,
                solid_capstyle="butt", zorder=3)
        ax.scatter([r["median"]], [i], s=float(np.clip(r["n_samples"] / 6.0, 10, 130)),
                   facecolor="white", edgecolor=col, linewidths=1.1, zorder=4)
        note = "n=%d" % int(r["n_samples"])
        if float(r.get("undated_fraction", 0.0)) > 0.1:
            note += " (%.0f%% undated)" % (100.0 * float(r["undated_fraction"]))
        if show_counts:
            ax.text(hi + 0.6, i, note, fontsize=5.6, va="center", color="#555555")
        if key in hl:
            ax.axhspan(i - 0.45, i + 0.45, color="#FFF3C4", zorder=0)

    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels(keys, fontsize=6.6)
    for tick, key in zip(ax.get_yticklabels(), keys):
        tick.set_color(colors.get(str(cont.get(key, "")), "#333333"))
    ax.set_ylim(-0.7, len(keys) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("collection year", fontsize=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.tick_params(axis="x", labelsize=6.6)
    ax.grid(axis="x", color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title(title if title is not None else
                 ("Sampling window per country\n"
                  "line = 5\u201395th pct, bar = IQR, dot = median (size = #genomes); "
                  "shaded = modelled individually"), fontsize=9)


def make_country_year_figure(samples_path: str, meta: pd.DataFrame,
                             highlight: Optional[Sequence[str]] = None,
                             save_prefix: Optional[str] = None):
    """
    国家采样时间分布的双面板图：左 = `国家 × 年份` 计数热图，右 = 采样窗口摘要。

    只统计进入分层分析的国家（`meta["key"]`），行序统一按中位采样年排序，
    因此两个面板可以逐行对读。返回 `(fig, counts, summary)`。
    """
    register_kp_colors()
    keys = meta["key"].astype(str).tolist()
    counts, summary = country_year_table(samples_path, keys=keys)
    order = summary["key"].astype(str).tolist()          # 按中位年份升序

    n = max(len(order), 1)
    fig = plt.figure(figsize=(13.0, 0.30 * n + 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.34)
    ax_h = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    panel_country_year_heatmap(ax_h, counts, order, meta=meta)
    panel_country_year_span(ax_s, summary, order, meta=meta, highlight=highlight)
    pf.panel_label(ax_h, "A", dx=-0.13, dy=1.06)
    pf.panel_label(ax_s, "B", dx=-0.14, dy=1.06)
    if save_prefix:
        for ext in ("png", "pdf"):
            fig.savefig("%s.%s" % (save_prefix, ext), dpi=330, bbox_inches="tight")
    return fig, counts, summary


def group_structure_test(scores: pd.DataFrame, labels: pd.Series,
                         n_perm: int = 5000, seed: int = 0,
                         min_group_size: int = 2) -> Dict[str, object]:
    """
    PC 空间中"某个标签是否解释了组间差异"的置换检验（PERMANOVA 式 pseudo-F）。

    F = (SSb / (k-1)) / (SSw / (n-k))，SS 用欧氏平方距离；p 值由标签置换给出。
    用途：判断国家特异动力学在 PCA 空间里是否**按大区成簇** —— 若显著，说明
    地理结构不仅是采样噪声；若不显著，则各国差异更像独立抖动。
    """
    lab = labels.reindex(scores.index).astype(str)
    ok = lab.notna() & (lab != "") & (lab.str.lower() != "nan")
    X = scores.loc[ok].values.astype(float)
    y = lab[ok].values
    vc = pd.Series(y).value_counts()
    keep = set(vc[vc >= min_group_size].index)
    m = np.array([v in keep for v in y])
    X, y = X[m], y[m]
    n, k = len(y), len(set(y))
    if n <= k or k < 2:
        return {"n": int(n), "k": int(k), "F": np.nan, "p_value": np.nan,
                "r2": np.nan, "note": "组数或样本数不足"}

    def ss(Xv: np.ndarray, yv: np.ndarray) -> Tuple[float, float]:
        gm = Xv.mean(axis=0)
        sst = float(((Xv - gm) ** 2).sum())
        ssw = 0.0
        for g in set(yv):
            Z = Xv[yv == g]
            ssw += float(((Z - Z.mean(axis=0)) ** 2).sum())
        return sst - ssw, ssw

    ssb, ssw = ss(X, y)
    f_obs = (ssb / (k - 1)) / (ssw / (n - k)) if ssw > 0 else np.inf
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(int(n_perm)):
        yp = rng.permutation(y)
        b, w = ss(X, yp)
        f = (b / (k - 1)) / (w / (n - k)) if w > 0 else np.inf
        if f >= f_obs:
            ge += 1
    return {"n": int(n), "k": int(k), "F": float(f_obs),
            "p_value": float((ge + 1) / (n_perm + 1)),
            "r2": float(ssb / (ssb + ssw)) if (ssb + ssw) > 0 else np.nan,
            "n_perm": int(n_perm)}


def stratification_strength(bubbles: Dict[str, pd.DataFrame],
                            meta: pd.DataFrame,
                            label: str = "geography") -> Dict[str, object]:
    """
    一种分层方式"把获得动力学分得有多开"的标量刻画（可跨分层方式比较）。

    做法：对分层组的展平 bubble 求两两欧氏距离，报告
      * `mean_pairwise_distance` —— 组间平均差异；
      * `distance_vs_size_spearman` —— 该差异与样本量的相关（若强负相关，说明差异
        主要由小样本组的分布抖动驱动，而非真实的地理/谱系信号）；
      * `mean_distance_large_only` —— 只用样本量前 50% 的组重算，剥离样本量效应。
    """
    keys = [k for k in bubbles if k in set(meta["key"])]
    X = np.vstack([bubbles[k].values.ravel() for k in keys])
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))
    iu = np.triu_indices(len(keys), 1)
    size = meta.set_index("key").reindex(keys)["n_transitions"].values.astype(float)
    per_group = np.array([np.delete(D[i], i).mean() for i in range(len(keys))])
    big = size >= np.median(size)
    Db = D[np.ix_(big, big)]
    iub = np.triu_indices(int(big.sum()), 1)
    return {
        "label": label, "n_groups": len(keys),
        "mean_pairwise_distance": float(D[iu].mean()),
        "sd_pairwise_distance": float(D[iu].std()),
        "distance_vs_size_spearman": hc.spearman(per_group, size),
        "mean_distance_large_only": float(Db[iub].mean()) if big.sum() > 1 else np.nan,
        "n_groups_large": int(big.sum()),
    }


def order_deviation_matrix(bubbles: Dict[str, pd.DataFrame],
                           reference: pd.DataFrame,
                           feature_order: Sequence[str],
                           column_order: Optional[Sequence[str]] = None,
                           min_mass: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    `feature × group` 的"期望获得序位 − 全局期望序位"偏差矩阵（论文 Fig 3 的量化底图）。

    返回 `(deviation, absolute)`。某组内某 character 从未被观测到（`total_mass <= min_mass`）
    时置 NaN —— 这些 cell 在热图上留白，避免把"没数据"读成"序位居中"。
    """
    ref = st.expected_ordering(reference).set_index("feature")["expected_order"]
    cols = list(column_order) if column_order is not None else list(bubbles)
    dev, absolute = {}, {}
    for g in cols:
        if g not in bubbles:
            continue
        eo = st.expected_ordering(bubbles[g]).set_index("feature")
        v = eo["expected_order"].where(eo["total_mass"] > min_mass)
        absolute[g] = v.reindex(feature_order)
        dev[g] = (v - ref).reindex(feature_order)
    return pd.DataFrame(dev), pd.DataFrame(absolute)


def train_two_stage_on(transitions_path: str, base_dir: str,
                       python_exe: Optional[str] = None,
                       epochs: int = 60, d_model: int = 128, n_layers: int = 3,
                       n_heads: int = 4, batch_size: int = 32, dropout: float = 0.30,
                       contrastive_weight: float = 0.05, mask_prob_pos: float = 0.5,
                       mask_prob_neg: float = 0.1, seed: int = 0,
                       force: bool = False) -> Dict[str, object]:
    """
    在指定的 transitions.csv 上训练一次 Stage1+Stage2，产物写入 `base_dir/{stage1,stage2}`。

    这是 `ensure_lineage_two_stage_model`（按 ST 抽子集）与国家特异模型（读逐国转移集）
    共用的底座 —— 唯一差别只是训练数据的范围，因此二者与全局模型三方可比。
    """
    import subprocess
    import sys as _sys
    import time as _time

    exe = python_exe or _sys.executable
    stage1_dir, stage2_dir = os.path.join(base_dir, "stage1"), os.path.join(base_dir, "stage2")
    model_path = os.path.join(stage2_dir, "model.pt")
    if os.path.isfile(model_path) and not force:
        return {"status": "cached", "model_path": model_path,
                "transitions_path": transitions_path, "elapsed_sec": None}
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)
    common = ["--transitions", transitions_path, "--arch", "transformer",
              "--d-model", str(d_model), "--n-layers", str(n_layers), "--n-heads", str(n_heads),
              "--dropout", str(dropout), "--batch-size", str(batch_size),
              "--epochs", str(epochs), "--seed", str(seed), "--no-meta"]
    t0 = _time.time()
    subprocess.run([exe, os.path.join(hc.MODELS_DIR, "train_state_pretrain.py"),
                    "--outdir", stage1_dir, "--mask-prob-pos", str(mask_prob_pos),
                    "--mask-prob-neg", str(mask_prob_neg),
                    "--contrastive-weight", str(contrastive_weight)] + common,
                   check=True, cwd=hc.EVOAMR_ROOT)
    t1 = _time.time()
    subprocess.run([exe, os.path.join(hc.MODELS_DIR, "train_transition_two_stage.py"),
                    "--outdir", stage2_dir,
                    "--pretrained", os.path.join(stage1_dir, "stage1_model.pt")] + common,
                   check=True, cwd=hc.EVOAMR_ROOT)
    t2 = _time.time()
    return {"status": "trained", "model_path": model_path,
            "transitions_path": transitions_path,
            "stage1_sec": round(t1 - t0, 1), "stage2_sec": round(t2 - t1, 1),
            "elapsed_sec": round(t2 - t0, 1)}


def ensure_country_two_stage_model(spec: "st.DatasetSpec", strat_dir: str, key: str,
                                   safe_name: Optional[str] = None, **kw
                                   ) -> Dict[str, object]:
    """
    只用某个国家的转移训练一次 Stage1+Stage2 —— 这正是论文"逐国拟合"的对应物。

    输出目录：`<results_dir>/country_<safe_name>/{stage1,stage2}`。
    """
    safe = safe_name or str(key).replace(" ", "_")
    tr_path = os.path.join(strat_dir, safe, "transitions.csv")
    if not os.path.isfile(tr_path):
        raise FileNotFoundError("未找到该国转移集: %s" % tr_path)
    base = os.path.join(spec.results_dir, "country_%s" % safe)
    out = train_two_stage_on(tr_path, base, **kw)
    out["key"] = str(key)
    out["safe_name"] = safe
    return out


# --------------------------------------------------------------------------------------
# 6b. 地理分层的绘图
# --------------------------------------------------------------------------------------

def _annotate_no_overlap(ax, xs: Sequence[float], ys: Sequence[float],
                         labels: Sequence[str], fontsize: float = 6.4,
                         arrow: bool = True) -> None:
    """
    贪心避让的点标注：候选偏移按"右上 → 右下 → 左上 → …"依次尝试，
    取第一个与已放标注不冲突（且不出画布）的位置，必要时用细引线连回点。

    调用前必须先固定 `xlim/ylim`（否则偏移量的归一化尺度会算错）。
    国家质心在欧洲一带极密，不做避让会叠成一团。
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span_x, span_y = (x1 - x0), (y1 - y0)
    cands = [(0.018, 0.028), (0.018, -0.052), (-0.072, 0.028), (-0.072, -0.052),
             (0.018, 0.082), (0.018, -0.106), (-0.072, 0.082), (-0.072, -0.106),
             (0.052, 0.145), (-0.105, 0.145), (0.052, -0.160), (-0.105, -0.160)]
    placed: List[Tuple[float, float]] = []
    for px, py, name in zip(xs, ys, labels):
        best = None
        for fx, fy in cands:
            tx, ty = px + fx * span_x, py + fy * span_y
            if not (x0 <= tx <= x1 and y0 <= ty <= y1):
                continue
            if all(abs(tx - qx) / span_x > 0.072 or abs(ty - qy) / span_y > 0.050
                   for qx, qy in placed):
                best = (tx, ty)
                break
        if best is None:
            best = (px + cands[0][0] * span_x, py + cands[0][1] * span_y)
        placed.append(best)
        ax.annotate(str(name), xy=(px, py), xytext=best, textcoords="data",
                    fontsize=fontsize, zorder=6,
                    arrowprops=(dict(arrowstyle="-", lw=0.45, color="#999999",
                                     shrinkA=0.5, shrinkB=2.0) if arrow else None))


def panel_geo_map(ax, meta: pd.DataFrame, value_col: str = "mean_depth",
                  annotate_top: int = 12, label: str = "country"):
    """
    A：国家级散点地图（论文 Fig 1A 的可得版本）。

    点大小 = 该国的获得事件数，颜色 = `value_col`（默认源状态平均耐药深度）。
    坐标是**国家质心**，故不画国界、只画经纬网格；这足以传达"分析覆盖了哪些地区"。
    """
    d = meta.dropna(subset=["latitude", "longitude"]).copy()
    lon, lat = d["longitude"].values.astype(float), d["latitude"].values.astype(float)
    x0, x1 = lon.min() - 18, lon.max() + 18
    y0, y1 = lat.min() - 12, lat.max() + 14
    for x in range(-180, 181, 30):
        if x0 <= x <= x1:
            ax.axvline(x, color="#eeeeee", lw=0.6, zorder=0)
    for y in range(-60, 91, 20):
        if y0 <= y <= y1:
            ax.axhline(y, color="#eeeeee", lw=0.6, zorder=0)
    ax.axhline(0, color="#dddddd", lw=0.9, zorder=0)
    sizes = np.clip(d["n_gain_events"].values.astype(float) / 3.0, 18, 420)
    sc = ax.scatter(lon, lat, s=sizes, c=d[value_col].values.astype(float),
                    cmap="magma_r", alpha=0.88, edgecolors="white", linewidths=0.7, zorder=3)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    top = d.sort_values("n_gain_events", ascending=False).head(annotate_top)
    _annotate_no_overlap(ax, top["longitude"].values, top["latitude"].values,
                         top["key"].astype(str).values)
    ax.set_xlabel("longitude", fontsize=8)
    ax.set_ylabel("latitude", fontsize=8)
    ax.set_title("Geographic coverage of the %s-stratified analysis\n"
                 "point size = #gain events, colour = mean resistance depth" % label,
                 fontsize=9)
    cb = ax.get_figure().colorbar(sc, ax=ax, pad=0.015, fraction=0.026)
    cb.set_label("mean #characters in source state", fontsize=7)
    cb.ax.tick_params(labelsize=6)


def panel_geo_pca(ax, scores: pd.DataFrame, evr: np.ndarray, meta: pd.DataFrame,
                  color_col: str = "continent",
                  pcs: Tuple[str, str] = ("PC1", "PC2"), annotate_top: int = 10,
                  test: Optional[Dict[str, object]] = None):
    """
    B：国家特异动力学的 PCA，按大洲着色（论文 Fig 2Bi 的原版口径）。

    右下角给出 `group_structure_test` 的结果 —— 大洲标签是否真的解释了 PC 空间里的
    组间差异。这一步是必需的：不做检验就把"看起来成簇"写成地理结构，是最常见的过度解读。
    """
    m = meta.set_index("key").reindex(scores.index)
    labels = m[color_col].astype(str) if color_col in m.columns else pd.Series("", index=m.index)
    cats = [c for c in sorted(set(labels)) if c and c.lower() != "nan"]
    cmap = plt.get_cmap("tab10")
    cols = {c: cmap(i % 10) for i, c in enumerate(cats)}
    sizes = np.clip(m["n_gain_events"].values.astype(float) / 3.0, 16, 320)
    for c in cats:
        sel = (labels == c).values
        ax.scatter(scores.loc[sel, pcs[0]], scores.loc[sel, pcs[1]], s=sizes[sel],
                   color=cols[c], alpha=0.85, edgecolors="white", linewidths=0.6, label=c)
    ax.set_xlim(*ax.get_xlim())      # 固化坐标范围，避让计算才有正确的归一化尺度
    ax.set_ylim(*ax.get_ylim())
    top = m["n_gain_events"].sort_values(ascending=False).head(annotate_top).index
    top = [g for g in top if g in scores.index]
    _annotate_no_overlap(ax, scores.loc[top, pcs[0]].values,
                         scores.loc[top, pcs[1]].values, [str(g) for g in top])
    i, j = int(pcs[0][2:]) - 1, int(pcs[1][2:]) - 1
    ax.set_xlabel("%s (%.0f%% var)" % (pcs[0], 100 * evr[i]), fontsize=8)
    ax.set_ylabel("%s (%.0f%% var)" % (pcs[1], 100 * evr[j]), fontsize=8)
    ax.set_title("Country-specific acquisition dynamics (PCA)\n"
                 "colour = continent, point size = #gain events", fontsize=9)
    ax.legend(fontsize=6.2, loc="best", frameon=True, ncol=2)
    if test:
        p = test.get("p_value")
        ax.text(0.99, 0.02,
                "continent structure: pseudo-F=%.2f, R$^2$=%.2f, p=%s"
                % (test.get("F", float("nan")), test.get("r2", float("nan")),
                   ("%.4f" % p) if isinstance(p, float) else str(p)),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3,
                bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#cccccc", lw=0.5))


def panel_order_heatmap(ax, dev: pd.DataFrame, short_map: Dict[str, str],
                        class_map: Dict[str, str], vmax: Optional[float] = None,
                        label: str = "country"):
    """
    C：`character × 国家` 的期望序位偏差热图（相对全局），论文 Fig 3 的底图。

    红 = 该国比全局更**晚**获得，蓝 = 更**早**；灰 cell = 该国从未观测到该 character。
    "未观测"必须与"偏差接近 0"在视觉上分开，否则会把缺数据读成"与全局一致"。
    列已按 PC1 排序，因此左右两侧分别是"分布锐利"与"分布弥散"的国家。
    """
    M = np.ma.masked_invalid(dev.values.astype(float))
    lim = vmax if vmax is not None else float(np.nanpercentile(np.abs(dev.values), 95))
    lim = max(lim, 1e-6)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim,
                   interpolation="nearest")
    ax.set_xticks(np.arange(dev.shape[1]))
    ax.set_xticklabels(dev.columns, rotation=90, fontsize=6.2)
    ax.set_yticks(np.arange(dev.shape[0]))
    ax.set_yticklabels([short_map.get(f, f) for f in dev.index], fontsize=6.8)
    for tick, f in zip(ax.get_yticklabels(), dev.index):
        tick.set_color(pf.CLASS_COLORS.get(class_map.get(str(f), "other"), "#333333"))
    ax.set_title("Deviation of expected acquisition order from the global order\n"
                 "(blue = earlier in this %s, red = later, grey = never observed)" % label,
                 fontsize=9)
    cb = ax.get_figure().colorbar(im, ax=ax, pad=0.015, fraction=0.025)
    cb.set_label("$\\Delta$ ordering step vs global", fontsize=7)
    cb.ax.tick_params(labelsize=6)


# Natural Earth 的国家名 -> 本项目 metadata 里的国家名
NE_NAME_ALIAS = {
    "United States of America": "USA",
    "United States": "USA",
    "United Kingdom": "UK",
    "Russian Federation": "Russia",
    "Viet Nam": "Vietnam",
    "Republic of Korea": "South Korea",
    "Dem. Rep. Congo": "Democratic Republic of the Congo",
    "Czechia": "Czech Republic",
    "Iran (Islamic Republic of)": "Iran",
    "Lao PDR": "Laos",
    "United Republic of Tanzania": "Tanzania",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "C\u00f4te d'Ivoire": "Cote d'Ivoire",
    "Ivory Coast": "Cote d'Ivoire",
}


def _plural(label: str) -> str:
    """把 'country' / 'region' 之类的分层名变成复数形式（仅用于图注）。"""
    text = str(label)
    if text.endswith("y") and not text.endswith(("ay", "ey", "oy", "uy")):
        return text[:-1] + "ies"
    return text if text.endswith("s") else text + "s"


def geo_map_projection(kind: str = "platecarree"):
    """
    返回世界地图用的 cartopy 投影；cartopy 不可用时返回 None（调用方退化为散点图）。

    默认 PlateCarree：本图 A 是通栏的扁轴，等距圆柱投影能把画面填满；
    需要更美观的等面积效果时可传 `kind="robinson"`。
    """
    try:
        import cartopy.crs as ccrs
    except Exception:
        return None
    return ccrs.Robinson() if str(kind).lower().startswith("rob") else ccrs.PlateCarree()


def panel_geo_map_world(ax, meta: pd.DataFrame, value_col: str = "mean_depth",
                        annotate_top: int = 12, label: str = "country"):
    """
    A：真正的世界地图版本（与 Figure A 的 `panel_overview_map` 同一套 cartopy 底图）。

    - 进入分层分析的国家：国界内浅色填充，突出"分析覆盖了哪些国家"；
    - 国家质心气泡：面积 = 该国获得事件数，颜色 = `value_col`（默认源状态平均耐药深度）；
    - 未纳入的国家填浅灰，与"纳入但事件少"在视觉上区分开。

    `ax` 必须是带 cartopy 投影的轴（见 `geo_map_projection()`）；数据经纬度按
    PlateCarree 变换后叠加。
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    import matplotlib.patheffects as mpatheffects

    register_kp_colors()
    d = meta.dropna(subset=["latitude", "longitude"]).copy()
    included = set(d["key"].astype(str))

    # 裁掉南极与极区空白：整行是一条很扁的轴，全球范围会把地图压得很小
    ax.set_extent([-168, 179, -56, 80], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="white",
                   edgecolor="none", zorder=0)

    shp = shpreader.natural_earth(resolution="110m", category="cultural",
                                  name="admin_0_countries")
    for record in shpreader.Reader(shp).records():
        attrs = record.attributes
        ne_name = attrs.get("ADMIN") or attrs.get("NAME_LONG") or attrs.get("NAME") or ""
        data_name = NE_NAME_ALIAS.get(ne_name, ne_name)
        face = "#DCE7F1" if data_name in included else "#F4F4F4"
        ax.add_geometries([record.geometry], crs=ccrs.PlateCarree(),
                          facecolor=face, edgecolor="#BFBFBF", linewidth=0.28, zorder=1)

    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#9E9E9E",
                   linewidth=0.35, zorder=2)
    ax.gridlines(draw_labels=False, linewidth=0.35, color="#DDDDDD",
                 alpha=0.6, zorder=0)

    sizes = np.clip(d["n_gain_events"].values.astype(float) / 3.0, 18, 420)
    sc = ax.scatter(d["longitude"].values.astype(float),
                    d["latitude"].values.astype(float),
                    s=sizes, c=d[value_col].values.astype(float),
                    cmap="magma_r", alpha=0.9, edgecolors="white", linewidths=0.7,
                    zorder=4, transform=ccrs.PlateCarree())

    top = d.sort_values("n_gain_events", ascending=False).head(annotate_top)
    for _, r in top.iterrows():
        ax.text(float(r["longitude"]), float(r["latitude"]) + 4.0, str(r["key"]),
                fontsize=6.2, ha="center", va="bottom", color="#222222", zorder=6,
                transform=ccrs.PlateCarree(),
                path_effects=[mpatheffects.withStroke(linewidth=1.6,
                                                      foreground="white")])

    ax.set_title("Geographic coverage of the %s-stratified analysis\n"
                 "shaded = %s included in the analysis; bubble area = #gain events, "
                 "colour = mean resistance depth" % (label, _plural(label)), fontsize=9)

    cb = ax.get_figure().colorbar(sc, ax=ax, pad=0.015, fraction=0.023)
    cb.set_label("mean #characters in source state", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    ref = list(np.percentile(d["n_gain_events"].astype(float), [25, 50, 95]))
    handles = [plt.scatter([], [], s=float(np.clip(v / 3.0, 18, 420)),
                           facecolor="#999999", edgecolor="white", linewidths=0.7,
                           label="%d gains" % int(round(v))) for v in ref]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0),
              fontsize=6.0, frameon=False, labelspacing=1.0, borderpad=0.2,
              handletextpad=0.8, title="#gain events", title_fontsize=6.2)


def _cluster_order(mat: np.ndarray) -> np.ndarray:
    """对矩阵的行做层次聚类并返回叶序；退化情形下返回原序。"""
    if mat.shape[0] < 3:
        return np.arange(mat.shape[0])
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist

        x = np.where(np.isfinite(mat), mat, 0.0)
        dist = pdist(x, metric="correlation")
        if not np.all(np.isfinite(dist)):
            dist = pdist(x, metric="euclidean")
        return np.asarray(leaves_list(linkage(dist, method="average")), dtype=int)
    except Exception:
        return np.arange(mat.shape[0])


def _adjusted_rand_index(labels_a: Sequence[object], labels_b: Sequence[object]) -> float:
    """不依赖 sklearn 的 adjusted Rand index。"""
    from scipy.special import comb

    ct = pd.crosstab(pd.Series(list(labels_a)), pd.Series(list(labels_b))).values
    n = int(ct.sum())
    if n < 2:
        return float("nan")
    sum_cells = float(comb(ct, 2).sum())
    sum_rows = float(comb(ct.sum(axis=1), 2).sum())
    sum_cols = float(comb(ct.sum(axis=0), 2).sum())
    total = float(comb(n, 2))
    expected = sum_rows * sum_cols / total
    maximum = 0.5 * (sum_rows + sum_cols)
    if abs(maximum - expected) < 1e-12:
        return float("nan")
    return (sum_cells - expected) / (maximum - expected)


def deviation_cluster_vs_continent(dev: pd.DataFrame, meta: pd.DataFrame,
                                   k: Optional[int] = None,
                                   n_perm: int = 2000, seed: int = 0) -> Dict[str, object]:
    """
    量化"偏差谱型聚类"与"大洲"的一致程度 —— 回答\"列到底该不该按大洲分组\"。

    做法：对国家的偏差向量做层次聚类，切成 `k` 类（默认 k = 大洲数），与大洲标签算
    adjusted Rand index（ARI），并用打乱大洲标签的置换检验给出 p 值。

    ARI ≈ 0 表示聚类结构与大洲基本无关，此时把列按大洲强行分组会**制造**出
    数据并不支持的地理分块；ARI 明显 > 0 才说明按大洲分组是数据本身的结构。
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    cont = meta.set_index("key")["continent"].astype(str)
    cols = [c for c in dev.columns.astype(str) if c in cont.index and cont[c].lower() != "nan"]
    if len(cols) < 4:
        return {"n": len(cols), "ari": float("nan"), "p_value": float("nan")}

    x = dev[cols].values.astype(float).T
    x = np.where(np.isfinite(x), x, 0.0)
    dist = pdist(x, metric="correlation")
    if not np.all(np.isfinite(dist)):
        dist = pdist(x, metric="euclidean")
    Z = linkage(dist, method="average")

    labels_cont = cont.reindex(cols).values
    k_use = int(k or len({str(v) for v in labels_cont}))
    labels_clu = fcluster(Z, t=k_use, criterion="maxclust")
    ari = _adjusted_rand_index(labels_clu, labels_cont)

    rng = np.random.default_rng(seed)
    null = np.array([_adjusted_rand_index(labels_clu, rng.permutation(labels_cont))
                     for _ in range(int(n_perm))], dtype=float)
    p = float((1.0 + np.sum(null >= ari)) / (1.0 + np.sum(np.isfinite(null))))

    return {"n": len(cols), "k": k_use, "ari": float(ari), "p_value": p,
            "n_perm": int(n_perm),
            "cluster_sizes": np.bincount(labels_clu)[1:].tolist(),
            "clusters": pd.Series(labels_clu, index=cols).to_dict()}


def panel_order_deviation_map(ax, dev: pd.DataFrame, short_map: Dict[str, str],
                              class_map: Dict[str, str],
                              meta: Optional[pd.DataFrame] = None,
                              corr: Optional[pd.DataFrame] = None,
                              vmax: Optional[float] = None,
                              label: str = "country",
                              cluster_columns: bool = True,
                              column_mode: Optional[str] = None,
                              annotate_cluster_test: bool = True):
    """
    D 的加强版：`character × 国家` 期望序位偏差热图 + 两条边缘摘要。

    相对旧版（单纯 imshow、列按 PC1 排）的改进：

    1. **列排序可选**（`column_mode`）：
       - `"cluster"`（默认）：按偏差谱型层次聚类的叶序，让"整体偏早 / 偏晚 / 混合"自己成块；
       - `"continent"`：先按大洲分组、组内再聚类，并用竖线分隔（便于按地理阅读，
         但要注意这是**人为施加**的分组，聚类结构本身未必支持）；
       - `"as_given"`：保持传入顺序（即 notebook 里的 PC1 升序）。
    2. **列上方大洲色条**：直接回答"聚类出来的国家组是否与地理一致"；
       `annotate_cluster_test=True` 时在图上标出聚类 vs 大洲的 ARI 与置换 p 值，
       避免只靠肉眼判断地理分块。
    3. **右侧边缘条形**：每个 character 跨国的平均 |Δ|（分歧幅度）+ 覆盖国家数；
    4. 分歧 character（与 PC2/PC3 相关，来自 `corr`）在 y 标签上加 ▲，与 C 图呼应。

    传入的 `ax` 只用来占位：函数会在其 subplotspec 上重建子网格。
    """
    register_kp_colors()
    fig = ax.get_figure()
    spec = ax.get_subplotspec()
    ax.remove()

    sub = spec.subgridspec(2, 2, height_ratios=[0.055, 1.0], width_ratios=[1.0, 0.14],
                           hspace=0.03, wspace=0.02)
    ax_strip = fig.add_subplot(sub[0, 0])
    ax_hm = fig.add_subplot(sub[1, 0])
    ax_bar = fig.add_subplot(sub[1, 1], sharey=ax_hm)

    mode = str(column_mode or ("cluster" if cluster_columns else "as_given")).lower()
    cont_series = None
    if meta is not None and "continent" in meta.columns:
        cont_series = meta.set_index("key")["continent"].astype(str)

    D = dev.copy()
    blocks: List[Tuple[str, int]] = []          # (大洲名, 该块列数)，仅 continent 模式使用

    if mode == "cluster" and D.shape[1] >= 3:
        D = D.iloc[:, _cluster_order(D.values.astype(float).T)]
        order_note = "columns clustered by deviation profile"
    elif mode == "continent" and cont_series is not None:
        cats = [c for c in sorted(set(cont_series)) if c and c.lower() != "nan"]
        ordered: List[str] = []
        for c in cats:
            members = [k for k in D.columns.astype(str) if cont_series.get(k) == c]
            if len(members) >= 3:                      # 组内再按谱型聚类，块内也可读
                sub_d = D[members]
                members = [members[i] for i in _cluster_order(sub_d.values.astype(float).T)]
            if members:
                blocks.append((c, len(members)))
                ordered += members
        rest = [k for k in D.columns.astype(str) if k not in set(ordered)]
        if rest:
            blocks.append(("unknown", len(rest)))
            ordered += rest
        D = D[ordered]
        order_note = "columns grouped by continent, clustered within continent"
    else:
        order_note = "columns in the order given (PC1 ascending)"

    M = np.ma.masked_invalid(D.values.astype(float))
    lim = vmax if vmax is not None else float(np.nanpercentile(np.abs(D.values), 95))
    lim = max(lim, 1e-6)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#D9D9D9")
    im = ax_hm.imshow(M, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim,
                      interpolation="nearest")

    divergent = set()
    if corr is not None and "class" in corr.columns:
        divergent = set(corr.loc[corr["class"].astype(str).str.contains("divergent",
                                                                       case=False),
                                 "feature"].astype(str))

    ax_hm.set_xticks(np.arange(D.shape[1]))
    ax_hm.set_xticklabels(D.columns, rotation=90, fontsize=6.2)
    ax_hm.set_yticks(np.arange(D.shape[0]))
    ax_hm.set_yticklabels([short_map.get(f, f) + (" \u25b2" if str(f) in divergent else "")
                           for f in D.index], fontsize=6.8)
    for tick, f in zip(ax_hm.get_yticklabels(), D.index):
        tick.set_color(pf.CLASS_COLORS.get(class_map.get(str(f), "other"), "#333333"))
    ax_hm.set_xlabel("%s (%s)" % (_plural(label), order_note), fontsize=8)

    if blocks:                                   # 大洲分组模式：画块间分隔线
        pos = 0
        for _c, n_c in blocks[:-1]:
            pos += n_c
            ax_hm.axvline(pos - 0.5, color="#FFFFFF", lw=1.6)
            ax_hm.axvline(pos - 0.5, color="#555555", lw=0.5)

    # --- 列上方的大洲色条 -------------------------------------------------
    ax_strip.set_xlim(-0.5, D.shape[1] - 0.5)
    ax_strip.set_ylim(0, 1)
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax_strip.spines[side].set_visible(False)

    if meta is not None and "continent" in meta.columns:
        cont = meta.set_index("key")["continent"].astype(str)
        # 与 B 图共用同一套 tab10 大洲配色（cats 取全量并排序，保证两图颜色一致）
        cats = [c for c in sorted(set(cont)) if c and c.lower() != "nan"]
        palette = plt.get_cmap("tab10")
        colors = {c: palette(i % 10) for i, c in enumerate(cats)}
        shown = [c for c in cats if c in set(cont.reindex(D.columns.astype(str)).dropna())]
        for j, key in enumerate(D.columns.astype(str)):
            c = cont.get(key)
            ax_strip.add_patch(Rectangle((j - 0.5, 0.05), 1.0, 0.9,
                                         facecolor=colors.get(str(c), "#DDDDDD"),
                                         edgecolor="white", linewidth=0.3))
        ax_strip.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=colors[c],
                                           edgecolor="none", label=c) for c in shown],
                        loc="lower left", bbox_to_anchor=(0.0, 1.05), ncol=len(shown),
                        fontsize=6.2, frameon=False, handlelength=1.0,
                        columnspacing=1.0, borderpad=0.1)

        if blocks:                              # 大洲分组模式：在色条上直接写块名
            pos = 0
            for c, n_c in blocks:
                ax_strip.text(pos + n_c / 2.0 - 0.5, 0.5, c, fontsize=5.8,
                              ha="center", va="center", color="white")
                pos += n_c

        # 聚类结构与大洲是否真的吻合：给出 ARI + 置换 p 值，而不是让读者靠色条猜
        if annotate_cluster_test:
            try:
                test = deviation_cluster_vs_continent(D, meta)
                if np.isfinite(test.get("ari", np.nan)):
                    ax_strip.text(1.0, 1.15,
                                  "deviation-profile clusters vs continent: "
                                  "ARI = %.2f, permutation p = %.3f (k = %d, n = %d)"
                                  % (test["ari"], test["p_value"], test["k"], test["n"]),
                                  transform=ax_strip.transAxes, fontsize=6.4,
                                  ha="right", va="bottom", color="#555555")
            except Exception as exc:
                print("聚类 vs 大洲一致性检验失败：%s" % exc)

    # --- 右侧：每个 character 的跨国分歧幅度 ------------------------------
    mean_abs = np.nanmean(np.abs(D.values.astype(float)), axis=1)
    n_obs = np.isfinite(D.values.astype(float)).sum(axis=1)
    ypos = np.arange(D.shape[0])
    bar_colors = [pf.CLASS_COLORS.get(class_map.get(str(f), "other"), "#999999")
                  for f in D.index]
    ax_bar.barh(ypos, mean_abs, color=bar_colors, alpha=0.85, height=0.72)
    ax_bar.set_xlim(0, float(np.nanmax(mean_abs)) * 1.45 if np.isfinite(mean_abs).any() else 1)
    # 注意：ax_bar 与 ax_hm 共享 y 轴，imshow 已经是"上小下大"的方向，
    # 这里若再 invert_yaxis() 会连带把热图整体上下翻转。
    ax_bar.tick_params(axis="y", left=False, labelleft=False)
    ax_bar.tick_params(axis="x", labelsize=6)
    ax_bar.set_xlabel("mean |$\\Delta$|", fontsize=7)
    for side in ("top", "right", "left"):
        ax_bar.spines[side].set_visible(False)
    for y, v, k in zip(ypos, mean_abs, n_obs):
        if np.isfinite(v):
            ax_bar.text(v + float(np.nanmax(mean_abs)) * 0.04, y, "n=%d" % int(k),
                        fontsize=5.4, va="center", color="#555555")

    # 标题挂在色条轴上，避免和大洲图例挤在同一条水平带里
    ax_strip.set_title("Deviation of expected acquisition order from the global order\n"
                       "(blue = earlier in this %s, red = later, grey = never observed; "
                       "\u25b2 = divergent character; right bars = mean |$\\Delta$| across %s)"
                       % (label, _plural(label)), fontsize=9, pad=26)

    cax = ax_bar.inset_axes([1.55, 0.30, 0.16, 0.40])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("$\\Delta$ ordering step vs global", fontsize=6.6)
    cb.ax.tick_params(labelsize=6)
    return ax_hm


def make_geo_figure(meta: pd.DataFrame, scores: pd.DataFrame, evr: np.ndarray,
                    dev: pd.DataFrame, corr: pd.DataFrame,
                    short_map: Dict[str, str], class_map: Dict[str, str],
                    structure_test: Optional[Dict[str, object]] = None,
                    label: str = "country", suptitle: str = "",
                    save_prefix: Optional[str] = None,
                    world_map: bool = True, cluster_columns: bool = True,
                    heatmap_column_mode: Optional[str] = None,
                    sampling_summary: Optional[pd.DataFrame] = None,
                    sampling_counts: Optional[pd.DataFrame] = None,
                    sampling_panel: str = "span",
                    sampling_highlight: Optional[Sequence[str]] = None):
    """组装地理分层图（A 世界地图 / B PCA / C 一致 vs 分歧 / D 偏差热图）。

    `world_map=True` 时 A 用 cartopy 世界地图（与 Figure A 一致）；cartopy 缺失时
    自动退化为经纬度散点图。D 的列排序由 `heatmap_column_mode` 控制
    （`"cluster"` / `"continent"` / `"as_given"`，缺省时由 `cluster_columns` 决定）。

    传入 `sampling_summary`（`country_year_table()` 的第二个返回值）时，第一行拆成
    **Ai 世界地图 + Aii 采样时间**：读者在看"哪些国家进入分析"的同时就能看到
    "这些国家的样本采于何时"，这是解读国家间顺序差异的必要背景。Aii 有两种画法：

    - `sampling_panel="span"`：采样窗口摘要（5–95 分位 / IQR / 中位年），最省版面；
    - `sampling_panel="heatmap"`：`国家 × 年份` 计数热图（需同时传 `sampling_counts`），
      能看出逐年的采样脉冲与断档，代价是列很密。
    """
    register_kp_colors()
    fig = plt.figure(figsize=(12.4, 14.6))
    # A 行给足高度：等距圆柱投影的世界地图是宽高比 ~2.5 的固定比例，
    # 行太矮会让地图两侧留下大片空白。
    gs = fig.add_gridspec(3, 2, height_ratios=[1.55, 1.45, 1.55], hspace=0.42, wspace=0.30)

    mode_s = str(sampling_panel or "span").lower()
    if mode_s.startswith("heat"):
        with_sampling = sampling_counts is not None and not sampling_counts.empty
    else:
        with_sampling = sampling_summary is not None and not sampling_summary.empty
    if with_sampling:
        ratio = [1.75, 1.0] if mode_s.startswith("heat") else [2.0, 1.0]
        gs_top = gs[0, :].subgridspec(1, 2, width_ratios=ratio, wspace=0.26)
        slot_map, slot_year = gs_top[0, 0], gs_top[0, 1]
    else:
        slot_map, slot_year = gs[0, :], None

    proj = geo_map_projection() if world_map else None
    if proj is not None:
        ax_a = fig.add_subplot(slot_map, projection=proj)
        try:
            panel_geo_map_world(ax_a, meta, label=label)
        except Exception as exc:                      # 底图数据缺失时不要让整张图失败
            print("世界地图绘制失败，退化为经纬度散点图：%s" % exc)
            ax_a.remove()
            ax_a = fig.add_subplot(slot_map)
            panel_geo_map(ax_a, meta, label=label)
    else:
        ax_a = fig.add_subplot(slot_map)
        panel_geo_map(ax_a, meta, label=label)
    pf.panel_label(ax_a, "Ai" if with_sampling else "A", dx=-0.045, dy=1.22)

    if with_sampling:
        ax_a2 = fig.add_subplot(slot_year)
        if sampling_summary is not None and not sampling_summary.empty:
            order = sampling_summary.sort_values("median")["key"].astype(str).tolist()
        else:
            order = sampling_counts.index.astype(str).tolist()
        if mode_s.startswith("heat"):
            panel_country_year_heatmap(
                ax_a2, sampling_counts, order, meta=meta,
                highlight=sampling_highlight, colorbar_fraction=0.042,
                tick_fontsize=6.0,
                title=("Sampling year distribution per %s\n"
                       "colour = log10(1 + #genomes); bold = modelled individually"
                       % label))
        else:
            panel_country_year_span(
                ax_a2, sampling_summary, order, meta=meta,
                highlight=sampling_highlight,
                title=("Sampling window per %s\n"
                       "line = 5\u201395th pct, bar = IQR, dot = median" % label))
        pf.panel_label(ax_a2, "Aii", dx=-0.34, dy=1.10)

    ax_b = fig.add_subplot(gs[1, 0])
    panel_geo_pca(ax_b, scores, evr, meta, test=structure_test)
    pf.panel_label(ax_b, "B")

    ax_d = fig.add_subplot(gs[1, 1])
    panel_d_consistency(ax_d, corr, short_map, class_map)
    ax_d.set_xlabel("expected acquisition order across %s (min \u2013 median \u2013 max)" % label,
                    fontsize=8)
    ax_d.set_title("Globally consistent vs divergent characters across %s\n"
                   "(\u25b2 = divergent: covaries with PC2/PC3 rather than PC1)" % label,
                   fontsize=9)
    pf.panel_label(ax_d, "C")

    ax_c = fig.add_subplot(gs[2, :])
    ax_c_main = panel_order_deviation_map(ax_c, dev, short_map, class_map,
                                          meta=meta, corr=corr, label=label,
                                          cluster_columns=cluster_columns,
                                          column_mode=heatmap_column_mode)
    pf.panel_label(ax_c_main, "D", dx=-0.055, dy=1.14)

    if suptitle:
        fig.suptitle(suptitle, y=0.995, fontsize=11.5, fontweight="bold")
    if save_prefix:
        for ext in ("png", "pdf"):
            fig.savefig(f"{save_prefix}.{ext}", dpi=350, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------------------
# 7. 汇总
# --------------------------------------------------------------------------------------

def write_summary(path_prefix: str, payload: Dict[str, object]) -> Tuple[str, str]:
    """写出 figure_kp_summary.json / .md。"""
    jpath, mpath = path_prefix + ".json", path_prefix + ".md"
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)

    def fmt(v, nd=4):
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) and v == v else str(v)

    d = payload.get("dataset", {})
    lines = ["# 肺炎克雷伯菌主图（Figure Y）核心数字", "",
             "- 数据：%s 条转移、L=%s、%s 个谱系分组、multi-gain 占比 %s"
             % (d.get("n_transitions"), d.get("L"), d.get("n_groups"),
                d.get("multi_gain_fraction")), ""]
    pred = payload.get("prediction", {})
    lines += ["## Panel E — 下一步预测", "",
              "| method | mean logL | Top-1 | Top-3 | Top-5 | MRR | AUC |",
              "|---|---|---|---|---|---|---|"]
    for m, v in (pred.get("overall") or {}).items():
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            m, fmt(v.get("mean_logl")), fmt(v.get("top1")), fmt(v.get("top3")),
            fmt(v.get("top5")), fmt(v.get("mrr")), fmt(v.get("auc"))))
    for k, v in (pred.get("tests") or {}).items():
        lines.append("")
        lines.append("- **%s**: Δ = %s, p = %s (%s)" % (
            k, fmt(v.get("delta_mean_logl")), fmt(v.get("p_value")), v.get("verdict")))
    if pred.get("hypertraps_status"):
        lines += ["", "> **可扩展性**：%s" % pred["hypertraps_status"]]

    bim = payload.get("bimodal") or []
    lines += ["", "## Panel B — 双峰 character（存在不同演化通路）", "",
              "| character | peaks at step | bimodality coef | valley ratio |",
              "|---|---|---|---|"]
    for r in bim:
        lines.append("| %s | %s | %s | %s |" % (r.get("short", r.get("feature")),
                                               r.get("peak_steps"),
                                               fmt(r.get("bimodality_coef"), 3),
                                               fmt(r.get("valley_ratio"), 3)))

    pca = payload.get("pca", {})
    lines += ["", "## Panel C/D — 谱系间变异", "",
              "- 纳入谱系数：%s（获得事件数 ≥ %s）" % (pca.get("n_groups"), pca.get("min_events")),
              "- 解释方差：%s" % ", ".join("PC%d %.1f%%" % (i + 1, 100 * v)
                                        for i, v in enumerate(pca.get("explained_variance") or [])),
              "", "| class | characters |", "|---|---|"]
    for cls, feats in (payload.get("consistency") or {}).items():
        lines.append("| %s | %s |" % (cls, ", ".join(feats)))

    geo = payload.get("geography") or {}
    if geo:
        lines += ["", "## 地理分层（按国家，论文原版口径）", "",
                  "- 纳入国家数：%s（逐国独立重建，转移合计 %s）"
                  % (geo.get("n_countries"), geo.get("n_transitions_sum")),
                  "- 解释方差：%s" % ", ".join(
                      "PC%d %.1f%%" % (i + 1, 100 * v)
                      for i, v in enumerate(geo.get("explained_variance") or []))]
        for nm, key in (("大洲", "continent_structure_test"),
                        ("大区", "region_structure_test")):
            t = geo.get(key) or {}
            if t:
                lines.append("- %s结构（PC 空间置换检验）：pseudo-F = %s, R2 = %s, p = %s"
                             % (nm, fmt(t.get("F"), 3), fmt(t.get("r2"), 3),
                                fmt(t.get("p_value"), 4)))
        cmp_rows = geo.get("stratification_comparison") or []
        if cmp_rows:
            lines += ["", "### 分层强度对照（真实 / 打乱 / 谱系）", "",
                      "| setting | #groups | mean pairwise dist | continent R2 | "
                      "continent p | #divergent |", "|---|---|---|---|---|---|"]
            for r in cmp_rows:
                lines.append("| %s | %s | %s | %s | %s | %s |" % (
                    r.get("setting"), r.get("n_groups"),
                    fmt(r.get("mean_pairwise_distance"), 3), fmt(r.get("continent_R2"), 3),
                    fmt(r.get("continent_p"), 4), r.get("n_divergent")))
        lines += ["", "| class (across countries) | characters |", "|---|---|"]
        for cls, feats in (geo.get("consistency") or {}).items():
            lines.append("| %s | %s |" % (cls, ", ".join(feats)))
        gsd = geo.get("geo_specific_divergence")
        if gsd is not None:
            lines += ["", "- **地理特异分歧**（跨国分歧但跨 ST 谱系一致 → 差异由地区而非克隆背景驱动）：%s"
                      % (", ".join(gsd) or "（无）")]
        cc = geo.get("country_case") or {}
        if cc:
            lines += ["", "### 代表国 %s 的三方对照（%s 条转移）"
                      % (cc.get("country"), cc.get("n_transitions")), "",
                      "| source | Spearman vs 该国经验 | MAE |", "|---|---|---|"]
            for r in (cc.get("fit_vs_country_empirical") or []):
                lines.append("| %s | %s | %s |" % (
                    r.get("source"), fmt(r.get("spearman_vs_lineage_empirical")),
                    fmt(r.get("mae_vs_lineage_empirical"))))
        rr = geo.get("region_robustness") or {}
        if rr:
            lines += ["", "- 大区层级稳健性：%s 个大区，国家/大区判定一致比例 %s；"
                      "两级都判分歧的 character：%s"
                      % (rr.get("n_regions"), fmt(rr.get("agreement_fraction"), 3),
                         ", ".join(rr.get("divergent_in_both") or []) or "（无）")]

    for note in (payload.get("caveats") or []):
        lines += ["", "> " + str(note)]
    with open(mpath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return jpath, mpath
