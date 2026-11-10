#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_figure_tb.py —— TB 主图（5 panel）的计算与绘图工具库

本模块只服务于论文正文 Figure "Two-stage model reconstructs clinically meaningful
resistance evolution in M. tuberculosis"，回答一个问题：

    Two-stage 是否既能准确预测下一步耐药事件，
    又能比 HyperTraPS 更好地恢复具有生物学意义的 TB 耐药演化结构？

Panel 结构（证据链）:
    A  Next-step prediction benchmark        —— 能不能预测下一步？（parity + 成本/可扩展性）
    B  Empirical acquisition hierarchy       —— 真实数据中的演化结构是什么？（数据参照系）
    C  Recovery of empirical ordering        —— 谁恢复得更好？（含 bootstrap CI）
    D  Representative evolutionary pathways  —— 事件如何组成路径？（drug-class 折叠 alluvial）
    E  Conditional evolutionary dependencies —— 为什么不同状态走向不同后续事件？
                                               （模型 log2 shift + 经验 log2 OR 双证据）

相对 evoamr_story / hypertraps_compare 新增的能力：
    1. `dependency_table`         : Panel E 的经验 odds ratio + 支持度 + Fisher 检验（去循环论证）
    2. `bootstrap_order_recovery` : Panel C 的 Δspearman / ΔMAE 的 cluster-bootstrap 置信区间
    3. `truncation_sensitivity`   : Panel C 预处理（截断点）敏感性，守 Supplementary
    4. `class_trajectory_flows`   : Panel D 的 drug-class 折叠轨迹（深度 4~5 的 alluvial）
    5. `paired_fold_test`         : Panel A 的逐折配对检验（Top-1 等非 logL 指标）

绘图文字全部英文（避免中文字体缺失），叙述性中文留在 notebook 的 markdown。
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath

import evoamr_story as st
import hypertraps_compare as hc

# --------------------------------------------------------------------------------------
# 0. 显示口径
# --------------------------------------------------------------------------------------

METHOD_LABELS = {
    "TS_pretrained": "Two-stage",
    "TS_random_init": "Two-stage\n(random init)",
    "HT_second-order": "HyperTraPS",
    "B1_prevalence": "Prevalence",
    "B0_uniform": "Uniform",
}
# 主角高亮，baseline 淡化
METHOD_COLORS = {
    "TS_pretrained": "#1f6fb4",
    "HT_second-order": "#c0392b",
    "TS_random_init": "#9ec9e8",
    "B1_prevalence": "#c9c9c9",
    "B0_uniform": "#e0e0e0",
}
MAIN_METHODS = ("B0_uniform", "B1_prevalence", "TS_random_init", "HT_second-order", "TS_pretrained")

PANEL_LABEL_KW = dict(fontsize=13, fontweight="bold", va="top", ha="right")


def panel_label(ax, text: str, dx: float = -0.09, dy: float = 1.06):
    ax.text(dx, dy, text, transform=ax.transAxes, **PANEL_LABEL_KW)


# --------------------------------------------------------------------------------------
# 1. TransitionSet 的子集 / 重采样
# --------------------------------------------------------------------------------------

def subset_transitions(ts: st.TransitionSet, idx: Sequence[int]) -> st.TransitionSet:
    """按行索引取子集（允许重复索引，用于 bootstrap）。"""
    idx = np.asarray(list(idx), dtype=int)
    return st.TransitionSet(
        frame=ts.frame.iloc[idx].reset_index(drop=True),
        src=ts.src[idx],
        dest=ts.dest[idx],
        gained=[ts.gained[i] for i in idx],
        feature_names=list(ts.feature_names),
    )


def cluster_bootstrap_indices(ts: st.TransitionSet, rng: np.random.Generator,
                              by_group: bool = True) -> np.ndarray:
    """
    重采样索引。

    `by_group=True` 时做**整簇（谱系）重采样**：转移在同一 clade 内高度相关，
    逐条重采样会低估不确定度。无 group_id 时退化为逐条重采样。
    """
    if by_group and "group_id" in ts.frame.columns:
        groups = ts.frame["group_id"].values
        uniq = pd.unique(groups)
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        out: List[int] = []
        for g in picked:
            out.extend(np.flatnonzero(groups == g).tolist())
        if out:
            return np.asarray(out, dtype=int)
    return rng.integers(0, ts.n, ts.n)


# --------------------------------------------------------------------------------------
# 2. Panel C：经验次序的恢复程度 + 不确定度
# --------------------------------------------------------------------------------------

def _expected_order_series(bubble: pd.DataFrame) -> pd.Series:
    eo = st.expected_ordering(bubble).set_index("feature")["expected_order"]
    return eo.reindex(bubble.index)


def ordering_recovery(bubble_empirical: pd.DataFrame,
                      model_bubbles: Dict[str, pd.DataFrame],
                      max_step: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """
    在**共同截断区间**内比较每个模型与经验 bubble 的获得次序。

    截断是必需的：经验序位由源状态的耐药深度决定（TB 数据最多 ~8），
    而模型前向游走会一直走到 L=19，两者序位尺度不可比。
    """
    ms = int(max_step or hc.empirical_max_step(bubble_empirical))
    emp = hc.truncate_and_renormalize(bubble_empirical, ms)
    eo_emp = _expected_order_series(emp)
    out: Dict[str, Dict[str, float]] = {}
    for name, b in model_bubbles.items():
        mod = hc.truncate_and_renormalize(b, ms)
        eo_mod = _expected_order_series(mod).reindex(eo_emp.index)
        diff = (eo_mod - eo_emp).values.astype(float)
        js = [hc.js_divergence(emp.loc[f].values, mod.loc[f].values) for f in eo_emp.index]
        out[name] = {
            "max_step": ms,
            "spearman": hc.spearman(eo_emp.values.astype(float), eo_mod.values.astype(float)),
            "mae": float(np.nanmean(np.abs(diff))),
            "median_js": float(np.nanmedian(js)),
        }
    return out


def ordering_scatter_frame(bubble_empirical: pd.DataFrame,
                           model_bubbles: Dict[str, pd.DataFrame],
                           max_step: Optional[int] = None) -> pd.DataFrame:
    """Panel C 散点用的长表：feature × model 的 (empirical order, model order)。"""
    ms = int(max_step or hc.empirical_max_step(bubble_empirical))
    eo_emp = _expected_order_series(hc.truncate_and_renormalize(bubble_empirical, ms))
    rows = []
    for name, b in model_bubbles.items():
        eo_mod = _expected_order_series(hc.truncate_and_renormalize(b, ms))
        for f in eo_emp.index:
            rows.append({"feature": f, "model": name,
                         "empirical_order": float(eo_emp.get(f, np.nan)),
                         "model_order": float(eo_mod.get(f, np.nan))})
    df = pd.DataFrame(rows)
    df["residual"] = df["model_order"] - df["empirical_order"]
    return df


def bootstrap_order_recovery(ts: st.TransitionSet,
                             model_bubbles: Dict[str, pd.DataFrame],
                             max_step: Optional[int] = None,
                             n_boot: int = 500,
                             seed: int = 0,
                             by_group: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对**经验侧**做 cluster bootstrap（模型 bubble 固定），得到
    每个模型的 spearman / MAE 及模型间差值的置信区间。

    这样量化的是"经验参照系本身的抽样不确定度"——正是 Panel C 结论
    (ρ=0.911 vs 0.760) 需要防守的地方。返回 (draws, summary)。
    """
    full_emp = st.empirical_bubble_matrix(ts)
    ms = int(max_step or hc.empirical_max_step(full_emp))
    rng = np.random.default_rng(seed)
    names = list(model_bubbles.keys())
    draws: List[Dict[str, float]] = []
    for b in range(n_boot):
        idx = cluster_bootstrap_indices(ts, rng, by_group=by_group)
        emp_b = st.empirical_bubble_matrix(subset_transitions(ts, idx))
        rec = ordering_recovery(emp_b, model_bubbles, max_step=ms)
        row: Dict[str, float] = {"boot": b}
        for name in names:
            row[f"spearman::{name}"] = rec[name]["spearman"]
            row[f"mae::{name}"] = rec[name]["mae"]
        draws.append(row)
    D = pd.DataFrame(draws)

    point = ordering_recovery(full_emp, model_bubbles, max_step=ms)
    rows = []
    for name in names:
        for metric in ("spearman", "mae"):
            v = D[f"{metric}::{name}"].values.astype(float)
            v = v[np.isfinite(v)]
            rows.append({
                "quantity": metric, "model": name,
                "point_estimate": point[name]["spearman" if metric == "spearman" else "mae"],
                "boot_mean": float(np.mean(v)) if len(v) else np.nan,
                "ci_lo": float(np.percentile(v, 2.5)) if len(v) else np.nan,
                "ci_hi": float(np.percentile(v, 97.5)) if len(v) else np.nan,
                "n_boot_valid": int(len(v)),
            })
    # 成对差值（配对：同一个 bootstrap 样本内比较，方差更小）
    for metric in ("spearman", "mae"):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b_ = names[i], names[j]
                d = (D[f"{metric}::{a}"] - D[f"{metric}::{b_}"]).values.astype(float)
                d = d[np.isfinite(d)]
                if not len(d):
                    continue
                obs = (point[a]["spearman"] - point[b_]["spearman"]) if metric == "spearman" \
                    else (point[a]["mae"] - point[b_]["mae"])
                p = 2.0 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0)))
                rows.append({
                    "quantity": f"delta_{metric}", "model": f"{a} - {b_}",
                    "point_estimate": float(obs), "boot_mean": float(np.mean(d)),
                    "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5)),
                    "n_boot_valid": int(len(d)), "p_two_sided": float(min(1.0, p)),
                })
    S = pd.DataFrame(rows)
    S.attrs["max_step"] = ms
    S.attrs["by_group"] = bool(by_group)
    return D, S


def truncation_sensitivity(bubble_empirical: pd.DataFrame,
                           model_bubbles: Dict[str, pd.DataFrame],
                           max_steps: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """
    ρ / MAE 随截断点变化的敏感性（Supplementary）。
    主图只报一个截断点会被质疑 cherry-picking，这条曲线是最直接的防守。
    """
    L = bubble_empirical.shape[1]
    default_hi = int(hc.empirical_max_step(bubble_empirical))
    steps = list(max_steps) if max_steps is not None else list(range(3, L + 1))
    rows = []
    for ms in steps:
        rec = ordering_recovery(bubble_empirical, model_bubbles, max_step=int(ms))
        row = {"max_step": int(ms), "is_default": int(ms) == default_hi}
        for name, vals in rec.items():
            row[f"spearman::{name}"] = vals["spearman"]
            row[f"mae::{name}"] = vals["mae"]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 3. Panel E：条件依赖的经验验证
# --------------------------------------------------------------------------------------

def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """2×2 表的双侧 Fisher 精确检验 p 值（不依赖 scipy；n 至多几百，comb 足够）。"""
    a, b, c, d = int(a), int(b), int(c), int(d)
    n = a + b + c + d
    if n == 0:
        return float("nan")
    r1, c1 = a + b, a + c
    denom = math.comb(n, c1)
    if denom == 0 or r1 == 0 or r1 == n or c1 == 0 or c1 == n:
        return 1.0

    def pmf(x: int) -> float:
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / denom

    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    p_obs = pmf(a)
    p = sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-9))
    return float(min(1.0, p))


def empirical_dependency(ts: st.TransitionSet, context: str, target: str,
                         haldane: float = 0.5) -> Dict[str, float]:
    """
    直接从转移数据估计"背景 j 是否改变下一步获得 i 的概率"。

    只在 **target 尚未获得（at risk）** 的转移上统计，按源状态是否已含 context 分两组：
        a = ctx+ 且本步获得 target   b = ctx+ 且未获得
        c = ctx- 且本步获得 target   d = ctx- 且未获得
    返回条件获得率、log2 rate ratio、log2 odds ratio（Haldane 校正）与 Fisher p。
    与模型侧的 log2 p(i|{j})/p(i|∅) 互为独立证据。
    """
    names = list(ts.feature_names)
    if context not in names or target not in names:
        return {}
    ci, ti = names.index(context), names.index(target)
    at_risk = ts.src[:, ti] == 0
    ctx = ts.src[:, ci] == 1
    gained = np.zeros(ts.n, dtype=bool)
    for r, g in enumerate(ts.gained):
        if ti in g:
            gained[r] = True
    a = int(np.sum(at_risk & ctx & gained))
    b = int(np.sum(at_risk & ctx & ~gained))
    c = int(np.sum(at_risk & ~ctx & gained))
    d = int(np.sum(at_risk & ~ctx & ~gained))
    r1 = a / (a + b) if (a + b) > 0 else np.nan
    r0 = c / (c + d) if (c + d) > 0 else np.nan
    h = haldane
    or_ = ((a + h) * (d + h)) / ((b + h) * (c + h))
    rr = ((a + h) / (a + b + 2 * h)) / ((c + h) / (c + d + 2 * h))
    return {
        "context_j": context, "next_i": target,
        "n_at_risk_ctx_present": a + b, "n_at_risk_ctx_absent": c + d,
        "n_gain_ctx_present": a, "n_gain_ctx_absent": c,
        "rate_ctx_present": r1, "rate_ctx_absent": r0,
        "emp_log2_or": float(np.log2(or_)), "emp_log2_rr": float(np.log2(rr)),
        "fisher_p": fisher_exact_2x2(a, b, c, d),
    }


def dependency_table(ts: st.TransitionSet, M: pd.DataFrame,
                     n_pos: int = 4, n_neg: int = 2,
                     min_support: int = 10,
                     max_fisher_p: float = 0.05,
                     candidate_pool: int = 40) -> pd.DataFrame:
    """
    Panel E 的数据：模型 log2 shift + 经验 log2 OR + 支持度。

    选取原则（**按经验证据筛，不按模型效应量筛**）：
      1. 从模型交互矩阵取效应量最强的 `candidate_pool` 个正/负 pair 作为候选；
      2. 支持度：ctx+ / ctx- 两侧 at-risk 转移数均需 ≥ `min_support`；
      3. 方向一致：模型与经验符号相同；
      4. 经验显著：Fisher p < `max_fisher_p`（设为 1.0 可关闭）；
      5. 剩下的按 |模型效应量| 取前 n_pos / n_neg。
    `M` 为 `st.model_interaction_matrix` 的输出（行=next_i，列=context_j）。
    """
    cand = st.top_interactions(M, top_n=int(candidate_pool))
    if cand.empty:
        return cand
    recs = []
    for _, r in cand.iterrows():
        emp = empirical_dependency(ts, str(r["context_j"]), str(r["next_i"]))
        if not emp:
            continue
        emp.update({"model_log2_shift": float(r["log2_shift"]), "effect": str(r["effect"])})
        recs.append(emp)
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["support_ok"] = ((df["n_at_risk_ctx_present"] >= min_support)
                        & (df["n_at_risk_ctx_absent"] >= min_support))
    df["same_direction"] = np.sign(df["model_log2_shift"]) == np.sign(df["emp_log2_or"])
    df["emp_significant"] = df["fisher_p"] < float(max_fisher_p)
    df["abs_model"] = df["model_log2_shift"].abs()
    keep = df[df["support_ok"] & df["same_direction"] & df["emp_significant"]]
    pos = keep[keep["model_log2_shift"] > 0].nlargest(n_pos, "abs_model")
    neg = keep[keep["model_log2_shift"] < 0].nlargest(n_neg, "abs_model")
    chosen = set(pos.index) | set(neg.index)
    out = df.copy()
    out["selected"] = out.index.isin(chosen)
    out["pair"] = out["context_j"] + " \u2192 " + out["next_i"]
    return out.sort_values(["selected", "model_log2_shift"], ascending=[False, False]) \
              .reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 4. Panel D：drug-class 折叠的代表性演化轨迹
# --------------------------------------------------------------------------------------

CLASS_SHORT = {
    # 基因层（drug_class 为药物/类别名）
    "isoniazid": "INH",
    "rifampicin": "RIF",
    "pyrazinamide": "PZA",
    "ethambutol": "EMB",
    "streptomycin": "STR",
    "second_line_injectable": "INJ",
    "fluoroquinolone": "FQ",
    "thioamide": "ETH",
    # 药物表型层（tuberculosis_antibiotics10）：把具体药物归入同一临床类别，
    # 使两层图共用同一套配色（AMI/CAP -> INJ，MOX/OFL -> FQ，PRO -> ETH）
    "amikacin": "INJ",
    "capreomycin": "INJ",
    "kanamycin": "INJ",
    "moxifloxacin": "FQ",
    "ofloxacin": "FQ",
    "levofloxacin": "FQ",
    "prothionamide": "ETH",
    "ethionamide": "ETH",
}
COMPENSATORY_LABEL = "RIF-comp"
CLASS_ORDER = ["INH", "RIF", "STR", "EMB", "ETH", "RIF-comp", "PZA", "INJ", "FQ"]
CLASS_COLORS = {
    "INH": "#1b5e20", "RIF": "#1f6fb4", "STR": "#7fb3d5", "EMB": "#8e6c8a",
    "ETH": "#b58b3c", "RIF-comp": "#e65100", "PZA": "#7a5195", "INJ": "#9e6b4f",
    "FQ": "#b71c1c",
}


def feature_class_map(feature_names: Sequence[str],
                      hierarchy: Optional[pd.DataFrame]) -> Dict[str, str]:
    """
    特征 → 展示类别。药物类别为主，`mechanism == compensatory_mutation` 单列一类
    （rpoA/rpoC 不产生耐药，而是补偿 rpoB 的适应度代价，不能与 RIF 靶点混为一类）。
    """
    out: Dict[str, str] = {}
    if hierarchy is None or "feature_id" not in hierarchy.columns:
        return {f: "other" for f in feature_names}
    h = hierarchy.set_index("feature_id")
    for f in feature_names:
        if f not in h.index:
            out[f] = "other"
            continue
        row = h.loc[f]
        mech = str(row.get("mechanism", ""))
        dc = str(row.get("drug_class", ""))
        out[f] = COMPENSATORY_LABEL if mech == "compensatory_mutation" \
            else CLASS_SHORT.get(dc, dc or "other")
    return out


def feature_palette(feature_names: Sequence[str],
                    hierarchy: Optional[pd.DataFrame]) -> Dict[str, str]:
    """Panel B/C/D/E 共用配色：同一基因在所有 panel 里颜色一致。"""
    cmap = feature_class_map(feature_names, hierarchy)
    return {f: CLASS_COLORS.get(cmap[f], "#999999") for f in feature_names}


def class_trajectory_flows(scorer, feature_names: Sequence[str], class_map: Dict[str, str],
                           depth: int = 4, n_walks: int = 3000, seed: int = 3
                           ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    在 drug-class 折叠空间里做前向游走，得到 alluvial 所需的三张表。

    基因级 top pathway 在 L=19 时每条概率都 <0.1（路径空间过大），无法支撑
    "共同早期骨架之后发生分支" 的叙事；折叠到药物类别后信号才可读。
    返回 (comp, flows, paths)：
      comp  : step × class 的占比
      flows : step→step+1 的类别流量
      paths : 折叠后的完整路径频率（top 12）
    """
    L = len(feature_names)
    rng = np.random.default_rng(seed)
    seqs: List[Tuple[str, ...]] = []
    for _ in range(n_walks):
        state = np.zeros(L, dtype=np.int8)
        seq: List[str] = []
        for _step in range(depth):
            p = np.asarray(scorer(state), dtype=float)
            p[state == 1] = 0.0
            tot = p.sum()
            if tot <= 0:
                break
            pick = int(rng.choice(L, p=p / tot))
            state[pick] = 1
            seq.append(class_map.get(feature_names[pick], "other"))
        if len(seq) == depth:
            seqs.append(tuple(seq))
    n = max(1, len(seqs))

    classes = [c for c in CLASS_ORDER if any(c in s for s in seqs)]
    extra = sorted({c for s in seqs for c in s} - set(classes))
    classes = classes + extra

    cidx = {c: i for i, c in enumerate(classes)}
    acc = np.zeros((depth, len(classes)), dtype=float)
    for s in seqs:
        for k, c in enumerate(s):
            acc[k, cidx[c]] += 1.0 / n
    comp = pd.DataFrame(acc, index=[f"step{k+1}" for k in range(depth)], columns=classes)

    frows = []
    for k in range(depth - 1):
        cnt: Dict[Tuple[str, str], int] = {}
        for s in seqs:
            key = (s[k], s[k + 1])
            cnt[key] = cnt.get(key, 0) + 1
        for (a, b), v in cnt.items():
            frows.append({"step": k + 1, "src": a, "dst": b, "prob": v / n})
    flows = pd.DataFrame(frows).sort_values(["step", "prob"], ascending=[True, False])

    pcnt: Dict[Tuple[str, ...], int] = {}
    for s in seqs:
        pcnt[s] = pcnt.get(s, 0) + 1
    paths = (pd.DataFrame([{"path": " \u2192 ".join(k), "probability": v / n}
                           for k, v in pcnt.items()])
             .sort_values("probability", ascending=False).head(12).reset_index(drop=True))
    return comp, flows, paths


def _ribbon(ax, x0: float, x1: float, y0a: float, y0b: float, y1a: float, y1b: float,
            color: str, alpha: float = 0.45):
    xm = (x0 + x1) / 2.0
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a),
             (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def plot_class_alluvial(ax, comp: pd.DataFrame, flows: pd.DataFrame,
                        min_flow: float = 0.01, bar_w: float = 0.16,
                        gap: float = 0.012, title: str = ""):
    """类别级 alluvial（不依赖 plotly，纯 matplotlib）。"""
    steps = list(comp.index)
    classes = [c for c in comp.columns if comp[c].sum() > 0]
    slot: Dict[Tuple[int, str], Tuple[float, float]] = {}
    for si, s in enumerate(steps):
        present = [c for c in classes if comp.loc[s, c] > 1e-9]
        total = sum(comp.loc[s, c] for c in present)
        y = 0.0
        pad = gap * max(0, len(present) - 1)
        scale = (1.0 - pad) / max(total, 1e-12)
        for c in present:
            hgt = comp.loc[s, c] * scale
            slot[(si, c)] = (y, y + hgt)
            ax.add_patch(Rectangle((si - bar_w / 2, y), bar_w, hgt,
                                   facecolor=CLASS_COLORS.get(c, "#999999"),
                                   edgecolor="white", linewidth=0.6))
            if hgt > 0.045:
                ax.text(si, y + hgt / 2, c, ha="center", va="center", fontsize=6.5,
                        color="white", fontweight="bold")
            y += hgt + gap

    out_cur = {k: v[0] for k, v in slot.items()}
    in_cur = {k: v[0] for k, v in slot.items()}
    for si in range(len(steps) - 1):
        sub = flows[flows["step"] == si + 1].sort_values("prob", ascending=False)
        for _, r in sub.iterrows():
            if r["prob"] < min_flow:
                continue
            ka, kb = (si, r["src"]), (si + 1, r["dst"])
            if ka not in slot or kb not in slot:
                continue
            ha = (slot[ka][1] - slot[ka][0]) * (r["prob"] / max(comp.iat[si, comp.columns.get_loc(r["src"])], 1e-12))
            hb = (slot[kb][1] - slot[kb][0]) * (r["prob"] / max(comp.iat[si + 1, comp.columns.get_loc(r["dst"])], 1e-12))
            y0a, y1a = out_cur[ka], in_cur[kb]
            _ribbon(ax, si + bar_w / 2, si + 1 - bar_w / 2,
                    y0a, y0a + ha, y1a, y1a + hb,
                    CLASS_COLORS.get(r["src"], "#999999"))
            out_cur[ka] = y0a + ha
            in_cur[kb] = y1a + hb

    ax.set_xlim(-0.45, len(steps) - 0.55)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([f"acquisition {i+1}" for i in range(len(steps))], fontsize=8)
    ax.set_yticks([])
    ax.grid(False)
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)


# --------------------------------------------------------------------------------------
# 5. Panel A：逐折配对检验（Top-k / MRR / AUC 这类无 per-item logL 的指标）
# --------------------------------------------------------------------------------------

def paired_fold_test(folds: pd.DataFrame, metric: str, left: str, right: str) -> Dict[str, float]:
    """
    逐折配对检验（正态近似的双侧 p）。

    注意口径：这是 **fold 级** 配对，K=5 时功效很低；per-transition 的配对 bootstrap
    只对 mean_logl 可得（评估流程只落盘了 per-item logL）。因此 Top-1 的差异
    只能作为描述性结果，不应作为主 claim。
    """
    piv = folds.pivot_table(index="fold", columns="method", values=metric)
    if left not in piv.columns or right not in piv.columns:
        return {}
    d = (piv[left] - piv[right]).dropna().values.astype(float)
    k = len(d)
    if k < 2:
        return {"metric": metric, "n_folds": k, "delta_mean": float(d.mean()) if k else np.nan}
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(k)
    t = float(d.mean() / se) if se > 0 else np.inf
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))
    return {"metric": metric, "left": left, "right": right, "n_folds": k,
            "delta_mean": float(d.mean()), "delta_sd": sd,
            "ci_lo": float(d.mean() - 1.96 * se), "ci_hi": float(d.mean() + 1.96 * se),
            "t": t, "p_two_sided": float(min(1.0, p)),
            "per_fold_delta": [round(float(x), 4) for x in d]}


# --------------------------------------------------------------------------------------
# 6. 五个 panel 的绘制
# --------------------------------------------------------------------------------------

def panel_a_prediction(ax_logl, ax_top1, overall: pd.DataFrame, folds: pd.DataFrame,
                       methods: Sequence[str] = MAIN_METHODS,
                       ax_delta=None,
                       focal: Tuple[str, str] = ("TS_pretrained", "HT_second-order"),
                       delta_metrics: Sequence[str] = ("mean_logl", "top1", "top3", "top5", "mrr", "auc"),
                       delta_logl_text: str = "", delta_top1_text: str = "",
                       overlay_fold_points: bool = False):
    """
    A：预测性能。主 claim 是 parity（Two-stage 不输 HyperTraPS 且无需 MCMC）。

    默认样式（`overlay_fold_points=False`）：
      * 条形 = 跨折均值，末端直接标数值；**不叠加抖动散点** ——
        逐折 marginal 散点会让"某个方法的最好一折"抢走视觉焦点，
        与配对比较的结论相反（HyperTraPS 的最优折 0.517/0.500 比 Two-stage 更靠右，
        但逐折配对里 Two-stage 在 4/5 折占优）。
      * 折级信息改用 `ax_delta` 的**配对差异**呈现（`plot_paired_fold_delta`）。

    `overlay_fold_points=True` 复现最初版本的样式（条形 + 抖动散点 + 条内 Δ 文本），
    仅用于保留 v1 图，不建议作为正文主图。
    """
    o = overall.set_index("method")
    use = [m for m in methods if m in o.index]
    labels = [METHOD_LABELS.get(m, m) for m in use]
    colors = [METHOD_COLORS.get(m, "#4C72B0") for m in use]
    y = np.arange(len(use))

    for ax, col, xlabel, title, note in (
        (ax_logl, "mean_logl", "held-out mean log-likelihood", "Next-step prediction (grouped CV)",
         delta_logl_text),
        (ax_top1, "top1", "Top-1 accuracy",
         "per-fold values overlaid" if overlay_fold_points else "Top-1 accuracy (bars = CV mean)",
         delta_top1_text),
    ):
        vals = np.asarray([o.loc[m, col] for m in use], dtype=float)
        ax.barh(y, vals, color=colors, edgecolor="none", height=0.66)
        span = float(np.nanmax(np.abs(vals))) or 1.0
        if overlay_fold_points:
            for i, m in enumerate(use):
                v = folds[folds["method"] == m][col].values.astype(float)
                if len(v):
                    ax.scatter(v, np.full(len(v), i) + np.linspace(-0.16, 0.16, len(v)),
                               s=9, color="#333333", alpha=0.75, zorder=3, linewidths=0)
        else:
            for yi, v in zip(y, vals):
                inside = v < 0
                ax.text(v + (0.02 if not inside else -0.02) * span, yi, "%.3f" % v,
                        va="center", ha="left" if not inside else "right",
                        fontsize=6.8, color="#222222")
        ax.set_yticks(y)
        ax.set_yticklabels(labels if col == "mean_logl" else [], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo - 0.10 * (hi - lo) if vals.min() < 0 else 0.0, hi + 0.16 * (hi - lo))
        if not note:
            continue
        if overlay_fold_points:                     # v1：文本压在条形内（保留原样式）
            ax.text(0.98, 0.04, note, transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6.8, color="#333333")
        elif not (ax_delta is not None and col == "top1"):
            ax.text(0.5, -0.34, note, transform=ax.transAxes, ha="center", va="top",
                    fontsize=6.6, color="#333333",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#f5f5f5", ec="#cccccc", lw=0.5))

    if ax_delta is None:
        return
    plot_paired_fold_delta(ax_delta, folds, focal=focal, metrics=delta_metrics)


def plot_paired_fold_delta(ax, folds: pd.DataFrame,
                           focal: Tuple[str, str] = ("TS_pretrained", "HT_second-order"),
                           metrics: Sequence[str] = ("mean_logl", "top1", "top3", "top5",
                                                     "mrr", "auc")):
    """
    逐折配对差异图（Supplementary）。

    与统计检验完全一致的视觉编码：每折一个灰点 + 均值 95% CI 菱形须。
    这是解释"为什么条形图上 Two-stage 更高、但散点看起来 HyperTraPS 更好"的关键图 ——
    marginal 分布丢掉了配对关系。
    """
    left, right = focal
    ks = list(metrics)
    for k, m in enumerate(ks):
        t = paired_fold_test(folds, m, left, right)
        if not t or "per_fold_delta" not in t:
            continue
        d = np.asarray(t["per_fold_delta"], dtype=float)
        ax.scatter(d, np.full(len(d), k) + np.linspace(-0.14, 0.14, len(d)),
                   s=13, color="#666666", alpha=0.85, zorder=3, linewidths=0)
        ax.errorbar(t["delta_mean"], k,
                    xerr=[[t["delta_mean"] - t["ci_lo"]], [t["ci_hi"] - t["delta_mean"]]],
                    fmt="D", ms=4.5, color=METHOD_COLORS.get(left, "#1f6fb4"),
                    ecolor=METHOD_COLORS.get(left, "#1f6fb4"),
                    elinewidth=1.4, capsize=3, zorder=4)
        ax.text(1.02, k, "p=%.2f%s" % (t["p_two_sided"], "*" if t["p_two_sided"] < 0.05 else ""),
                transform=ax.get_yaxis_transform(), va="center",
                fontsize=6.4, color="#c0392b" if t["p_two_sided"] < 0.05 else "#666666")
    ax.axvline(0, color="#333333", lw=1.0)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels(ks, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("per-fold paired difference\n(%s \u2212 %s)"
                  % (METHOD_LABELS.get(left, left).replace("\n", " "),
                     METHOD_LABELS.get(right, right)), fontsize=8)
    ax.set_title("Paired comparison (K=5 folds)", fontsize=9)
    lim = max(0.02, float(np.nanmax(np.abs(ax.get_xlim()))))
    ax.set_xlim(-lim, lim)
    ax.text(0.02, 0.02, "\u2190 %s better   |   %s better \u2192"
            % (METHOD_LABELS.get(right, right), METHOD_LABELS.get(left, left).replace("\n", " ")),
            transform=ax.transAxes, fontsize=6.2, color="#555555")


def panel_b_dynamics(ax_left, ax_right, bubbles: Dict[str, pd.DataFrame],
                     feature_order: Sequence[str],
                     class_map: Dict[str, str], mech_map: Dict[str, str],
                     model_keys: Sequence[str] = ("HT_second-order", "TS_pretrained"),
                     empirical_key: str = "empirical",
                     observed: Optional[pd.DataFrame] = None,
                     row_h: float = 0.86, min_support: int = 15,
                     support: Optional[Dict[str, int]] = None,
                     highlight: Optional[Sequence[str]] = None,
                     header: Optional[str] = None,
                     show_empirical: bool = True):
    """
    B：posterior acquisition dynamics 的**左右对照**（对齐文献 Fig 7C 的呈现方式）。

    每个特征一行、横轴为获得序位；两列分别是 HyperTraPS 与 two-stage 的后验分布。
    关键设计——**经验分布以灰色底图叠在两列的同一行上**，并用红色虚线标出经验期望序位：
    这样"哪一侧的蓝色分布更贴合灰色底图"就是可直接目视判读的模型-数据一致性，
    而不需要读者在两张独立的图之间来回比对。
    浅灰横带 = 数据中从未观测到该特征被获得的序位区间（模型落在此区间的质量不可验证）。

    `show_empirical=False` 时只绘制模型后验（蓝色分布），移除灰色经验分布底图、
    红色经验期望序位虚线以及浅灰 never-observed 横带，图例只保留 model posterior。

    `ax_left is ax_right` 时退化为**单列**模式（用于没有可对照模型的数据集，例如
    HyperTraPS 无法运行的 K. pneumoniae）。`highlight` 中的特征会加一条淡黄色底纹。
    """
    ref = bubbles[model_keys[0]]
    steps = np.arange(1, ref.shape[1] + 1, dtype=float)
    order = [f for f in feature_order if f in ref.index]
    n = len(order)
    emp = bubbles.get(empirical_key)
    obs = observed.set_index("feature") if observed is not None else None
    eo_emp = _expected_order_series(emp) if emp is not None else None
    hl = set(highlight or [])
    # ax_left 也可以直接传入一个 axes 序列 -> 支持任意列数（例如三方对照）
    if isinstance(ax_left, (list, tuple, np.ndarray)):
        axs = list(ax_left)
        ax_left, ax_right = axs[0], axs[-1]
    else:
        axs = [ax_left] if ax_left is ax_right else [ax_left, ax_right]

    for ax, key in zip(axs, list(model_keys)[:len(axs)]):
        b = bubbles[key]
        # 当"模型列"本身就是经验分布时（例如尚未训练模型的数据集），不再重复叠灰底
        same_as_emp = (key == empirical_key)
        for r, f in enumerate(order):
            base = float(n - r)
        v = b.loc[f].values.astype(float) if f in b.index else np.zeros(len(steps))
        ve = (emp.loc[f].values.astype(float)
              if (show_empirical and emp is not None and f in emp.index and not same_as_emp) else None)
        # 每条分布各自归一化到 row_h：本 panel 比较的是"质量落在哪个序位"，
        # 而非峰高（模型把质量摊在 19 步、经验集中在 <=10 步，共用尺度会把模型压平）
        # show_empirical=False 时放大有效行高，让仅剩的模型后验清晰可读
        eff_row_h = row_h * (2.0 if not show_empirical else 1.0)
        vmax_m = max(float(v.max()), 1e-9)
        vmax_e = max(float(ve.max()), 1e-9) if ve is not None else 1.0
        if f in hl:
            ax.add_patch(Rectangle((0.5, base - 0.06), steps[-1], row_h + 0.12,
                                   facecolor="#FFF3C4", edgecolor="none", zorder=0))
        if show_empirical and obs is not None and f in obs.index:
            lo = obs.loc[f, "first_observed_step"]
            hi = obs.loc[f, "last_observed_step"]
            if np.isfinite(lo) and lo > 1:
                ax.add_patch(Rectangle((0.5, base), lo - 1.0, row_h,
                                       facecolor="#F2F2F2", edgecolor="none", zorder=0))
            if np.isfinite(hi) and hi < steps[-1]:
                ax.add_patch(Rectangle((hi + 0.5, base), steps[-1] + 0.5 - (hi + 0.5), row_h,
                                       facecolor="#F2F2F2", edgecolor="none", zorder=0))
        if ve is not None:
            ax.fill_between(steps, base, base + ve / vmax_e * row_h, step="mid",
                            color="#AFAFAF", alpha=0.55, linewidth=0, zorder=1)
        weak = support is not None and support.get(f, 10 ** 9) < min_support
        ax.bar(steps, v / vmax_m * eff_row_h, bottom=base, width=0.70, zorder=2,
               color=CLASS_COLORS.get(class_map.get(f, "other"), "#999999"),
               alpha=0.45 if weak else 0.95, linewidth=0)
        if ve is not None:   # 经验分布再叠一条深灰轮廓，避免被模型柱体完全遮住
            ax.step(np.append(steps, steps[-1] + 1) - 0.5,
                    np.append(base + ve / vmax_e * row_h, base),
                    where="post", color="#404040", lw=0.7, zorder=5)
        ax.hlines(base, 0.5, steps[-1] + 0.5, color="#555555", lw=0.5, zorder=3)
        if show_empirical and eo_emp is not None and np.isfinite(eo_emp.get(f, np.nan)):
            ax.vlines(float(eo_emp[f]), base, base + row_h, color="#c0392b",
                      lw=1.0, ls="--", zorder=4)

        ax.set_xlim(0.5, steps[-1] + 0.5)
        ax.set_ylim(0.6, n + (1.05 if show_empirical else eff_row_h + 0.15))
        tick_step = max(1, int(np.ceil(steps[-1] / 8.0)))   # 最多 ~8 个刻度，窄栏也不挤
        ax.set_xticks(np.arange(1, steps[-1] + 1, tick_step))
        ax.set_xlabel("number of features already acquired (ordering step)", fontsize=8)
        ax.grid(False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(METHOD_LABELS.get(key, key).replace("\n", " "), fontsize=9)

    ax_left.set_yticks([float(n - r) + row_h / 2 for r in range(n)])
    ax_left.set_yticklabels(
        [f + (" \u25b2" if mech_map.get(f) == "compensatory_mutation" else "") for f in order],
        fontsize=7.2)
    for tick, f in zip(ax_left.get_yticklabels(), order):
        tick.set_color(CLASS_COLORS.get(class_map.get(f, "other"), "#333333"))
    for ax in axs[1:]:
        ax.set_yticks([])

    handles = [
        Rectangle((0, 0), 1, 1, facecolor="#2C6FA6", edgecolor="none", label="model posterior (each row scaled to its own max)"),
    ]
    if show_empirical:
        handles += [
            Rectangle((0, 0), 1, 1, facecolor="#AFAFAF", alpha=0.55, edgecolor="#404040",
                      label="empirical distribution"),
            plt.Line2D([0], [0], color="#c0392b", ls="--", lw=1.2, label="empirical expected order"),
            Rectangle((0, 0), 1, 1, facecolor="#F2F2F2", edgecolor="#DDDDDD", label="never observed"),
        ]
    if hl:
        handles.append(Rectangle((0, 0), 1, 1, facecolor="#FFF3C4", edgecolor="#E0CF7A",
                                 label="bimodal (distinct pathways)"))
    ax_left.legend(handles=handles, fontsize=6.6, ncol=4, loc="upper left",
                   bbox_to_anchor=(0.0, -0.095), frameon=False, handlelength=1.4,
                   columnspacing=1.1, borderpad=0.2)
    has_comp = any(mech_map.get(f) == "compensatory_mutation" for f in order)
    if header is None:
        if show_empirical:
            header = ("Posterior acquisition dynamics: rows ordered by empirical expected order, "
                      "coloured by drug class"
                      + (" ( \u25b2 = compensatory )" if has_comp else ""))
        else:
            header = ("Posterior acquisition dynamics, coloured by drug class"
                      + (" ( \u25b2 = compensatory )" if has_comp else ""))
    if header:
        ax_left.text(0.0, 1.075, header, transform=ax_left.transAxes,
                     fontsize=7.6, va="bottom", color="#333333")

def panel_b_hierarchy(ax, order_df: pd.DataFrame, class_map: Dict[str, str],
                      mech_map: Dict[str, str], min_support: int = 15):
    """
    经验获得层级的 lollipop 版本（现降为 Supplementary）。

    lollipop + `order_sd` 误差棒；点大小 = 获得事件数；低支持度特征淡化。
    不人为划"波次"——让连续的 hierarchy 自己呈现。
    """
    d = order_df.sort_values("expected_order").reset_index(drop=True)
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        f = str(r["feature"])
        cls = class_map.get(f, "other")
        col = CLASS_COLORS.get(cls, "#999999")
        weak = int(r["n_gain_events"]) < min_support
        alpha = 0.35 if weak else 1.0
        ax.hlines(yi, 0.6, r["expected_order"], color=col, alpha=alpha * 0.55, lw=1.6)
        ax.errorbar(r["expected_order"], yi, xerr=r["order_sd"], fmt="none",
                    ecolor=col, alpha=alpha * 0.7, elinewidth=1.1, capsize=2)
        ax.scatter(r["expected_order"], yi,
                   s=18 + 2.0 * float(r["n_gain_events"]),
                   marker="^" if mech_map.get(f) == "compensatory_mutation" else "o",
                   color=col, alpha=alpha, edgecolors="white", linewidths=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([str(f) for f in d["feature"]], fontsize=7.5)
    for tick, f in zip(ax.get_yticklabels(), d["feature"]):
        tick.set_color(CLASS_COLORS.get(class_map.get(str(f), "other"), "#333333"))
    ax.set_xlabel("empirical expected acquisition order  (earlier \u2190 \u2192 later)", fontsize=8)
    ax.set_title("Empirical resistance acquisition hierarchy\n"
                 "(marker size = #gain events; \u25b2 = compensatory; faded = low support)",
                 fontsize=9)
    ax.set_xlim(0.4, float(np.nanmax(d["expected_order"] + d["order_sd"])) + 0.6)

    seen: List[str] = []
    handles = []
    for f in d["feature"]:
        cls = class_map.get(str(f), "other")
        if cls not in seen:
            seen.append(cls)
            handles.append(plt.Line2D([0], [0], marker="o", linestyle="none",
                                      markersize=5, markerfacecolor=CLASS_COLORS.get(cls, "#999"),
                                      markeredgecolor="white", label=cls))
    ax.legend(handles=handles, fontsize=6.2, ncol=2, loc="lower right", frameon=True,
              handletextpad=0.3, columnspacing=0.8, borderpad=0.4)


def panel_c_recovery(axes, scatter: pd.DataFrame, stats: Dict[str, Dict[str, float]],
                     palette: Dict[str, str], model_order: Sequence[str],
                     ci_text: Dict[str, str] = None, annotate: int = 4):
    """C：两个模型各自与经验次序的一致程度（共用坐标 + identity line）。"""
    ci_text = ci_text or {}
    lo = float(np.nanmin([scatter["empirical_order"].min(), scatter["model_order"].min()])) - 0.4
    hi = float(np.nanmax([scatter["empirical_order"].max(), scatter["model_order"].max()])) + 0.4
    for ax, name in zip(axes, model_order):
        sub = scatter[scatter["model"] == name]
        ax.plot([lo, hi], [lo, hi], "--", color="#999999", lw=1, zorder=1)
        ax.scatter(sub["empirical_order"], sub["model_order"], s=42, zorder=3,
                   c=[palette.get(f, "#999999") for f in sub["feature"]],
                   edgecolors="white", linewidths=0.7)
        for _, r in sub.reindex(sub["residual"].abs().sort_values(ascending=False).index) \
                        .head(annotate).iterrows():
            ax.annotate(str(r["feature"]), (r["empirical_order"], r["model_order"]),
                        fontsize=6.5, xytext=(4, 3), textcoords="offset points")
        s = stats.get(name, {})
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("empirical expected order", fontsize=8)
        ax.set_title("%s\n$\\rho$ = %.3f   MAE = %.2f%s"
                     % (METHOD_LABELS.get(name, name).replace("\n", " "),
                        s.get("spearman", np.nan), s.get("mae", np.nan),
                        ("\n" + ci_text[name]) if name in ci_text else ""), fontsize=8.5)
    axes[0].set_ylabel("model expected order", fontsize=8)


def panel_e_dependencies(ax, dep: pd.DataFrame, show: Optional[int] = None):
    """E：条件依赖的双证据条形图（模型 log2 shift vs 经验 log2 OR）。"""
    d = (dep[dep["selected"].astype(bool)] if "selected" in dep.columns else dep).copy()
    if show:
        d = d.head(show)
    d = d.sort_values("model_log2_shift")
    y = np.arange(len(d))
    h = 0.36
    ax.barh(y + h / 2, d["model_log2_shift"], height=h, color="#1f6fb4",
            edgecolor="none", label="model  $\\log_2\\frac{p(i\\mid\\{j\\})}{p(i\\mid\\emptyset)}$")
    ax.barh(y - h / 2, d["emp_log2_or"], height=h, color="#8c8c8c",
            edgecolor="none", label="empirical  $\\log_2$ OR")
    ax.axvline(0, color="#333333", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(d["pair"], fontsize=7.5)
    for yi, (_, r) in zip(y, d.iterrows()):
        star = "*" if float(r["fisher_p"]) < 0.05 else ""
        vals = [float(r["model_log2_shift"]), float(r["emp_log2_or"])]
        # 正向条标在右端；负向条的右半区是空的，标在 0 右侧最清晰
        xpos = (max(vals) + 0.15) if max(vals) > 0 else 0.15
        ax.text(xpos, yi, "%d/%d gains, n=%d/%d%s"
                % (int(r["n_gain_ctx_present"]), int(r["n_gain_ctx_absent"]),
                   int(r["n_at_risk_ctx_present"]), int(r["n_at_risk_ctx_absent"]), star),
                fontsize=6.0, va="center", ha="left", color="#444444")
    ax.set_xlabel("$\\log_2$ change in next-step probability of $i$ given context $j$", fontsize=8)
    ax.set_title("Conditional evolutionary dependencies\n"
                 "(model prediction vs independent empirical estimate)", fontsize=9)
    ax.legend(fontsize=6.4, loc="upper left", frameon=True)
    lim = float(np.nanmax(np.abs(np.concatenate([d["model_log2_shift"].values,
                                                 d["emp_log2_or"].values])))) + 3.4
    ax.set_xlim(-lim, lim)


def make_main_figure(overall: pd.DataFrame, folds: pd.DataFrame,
                     order_df: pd.DataFrame, class_map: Dict[str, str],
                     mech_map: Dict[str, str],
                     scatter: pd.DataFrame, recovery_stats: Dict[str, Dict[str, float]],
                     palette: Dict[str, str], model_order: Sequence[str],
                     comp: pd.DataFrame, flows: pd.DataFrame,
                     dep: pd.DataFrame,
                     bubbles: Optional[Dict[str, pd.DataFrame]] = None,
                     observed: Optional[pd.DataFrame] = None,
                     ci_text: Optional[Dict[str, str]] = None,
                     delta_logl_text: str = "", delta_top1_text: str = "",
                     show_paired_panel: bool = False,
                     overlay_fold_points: bool = False,
                     show_empirical: bool = True,
                     suptitle: str = "", save_prefix: Optional[str] = None):
    """
    组装 5-panel 主图。

    `bubbles` 给出时，Panel B 使用 posterior acquisition dynamics 的左右对照
    （对齐文献 Fig 7C）；否则退化为经验层级 lollipop。
    `show_paired_panel=True` 时 Panel A 追加逐折配对差异子图；默认关闭，
    该图作为 Supplementary（`plot_paired_fold_delta`）单独输出。
    `overlay_fold_points=True` + `bubbles=None` 即最初版本（v1）的样式。
    """
    dynamics = bubbles is not None
    fig = plt.figure(figsize=(11.6, 17.6) if dynamics else (11.6, 15.4))
    gs = fig.add_gridspec(4, 2,
                          height_ratios=[1.0, 3.15, 1.20, 1.30] if dynamics
                          else [1.0, 2.0, 1.25, 1.35],
                          hspace=0.60 if dynamics else 0.55, wspace=0.28)

    if show_paired_panel:
        gs_a = gs[0, :].subgridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.38)
        ax_a3 = fig.add_subplot(gs_a[0, 2])
    else:
        gs_a = gs[0, :].subgridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.22)
        ax_a3 = None
    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[0, 1])
    panel_a_prediction(ax_a1, ax_a2, overall, folds, ax_delta=ax_a3,
                       delta_logl_text=delta_logl_text, delta_top1_text=delta_top1_text,
                       overlay_fold_points=overlay_fold_points)
    panel_label(ax_a1, "A")

    feature_order = order_df.sort_values("expected_order")["feature"].astype(str).tolist()
    if dynamics:
        gs_b = gs[1, :].subgridspec(1, 2, wspace=0.06)
        ax_b1 = fig.add_subplot(gs_b[0, 0])
        ax_b2 = fig.add_subplot(gs_b[0, 1])
        support = dict(zip(order_df["feature"].astype(str),
                           order_df["n_gain_events"].astype(int))) \
            if "n_gain_events" in order_df.columns else None
        panel_b_dynamics(ax_b1, ax_b2, bubbles, feature_order, class_map, mech_map,
                         model_keys=list(model_order), observed=observed, support=support,
                         show_empirical=show_empirical)
        panel_label(ax_b1, "B", dx=-0.115)
    else:
        ax_b1 = fig.add_subplot(gs[1, :])
        panel_b_hierarchy(ax_b1, order_df, class_map, mech_map)
        panel_label(ax_b1, "B", dx=-0.055)

    ax_c1 = fig.add_subplot(gs[2, 0])
    ax_c2 = fig.add_subplot(gs[2, 1])
    panel_c_recovery([ax_c1, ax_c2], scatter, recovery_stats, palette, model_order,
                     ci_text=ci_text)
    panel_label(ax_c1, "C")

    ax_d = fig.add_subplot(gs[3, 0])
    plot_class_alluvial(ax_d, comp, flows,
                        title="Representative evolutionary pathways\n"
                              "(two-stage forward walks, drug-class level)")
    panel_label(ax_d, "D")

    ax_e = fig.add_subplot(gs[3, 1])
    panel_e_dependencies(ax_e, dep)
    panel_label(ax_e, "E")

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
    """写出 figure_tb_main_summary.json / .md（论文正文可直接引用的数字）。"""
    jpath, mpath = path_prefix + ".json", path_prefix + ".md"
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)

    def fmt(v, nd=4):
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) and v == v else str(v)

    lines = ["# TB 主图（Figure X）核心数字", ""]
    pred = payload.get("prediction", {})
    lines += ["## Panel A — Prediction (parity)", "",
              "| method | mean logL | Top-1 | Top-3 | AUC |", "|---|---|---|---|---|"]
    for m, v in (pred.get("overall") or {}).items():
        lines.append("| %s | %s | %s | %s | %s |" % (m, fmt(v.get("mean_logl")), fmt(v.get("top1")),
                                                    fmt(v.get("top3")), fmt(v.get("auc"))))
    for k, v in (pred.get("tests") or {}).items():
        lines.append("")
        lines.append("- **%s**: Δ = %s, p = %s%s" % (
            k, fmt(v.get("delta_mean", v.get("delta_mean_logl"))), fmt(v.get("p_two_sided", v.get("p_value"))),
            ("，per-fold Δ = %s" % v.get("per_fold_delta")) if v.get("per_fold_delta") else ""))

    rec = payload.get("ordering_recovery", {})
    lines += ["", "## Panel C — Recovery of empirical ordering", "",
              "| model | Spearman ρ | MAE | median JS |", "|---|---|---|---|"]
    for m, v in (rec.get("point") or {}).items():
        lines.append("| %s | %s | %s | %s |" % (m, fmt(v.get("spearman")), fmt(v.get("mae")),
                                               fmt(v.get("median_js"))))
    for row in (rec.get("bootstrap_delta") or []):
        lines.append("")
        lines.append("- **%s (%s)**: Δ = %s, 95%% CI [%s, %s], p = %s" % (
            row.get("quantity"), row.get("model"), fmt(row.get("point_estimate")),
            fmt(row.get("ci_lo")), fmt(row.get("ci_hi")), fmt(row.get("p_two_sided"))))
    lines += ["", "> 截断口径：max_step = %s（经验 bubble 覆盖的最大序位）；"
              "敏感性曲线见 Supplementary。" % rec.get("max_step")]

    dep = payload.get("dependencies") or []
    lines += ["", "## Panel E — Conditional dependencies (双证据)", "",
              "| pair | model log2 shift | empirical log2 OR | n(ctx+/ctx−) | Fisher p |",
              "|---|---|---|---|---|"]
    for r in dep:
        lines.append("| %s | %s | %s | %d/%d | %s |" % (
            r.get("pair"), fmt(r.get("model_log2_shift"), 2), fmt(r.get("emp_log2_or"), 2),
            int(r.get("n_at_risk_ctx_present", 0)), int(r.get("n_at_risk_ctx_absent", 0)),
            fmt(r.get("fisher_p"), 4)))

    paths = payload.get("pathways") or []
    if paths:
        lines += ["", "## Panel D — Top drug-class trajectories", "",
                  "| path | probability |", "|---|---|"]
        for r in paths:
            lines.append("| %s | %s |" % (r.get("path"), fmt(r.get("probability"), 3)))

    for note in (payload.get("caveats") or []):
        lines += ["", "> " + str(note)]

    with open(mpath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return jpath, mpath
