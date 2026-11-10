#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hypertraps_compare.py —— 两阶段模型 vs HyperTraPS 的"路径级"对照工具

现有 `40_eval_grouped_repr_evo_hypertraps_vs_dl.sh` 只在 grouped-CV 下比较**预测指标**
（mean logL / Top-k / AUC）。本模块补齐文献中真正用于讲故事的那一层比较，即
Greenbury/Renz 等在 TB 数据上给出的 Figure 7 式分析：

    A/B. HyperTraPS transition graph        —— 超立方状态图，边宽 = 单步获得概率
    C.   Posterior acquisition dynamics     —— 每个特征"在第几步被获得"的后验分布，
                                               基因位点与其对应药物用颜色关联，
                                               灰色区域 = 数据中未观测到的获得区间

对应到本项目：

| 文献侧 | 本项目侧 |
|---|---|
| HyperTraPS θ 后验（forwards.txt） | Stage-2 模型的条件分布 p(i \\| s) |
| 前向游走（rate_j = exp(F_jj + Σ_{i∈s} F_ji)） | 前向游走（softmax over 候选集合） |
| posterior bubble / ordering 直方图 | model bubble / ordering 直方图 |
| transition graph | 同构的状态转移图 |

因此二者可以在**完全相同的前向游走协议**下逐图对照，而不只是比一个标量指标。

注意：本模块跑的 HyperTraPS 是**全量数据**上的一次 MCMC（用于路径可视化与机制解读），
与 grouped-CV 中"每折内重训"的评估口径不同，不可混用于性能声明。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EVOAMR_ROOT = os.path.dirname(HERE)
MODELS_DIR = os.path.join(EVOAMR_ROOT, "models")
EVAL_DIR = os.path.join(EVOAMR_ROOT, "eval")
for _p in (HERE, MODELS_DIR, EVAL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evoamr_story as st  # noqa: E402

Scorer = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------------------
# 1. 运行 HyperTraPS 并保留后验
# --------------------------------------------------------------------------------------

LOCAL_HT_BIN = os.path.join(HERE, "bin", "RUN_MCMC_SAMPLER")
HT_SOURCE_DIR = os.path.join(os.path.dirname(EVOAMR_ROOT), "refs", "HyperTraPS", "src", "cpp")
HT_SOURCES = ["graph.cpp", "likelihood_sampler_thread.cpp", "mcmc_sampler.cpp", "state.cpp",
              "trajectory.cpp", "transition.cpp", "updater_fast.cpp", "utilities.cpp",
              "run_mcmc_sampler.cpp"]


def _binary_runnable(path: str) -> bool:
    """
    检测二进制能否在当前机器上真正执行（预编译版本常因 glibc/libstdc++ 版本过新而失败）。
    注意 legacy 采样器打印用法后以非 0 退出，因此以"是否打印出自身 banner"为判据。
    """
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        return False
    try:
        proc = subprocess.run([path, "-h"], capture_output=True, timeout=60)
        return b"HyperTraPS" in (proc.stdout + proc.stderr)
    except (OSError, subprocess.SubprocessError):
        return False


def build_local_ht_binary(out_path: str = LOCAL_HT_BIN, cxx: str = "g++",
                          force: bool = False) -> str:
    """
    从 `refs/HyperTraPS/src/cpp` 本地编译 legacy 采样器。

    仓库自带的 `refs/HyperTraPS/bin/RUN_MCMC_SAMPLER` 是在较新的发行版上编译的，
    在 glibc 较旧的机器上会报 `GLIBC_2.34/2.38 not found` 而无法运行；
    本函数用本机编译器重新构建一份可用二进制（功能与源码完全一致）。
    """
    if _binary_runnable(out_path) and not force:
        return out_path
    if not os.path.isdir(HT_SOURCE_DIR):
        raise RuntimeError(f"未找到 HyperTraPS 源码目录: {HT_SOURCE_DIR}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [cxx, "-O2", "-std=c++17", "-pthread"] + \
          [os.path.join(HT_SOURCE_DIR, s) for s in HT_SOURCES] + ["-o", out_path]
    subprocess.run(cmd, check=True)
    if not _binary_runnable(out_path):
        raise RuntimeError("本地编译完成但二进制仍无法运行: " + out_path)
    return out_path


def resolve_ht_runtime(ht_bin: Optional[str] = None, allow_local_build: bool = True) -> Dict[str, str]:
    """
    解析可用的 HyperTraPS 运行时。优先顺序：
      1. 显式指定的 `ht_bin`；
      2. 本地编译的 `analysis_repr_evo/bin/RUN_MCMC_SAMPLER`（若可运行）；
      3. 仓库自带 `refs/HyperTraPS/bin/RUN_MCMC_SAMPLER`（若可运行）；
      4. 现场本地编译（`allow_local_build=True`）。
    """
    from pathlib import Path

    from run_phylo_eval import resolve_hypertraps_runtime  # noqa: E402

    if ht_bin:
        return resolve_hypertraps_runtime(Path(EVOAMR_ROOT), ht_bin=ht_bin)
    if _binary_runnable(LOCAL_HT_BIN):
        return {"backend": "legacy_sampler", "bin": LOCAL_HT_BIN, "post_process": None,
                "provenance": "locally compiled"}
    try:
        runtime = resolve_hypertraps_runtime(Path(EVOAMR_ROOT))
        if _binary_runnable(runtime["bin"]):
            runtime["provenance"] = "repository prebuilt"
            return runtime
    except RuntimeError:
        pass
    if not allow_local_build:
        raise RuntimeError("没有可运行的 HyperTraPS 二进制，且已禁用本地编译。")
    path = build_local_ht_binary()
    return {"backend": "legacy_sampler", "bin": path, "post_process": None,
            "provenance": "locally compiled (on demand)"}


def write_ht_transitions(ts: "st.TransitionSet", path: str) -> int:
    """
    写 HyperTraPS legacy 采样器所需的输入：每条转移占两行（源状态 / 仅含增益的目标状态），
    空格分隔的 0/1，无表头。与 `eval/run_phylo_eval.py` 完全同口径。
    """
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(ts.n):
            src = ts.src[i].astype(int)
            dst = src.copy()
            for j in ts.gained[i]:
                dst[j] = 1
            if not ts.gained[i]:
                continue
            fh.write(" ".join(str(int(x)) for x in src.tolist()) + "\n")
            fh.write(" ".join(str(int(x)) for x in dst.tolist()) + "\n")
            n += 1
    return n


def run_hypertraps_posterior(
    ts: "st.TransitionSet",
    outdir: str,
    model: str = "second-order",
    n_posterior: int = 120,
    burn: int = 3000,
    r: int = 50,
    threads: int = 4,
    timeout_sec: int = 1800,
    seed: int = 1,
    force: bool = False,
    ht_bin: Optional[str] = None,
) -> Dict[str, object]:
    """
    在**全量转移**上跑一次 HyperTraPS MCMC，并把 θ 后验（forwards.txt）保留下来。

    返回 dict: status ∈ {"cached", "ok", "timeout", "failed"}，含 elapsed_sec、forwards、cmd、log 等。
    超时/失败不会抛异常——超时本身就是可扩展性对比的一部分证据。
    """
    os.makedirs(outdir, exist_ok=True)
    forwards = os.path.join(outdir, "forwards.txt")
    meta_path = os.path.join(outdir, "ht_run_meta.json")
    requested = {"model": model, "N": n_posterior, "burn": burn, "r": r,
                 "threads": threads, "seed": seed, "timeout_sec": timeout_sec}

    if os.path.isfile(forwards) and os.path.getsize(forwards) > 0 and not force:
        meta: Dict[str, object] = {}
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        cached_params = meta.get("ht_params") or {}
        # 仅当采样参数与本次请求一致时才复用缓存，避免"声明参数"与"实际后验"不符
        comparable = {k: cached_params.get(k) for k in ("model", "N", "burn", "r")}
        if comparable == {k: requested[k] for k in ("model", "N", "burn", "r")}:
            meta.update({"status": "cached", "forwards": forwards})
            return meta
        print("[ht] 缓存后验的采样参数与本次请求不一致，重新运行 MCMC。",
              "cached=", comparable,
              "requested=", {k: requested[k] for k in ("model", "N", "burn", "r")})

    runtime = resolve_ht_runtime(ht_bin)
    if runtime.get("backend") != "legacy_sampler":
        raise RuntimeError("本模块目前只支持 legacy RUN_MCMC_SAMPLER 后端，"
                           f"实际解析到 backend={runtime.get('backend')}")

    inp = os.path.join(outdir, "transitions.txt")
    n_used = write_ht_transitions(ts, inp)
    with open(os.path.join(outdir, "labels.csv"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(ts.feature_names) + "\n")

    cmd = [runtime["bin"], "-f", "transitions.txt", "-M", model,
           "-N", str(n_posterior), "-r", str(r), "-b", str(burn),
           "-i", "100", "-S", str(seed), "-t", str(threads)]
    log_path = os.path.join(outdir, "ht_run.log")
    t0 = time.time()
    status, error = "ok", None
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            subprocess.run(cmd, cwd=outdir, check=True, timeout=timeout_sec,
                           stdout=logf, stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        status, error = "timeout", f"TimeoutExpired after {timeout_sec}s"
    except (subprocess.CalledProcessError, OSError) as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - t0

    if status == "ok" and not (os.path.isfile(forwards) and os.path.getsize(forwards) > 0):
        status, error = "failed", "finished without forwards.txt"

    meta = {
        "status": status,
        "error": error,
        "elapsed_sec": round(elapsed, 1),
        "cmd": " ".join(cmd),
        "backend": runtime.get("backend"),
        "bin": runtime.get("bin"),
        "outdir": os.path.abspath(outdir),
        "forwards": forwards if status == "ok" else None,
        "log": log_path,
        "n_transitions_used": n_used,
        "L": int(ts.L),
        "ht_params": requested,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return meta


def load_ht_theta(forwards_path: str, L: int, max_samples: int = 200) -> List[np.ndarray]:
    samples: List[np.ndarray] = []
    with open(forwards_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            vals = [float(x) for x in parts]
            if len(vals) == 1 + L * L:
                vals = vals[1:]
            elif len(vals) != L * L:
                continue
            samples.append(np.asarray(vals, dtype=float).reshape(L, L))
    samples = [F for F in samples if np.abs(F).sum() > 1e-9] or samples
    if max_samples and len(samples) > max_samples:
        idx = np.linspace(0, len(samples) - 1, max_samples).astype(int)
        samples = [samples[i] for i in idx]
    return samples


def ht_scorer(samples: Sequence[np.ndarray]) -> Scorer:
    """
    HyperTraPS 后验预测分布：rate_j(s) = exp(F_jj + Σ_{i∈s} F_ji)，
    在候选集合内归一化后对后验样本求平均（与 models/baselines.HyperTraPSThetaBaseline 一致）。
    """
    mats = [np.asarray(F, dtype=float) for F in samples]

    def _score(state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=float)
        cand = s == 0
        acc = np.zeros(len(s), dtype=float)
        for F in mats:
            logr = np.clip(F.diagonal() + F.dot(s), -30, 30)
            rate = np.where(cand, np.exp(logr), 0.0)
            tot = rate.sum()
            if tot > 0:
                acc += rate / tot
        if acc.sum() <= 0:
            acc = np.where(cand, 1.0, 0.0)
        return acc / max(acc.sum(), 1e-12)

    return _score


def two_stage_scorer(model) -> Scorer:
    def _score(state: np.ndarray) -> np.ndarray:
        v = np.asarray(model.predict_scores(state), dtype=float)
        v = np.where(np.asarray(state) == 0, v, 0.0)
        return v / max(v.sum(), 1e-12)
    return _score


# --------------------------------------------------------------------------------------
# 2. 训练/获取 Stage-2 模型（供缺少 default/stage2 的数据集使用）
# --------------------------------------------------------------------------------------

def ensure_two_stage_model(spec: "st.DatasetSpec", python_exe: Optional[str] = None,
                           epochs: int = 120, d_model: int = 256, n_layers: int = 3,
                           n_heads: int = 4, batch_size: int = 16, dropout: float = 0.30,
                           contrastive_weight: float = 0.05, mask_prob_pos: float = 0.5,
                           mask_prob_neg: float = 0.1, seed: int = 0,
                           force: bool = False) -> Dict[str, object]:
    """
    若 `default/stage2/model.pt` 不存在则在全量转移上训练一次（Stage 1 + Stage 2），
    超参与 `scripts_repr_evo/common.sh` 主线一致。返回 dict(model_path, elapsed_sec, status)。
    """
    exe = python_exe or sys.executable
    stage1_dir = os.path.join(spec.results_dir, "default", "stage1")
    stage2_dir = os.path.join(spec.results_dir, "default", "stage2")
    model_path = os.path.join(stage2_dir, "model.pt")
    if os.path.isfile(model_path) and not force:
        return {"status": "cached", "model_path": model_path, "elapsed_sec": None}

    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)
    common = ["--transitions", spec.transitions_path, "--arch", "transformer",
              "--d-model", str(d_model), "--n-layers", str(n_layers), "--n-heads", str(n_heads),
              "--dropout", str(dropout), "--batch-size", str(batch_size),
              "--epochs", str(epochs), "--seed", str(seed), "--no-meta"]
    t0 = time.time()
    subprocess.run([exe, os.path.join(MODELS_DIR, "train_state_pretrain.py"),
                    "--outdir", stage1_dir, "--mask-prob-pos", str(mask_prob_pos),
                    "--mask-prob-neg", str(mask_prob_neg),
                    "--contrastive-weight", str(contrastive_weight)] + common,
                   check=True, cwd=EVOAMR_ROOT)
    t_stage1 = time.time() - t0
    t1 = time.time()
    subprocess.run([exe, os.path.join(MODELS_DIR, "train_transition_two_stage.py"),
                    "--outdir", stage2_dir,
                    "--pretrained", os.path.join(stage1_dir, "stage1_model.pt")] + common,
                   check=True, cwd=EVOAMR_ROOT)
    t_stage2 = time.time() - t1
    return {"status": "trained", "model_path": model_path,
            "elapsed_sec": round(t_stage1 + t_stage2, 1),
            "stage1_sec": round(t_stage1, 1), "stage2_sec": round(t_stage2, 1)}


# --------------------------------------------------------------------------------------
# 3. 统一的前向游走：bubble / 转移图 / 主路径
# --------------------------------------------------------------------------------------

def bubble_from_scorer(scorer: Scorer, feature_names: Sequence[str], n_walks: int = 2000,
                       max_steps: Optional[int] = None, seed: int = 0) -> pd.DataFrame:
    """任意打分器（HT 或两阶段模型）在同一协议下的 bubble 矩阵（feature × step）。"""
    L = len(feature_names)
    steps = L if max_steps is None else min(L, max_steps)
    rng = np.random.default_rng(seed)
    mat = np.zeros((L, steps), dtype=float)
    for _ in range(n_walks):
        state = np.zeros(L, dtype=np.int8)
        for step in range(1, steps + 1):
            p = np.asarray(scorer(state), dtype=float)
            p[state == 1] = 0.0
            tot = p.sum()
            if tot <= 0:
                break
            pick = int(rng.choice(L, p=p / tot))
            mat[pick, step - 1] += 1.0
            state[pick] = 1
    mat /= max(1, n_walks)
    return pd.DataFrame(mat, index=list(feature_names),
                        columns=[f"step{i+1}" for i in range(steps)])


def transition_graph(scorer: Scorer, feature_names: Sequence[str], n_walks: int = 3000,
                     max_steps: Optional[int] = None, seed: int = 0,
                     min_edge_prob: float = 0.01) -> Dict[str, object]:
    """
    前向游走统计得到的超立方转移图（对应文献 Fig 7A/B）。

    节点 = 状态（用已获得特征索引的排序元组表示），分层 = 已获得特征数；
    边 = 单步获得某特征的转移，权重 = 该转移在全部游走中的出现频率。
    """
    L = len(feature_names)
    steps = L if max_steps is None else min(L, max_steps)
    rng = np.random.default_rng(seed)
    node_visits: Dict[Tuple[int, ...], float] = {(): 0.0}
    edges: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
    for _ in range(n_walks):
        state = np.zeros(L, dtype=np.int8)
        key: Tuple[int, ...] = ()
        node_visits[key] = node_visits.get(key, 0.0) + 1.0
        for _step in range(steps):
            p = np.asarray(scorer(state), dtype=float)
            p[state == 1] = 0.0
            tot = p.sum()
            if tot <= 0:
                break
            pick = int(rng.choice(L, p=p / tot))
            state[pick] = 1
            nxt = tuple(sorted(key + (pick,)))
            edges[(key, nxt)] = edges.get((key, nxt), 0.0) + 1.0
            node_visits[nxt] = node_visits.get(nxt, 0.0) + 1.0
            key = nxt
    edges = {k: v / n_walks for k, v in edges.items() if v / n_walks >= min_edge_prob}
    nodes = {k: v / n_walks for k, v in node_visits.items()}
    keep = {k for e in edges for k in e}
    nodes = {k: v for k, v in nodes.items() if k in keep or k == ()}
    return {"nodes": nodes, "edges": edges, "feature_names": list(feature_names),
            "n_walks": n_walks, "max_steps": steps}


def plot_transition_graph(graph: Dict[str, object], title: str, max_nodes_per_layer: int = 6,
                          annotate_layers: int = 4, save: Optional[str] = None):
    """把转移图画成分层图：x = 已获得特征数，边宽 ∝ 转移概率，节点大小 ∝ 访问频率。"""
    nodes: Dict[Tuple[int, ...], float] = graph["nodes"]          # type: ignore[assignment]
    edges: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = graph["edges"]  # type: ignore[assignment]
    names: List[str] = graph["feature_names"]                     # type: ignore[assignment]

    layers: Dict[int, List[Tuple[int, ...]]] = {}
    for key, w in nodes.items():
        layers.setdefault(len(key), []).append(key)
    pos: Dict[Tuple[int, ...], Tuple[float, float]] = {}
    for layer, keys in layers.items():
        keys = sorted(keys, key=lambda k: -nodes[k])[:max_nodes_per_layer]
        layers[layer] = keys
        for rank, key in enumerate(keys):
            pos[key] = (layer, rank - (len(keys) - 1) / 2.0)

    fig, ax = plt.subplots(figsize=(max(6.5, 0.62 * (max(layers) + 1)), 5.2))
    for (a, b), w in sorted(edges.items(), key=lambda kv: kv[1]):
        if a not in pos or b not in pos:
            continue
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        ax.plot([x0, x1], [y0, y1], color="#666666", alpha=min(0.9, 0.25 + 2.5 * w),
                lw=0.4 + 7.0 * w, zorder=1, solid_capstyle="round")
    xs = [pos[k][0] for k in pos]
    ys = [pos[k][1] for k in pos]
    sz = [40 + 900 * nodes[k] for k in pos]
    cs = [len(k) for k in pos]
    sc = ax.scatter(xs, ys, s=sz, c=cs, cmap="coolwarm", zorder=3, edgecolors="white", linewidths=0.6)
    for key in pos:
        if 0 < len(key) <= annotate_layers and nodes[key] > 0.02:
            ax.annotate(names[key[-1]][:12], pos[key], fontsize=6, ha="center", va="center",
                        xytext=(0, 9), textcoords="offset points", zorder=4)
    ax.set_xlabel("number of features acquired")
    ax.set_yticks([])
    ax.set_title(title + "\n(node size = visit frequency, edge width = transition probability)")
    ax.grid(axis="x", alpha=0.25)
    fig.colorbar(sc, ax=ax, label="layer (#acquired)")
    if save:
        fig.savefig(save)
    return fig


def top_pathways(scorer: Scorer, feature_names: Sequence[str], depth: int = 4,
                 n_walks: int = 3000, seed: int = 0, top_k: int = 10) -> pd.DataFrame:
    """最常见的前 `depth` 步获得路径及其概率（对应文献里"不同 pathway class"的讨论）。"""
    L = len(feature_names)
    rng = np.random.default_rng(seed)
    counts: Dict[Tuple[int, ...], int] = {}
    for _ in range(n_walks):
        state = np.zeros(L, dtype=np.int8)
        path: List[int] = []
        for _step in range(depth):
            p = np.asarray(scorer(state), dtype=float)
            p[state == 1] = 0.0
            tot = p.sum()
            if tot <= 0:
                break
            pick = int(rng.choice(L, p=p / tot))
            state[pick] = 1
            path.append(pick)
        key = tuple(path)
        counts[key] = counts.get(key, 0) + 1
    rows = [{"path": " -> ".join(feature_names[i] for i in k), "probability": v / n_walks,
             "path_idx": k} for k, v in counts.items()]
    return (pd.DataFrame(rows).sort_values("probability", ascending=False)
            .head(top_k).reset_index(drop=True))


def compare_pathways(paths_a: pd.DataFrame, paths_b: pd.DataFrame,
                     name_a: str = "HyperTraPS", name_b: str = "two-stage",
                     top_k: int = 10) -> Tuple[pd.DataFrame, Dict[str, float]]:
    a = paths_a.head(top_k)
    b = paths_b.head(top_k)
    set_a, set_b = set(a["path"]), set(b["path"])
    inter = set_a & set_b
    merged = (a[["path", "probability"]].rename(columns={"probability": name_a})
              .merge(b[["path", "probability"]].rename(columns={"probability": name_b}),
                     on="path", how="outer").fillna(0.0)
              .sort_values(name_a, ascending=False).reset_index(drop=True))
    stats = {
        "top_k": top_k,
        "n_shared_paths": len(inter),
        "jaccard": len(inter) / max(1, len(set_a | set_b)),
        "prob_mass_shared_%s" % name_a: float(a[a["path"].isin(inter)]["probability"].sum()),
        "prob_mass_shared_%s" % name_b: float(b[b["path"].isin(inter)]["probability"].sum()),
    }
    return merged, stats


# --------------------------------------------------------------------------------------
# 4. Figure 7C 式的后验获得动力学图
# --------------------------------------------------------------------------------------

def feature_color_map(feature_names: Sequence[str], hierarchy: Optional[pd.DataFrame] = None,
                      key: str = "drug_class") -> Dict[str, str]:
    """按药物类别（或其他层级字段）给特征上色，使基因位点与其对应药物在图中被颜色关联。"""
    if hierarchy is None or key not in hierarchy.columns:
        return {f: "#333333" for f in feature_names}
    mapping = dict(zip(hierarchy["feature_id"].astype(str), hierarchy[key].astype(str)))
    groups = sorted({mapping.get(f, "unknown") for f in feature_names})
    cmap = plt.get_cmap("tab20")
    palette = {g: cmap(i % 20) for i, g in enumerate(groups)}
    return {f: palette[mapping.get(f, "unknown")] for f in feature_names}


def observed_step_range(ts: "st.TransitionSet") -> pd.DataFrame:
    """每个特征在数据中实际被观测到获得的序位区间（用于 Fig 7C 的灰色未观测区）。"""
    lo = np.full(ts.L, np.nan)
    hi = np.full(ts.L, np.nan)
    depth = ts.frame["n_src_features"].values
    for row_i, g in enumerate(ts.gained):
        step = int(depth[row_i]) + 1
        for f in g:
            lo[f] = step if np.isnan(lo[f]) else min(lo[f], step)
            hi[f] = step if np.isnan(hi[f]) else max(hi[f], step)
    return pd.DataFrame({"feature": ts.feature_names, "first_observed_step": lo,
                         "last_observed_step": hi})


def plot_posterior_ordering(bubbles: Dict[str, pd.DataFrame], title: str,
                            feature_order: Optional[Sequence[str]] = None,
                            colors: Optional[Dict[str, str]] = None,
                            observed: Optional[pd.DataFrame] = None,
                            save: Optional[str] = None):
    """
    复现文献 Fig 7C：每个特征一行，横轴为"已获得特征数"，条形高度为该特征在该序位被获得的
    后验概率；多个来源（HyperTraPS / 两阶段模型 / 经验）并列成多列以便直接对照。
    行标签按药物类别着色；灰色区域标注数据中未观测到的获得区间。
    """
    names = list(bubbles)
    ref = bubbles[names[0]]
    order = list(feature_order) if feature_order is not None else list(ref.index)
    order = [f for f in order if f in ref.index]
    n_f = len(order)
    n_src = len(names)
    obs = observed.set_index("feature") if observed is not None else None

    fig, axes = plt.subplots(n_f, n_src, figsize=(3.6 * n_src, max(3.0, 0.30 * n_f)),
                             sharex=True, squeeze=False)
    steps = np.arange(1, ref.shape[1] + 1)
    vmax = max(float(np.nanmax(b.values)) for b in bubbles.values())
    for r, feat in enumerate(order):
        for c, src_name in enumerate(names):
            ax = axes[r][c]
            vals = bubbles[src_name].loc[feat].values if feat in bubbles[src_name].index \
                else np.zeros(len(steps))
            if obs is not None and feat in obs.index:
                lo, hi = obs.loc[feat, "first_observed_step"], obs.loc[feat, "last_observed_step"]
                if np.isfinite(lo):
                    ax.axvspan(0.5, lo - 0.5, color="#DDDDDD", zorder=0)
                    ax.axvspan(hi + 0.5, steps[-1] + 0.5, color="#DDDDDD", zorder=0)
            ax.bar(steps, vals, width=0.85,
                   color=(colors or {}).get(feat, "#2C6FA6"), zorder=2)
            ax.set_ylim(0, vmax * 1.05)
            ax.set_yticks([])
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.grid(False)
            if c == 0:
                ax.set_ylabel(feat, rotation=0, ha="right", va="center", fontsize=7,
                              color=(colors or {}).get(feat, "#333333"))
            if r == 0:
                ax.set_title(src_name, fontsize=10)
            if r == n_f - 1:
                ax.set_xlabel("number of features acquired")
    fig.suptitle(title, y=1.005, fontsize=11)
    fig.tight_layout(h_pad=0.05)
    if save:
        fig.savefig(save)
    return fig


# --------------------------------------------------------------------------------------
# 5. 定量一致性
# --------------------------------------------------------------------------------------

def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # 处理并列
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for gi, cnt in enumerate(counts):
        if cnt > 1:
            idx = np.flatnonzero(inv == gi)
            ranks[idx] = ranks[idx].mean()
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = _rank(a[ok]), _rank(b[ok])
    return float(np.corrcoef(ra, rb)[0, 1])


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon 散度（以 2 为底，取值 0~1）。"""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / np.maximum(y[mask], 1e-12))))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def compare_orderings(bubble_a: pd.DataFrame, bubble_b: pd.DataFrame,
                      name_a: str = "HyperTraPS", name_b: str = "two-stage") -> Tuple[pd.DataFrame, Dict[str, float]]:
    """逐特征比较两侧的期望获得次序与获得序位分布（JS 散度）。"""
    ea = st.expected_ordering(bubble_a).set_index("feature")["expected_order"]
    eb = st.expected_ordering(bubble_b).set_index("feature")["expected_order"]
    feats = [f for f in bubble_a.index if f in bubble_b.index]
    rows = []
    for f in feats:
        rows.append({
            "feature": f,
            f"order_{name_a}": ea.get(f, np.nan),
            f"order_{name_b}": eb.get(f, np.nan),
            "order_diff": (eb.get(f, np.nan) - ea.get(f, np.nan)),
            "js_divergence": js_divergence(bubble_a.loc[f].values, bubble_b.loc[f].values),
        })
    df = pd.DataFrame(rows)
    stats = {
        "spearman_ordering": spearman(df[f"order_{name_a}"].values, df[f"order_{name_b}"].values),
        "pearson_ordering": float(np.corrcoef(
            df[[f"order_{name_a}", f"order_{name_b}"]].dropna().T.values)[0, 1])
        if len(df.dropna()) > 2 else float("nan"),
        "mean_abs_order_diff": float(df["order_diff"].abs().mean()),
        "median_js_divergence": float(df["js_divergence"].median()),
    }
    return df.sort_values("order_diff", key=lambda s: s.abs(), ascending=False).reset_index(drop=True), stats


def plot_ordering_agreement(cmp_df: pd.DataFrame, stats: Dict[str, float],
                            name_a: str = "HyperTraPS", name_b: str = "two-stage",
                            title: str = "", save: Optional[str] = None):
    xa, xb = f"order_{name_a}", f"order_{name_b}"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].scatter(cmp_df[xa], cmp_df[xb], s=45, alpha=0.8, color="#4C72B0", edgecolors="white")
    lims = [min(cmp_df[xa].min(), cmp_df[xb].min()) - 0.5,
            max(cmp_df[xa].max(), cmp_df[xb].max()) + 0.5]
    axes[0].plot(lims, lims, "--", color="grey", lw=1)
    for _, r in cmp_df.head(10).iterrows():
        axes[0].annotate(str(r["feature"])[:14], (r[xa], r[xb]), fontsize=6.5,
                         xytext=(3, 3), textcoords="offset points")
    axes[0].set_xlabel(f"expected ordering — {name_a}")
    axes[0].set_ylabel(f"expected ordering — {name_b}")
    axes[0].set_title("Ordering agreement (Spearman = %.3f)" % stats.get("spearman_ordering", float("nan")))

    d = cmp_df.dropna(subset=["js_divergence"]).sort_values("js_divergence")
    axes[1].barh(range(len(d)), d["js_divergence"], color="#DD8452")
    axes[1].set_yticks(range(len(d)))
    axes[1].set_yticklabels(d["feature"], fontsize=6.5)
    axes[1].set_xlabel("JS divergence of acquisition-step distribution")
    axes[1].set_title("Per-feature disagreement\n(median = %.3f)" % stats.get("median_js_divergence", float("nan")))
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


def truncate_and_renormalize(bubble: pd.DataFrame, max_step: int) -> pd.DataFrame:
    """
    把 bubble 截断到前 `max_step` 个序位并逐特征重新归一化。

    这一步在"与经验观测比较"时是必需的：经验 bubble 的序位由数据中源状态的耐药深度决定
    （TB 数据平均只有 4~6 个已获得特征），而模型/HyperTraPS 的前向游走会一直走到 L 步，
    两者的序位尺度不同。截断到共同可观测区间后，期望次序与 JS 散度才具备可比性。
    """
    sub = bubble.iloc[:, :max(1, int(max_step))].copy()
    row = sub.sum(axis=1).values.reshape(-1, 1)
    vals = np.divide(sub.values, row, out=np.zeros_like(sub.values), where=row > 0)
    return pd.DataFrame(vals, index=sub.index, columns=sub.columns)


def empirical_max_step(bubble_empirical: pd.DataFrame, quantile: float = 1.0) -> int:
    """经验 bubble 中最后一个仍有概率质量的序位（默认取全部质量所覆盖的最大序位）。"""
    mass = bubble_empirical.sum(axis=0).values
    nz = np.flatnonzero(mass > 0)
    if len(nz) == 0:
        return bubble_empirical.shape[1]
    if quantile >= 1.0:
        return int(nz[-1] + 1)
    cum = np.cumsum(mass) / max(mass.sum(), 1e-12)
    return int(np.searchsorted(cum, quantile) + 1)


def cost_comparison(ht_meta: Dict[str, object], ts_meta: Dict[str, object]) -> pd.DataFrame:
    """墙钟成本对照（HyperTraPS MCMC vs 两阶段训练）。"""
    rows = [
        {"method": "HyperTraPS (MCMC, full data)", "status": ht_meta.get("status"),
         "elapsed_sec": ht_meta.get("elapsed_sec"),
         "params": json.dumps(ht_meta.get("ht_params"), ensure_ascii=False)},
        {"method": "Two-stage (Stage1+Stage2, full data)", "status": ts_meta.get("status"),
         "elapsed_sec": ts_meta.get("elapsed_sec"),
         "params": "stage1=%s s, stage2=%s s" % (ts_meta.get("stage1_sec"), ts_meta.get("stage2_sec"))},
    ]
    return pd.DataFrame(rows)
