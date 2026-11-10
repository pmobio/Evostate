#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_two_stage_inputs.py
==========================
把 ecoli2026 Route-B 的"演化终点横截面"数据集转换为 two-stage 模型
(train_state_pretrain / train_transition_two_stage / run_two_stage_eval) 可直接
消费的输入:

    <out>/<DRUG>/
        transitions/transitions.csv   # src_state,dest_state,n_gained,gained_features,n_lost,lost_features,group_id,transition_strategy
        feature_hierarchy.csv         # feature_id,layer,drug_class,mechanism,location,reversible
        samples.csv                   # sample_id,group_id,lineage_group,lineage_group_size,tree_leaf_order,n_positive_features,feature_set_name,source_dataset
        manifest.json

数据语义 (已逐项核实 derived/genotype_route_b/mutations_long.csv 与 fsepi_data):
  * 关键事实: 基因型只有**演化终点快照** (mutations_long 全 event_type=endpoint,
    每个 population 仅 1 个终点突变集合); 同一 strain_group 的 3 个样本是**平行重复板
    (实验上去偏差用)**, 不是时间序 —— 故不存在"祖先->终点之外"的基因型中间态。
    真实时间轴在**表型**里: morbidostat 每 ~4.2h 一个稀释周期记录每个 well 的
    IC50/生长速率 (derived/phenotype/phenotype_trajectories.csv, ~109 时间点)。
  * --temporal none (默认): 每个 population 构造为单条 transition
        祖先 background 初态  ->  终点 resistance 基因型  (irreversible, 全为 gain)
  * --temporal inferred (推荐, 由表型轨迹推断突变获得顺序): 用 4h IC50 轨迹把终点突变集
    拆成时间有序的获得序列 —— 轨迹决定"几步/何时", 突变按群体出现频次(高频=强驱动先固定)
    排序, 构造 祖先 -> 中间态 -> ... -> 终点 的多步 transition。详见 build_drug_inferred。
  * --temporal monotonic/full_temporal 旧版把"3 平行板"误当时间链, 与实验设计不符, 已废弃。
  * 推断假设 (均可在 build_drug_inferred 内替换):
      (1) 每个终态突变对应一次获得事件 (步数 M = 演化突变数);
      (2) 突变按群体频次降序获得 (高频=强驱动先固定);
      (3) 各步时间锚定到 IC50 轨迹的等累积耐药增量位置。
    这是**推断而非实测**的演化顺序, 仅利用真实 4h 表型时间轴作为时序骨架。

为什么要保留 background marker (默认 --background-mode marker):
  文献核心结论是 function-specific epistasis —— 缺失背景决定了走哪条 epistatic 路径。
  若 src 全 0, 则所有 population 初态完全相同, background 信息彻底丢失, 模型无法学到
  "不同背景 -> 不同突变组合"。把 background 编码为**贯穿 src/dest 的持久 marker
  特征** (1=该 lineage 起始于某缺失背景), 模型既能看到"来自哪个背景", 又不会把 background
  误判为"被获得的突变" (marker 在 src 已为 1, 不在 gain 候选里)。

  background 名 (如 acrA, pntB) 与突变基因列名大量重合, 故统一加前缀 `background__`
  命名空间, 避免与突变特征冲突。

用法:
  python build_two_stage_inputs.py --drugs MEC,NIT,TMP
  python build_two_stage_inputs.py --drugs ALL
  python build_two_stage_inputs.py --drugs MEC --background-mode none   # 消融: 忽略背景

时间线模式:
  # inferred (推荐): 由 4h 表型(IC50)轨迹推断突变获得顺序, 构造多步 transition
  python build_two_stage_inputs.py --drugs MEC --temporal inferred
  python build_two_stage_inputs.py --drugs MEC --temporal inferred --env-mode bits
  python build_two_stage_inputs.py --drugs MEC --temporal inferred --order-by name   # 对照: 字典序
  # monotonic/full_temporal (已废弃, 把 3 平行板误当时间链):
  python build_two_stage_inputs.py --drugs MEC --temporal monotonic
  python build_two_stage_inputs.py --drugs MEC --temporal full_temporal
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EVOAMR_DIR = os.path.dirname(os.path.dirname(HERE))
ENDPOINT_ROOT = os.path.join(EVOAMR_DIR, "data", "ecoli2026", "derived", "endpoint")
TWOSTAGE_ROOT = os.path.join(EVOAMR_DIR, "data", "ecoli2026", "derived", "two_stage")
PHENO_ROOT = os.path.join(EVOAMR_DIR, "data", "ecoli2026", "derived", "phenotype")
PHENO_CSV = os.path.join(PHENO_ROOT, "phenotype_populations.csv")
# 按真实时间线构造的有序 genotype transition (t1->t2->t3 跨 3 块连续板)。
# monotonic=仅增益(irreversible); full_temporal=增益+丢失(reversible, 双头)。
GENO_ROOT = os.path.join(EVOAMR_DIR, "data", "ecoli2026", "derived", "genotype_route_b")
# 从表型时间轴(AB 浓度轨迹, 109 时间点)提取的可量化"进化信息"协变量。
# 基因型突变矩阵无时间轴, 但 morbidostat 的 AB 浓度轨迹含真实连续时间轴;
# 这些汇总统计量刻画了每个 population 经历的"选择压力强度/演化程度"。
PHENO_COLS = ["initial_ic50", "final_ic50", "max_ic50", "fold_increase",
              "log10_ic50_auc", "duration_h", "n_timepoints",
              "initial_growth_rate", "final_time_h_paper"]

FEATURE_COLS = ["feature_id", "layer", "drug_class", "mechanism", "location", "reversible"]


def sanitize_bg(bg):
    """把背景标签 (可能含 Δ/特殊字符) 转成安全且不与突变基因列冲突的 feature_id。"""
    s = (bg.replace("Δ", "delta").replace("△", "delta")
           .replace("Φ", "phi").replace("Ω", "omega"))
    s = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in s).strip("_")
    s = s.strip("_")
    return f"background__{s}"


def align_pheno(pop_keys, pheno_pop, drug):
    """把表型时间轴协变量(109 时间点汇总 9 统计量)对齐到 pop_keys 指定的 sample。

    pop_keys: DataFrame, 须含 population_id, plate_id, well, replicate_index。
    返回 (env_bits, used): env_bits={population_id: [0/1,...]}, used=[pheno__*...]。
    对齐键 (plate_id, well, replicate_index) 在 endpoint/phenotype 两边一致。
    未提供 phenotype 文件或无可对齐列时返回 ({}, [])。
    """
    if pheno_pop is None or not os.path.isfile(PHENO_CSV):
        return {}, []
    sub = pheno_pop.copy()
    if "antibiotic" in sub.columns:
        sub = sub[sub["antibiotic"] == drug]
    sub["_jk"] = (sub["plate_id"].astype(str) + "|" + sub["well"].astype(str)
                  + "|" + sub["replicate_index"].astype(str))
    sub = sub.rename(columns={c: f"pheno__{c}" for c in PHENO_COLS if c in sub.columns})
    used = [f"pheno__{c}" for c in PHENO_COLS if f"pheno__{c}" in sub.columns]
    if not used:
        return {}, []
    sub = sub[["_jk"] + used].drop_duplicates("_jk")
    medians = {c: pd.to_numeric(sub[c], errors="coerce").median() for c in used}
    lut = {}
    for _, r in sub.iterrows():
        bits = []
        for c in used:
            v = pd.to_numeric(r[c], errors="coerce")
            m = medians[c]
            bits.append(1 if (pd.notna(v) and pd.notna(m) and v >= m) else 0)
        lut[r["_jk"]] = bits
    out = {}
    for _, r in pop_keys.iterrows():
        jk = f"{r['plate_id']}|{r['well']}|{r['replicate_index']}"
        out[r["population_id"]] = lut.get(jk, [0] * len(used))
    return out, used


def build_drug_temporal(drug, encoding, out_root, env_mode="none", pheno_pop=None):
    """消费 derived/genotype_route_b/<encoding>/ 中已按时间线构造的有序 transition。

    encoding: monotonic (仅增益, irreversible) | full_temporal (增益+丢失, reversible)。
    每个 trajectory (= strain_group, 如 MEC:A10_pntB) 由 3 块连续运行板的端点基因型
    构成 time_order 1->2->3 的有序快照; transitions.csv 的每条即 t->t+1 的获得/丢失。
    可选 --env-mode bits: 把表型时间轴协变量追加为 persistent 环境特征位(尾部)。
    """
    gdir = os.path.join(GENO_ROOT, encoding)
    tpath = os.path.join(gdir, "transitions.csv")
    hpath = os.path.join(gdir, "feature_hierarchy.csv")
    spath = os.path.join(gdir, "samples.csv")
    if not (os.path.isfile(tpath) and os.path.isfile(hpath) and os.path.isfile(spath)):
        print(f"[skip] 缺少 temporal({encoding}) 文件: {gdir}", file=sys.stderr)
        return None

    trans = pd.read_csv(tpath, dtype=str, keep_default_na=False)
    hier = pd.read_csv(hpath, dtype=str, keep_default_na=False)
    samples = pd.read_csv(spath, dtype=str, keep_default_na=False)

    # 仅保留当前药物
    if "group_id" in trans.columns:
        trans = trans[trans["group_id"].str.startswith(f"{drug}:")].copy()
    if "antibiotic" in samples.columns:
        samples = samples[samples["antibiotic"] == drug].copy()
    elif "group_id" in samples.columns:
        samples = samples[samples["group_id"].str.startswith(f"{drug}:")].copy()
    if len(trans) == 0:
        print(f"[skip] {drug}: temporal({encoding}) 无 transition", file=sys.stderr)
        return None

    feat_ids = hier["feature_id"].tolist()
    L0 = len(feat_ids)

    # 可选: 融表型时间轴 env 位 (persistent, 追加到 bitstring 尾与 hierarchy)
    env_fids = []
    env_bits = {}
    if env_mode == "bits" and pheno_pop is not None:
        # temporal samples.csv: population_id=sample_id, replicate_index=replicate
        pop_keys = samples.rename(columns={"sample_id": "population_id", "replicate": "replicate_index"})[
            ["population_id", "plate_id", "well", "replicate_index"]].drop_duplicates()
        env_bits, used = align_pheno(pop_keys, pheno_pop, drug)
        env_fids = [f"env__{c.replace('pheno__', '')}" for c in used]
        add = pd.DataFrame([{
            "feature_id": f, "layer": "environment",
            "drug_class": "phenotype_trajectory", "mechanism": "selection_pressure",
            "location": "", "reversible": "0"} for f in env_fids])
        hier = pd.concat([hier, add], ignore_index=True)

    if env_fids:
        new_rows = []
        for _, r in trans.iterrows():
            src = list(r["src_state"]) + [str(x) for x in env_bits.get(r["src_sample_id"], [0] * len(env_fids))]
            dst = list(r["dest_state"]) + [str(x) for x in env_bits.get(r["dest_sample_id"], [0] * len(env_fids))]
            nr = r.to_dict()
            nr["src_state"] = "".join(src)
            nr["dest_state"] = "".join(dst)
            new_rows.append(nr)
        trans = pd.DataFrame(new_rows)
    L = L0 + len(env_fids)

    # 写出 transitions
    tdir = os.path.join(out_root, drug, "transitions")
    os.makedirs(tdir, exist_ok=True)
    trans.to_csv(os.path.join(tdir, "transitions.csv"), index=False)
    hier.to_csv(os.path.join(out_root, drug, "feature_hierarchy.csv"), index=False)

    # 写出 samples (统一 schema, 便于下游 collector)
    sg_col = "strain_group_id" if "strain_group_id" in samples.columns else (
        "group_id" if "group_id" in samples.columns else None)
    traj_col = "trajectory_id" if "trajectory_id" in samples.columns else sg_col
    sizes = samples.groupby(sg_col).size().to_dict() if sg_col else {}
    out_rows = []
    for _, r in samples.iterrows():
        gid = r[sg_col] if sg_col else r.get("sample_id", "")
        lg = r[traj_col] if traj_col else gid
        out_rows.append({
            "sample_id": r["sample_id"],
            "group_id": gid,
            "lineage_group": lg,
            "lineage_group_size": sizes.get(gid, 1) if sg_col else 1,
            "tree_leaf_order": r.get("time_order", "1"),
            "n_positive_features": r.get("n_positive_features", ""),
            "feature_set_name": drug,
            "source_dataset": f"ecoli2026_route_b_{encoding}",
        })
    samples_df = pd.DataFrame(out_rows, columns=[
        "sample_id", "group_id", "lineage_group", "lineage_group_size",
        "tree_leaf_order", "n_positive_features", "feature_set_name", "source_dataset"])
    samples_df.to_csv(os.path.join(out_root, drug, "samples.csv"), index=False)

    # phenotype covariates (分析/对齐用)
    pheno_path = None
    if env_fids:
        bits_df = pd.DataFrame([
            {"sample_id": k, **{f: int(v[i]) for i, f in enumerate(env_fids)}}
            for k, v in env_bits.items()])
        cov = samples[["sample_id", "group_id"]].drop_duplicates().merge(bits_df, on="sample_id", how="left")
        pheno_path = os.path.join(out_root, drug, "phenotype_covariates.csv")
        cov.to_csv(pheno_path, index=False)

    manifest = {
        "drug": drug,
        "temporal_encoding": encoding,
        "n_transitions": int(len(trans)),
        "n_features": int(L),
        "n_gene_features": int(L0),
        "n_environment_features": int(len(env_fids)),
        "env_mode": env_mode,
        "transition_strategy": "irreversible" if encoding == "monotonic" else "full_temporal",
        "files": {
            "transitions": os.path.join(tdir, "transitions.csv"),
            "feature_hierarchy": os.path.join(out_root, drug, "feature_hierarchy.csv"),
            "samples": os.path.join(out_root, drug, "samples.csv"),
        },
    }
    if pheno_path:
        manifest["files"]["phenotype_covariates"] = pheno_path
    with open(os.path.join(out_root, drug, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    env_msg = f"env_bits={len(env_fids)}" if env_fids else "env_bits=0"
    print(f"[ok] {drug} (temporal={encoding}): transitions={len(trans)} features={L} "
          f"(genes={L0}, env={len(env_fids)}) {env_msg}")
    return manifest


def infer_step_times(time_h, ic50, M):
    """把一个 population 的 4h IC50 轨迹切成 M 步, 返回每步的终点时间(time_h)。

    以 IC50 从初值到终值的"等累积耐药增量"位置锚定各步时间, 即第 j 步落在
    累积耐药达到总增益 j/M 处。无有效轨迹(点<2 或无上升)返回 None。
    """
    th = np.array(time_h, dtype=float)
    ic = np.array(ic50, dtype=float)
    mask = ~np.isnan(ic)
    th, ic = th[mask], ic[mask]
    if len(ic) < 2:
        return None
    o = np.argsort(th)
    th, ic = th[o], ic[o]
    base, top = ic[0], ic[-1]
    if top <= base:
        return None
    total = top - base
    out = []
    for j in range(1, M + 1):
        target = base + total * (j / M)
        pos = int(np.searchsorted(ic, target))
        pos = min(pos, len(th) - 1)
        out.append(float(th[pos]))
    return out


def build_drug_inferred(drug, endpoint_root, out_root, background_mode,
                        pheno_pop=None, pheno_traj=None, env_mode="none",
                        order_by="frequency"):
    """由 4h 表型(IC50)轨迹推断每个 population 的突变获得顺序, 构造多步 transition。

    基因型只有终点快照, 真实时间轴在表型轨迹 (pheno_traj: time_h, ic50)。构造规则
    (均为可调启发式, 见模块 docstring):
      1) 步数 = 该 population 的演化突变数 M; 各步时间锚定到 IC50 等累积耐药增量位置。
      2) 突变按群体出现频次降序排序 (高频=强驱动, 先被固定) -> 获得顺序。
      3) state_0 = background 初态; 逐步加入排序后的突变 -> 多步 irreversible transition。
    产物 schema (transitions/feature_hierarchy/samples) 与模型 loader 完全一致。
    额外写出 transition_timings.csv 记录每步的 (src/dest)_time_h, 供校验"时间序"用。
    可选 --env-mode bits: 追加表型时间轴 env 持久位 (persistent)。
    """
    edir = os.path.join(endpoint_root, drug)
    pm_path = os.path.join(edir, "population_matrix.csv")
    pop_path = os.path.join(edir, "populations.csv")
    if not (os.path.isfile(pm_path) and os.path.isfile(pop_path)):
        print(f"[skip] 缺少 endpoint 文件: {edir}", file=sys.stderr)
        return None

    pm = pd.read_csv(pm_path, dtype=str, keep_default_na=False)
    pop = pd.read_csv(pop_path, dtype=str, keep_default_na=False)
    if "population_id" in pm.columns:
        pm = pm.set_index("population_id")
    pm = pm.astype(int)
    gene_cols = list(pm.columns)

    if "background" not in pop.columns:
        raise ValueError(f"{pop_path} 缺少 background 列")
    pid_col = "population_id" if "population_id" in pop.columns else pop.columns[0]
    sg_col = "strain_group_id" if "strain_group_id" in pop.columns else None

    pm_ids = set(pm.index.tolist())
    pop = pop[pop[pid_col].isin(pm_ids)].copy()
    pop = pop.set_index(pid_col, drop=False)

    # env 协变量对齐 (同 build_drug)
    pheno_cols_used = []
    if pheno_pop is not None and os.path.isfile(PHENO_CSV):
        pop = pop.reset_index(drop=True)
        pop = pop.drop(columns=[c for c in PHENO_COLS if c in pop.columns], errors="ignore")
        pop["_jk"] = (pop["plate_id"].astype(str) + "|" + pop["well"].astype(str)
                      + "|" + pop["replicate_index"].astype(str))
        sub = pheno_pop.copy()
        if "antibiotic" in pheno_pop.columns:
            sub = sub[sub["antibiotic"] == drug]
        sub["_jk"] = (sub["plate_id"].astype(str) + "|" + sub["well"].astype(str)
                      + "|" + sub["replicate_index"].astype(str))
        sub = sub.rename(columns={c: f"pheno__{c}" for c in PHENO_COLS if c in sub.columns})
        keep = ["_jk"] + [f"pheno__{c}" for c in PHENO_COLS if f"pheno__{c}" in sub.columns]
        sub = sub[keep].drop_duplicates("_jk")
        pop = pop.merge(sub, on="_jk", how="left").drop(columns=["_jk"])
        pop = pop.set_index(pid_col, drop=False)
        pheno_cols_used = [f"pheno__{c}" for c in PHENO_COLS if f"pheno__{c}" in pop.columns]

    bg_labels = sorted({b for b in pop["background"].tolist() if b})
    bg_markers = [sanitize_bg(b) for b in bg_labels]
    features = list(bg_markers) + list(gene_cols)
    env_fids, env_bitmap = [], {}
    if env_mode == "bits" and pheno_cols_used:
        for col in pheno_cols_used:
            env_fids.append(f"env__{col.replace('pheno__', '')}")
        medians = {col: pd.to_numeric(pop[col], errors="coerce").median() for col in pheno_cols_used}
        for pid in pm.index.tolist():
            bits = []
            for col in pheno_cols_used:
                v = pd.to_numeric(pop.loc[pid, col], errors="coerce")
                m = medians[col]
                bits.append(1 if (pd.notna(v) and pd.notna(m) and v >= m) else 0)
            env_bitmap[pid] = bits
        features = features + env_fids
    if len(features) != len(set(features)):
        seen, dups = set(), []
        for f in features:
            if f in seen:
                dups.append(f)
            seen.add(f)
        raise ValueError(f"feature 命名冲突: {sorted(set(dups))}")

    bg_to_marker = dict(zip(bg_labels, bg_markers))
    L = len(features)
    feat_idx = {f: i for i, f in enumerate(features)}

    # 突变群体频次 (驱动优先排序); order_by=name 时退化为字典序
    if order_by == "frequency":
        freq = {g: int((pm[g] == 1).sum()) for g in gene_cols}
        keyfn = lambda g: (-freq.get(g, 0), g)
    else:
        keyfn = lambda g: (0, g)

    # 表型轨迹索引: (plate_id, well, replicate_index) -> (time_h, ic50)
    traj_lut = {}
    if pheno_traj is not None and "ic50" in pheno_traj.columns:
        tr = pheno_traj.copy()
        if "antibiotic" in tr.columns:
            tr = tr[tr["antibiotic"] == drug]
        tr["time_h"] = pd.to_numeric(tr["time_h"], errors="coerce")
        tr["ic50"] = pd.to_numeric(tr["ic50"], errors="coerce")
        tr = tr.dropna(subset=["time_h", "ic50"])
        for (pl, wl, rp), grp in tr.groupby(["plate_id", "well", "replicate_index"]):
            grp = grp.sort_values("time_h")
            traj_lut[(str(pl), str(wl), str(rp))] = (grp["time_h"].tolist(), grp["ic50"].tolist())

    rows, samples, timings = [], [], []
    n_gained_total = 0
    for pid in pm.index.tolist():
        bg = pop.loc[pid, "background"] if pid in pop.index else ""
        sg = pop.loc[pid, sg_col] if (sg_col and pid in pop.index) else pid
        bg_marker = bg_to_marker.get(bg) if (background_mode == "marker" and bg) else None

        evolved = [g for g in gene_cols if int(pm.loc[pid, g]) == 1]
        if not evolved:
            continue  # 无演化突变, 跳过
        evolved_ordered = sorted(evolved, key=keyfn)
        M = len(evolved_ordered)

        # 时间锚定 (真实 4h 表型轨迹)
        jk = (str(pop.loc[pid, "plate_id"]) if pid in pop.index else None,
              str(pop.loc[pid, "well"]) if pid in pop.index else None,
              str(pop.loc[pid, "replicate_index"]) if pid in pop.index else None)
        th_t, ic_t = traj_lut.get(jk, (None, None))
        step_times = infer_step_times(th_t, ic_t, M) if th_t is not None else None

        # 状态链: background 初态 -> 逐步加入排序突变
        state = np.zeros(L, dtype=np.int8)
        if bg_marker is not None:
            state[feat_idx[bg_marker]] = 1
        if env_fids:
            for k, fid in enumerate(env_fids):
                state[feat_idx[fid]] = env_bitmap.get(pid, [0] * len(env_fids))[k]
        prev_time = 0.0
        for j, g in enumerate(evolved_ordered, start=1):
            nxt = state.copy()
            nxt[feat_idx[g]] = 1
            gained = [features[feat_idx[g]]]
            rows.append({
                "src_state": "".join(str(int(b)) for b in state),
                "dest_state": "".join(str(int(b)) for b in nxt),
                "n_gained": 1,
                "gained_features": ";".join(gained),
                "n_lost": 0,
                "lost_features": "",
                "group_id": sg,
                "transition_strategy": "irreversible",
            })
            n_gained_total += 1
            dt = step_times[j - 1] if step_times else float("nan")
            timings.append({
                "population_id": pid, "group_id": sg, "step": j,
                "gained_feature": gained[0], "src_time_h": prev_time,
                "dest_time_h": dt,
            })
            prev_time = dt if step_times else float("nan")
            state = nxt

        samples.append({
            "sample_id": pid, "group_id": sg, "lineage_group": sg,
            "lineage_group_size": 1, "tree_leaf_order": M,
            "n_positive_features": M, "feature_set_name": drug,
            "source_dataset": "ecoli2026_route_b_inferred",
        })

    if not rows:
        print(f"[skip] {drug}: inferred 无有效 transition", file=sys.stderr)
        return None

    trans_df = pd.DataFrame(rows, columns=[
        "src_state", "dest_state", "n_gained", "gained_features",
        "n_lost", "lost_features", "group_id", "transition_strategy"])
    samples_df = pd.DataFrame(samples, columns=[
        "sample_id", "group_id", "lineage_group", "lineage_group_size",
        "tree_leaf_order", "n_positive_features", "feature_set_name", "source_dataset"])
    hier_rows = []
    for f in features:
        if f.startswith("background__"):
            hier_rows.append({"feature_id": f, "layer": "ancestral_background",
                              "drug_class": "background", "mechanism": "ancestral_deletion",
                              "location": "", "reversible": "0"})
        elif f.startswith("env__"):
            hier_rows.append({"feature_id": f, "layer": "environment",
                              "drug_class": "phenotype_trajectory", "mechanism": "selection_pressure",
                              "location": "", "reversible": "0"})
        else:
            hier_rows.append({"feature_id": f, "layer": "mutation_marker",
                              "drug_class": drug, "mechanism": "",
                              "location": "", "reversible": "0"})
    hier_df = pd.DataFrame(hier_rows, columns=FEATURE_COLS)

    out_dir = os.path.join(out_root, drug)
    tdir = os.path.join(out_dir, "transitions")
    os.makedirs(tdir, exist_ok=True)
    trans_df.to_csv(os.path.join(tdir, "transitions.csv"), index=False)
    hier_df.to_csv(os.path.join(out_dir, "feature_hierarchy.csv"), index=False)
    samples_df.to_csv(os.path.join(out_dir, "samples.csv"), index=False)
    timings_df = pd.DataFrame(timings)
    timings_path = os.path.join(out_dir, "transition_timings.csv")
    timings_df.to_csv(timings_path, index=False)

    pheno_path = None
    if pheno_cols_used:
        cov = pop[["population_id", "strain_group_id"] + pheno_cols_used].copy()
        pheno_path = os.path.join(out_dir, "phenotype_covariates.csv")
        cov.to_csv(pheno_path, index=False)
    if sg_col:
        sizes = samples_df.groupby("group_id").size().to_dict()
        samples_df["lineage_group_size"] = samples_df["group_id"].map(sizes)
        samples_df.to_csv(os.path.join(out_dir, "samples.csv"), index=False)

    n_with_traj = int(timings_df["dest_time_h"].notna().sum()) if len(timings_df) else 0
    manifest = {
        "drug": drug, "mode": "inferred", "order_by": order_by,
        "background_mode": background_mode, "env_mode": env_mode,
        "n_populations": int(len(samples_df)), "n_transitions": int(len(trans_df)),
        "n_features": int(L), "n_background_markers": int(len(bg_markers)),
        "n_gene_features": int(len(gene_cols)), "n_gained_total": int(n_gained_total),
        "n_environment_features": int(len(env_fids)),
        "population_id_time_anchored": n_with_traj,
        "transition_strategy": "irreversible",
        "files": {
            "transitions": os.path.join(tdir, "transitions.csv"),
            "feature_hierarchy": os.path.join(out_dir, "feature_hierarchy.csv"),
            "samples": os.path.join(out_dir, "samples.csv"),
            "transition_timings": timings_path,
        },
    }
    if pheno_path:
        manifest["files"]["phenotype_covariates"] = pheno_path
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    traj_msg = f"time_anchored={n_with_traj}/{len(samples_df)}" if n_with_traj else "time_anchored=0(no_traj)"
    print(f"[ok] {drug} (inferred, order={order_by}): pops={len(samples_df)} "
          f"transitions={len(trans_df)} features={L} (bg={len(bg_markers)},genes={len(gene_cols)}) "
          f"{traj_msg}\n       -> {out_dir}")
    return manifest


def build_drug(drug, endpoint_root, out_root, background_mode, pheno_pop=None, env_mode="none", rename_map=None):
    edir = os.path.join(endpoint_root, drug)
    pm_path = os.path.join(edir, "population_matrix.csv")
    pop_path = os.path.join(edir, "populations.csv")
    if not (os.path.isfile(pm_path) and os.path.isfile(pop_path)):
        print(f"[skip] 缺少 endpoint 文件: {edir}", file=sys.stderr)
        return None

    pm = pd.read_csv(pm_path, dtype=str, keep_default_na=False)
    pop = pd.read_csv(pop_path, dtype=str, keep_default_na=False)

    if "population_id" in pm.columns:
        pm = pm.set_index("population_id")
    pm = pm.astype(int)

    # 突变基因特征 (列名)
    gene_cols = list(pm.columns)
    # 背景标签 (来自 populations.csv 的 background 列)
    if "background" not in pop.columns:
        raise ValueError(f"{pop_path} 缺少 background 列")
    # population_id / strain_group_id
    pid_col = "population_id" if "population_id" in pop.columns else pop.columns[0]
    sg_col = "strain_group_id" if "strain_group_id" in pop.columns else None

    # 仅保留 endpoint 里实际出现的 population
    pm_ids = set(pm.index.tolist())
    pop = pop[pop[pid_col].isin(pm_ids)].copy()
    pop = pop.set_index(pid_col, drop=False)

    # ---- 融入表型时间轴协变量(真实进化信息) ----
    # AB 浓度轨迹(derived/phenotype, 109 时间点)才是真正的演化时间轴; 这里提取其
    # 汇总统计量作为每个 population 的"选择压力强度/演化程度"量化, 写入
    # phenotype_covariates.csv 供分析/分层/未来模型扩展。基因型本身仍只有终点,
    # 故 transition 的 src/dest/gained 逻辑不变, 仅追加协变量元数据。
    # 注意: 两数据集 population_id 编码不同(endpoint=17221_19353_A10_pntB,
    # phenotype=MEC:21721:A10_R6), 但 (plate_id, well, replicate_index) 在两边一致, 用作对齐键。
    pheno_cols_used = []
    if pheno_pop is not None and os.path.isfile(PHENO_CSV):
        # endpoint populations.csv 已自带若干同名表型字段; 为避免 merge 冲突,
        # 统一从 phenotype 取全部协变量并加 "pheno__" 前缀, 丢弃 pop 中同名裸列,
        # 保证协变量来源一致、无歧义。
        pop = pop.reset_index(drop=True)
        pop = pop.drop(columns=[c for c in PHENO_COLS if c in pop.columns], errors="ignore")
        pop["_jk"] = pop["plate_id"].astype(str) + "|" + pop["well"].astype(str) + "|" + pop["replicate_index"].astype(str)
        sub = pheno_pop
        if "antibiotic" in pheno_pop.columns:
            sub = pheno_pop[pheno_pop["antibiotic"] == drug]
        sub = sub.copy()
        sub["_jk"] = sub["plate_id"].astype(str) + "|" + sub["well"].astype(str) + "|" + sub["replicate_index"].astype(str)
        sub = sub.rename(columns={c: f"pheno__{c}" for c in PHENO_COLS if c in sub.columns})
        keep = ["_jk"] + [f"pheno__{c}" for c in PHENO_COLS if f"pheno__{c}" in sub.columns]
        sub = sub[keep].drop_duplicates("_jk")
        pop = pop.merge(sub, on="_jk", how="left").drop(columns=["_jk"])
        pop = pop.set_index(pid_col, drop=False)
        pheno_cols_used = [f"pheno__{c}" for c in PHENO_COLS if f"pheno__{c}" in pop.columns]

    bg_labels = sorted({b for b in pop["background"].tolist() if b})
    bg_markers = [sanitize_bg(b) for b in bg_labels]
    # background 与突变基因列名去重保护 (理论上已靠前缀避免)
    features = list(bg_markers) + list(gene_cols)
    # 环境特征位 (--env-mode bits): 把连续表型协变量按 drug 内中位数二值化为 persistent
    # 状态位, 让 two-stage 模型直接感知"该 lineage 经历的选择压力/演化程度"这一真实时间轴信息。
    env_fids = []
    env_bitmap = {}
    if env_mode == "bits" and pheno_cols_used:
        for col in pheno_cols_used:
            env_fids.append(f"env__{col.replace('pheno__', '')}")
        medians = {col: pd.to_numeric(pop[col], errors="coerce").median() for col in pheno_cols_used}
        for pid in pm.index.tolist():
            bits = []
            for col in pheno_cols_used:
                v = pd.to_numeric(pop.loc[pid, col], errors="coerce")
                m = medians[col]
                bits.append(1 if (pd.notna(v) and pd.notna(m) and v >= m) else 0)
            env_bitmap[pid] = bits
        features = features + env_fids
    # 健壮性: 命名冲突检查
    dup = len(features) != len(set(features))
    if dup:
        seen, dups = set(), []
        for f in features:
            if f in seen:
                dups.append(f)
            seen.add(f)
        raise ValueError(f"feature 命名冲突: {sorted(set(dups))}")

    bg_to_marker = dict(zip(bg_labels, bg_markers))
    L = len(features)
    feat_idx = {f: i for i, f in enumerate(features)}

    rows = []
    samples = []
    n_gained_total = 0
    for pid in pm.index.tolist():
        bg = pop.loc[pid, "background"] if pid in pop.index else ""
        sg = pop.loc[pid, sg_col] if (sg_col and pid in pop.index) else pid
        bg_marker = bg_to_marker.get(bg) if (background_mode == "marker" and bg) else None

        src = np.zeros(L, dtype=np.int8)
        if bg_marker is not None:
            src[feat_idx[bg_marker]] = 1
        if env_fids:
            for k, fid in enumerate(env_fids):
                src[feat_idx[fid]] = env_bitmap.get(pid, [0] * len(env_fids))[k]
        dest = src.copy()
        gained_idx = []
        for g in gene_cols:
            if int(pm.loc[pid, g]) == 1:
                dest[feat_idx[g]] = 1
                gained_idx.append(feat_idx[g])
        gained_names = [features[i] for i in gained_idx]
        n_gained = len(gained_names)
        n_gained_total += n_gained

        rows.append({
            "src_state": "".join(str(int(b)) for b in src),
            "dest_state": "".join(str(int(b)) for b in dest),
            "n_gained": n_gained,
            "gained_features": ";".join(gained_names),
            "n_lost": 0,
            "lost_features": "",
            "group_id": sg,
            "transition_strategy": "irreversible",
        })
        samples.append({
            "sample_id": pid,
            "group_id": sg,
            "lineage_group": sg,
            "lineage_group_size": 1,
            "tree_leaf_order": 1,
            "n_positive_features": n_gained,
            "feature_set_name": drug,
            "source_dataset": "ecoli2026_route_b_endpoint",
        })

    trans_df = pd.DataFrame(rows, columns=[
        "src_state", "dest_state", "n_gained", "gained_features",
        "n_lost", "lost_features", "group_id", "transition_strategy"])
    samples_df = pd.DataFrame(samples, columns=[
        "sample_id", "group_id", "lineage_group", "lineage_group_size",
        "tree_leaf_order", "n_positive_features", "feature_set_name", "source_dataset"])

    # feature_hierarchy: 行序 == 比特序
    hier_rows = []
    for f in features:
        if f.startswith("background__"):
            hier_rows.append({
                "feature_id": f, "layer": "ancestral_background",
                "drug_class": "background", "mechanism": "ancestral_deletion",
                "location": "", "reversible": "0"})
        elif f.startswith("env__"):
            hier_rows.append({
                "feature_id": f, "layer": "environment",
                "drug_class": "phenotype_trajectory", "mechanism": "selection_pressure",
                "location": "", "reversible": "0"})
        else:
            hier_rows.append({
                "feature_id": f, "layer": "mutation_marker",
                "drug_class": drug, "mechanism": "",
                "location": "", "reversible": "0"})
    hier_df = pd.DataFrame(hier_rows, columns=FEATURE_COLS)

    out_dir = os.path.join(out_root, drug)
    tdir = os.path.join(out_dir, "transitions")
    os.makedirs(tdir, exist_ok=True)
    trans_df.to_csv(os.path.join(tdir, "transitions.csv"), index=False)
    hier_df.to_csv(os.path.join(out_dir, "feature_hierarchy.csv"), index=False)
    samples_df.to_csv(os.path.join(out_dir, "samples.csv"), index=False)

    # 表型协变量(真实时间轴的量化) -> 单独文件, 供分析/分层/未来模型扩展。
    # 不写入 transitions/feature_hierarchy, 以避免破坏 two-stage loader 的 schema。
    pheno_path = None
    if pheno_cols_used:
        cov = pop[["population_id", "strain_group_id"] + pheno_cols_used].copy()
        pheno_path = os.path.join(out_dir, "phenotype_covariates.csv")
        cov.to_csv(pheno_path, index=False)

    # lineage_group_size 回填 (同一 strain_group 的 replicate 数)
    if sg_col:
        sizes = samples_df.groupby("group_id").size().to_dict()
        samples_df["lineage_group_size"] = samples_df["group_id"].map(sizes)
        samples_df.to_csv(os.path.join(out_dir, "samples.csv"), index=False)

    manifest = {
        "drug": drug,
        "background_mode": background_mode,
        "n_populations": int(len(trans_df)),
        "n_features": int(L),
        "n_background_markers": int(len(bg_markers)),
        "n_gene_features": int(len(gene_cols)),
        "n_transitions": int(len(trans_df)),
        "n_gained_total": int(n_gained_total),
        "env_mode": env_mode,
        "n_environment_features": int(len(env_fids)),
        "phenotype_covariates": pheno_path,
        "phenotype_covariate_fields": pheno_cols_used,
        "backgrounds": bg_labels,
        "transition_strategy": "irreversible",
        "files": {
            "transitions": os.path.join(tdir, "transitions.csv"),
            "feature_hierarchy": os.path.join(out_dir, "feature_hierarchy.csv"),
            "samples": os.path.join(out_dir, "samples.csv"),
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    cov_msg = f"covariates={len(pheno_cols_used)}" if pheno_cols_used else "covariates=0(none)"
    env_msg = f"env_bits={len(env_fids)}" if env_fids else "env_bits=0"
    print(f"[ok] {drug}: populations={len(trans_df)} features={L} "
          f"(bg={len(bg_markers)}, genes={len(gene_cols)}) gained_total={n_gained_total} {cov_msg} {env_msg}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drugs", default="MEC,NIT,TMP",
                    help="逗号分隔; 支持 ALL/MEC/NIT/TMP")
    ap.add_argument("--endpoint-root", default=ENDPOINT_ROOT)
    ap.add_argument("--out-root", default=TWOSTAGE_ROOT)
    ap.add_argument("--background-mode", choices=["marker", "none"], default="marker",
                    help="marker=把 background 编码为贯穿 src/dest 的持久 marker 特征; "
                         "none=src 全 0 (忽略背景, 消融对照)")
    ap.add_argument("--temporal", choices=["none", "monotonic", "full_temporal", "inferred"], default="none",
                    help="none=终点横截面(祖先->终点, 含 background marker); "
                         "monotonic/full_temporal=旧版(把 3 平行板误当时间链, 已废弃); "
                         "inferred=由 4h 表型(IC50)轨迹推断突变获得顺序, 构造多步 transition(推荐)")
    ap.add_argument("--order-by", choices=["frequency", "name"], default="frequency",
                    help="inferred 模式下突变获得顺序的排序键: frequency=按群体出现频次降序(驱动先固定); "
                         "name=字典序(可复现对照)")
    ap.add_argument("--env-mode", choices=["none", "bits"], default="none",
                    help="none=纯基因型 transition; bits=把表型时间轴协变量二值化为 persistent "
                         "环境特征位(让 two-stage 模型直接感知选择压力/演化程度这一真实时间轴信息)")
    args = ap.parse_args()
    # 输出目录后缀: temporal 编码 + env 位, 与 baseline 并存可对比
    suffix = ""
    if args.temporal != "none":
        suffix += f"_temporal_{args.temporal}"
    if args.env_mode == "bits":
        suffix += "_env"
    out_root = args.out_root.rstrip("/") + suffix

    drugs = [d.strip() for d in args.drugs.split(",") if d.strip()]
    os.makedirs(out_root, exist_ok=True)
    pheno_pop = None
    if os.path.isfile(PHENO_CSV):
        pheno_pop = pd.read_csv(PHENO_CSV, dtype=str, keep_default_na=False)
    # 4h 表型轨迹 (真实时间轴), 仅 inferred 模式需要
    pheno_traj = None
    if args.temporal == "inferred":
        pt_path = os.path.join(PHENO_ROOT, "phenotype_trajectories.csv")
        if os.path.isfile(pt_path):
            pheno_traj = pd.read_csv(pt_path, dtype=str, keep_default_na=False)
        else:
            print("[warn] 缺少 phenotype_trajectories.csv, inferred 将退化为无时间锚定", file=sys.stderr)
    manifests = {}
    for d in drugs:
        if args.temporal == "none":
            m = build_drug(d, args.endpoint_root, out_root, args.background_mode,
                           pheno_pop=pheno_pop, env_mode=args.env_mode)
        elif args.temporal == "inferred":
            m = build_drug_inferred(d, args.endpoint_root, out_root, args.background_mode,
                                    pheno_pop=pheno_pop, pheno_traj=pheno_traj,
                                    env_mode=args.env_mode, order_by=args.order_by)
        else:
            m = build_drug_temporal(d, args.temporal, out_root,
                                    env_mode=args.env_mode, pheno_pop=pheno_pop)
        if m:
            manifests[d] = m
    if manifests:
        with open(os.path.join(out_root, "manifest.json"), "w") as f:
            json.dump(manifests, f, indent=2, ensure_ascii=False)
        print(f"[done] 已写出 {len(manifests)} 个药物数据集 -> {out_root}")
    else:
        print("[warn] 没有成功构造任何数据集", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
