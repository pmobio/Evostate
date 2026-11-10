#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_transitions.py —— 把谱系(树 或 cgMLST HierCC 层级) + barcode 转成
EvAM/HyperTraPS 的监督信号: (祖先状态 -> 后代状态) transition 表。

复刻 Aga2026 的路线(cgMLST/LIN -> 树 -> curate.tree -> transition), 但谱系来源灵活:

  输入模式 A —— Newick 树 (--tree):
    直接对给定树做祖先重建。默认产出两套 transition:
      1) 不可逆 (legacy): 与 Aga2026 / HyperTraPS 对齐, 只允许 gain
      2) 选择性可逆: 仅对有可逆先验的特征(优先 feature_hierarchy.csv 的 reversible=1,
         否则回退 location=plasmid / layer=plasmid_arg)允许净回退, 其它特征仍按不可逆。
    为兼容旧下游, transitions.csv 固定保留为不可逆版本。

  输入模式 B —— cgMLST HierCC 层级 (--hiercc):
    HierCC 的 hc0 ⊂ hc2 ⊂ ... ⊂ hc100 是天然的嵌套聚类, 可直接构造一棵"层级树":
    root -> hc100 簇 -> hc50 簇 -> ... -> hc2簇 -> 叶(样本)。
    然后同样输出两套 transition。

输出文件:
  transitions.csv                         : legacy 别名 = 不可逆版本 (供 HyperTraPS / 旧评估)
  transitions_irreversible.csv            : 不可逆版本显式文件名
  transitions_selective_reversible.csv    : 选择性可逆版本 (允许 gain + selective loss)
  transitions_summary.json                : legacy 默认摘要 (= 不可逆版本摘要)
  transitions_irreversible_summary.json   : 不可逆摘要
  transitions_selective_reversible_summary.json : 选择性可逆摘要
  manifest.json                           : 双策略清单, 标明哪个文件可直接用于 HyperTraPS

用法:
  # 模式A: 已有 Newick 树
  python build_transitions.py --barcode <evam_matrix或barcode.csv> --id-col sample_id \
      --tree Romania.nwk --outdir <out>
  # 模式B: cgMLST HierCC
  python build_transitions.py --barcode <barcode.csv> --id-col sample_id \
      --hiercc <bvbrc.csv> --hiercc-id orig_id \
      --hiercc-levels cgmlst_hc100,cgmlst_hc50,cgmlst_hc20,cgmlst_hc10,cgmlst_hc5,cgmlst_hc2 \
      --outdir <out>
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
from collections import Counter

import pandas as pd


# ---------------------------------------------------------------------------
# barcode 读取: 行=样本, 列=特征 0/1。id-col 指定样本 ID 列 (若无则用行号)。
# ---------------------------------------------------------------------------
def load_barcode(path: str, id_col: str | None):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if id_col and id_col in df.columns:
        ids = df[id_col].astype(str).tolist()
        feat_cols = [c for c in df.columns if c != id_col]
    else:
        ids = [str(i) for i in range(len(df))]
        feat_cols = list(df.columns)
    state = {}
    for i, rid in enumerate(ids):
        state[rid] = tuple(int(x) if str(x).strip() in ("0", "1") else 0
                           for x in df.iloc[i][feat_cols])
    return state, feat_cols


def find_hierarchy_path(barcode_path: str, explicit: str | None = None):
    """自动定位与 barcode 同数据集下的 feature_hierarchy.csv。"""
    cands = []
    if explicit:
        cands.append(os.path.abspath(explicit))
    bp = os.path.abspath(barcode_path)
    cands.extend([
        os.path.join(os.path.dirname(bp), "feature_hierarchy.csv"),
        os.path.join(os.path.dirname(os.path.dirname(bp)), "feature_hierarchy.csv"),
    ])
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


# ---------------------------------------------------------------------------
# 可逆先验: 优先 hierarchy.reversible==1, 否则回退 location=plasmid / layer=plasmid_arg。
# 这是一个"选择性可逆"弱先验, 不是对真实时间线的强断言。
#
# 也支持显式覆盖(用于没有 feature_hierarchy.csv 的数据集, 如 Kleborate22):
#   mode="auto" : 按 hierarchy 推断(默认, 向后兼容); 若同时给了 patterns 则取并集
#   mode="none" : 强制全部不可逆(单调累积, HyperTraPS 同口径)
#   mode="all"  : 全部特征可逆
#   mode="list" : 只用 patterns 指定的特征(支持 fnmatch 通配, 如 "*_acquired")
# ---------------------------------------------------------------------------
def _match_patterns(feat_cols: list[str], patterns: list[str]):
    hit = [False] * len(feat_cols)
    unmatched = []
    for pat in patterns:
        pat = pat.strip()
        if not pat:
            continue
        matched = False
        for i, feat in enumerate(feat_cols):
            if feat == pat or fnmatch.fnmatchcase(feat, pat):
                hit[i] = True
                matched = True
        if not matched:
            unmatched.append(pat)
    return hit, unmatched


def load_reversible_mask(hierarchy_path: str | None, feat_cols: list[str],
                         mode: str = "auto", patterns: list[str] | None = None):
    mask = [False] * len(feat_cols)
    patterns = [p for p in (patterns or []) if p.strip()]
    info = {
        "hierarchy": (os.path.abspath(hierarchy_path) if hierarchy_path and os.path.isfile(hierarchy_path) else None),
        "mode": mode,
        "patterns": patterns,
        "unmatched_patterns": [],
        "source": None,
        "reason": None,
        "n_reversible": 0,
        "reversible_features": [],
        "n_missing_feature_ids": 0,
        "missing_feature_ids": [],
    }

    if mode == "none":
        info.update({"source": "explicit --reversible-mode=none",
                     "reason": "all features forced irreversible"})
        return mask, info

    if mode == "all":
        mask = [True] * len(feat_cols)
        info.update({"source": "explicit --reversible-mode=all",
                     "reason": "all features forced reversible",
                     "n_reversible": len(feat_cols),
                     "reversible_features": list(feat_cols)})
        return mask, info

    if mode == "list":
        if not patterns:
            raise SystemExit("[trans] --reversible-mode=list 需要同时提供 --reversible-features")
        mask, unmatched = _match_patterns(feat_cols, patterns)
        info.update({"source": "explicit --reversible-features",
                     "reason": "ok" if any(mask) else "no feature matched the given patterns",
                     "unmatched_patterns": unmatched,
                     "n_reversible": int(sum(mask)),
                     "reversible_features": [feat_cols[i] for i, ok in enumerate(mask) if ok]})
        return mask, info

    if mode != "auto":
        raise SystemExit(f"[trans] 未知的 --reversible-mode: {mode}")

    if not hierarchy_path or not os.path.isfile(hierarchy_path):
        info["reason"] = "feature_hierarchy.csv not found"
        if patterns:
            mask, unmatched = _match_patterns(feat_cols, patterns)
            info.update({"source": "explicit --reversible-features (no hierarchy)",
                         "reason": "ok" if any(mask) else "no feature matched the given patterns",
                         "unmatched_patterns": unmatched,
                         "n_reversible": int(sum(mask)),
                         "reversible_features": [feat_cols[i] for i, ok in enumerate(mask) if ok]})
        return mask, info

    h = pd.read_csv(hierarchy_path, dtype=str, keep_default_na=False)
    if "feature_id" not in h.columns:
        info["reason"] = "feature_hierarchy.csv missing feature_id"
        return mask, info

    h = h.set_index("feature_id").reindex(feat_cols)
    missing_rows = h.isna().all(axis=1) if len(h.columns) else pd.Series(True, index=h.index)
    missing = [feat for feat, miss in zip(feat_cols, missing_rows.tolist()) if miss]

    def _truthy(s):
        return s.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})

    source = None
    cand = None
    if "reversible" in h.columns:
        cand = _truthy(h["reversible"])
        if cand.any():
            source = "feature_hierarchy.reversible==1"
    if source is None and "location" in h.columns:
        cand = h["location"].fillna("").astype(str).str.strip().eq("plasmid")
        if cand.any():
            source = "feature_hierarchy.location==plasmid"
    if source is None and "layer" in h.columns:
        cand = h["layer"].fillna("").astype(str).str.strip().eq("plasmid_arg")
        if cand.any():
            source = "feature_hierarchy.layer==plasmid_arg"
    if cand is not None and source is not None:
        mask = cand.fillna(False).astype(bool).tolist()
    else:
        info["reason"] = "no reversible prior found in hierarchy"

    if patterns:  # auto 模式下显式列表与 hierarchy 推断取并集
        extra, unmatched = _match_patterns(feat_cols, patterns)
        mask = [a or b for a, b in zip(mask, extra)]
        info["unmatched_patterns"] = unmatched
        source = ("%s + explicit --reversible-features" % source) if source \
            else "explicit --reversible-features"

    info.update({
        "source": source,
        "n_reversible": int(sum(mask)),
        "reversible_features": [feat_cols[i] for i, ok in enumerate(mask) if ok],
        "n_missing_feature_ids": len(missing),
        "missing_feature_ids": missing,
    })
    if source is not None and info["reason"] in (None, "feature_hierarchy.csv not found"):
        info["reason"] = "ok"
    return mask, info


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def _postorder_nodes(children, root):
    order = []
    stack = [(root, False)]
    while stack:
        node, done = stack.pop()
        if done:
            order.append(node)
            continue
        stack.append((node, True))
        for ch in children.get(node, []):
            stack.append((ch, False))
    return order


def _state_bit(v: int):
    return 1 if int(v) == 0 else 2  # {0}->{1}, {1}->{2}


def _pick_from_mask(mask: int, preferred: int = 0):
    if mask == 0:
        return int(preferred)
    pref_bit = _state_bit(preferred)
    if mask & pref_bit:
        return int(preferred)
    return 0 if (mask & 1) else 1


def extract_transitions(children: dict, node_state: dict, n_feat: int):
    """从 parent/child 精确状态抽取净变化: gain + loss。"""
    transitions = []
    for parent, kids in children.items():
        ps = node_state.get(parent)
        if ps is None:
            continue
        for child in kids:
            cs = node_state.get(child)
            if cs is None:
                continue
            gained = [k for k in range(n_feat) if cs[k] == 1 and ps[k] == 0]
            lost = [k for k in range(n_feat) if cs[k] == 0 and ps[k] == 1]
            if gained or lost:
                transitions.append((ps, cs, gained, lost, child))
    return transitions


# ---------------------------------------------------------------------------
# 策略1: 不可逆最大简约 (legacy / HyperTraPS 对齐)。
#   内部节点第 k 位 = AND over 所有有效子节点第 k 位。
# ---------------------------------------------------------------------------
def reconstruct_irreversible(children: dict, root, leaf_state: dict, n_feat: int):
    node_state = {}
    for node in _postorder_nodes(children, root):
        kids = children.get(node, [])
        if not kids:  # 叶
            node_state[node] = leaf_state.get(node)
            continue
        valid = [node_state.get(c) for c in kids if node_state.get(c) is not None]
        if not valid:
            node_state[node] = None
            continue
        agg = tuple(1 if all(s[k] == 1 for s in valid) else 0 for k in range(n_feat))
        node_state[node] = agg
    return node_state, extract_transitions(children, node_state, n_feat)


# ---------------------------------------------------------------------------
# 策略2: 选择性可逆。
#   - 对无可逆先验的特征: 仍按不可逆 AND 聚合 (与 legacy 一致)
#   - 对有可逆先验的特征: 用 binary Fitch parsimony 重建, 允许净 gain / net loss。
#     这里的 loss 解释为 sampled branch 两端的"净消失/回退", 不是已知真实时间线事件。
# ---------------------------------------------------------------------------
def reconstruct_selective_reversible(children: dict, root, leaf_state: dict,
                                     n_feat: int, reversible_mask: list[bool]):
    if not any(reversible_mask):
        return reconstruct_irreversible(children, root, leaf_state, n_feat)

    rev_idx = [k for k, ok in enumerate(reversible_mask) if ok]
    node_state = {}
    node_sets = {}

    for node in _postorder_nodes(children, root):
        kids = children.get(node, [])
        if not kids:  # 叶
            st = leaf_state.get(node)
            if st is None:
                node_state[node] = None
                node_sets[node] = None
                continue
            vals = list(st)
            sets = [0] * n_feat
            for k in rev_idx:
                sets[k] = _state_bit(st[k])
            node_state[node] = vals
            node_sets[node] = sets
            continue

        valid = [c for c in kids if node_state.get(c) is not None]
        if not valid:
            node_state[node] = None
            node_sets[node] = None
            continue

        vals = [0] * n_feat
        sets = [0] * n_feat
        for k in range(n_feat):
            if not reversible_mask[k]:
                vals[k] = 1 if all(node_state[c][k] == 1 for c in valid) else 0
        for k in rev_idx:
            child_masks = [node_sets[c][k] for c in valid if node_sets[c][k] != 0]
            if not child_masks:
                sets[k] = 0
                continue
            inter = child_masks[0]
            union = child_masks[0]
            for m in child_masks[1:]:
                inter &= m
                union |= m
            sets[k] = inter if inter else union
        node_state[node] = vals
        node_sets[node] = sets

    if node_state.get(root) is None:
        return {k: (None if v is None else tuple(v)) for k, v in node_state.items()}, []

    # 根偏好 0: 与 all-zero 起点 / HyperTraPS 传统口径兼容, 同时保持可逆子集最少变更。
    for k in rev_idx:
        node_state[root][k] = _pick_from_mask(node_sets[root][k], preferred=0)

    stack = [root]
    while stack:
        parent = stack.pop()
        ps = node_state[parent]
        for child in children.get(parent, []):
            cs = node_state.get(child)
            if cs is None:
                continue
            for k in rev_idx:
                cs[k] = _pick_from_mask(node_sets[child][k], preferred=ps[k])
            stack.append(child)

    final_state = {n: (None if st is None else tuple(int(x) for x in st)) for n, st in node_state.items()}
    return final_state, extract_transitions(children, final_state, n_feat)


def collect_leaves(children, node):
    """Collect all leaf nodes under node."""
    out = []
    stack = [node]
    while stack:
        x = stack.pop()
        kids = children.get(x, [])
        if not kids:
            out.append(x)
        else:
            stack.extend(kids)
    return out


def group_of_child(child, children, leaf_group, mode, group_level=None):
    """Assign a lineage-group label to a transition's child (for grouped CV).
    group_level: e.g. "cgmlst_hc100"; pick that segment from the HierCC path key.
    If None, use the coarsest (first) segment."""
    if mode == "cgmlst_hiercc" and isinstance(child, str) and "=" in child:
        segs = child.split("|")
        if group_level:
            for seg in segs:
                if seg.startswith(group_level + "="):
                    return seg
        return segs[0]
    if leaf_group:
        leaves = collect_leaves(children, child)
        gs = [leaf_group.get(lf) for lf in leaves if leaf_group.get(lf)]
        if gs:
            return Counter(gs).most_common(1)[0][0]
    return str(child)


# ---------------------------------------------------------------------------
# 模式A: Newick 树 -> children/leaf_state
# ---------------------------------------------------------------------------
def build_from_tree(tree_path, tree_format, barcode_state, n_feat):
    from Bio import Phylo

    tree = Phylo.read(tree_path, tree_format)
    children = {}
    leaf_state = {}
    counter = [0]

    def uid(clade):
        if clade.name:
            nm = str(clade.name).strip()
            # Newick 引用标签: Biopython 会保留成对的首尾单/双引号 (如 'GCF_x.1'),
            # 必须去掉, 否则叶名与 barcode 的 sample_id 对不上 (模式A 关键)。
            if len(nm) >= 2 and nm[0] in ("'", '"') and nm[-1] == nm[0]:
                nm = nm[1:-1]
            return nm
        counter[0] += 1
        return f"_node{counter[0]}"

    name_of = {}

    def walk(clade):
        nid = name_of.setdefault(id(clade), uid(clade))
        kids = clade.clades
        if not kids:
            leaf_state[nid] = barcode_state.get(nid)
            children.setdefault(nid, [])
        else:
            child_ids = []
            for ch in kids:
                cid = name_of.setdefault(id(ch), uid(ch))
                child_ids.append(cid)
                walk(ch)
            children[nid] = child_ids

    walk(tree.root)
    root_id = name_of[id(tree.root)]
    matched = sum(1 for v in leaf_state.values() if v is not None)
    return children, root_id, leaf_state, matched


# ---------------------------------------------------------------------------
# 模式B: cgMLST HierCC 层级 -> 构造嵌套层级树
#   levels 从粗到细,如 [hc100, hc50, hc20, hc10, hc5, hc2]。
#   路径: root -> hc100=X -> hc50=Y -> ... -> hc2=Z -> 叶(sample_id)。
#   同一前缀共享内部节点 (用累积路径元组作节点键)。
# ---------------------------------------------------------------------------
def build_from_hiercc(hiercc_df, hiercc_id, levels, id_map, barcode_state, n_feat):
    children = {}
    leaf_state = {}
    root = "ROOT"
    children[root] = []

    def add_child(parent, child):
        lst = children.setdefault(parent, [])
        if child not in lst:
            lst.append(child)

    n_leaf = 0
    for _, row in hiercc_df.iterrows():
        orig = str(row[hiercc_id])
        sample_id = id_map.get(orig)
        if sample_id is None or sample_id not in barcode_state:
            continue
        # 构造从粗到细的路径节点
        parent = root
        path_key = ()
        skip = False
        for lv in levels:
            val = str(row.get(lv, "")).strip()
            if val == "":
                skip = True
                break
            path_key = path_key + (f"{lv}={val}",)
            node = "|".join(path_key)
            add_child(parent, node)
            parent = node
        if skip:
            # 层级不全: 直接挂到 root 下 (退化为独立观测)
            parent = root
        # 叶 = 样本
        add_child(parent, sample_id)
        leaf_state[sample_id] = barcode_state.get(sample_id)
        children.setdefault(sample_id, [])
        n_leaf += 1
    return children, root, leaf_state, n_leaf


def state_to_str(state):
    return "".join(str(b) for b in state)


def build_rows(transitions, feat_cols, children, leaf_group, mode, group_level, strategy):
    rows = []
    gain_counter = Counter()
    loss_counter = Counter()
    group_counter = Counter()
    for ps, cs, gained, lost, child in transitions:
        grp = group_of_child(child, children, leaf_group, mode, group_level)
        rows.append({
            "src_state": state_to_str(ps),
            "dest_state": state_to_str(cs),
            "n_gained": len(gained),
            "gained_features": ";".join(feat_cols[k] for k in gained),
            "n_lost": len(lost),
            "lost_features": ";".join(feat_cols[k] for k in lost),
            "group_id": grp,
            "transition_strategy": strategy,
        })
        group_counter[grp] += 1
        for k in gained:
            gain_counter[feat_cols[k]] += 1
        for k in lost:
            loss_counter[feat_cols[k]] += 1
    df = pd.DataFrame(rows, columns=[
        "src_state", "dest_state", "n_gained", "gained_features",
        "n_lost", "lost_features", "group_id", "transition_strategy",
    ])
    stats = {
        "n_informative_transitions": len(transitions),
        "n_multi_gain_transitions": sum(1 for _, _, g, _, _ in transitions if len(g) > 1),
        "n_loss_transitions": sum(1 for _, _, _, l, _ in transitions if len(l) > 0),
        "n_multi_loss_transitions": sum(1 for _, _, _, l, _ in transitions if len(l) > 1),
        "n_groups": len(group_counter),
        "gained_feature_counts": dict(gain_counter.most_common()),
        "lost_feature_counts": dict(loss_counter.most_common()),
    }
    return df, stats


def write_strategy(outdir, csv_name, summary_name, df, summary):
    df.to_csv(os.path.join(outdir, csv_name), index=False)
    with open(os.path.join(outdir, summary_name), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description="Build ancestor->descendant transitions")
    ap.add_argument("--barcode", required=True, help="barcode csv (行=样本,列=特征0/1)")
    ap.add_argument("--id-col", default="sample_id", help="barcode 里的样本 ID 列名")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--hierarchy", default=None,
                    help="feature_hierarchy.csv 路径 (默认自动定位 barcode 同数据集下)")
    # 模式A
    ap.add_argument("--tree", default=None)
    ap.add_argument("--tree-format", default="newick")
    # 模式B
    ap.add_argument("--hiercc", default=None, help="BV-BRC取数 csv")
    ap.add_argument("--hiercc-id", default="orig_id", help="hiercc 文件里对应 barcode 的 ID 列")
    ap.add_argument("--hiercc-sample-prefix", default=None,
                    help="若 barcode 的 sample_id 形如 source:species:orig_id, 传该前缀以映射")
    ap.add_argument("--hiercc-levels",
                    default="cgmlst_hc100,cgmlst_hc50,cgmlst_hc20,cgmlst_hc10,cgmlst_hc5,cgmlst_hc2")
    ap.add_argument("--group-level", default="cgmlst_hc100",
                    help="lineage group resolution for CV, e.g. cgmlst_hc10 / cgmlst_hc100")
    # tree 模式(模式A)可选: 外接 叶id->分组 映射(如 samples.csv 的 ST), 用于分组CV 防近缘泄漏;
    # 内部边的分组 = 其子树叶子的多数分组。缺省 None 时保持原行为(每个 transition 各自成组)。
    ap.add_argument("--leaf-group-file", default=None,
                    help="tree模式: CSV, 提供 叶id->分组(如 ST) 映射, 供分组CV")
    ap.add_argument("--leaf-group-id-col", default="sample_id")
    ap.add_argument("--leaf-group-col", default="ST")
    # 可逆性控制: 决定 selective_reversible 策略允许哪些特征发生丢失。
    ap.add_argument("--reversible-mode", default="auto",
                    choices=["auto", "none", "all", "list"],
                    help="可逆先验来源: auto=按 feature_hierarchy 推断(默认, 若同时给 "
                         "--reversible-features 则取并集); none=强制全部不可逆; "
                         "all=全部可逆; list=仅用 --reversible-features")
    ap.add_argument("--reversible-features", default="",
                    help="逗号分隔的可逆特征名, 支持 fnmatch 通配, 例如 "
                         "\"*_acquired,Omp_mutations,Col_mutations\"")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    barcode_state, feat_cols = load_barcode(args.barcode, args.id_col)
    n_feat = len(feat_cols)
    hierarchy_path = find_hierarchy_path(args.barcode, args.hierarchy)
    reversible_mask, reversible_info = load_reversible_mask(
        hierarchy_path, feat_cols,
        mode=args.reversible_mode,
        patterns=[p for p in args.reversible_features.split(",") if p.strip()])
    print(f"[trans] barcode: {len(barcode_state)} 样本 x {n_feat} 特征")
    print("[trans] reversible prior: mode=%s | %s | n=%d%s"
          % (args.reversible_mode,
             reversible_info.get("source") or reversible_info.get("reason"),
             reversible_info["n_reversible"],
             (" | unmatched=%s" % reversible_info["unmatched_patterns"])
             if reversible_info.get("unmatched_patterns") else ""))

    if args.tree:
        print(f"[trans] 模式A: Newick 树 {args.tree}")
        children, root, leaf_state, matched = build_from_tree(
            args.tree, args.tree_format, barcode_state, n_feat)
        mode = "tree"
    elif args.hiercc:
        print(f"[trans] 模式B: cgMLST HierCC {args.hiercc}")
        levels = [x.strip() for x in args.hiercc_levels.split(",") if x.strip()]
        hdf = pd.read_csv(args.hiercc, dtype=str, keep_default_na=False)
        # id_map: hiercc 的 orig_id -> barcode 的 sample_id
        if args.hiercc_sample_prefix:
            id_map = {o: f"{args.hiercc_sample_prefix}{o}" for o in hdf[args.hiercc_id]}
        else:
            id_map = {o: o for o in hdf[args.hiercc_id]}
        children, root, leaf_state, matched = build_from_hiercc(
            hdf, args.hiercc_id, levels, id_map, barcode_state, n_feat)
        mode = "cgmlst_hiercc"
    else:
        raise SystemExit("必须指定 --tree 或 --hiercc 之一")

    # tree 模式可外接 leaf->group 映射(如 ST); 默认 None (HierCC 从节点键提取分组)
    leaf_group = None
    if mode == "tree" and args.leaf_group_file:
        lg = pd.read_csv(args.leaf_group_file, dtype=str, keep_default_na=False)
        gid, gcol = args.leaf_group_id_col, args.leaf_group_col
        if gid in lg.columns and gcol in lg.columns:
            leaf_group = {str(r[gid]): str(r[gcol]) for _, r in lg.iterrows()
                          if str(r[gcol]).strip() != ""}
            print("[trans] leaf-group: %d 叶 id -> '%s' 分组 (来自 %s)"
                  % (len(leaf_group), gcol, os.path.basename(args.leaf_group_file)))
        else:
            print("[trans] WARN: leaf-group 列缺失 (%s / %s), 忽略分组" % (gid, gcol))
    print("[trans] leaves matched to barcode: %d" % matched)

    _, transitions_irrev = reconstruct_irreversible(children, root, leaf_state, n_feat)
    _, transitions_selrev = reconstruct_selective_reversible(
        children, root, leaf_state, n_feat, reversible_mask)

    df_irrev, stats_irrev = build_rows(
        transitions_irrev, feat_cols, children, leaf_group, mode, args.group_level,
        strategy="irreversible")
    df_selrev, stats_selrev = build_rows(
        transitions_selrev, feat_cols, children, leaf_group, mode, args.group_level,
        strategy="selective_reversible")

    # 是否与不可逆版本在"实质内容"上完全一致。忽略 transition_strategy 这一标识列,
    # 否则即便两份 transition 的状态/gain/loss 全相同, 也会因策略标签不同而误判为 False。
    same_outputs = df_irrev.drop(columns=["transition_strategy"], errors="ignore").equals(
        df_selrev.drop(columns=["transition_strategy"], errors="ignore"))

    summary_base = {
        "mode": mode,
        "barcode": os.path.abspath(args.barcode),
        "hierarchy": reversible_info.get("hierarchy"),
        "n_features": n_feat,
        "feature_columns": feat_cols,
        "n_leaves_matched": matched,
        "reversible_prior": reversible_info,
    }
    irrev_summary = {
        **summary_base,
        "strategy": "irreversible",
        "description": "legacy irreversible AND parsimony (HyperTraPS-compatible default)",
        "hypertraps_compatible": True,
        **stats_irrev,
    }
    selrev_summary = {
        **summary_base,
        "strategy": "selective_reversible",
        "description": "irreversible for most features; reversible prior subset uses Fitch parsimony and may yield net loss",
        "hypertraps_compatible": False,
        "identical_to_irreversible": bool(same_outputs),
        **stats_selrev,
    }

    # legacy 默认: transitions.csv = 不可逆版本, 确保 HyperTraPS / 旧评估脚本完全不受影响。
    write_strategy(args.outdir, "transitions.csv", "transitions_summary.json", df_irrev, irrev_summary)
    write_strategy(args.outdir, "transitions_irreversible.csv", "transitions_irreversible_summary.json",
                   df_irrev, irrev_summary)
    write_strategy(args.outdir, "transitions_selective_reversible.csv",
                   "transitions_selective_reversible_summary.json", df_selrev, selrev_summary)

    manifest = {
        **summary_base,
        "default_strategy": "irreversible",
        "default_csv": "transitions.csv",
        "default_summary": "transitions_summary.json",
        "strategies": {
            "irreversible": {
                "csv": "transitions.csv",
                "alias_csv": "transitions_irreversible.csv",
                "summary": "transitions_irreversible_summary.json",
                "hypertraps_compatible": True,
                "gain_only": True,
            },
            "selective_reversible": {
                "csv": "transitions_selective_reversible.csv",
                "summary": "transitions_selective_reversible_summary.json",
                "hypertraps_compatible": False,
                "gain_only": False,
                "identical_to_irreversible": bool(same_outputs),
            },
        },
    }
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("[trans] DONE. irreversible=%d (multi-gain=%d, loss=%d) | "
          "selective_reversible=%d (multi-gain=%d, loss=%d) -> %s"
          % (stats_irrev["n_informative_transitions"],
             stats_irrev["n_multi_gain_transitions"],
             stats_irrev["n_loss_transitions"],
             stats_selrev["n_informative_transitions"],
             stats_selrev["n_multi_gain_transitions"],
             stats_selrev["n_loss_transitions"],
             args.outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
