#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stratify_transitions_tree.py —— **树模式**下按国家（或任意元数据键）分层构造 transition，
复刻 Aga et al. 2026 (PLoS Biol) 的"每国一棵树 → 每国独立推断"路线。

与既有 `stratify_transitions.py` 的分工
-------------------------------------
| 脚本 | 输入拓扑 | 适用数据集 |
|---|---|---|
| `stratify_transitions.py` | BV-BRC cgMLST HierCC 层级（模式B） | AMRGlobal16 系列 |
| `stratify_transitions_tree.py`（本脚本） | Newick 系统发生树（模式A） | `microreact_kp_kleborate22` 等带树的数据集 |

方法
----
1. 一次读入整棵 Newick 树，得到 `children / root / leaf_state`（复用 `build_from_tree`，
   因此叶名去引号等细节与全局口径完全一致）；
2. 从 `samples.csv` 读 `sample_id -> 分层键`（如 `Country`）；
3. 对每个分层键取该组的叶集合，把树**裁剪**（prune）到这些叶：
   删除不在组内的叶，并折叠只剩单个子节点的内部节点 —— 这一步保留了组内叶之间的
   真实拓扑关系，等价于"只用该国样本重建一棵树"，但避免了重新跑建树；
4. 在裁剪后的子树上独立做**不可逆最大简约**祖先重建 → 该组的 transitions；
5. 样本数 < `--min-samples` 或 informative transition < `--min-transitions` 的组丢弃
   （= 论文的 informative-country 过滤）。

输出格式与全局 `build_transitions.py` 完全一致
（`src_state,dest_state,n_gained,gained_features,n_lost,lost_features,group_id,transition_strategy`），
因此下游 `evoamr_story.load_transition_set` / `export_hypertraps.py` 均可直接读取。

**重要口径说明**：裁剪子树上的内部节点状态由该组的叶子重新 AND 聚合得到，因此某国的
转移集**不是**全局转移集的行子集 —— 这与论文"逐国独立重建"一致，也是它必须的：
若只按叶的国家给全局转移打标签，跨国共享的祖先节点会被重复计数。

用法
----
    python pipeline/stratify_transitions_tree.py \
        --barcode data/processed/microreact_kp_kleborate22/category_barcode.csv \
        --tree data/kleborate/2021-Klebsiella_tree.nwk \
        --samples data/processed/microreact_kp_kleborate22/samples.csv \
        --stratify-by Country --leaf-group-col ST \
        --min-samples 100 --min-transitions 30 \
        --outdir data/processed/microreact_kp_kleborate22/transitions_by_country
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_transitions import (  # noqa: E402
    build_from_tree,
    build_rows,
    load_barcode,
    reconstruct_irreversible,
    _postorder_nodes,
)

MISSING_TOKENS = {"", "nan", "na", "n/a", "none", "null", "unknown", "missing", "-"}


def safe_name(name: str) -> str:
    """把分层键值转成安全的目录名（空格 -> 下划线，其余非法字符 -> 下划线）。"""
    s = str(name).strip().replace(" ", "_")
    s = re.sub(r"[^0-9A-Za-z_.\-]", "_", s)
    return s or "UNKNOWN"


def prune_to_leaves(children: Dict[str, List[str]], root: str,
                    keep: Sequence[str]) -> Tuple[Dict[str, List[str]], Optional[str], int]:
    """
    把树裁剪到 `keep` 指定的叶集合，并折叠只剩单个子节点的内部节点。

    返回 `(new_children, new_root, n_leaves)`；若组内有效叶 < 2 则 `new_root=None`。

    折叠是必要的：不折叠会留下大量"父子状态必然相同"的链，虽然不产生 transition，
    但会让 `group_of_child` 的叶收集与节点计数失真。
    """
    keep_set = set(keep)
    rep: Dict[str, Optional[str]] = {}      # 节点 -> 裁剪后代表它的节点（None = 整棵子树被删）
    new_children: Dict[str, List[str]] = {}
    for node in _postorder_nodes(children, root):
        kids = children.get(node, [])
        if not kids:                        # 叶
            rep[node] = node if node in keep_set else None
            continue
        alive = [rep.get(c) for c in kids]
        alive = [c for c in alive if c is not None]
        if not alive:
            rep[node] = None
        elif len(alive) == 1:               # 单子节点 -> 折叠掉本节点
            rep[node] = alive[0]
        else:
            rep[node] = node
            new_children[node] = alive
    new_root = rep.get(root)
    if new_root is None or new_root not in new_children:
        return {}, None, 0
    # 叶不需要在 new_children 里显式建 key：collect_leaves 以 `children.get(x, [])`
    # 为空来判定叶节点
    n_leaves = sum(1 for lf in keep_set if rep.get(lf) == lf)
    return new_children, new_root, n_leaves


def read_stratum(samples_path: str, key_col: str, id_col: str = "sample_id"
                 ) -> Dict[str, str]:
    df = pd.read_csv(samples_path, dtype=str, keep_default_na=False)
    for col in (id_col, key_col):
        if col not in df.columns:
            raise SystemExit("samples.csv 缺少列 %r（现有列: %s）" % (col, list(df.columns)))
    out: Dict[str, str] = {}
    for sid, val in zip(df[id_col].astype(str), df[key_col].astype(str)):
        v = val.strip()
        if v.lower() in MISSING_TOKENS:
            continue                        # 无分层信息的样本不参与（与论文丢弃 missing 一致）
        out[sid] = v
    return out


def read_leaf_group(samples_path: str, group_col: Optional[str],
                    id_col: str = "sample_id") -> Optional[Dict[str, str]]:
    """叶 -> 谱系标签（用于组内 grouped CV 的 group_id）。"""
    if not group_col:
        return None
    df = pd.read_csv(samples_path, dtype=str, keep_default_na=False)
    if group_col not in df.columns:
        raise SystemExit("samples.csv 缺少谱系列 %r" % group_col)
    return {str(s): str(g).strip() for s, g in zip(df[id_col], df[group_col])
            if str(g).strip().lower() not in MISSING_TOKENS}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="树模式下按国家/地区分层构造 transition（复刻 Aga2026 逐国推断）")
    ap.add_argument("--barcode", required=True, help="barcode csv（行=样本，列=特征 0/1）")
    ap.add_argument("--id-col", default="sample_id")
    ap.add_argument("--tree", required=True, help="Newick 树，叶名 = barcode 的 sample_id")
    ap.add_argument("--tree-format", default="newick")
    ap.add_argument("--samples", required=True, help="samples.csv，需含 sample_id + 分层键列")
    ap.add_argument("--stratify-by", default="Country", help="分层键列名（如 Country / Region）")
    ap.add_argument("--leaf-group-col", default="ST",
                    help="组内 grouped CV 的谱系列名；置空则用节点名作 group_id")
    ap.add_argument("--min-samples", type=int, default=100,
                    help="每组最少样本数（复刻 informative-country 过滤）")
    ap.add_argument("--min-transitions", type=int, default=30,
                    help="每组最少 informative transition 数")
    ap.add_argument("--max-groups", type=int, default=0, help="0 = 不限")
    ap.add_argument("--shuffle-seed", type=int, default=-1,
                    help=">=0 时**打乱**样本与分层键的对应关系（保持每组样本数不变），"
                         "用于生成零假设对照：此时组间的动力学差异只可能来自采样噪声。")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    barcode_state, feat_cols = load_barcode(args.barcode, args.id_col)
    n_feat = len(feat_cols)
    print("[stratify-tree] barcode: %d 样本 x %d 特征" % (len(barcode_state), n_feat))

    children, root, leaf_state, matched = build_from_tree(
        args.tree, args.tree_format, barcode_state, n_feat)
    print("[stratify-tree] 树: %d 节点 | 叶与 barcode 匹配 %d" % (len(children), matched))

    stratum = read_stratum(args.samples, args.stratify_by, args.id_col)
    leaf_group = read_leaf_group(args.samples, args.leaf_group_col, args.id_col)

    # 只保留"既在树上、又有 barcode、又有分层键"的叶
    usable = {lf for lf, stt in leaf_state.items() if stt is not None}
    stratum = {s: k for s, k in stratum.items() if s in usable}

    if args.shuffle_seed >= 0:
        # 零假设对照：保持每个分层键的样本数不变，只打乱"哪个样本属于哪个键"。
        # 这样一来，各组的样本量、树上的分布密度都与真实分层同阶，
        # 组间动力学差异**只可能**来自采样噪声与树的局部结构。
        import random
        rnd = random.Random(args.shuffle_seed)
        sids = list(stratum)
        keys = [stratum[s] for s in sids]
        rnd.shuffle(keys)
        stratum = dict(zip(sids, keys))
        print("[stratify-tree] 已打乱分层标签（shuffle-seed=%d）—— 本次产物是零假设对照"
              % args.shuffle_seed)

    key_to_leaves: Dict[str, List[str]] = {}
    for sid, key in stratum.items():
        key_to_leaves.setdefault(key, []).append(sid)

    manifest = {
        "mode": "tree",
        "stratify_by": args.stratify_by,
        "tree": os.path.abspath(args.tree),
        "barcode": os.path.abspath(args.barcode),
        "samples": os.path.abspath(args.samples),
        "leaf_group_col": args.leaf_group_col,
        "min_samples": args.min_samples,
        "min_transitions": args.min_transitions,
        "shuffle_seed": (args.shuffle_seed if args.shuffle_seed >= 0 else None),
        "is_null_control": bool(args.shuffle_seed >= 0),
        "n_features": n_feat,
        "feature_columns": feat_cols,
        "strategy": "irreversible",
        "note": ("每组的内部节点状态由该组叶子独立 AND 聚合得到，"
                 "因此组内转移集不是全局转移集的行子集（与逐国独立重建一致）。"),
        "groups": [],
        "dropped_small_sample": [],
        "dropped_few_transitions": [],
    }

    kept = 0
    ordered = sorted(key_to_leaves, key=lambda k: -len(key_to_leaves[k]))
    for key in ordered:
        leaves = key_to_leaves[key]
        if len(leaves) < args.min_samples:
            manifest["dropped_small_sample"].append({"key": key, "n_samples": len(leaves)})
            continue
        if args.max_groups and kept >= args.max_groups:
            break

        sub_children, sub_root, n_leaves = prune_to_leaves(children, root, leaves)
        if sub_root is None:
            manifest["dropped_few_transitions"].append(
                {"key": key, "n_samples": len(leaves), "n_transitions": 0,
                 "reason": "pruned tree has < 2 leaves"})
            continue
        sub_leaf_state = {lf: leaf_state[lf] for lf in leaves}
        _, transitions = reconstruct_irreversible(sub_children, sub_root,
                                                  sub_leaf_state, n_feat)
        if len(transitions) < args.min_transitions:
            manifest["dropped_few_transitions"].append(
                {"key": key, "n_samples": len(leaves), "n_leaves_matched": n_leaves,
                 "n_transitions": len(transitions)})
            continue

        df, stats = build_rows(transitions, feat_cols, sub_children, leaf_group,
                               "tree", None, "irreversible")
        safe = safe_name(key)
        gdir = os.path.join(args.outdir, safe)
        os.makedirs(gdir, exist_ok=True)
        df.to_csv(os.path.join(gdir, "transitions.csv"), index=False)

        g_summary = {"key": key, "safe_name": safe, "n_samples": len(leaves),
                     "n_leaves_matched": n_leaves, "dir": os.path.abspath(gdir), **stats}
        with open(os.path.join(gdir, "transitions_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(g_summary, fh, indent=2, ensure_ascii=False)
        manifest["groups"].append({k: g_summary[k] for k in (
            "key", "safe_name", "n_samples", "n_leaves_matched",
            "n_informative_transitions", "n_multi_gain_transitions", "n_groups", "dir")})
        kept += 1
        print("[stratify-tree] %-24s samples=%5d leaves=%5d transitions=%5d "
              "multi-gain=%5d cv-groups=%4d -> %s/"
              % (key, len(leaves), n_leaves, stats["n_informative_transitions"],
                 stats["n_multi_gain_transitions"], stats["n_groups"], safe))

    manifest["n_groups_kept"] = kept
    with open(os.path.join(args.outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print("\n[stratify-tree] DONE. kept %d groups (dropped %d too-few-samples, "
          "%d too-few-transitions) -> %s/manifest.json"
          % (kept, len(manifest["dropped_small_sample"]),
             len(manifest["dropped_few_transitions"]), args.outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
