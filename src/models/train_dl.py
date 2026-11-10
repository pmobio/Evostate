#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_dl.py —— 自研深度学习集合预测模型: 全数据训练 (对齐 HyperTraPS step3)

输入:
  1) transitions.csv / transitions_irreversible.csv
     -> gain-only SetPredDL.fit(全数据) -> model.pt
  2) transitions_selective_reversible.csv
     -> V1 dual-head selective-reversible Transformer
        gain head: 预测净获得
        loss head: 仅在 selective reversible 子集预测端点净消失/回退

额外产出:
  fit_report.json       : 过拟合诊断 + 元数据使用情况 + loss head 使用情况
  first_step_ranking.csv: 从空状态 s=0 的 gain 排序
  loss_step_ranking.csv : loss head 启用时, 在"全部可逆位均已存在"状态上的 loss 排序

方案A (特征元数据 token 化): 自动定位 transitions.csv 同数据集下的 feature_hierarchy.csv,
  把 layer/drug_class/mechanism/location/reversible 按位序对齐后喂给 transformer 的每个
  token (见 model_dl.py)。位序对齐由 assembler.py 的 .reindex(kept_feats) 保证, 训练前用
  gained_features / lost_features 名字反查再做一次运行时校验。更重要的是: **按输入数据自适应** ——
  缺失字段自动跳过, 只有单一取值(如纯 gene 数据里 layer 恒为 gene / reversible 恒为0)
  的字段也自动禁用, 只把当前数据里真正有区分度的元数据送进模型。--no-meta 可关闭
  (消融), --hierarchy 可显式指定。

设计红线: 仅用 barcode 状态 + 特征内生分子属性作输入; 环境协变量/时间不进模型 (docs/05 §7)。

用法:
  python3 train_dl.py --transitions .../transitions.csv --outdir .../dl_<DATASET> \
      --arch transformer --epochs 200
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_dl as MDL


def _truthy(s):
    return s.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


# 允许的 transition 构造策略
#   irreversible         : 只允许 0->1 (单调累积, HyperTraPS 传统口径)
#   selective_reversible : 系统发育 Fitch 重建, 可逆先验子集允许净 1->0
#   full_temporal        : 直接观测的时序采样点配对, gain/loss 均为观测量
STRATEGY_IRREVERSIBLE = "irreversible"
STRATEGY_SELECTIVE_REVERSIBLE = "selective_reversible"
STRATEGY_FULL_TEMPORAL = "full_temporal"
KNOWN_STRATEGIES = {
    STRATEGY_IRREVERSIBLE,
    STRATEGY_SELECTIVE_REVERSIBLE,
    STRATEGY_FULL_TEMPORAL,
}
LOSS_CAPABLE_STRATEGIES = {STRATEGY_SELECTIVE_REVERSIBLE, STRATEGY_FULL_TEMPORAL}


def load_transitions(path):
    """逐行解析 src/dest bitstring -> gained/lost, 并保留 gained/lost feature 名字。"""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    strategies = []
    if "transition_strategy" in df.columns:
        strategies = sorted({str(x).strip() for x in df["transition_strategy"].tolist()
                             if str(x).strip()})
        bad = sorted(set(strategies) - KNOWN_STRATEGIES)
        if bad:
            raise ValueError(
                f"不支持的 transition_strategy: {bad}; 仅支持 {sorted(KNOWN_STRATEGIES)}")
        if len(strategies) > 1:
            raise ValueError(
                f"同一 transitions 文件混用了多种 strategy={strategies}; "
                "请分别使用单一策略文件。")

    trans = []
    Ln = 0
    n_loss_total = 0
    n_loss_transitions = 0
    for _, r in df.iterrows():
        src = np.array([int(c) for c in r["src_state"]], dtype=np.int8)
        dst = np.array([int(c) for c in r["dest_state"]], dtype=np.int8)
        Ln = len(src)
        gained = [i for i in range(Ln) if dst[i] == 1 and src[i] == 0]
        lost = [i for i in range(Ln) if dst[i] == 0 and src[i] == 1]

        strategy = str(r.get("transition_strategy", "")).strip()
        if not strategy:
            strategy = STRATEGY_SELECTIVE_REVERSIBLE if lost else STRATEGY_IRREVERSIBLE
        if strategy == STRATEGY_IRREVERSIBLE and lost:
            raise ValueError(
                "输入声明为 irreversible, 但检测到 src=1,dst=0 的 loss 位; "
                "请检查是否误用了 selective_reversible 文件。")

        gnames = r.get("gained_features", "") or ""
        lnames = r.get("lost_features", "") or ""
        gained_names = [x for x in gnames.split(";") if x]
        lost_names = [x for x in lnames.split(";") if x]

        trans.append({
            "src": src,
            "dest": dst,
            "gained": gained,
            "lost": lost,
            "gained_names": gained_names,
            "lost_names": lost_names,
            "group": r.get("group_id", ""),
            "transition_strategy": strategy,
        })
        n_loss_total += len(lost)
        n_loss_transitions += int(len(lost) > 0)

    if not strategies:
        strategies = ([STRATEGY_SELECTIVE_REVERSIBLE] if n_loss_total > 0
                      else [STRATEGY_IRREVERSIBLE])

    loss_capable = bool(
        (set(strategies) & LOSS_CAPABLE_STRATEGIES) or n_loss_total > 0)
    info = {
        "path": os.path.abspath(path),
        "strategies": strategies,
        "n_loss_total": int(n_loss_total),
        "n_loss_transitions": int(n_loss_transitions),
        # 历史字段名保留 (下游 collector 依赖); 语义为"输入含 loss 侧信息"
        "selective_reversible_input": loss_capable,
        "loss_capable_input": loss_capable,
        "requested_head_mode": (
            "dual_head_selective_reversible" if loss_capable else "gain_only"
        ),
    }
    return trans, Ln, info


def find_hierarchy_path(transitions_path):
    """transitions.csv 通常在 <dataset>/transitions/transitions*.csv,
    feature_hierarchy.csv 在 <dataset>/feature_hierarchy.csv。自动上溯定位。"""
    tp = os.path.abspath(transitions_path)
    cand = os.path.join(os.path.dirname(os.path.dirname(tp)), "feature_hierarchy.csv")
    if os.path.isfile(cand):
        return cand
    cand2 = os.path.join(os.path.dirname(tp), "feature_hierarchy.csv")
    if os.path.isfile(cand2):
        return cand2
    return None


META_FIELDS = ["layer", "drug_class", "mechanism", "location", "reversible"]


def parse_meta_fields_arg(spec):
    """解析 --meta-fields; None/空字符串表示默认全量自适应扫描。"""
    if spec is None:
        return None
    if isinstance(spec, str):
        toks = [x.strip() for x in spec.replace(";", ",").split(",")]
    else:
        toks = [str(x).strip() for x in spec]
    toks = [x for x in toks if x]
    if not toks:
        return None

    out = []
    bad = []
    for tok in toks:
        if tok not in META_FIELDS:
            bad.append(tok)
            continue
        if tok not in out:
            out.append(tok)
    if bad:
        raise ValueError(f"--meta-fields 含未知字段: {bad}; 仅支持 {META_FIELDS}")
    return out


def load_feature_hierarchy(hierarchy_path, L):
    df = pd.read_csv(hierarchy_path, dtype=str, keep_default_na=False)
    if len(df) != L:
        raise ValueError(
            f"feature_hierarchy.csv 行数({len(df)}) 与 bitstring 长度 L({L}) 不一致, "
            f"无法按位对齐: {hierarchy_path}")
    if "feature_id" not in df.columns:
        raise ValueError(f"feature_hierarchy.csv 缺少 feature_id 列: {hierarchy_path}")
    return df


def load_feature_meta(hdf, requested_meta_fields=None):
    """按当前输入数据自适应选择元数据字段, 可选字段白名单。"""
    requested_meta_fields = parse_meta_fields_arg(requested_meta_fields)
    scan_fields = META_FIELDS if requested_meta_fields is None else requested_meta_fields

    feat_ids = hdf["feature_id"].tolist()
    fields, vocab, ids, vocab_labels = [], {}, {}, {}
    meta_scan = {
        "requested_fields": list(scan_fields),
        "enabled": {},
        "skipped": {},
    }
    if requested_meta_fields is not None:
        for f in META_FIELDS:
            if f not in scan_fields:
                meta_scan["skipped"][f] = {"reason": "not_requested"}

    for f in scan_fields:
        if f not in hdf.columns:
            meta_scan["skipped"][f] = {"reason": "missing_column"}
            continue
        col = hdf[f].astype(str)
        cats = sorted(col.unique())
        if len(cats) <= 1:
            meta_scan["skipped"][f] = {
                "reason": "constant_value",
                "value": (cats[0] if cats else ""),
                "n_unique": len(cats),
            }
            continue
        cat2i = {c: i for i, c in enumerate(cats)}
        ids[f] = np.array([cat2i[v] for v in col], dtype=np.int64)
        vocab[f] = len(cats)
        vocab_labels[f] = cats
        fields.append(f)
        meta_scan["enabled"][f] = {"n_unique": len(cats), "values": cats}
    feat_meta = None
    if fields:
        feat_meta = {
            "fields": fields,
            "vocab": vocab,
            "ids": ids,
            "vocab_labels": vocab_labels,
        }
    return feat_meta, feat_ids, meta_scan


def load_selective_loss_mask(hdf):
    """用与 build_transitions 相同的弱先验构造 selective loss mask。"""
    feat_ids = hdf["feature_id"].tolist()
    info = {
        "used": False,
        "reason": None,
        "source": None,
        "n_loss_eligible_features": 0,
        "loss_eligible_features": [],
    }

    source = None
    cand = None
    if "reversible" in hdf.columns:
        cand = _truthy(hdf["reversible"])
        if cand.any():
            source = "feature_hierarchy.reversible==1"
    if source is None and "location" in hdf.columns:
        cand = hdf["location"].fillna("").astype(str).str.strip().eq("plasmid")
        if cand.any():
            source = "feature_hierarchy.location==plasmid"
    if source is None and "layer" in hdf.columns:
        cand = hdf["layer"].fillna("").astype(str).str.strip().eq("plasmid_arg")
        if cand.any():
            source = "feature_hierarchy.layer==plasmid_arg"

    if source is None:
        info["reason"] = "no reversible prior found in hierarchy"
        return np.zeros(len(hdf), dtype=np.float32), info

    mask = cand.fillna(False).astype(bool).to_numpy(dtype=np.float32)
    info.update({
        "used": True,
        "reason": "ok",
        "source": source,
        "n_loss_eligible_features": int(mask.sum()),
        "loss_eligible_features": [feat_ids[i] for i, ok in enumerate(mask) if ok],
    })
    return mask, info


def _validate_names(feat_ids, idxs, names, label):
    by_bits = {feat_ids[i] for i in idxs if 0 <= i < len(feat_ids)}
    if set(names) != by_bits:
        missing = set(names) - by_bits
        extra = by_bits - set(names)
        raise ValueError(
            "特征位序校验失败: "
            f"{label}_names={sorted(names)} vs bits->id={sorted(by_bits)}; "
            f"仅名字有={sorted(missing)}, 仅位号有={sorted(extra)}. "
            "请检查 feature_hierarchy.csv 与 transitions.csv 是否同一数据集/同一位序。")


def validate_feature_alignment(trans, feat_ids, n_check=500):
    """同时校验 gained / lost 名字与位号的一致性。"""
    checked = 0
    for t in trans:
        touched = False
        gained_names = t.get("gained_names") or []
        lost_names = t.get("lost_names") or []
        if gained_names:
            _validate_names(feat_ids, t["gained"], gained_names, "gained")
            touched = True
        if lost_names:
            _validate_names(feat_ids, t["lost"], lost_names, "lost")
            touched = True
        if touched:
            checked += 1
            if checked >= n_check:
                break
    return checked


def attach_feature_labels(rank_cols, order, feat_ids, hdf):
    if feat_ids is None or hdf is None:
        return rank_cols
    rank_cols["feature_id"] = [feat_ids[i] for i in order]
    for col in ("layer", "drug_class"):
        if col in hdf.columns:
            vals = hdf[col].tolist()
            rank_cols[col] = [vals[i] for i in order]
    return rank_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--arch", default="transformer", choices=["mlp", "transformer"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hierarchy", default=None,
                    help="feature_hierarchy.csv 路径 (默认自动定位 transitions 同数据集下)")
    ap.add_argument("--no-meta", action="store_true",
                    help="关闭方案A 特征元数据 token 化 (消融对照: 仅身份+状态)")
    ap.add_argument("--meta-fields", default="",
                    help="仅启用指定 metadata 字段, 逗号分隔: layer,drug_class,mechanism,location,reversible")
    args = ap.parse_args()

    requested_meta_fields = parse_meta_fields_arg(args.meta_fields)
    if args.no_meta and requested_meta_fields is not None:
        raise ValueError("--no-meta 与 --meta-fields 不能同时使用")

    os.makedirs(args.outdir, exist_ok=True)
    trans, Ln, trans_info = load_transitions(args.transitions)
    print("[train_dl] %d transitions, L=%d, arch=%s epochs=%d strategies=%s n_loss=%d"
          % (len(trans), Ln, args.arch, args.epochs,
             trans_info["strategies"], trans_info["n_loss_total"]))

    hpath = args.hierarchy or find_hierarchy_path(args.transitions)
    hdf = None
    feat_ids = None
    n_ok = 0
    if hpath and os.path.isfile(hpath):
        hdf = load_feature_hierarchy(hpath, Ln)
        feat_ids = hdf["feature_id"].tolist()
        n_ok = validate_feature_alignment(trans, feat_ids)
        print("[train_dl] hierarchy 对齐校验通过: %d 条" % n_ok)
    elif trans_info["selective_reversible_input"]:
        raise ValueError(
            "selective_reversible 训练需要 feature_hierarchy.csv 来构造 selective loss mask; "
            "请用 --hierarchy 指定或把文件放回数据集目录。")

    if args.no_meta:
        requested_fields_for_report = []
        requested_fields_source = "--no-meta"
    elif requested_meta_fields is None:
        requested_fields_for_report = list(META_FIELDS)
        requested_fields_source = "default_all"
    else:
        requested_fields_for_report = list(requested_meta_fields)
        requested_fields_source = "--meta-fields"

    feat_meta = None
    meta_info = {
        "used": False,
        "reason": None,
        "fields": [],
        "requested_fields": requested_fields_for_report,
        "requested_fields_source": requested_fields_source,
        "hierarchy": (None if hpath is None else os.path.abspath(hpath)),
        "enabled": {},
        "skipped": {},
        "n_alignment_checked": int(n_ok),
    }
    if args.no_meta:
        meta_info["reason"] = "disabled by --no-meta"
    elif args.arch != "transformer":
        meta_info["reason"] = f"arch={args.arch} 不使用元数据 (仅 transformer 生效)"
    elif hdf is None:
        meta_info["reason"] = "未找到 feature_hierarchy.csv, 退化为纯身份+状态"
        print("[train_dl][warn] %s" % meta_info["reason"])
    else:
        feat_meta, _, meta_scan = load_feature_meta(
            hdf, requested_meta_fields=requested_meta_fields)
        meta_info.update(
            enabled=meta_scan["enabled"],
            skipped=meta_scan["skipped"],
        )
        if feat_meta is None:
            meta_info["reason"] = "hierarchy 存在, 但所请求元数据字段在当前数据里都无区分度"
            print("[train_dl][warn] %s" % meta_info["reason"])
        else:
            meta_info.update(
                used=True,
                fields=feat_meta["fields"],
                vocab={k: int(v) for k, v in feat_meta["vocab"].items()},
            )
            print("[train_dl] 方案A 元数据已接入: fields=%s vocab=%s"
                  % (feat_meta["fields"], meta_info["vocab"]))
            if meta_info["skipped"]:
                print("[train_dl] 方案A 自动跳过无信息量字段: %s"
                      % meta_info["skipped"])

    loss_mask = None
    head_mode = "gain_only"
    loss_info = {
        "requested": bool(trans_info["selective_reversible_input"]),
        "used": False,
        "reason": None,
        "source": None,
        "hierarchy": (None if hpath is None else os.path.abspath(hpath)),
        "n_alignment_checked": int(n_ok),
        "n_loss_total": int(trans_info["n_loss_total"]),
        "n_loss_transitions": int(trans_info["n_loss_transitions"]),
        "n_loss_eligible_features": 0,
        "loss_eligible_features": [],
    }
    if trans_info["selective_reversible_input"]:
        if args.arch != "transformer":
            raise ValueError(
                "V1 selective-reversible dual-head 当前仅支持 --arch transformer")
        loss_mask, loss_scan = load_selective_loss_mask(hdf)
        loss_info.update(loss_scan)
        if trans_info["n_loss_total"] > 0 and int(loss_mask.sum()) <= 0:
            raise ValueError(
                "transition 中检测到 loss, 但 hierarchy 未找到可逆先验; "
                "请检查 transitions_selective_reversible.csv 与 feature_hierarchy.csv 是否匹配。")
        if int(loss_mask.sum()) > 0:
            head_mode = "dual_head_selective_reversible"
            print("[train_dl] V1 selective-reversible 已启用: source=%s n=%d"
                  % (loss_info["source"], loss_info["n_loss_eligible_features"]))
        else:
            loss_info["reason"] = (
                "selective_reversible 输入, 但 hierarchy 中未找到可逆先验 -> 退化为 gain-only")
            print("[train_dl][warn] %s" % loss_info["reason"])
    else:
        loss_info["reason"] = "irreversible/gain-only input"

    m = MDL.SetPredDL(
        Ln,
        arch=args.arch,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        verbose=True,
        feat_meta=feat_meta,
        head_mode=head_mode,
        loss_mask=loss_mask,
    )
    m.fit(trans)

    loss_info.update({
        "effective_head_mode": m.fit_report.get("head_mode"),
        "loss_head_enabled": bool(m.fit_report.get("loss_head_enabled", False)),
        "loss_head_disable_reason": m.fit_report.get("loss_head_disable_reason"),
    })

    model_path = os.path.join(args.outdir, "model.pt")
    m.save(model_path)

    with open(os.path.join(args.outdir, "fit_report.json"), "w") as f:
        json.dump({
            "L": Ln,
            "n_transitions": len(trans),
            "arch": args.arch,
            "epochs": args.epochs,
            "transition_input": trans_info,
            "feat_meta": meta_info,
            "loss_head": loss_info,
            **m.fit_report,
        }, f, indent=2, ensure_ascii=False)

    s0 = np.zeros(Ln, dtype=np.int8)
    gain_scores = m.predict_fn(s0)
    order = np.argsort(-gain_scores)
    rank_cols = {
        "feature_index": order,
        "score_from_empty": gain_scores[order],
        "rank": np.arange(1, Ln + 1),
    }
    rank_cols = attach_feature_labels(rank_cols, order, feat_ids, hdf)
    pd.DataFrame(rank_cols).to_csv(
        os.path.join(args.outdir, "first_step_ranking.csv"), index=False)

    if m.fit_report.get("loss_head_enabled"):
        s_loss = np.zeros(Ln, dtype=np.int8)
        loss_idx = np.where(np.asarray(loss_mask) > 0)[0]
        s_loss[loss_idx] = 1
        loss_scores = m.predict_loss_fn(s_loss)
        loss_order = loss_idx[np.argsort(-loss_scores[loss_idx])]
        loss_cols = {
            "feature_index": loss_order,
            "loss_score_from_reversible_present": loss_scores[loss_order],
            "rank": np.arange(1, len(loss_order) + 1),
        }
        loss_cols = attach_feature_labels(loss_cols, loss_order, feat_ids, hdf)
        pd.DataFrame(loss_cols).to_csv(
            os.path.join(args.outdir, "loss_step_ranking.csv"), index=False)

    print("[train_dl] saved model : %s" % model_path)
    print("[train_dl] fit_report  : %s" % m.fit_report)
    if feat_ids is not None:
        print("[train_dl] first_step_ranking top5: %s"
              % [feat_ids[i] for i in order[:5]])
    else:
        print("[train_dl] first_step_ranking top5 feature_index: %s"
              % order[:5].tolist())

    if m.fit_report.get("loss_head_enabled"):
        loss_path = os.path.join(args.outdir, "loss_step_ranking.csv")
        if feat_ids is not None:
            print("[train_dl] loss_step_ranking top5: %s"
                  % [feat_ids[i] for i in loss_order[:5]])
        else:
            print("[train_dl] loss_step_ranking top5 feature_index: %s"
                  % loss_order[:5].tolist())
        print("[train_dl] loss_step_ranking : %s" % loss_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
