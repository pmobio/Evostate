#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluator.py  —— 统一评价器 (所有模型同口径)

评价对象: 每个 transition 的"下一步预测"。每个模型对给定源状态 s 输出候选特征
的一个概率分布 P(next=i | s)。评价器据此算指标。

双通道 (方案B) 评估口径:
  - gain 侧: 候选 = s 中尚未获得的位 (s_i == 0), 真值 = t["gained"]
  - loss 侧: 候选 = s 中已存在且属可逆先验子集的位 (s_i == 1 且 loss_mask_i),
             真值 = t["lost"]
两侧各自独立归一化 (两次 softmax/归一), 因此 gain 侧指标与历史 gain-only 结果
口径完全一致, 向后兼容; loss 侧作为新增能力单独报告。

注意: loss 侧的 1->0 解释为"端点净状态消失/回退", 不等于已确证的回复突变事件
(与 build_transitions.py 的声明一致)。

模型以一个可调用对象接入:
    predict_fn(src_state: np.int8[L]) -> np.ndarray[L]  (对所有位的非负分数)
评价器内部只在对应侧的候选位上归一化, 保证各模型口径一致。

指标(层1, 全部并列报告, 不设首要):
  - logL   : 真实下一步特征的 log P (在候选上归一化后)
  - top1/top3/top5 : 命中率
  - mrr: 平均倒数排名
  - auc    : 候选内区分"被获取vs 未获取"的 AUC

|G|>1 的边处理:
  - 主指标(clean): 只用 |G|=1 的边(干净单步真值)
  - 辅助(multi) : |G|>1 的边, 命中口径为"G 中任一进入top-k 即算命中"

划分:
  - 谱系分组 CV (group K-fold, 按 child 所属进化枝整组进出) —— 主
  - 随机 K 折 —— 泄漏对照

显著性:
  - 对同一测试集的per-item logL 做 paired bootstrap, 比较两模型
"""
from __future__ import annotations

import numpy as np


GAIN = "gain"
LOSS = "loss"
COMBINED = "combined"
TRUTH_KEY = {GAIN: "gained", LOSS: "lost"}


def _all_edge_ranking_metrics(p, cand_idx, truth, topks):
    """在"所有含事件的边"上计算排序质量, 不再只用 |T|==1 的干净边。

    动机: 干净边 (恰好 1 个事件) 在多事件数据上极其稀少 —— ecoli2026 full_temporal
    每条边平均约 19 个事件, 全库只有 6 条 combined 干净边, 导致 clean 口径的
    logL/top-k/MRR 完全没有统计功效。而把 multi 边压成一个二值 multi_top-k 又浪费了
    每条边约 19 个真事件所含的排序信息。

    这里返回三类不依赖"单事件"假设的量:
      event_rr : 每个真事件各自的倒数排名 (微平均, 事件级样本量)
      rec@k    : |T ∩ top-k| / |T|      (该边真事件被 top-k 覆盖的比例)
      prec@k   : |T ∩ top-k| / k        (top-k 中命中真事件的比例)
      auc_all  : 一对多 AUC, 正类=全部真事件, 负类=其余候选 (与候选集大小无关)
    """
    from sklearn.metrics import roc_auc_score

    order = cand_idx[np.argsort(-p[cand_idx])]
    order_list = order.tolist()
    pos = {int(x) for x in truth}
    rank_of = {int(e): i + 1 for i, e in enumerate(order_list)}

    event_rr = [1.0 / rank_of[e] for e in pos if e in rank_of]

    prec, rec = {}, {}
    n_pos = max(1, len(pos))
    for k in topks:
        hit = len(pos & set(order_list[:k]))
        prec[k] = hit / float(k)
        rec[k] = hit / float(n_pos)

    auc_all = None
    if len(cand_idx) >= 2:
        y = np.array([1 if int(i) in pos else 0 for i in cand_idx])
        if 0 < int(y.sum()) < len(y):
            try:
                auc_all = float(roc_auc_score(y, p[cand_idx]))
            except ValueError:
                auc_all = None
    return event_rr, prec, rec, auc_all


def _candidate_mask(src, side=GAIN, loss_mask=None):
    """候选位 bool mask。

    side="gain": 尚未获得(==0)的位 (历史口径, 默认)。
    side="loss": 当前已存在(==1) 且落在可逆先验子集内的位。
    """
    src = np.asarray(src)
    if side == GAIN:
        return src == 0
    if side == LOSS:
        mask = src == 1
        if loss_mask is not None:
            mask = mask & (np.asarray(loss_mask).reshape(-1) > 0)
        return mask
    raise ValueError(f"未知评估侧 side={side} (仅支持 'gain'/'loss')")


def _normalize_on_candidates(scores, src, side=GAIN, loss_mask=None):
    mask = _candidate_mask(src, side=side, loss_mask=loss_mask)
    s = np.array(scores, dtype=float)
    s = np.where(mask, s, 0.0)
    total = s.sum()
    if total <= 0:
        # 退化: 候选上均匀
        p = np.zeros_like(s)
        if mask.sum() > 0:
            p[mask] = 1.0 / mask.sum()
        return p
    return s / total


def eval_transitions(trans, predict_fn, eps=1e-12, topks=(1, 3, 5),
                     side=GAIN, loss_mask=None):
    """
    对一组 transition 计算 per-item 指标。返回 dict:
      per_item_logl(list), hits{k:list}, rr(list), auc(list), n_clean, n_multi

    side="gain" (默认, 与历史行为逐位一致): 真值取 t["gained"], 候选 s_i==0。
    side="loss": 真值取 t["lost"], 候选 s_i==1 且属 loss_mask。

    只在 |truth|==1 的边上累计主指标; |truth|>1 记入 multi 辅助命中;
    |truth|==0 的边在该侧被跳过 (不计入 n_clean/n_multi)。
    """
    from sklearn.metrics import roc_auc_score

    if side not in TRUTH_KEY:
        raise ValueError(f"未知评估侧 side={side} (仅支持 'gain'/'loss')")
    truth_key = TRUTH_KEY[side]

    per_logl = []
    hits = {k: [] for k in topks}
    rr = []
    aucs = []
    multi_hits = {k: [] for k in topks}
    cand_sizes = []
    # 全边口径 (不限 |T|==1): 事件级 RR / 覆盖率 / 一对多 AUC
    all_event_rr = []
    all_prec = {k: [] for k in topks}
    all_rec = {k: [] for k in topks}
    all_aucs = []
    n_events_scored = 0
    n_edges_scored = 0
    n_clean = 0
    n_multi = 0
    n_skipped_no_truth = 0
    n_skipped_no_candidate = 0
    n_skipped_truth_outside_candidates = 0

    for t in trans:
        src = t["src"]
        truth = list(t.get(truth_key, []) or [])
        if not truth:
            n_skipped_no_truth += 1
            continue

        mask = _candidate_mask(src, side=side, loss_mask=loss_mask)
        cand_idx = np.where(mask)[0]
        if len(cand_idx) == 0:
            n_skipped_no_candidate += 1
            continue
        # 真值必须落在候选集合内; 否则该侧无法评估 (通常说明 loss_mask 与数据不一致)
        if not set(truth).issubset(set(cand_idx.tolist())):
            n_skipped_truth_outside_candidates += 1
            continue

        p = _normalize_on_candidates(
            predict_fn(src), src, side=side, loss_mask=loss_mask)
        # 排名(降序)
        order = cand_idx[np.argsort(-p[cand_idx])]

        # 全边口径: 每条含事件的边都参与, 提供足够的统计功效
        e_rr, e_prec, e_rec, e_auc = _all_edge_ranking_metrics(
            p, cand_idx, truth, topks)
        all_event_rr.extend(e_rr)
        n_events_scored += len(e_rr)
        n_edges_scored += 1
        for k in topks:
            all_prec[k].append(e_prec[k])
            all_rec[k].append(e_rec[k])
        if e_auc is not None:
            all_aucs.append(e_auc)

        if len(truth) == 1:
            n_clean += 1
            cand_sizes.append(int(len(cand_idx)))
            g = truth[0]
            per_logl.append(float(np.log(p[g] + eps)))
            rank = int(np.where(order == g)[0][0]) + 1
            rr.append(1.0 / rank)
            for k in topks:
                hits[k].append(1.0 if rank <= k else 0.0)
            # AUC: 候选内, 正类=发生事件的位(仅 g), 负类=其余候选
            if len(cand_idx) >= 2:
                y = np.array([1 if i == g else 0 for i in cand_idx])
                try:
                    aucs.append(roc_auc_score(y, p[cand_idx]))
                except ValueError:
                    pass
        else:
            n_multi += 1
            gset = set(truth)
            for k in topks:
                topk = set(order[:k].tolist())
                multi_hits[k].append(1.0 if gset & topk else 0.0)

    return {
        "side": side,
        "per_item_logl": per_logl,
        "hits": hits,
        "rr": rr,
        "auc": aucs,
        "multi_hits": multi_hits,
        "n_candidates": cand_sizes,
        "n_clean": n_clean,
        "n_multi": n_multi,
        "all_event_rr": all_event_rr,
        "all_prec": all_prec,
        "all_rec": all_rec,
        "all_auc": all_aucs,
        "n_events_scored": n_events_scored,
        "n_edges_scored": n_edges_scored,
        "n_skipped_no_truth": n_skipped_no_truth,
        "n_skipped_no_candidate": n_skipped_no_candidate,
        "n_skipped_truth_outside_candidates": n_skipped_truth_outside_candidates,
    }


def estimate_gain_event_share(trans, default=1.0):
    """从一组转移估计 side 混合权重 pi = #gain事件 / (#gain事件 + #loss事件)。

    以"事件"(标签) 而非"边"为单位计数, 因为综合评估预测的是"下一个事件"。
    无任何事件时返回 default。gain-only 数据 (无 loss 标签) 恒返回 1.0, 此时
    综合评估与 gain 侧评估完全等价。
    """
    n_gain = sum(len(t.get("gained", []) or []) for t in trans)
    n_loss = sum(len(t.get("lost", []) or []) for t in trans)
    total = n_gain + n_loss
    if total <= 0:
        return float(default)
    return float(n_gain) / float(total)


def eval_transitions_combined(trans, gain_predict_fn, loss_predict_fn=None,
                              loss_mask=None, pi=1.0, eps=1e-12, topks=(1, 3, 5)):
    """在统一事件空间上评估双头模型, 输出单一综合性能。

    事件空间 (最多 2L 个候选):
        (gain, i)  当 s_i == 0
        (loss, j)  当 s_j == 1 且 j 属可逆先验 loss_mask
    索引编码: gain 事件用 i, loss 事件用 L + j。

    概率构造 (两个 head 各自归一, 再用 side 权重 pi 混合成 C 上的合法分布):
        P(gain, i) = pi       * P_g(i),  P_g 在 gain 候选上归一
        P(loss, j) = (1 - pi) * P_l(j),  P_l 在 loss 候选上归一
    因此 sum_C P = pi + (1 - pi) = 1。

    pi 必须由外部给定 (建议用训练折的 estimate_gain_event_share), 不能默认 0.5:
    直接拼接两侧分数等价于隐式假设 gain/loss 各占一半, 属无依据的先验。

    退化性质: loss 候选为空 或 pi==1 时, 本函数与 eval_transitions(side="gain")
    完全等价, 因此 gain-only 数据集上综合指标 == 历史 gain 侧指标。
    """
    from sklearn.metrics import roc_auc_score

    pi = float(pi)
    if not (0.0 <= pi <= 1.0):
        raise ValueError(f"pi 必须在 [0,1] 内, 收到 {pi}")
    if pi < 1.0 and loss_predict_fn is None:
        raise ValueError("pi < 1 时必须提供 loss_predict_fn")

    per_logl = []
    hits = {k: [] for k in topks}
    rr = []
    aucs = []
    multi_hits = {k: [] for k in topks}
    cand_sizes = []
    all_event_rr = []
    all_prec = {k: [] for k in topks}
    all_rec = {k: [] for k in topks}
    all_aucs = []
    n_events_scored = 0
    n_edges_scored = 0
    n_clean = 0
    n_multi = 0
    n_skipped_no_truth = 0
    n_skipped_no_candidate = 0
    n_clean_gain = 0
    n_clean_loss = 0

    for t in trans:
        src = np.asarray(t["src"])
        L = src.shape[0]
        gain_mask = _candidate_mask(src, side=GAIN)
        loss_cand_mask = _candidate_mask(src, side=LOSS, loss_mask=loss_mask)

        gained = [int(i) for i in (t.get("gained", []) or [])]
        lost = [int(j) for j in (t.get("lost", []) or [])]
        # 真值统一编码到 [0, 2L)
        truth = [i for i in gained] + [L + j for j in lost]
        if not truth:
            n_skipped_no_truth += 1
            continue

        p = np.zeros(2 * L, dtype=float)
        if gain_mask.sum() > 0 and pi > 0.0:
            p[:L] = pi * _normalize_on_candidates(
                gain_predict_fn(src), src, side=GAIN)
        if loss_cand_mask.sum() > 0 and pi < 1.0:
            p[L:] = (1.0 - pi) * _normalize_on_candidates(
                loss_predict_fn(src), src, side=LOSS, loss_mask=loss_mask)

        cand = np.concatenate([gain_mask, loss_cand_mask])
        # pi 落在端点时, 被赋权 0 的一侧不应参与排名
        if pi >= 1.0:
            cand[L:] = False
        elif pi <= 0.0:
            cand[:L] = False
        cand_idx = np.where(cand)[0]
        if len(cand_idx) == 0:
            n_skipped_no_candidate += 1
            continue
        # 真值必须落在候选内; 越界通常说明 loss_mask 与数据不一致
        truth = [e for e in truth if cand[e]]
        if not truth:
            n_skipped_no_candidate += 1
            continue

        order = cand_idx[np.argsort(-p[cand_idx])]

        # 全边口径: combined 的干净边 (整条边恰好 1 个事件) 在多事件数据上几乎不存在,
        # 因此这里的事件级指标才是 combined 通道可用的主指标
        e_rr, e_prec, e_rec, e_auc = _all_edge_ranking_metrics(
            p, cand_idx, truth, topks)
        all_event_rr.extend(e_rr)
        n_events_scored += len(e_rr)
        n_edges_scored += 1
        for k in topks:
            all_prec[k].append(e_prec[k])
            all_rec[k].append(e_rec[k])
        if e_auc is not None:
            all_aucs.append(e_auc)

        if len(truth) == 1:
            n_clean += 1
            cand_sizes.append(int(len(cand_idx)))
            e = truth[0]
            if e < L:
                n_clean_gain += 1
            else:
                n_clean_loss += 1
            per_logl.append(float(np.log(p[e] + eps)))
            rank = int(np.where(order == e)[0][0]) + 1
            rr.append(1.0 / rank)
            for k in topks:
                hits[k].append(1.0 if rank <= k else 0.0)
            if len(cand_idx) >= 2:
                y = np.array([1 if i == e else 0 for i in cand_idx])
                try:
                    aucs.append(roc_auc_score(y, p[cand_idx]))
                except ValueError:
                    pass
        else:
            n_multi += 1
            tset = set(truth)
            for k in topks:
                if tset & set(order[:k].tolist()):
                    multi_hits[k].append(1.0)
                else:
                    multi_hits[k].append(0.0)

    return {
        "side": COMBINED,
        "pi_gain": pi,
        "per_item_logl": per_logl,
        "hits": hits,
        "rr": rr,
        "auc": aucs,
        "multi_hits": multi_hits,
        "n_candidates": cand_sizes,
        "all_event_rr": all_event_rr,
        "all_prec": all_prec,
        "all_rec": all_rec,
        "all_auc": all_aucs,
        "n_events_scored": n_events_scored,
        "n_edges_scored": n_edges_scored,
        "n_clean": n_clean,
        "n_multi": n_multi,
        "n_clean_gain_event": n_clean_gain,
        "n_clean_loss_event": n_clean_loss,
        "n_skipped_no_truth": n_skipped_no_truth,
        "n_skipped_no_candidate": n_skipped_no_candidate,
        "n_skipped_truth_outside_candidates": 0,
    }


def summarize(ev, topks=(1, 3, 5)):
    """把 per-item 指标聚合成标量。"""
    out = {}
    out["mean_logl"] = float(np.mean(ev["per_item_logl"])) if ev["per_item_logl"] else float("nan")
    for k in topks:
        out[f"top{k}"] = float(np.mean(ev["hits"][k])) if ev["hits"][k] else float("nan")
    out["mrr"] = float(np.mean(ev["rr"])) if ev["rr"] else float("nan")
    out["auc"] = float(np.mean(ev["auc"])) if ev["auc"] else float("nan")
    for k in topks:
        out[f"multi_top{k}"] = float(np.mean(ev["multi_hits"][k])) if ev["multi_hits"][k] else float("nan")
    out["n_clean"] = ev["n_clean"]
    out["n_multi"] = ev["n_multi"]
    cand = ev.get("n_candidates") or []
    out["mean_n_candidates"] = float(np.mean(cand)) if cand else float("nan")

    # ---- 全边/事件级指标: 不依赖"单事件干净边", 统计功效高得多 ----
    # 多事件数据 (如 ecoli2026 full_temporal 每边约 19 个事件) 上, clean 口径样本量
    # 可能只有个位数甚至 0, 应以这组指标为主。
    ev_rr = ev.get("all_event_rr") or []
    out["event_mrr"] = float(np.mean(ev_rr)) if ev_rr else float("nan")
    auc_all = ev.get("all_auc") or []
    out["auc_all"] = float(np.mean(auc_all)) if auc_all else float("nan")
    for k in topks:
        pk = (ev.get("all_prec") or {}).get(k) or []
        rk = (ev.get("all_rec") or {}).get(k) or []
        out[f"prec{k}"] = float(np.mean(pk)) if pk else float("nan")
        out[f"rec{k}"] = float(np.mean(rk)) if rk else float("nan")
    out["n_events_scored"] = int(ev.get("n_events_scored", 0) or 0)
    out["n_edges_scored"] = int(ev.get("n_edges_scored", 0) or 0)
    if "side" in ev:
        out["side"] = ev["side"]
    if "pi_gain" in ev:
        out["pi_gain"] = float(ev["pi_gain"])
    for key in ("n_skipped_no_truth", "n_skipped_no_candidate",
                "n_skipped_truth_outside_candidates",
                "n_clean_gain_event", "n_clean_loss_event"):
        if key in ev:
            out[key] = int(ev[key])
    return out


def group_kfold_indices(group_labels, n_splits, seed=0):
    """
    谱系分组 K 折: 把不同 group 分配到 K 折(整组进出)。
    转移集中的大 group 会被拆成多个"虚拟子组"以均衡每折样本量。
    返回 list of (train_idx, test_idx)。
    """
    rng = np.random.default_rng(seed)
    group_labels = np.asarray(group_labels)
    n = len(group_labels)
    # 把每个 group 的样本索引收集; 大 group 切成子块以均衡
    from collections import defaultdict

    gidx = defaultdict(list)
    for i, g in enumerate(group_labels):
        gidx[g].append(i)

    target = max(1, n // (n_splits * 3))  # 每个子块目标大小
    blocks = []
    for g, idxs in gidx.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        if len(idxs) <= target:
            blocks.append(idxs)
        else:
            for j in range(0, len(idxs), target):
                blocks.append(idxs[j : j + target])
    rng.shuffle(blocks)
    # 贪心均衡地把block 分到 K 折
    fold_sizes = [0] * n_splits
    fold_members = [[] for _ in range(n_splits)]
    for blk in sorted(blocks, key=len, reverse=True):
        f = int(np.argmin(fold_sizes))
        fold_members[f].extend(blk)
        fold_sizes[f] += len(blk)

    splits = []
    all_idx = set(range(n))
    for f in range(n_splits):
        test = sorted(fold_members[f])
        train = sorted(all_idx - set(test))
        if test and train:
            splits.append((np.array(train), np.array(test)))
    return splits


def random_kfold_indices(n, n_splits, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    splits = []
    for f in range(n_splits):
        test = np.sort(folds[f])
        train = np.sort(np.concatenate([folds[j] for j in range(n_splits) if j != f]))
        splits.append((train, test))
    return splits


def paired_bootstrap_logl(logl_a, logl_b, n_boot=2000, seed=0):
    """
    对齐的 per-item logL(同一批测试转移, 同顺序)做配对 bootstrap。
    返回 (mean_diff = a-b, p_value 双侧)。要求 logl_a/b 一一对应。
    """
    a = np.asarray(logl_a, dtype=float)
    b = np.asarray(logl_b, dtype=float)
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]
    diff = a - b
    obs = diff.mean()
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        samp = rng.integers(0, m, m)
        boots[i] = diff[samp].mean()
    # 双侧 p: bootstrap 分布跨过 0 的比例
    if obs >= 0:
        p = 2.0 * np.mean(boots <= 0)
    else:
        p = 2.0 * np.mean(boots >= 0)
    return float(obs), float(min(1.0, p))
