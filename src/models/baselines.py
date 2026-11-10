#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baselines.py —— 基线 next-step预测器

每个基线暴露一个 predict_fn(src_state: np.int8[L]) -> scores[L] (非负, 评价器在候选上归一化)。

  B0 UniformBaseline    : 候选内均匀 (随机下限)
  B1 PrevalenceBaseline : 按特征在训练集出现频率打分 (忽略当前状态) —— 关键坎
  B2/B3 HyperTraPSEdgeBaseline : 直接用 HyperTraPS 随机游走的 edge-list-long 文件,
        state 用bitmask 整数编码, 边权 weight = 从 s 获取某特征的经验频次。
        这样直接对接你已跑通的 HyperTraPS 输出, 不碰内部 θ 排列。

位序约定(与 HyperTraPS 输出一致, 已在数据上校验):
  状态整数 S的第 i 位(bit i,权重 2^i) = 特征 i 是否已获得。
"""
from __future__ import annotations

import numpy as np


class UniformBaseline:
    name = "B0_uniform"

    def __init__(self, L):
        self.L = L

    def fit(self, trans):
        return self

    def predict_fn(self, src):
        return np.ones(self.L, dtype=float)


class PrevalenceBaseline:
    name = "B1_prevalence"

    def __init__(self, L):
        self.L = L
        self.freq = np.ones(L)

    def fit(self, trans):
        # 用训练转移里"被获取特征"的频次作为流行率打分
        cnt = np.zeros(self.L)
        for t in trans:
            for g in t["gained"]:
                cnt[g] += 1
        self.freq = cnt + 1e-6  # 平滑, 避免全0
        return self

    def predict_fn(self, src):
        return self.freq.copy()


class LossPrevalenceBaseline:
    """loss 侧的流行率基线: 用训练转移里"被丢失特征"的频次打分。

    与 PrevalenceBaseline 同构, 只是统计对象从 gained 换成 lost, 使 loss 侧
    有一个与 gain 侧口径一致的"忽略状态的边际频率"对照。
    """

    name = "B1_prevalence"

    def __init__(self, L):
        self.L = L
        self.freq = np.ones(L)

    def fit(self, trans):
        cnt = np.zeros(self.L)
        for t in trans:
            for l in t.get("lost", []) or []:
                if 0 <= l < self.L:
                    cnt[l] += 1
        self.freq = cnt + 1e-6  # 平滑, 避免全0
        return self

    def predict_fn(self, src):
        return self.freq.copy()


def state_to_int(arr):
    """0/1 array -> bitmask int, bit i = 特征 i。"""
    s = 0
    for i, v in enumerate(arr):
        if v:
            s |= (1 << i)
    return s


class HyperTraPSEdgeBaseline:
    """
    从 HyperTraPS 的 edge-list-long 文件构建 P(next|s)。
    文件格式(带表头): 'from to weight', from/to 为状态 bitmask 整数。
    从 s 获取特征 i 的分数 = weight(from=S, to=S|1<<i)。
    """

    def __init__(self, L, edge_file, name="B3_hypertraps"):
        self.L = L
        self.name = name
        # from_state -> {feature_i: weight}
        self.trans_w = {}
        with open(edge_file) as f:
            header = f.readline()  # skip 'from to weight'
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a, b, w = line.split()
                a, b, w = int(a), int(b), float(w)
                diff = b & ~a  # b 相比 a 新增的 bit
                # 只处理单特征获取(diff 是2的幂)
                if diff and (diff & (diff - 1)) == 0:
                    i = diff.bit_length() - 1
                    self.trans_w.setdefault(a, {})[i] = self.trans_w.setdefault(a, {}).get(i, 0.0) + w

    def fit(self, trans):
        return self

    def predict_fn(self, src):
        S = state_to_int(src)
        scores = np.zeros(self.L, dtype=float)
        wmap = self.trans_w.get(S)
        if wmap:
            for i, w in wmap.items():
                if i < self.L:
                    scores[i] = w
        # 若该状态在游走中未出现, 退化为均匀(交给评价器在候选上归一)
        if scores.sum() <= 0:
            scores = np.ones(self.L, dtype=float)
        return scores


class HyperTraPSThetaBaseline:
    """
    基于 HyperTraPS 的 second-order θ 后验(forwards.txt)直接算P(next|s)。

    速率公式(已从 C++ updater_fast.cpp 确认, 行98-102):
        rate(获取特征 j | 状态 s) = exp( F[j][j] + sum_{i in s} F[j][i] )
    forwards.txt 每行: iter + L*L 个值, 按行展平 F[0][0..L-1], F[1][...], ...
    多个后验样本 -> 对每个样本算候选上的归一概率, 再对样本求平均(后验预测)。

    公平性说明: forwards.txt 是全数据 MCMC 后验。严格每折公平需在每折训练集重跑
    MCMC; 本类用全数据 θ 属"轻度乐观"基线, 已在文档标注 caveat。
    """

    def __init__(self, L, forwards_file, name="B3_hypertraps_theta", max_samples=300):
        self.L = L
        self.name = name
        self.samples = []  # list of F matrix (L x L)
        with open(forwards_file) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                vals = [float(x) for x in parts]
                # 去掉首列迭代号(若列数 =1 + L*L)
                if len(vals) ==1 + L * L:
                    vals = vals[1:]
                elif len(vals) != L * L:
                    continue
                F = np.array(vals, dtype=float).reshape(L, L)
                self.samples.append(F)
        if max_samples and len(self.samples) > max_samples:
            idx = np.linspace(0, len(self.samples) - 1, max_samples).astype(int)
            self.samples = [self.samples[i] for i in idx]

    def fit(self, trans):
        return self

    def predict_fn(self, src):
        s = src.astype(float)
        cand = src == 0
        acc = np.zeros(self.L, dtype=float)
        for F in self.samples:
            # rate_j = exp(F[j][j] + sum_{i in s} F[j][i]) = exp(F[j] . s + F[j][j])
            logr = F.diagonal() + F.dot(s)
            logr = np.clip(logr, -30, 30)
            r = np.exp(logr)
            r = np.where(cand, r, 0.0)
            tot = r.sum()
            if tot > 0:
                acc += r / tot
        if acc.sum() <= 0:
            acc = np.where(cand, 1.0, 0.0)
        return acc
