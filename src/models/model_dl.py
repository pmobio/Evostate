#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_dl.py —— 深度学习集合预测模型 (对齐 docs/05)

任务:
  - gain head: 给定源状态 s in {0,1}^L, 预测"下一条演化边一次性获得的特征集合 G"
  - V1 selective-reversible / dual-head:
      在 gain 之外, 再对选择性可逆子集预测"端点净消失/回退"集合 L_loss。
      loss head 只在 loss_mask 指定的可逆先验子集上启用, 且候选仅限当前已存在(s_j==1)的位。
      这里的 1→0 解释为 sampled branch 两端的净消失, 不是已知真实时间线事件。

接口:
    m = SetPredDL(L, arch="transformer", ...).fit(trans)
    m.predict_fn(src) -> gain scores[L]            # 向后兼容旧 evaluator
    m.predict_heads(src) -> {"gain": ..., "loss": ...}

架构:
    - "mlp"        : 状态向量 -> MLP -> L 个 gain logit (最简基线; 不吃元数据)
    - "transformer": 特征 token 化 + self-attention
                     可选升级为 V1 dual-head selective-reversible

方案A (特征元数据 token 化, 仅 transformer 生效):
    token_j = feat_emb[j] + state_emb[s_j]
            + layer_emb + class_emb + mech_emb + loc_emb + rev_emb
    feat_meta=None 时退化为原始"身份+状态"两路 embedding (向后兼容)。

设计红线:
    仅用 barcode 状态 + 特征内生分子属性作输入; 环境协变量/时间不进模型。
"""
from __future__ import annotations

import numpy as np


class SetPredDL:
    def __init__(self, L, arch="transformer", d_model=64, n_layers=2, n_heads=4,
                 dropout=0.3, weight_decay=1e-3, lr=1e-3, epochs=200, batch_size=64,
                 val_frac=0.15, patience=20, seed=0, name=None, verbose=False,
                 feat_meta=None, head_mode="gain_only", loss_mask=None, loss_weight=1.0):
        self.L = int(L)
        self.arch = arch
        self.feat_meta = feat_meta
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.patience = patience
        self.seed = seed
        self.name = name or f"DL_{arch}"
        self.verbose = verbose
        self.net = None
        self._torch = None
        self.fit_report = {}

        if head_mode not in {"gain_only", "dual_head_selective_reversible"}:
            raise ValueError(f"未知 head_mode={head_mode}")
        if head_mode == "dual_head_selective_reversible" and arch != "transformer":
            raise ValueError("V1 selective-reversible dual-head 当前仅支持 arch=transformer")
        self.head_mode = head_mode               # 用户请求/配置
        self._effective_head_mode = head_mode    # 实际生效 (fit 时可能降级)
        self.loss_weight = float(loss_weight)

        if loss_mask is None:
            self.loss_mask = None
        else:
            arr = np.asarray(loss_mask, dtype=np.float32).reshape(-1)
            if arr.shape[0] != self.L:
                raise ValueError(
                    f"loss_mask 长度({arr.shape[0]}) 与 L({self.L}) 不一致")
            arr = (arr > 0).astype(np.float32)
            self.loss_mask = arr
        if self.head_mode == "dual_head_selective_reversible":
            if self.loss_mask is None or int(self.loss_mask.sum()) <= 0:
                raise ValueError(
                    "dual_head_selective_reversible 需要非空 loss_mask "
                    "(选择性可逆特征子集)")

        self._fallback_gain = None
        self._fallback_loss = None

    # ------------------------------------------------------------------
    def _uses_dual_head(self):
        return self.arch == "transformer" and self._effective_head_mode == "dual_head_selective_reversible"

    def _split_logits(self, out):
        if isinstance(out, tuple):
            return out
        return out, None

    def _sigmoid(self, logits):
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

    # ------------------------------------------------------------------
    def _build_net(self):
        import torch
        import torch.nn as nn

        L, d = self.L, self.d_model
        arch = self.arch
        drop = self.dropout
        nheads, nlayers = self.n_heads, self.n_layers
        feat_meta = self.feat_meta
        dual_head = self._uses_dual_head()

        class MLPNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(L, d), nn.ReLU(), nn.Dropout(drop),
                    nn.Linear(d, d), nn.ReLU(), nn.Dropout(drop),
                    nn.Linear(d, L),
                )

            def forward(self, s):            # s: [B, L]
                return self.net(s)           # logits: [B, L]

        class TransformerNet(nn.Module):
            """每个特征是一个 token: embedding(特征身份) + 状态标记(已获得/候选)
            + 方案A 生物学元数据(layer/drug_class/mechanism/location/reversible)。
            dual_head=False: 仅输出 gain logits;
            dual_head=True : 同时输出 gain / loss 两个 head。"""
            def __init__(self):
                super().__init__()
                self.dual_head = dual_head
                self.feat_emb = nn.Embedding(L, d)
                self.state_emb = nn.Embedding(2, d)
                self.meta_fields = []
                self.meta_embs = nn.ModuleDict()
                if feat_meta is not None:
                    for f in feat_meta["fields"]:
                        vocab = int(feat_meta["vocab"][f])
                        self.meta_embs[f] = nn.Embedding(vocab, d)
                        self.register_buffer(
                            "meta_ids_" + f,
                            torch.as_tensor(feat_meta["ids"][f], dtype=torch.long))
                        self.meta_fields.append(f)
                enc = nn.TransformerEncoderLayer(
                    d_model=d, nhead=nheads, dim_feedforward=2 * d,
                    dropout=drop, batch_first=True)
                self.encoder = nn.TransformerEncoder(enc, num_layers=nlayers)
                if self.dual_head:
                    self.gain_head = nn.Sequential(
                        nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, 1))
                    self.loss_head = nn.Sequential(
                        nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, 1))
                else:
                    self.head = nn.Sequential(
                        nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, 1))
                self.register_buffer("feat_ids", torch.arange(L))

            def forward(self, s):            # s: [B, L] (0/1)
                B = s.shape[0]
                tok = (self.feat_emb(self.feat_ids).unsqueeze(0).expand(B, -1, -1)
                       + self.state_emb(s.long()))
                for f in self.meta_fields:
                    ids = getattr(self, "meta_ids_" + f)
                    tok = tok + self.meta_embs[f](ids).unsqueeze(0)
                h = self.encoder(tok)
                if self.dual_head:
                    return (self.gain_head(h).squeeze(-1),
                            self.loss_head(h).squeeze(-1))
                return self.head(h).squeeze(-1)

        if arch == "mlp":
            return MLPNet()
        if arch == "transformer":
            return TransformerNet()
        raise ValueError(f"未知 arch={arch} (支持 mlp/transformer; gnn 待接入)")

    # ------------------------------------------------------------------
    def _prepare(self, trans):
        """trans -> X, Y_gain, M_gain, Y_loss, M_loss。
        gain 候选: s_j==0
        loss 候选: s_j==1 且 j 属于 selective loss_mask
        """
        X, Yg, Mg, Yl, Ml = [], [], [], [], []
        use_loss = self.head_mode == "dual_head_selective_reversible"
        loss_mask = (np.zeros(self.L, dtype=np.float32)
                     if self.loss_mask is None else self.loss_mask.astype(np.float32))

        n_gain_labels = 0
        n_loss_labels = 0
        n_loss_transitions = 0

        for t in trans:
            s = np.asarray(t["src"], dtype=np.float32)
            if s.shape[0] != self.L:
                continue

            yg = np.zeros(self.L, dtype=np.float32)
            yl = np.zeros(self.L, dtype=np.float32)
            for g in t.get("gained", []):
                if 0 <= g < self.L:
                    yg[g] = 1.0
            for l in t.get("lost", []):
                if 0 <= l < self.L:
                    yl[l] = 1.0

            mg = (s == 0).astype(np.float32)
            if use_loss:
                ml = ((s == 1).astype(np.float32) * loss_mask).astype(np.float32)
                bad_loss = np.where((yl > 0) & (ml <= 0))[0]
                if len(bad_loss):
                    raise ValueError(
                        "检测到 loss 标签落在非法候选位上: "
                        f"indices={bad_loss.tolist()} | 这通常说明 transitions 与 loss_mask "
                        "不一致，或把不可逆位错误地标成了 loss。")
            else:
                ml = np.zeros(self.L, dtype=np.float32)
                yl[:] = 0.0

            X.append(s)
            Yg.append(yg)
            Mg.append(mg)
            Yl.append(yl)
            Ml.append(ml)

            n_gain_labels += int(yg.sum())
            n_loss_labels += int(yl.sum())
            n_loss_transitions += int(yl.sum() > 0)

        X = np.array(X)
        Yg = np.array(Yg)
        Mg = np.array(Mg)
        Yl = np.array(Yl)
        Ml = np.array(Ml)
        prep_info = {
            "n_samples": int(len(X)),
            "n_gain_labels": int(n_gain_labels),
            "n_loss_labels": int(n_loss_labels),
            "n_loss_transitions": int(n_loss_transitions),
            "n_loss_eligible_features": int(loss_mask.sum()) if use_loss else 0,
            "n_loss_candidates": int(Ml.sum()) if len(Ml) else 0,
        }
        return X, Yg, Mg, Yl, Ml, prep_info

    def fit(self, trans):
        import torch
        import torch.nn as nn

        self._torch = torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X, Yg, Mg, Yl, Ml, prep = self._prepare(trans)

        requested_dual = (self.head_mode == "dual_head_selective_reversible")
        dual_disable_reason = None
        dual_enabled = requested_dual and prep["n_loss_eligible_features"] > 0 and prep["n_loss_labels"] > 0
        if requested_dual and not dual_enabled:
            if prep["n_loss_eligible_features"] <= 0:
                dual_disable_reason = "no_loss_eligible_features"
            elif prep["n_loss_labels"] <= 0:
                dual_disable_reason = "no_positive_loss_labels"
        self._effective_head_mode = (
            "dual_head_selective_reversible" if dual_enabled else "gain_only"
        )

        if len(X) < 10:
            self._fallback_gain = Yg.sum(0) + 1e-6
            self._fallback_loss = (Yl.sum(0) + 1e-6) if dual_enabled else np.zeros(self.L, dtype=float)
            self.net = None
            self.fit_report = {
                "head_mode": self._effective_head_mode,
                "requested_head_mode": self.head_mode,
                "loss_head_enabled": bool(dual_enabled),
                "loss_head_disable_reason": dual_disable_reason,
                **prep,
            }
            return self

        n = len(X)
        idx = np.random.permutation(n)
        n_val = max(1, int(n * self.val_frac))
        vi, ti = idx[:n_val], idx[n_val:]
        Xt, Ygt, Mgt = X[ti], Yg[ti], Mg[ti]
        Xv, Ygv, Mgv = X[vi], Yg[vi], Mg[vi]
        Ylt, Mlt = Yl[ti], Ml[ti]
        Ylv, Mlv = Yl[vi], Ml[vi]

        dev = torch.device("cpu")
        self.net = self._build_net().to(dev)
        opt = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def masked_bce(logits, y, m):
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, y, reduction="none")
            bce = bce * m
            denom = m.sum().clamp(min=1.0)
            return bce.sum() / denom

        def to_t(a):
            return torch.tensor(a, dtype=torch.float32, device=dev)

        Xt_, Ygt_, Mgt_ = to_t(Xt), to_t(Ygt), to_t(Mgt)
        Xv_, Ygv_, Mgv_ = to_t(Xv), to_t(Ygv), to_t(Mgv)
        Ylt_, Mlt_ = to_t(Ylt), to_t(Mlt)
        Ylv_, Mlv_ = to_t(Ylv), to_t(Mlv)

        best_val = float("inf")
        best_state = None
        best_parts = None
        wait = 0
        hist = []
        bs = self.batch_size

        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Xt_))
            tr_total = 0.0
            tr_gain = 0.0
            tr_loss = 0.0
            nb = 0
            for b in range(0, len(perm), bs):
                bi = perm[b:b + bs]
                opt.zero_grad()
                gain_logits, loss_logits = self._split_logits(self.net(Xt_[bi]))
                loss_gain = masked_bce(gain_logits, Ygt_[bi], Mgt_[bi])
                loss = loss_gain
                loss_loss = None
                if dual_enabled:
                    loss_loss = masked_bce(loss_logits, Ylt_[bi], Mlt_[bi])
                    loss = loss + self.loss_weight * loss_loss
                loss.backward()
                opt.step()
                tr_total += loss.item()
                tr_gain += loss_gain.item()
                tr_loss += (0.0 if loss_loss is None else loss_loss.item())
                nb += 1
            tr_total /= max(nb, 1)
            tr_gain /= max(nb, 1)
            tr_loss /= max(nb, 1)

            self.net.eval()
            with torch.no_grad():
                gain_logits_v, loss_logits_v = self._split_logits(self.net(Xv_))
                val_gain = masked_bce(gain_logits_v, Ygv_, Mgv_).item()
                val_loss_head = None
                val_total = val_gain
                if dual_enabled:
                    val_loss_head = masked_bce(loss_logits_v, Ylv_, Mlv_).item()
                    val_total = val_total + self.loss_weight * val_loss_head
            hist.append({
                "train_total": tr_total,
                "train_gain": tr_gain,
                "train_loss": tr_loss if dual_enabled else None,
                "val_total": float(val_total),
                "val_gain": float(val_gain),
                "val_loss": (None if val_loss_head is None else float(val_loss_head)),
            })

            if val_total < best_val - 1e-5:
                best_val = float(val_total)
                best_parts = {
                    "val_gain": float(val_gain),
                    "val_loss": (None if val_loss_head is None else float(val_loss_head)),
                }
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

            if self.verbose and ep % 20 == 0:
                if dual_enabled:
                    print(f"  [ep{ep}] train={tr_total:.4f} "
                          f"(g={tr_gain:.4f}, l={tr_loss:.4f}) "
                          f"val={val_total:.4f} "
                          f"(g={val_gain:.4f}, l={val_loss_head:.4f})")
                else:
                    print(f"  [ep{ep}] train={tr_total:.4f} val={val_total:.4f}")

        if best_state is not None:
            self.net.load_state_dict(best_state)

        final = hist[-1] if hist else {
            "train_total": float("nan"),
            "train_gain": float("nan"),
            "train_loss": None,
        }
        self._fallback_gain = None
        self._fallback_loss = None
        self.fit_report = {
            "n_train": int(len(ti)),
            "n_val": int(n_val),
            "best_val_loss": float(best_val),
            "best_val_gain_loss": (None if best_parts is None else best_parts["val_gain"]),
            "best_val_loss_head": (None if best_parts is None else best_parts["val_loss"]),
            "final_train_loss": float(final["train_total"]),
            "final_train_gain_loss": float(final["train_gain"]),
            "final_train_loss_head": final["train_loss"],
            "train_val_gap": float(final["train_total"] - best_val),
            "epochs_ran": len(hist),
            "head_mode": self._effective_head_mode,
            "requested_head_mode": self.head_mode,
            "loss_head_enabled": bool(dual_enabled),
            "loss_head_disable_reason": dual_disable_reason,
            "loss_weight": float(self.loss_weight),
            **prep,
        }
        if self.verbose:
            print(f"  fit done: {self.fit_report}")
        return self

    # ------------------------------------------------------------------
    def predict_heads(self, src):
        """返回 {"gain": P_gain, "loss": P_loss}。
        gain 仅保留 src==0 的候选位;
        loss 仅保留 src==1 且属 selective loss_mask 的候选位。
        """
        src = np.asarray(src, dtype=np.int8).reshape(-1)
        if src.shape[0] != self.L:
            raise ValueError(f"src 长度({src.shape[0]}) 与 L({self.L}) 不一致")

        gain_cand = (src == 0)
        if self.loss_mask is None:
            loss_cand = np.zeros(self.L, dtype=bool)
        else:
            loss_cand = (src == 1) & (self.loss_mask > 0)

        if self.net is None:
            gain = (np.ones(self.L, dtype=float) if self._fallback_gain is None
                    else np.asarray(self._fallback_gain, dtype=float).copy())
            loss = (np.zeros(self.L, dtype=float) if self._fallback_loss is None
                    else np.asarray(self._fallback_loss, dtype=float).copy())
        else:
            torch = self._torch
            self.net.eval()
            with torch.no_grad():
                s = torch.tensor(src.astype(np.float32)[None, :], dtype=torch.float32)
                gain_logits, loss_logits = self._split_logits(self.net(s))
                gain = self._sigmoid(gain_logits[0].detach().cpu().numpy())
                if loss_logits is None:
                    loss = np.zeros(self.L, dtype=float)
                else:
                    loss = self._sigmoid(loss_logits[0].detach().cpu().numpy())

        gain = np.where(gain_cand, gain, 0.0)
        loss = np.where(loss_cand, loss, 0.0)
        return {"gain": gain.astype(float), "loss": loss.astype(float)}

    def predict_fn(self, src):
        """向后兼容旧 evaluator: 只返回 gain head 分数。"""
        return self.predict_heads(src)["gain"]

    def predict_loss_fn(self, src):
        """返回 selective-reversible loss head 分数。"""
        return self.predict_heads(src)["loss"]

    # ------------------------------------------------------------------
    # 持久化(对齐 HyperTraPS step3 的 forwards.txt: 训练产物 = 可复用模型)
    # ------------------------------------------------------------------
    def save(self, path):
        """存超参 + net权重 (或 fallback 频率) + fit_report 到单个 .pt 文件。"""
        import torch
        blob = {
            "L": self.L,
            "arch": self.arch,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
            "name": self.name,
            "fit_report": self.fit_report,
            "head_mode": self._effective_head_mode,
            "requested_head_mode": self.head_mode,
            "loss_weight": self.loss_weight,
            "loss_mask": (None if self.loss_mask is None
                          else np.asarray(self.loss_mask).astype(int).tolist()),
            "feat_meta": (None if self.feat_meta is None else {
                "fields": list(self.feat_meta["fields"]),
                "vocab": {k: int(v) for k, v in self.feat_meta["vocab"].items()},
                "ids": {k: np.asarray(v).astype(int).tolist()
                        for k, v in self.feat_meta["ids"].items()},
            }),
            "state_dict": (None if self.net is None else self.net.state_dict()),
            "fallback_gain": (None if self._fallback_gain is None
                              else np.asarray(self._fallback_gain).tolist()),
            "fallback_loss": (None if self._fallback_loss is None
                              else np.asarray(self._fallback_loss).tolist()),
            # 兼容旧字段名: fallback_freq == gain-only fallback
            "fallback_freq": (None if self._fallback_gain is None
                              else np.asarray(self._fallback_gain).tolist()),
        }
        torch.save(blob, path)
        return path

    @classmethod
    def load(cls, path):
        """从 save() 产物重建模型, predict_fn / predict_heads 立即可用。"""
        import torch
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location="cpu")

        fm = blob.get("feat_meta")
        feat_meta = None
        if fm is not None:
            feat_meta = {
                "fields": list(fm["fields"]),
                "vocab": {k: int(v) for k, v in fm["vocab"].items()},
                "ids": {k: np.asarray(v, dtype=np.int64) for k, v in fm["ids"].items()},
            }

        loss_mask = blob.get("loss_mask")
        if loss_mask is not None:
            loss_mask = np.asarray(loss_mask, dtype=np.float32)

        actual_head_mode = blob.get("head_mode", "gain_only")
        m = cls(
            blob["L"],
            arch=blob["arch"],
            d_model=blob["d_model"],
            n_layers=blob["n_layers"],
            n_heads=blob["n_heads"],
            dropout=blob["dropout"],
            name=blob.get("name"),
            feat_meta=feat_meta,
            head_mode=actual_head_mode,
            loss_mask=loss_mask,
            loss_weight=blob.get("loss_weight", 1.0),
        )
        m._torch = torch
        m.fit_report = blob.get("fit_report", {})
        m.head_mode = blob.get("requested_head_mode", actual_head_mode)
        m._effective_head_mode = actual_head_mode

        if blob.get("state_dict") is not None:
            m.net = m._build_net()
            m.net.load_state_dict(blob["state_dict"])
            m.net.eval()
            m._fallback_gain = None
            m._fallback_loss = None
        else:
            m.net = None
            fg = blob.get("fallback_gain", blob.get("fallback_freq"))
            fl = blob.get("fallback_loss")
            m._fallback_gain = None if fg is None else np.asarray(fg, dtype=float)
            m._fallback_loss = None if fl is None else np.asarray(fl, dtype=float)
        return m
