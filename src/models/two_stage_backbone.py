#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
two_stage_backbone.py —— 两阶段 EvoAMR 骨架

Stage 1: masked-state + state-contrastive pretraining
    给定部分 mask 的耐药状态向量, 重建被 mask 的 0/1 位；
    并约束同一 state 的不同 masking / dropout 视图保持相近表示。

Stage 2: next-event transition fine-tuning
    给定源状态 src, 预测下一步 gained features 的分布。

设计目标:
- 尽量复用现有 transitions.csv / feature_hierarchy.csv 输入口径
- 支持 identity+state 与 metadata token 化两种模式
- 提供最小可用的 save/load/predict 接口, 供脚本与 CV 评估复用
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import train_dl as TDL


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seeded_cuda_devices() -> List[int]:
    return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []


def _build_seeded_net(seed: int, **kwargs) -> "TwoStageNet":
    with torch.random.fork_rng(devices=_seeded_cuda_devices()):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return TwoStageNet(**kwargs)


def _jsonable(obj: Any):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if math.isnan(x) else x
    return obj


def serialize_feat_meta(feat_meta: Optional[dict]) -> Optional[dict]:
    if feat_meta is None:
        return None
    return {
        "fields": list(feat_meta["fields"]),
        "vocab": {k: int(v) for k, v in feat_meta["vocab"].items()},
        "ids": {k: np.asarray(v).astype(int).tolist() for k, v in feat_meta["ids"].items()},
        "vocab_labels": {k: list(v) for k, v in feat_meta.get("vocab_labels", {}).items()},
    }


def deserialize_feat_meta(blob: Optional[dict]) -> Optional[dict]:
    if blob is None:
        return None
    return {
        "fields": list(blob["fields"]),
        "vocab": {k: int(v) for k, v in blob["vocab"].items()},
        "ids": {k: np.asarray(v, dtype=np.int64) for k, v in blob["ids"].items()},
        "vocab_labels": {k: list(v) for k, v in blob.get("vocab_labels", {}).items()},
    }


@dataclass
class FeatureSetup:
    L: int
    transitions: List[dict]
    transition_info: dict
    hierarchy_path: Optional[str]
    hierarchy_df: Any
    feat_ids: Optional[List[str]]
    feat_meta: Optional[dict]
    meta_info: dict


def prepare_feature_setup(
    transitions_path: str,
    hierarchy_path: Optional[str] = None,
    no_meta: bool = False,
    meta_fields: Optional[Sequence[str]] = None,
) -> FeatureSetup:
    trans, L, trans_info = TDL.load_transitions(transitions_path)
    hpath = hierarchy_path or TDL.find_hierarchy_path(transitions_path)
    hdf = None
    feat_ids = None
    n_ok = 0
    if hpath and os.path.isfile(hpath):
        hdf = TDL.load_feature_hierarchy(hpath, L)
        feat_ids = hdf["feature_id"].tolist()
        n_ok = TDL.validate_feature_alignment(trans, feat_ids)
    elif not no_meta:
        hpath = None

    requested_meta_fields = TDL.parse_meta_fields_arg(meta_fields)
    feat_meta = None
    meta_info = {
        "used": False,
        "reason": None,
        "fields": [],
        "requested_fields": ([] if no_meta else list(TDL.META_FIELDS) if requested_meta_fields is None else list(requested_meta_fields)),
        "requested_fields_source": ("--no-meta" if no_meta else "default_all" if requested_meta_fields is None else "--meta-fields"),
        "hierarchy": (None if hpath is None else os.path.abspath(hpath)),
        "enabled": {},
        "skipped": {},
        "n_alignment_checked": int(n_ok),
    }
    if no_meta:
        meta_info["reason"] = "disabled by --no-meta"
    elif hdf is None:
        meta_info["reason"] = "未找到 feature_hierarchy.csv, 退化为纯 identity+state"
    else:
        feat_meta, _, meta_scan = TDL.load_feature_meta(hdf, requested_meta_fields=requested_meta_fields)
        meta_info.update(enabled=meta_scan["enabled"], skipped=meta_scan["skipped"])
        if feat_meta is None:
            meta_info["reason"] = "hierarchy 存在, 但所请求元数据字段在当前数据里都无区分度"
        else:
            meta_info.update(
                used=True,
                fields=feat_meta["fields"],
                vocab={k: int(v) for k, v in feat_meta["vocab"].items()},
            )

    return FeatureSetup(
        L=L,
        transitions=trans,
        transition_info=trans_info,
        hierarchy_path=(None if hpath is None else os.path.abspath(hpath)),
        hierarchy_df=hdf,
        feat_ids=feat_ids,
        feat_meta=feat_meta,
        meta_info=meta_info,
    )


def build_state_examples(
    transitions: Sequence[dict],
    state_source: str = "src_dest",
    dedupe: bool = True,
) -> np.ndarray:
    states: List[np.ndarray] = []
    for t in transitions:
        if state_source in {"src", "src_dest", "all"}:
            states.append(np.asarray(t["src"], dtype=np.int8))
        if state_source in {"dest", "src_dest", "all"}:
            states.append(np.asarray(t["dest"], dtype=np.int8))
    if not states:
        raise ValueError("没有可用于 Stage 1 的状态样本")
    arr = np.asarray(states, dtype=np.int8)
    if dedupe:
        arr = np.unique(arr, axis=0)
    return arr


def filter_gain_transitions(transitions: Sequence[dict]) -> List[dict]:
    return [t for t in transitions if len(t.get("gained", [])) > 0]


def filter_loss_transitions(transitions: Sequence[dict]) -> List[dict]:
    return [t for t in transitions if len(t.get("lost", [])) > 0]


def filter_event_transitions(transitions: Sequence[dict],
                             include_loss: bool = False) -> List[dict]:
    """保留"至少含一个可监督事件"的边。

    include_loss=False 时与 filter_gain_transitions 完全等价 (gain-only 主线
    行为不变); include_loss=True 时额外纳入 loss-only 边, 使 1->0 事件不再被丢弃。
    """
    if not include_loss:
        return filter_gain_transitions(transitions)
    return [
        t for t in transitions
        if len(t.get("gained", [])) > 0 or len(t.get("lost", [])) > 0
    ]


class TransformerBackbone(nn.Module):
    def __init__(self, L: int, d_model: int, n_layers: int, n_heads: int,
                 dropout: float, feat_meta: Optional[dict] = None):
        super().__init__()
        self.L = int(L)
        self.d_model = int(d_model)
        self.feat_meta = feat_meta
        self.feat_emb = nn.Embedding(self.L, d_model)
        self.state_emb = nn.Embedding(3, d_model)
        self.meta_embs = nn.ModuleDict()
        if feat_meta is not None:
            for field in feat_meta["fields"]:
                self.meta_embs[field] = nn.Embedding(int(feat_meta["vocab"][field]), d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=d_model * 4,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        B, L = states.shape
        if L != self.L:
            raise ValueError(f"state length mismatch: got {L}, expected {self.L}")
        device = states.device
        feat_ids = torch.arange(self.L, device=device).unsqueeze(0).expand(B, -1)
        tok = self.feat_emb(feat_ids) + self.state_emb(states.clamp(min=0, max=2))
        if self.feat_meta is not None:
            for field in self.feat_meta["fields"]:
                ids = torch.as_tensor(self.feat_meta["ids"][field], dtype=torch.long, device=device)
                tok = tok + self.meta_embs[field](ids).unsqueeze(0)
        tok = self.drop(tok)
        return self.norm(self.encoder(tok))


class MLPBackbone(nn.Module):
    def __init__(self, L: int, d_model: int, dropout: float, feat_meta: Optional[dict] = None):
        super().__init__()
        self.L = int(L)
        self.d_model = int(d_model)
        self.feat_meta = feat_meta
        self.feat_emb = nn.Embedding(self.L, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(L, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        B, L = states.shape
        if L != self.L:
            raise ValueError(f"state length mismatch: got {L}, expected {self.L}")
        device = states.device
        dense = states.float().clone()
        dense = torch.where(dense >= 2, torch.zeros_like(dense), dense)
        ctx = self.mlp(dense)
        feat_ids = torch.arange(self.L, device=device).unsqueeze(0).expand(B, -1)
        tok = self.feat_emb(feat_ids) + ctx.unsqueeze(1)
        tok = self.drop(tok)
        return self.norm(tok)


class TwoStageNet(nn.Module):
    def __init__(self, L: int, arch: str = "transformer", d_model: int = 64,
                 n_layers: int = 2, n_heads: int = 4, dropout: float = 0.3,
                 feat_meta: Optional[dict] = None):
        super().__init__()
        self.L = int(L)
        self.arch = arch
        self.d_model = int(d_model)
        self.feat_meta = feat_meta
        if arch == "transformer":
            self.backbone = TransformerBackbone(
                L=L,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                dropout=dropout,
                feat_meta=feat_meta,
            )
        elif arch == "mlp":
            self.backbone = MLPBackbone(L=L, d_model=d_model, dropout=dropout, feat_meta=feat_meta)
        else:
            raise ValueError(f"unsupported arch={arch}")
        self.reconstruct_head = nn.Linear(d_model, 1)
        self.gain_head = nn.Linear(d_model, 1)
        # loss head: 预测"端点净状态消失/回退"(1->0)。始终构建以保证 state_dict 结构
        # 稳定; 是否参与训练与预测由 TwoStageTransitionModel.head_mode 决定。
        # 旧 checkpoint 不含该权重, 各处 load_state_dict 均为 strict=False, 故兼容。
        self.loss_head = nn.Linear(d_model, 1)
        self.state_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        return self.backbone(states)

    def state_projection(self, states: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        pooled = self.encode(states).mean(dim=1)
        projected = self.state_projector(pooled)
        return F.normalize(projected, dim=1) if normalize else projected

    def reconstruct_logits(self, states: torch.Tensor) -> torch.Tensor:
        return self.reconstruct_head(self.encode(states)).squeeze(-1)

    def gain_logits(self, states: torch.Tensor) -> torch.Tensor:
        return self.gain_head(self.encode(states)).squeeze(-1)

    def loss_logits(self, states: torch.Tensor) -> torch.Tensor:
        return self.loss_head(self.encode(states)).squeeze(-1)

    def head_by_side(self, side: str) -> nn.Module:
        if side == "gain":
            return self.gain_head
        if side == "loss":
            return self.loss_head
        raise ValueError(f"未知 side={side} (仅支持 'gain'/'loss')")


class TwoStageTransitionModel:
    def __init__(self, L: int, arch: str = "transformer", d_model: int = 64,
                 n_layers: int = 2, n_heads: int = 4, dropout: float = 0.3,
                 lr: float = 1e-3, weight_decay: float = 1e-3, batch_size: int = 64,
                 val_frac: float = 0.15, patience: int = 20, seed: int = 0,
                 device: Optional[str] = None, feat_meta: Optional[dict] = None,
                 head_mode: str = "gain_only", loss_mask: Optional[np.ndarray] = None,
                 loss_weight: float = 1.0):
        # head_mode / loss_mask / loss_weight 刻意不进 self.config:
        # config 会被 load() 以 cls(**cfg) 反构造, 保持其字段集稳定可避免破坏旧
        # checkpoint; 这三项改由 save/load 单独持久化。
        self.config = {
            "L": int(L),
            "arch": arch,
            "d_model": int(d_model),
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "dropout": float(dropout),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "val_frac": float(val_frac),
            "patience": int(patience),
            "seed": int(seed),
        }
        self.feat_meta = feat_meta
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = _build_seeded_net(
            self.config["seed"],
            feat_meta=feat_meta,
            **{k: self.config[k] for k in ["L", "arch", "d_model", "n_layers", "n_heads", "dropout"]},
        ).to(self.device)
        self.pretrain_report: Dict[str, Any] = {}
        self.alignment_report: Dict[str, Any] = {}
        self.fit_report: Dict[str, Any] = {}

        if head_mode not in {"gain_only", "dual_head_selective_reversible"}:
            raise ValueError(f"未知 head_mode={head_mode}")
        self.head_mode = head_mode
        self._effective_head_mode = head_mode
        self.loss_weight = float(loss_weight)
        if loss_mask is None:
            self.loss_mask = None
        else:
            arr = np.asarray(loss_mask, dtype=np.float32).reshape(-1)
            if arr.shape[0] != self.config["L"]:
                raise ValueError(
                    f"loss_mask 长度({arr.shape[0]}) 与 L({self.config['L']}) 不一致")
            self.loss_mask = (arr > 0).astype(np.float32)
        if head_mode == "dual_head_selective_reversible":
            if self.loss_mask is None or int(self.loss_mask.sum()) <= 0:
                raise ValueError(
                    "dual_head_selective_reversible 需要非空 loss_mask (可逆特征子集)")

    # ------------------------------------------------------------------
    def _loss_head_enabled(self) -> bool:
        return (self._effective_head_mode == "dual_head_selective_reversible"
                and self.loss_mask is not None
                and int(self.loss_mask.sum()) > 0)

    def _side_candidate(self, src: np.ndarray, side: str) -> np.ndarray:
        """该侧的候选位 bool mask。gain: src==0; loss: src==1 且属可逆先验。"""
        if side == "gain":
            return src == 0
        if side == "loss":
            mask = src == 1
            if self.loss_mask is not None:
                mask = mask & (self.loss_mask > 0)
            return mask
        raise ValueError(f"未知 side={side}")

    def _make_optimizer(
        self,
        params: Optional[Iterable[torch.nn.Parameter]] = None,
        lr: Optional[float] = None,
    ) -> torch.optim.Optimizer:
        target_params = self.net.parameters() if params is None else params
        return torch.optim.AdamW(
            target_params,
            lr=(self.config["lr"] if lr is None else float(lr)),
            weight_decay=self.config["weight_decay"],
        )

    def _split_indices(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.config["seed"])
        idx = np.arange(n)
        rng.shuffle(idx)
        if n <= 1 or self.config["val_frac"] <= 0:
            return idx, np.asarray([], dtype=int)
        n_val = max(1, int(round(n * self.config["val_frac"])))
        if n_val >= n:
            n_val = max(0, n - 1)
        if n_val <= 0:
            return idx, np.asarray([], dtype=int)
        return idx[n_val:], idx[:n_val]

    def _iter_batches(self, idx: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
        for i in range(0, len(idx), batch_size):
            yield idx[i:i + batch_size]

    def _mask_batch(
        self,
        states: np.ndarray,
        mask_prob: float,
        rng: np.random.Generator,
        mask_prob_pos: Optional[float] = None,
        mask_prob_neg: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if mask_prob_pos is None and mask_prob_neg is None:
            probs = np.full(states.shape, float(mask_prob), dtype=np.float32)
        else:
            pos_prob = float(mask_prob if mask_prob_pos is None else mask_prob_pos)
            neg_prob = float(mask_prob if mask_prob_neg is None else mask_prob_neg)
            probs = np.where(states > 0, pos_prob, neg_prob).astype(np.float32, copy=False)
        mask = rng.random(states.shape) < probs
        empty_rows = np.where(mask.sum(axis=1) == 0)[0]
        for row in empty_rows:
            positives = np.flatnonzero(states[row] > 0)
            candidates = positives if positives.size > 0 else np.arange(states.shape[1])
            mask[row, int(rng.choice(candidates))] = True
        masked = states.copy()
        masked[mask] = 2
        return masked, mask

    def _contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float,
        states_binary: Optional[np.ndarray] = None,
        similar_state_neg_threshold: float = 1.0,
    ) -> torch.Tensor:
        if z1.ndim != 2 or z2.ndim != 2:
            raise ValueError("contrastive inputs 必须是二维张量")
        if z1.shape != z2.shape:
            raise ValueError("contrastive 两个视图的表示维度必须一致")
        batch_size = z1.shape[0]
        if batch_size <= 1:
            return z1.new_tensor(0.0)
        reps = torch.cat([z1, z2], dim=0)
        logits = torch.matmul(reps, reps.T) / float(temperature)
        diag_mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
        if states_binary is not None and float(similar_state_neg_threshold) < 1.0:
            states_arr = np.asarray(states_binary)
            if states_arr.ndim != 2 or states_arr.shape[0] != batch_size:
                raise ValueError("states_binary 必须是与 contrastive batch 对齐的二维数组")
            state_tensor = torch.as_tensor(states_arr > 0, dtype=torch.float32, device=logits.device)
            intersection = state_tensor @ state_tensor.T
            cardinality = state_tensor.sum(dim=1, keepdim=True)
            union = (cardinality + cardinality.T - intersection).clamp(min=1e-6)
            jaccard_similarity = intersection / union
            similar_neg_mask = (jaccard_similarity > float(similar_state_neg_threshold)).repeat(2, 2)
            pos_rows = torch.arange(logits.size(0), device=logits.device)
            pos_cols = (pos_rows + batch_size) % (2 * batch_size)
            similar_neg_mask[pos_rows, pos_cols] = False
            logits = logits.masked_fill(similar_neg_mask & (~diag_mask), -1e9)
        logits = logits.masked_fill(diag_mask, -1e9)
        pos_idx = torch.arange(logits.size(0), device=logits.device)
        pos_idx = (pos_idx + batch_size) % (2 * batch_size)
        return F.cross_entropy(logits, pos_idx)

    def _pretrain_epoch(
        self,
        states: np.ndarray,
        idx: np.ndarray,
        optimizer: Optional[torch.optim.Optimizer],
        mask_prob: float,
        seed_offset: int,
        mask_prob_pos: Optional[float] = None,
        mask_prob_neg: Optional[float] = None,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.1,
        similar_state_neg_threshold: float = 1.0,
    ) -> Dict[str, float]:
        is_train = optimizer is not None
        self.net.train(is_train)
        rng = np.random.default_rng(self.config["seed"] + seed_offset)
        total_loss = 0.0
        total_recon_loss = 0.0
        total_contrastive_loss = 0.0
        total_samples = 0
        total_masked = 0
        total_correct = 0
        batch_size = self.config["batch_size"]
        contrastive_enabled = float(contrastive_weight) > 0.0
        for b_idx in self._iter_batches(idx, batch_size):
            if len(b_idx) == 0:
                continue
            batch = states[b_idx]
            masked_1, mask_1 = self._mask_batch(
                batch,
                mask_prob=mask_prob,
                rng=rng,
                mask_prob_pos=mask_prob_pos,
                mask_prob_neg=mask_prob_neg,
            )
            x1 = torch.as_tensor(masked_1, dtype=torch.long, device=self.device)
            y = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
            m1 = torch.as_tensor(mask_1, dtype=torch.bool, device=self.device)
            logits_1 = self.net.reconstruct_logits(x1)
            loss_raw_1 = F.binary_cross_entropy_with_logits(logits_1, y, reduction="none")
            masked_loss_1 = loss_raw_1[m1]
            if masked_loss_1.numel() == 0:
                continue
            recon_loss = masked_loss_1.mean()
            contrastive_loss = y.new_tensor(0.0)
            if contrastive_enabled:
                masked_2, _ = self._mask_batch(
                    batch,
                    mask_prob=mask_prob,
                    rng=rng,
                    mask_prob_pos=mask_prob_pos,
                    mask_prob_neg=mask_prob_neg,
                )
                x2 = torch.as_tensor(masked_2, dtype=torch.long, device=self.device)
                z1 = self.net.state_projection(x1)
                z2 = self.net.state_projection(x2)
                contrastive_loss = self._contrastive_loss(
                    z1,
                    z2,
                    contrastive_temperature,
                    states_binary=batch,
                    similar_state_neg_threshold=similar_state_neg_threshold,
                )
            loss = recon_loss + float(contrastive_weight) * contrastive_loss
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                pred = (torch.sigmoid(logits_1[m1]) >= 0.5).to(torch.float32)
                batch_n = int(len(b_idx))
                total_correct += int((pred == y[m1]).sum().item())
                total_masked += int(m1.sum().item())
                total_samples += batch_n
                total_loss += float(loss.item()) * batch_n
                total_recon_loss += float(recon_loss.item()) * batch_n
                total_contrastive_loss += float(contrastive_loss.item()) * batch_n
        return {
            "loss": (float(total_loss / total_samples) if total_samples else float("nan")),
            "recon_loss": (float(total_recon_loss / total_samples) if total_samples else float("nan")),
            "contrastive_loss": (float(total_contrastive_loss / total_samples) if total_samples else float("nan")),
            "masked_accuracy": (float(total_correct / total_masked) if total_masked else float("nan")),
            "n_masked": int(total_masked),
            "n_samples": int(total_samples),
        }

    def _resolve_pretrain_contrastive_weight(
        self,
        epoch: int,
        target_weight: float,
        warmup_epochs: int,
    ) -> float:
        target = float(target_weight)
        if target <= 0.0:
            return 0.0
        if warmup_epochs <= 1:
            return target
        progress = min(1.0, max(0.0, float(epoch - 1) / float(warmup_epochs - 1)))
        return target * progress

    def _pretrain_checkpoint_score(
        self,
        val_report: Dict[str, float],
        checkpoint_metric: str,
    ) -> Tuple[float, bool]:
        metric = str(checkpoint_metric or "val_loss")
        if metric == "val_masked_accuracy":
            return float(val_report.get("masked_accuracy", float("nan"))), True
        if metric == "val_recon_loss":
            return float(val_report.get("recon_loss", float("nan"))), False
        return float(val_report.get("loss", float("nan"))), False

    def pretrain(
        self,
        states: np.ndarray,
        epochs: int = 100,
        mask_prob: float = 0.15,
        mask_prob_pos: Optional[float] = None,
        mask_prob_neg: Optional[float] = None,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.1,
        contrastive_warmup_epochs: int = 0,
        similar_state_neg_threshold: float = 1.0,
        checkpoint_metric: str = "val_loss",
    ) -> Dict[str, Any]:
        if len(states) == 0:
            raise ValueError("Stage 1 没有状态样本")
        set_global_seed(self.config["seed"])
        train_idx, val_idx = self._split_indices(len(states))
        optimizer = self._make_optimizer()
        metric_name = str(checkpoint_metric or "val_loss")
        metric_higher_is_better = metric_name == "val_masked_accuracy"
        best_score = (-float("inf") if metric_higher_is_better else float("inf"))
        best_val_loss = float("inf")
        best_state = copy.deepcopy(self.net.state_dict())
        best_epoch = 0
        bad_epochs = 0
        history = []
        resolved_mask_prob_pos = float(mask_prob if mask_prob_pos is None else mask_prob_pos)
        resolved_mask_prob_neg = float(mask_prob if mask_prob_neg is None else mask_prob_neg)
        resolved_contrastive_weight = float(contrastive_weight)
        resolved_contrastive_temperature = float(contrastive_temperature)
        resolved_similar_state_neg_threshold = float(similar_state_neg_threshold)
        resolved_warmup_epochs = max(0, int(contrastive_warmup_epochs))
        for epoch in range(1, epochs + 1):
            epoch_contrastive_weight = self._resolve_pretrain_contrastive_weight(
                epoch,
                resolved_contrastive_weight,
                resolved_warmup_epochs,
            )
            tr = self._pretrain_epoch(
                states,
                train_idx,
                optimizer,
                mask_prob,
                seed_offset=epoch * 17,
                mask_prob_pos=resolved_mask_prob_pos,
                mask_prob_neg=resolved_mask_prob_neg,
                contrastive_weight=epoch_contrastive_weight,
                contrastive_temperature=resolved_contrastive_temperature,
                similar_state_neg_threshold=resolved_similar_state_neg_threshold,
            )
            va = (
                self._pretrain_epoch(
                    states,
                    val_idx,
                    None,
                    mask_prob,
                    seed_offset=epoch * 17 + 1,
                    mask_prob_pos=resolved_mask_prob_pos,
                    mask_prob_neg=resolved_mask_prob_neg,
                    contrastive_weight=epoch_contrastive_weight,
                    contrastive_temperature=resolved_contrastive_temperature,
                    similar_state_neg_threshold=resolved_similar_state_neg_threshold,
                )
                if len(val_idx) > 0 else {
                    "loss": tr["loss"],
                    "recon_loss": tr["recon_loss"],
                    "contrastive_loss": tr["contrastive_loss"],
                    "masked_accuracy": tr["masked_accuracy"],
                    "n_masked": 0,
                    "n_samples": 0,
                }
            )
            history.append({
                "epoch": epoch,
                "contrastive_weight": float(epoch_contrastive_weight),
                "train": tr,
                "val": va,
            })
            score, higher_is_better = self._pretrain_checkpoint_score(va, metric_name)
            val_loss = float(va.get("loss", float("nan")))
            improved = False
            if not math.isnan(score):
                improved = score > best_score if higher_is_better else score < best_score
            if improved:
                best_score = score
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.net.state_dict())
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= self.config["patience"]:
                break
        self.net.load_state_dict(best_state)
        final = history[best_epoch - 1] if history and best_epoch > 0 else (history[-1] if history else {"train": {}, "val": {}})
        self.pretrain_report = {
            "stage": "state_pretraining",
            "objective": "masked_reconstruction_plus_state_contrastive" if resolved_contrastive_weight > 0 else "masked_reconstruction",
            "arch": self.config["arch"],
            "epochs_requested": int(epochs),
            "epochs_ran": int(len(history)),
            "best_epoch": int(best_epoch),
            "checkpoint_metric": metric_name,
            "checkpoint_metric_higher_is_better": bool(metric_higher_is_better),
            "best_checkpoint_metric_value": (None if math.isnan(best_score) else float(best_score)),
            "best_val_loss": (None if math.isnan(best_val_loss) else float(best_val_loss)),
            "train_loss": final.get("train", {}).get("loss"),
            "train_recon_loss": final.get("train", {}).get("recon_loss"),
            "train_contrastive_loss": final.get("train", {}).get("contrastive_loss"),
            "train_masked_accuracy": final.get("train", {}).get("masked_accuracy"),
            "val_loss": final.get("val", {}).get("loss"),
            "val_recon_loss": final.get("val", {}).get("recon_loss"),
            "val_contrastive_loss": final.get("val", {}).get("contrastive_loss"),
            "val_masked_accuracy": final.get("val", {}).get("masked_accuracy"),
            "selected_epoch_contrastive_weight": final.get("contrastive_weight"),
            "contrastive_warmup_epochs": int(resolved_warmup_epochs),
            "similar_state_neg_threshold": float(resolved_similar_state_neg_threshold),
            "n_states": int(len(states)),
            "n_train_states": int(len(train_idx)),
            "n_val_states": int(len(val_idx)),
            "mask_prob": float(mask_prob),
            "mask_prob_pos": float(resolved_mask_prob_pos),
            "mask_prob_neg": float(resolved_mask_prob_neg),
            "contrastive_weight": float(resolved_contrastive_weight),
            "contrastive_temperature": float(resolved_contrastive_temperature),
        }
        return self.pretrain_report

    def _transition_batch(self, transitions: Sequence[dict],
                          side: str = "gain") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """构造某一侧的 (状态, 目标分布, 候选掩码)。

        side="gain": 候选 src==0, 目标为 gained 上的均匀分布 (与历史行为一致)。
        side="loss": 候选 src==1 且属可逆先验, 目标为 lost 上的均匀分布。
        目标写成候选内的分布, 与 _transition_epoch 里的 masked log_softmax 对应。
        """
        B = len(transitions)
        L = self.config["L"]
        event_key = "gained" if side == "gain" else "lost"
        x = np.zeros((B, L), dtype=np.int64)
        target = np.zeros((B, L), dtype=np.float32)
        candidate = np.zeros((B, L), dtype=bool)
        for i, t in enumerate(transitions):
            src = np.asarray(t["src"], dtype=np.int8)
            events = [int(j) for j in (t.get(event_key, []) or [])]
            x[i] = src.astype(np.int64)
            cand = self._side_candidate(src, side)
            candidate[i] = cand
            if events:
                bad = [j for j in events if not cand[j]]
                if bad:
                    raise ValueError(
                        f"{side} 事件落在非法候选位上: indices={bad}; "
                        "通常说明 transitions 与 loss_mask/可逆先验不一致。")
                target[i, events] = 1.0 / float(len(events))
        return (
            torch.as_tensor(x, dtype=torch.long, device=self.device),
            torch.as_tensor(target, dtype=torch.float32, device=self.device),
            torch.as_tensor(candidate, dtype=torch.bool, device=self.device),
        )

    def _is_transformer_backbone(self) -> bool:
        return self.config.get("arch") == "transformer" and hasattr(self.net.backbone, "encoder")

    def _trainable_backbone_named_parameters(
        self,
        unfreeze_last_n_layers: int = 0,
    ) -> List[Tuple[str, torch.nn.Parameter]]:
        named_params = list(self.net.backbone.named_parameters())
        if not named_params:
            return []
        resolved_last_n_layers = max(0, int(unfreeze_last_n_layers))
        if resolved_last_n_layers <= 0:
            return named_params
        if not self._is_transformer_backbone():
            return named_params
        encoder_layers = list(getattr(self.net.backbone.encoder, "layers", []))
        if not encoder_layers:
            return named_params
        total_layers = len(encoder_layers)
        if resolved_last_n_layers >= total_layers:
            return named_params
        start_layer = max(0, total_layers - resolved_last_n_layers)
        allowed_prefixes = tuple(f"encoder.layers.{idx}." for idx in range(start_layer, total_layers))
        selected = [
            (name, param)
            for name, param in named_params
            if name.startswith("norm.") or name.startswith(allowed_prefixes)
        ]
        return selected or named_params

    def _set_backbone_trainability(
        self,
        freeze_backbone: bool = False,
        unfreeze_last_n_layers: int = 0,
    ) -> Dict[str, Any]:
        named_params = list(self.net.backbone.named_parameters())
        selected_named_params = [] if freeze_backbone else self._trainable_backbone_named_parameters(
            unfreeze_last_n_layers=unfreeze_last_n_layers,
        )
        selected_param_ids = {id(param) for _, param in selected_named_params}
        for _, param in named_params:
            param.requires_grad = id(param) in selected_param_ids
        total_backbone_params = int(sum(param.numel() for _, param in named_params))
        trainable_backbone_params = int(sum(param.numel() for _, param in selected_named_params))
        if freeze_backbone or trainable_backbone_params == 0:
            trainable_scope = "frozen"
        elif int(unfreeze_last_n_layers) > 0 and self._is_transformer_backbone():
            encoder_layers = list(getattr(self.net.backbone.encoder, "layers", []))
            resolved_last_n_layers = min(int(unfreeze_last_n_layers), len(encoder_layers))
            trainable_scope = f"last_{resolved_last_n_layers}_transformer_layers_plus_norm"
        elif int(unfreeze_last_n_layers) > 0:
            trainable_scope = "full_backbone_non_transformer"
        else:
            trainable_scope = "full_backbone"
        return {
            "backbone_frozen_for_forward": bool(freeze_backbone),
            "backbone_trainable_scope": trainable_scope,
            "n_total_backbone_params": total_backbone_params,
            "n_trainable_backbone_params": trainable_backbone_params,
            "trainable_backbone_named_params": selected_named_params,
        }

    def _make_transition_optimizer(
        self,
        freeze_backbone: bool = False,
        unfreeze_last_n_layers: int = 0,
        backbone_lr_scale: float = 1.0,
        head_lr_scale: float = 1.0,
    ) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
        if float(head_lr_scale) <= 0:
            raise ValueError("head_lr_scale 必须大于 0")
        if (not freeze_backbone) and float(backbone_lr_scale) <= 0:
            raise ValueError("backbone_lr_scale 必须大于 0；若需冻结 backbone，请使用 freeze_backbone=True")
        phase_info = self._set_backbone_trainability(
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_layers=unfreeze_last_n_layers,
        )
        base_lr = float(self.config["lr"])
        param_groups = []
        trainable_backbone_params = [param for _, param in phase_info["trainable_backbone_named_params"]]
        if trainable_backbone_params:
            param_groups.append({
                "params": trainable_backbone_params,
                "lr": base_lr * float(backbone_lr_scale),
            })
        head_params = list(self.net.gain_head.parameters())
        if self._loss_head_enabled():
            # loss head 与 gain head 同属 "head" 参数组, 共享 head_lr_scale,
            # 保证 discriminative LR / partial unfreeze 等策略对两个 head 一致
            head_params += list(self.net.loss_head.parameters())
        for param in head_params:
            param.requires_grad = True
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": base_lr * float(head_lr_scale),
            })
        if not param_groups:
            raise ValueError("Stage 2 没有可训练参数")
        optimizer = self._make_optimizer(params=param_groups)
        phase_info.update({
            "backbone_lr": (0.0 if freeze_backbone else base_lr * float(backbone_lr_scale)),
            "head_lr": base_lr * float(head_lr_scale),
        })
        return optimizer, phase_info

    def _encode_for_transition(self, states: torch.Tensor,
                               backbone_frozen: bool = False) -> torch.Tensor:
        """编码一次, 供 gain / loss 两个 head 共用, 避免重复前向。"""
        if not backbone_frozen:
            return self.net.encode(states)
        self.net.backbone.eval()
        with torch.no_grad():
            return self.net.encode(states)

    def _transition_logits(self, states: torch.Tensor, backbone_frozen: bool = False,
                           side: str = "gain") -> torch.Tensor:
        features = self._encode_for_transition(states, backbone_frozen=backbone_frozen)
        return self.net.head_by_side(side)(features).squeeze(-1)

    def _transition_epoch(
        self,
        transitions: Sequence[dict],
        idx: np.ndarray,
        optimizer: Optional[torch.optim.Optimizer],
        backbone_frozen: bool = False,
    ) -> Dict[str, float]:
        is_train = optimizer is not None
        self.net.train(is_train)
        if backbone_frozen:
            self.net.backbone.eval()
        use_loss_head = self._loss_head_enabled()

        total_obj = 0.0
        total_obj_items = 0
        side_stats = {
            "gain": {"loss_sum": 0.0, "hits": 0, "n": 0},
            "loss": {"loss_sum": 0.0, "hits": 0, "n": 0},
        }
        batch_size = self.config["batch_size"]

        for b_idx in self._iter_batches(idx, batch_size):
            rows = [
                transitions[i] for i in b_idx
                if len(transitions[i].get("gained", [])) > 0
                or (use_loss_head and len(transitions[i].get("lost", [])) > 0)
            ]
            if not rows:
                continue

            # 一次编码, 两个 head 共用
            x = torch.as_tensor(
                np.stack([np.asarray(t["src"], dtype=np.int64) for t in rows]),
                dtype=torch.long, device=self.device)
            features = self._encode_for_transition(x, backbone_frozen=backbone_frozen)

            objective = None
            per_side_batch: Dict[str, Dict[str, Any]] = {}
            sides = ["gain"] + (["loss"] if use_loss_head else [])
            for side in sides:
                event_key = "gained" if side == "gain" else "lost"
                sel = [i for i, t in enumerate(rows) if len(t.get(event_key, []) or []) > 0]
                if not sel:
                    continue
                sub_rows = [rows[i] for i in sel]
                _, target, candidate = self._transition_batch(sub_rows, side=side)
                sel_t = torch.as_tensor(sel, dtype=torch.long, device=self.device)
                logits = self.net.head_by_side(side)(features.index_select(0, sel_t)).squeeze(-1)
                masked_logits = logits.masked_fill(~candidate, -1e9)
                log_probs = F.log_softmax(masked_logits, dim=1)
                side_loss = -(target * log_probs).sum(dim=1).mean()
                weight = 1.0 if side == "gain" else self.loss_weight
                objective = side_loss * weight if objective is None else objective + side_loss * weight
                per_side_batch[side] = {
                    "loss": side_loss,
                    "masked_logits": masked_logits,
                    "rows": sub_rows,
                    "event_key": event_key,
                }

            if objective is None:
                continue
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                optimizer.step()

            with torch.no_grad():
                total_obj += float(objective.item()) * len(rows)
                total_obj_items += len(rows)
                for side, info in per_side_batch.items():
                    pred = info["masked_logits"].argmax(dim=1).detach().cpu().numpy()
                    n = len(info["rows"])
                    hits = sum(
                        1 for r_i, t in enumerate(info["rows"])
                        if pred[r_i] in set(t.get(info["event_key"], []) or [])
                    )
                    side_stats[side]["loss_sum"] += float(info["loss"].item()) * n
                    side_stats[side]["hits"] += hits
                    side_stats[side]["n"] += n

        def _mean(side: str, key: str) -> float:
            n = side_stats[side]["n"]
            if not n:
                return float("nan")
            return float(side_stats[side][key] / n)

        out = {
            # "loss"/"top1"/"n_items" 保持历史语义 (gain-only 时与旧实现一致):
            # loss = 优化目标, top1/n_items = gain 侧命中与样本数
            "loss": (float(total_obj / total_obj_items) if total_obj_items else float("nan")),
            "top1": _mean("gain", "hits"),
            "n_items": int(side_stats["gain"]["n"]),
            "gain_loss": _mean("gain", "loss_sum"),
            "gain_top1": _mean("gain", "hits"),
            "n_gain_items": int(side_stats["gain"]["n"]),
            "loss_head_enabled": bool(use_loss_head),
        }
        if use_loss_head:
            out.update({
                "loss_loss": _mean("loss", "loss_sum"),
                "loss_top1": _mean("loss", "hits"),
                "n_loss_items": int(side_stats["loss"]["n"]),
            })
        return out

    def align_transitions(
        self,
        transitions: Sequence[dict],
        epochs: int = 20,
        lr_scale: float = 0.5,
    ) -> Dict[str, Any]:
        trans = filter_gain_transitions(transitions)
        if not trans:
            raise ValueError("Alignment 阶段没有可用于 gained prediction 的 transition")
        if epochs <= 0:
            self.alignment_report = {
                "stage": "transition_alignment",
                "enabled": False,
                "reason": "epochs<=0",
                "n_transitions": int(len(trans)),
            }
            return self.alignment_report
        set_global_seed(self.config["seed"])
        train_idx, val_idx = self._split_indices(len(trans))
        optimizer = self._make_optimizer(lr=self.config["lr"] * float(lr_scale))
        best_val = float("inf")
        best_state = copy.deepcopy(self.net.state_dict())
        best_epoch = 0
        bad_epochs = 0
        history = []
        for epoch in range(1, epochs + 1):
            tr = self._transition_epoch(trans, train_idx, optimizer)
            va = (
                self._transition_epoch(trans, val_idx, None)
                if len(val_idx) > 0 else {"loss": tr["loss"], "top1": tr["top1"], "n_items": 0}
            )
            history.append({"epoch": epoch, "train": tr, "val": va})
            val_loss = va["loss"]
            if not math.isnan(val_loss) and val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(self.net.state_dict())
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= self.config["patience"]:
                break
        self.net.load_state_dict(best_state)
        final = history[best_epoch - 1] if history and best_epoch > 0 else (history[-1] if history else {"train": {}, "val": {}})
        self.alignment_report = {
            "stage": "transition_alignment",
            "enabled": True,
            "objective": "candidate_masked_gain",
            "arch": self.config["arch"],
            "epochs_requested": int(epochs),
            "epochs_ran": int(len(history)),
            "best_epoch": int(best_epoch),
            "best_val_loss": (None if math.isnan(best_val) else float(best_val)),
            "train_loss": final.get("train", {}).get("loss"),
            "train_top1": final.get("train", {}).get("top1"),
            "val_loss": final.get("val", {}).get("loss"),
            "val_top1": final.get("val", {}).get("top1"),
            "n_transitions": int(len(trans)),
            "n_train_transitions": int(len(train_idx)),
            "n_val_transitions": int(len(val_idx)),
            "lr_scale": float(lr_scale),
            "train_val_gap": (
                None
                if any(math.isnan(float(x)) for x in [final.get("train", {}).get("loss", float("nan")), final.get("val", {}).get("loss", float("nan"))])
                else float(final["train"]["loss"] - final["val"]["loss"])
            ),
        }
        return self.alignment_report

    def _resolve_transition_strategy(
        self,
        freeze_backbone: bool,
        freeze_backbone_epochs: int,
        unfreeze_last_n_layers: int,
        backbone_lr_scale: float,
        head_lr_scale: float,
    ) -> str:
        uses_scaled_learning_rates = (
            not math.isclose(float(backbone_lr_scale), 1.0, rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(float(head_lr_scale), 1.0, rel_tol=1e-12, abs_tol=1e-12)
        )
        uses_discriminative_lr = not math.isclose(
            float(backbone_lr_scale),
            float(head_lr_scale),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        if freeze_backbone:
            return "frozen_backbone"
        if int(freeze_backbone_epochs) > 0:
            if int(unfreeze_last_n_layers) > 0:
                return "freeze_then_partial_unfreeze_discriminative_lr" if uses_discriminative_lr else "freeze_then_partial_unfreeze"
            return "freeze_then_unfreeze_discriminative_lr" if uses_discriminative_lr else "freeze_then_unfreeze"
        if int(unfreeze_last_n_layers) > 0:
            return "partial_unfreeze_discriminative_lr" if uses_discriminative_lr else "partial_unfreeze"
        if uses_discriminative_lr:
            return "discriminative_lr"
        if uses_scaled_learning_rates:
            return "scaled_lr_full_finetune"
        return "full_finetune"

    def fit_transitions(
        self,
        transitions: Sequence[dict],
        epochs: int = 100,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        freeze_backbone_epochs: int = 0,
        backbone_lr_scale: float = 1.0,
        head_lr_scale: float = 1.0,
        unfreeze_last_n_layers: int = 0,
    ) -> Dict[str, Any]:
        # 双头启用时把 loss-only 边一并纳入 Stage 2 训练; gain-only 时行为不变
        requested_dual = (self.head_mode == "dual_head_selective_reversible")
        n_loss_labels = sum(len(t.get("lost", []) or []) for t in transitions)
        dual_enabled = bool(
            requested_dual
            and self.loss_mask is not None
            and int(self.loss_mask.sum()) > 0
            and n_loss_labels > 0
        )
        dual_disable_reason = None
        if requested_dual and not dual_enabled:
            if self.loss_mask is None or int(self.loss_mask.sum()) <= 0:
                dual_disable_reason = "no_loss_eligible_features"
            elif n_loss_labels <= 0:
                dual_disable_reason = "no_positive_loss_labels"
        self._effective_head_mode = (
            "dual_head_selective_reversible" if dual_enabled else "gain_only")

        trans = filter_event_transitions(transitions, include_loss=dual_enabled)
        if not trans:
            raise ValueError("Stage 2 没有可用于 next-step prediction 的 transition")
        resolved_freeze_backbone_epochs = max(0, int(freeze_backbone_epochs))
        resolved_unfreeze_last_n_layers = int(unfreeze_last_n_layers)
        if resolved_unfreeze_last_n_layers < 0:
            raise ValueError("unfreeze_last_n_layers 不能为负数")
        resolved_backbone_lr_scale = float(backbone_lr_scale)
        resolved_head_lr_scale = float(head_lr_scale)
        effective_freeze_backbone_epochs = (0 if freeze_backbone else resolved_freeze_backbone_epochs)
        set_global_seed(self.config["seed"])
        train_idx, val_idx = self._split_indices(len(trans))
        backbone_requires_grad = [param.requires_grad for param in self.net.backbone.parameters()]
        best_val = float("inf")
        best_state = copy.deepcopy(self.net.state_dict())
        best_epoch = 0
        bad_epochs = 0
        history = []
        current_phase_signature = None
        optimizer = None
        current_phase_info: Dict[str, Any] = {
            "backbone_frozen_for_forward": bool(freeze_backbone),
            "backbone_trainable_scope": "frozen" if freeze_backbone else "full_backbone",
            "n_total_backbone_params": int(sum(param.numel() for param in self.net.backbone.parameters())),
            "n_trainable_backbone_params": 0,
            "backbone_lr": 0.0,
            "head_lr": float(self.config["lr"]) * resolved_head_lr_scale,
        }
        strategy = self._resolve_transition_strategy(
            freeze_backbone=bool(freeze_backbone),
            freeze_backbone_epochs=effective_freeze_backbone_epochs,
            unfreeze_last_n_layers=resolved_unfreeze_last_n_layers,
            backbone_lr_scale=resolved_backbone_lr_scale,
            head_lr_scale=resolved_head_lr_scale,
        )
        try:
            for epoch in range(1, epochs + 1):
                if freeze_backbone:
                    phase_name = "head_only"
                    phase_signature = (True, 0)
                    phase_freeze_backbone = True
                    phase_unfreeze_last_n_layers = 0
                elif epoch <= effective_freeze_backbone_epochs:
                    phase_name = "head_only_warmup"
                    phase_signature = (True, 0)
                    phase_freeze_backbone = True
                    phase_unfreeze_last_n_layers = 0
                else:
                    phase_name = "adaptation"
                    phase_signature = (False, resolved_unfreeze_last_n_layers)
                    phase_freeze_backbone = False
                    phase_unfreeze_last_n_layers = resolved_unfreeze_last_n_layers
                if phase_signature != current_phase_signature:
                    optimizer, current_phase_info = self._make_transition_optimizer(
                        freeze_backbone=phase_freeze_backbone,
                        unfreeze_last_n_layers=phase_unfreeze_last_n_layers,
                        backbone_lr_scale=resolved_backbone_lr_scale,
                        head_lr_scale=resolved_head_lr_scale,
                    )
                    current_phase_signature = phase_signature
                tr = self._transition_epoch(
                    trans,
                    train_idx,
                    optimizer,
                    backbone_frozen=bool(current_phase_info.get("backbone_frozen_for_forward", False)),
                )
                va = (
                    self._transition_epoch(
                        trans,
                        val_idx,
                        None,
                        backbone_frozen=bool(current_phase_info.get("backbone_frozen_for_forward", False)),
                    )
                    if len(val_idx) > 0 else {"loss": tr["loss"], "top1": tr["top1"], "n_items": 0}
                )
                history.append({
                    "epoch": epoch,
                    "phase": phase_name,
                    "backbone_trainable_scope": current_phase_info.get("backbone_trainable_scope"),
                    "n_trainable_backbone_params": int(current_phase_info.get("n_trainable_backbone_params", 0)),
                    "n_total_backbone_params": int(current_phase_info.get("n_total_backbone_params", 0)),
                    "backbone_lr": float(current_phase_info.get("backbone_lr", 0.0)),
                    "head_lr": float(current_phase_info.get("head_lr", 0.0)),
                    "train": tr,
                    "val": va,
                })
                val_loss = va["loss"]
                if not math.isnan(val_loss) and val_loss < best_val:
                    best_val = val_loss
                    best_state = copy.deepcopy(self.net.state_dict())
                    best_epoch = epoch
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if bad_epochs >= self.config["patience"]:
                    break
            self.net.load_state_dict(best_state)
        finally:
            for param, requires_grad in zip(self.net.backbone.parameters(), backbone_requires_grad):
                param.requires_grad = requires_grad
        final = history[best_epoch - 1] if history and best_epoch > 0 else (history[-1] if history else {"train": {}, "val": {}})
        uses_scaled_learning_rates = (
            not math.isclose(resolved_backbone_lr_scale, 1.0, rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(resolved_head_lr_scale, 1.0, rel_tol=1e-12, abs_tol=1e-12)
        )
        uses_discriminative_lr = not math.isclose(
            resolved_backbone_lr_scale,
            resolved_head_lr_scale,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        self.fit_report = {
            "stage": "transition_finetuning",
            "arch": self.config["arch"],
            "epochs_requested": int(epochs),
            "epochs_ran": int(len(history)),
            "best_epoch": int(best_epoch),
            "best_val_loss": (None if math.isnan(best_val) else float(best_val)),
            "train_loss": final.get("train", {}).get("loss"),
            "train_top1": final.get("train", {}).get("top1"),
            "val_loss": final.get("val", {}).get("loss"),
            "val_top1": final.get("val", {}).get("top1"),
            "n_transitions": int(len(trans)),
            "n_train_transitions": int(len(train_idx)),
            "n_val_transitions": int(len(val_idx)),
            # gain + loss 双头
            "head_mode": self._effective_head_mode,
            "requested_head_mode": self.head_mode,
            "loss_head_enabled": bool(dual_enabled),
            "loss_head_disable_reason": dual_disable_reason,
            "loss_weight": float(self.loss_weight),
            "n_loss_eligible_features": (
                0 if self.loss_mask is None else int(self.loss_mask.sum())),
            "n_loss_labels": int(n_loss_labels),
            "n_loss_transitions": int(sum(1 for t in trans if len(t.get("lost", []) or []) > 0)),
            "n_gain_transitions": int(sum(1 for t in trans if len(t.get("gained", []) or []) > 0)),
            "train_gain_loss": final.get("train", {}).get("gain_loss"),
            "train_loss_head_loss": final.get("train", {}).get("loss_loss"),
            "train_loss_head_top1": final.get("train", {}).get("loss_top1"),
            "val_gain_loss": final.get("val", {}).get("gain_loss"),
            "val_loss_head_loss": final.get("val", {}).get("loss_loss"),
            "val_loss_head_top1": final.get("val", {}).get("loss_top1"),
            "used_pretrained": bool(pretrained),
            "freeze_backbone": bool(freeze_backbone),
            "requested_freeze_backbone_epochs": int(resolved_freeze_backbone_epochs),
            "freeze_backbone_epochs": int(effective_freeze_backbone_epochs),
            "unfreeze_last_n_layers": int(resolved_unfreeze_last_n_layers),
            "backbone_lr_scale": float(resolved_backbone_lr_scale),
            "head_lr_scale": float(resolved_head_lr_scale),
            "uses_scaled_learning_rates": bool(uses_scaled_learning_rates),
            "uses_discriminative_lr": bool(uses_discriminative_lr),
            "adaptation_strategy": strategy,
            "training_scope": final.get("backbone_trainable_scope", "gain_head_only" if freeze_backbone else "full_backbone"),
            "selected_phase": final.get("phase"),
            "backbone_trainable_scope": final.get("backbone_trainable_scope"),
            "n_trainable_backbone_params": final.get("n_trainable_backbone_params"),
            "n_total_backbone_params": final.get("n_total_backbone_params"),
            "selected_backbone_lr": final.get("backbone_lr"),
            "selected_head_lr": final.get("head_lr"),
            "train_val_gap": (
                None
                if any(math.isnan(float(x)) for x in [final.get("train", {}).get("loss", float("nan")), final.get("val", {}).get("loss", float("nan"))])
                else float(final["train"]["loss"] - final["val"]["loss"])
            ),
            "phase_history": [
                {
                    "epoch": int(item["epoch"]),
                    "phase": item.get("phase"),
                    "backbone_trainable_scope": item.get("backbone_trainable_scope"),
                    "n_trainable_backbone_params": item.get("n_trainable_backbone_params"),
                    "backbone_lr": item.get("backbone_lr"),
                    "head_lr": item.get("head_lr"),
                }
                for item in history
            ],
        }
        return self.fit_report

    @torch.no_grad()
    def predict_scores(self, src_state: np.ndarray) -> np.ndarray:
        self.net.eval()
        x = torch.as_tensor(np.asarray(src_state, dtype=np.int64)[None, :], dtype=torch.long, device=self.device)
        logits = self.net.gain_logits(x)[0]
        scores = torch.softmax(logits, dim=0).detach().cpu().numpy().astype(float)
        scores = np.where(np.asarray(src_state) == 0, scores, 0.0)
        if scores.sum() <= 0:
            mask = (np.asarray(src_state) == 0)
            scores = np.zeros_like(scores)
            if mask.sum() > 0:
                scores[mask] = 1.0 / mask.sum()
        return scores

    predict_fn = predict_scores

    @torch.no_grad()
    def predict_loss_scores(self, src_state: np.ndarray) -> np.ndarray:
        """loss 侧分数: 候选 = src==1 且属可逆先验; 未启用时返回全 0。"""
        src = np.asarray(src_state)
        cand = self._side_candidate(src.astype(np.int8), "loss")
        if not self._loss_head_enabled() or cand.sum() == 0:
            return np.zeros(self.config["L"], dtype=float)
        self.net.eval()
        x = torch.as_tensor(src.astype(np.int64)[None, :], dtype=torch.long, device=self.device)
        logits = self.net.loss_logits(x)[0]
        scores = torch.softmax(logits, dim=0).detach().cpu().numpy().astype(float)
        scores = np.where(cand, scores, 0.0)
        if scores.sum() <= 0:
            scores = np.zeros_like(scores)
            scores[cand] = 1.0 / cand.sum()
        return scores

    predict_loss_fn = predict_loss_scores

    def load_pretrained(self, checkpoint_path: str) -> dict:
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = blob["state_dict"]
        backbone_state = {k: v for k, v in state_dict.items() if k.startswith("backbone.")}
        self.net.load_state_dict(backbone_state, strict=False)
        self.pretrain_report = blob.get("pretrain_report") or {}
        self.pretrain_report.setdefault("transfer_scope", "backbone_only")
        self.pretrain_report.setdefault("loaded_param_keys", sorted(backbone_state.keys()))
        return blob

    def save(self, path: str, extra: Optional[dict] = None) -> str:
        blob = {
            "config": copy.deepcopy(self.config),
            "feat_meta": serialize_feat_meta(self.feat_meta),
            "state_dict": self.net.state_dict(),
            "pretrain_report": _jsonable(self.pretrain_report),
            "fit_report": _jsonable(self.fit_report),
            # gain+loss 双头: 不放进 config, 以免破坏 load() 的 cls(**cfg) 反构造
            "head_mode": self._effective_head_mode,
            "requested_head_mode": self.head_mode,
            "loss_weight": float(self.loss_weight),
            "loss_mask": (None if self.loss_mask is None
                          else np.asarray(self.loss_mask).astype(int).tolist()),
        }
        if extra:
            blob.update(extra)
        torch.save(blob, path)
        return path

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "TwoStageTransitionModel":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        feat_meta = deserialize_feat_meta(blob.get("feat_meta"))
        cfg = dict(blob["config"])
        loss_mask = blob.get("loss_mask")
        if loss_mask is not None:
            loss_mask = np.asarray(loss_mask, dtype=np.float32)
        actual_head_mode = blob.get("head_mode", "gain_only")
        model = cls(
            feat_meta=feat_meta,
            device=device,
            head_mode=actual_head_mode,
            loss_mask=loss_mask,
            loss_weight=float(blob.get("loss_weight", 1.0)),
            **cfg,
        )
        model.head_mode = blob.get("requested_head_mode", actual_head_mode)
        model._effective_head_mode = actual_head_mode
        model.net.load_state_dict(blob["state_dict"], strict=False)
        model.pretrain_report = blob.get("pretrain_report") or {}
        model.alignment_report = blob.get("alignment_report") or {}
        model.fit_report = blob.get("fit_report") or {}
        return model


def first_step_ranking_table(model: TwoStageTransitionModel, feat_ids: Optional[List[str]], hdf: Any) -> Any:
    import pandas as pd

    L = model.config["L"]
    s0 = np.zeros(L, dtype=np.int8)
    scores = model.predict_scores(s0)
    order = np.argsort(-scores)
    rank_cols = {
        "feature_index": order,
        "score_from_empty": scores[order],
        "rank": np.arange(1, L + 1),
    }
    rank_cols = TDL.attach_feature_labels(rank_cols, order, feat_ids, hdf)
    return pd.DataFrame(rank_cols)
