"""从零实现的棋局序列 Transformer 与指针式行棋头。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import SEQUENCE_LENGTH, TOKENS


@dataclass
class ModelConfig:
    vocab_size: int = len(TOKENS)
    hidden_size: int = 64
    num_layers: int = 3
    num_heads: int = 4
    intermediate_size: int = 192
    max_seq_len: int = SEQUENCE_LENGTH
    dropout: float = 0.1


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.hidden_size % config.num_heads:
            raise ValueError("hidden_size 必须能被 num_heads 整除")
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, capture_attention: bool = False) -> torch.Tensor:
        batch, length, hidden = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (item.transpose(1, 2) for item in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        if capture_attention:
            self.last_attention = weights.detach()
        context = self.dropout(weights) @ v
        context = context.transpose(1, 2).contiguous().view(batch, length, hidden)
        return self.out(context)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.up = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.gelu(self.up(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.attention = MultiHeadSelfAttention(config)
        self.norm2 = RMSNorm(config.hidden_size)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, capture_attention: bool = False) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), capture_attention=capture_attention)
        x = x + self.ffn(self.norm2(x))
        return x


class QiYuTransformer(nn.Module):
    """读取局面 token，输出来源格与目标格的联合策略分数。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.source_query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.source_key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.destination_query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.destination_key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Tanh(),
        )

    def forward(
        self, input_ids: torch.Tensor, capture_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        for index, block in enumerate(self.blocks):
            x = block(x, capture_attention=capture_attention and index == len(self.blocks) - 1)
        x = self.norm(x)

        squares = x[:, 1:91, :]
        decision = x[:, -1, :]
        source_logits = torch.einsum(
            "bd,bsd->bs", self.source_query(decision), self.source_key(squares)
        ) / math.sqrt(self.config.hidden_size)
        destination_logits = torch.einsum(
            "bsd,btd->bst", self.destination_query(squares), self.destination_key(squares)
        ) / math.sqrt(self.config.hidden_size)
        value = self.value_head(decision).squeeze(-1)

        attention = None
        if capture_attention:
            raw = self.blocks[-1].attention.last_attention
            if raw is not None:
                attention = raw[:, :, -1, 1:91].mean(dim=1)
        return source_logits, destination_logits, value, attention

    def loss(
        self,
        input_ids: torch.Tensor,
        sources: torch.Tensor,
        destinations: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        value_weight: float = 0.25,
        legal_move_mask: Optional[torch.Tensor] = None,
        policy_loss: str = "factorized",
    ) -> torch.Tensor:
        source_logits, destination_logits, predicted_values, _ = self(input_ids)
        batch_index = torch.arange(input_ids.size(0), device=input_ids.device)
        destination_for_source = destination_logits[batch_index, sources]
        if policy_loss == "factorized":
            policy_objective = F.cross_entropy(source_logits, sources) + F.cross_entropy(destination_for_source, destinations)
        elif policy_loss == "masked_joint":
            if legal_move_mask is None:
                raise ValueError("masked_joint 损失需要 legal_move_mask")
            joint_logits = source_logits.unsqueeze(-1) + destination_logits
            joint_logits = joint_logits.masked_fill(~legal_move_mask, torch.finfo(joint_logits.dtype).min)
            policy_objective = F.cross_entropy(joint_logits.flatten(1), sources * 90 + destinations)
        else:
            raise ValueError(f"未知策略损失：{policy_loss}")
        if values is None:
            return policy_objective
        return policy_objective + value_weight * F.mse_loss(predicted_values, values)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def joint_move_log_probs(source_logits: torch.Tensor, destination_logits: torch.Tensor) -> torch.Tensor:
    """返回与分解交叉熵严格一致的 log p(source, destination)。"""
    return F.log_softmax(source_logits, dim=-1).unsqueeze(-1) + F.log_softmax(destination_logits, dim=-1)


def save_checkpoint(path: str, model: QiYuTransformer, extra: Optional[Dict] = None) -> None:
    payload = {"config": asdict(model.config), "model_state": model.state_dict(), "extra": extra or {}}
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device) -> Tuple[QiYuTransformer, Dict]:
    # 项目 checkpoint 是本地训练产物，包含 config/extra 等非张量元数据；
    # 新版 PyTorch 默认 weights_only=True 时会拒绝这些旧文件。
    payload = torch.load(path, map_location=device, weights_only=False)
    model = QiYuTransformer(ModelConfig(**payload["config"]))
    # strict=False 保持第一版（没有 value head）的旧权重仍可打开。
    model.load_state_dict(payload["model_state"], strict=False)
    model.to(device).eval()
    return model, payload.get("extra", {})
