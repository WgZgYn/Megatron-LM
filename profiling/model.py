"""Standalone mini GPT-style transformer for distributed-training profiling.

The model is written *TP-native*: at ``tp_size == 1`` it is an ordinary
transformer (``nn.Linear`` everywhere); at ``tp_size > 1`` the caller injects
``ColumnParallelLinear`` / ``RowParallelLinear`` factories and the attention
heads are split across the TP ranks. This keeps a single forward path shared
by the DP / TP / PP modes, which makes the per-mode communication profile
directly comparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 50304
    hidden_size: int = 1024
    num_layers: int = 8
    num_heads: int = 16
    ffn_hidden: int = 4096
    seq_len: int = 256
    dropout: float = 0.0
    dtype: torch.dtype = torch.float32


def _init_weights(module: nn.Module) -> None:
    """Deterministic GPT-2-style init shared by every mode.

    NOTE: only touches ``nn.Linear`` / ``nn.Embedding``. The TP-parallel
    linear layers are custom ``nn.Module`` subclasses and already initialise
    their (sliced) weights in ``__init__``, so ``model.apply`` skips them.
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, tp_size: int, col_factory, row_factory):
        super().__init__()
        self.cfg = cfg
        self.tp_size = tp_size
        self.local_heads = cfg.num_heads // tp_size
        self.head_dim = cfg.hidden_size // cfg.num_heads

        # Column-parallel QKV projection: each rank owns heads/tp * 3 outputs.
        self.qkv = col_factory(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        # Row-parallel output projection: input sharded, output replicated.
        self.proj = row_factory(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)                                    # [B, T, 3*C/tp]
        qkv = qkv.reshape(B, T, 3, self.local_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                          # each [B, T, h_local, d]

        q = q.transpose(1, 2)                                # [B, h_local, T, d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v                                          # [B, h_local, T, d]
        y = y.transpose(1, 2).contiguous().reshape(B, T, self.local_heads * self.head_dim)
        return self.proj(y)                                  # [B, T, C]


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig, tp_size: int, col_factory, row_factory):
        super().__init__()
        self.up = col_factory(cfg.hidden_size, cfg.ffn_hidden, bias=False)   # column
        self.down = row_factory(cfg.ffn_hidden, cfg.hidden_size, bias=False)  # row

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, tp_size: int, col_factory, row_factory):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = CausalSelfAttention(cfg, tp_size, col_factory, row_factory)
        self.ln2 = nn.LayerNorm(cfg.hidden_size)
        self.mlp = MLP(cfg, tp_size, col_factory, row_factory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        tp_size: int = 1,
        col_factory=None,
        row_factory=None,
    ):
        super().__init__()
        assert cfg.num_heads % tp_size == 0, (cfg.num_heads, tp_size)
        assert cfg.hidden_size % cfg.num_heads == 0
        self.cfg = cfg
        self.tp_size = tp_size

        col = col_factory if col_factory is not None else nn.Linear
        row = row_factory if row_factory is not None else nn.Linear

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.hidden_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, tp_size, col, row) for _ in range(cfg.num_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # Replicate GPT-2 init for the non-parallel layers (embeddings/head).
        self.apply(_init_weights)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
