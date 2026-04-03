import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import subprocess as sp
import pandas as pd
try:
    import flash_attn

    if flash_attn.__version__.split(".")[0] == "1":
        from flash_attn.flash_attention import FlashAttention  # pyright: ignore[reportMissingImports]

        flash_attention_version = 1
    else:
        from flash_attn import flash_attn_qkvpacked_func

        flash_attention_version = 2
except ImportError:
    flash_attn = None
    flash_attention_version = 0


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, index=None):
        y = sublayer(self.norm(x))
        if index is not None:
            y1 = y[index]
            return x + self.dropout(y1), y[-1]
        return x + self.dropout(y)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class TokenEmbedding(nn.Embedding):
    def __init__(self, config):
        super().__init__(config.vocab_size, config.hidden_dim, config.token_id_pad)


class PositionalEmbeddingTrainable(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pe = nn.Embedding(config.n_datasets, config.hidden_dim)
        self.d_model = config.hidden_dim
        self.n_datasets = config.n_datasets

    def forward(self, x):
        return self.pe(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.pe_mode != "train":
            raise ValueError(f"only support train mode for positional embedding! {config.pe_mode} is not supported!")
        self.pe = PositionalEmbeddingTrainable(config)

    def forward(self, x):
        return self.pe(x)


class BERTEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token = TokenEmbedding(config)
        self.position = PositionalEmbedding(config)
        self.dropout = nn.Dropout(p=config.dropout)
        self.embed_size = config.hidden_dim
        self.dtype = config.dtype
        self.config = config

    def forward(self, sequence, position_ids):
        sequence = sequence.long()
        x = self.token(sequence) + self.position(position_ids)
        return self.dropout(x).to(self.dtype)


class SelfAttentionFlashMHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_dim
        self.causal = config.flash_causal
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_attention_heads")

        factory_kwargs = {"device": config.flash_device, "dtype": config.dtype}
        self.Wqkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.flash_bias, **factory_kwargs)
        self.dropout_p = config.dropout
        self.dtype = config.dtype
        if flash_attention_version == 1:
            self.inner_attn = FlashAttention(attention_dropout=config.dropout)
        elif flash_attention_version == 2:
            self.inner_attn = flash_attn_qkvpacked_func
        else:
            self.inner_attn = None

    def _split_qkv(self, qkv):
        batch_size, seq_len, _ = qkv.shape
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        return q, k, v

    def _torch_attention(self, qkv, key_padding_mask=None):
        q, k, v = self._split_qkv(qkv)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(torch.bool)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        if self.training and self.dropout_p > 0:
            attn = F.dropout(attn, p=self.dropout_p)

        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        batch_size, seq_len, _, _ = context.shape
        context = context.view(batch_size, seq_len, self.embed_dim)
        return context, attn

    def _flash_attention(self, qkv, key_padding_mask=None, need_weights=False):
        if flash_attention_version == 2:
            context = self.inner_attn(qkv, dropout_p=self.dropout_p, causal=self.causal)
            context = context.contiguous().view(qkv.shape[0], qkv.shape[1], self.embed_dim)
            return context, None

        if flash_attention_version == 1:
            context, _ = self.inner_attn(
                qkv,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                causal=self.causal,
            )
            context = context.contiguous().view(qkv.shape[0], qkv.shape[1], self.embed_dim)
            return context, None

        return self._torch_attention(qkv, key_padding_mask=key_padding_mask)

    def forward(self, x, key_padding_mask=None, need_weights=False, attn_weight=False):
        x = x.to(self.dtype)
        qkv = self.Wqkv(x)

        qkv_flash = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, self.head_dim)
        context, attn = self._flash_attention(
            qkv_flash,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )

        if attn_weight:
            if attn is None:
                attn = self._torch_attention(qkv, key_padding_mask=key_padding_mask)[1]
            return context, attn.detach().cpu()
        return context


class EncoderTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = SelfAttentionFlashMHA(config)
        self.feed_forward = PositionwiseFeedForward(
            d_model=config.hidden_dim,
            d_ff=config.feed_forward_dim,
            dropout=config.dropout,
        )
        self.input_sublayer = SublayerConnection(size=config.hidden_dim, dropout=config.dropout)
        self.output_sublayer = SublayerConnection(size=config.hidden_dim, dropout=config.dropout)
        self.dropout = nn.Dropout(p=config.dropout)
        self.dtype = config.dtype

    def forward(self, x, mask, attn_weight=False):
        x = x.to(self.dtype)
        if attn_weight:
            x, out_attn = self.input_sublayer(
                x,
                lambda x: self.attention.forward(x, mask, need_weights=False, attn_weight=attn_weight),
                index=0,
            )
        else:
            x = self.input_sublayer(x, lambda x: self.attention.forward(x, mask))

        x = self.output_sublayer(x, self.feed_forward)
        if attn_weight:
            return x, out_attn
        return x
    
    
