from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint
from chrombert_hf.pretrain_model.emb_manager import ChromBERTEmbedding, MaskProjector



def extract_last_hidden_state(outputs):
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.norm = LayerNorm(out_features)
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.norm(self.fc2(out))
        out = self.dropout(out)
        out = out + self.shortcut(x)
        return F.relu(out)


class GeneralHeader(nn.Module):
    def __init__(self, hidden_dim, dim_output, mtx_mask, ignore=False, ignore_index=None, dropout=0.1, medium_dim=256):
        super().__init__()
        self.interface = MaskProjector(mtx_mask, ignore=ignore, ignore_index=ignore_index)
        self.conv = nn.Conv2d(1, 1, (1, hidden_dim))
        self.activation = nn.ReLU()
        self.res1 = ResidualBlock(in_features=self.interface.normalized_mask.shape[1], out_features=1024, dropout=dropout)
        self.res2 = ResidualBlock(in_features=1024, out_features=hidden_dim, dropout=dropout)
        self.res3 = ResidualBlock(in_features=hidden_dim, out_features=medium_dim, dropout=dropout)
        self.fc = nn.Linear(in_features=medium_dim, out_features=dim_output, bias=True)

    def forward(self, x, return_emb=False):
        x = self.interface(x)
        x = x.permute(0, 2, 1)
        x = self.res1(x)
        x = self.res2(x)
        x = x[:, None, :, :]
        x = self.conv(x)
        x = self.activation(x)
        x = x.view(x.shape[0], -1)
        if return_emb:
            return x
        x = self.res3(x)
        return self.fc(x)


class GepHeader(nn.Module):
    def __init__(self, hidden_dim, dim_output, mtx_mask, ignore=False, ignore_index=None, dropout=0.1, medium_dim=256):
        super().__init__()
        self.interface = MaskProjector(mtx_mask, ignore=ignore, ignore_index=ignore_index)
        self.conv = nn.Conv2d(1, 1, (1, hidden_dim))
        self.activation = nn.ReLU()
        self.res1 = ResidualBlock(in_features=self.interface.normalized_mask.shape[1], out_features=1024, dropout=dropout)
        self.res2 = ResidualBlock(in_features=1024, out_features=hidden_dim, dropout=dropout)
        self.res3 = ResidualBlock(in_features=hidden_dim, out_features=medium_dim, dropout=dropout)
        self.zero_inflation = nn.Linear(in_features=medium_dim, out_features=1)
        self.regression = nn.Linear(in_features=medium_dim, out_features=dim_output, bias=True)

    def forward(self, x, **kwargs):
        x = self.interface(x)
        x = x.permute(0, 2, 1)
        x = self.res1(x)
        x = self.res2(x)
        x = x[:, None, :, :]
        x = self.conv(x)
        x = self.activation(x)
        x = x.view(x.shape[0], -1)
        x = self.res3(x)
        zero_prob_logit = self.zero_inflation(x)
        reg_value = self.regression(x)
        return zero_prob_logit, reg_value


class PoolFlankWindow(nn.Module):
    def __init__(self, flank_region_num=9, pretrain_model=None, parallel_embedding=False, gradient_checkpoint=False):
        super().__init__()
        self.flank_region_num = flank_region_num
        self.pretrain_model = pretrain_model
        self.parallel_embedding = parallel_embedding
        self.gradient_checkpoint = gradient_checkpoint

    def forward(self, x, position_ids):
        batch_size = x.shape[0]
        seq_len = x.shape[-1]
        x = x.float()
        x.requires_grad = True
        if not self.parallel_embedding:
            embeddings = []
            if not self.gradient_checkpoint:
                for i in range(self.flank_region_num):
                    x_i = x[:, i, :].clone()
                    position_ids_i = position_ids[:, i, :].clone()
                    x_i = self.pretrain_model(
                        input_ids=x_i.long(),
                        position_ids=position_ids_i.long(),
                        return_dict=True,
                    )
                    x_i = extract_last_hidden_state(x_i)
                    embeddings.append(x_i)
                x = torch.stack(embeddings, dim=1)
            else:
                hidden_dim = getattr(self.pretrain_model.config, "hidden_dim", 768)
                all_embeddings = torch.zeros((batch_size, self.flank_region_num, seq_len, hidden_dim), device=x.device)
                for i in range(self.flank_region_num):
                    all_embeddings[:, i, :, :] = checkpoint(
                        lambda a, b: extract_last_hidden_state(
                            self.pretrain_model(input_ids=a.long(), position_ids=b.long(), return_dict=True)
                        ),
                        x[:, i, :],
                        position_ids[:, i, :],
                    )
                x = all_embeddings
        else:
            x = rearrange(x, "b n l -> (b n) l")
            position_ids = rearrange(position_ids, "b n l -> (b n) l")
            x = self.pretrain_model(
                input_ids=x.long(),
                position_ids=position_ids.long(),
                return_dict=True,
            )
            x = extract_last_hidden_state(x)
            x = rearrange(x, "(b n) l h -> b n l h", b=batch_size)

        x = rearrange(x, "b n l h -> b l n h", b=batch_size)
        return torch.max(x, dim=-2).values


__all__ = [
    "ChromBERTEmbedding",
    "extract_last_hidden_state",
    "GepHeader",
    "GeneralHeader",
    "PoolFlankWindow",
]
