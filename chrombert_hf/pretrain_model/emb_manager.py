from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn


class MaskProjector(nn.Module):
    """
    Project dataset/cistrome-level hidden states to regulator-level hidden states.

    input:
        hidden_states: [B, D, H]
    output:
        regulator_hidden: [B, F, H]

    D = number of datasets/cistromes
    F = number of regulators/factors
    """

    def __init__(
        self,
        mask_path: str,
        ignore: bool = False,
        ignore_index=None,
    ):
        super().__init__()

        if mask_path is None:
            raise ValueError("mask_path must be specified")
        if not isinstance(mask_path, str):
            raise TypeError("mask_path must be a string")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"{mask_path} does not exist")

        mask_path = str(Path(mask_path).resolve())
        mask_df = pd.read_csv(mask_path, sep="\t", index_col=0)
        mask_df = mask_df[sorted(mask_df.columns)]

        mask = torch.tensor(mask_df.values, dtype=torch.float32)
        gsmid_names = mask_df.index.tolist()
        regulator_names = mask_df.columns.tolist()

        if ignore:
            if ignore_index is None or len(ignore_index) != 2:
                raise ValueError(
                    "ignore_index must be a tuple/list of "
                    "(ignore_gsmid_index, ignore_regulator_index)"
                )

            ignore_gsmid_index = set(ignore_index[0])
            ignore_regulator_index = set(ignore_index[1])

            rows_to_keep = torch.tensor(
                [i not in ignore_gsmid_index for i in range(mask.shape[0])]
            )
            cols_to_keep = torch.tensor(
                [j not in ignore_regulator_index for j in range(mask.shape[1])]
            )

            mask = mask[rows_to_keep][:, cols_to_keep]
            mask_df = mask_df.iloc[rows_to_keep.numpy(), cols_to_keep.numpy()]
            gsmid_names = mask_df.index.tolist()
            regulator_names = mask_df.columns.tolist()

        factor_num = (mask != 0).sum(dim=0).clamp(min=1).to(torch.float32)
        normalized_mask = mask / factor_num

        self.mask_path = mask_path
        self.gsmid_names = gsmid_names
        self.regulator_names = regulator_names
        self.gsmid_to_idx = {x.lower(): i for i, x in enumerate(self.gsmid_names)}
        self.regulator_to_idx = {x.lower(): i for i, x in enumerate(self.regulator_names)}

        # Not trainable, but will follow model.to(device)
        self.register_buffer("normalized_mask", normalized_mask)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Input: hidden_states: [B, D, H]
        return: [B, F, H]
        """
        mask = self.normalized_mask.to(dtype=hidden_states.dtype)
        return torch.matmul(hidden_states.transpose(1, 2), mask).transpose(1, 2)

    def get_cistrome_embedding(self, hidden_states: torch.Tensor, gsmid: str) -> torch.Tensor:
        idx = self.gsmid_to_idx[gsmid.lower()]
        return hidden_states[:, idx, :]

    def get_regulator_embedding_from_all(
        self,
        regulator_hidden: torch.Tensor,
        regulator: str,
    ) -> torch.Tensor:
        """
        Input: regulator_hidden: [B, F, H]
        return: [B, H]
        """
        idx = self.regulator_to_idx[regulator.lower()]
        return regulator_hidden[:, idx, :]

    def get_regulator_embedding(
        self,
        hidden_states: torch.Tensor,
        regulator: str,
    ) -> torch.Tensor:
        """
        Get the embedding of a single regulator from the cistrome hidden states.
        No need to compute all regulator hidden states first.

        Input: hidden_states: [B, D, H]
        return: [B, H]
        """
        idx = self.regulator_to_idx[regulator.lower()]
        mask_col = self.normalized_mask[:, idx : idx + 1].to(dtype=hidden_states.dtype).to(hidden_states.device)
        out = torch.matmul(hidden_states.transpose(1, 2), mask_col).transpose(1, 2)
        return out[:, 0, :]

    def get_region_embedding(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.mean(dim=1)


class ChromBERTEmbedding(nn.Module):
    """
    A lightweight wrapper:
    - forward(batch): Run the backbone once, and cache the cistrome / regulator hidden states
    - get_regulator_embedding(name): Get the embedding of a single regulator from the cached hidden states
    - get_single_regulator_from_batch(batch, name): Do not cache all regulator hidden states, only get one as needed
    """

    def __init__(
        self,
        backbone: nn.Module,
        mask_path: str,
        ignore: bool = False,
        ignore_index=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.projector = MaskProjector(
            mask_path,
            ignore=ignore,
            ignore_index=ignore_index,
        )

        self.hidden_cistrome: Optional[torch.Tensor] = None
        self.hidden_regulator: Optional[torch.Tensor] = None

    @property
    def gsmid_names(self):
        return self.projector.gsmid_names

    @property
    def regulator_names(self):
        return self.projector.regulator_names

    @torch.no_grad()
    def encode_cistrome(self, batch) -> torch.Tensor:
        was_training = self.backbone.training
        self.backbone.eval()
        if hasattr(self.backbone, "config"):
            hidden = self.backbone(
                batch["input_ids"].long(),
                batch["position_ids"].long(),
            )
        else:
            hidden = self.backbone(batch["input_ids"], batch["position_ids"])
        if hasattr(hidden, "last_hidden_state"):
            hidden = hidden.last_hidden_state
        elif isinstance(hidden, tuple):
            hidden = hidden[0]
        if was_training:
            self.backbone.train()
        return hidden

    @torch.no_grad()
    def forward(self, batch) -> torch.Tensor:
        """
        Compute all regulator embeddings at once and cache them.
        return: [B, F, H]
        """
        hidden = self.encode_cistrome(batch)
        regulator_hidden = self.projector(hidden)

        self.hidden_cistrome = hidden
        self.hidden_regulator = regulator_hidden
        return regulator_hidden

    def get_hidden_state(self):
        if self.hidden_cistrome is None:
            raise ValueError("No cached cistrome hidden state. Please call forward(batch) first.")
        return self.hidden_cistrome

    def get_cistrome_embedding(self, gsmid: str):
        if self.hidden_cistrome is None:
            raise ValueError("No cached cistrome hidden state. Please call forward(batch) first.")
        return self.projector.get_cistrome_embedding(self.hidden_cistrome, gsmid)

    def get_regulator_embedding(self, regulator: str):
        """
        Get the embedding of a single regulator from the cached hidden states.
        """
        if self.hidden_regulator is None:
            raise ValueError("No cached regulator hidden state. Please call forward(batch) first.")
        return self.projector.get_regulator_embedding_from_all(
            self.hidden_regulator,
            regulator,
        )

    def get_region_embedding(self):
        if self.hidden_cistrome is None:
            raise ValueError("No cached cistrome hidden state. Please call forward(batch) first.")
        return self.projector.get_region_embedding(self.hidden_cistrome)

    @torch.no_grad()
    def get_single_regulator_from_batch(self, batch, regulator: str):
        """
        Get the embedding of a single regulator:
        - The backbone is only forward once
        - The entire regulator hidden states are not projected
        """
        hidden = self.encode_cistrome(batch)
        return self.projector.get_regulator_embedding(hidden, regulator)

    @torch.no_grad()
    def get_single_cistrome_from_batch(self, batch, gsmid: str):
        hidden = self.encode_cistrome(batch)
        return self.projector.get_cistrome_embedding(hidden, gsmid)

    @torch.no_grad()
    def get_region_from_batch(self, batch):
        hidden = self.encode_cistrome(batch)
        return self.projector.get_region_embedding(hidden)
