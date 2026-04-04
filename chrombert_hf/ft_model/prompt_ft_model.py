from __future__ import annotations

import torch
from .basic_model import BasicModel
from .ft_utils import (
    PromptHeader,
    PromptsEmb,
    AdapterExternalEmb,
    extract_last_hidden_state,
)


class ChromBERTPrompt(BasicModel):
    """
    Fine-tune a Hugging Face ChromBERT checkpoint using enhanced prompt
    for TFBS prediction.
    """

    def create_layers(self):
        self.pretrain_model = self.init_pretrain_model()

        if self.finetune_config.prompt_kind == "expression":
            self.adapter_cell_emb = AdapterExternalEmb(
                prompt_dim_external=self.finetune_config.prompt_dim_external,
                dropout=self.finetune_config.dropout,
            )

        self.gather_emb = PromptsEmb()
        self.ft_header = PromptHeader(
            n_parts=self.finetune_config.n_prompt_parts + 1,
            dropout=self.finetune_config.dropout,
        )
        return None

    def forward(self, batch):
        emb_cell, emb_regulator, emb_all = self.get_emb_parts(
            batch, dtype=self.ft_header.fcs[0].fc1.weight.dtype
        )
        header_out = self.ft_header(emb_cell, emb_regulator, emb_all)
        return header_out

    def get_emb_parts(self, batch, dtype=torch.bfloat16):
        """
        Gather the necessary inputs for forwarding,
        handling cached embedding or forwarding directly.
        """
        if "emb_cell" not in batch or "emb_regulator" not in batch:
            input_ids = batch["input_ids"]
            position_ids = batch["position_ids"]
            chrombert_out = self.pretrain_model(
                input_ids=input_ids.long(),
                position_ids=position_ids.long(),
                return_dict=True,
            )
            chrombert_out = extract_last_hidden_state(chrombert_out)

        if "emb_cell" in batch:
            emb_cell = batch["emb_cell"]
        else:
            prompts_cell = batch["prompts_cell"]
            emb_cell = self.gather_emb(chrombert_out, prompts_cell)

        if "emb_regulator" in batch:
            emb_regulator = batch["emb_regulator"]
            emb_all = batch["emb_all"]
        else:
            prompts_all = batch["prompts_all"]
            prompts_regulator = batch["prompts_regulator"]
            emb_regulator = self.gather_emb(chrombert_out, prompts_regulator)
            emb_all = self.gather_emb(chrombert_out, prompts_all)

        if self.finetune_config.prompt_kind == "expression":
            emb_cell = self.adapter_cell_emb(emb_cell)

        return emb_cell.to(dtype), emb_regulator.to(dtype), emb_all.to(dtype)
