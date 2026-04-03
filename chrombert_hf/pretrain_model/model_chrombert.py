from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from .emb_manager import ChromBERTEmbedding
from .config_chrombert import ChromBERTConfig
from .utils import BERTEmbedding, EncoderTransformerBlock


class ChromBERT(nn.Module):
    def __init__(self, config: ChromBERTConfig):
        super().__init__()
        self.config = config
        self.hidden = config.hidden_dim
        self.n_layers = config.num_layers
        self.attn_heads = config.num_attention_heads
        self.feed_forward_hidden = config.feed_forward_dim

        self.embedding = BERTEmbedding(config)
        self.transformer_blocks = nn.ModuleList(
            [EncoderTransformerBlock(config) for _ in range(self.n_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_weight: bool = False,
        attn_layer: Optional[int] = None,
    ):
        x = self.embedding(x, position_ids)

        if attn_layer == -1:
            attn = []

        for i, transformer in enumerate(self.transformer_blocks):
            if attn_weight:
                if attn_layer == -1:
                    x, attn_score = transformer.forward(
                        x, key_padding_mask, attn_weight=True
                    )
                    attn.append(attn_score)
                elif i == attn_layer:
                    x, attn = transformer.forward(
                        x, key_padding_mask, attn_weight=True
                    )
                else:
                    x = transformer.forward(x, key_padding_mask, attn_weight=False)
            else:
                x = transformer.forward(x, key_padding_mask, attn_weight=False)

        return (x, attn) if attn_weight else x

    def load_ckpt(self, ckpt_path: str):
        ck = torch.load(ckpt_path, map_location=torch.device("cpu"))
        self.load_state_dict(ck)
        return None

    def freeze(self, trainable: int = 2):
        if not isinstance(trainable, int):
            raise TypeError("trainable should be an integer")
        if trainable < 0:
            raise ValueError("trainable should be non-negative")

        for _, parameter in self.named_parameters():
            parameter.requires_grad = False

        total_layers = len(self.transformer_blocks)
        if trainable > total_layers:
            raise ValueError(
                "trainable should not be greater than total transformer blocks"
            )

        for i in range(total_layers - trainable, total_layers):
            for _, parameter in self.transformer_blocks[i].named_parameters():
                parameter.requires_grad = True
        return None

    def display_trainable_parameters(self, verbose: bool = True):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        output = {"total_params": total_params, "trainable_params": trainable_params}
        print(output)
        if verbose:
            for name, parameter in self.named_parameters():
                print(name, ": trainable" if parameter.requires_grad else ": frozen")
        return output


class ChromBERTModel(PreTrainedModel):
    config_class = ChromBERTConfig
    base_model_prefix = "chrombert"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False

    def __init__(self, config: ChromBERTConfig):
        super().__init__(config)
        self.chrombert = ChromBERT(config)
        self.post_init()

    @staticmethod
    def _resolve_local_mask_path(
        model_dir: str | os.PathLike,
        mask_matrix: Optional[str],
    ) -> Optional[str]:
        if mask_matrix is None:
            return None

        if os.path.isabs(mask_matrix) and os.path.exists(mask_matrix):
            return str(Path(mask_matrix).resolve())

        model_dir = Path(model_dir)
        candidate = model_dir / mask_matrix
        if candidate.exists():
            return str(candidate.resolve())

        return None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        mask_matrix = model.config.mask_matrix
        if mask_matrix is None:
            return model

        # 1) local directory
        local_resolved = cls._resolve_local_mask_path(
            pretrained_model_name_or_path, mask_matrix
        )
        if local_resolved is not None:
            model.config.mask_matrix = local_resolved
            return model

        # 2) HF Hub
        try:
            downloaded = hf_hub_download(
                repo_id=str(pretrained_model_name_or_path),
                filename=mask_matrix,
            )
            model.config.mask_matrix = downloaded
            return model
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not resolve mask_matrix='{mask_matrix}' from "
                f"local dir or Hub repo '{pretrained_model_name_or_path}'."
            ) from exc

    def save_pretrained(self, save_directory, **kwargs):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        original_mask_matrix = self.config.mask_matrix
        resolved_mask_src = None
        portable_name = None

        if original_mask_matrix is not None:
            mask_src = Path(original_mask_matrix)

            candidates = [
                mask_src,
                Path(__file__).resolve().parent / mask_src,
                Path.cwd() / mask_src,
            ]

            for cand in candidates:
                if cand.exists():
                    resolved_mask_src = cand.resolve()
                    portable_name = resolved_mask_src.name
                    break

            if resolved_mask_src is None:
                raise FileNotFoundError(
                    f"Configured mask_matrix file does not exist: {original_mask_matrix}\n"
                    f"Tried: {[str(c) for c in candidates]}"
                )

            # Write portable filename to config.json on disk
            self.config.mask_matrix = portable_name

        super().save_pretrained(save_directory, **kwargs)

        if resolved_mask_src is not None:
            mask_dst = save_directory / portable_name
            if not mask_dst.exists() or resolved_mask_src != mask_dst.resolve():
                shutil.copy2(resolved_mask_src, mask_dst)

            self.config.mask_matrix = portable_name
            self.config.save_pretrained(save_directory)

        # Restore runtime absolute path
        self.config.mask_matrix = (
            str(resolved_mask_src) if resolved_mask_src is not None else original_mask_matrix
        )

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        output_attentions=None,
        return_dict=None,
        **kwargs,
    ):
        del kwargs

        output_attentions = (
            self.config.output_attentions
            if output_attentions is None
            else output_attentions
        )
        return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )

        if input_ids is None:
            raise ValueError("input_ids must be provided")
        if position_ids is None:
            raise ValueError("position_ids must be provided for ChromBERT")

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        outputs = self.chrombert(
            input_ids,
            position_ids,
            key_padding_mask=key_padding_mask,
            attn_weight=output_attentions,
            attn_layer=-1 if output_attentions else None,
        )

        if output_attentions:
            last_hidden_state, attentions = outputs
        else:
            last_hidden_state = outputs
            attentions = None

        if not return_dict:
            if output_attentions:
                return last_hidden_state, attentions
            return last_hidden_state

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            attentions=attentions,
        )

    def get_embedding_manager(self, **kwargs):
        if self.config.mask_matrix is None:
            raise ValueError(
                "config.mask_matrix is None. Please provide a valid mask matrix file."
            )

        return ChromBERTEmbedding(
            self.chrombert,
            self.config.mask_matrix,
            **kwargs,
        )

    