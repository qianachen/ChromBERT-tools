from __future__ import annotations

import os
import torch
from transformers import PretrainedConfig


class ChromBERTConfig(PretrainedConfig):
    model_type = "chrombert"

    def __init__(
        self,
        genome: str = "hg38",
        dropout: float = 0.1,
        dtype_str: str = "bfloat16",
        ckpt: str | None = None,
        hidden_dim: int = 768,
        num_layers: int = 8,
        feed_forward_dim: int = 3072,
        num_attention_heads: int = 8,
        vocab_size: int = 10,
        vocab_size_shift: int = 5,
        token_id_pad: int = 0,
        pe_mode: str = "train",
        flash_bias: bool = True,
        flash_batch_first: bool = True,
        flash_causal: bool = False,
        flash_device: str | None = None,
        mask_matrix: str | None = "hg38_6k_mask_matrix.tsv",
        lite: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.genome = genome
        self.dropout = dropout
        self.dtype_str = dtype_str
        self.ckpt = ckpt

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.feed_forward_dim = feed_forward_dim
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.vocab_size_shift = vocab_size_shift
        self.token_id_pad = token_id_pad
        self.pe_mode = pe_mode

        self.flash_bias = flash_bias
        self.flash_batch_first = flash_batch_first
        self.flash_causal = flash_causal
        self.flash_device = flash_device

        self.mask_matrix = mask_matrix

        self.hidden_size = hidden_dim
        self.num_hidden_layers = num_layers
        self.attention_probs_dropout_prob = dropout
        self.intermediate_size = feed_forward_dim
        self.pad_token_id = token_id_pad
        self.lite = lite
        if self.lite:
            self.num_layers = 4
            self.num_hidden_layers = 4
            

        if self.genome not in ["hg38", "mm10"]:
            raise ValueError(f"genome should be hg38 or mm10, but got {self.genome}")

    @property
    def n_datasets(self) -> int:
        if self.genome == "hg38" and not self.lite:
            return 6392
        if self.genome == "hg38" and self.lite:
            return 3884
        if self.genome == "mm10":
            return 5616
        raise ValueError(f"Unsupported genome: {self.genome}")

    @property
    def dtype(self):
        try:
            return getattr(torch, self.dtype_str)
        except AttributeError as exc:
            raise ValueError(f"Unsupported dtype_str: {self.dtype_str}") from exc
    
    def init_model(self, ckpt=None):
        '''
        Instantiate the model using the configuration.
        '''
        from .model_chrombert import ChromBERTModel
        model = ChromBERTModel(self)
        if ckpt is None:
            ckpt = self.ckpt
        if ckpt is None:
            print(f"Warning: no ckpt provided, use random initialization!")
        elif os.path.exists(ckpt):
            model.chrombert.load_ckpt(ckpt)
            print(f"Load pretrained ckpt {ckpt} successfully!")
        else:
            print(f"Warning: ckpt {ckpt} not exists, use random initialization!")
        return model