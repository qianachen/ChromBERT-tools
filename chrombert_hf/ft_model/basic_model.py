from __future__ import annotations

import os
from abc import ABC, abstractmethod
from copy import deepcopy

import torch
from torch import nn
from transformers import AutoModel

from chrombert_hf.pretrain_model import ChromBERTEmbedding
from chrombert_hf.pretrain_model import ChromBERTModel
from chrombert_hf.pretrain_model import ChromBERTConfig

def _clone_config(config):
    if hasattr(config, "clone"):
        return config.clone()
    return deepcopy(config)


def _update_config(config, **kwargs):
    if hasattr(config, "update"):
        config.update(**kwargs)
        return config
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


class BasicModel(nn.Module, ABC):
    """
    Fine-tuning base class backed by a Hugging Face ChromBERT model.
    """

    def __init__(
        self,
        finetune_config,
        pretrained_model_name_or_path=None,
        pretrain_model=None,
        trust_remote_code: bool = True,
        pretrained_model_kwargs: dict | None = None,
    ):
        super().__init__()
        if pretrained_model_name_or_path is not None and pretrain_model is not None:
            raise ValueError("Only one of pretrained_model_name_or_path or pretrain_model can be provided")
        if pretrain_model is not None and not isinstance(pretrain_model, nn.Module):
            raise TypeError("pretrain_model must be an nn.Module")
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.pretrain_model = pretrain_model
        self.finetune_config = finetune_config
        self.trust_remote_code = trust_remote_code
        self.pretrained_model_kwargs = pretrained_model_kwargs or {}
        self.create_layers()

    def _load_pretrain_from_path(self, path):
        """Load ChromBERT pretrain from a local dir or Hub id. Same strategy as a single path."""
        try:
            return AutoModel.from_pretrained(
                path,
                trust_remote_code=self.trust_remote_code,
                **self.pretrained_model_kwargs,
            )
        except (KeyError, ValueError):
            # Fallback for local/custom ChromBERT exports without auto_map.
            return ChromBERTModel.from_pretrained(
                path,
                **self.pretrained_model_kwargs,
            )
    def init_pretrain_model(self):
        if self.pretrain_model is not None:
            pretrain_model = self.pretrain_model
        elif getattr(self.finetune_config, "pretrain_ckpt", None) is not None and os.path.exists(getattr(self.finetune_config, "pretrain_ckpt", None)) and os.path.exists(self.finetune_config.mtx_mask):
            pretrain_config = ChromBERTConfig(genome=self.finetune_config.genome,
                                              dropout=self.finetune_config.dropout,
                                            mtx_mask=self.finetune_config.mtx_mask)
            pretrain_model = pretrain_config.init_model(
                getattr(self.finetune_config, "pretrain_ckpt", None),
            )
        else:
            primary = self.pretrained_model_name_or_path
            raw_pretrain_ckpt = getattr(self.finetune_config, "pretrain_ckpt", None)

            last_error = None
            pretrain_model = None
            # 1) try HF / local export directory
            if primary:
                try:
                    pretrain_model = self._load_pretrain_from_path(primary)
                    self.pretrained_model_name_or_path = primary
                except Exception as err:
                    last_error = err

            if pretrain_model is None:
                raise RuntimeError(
                    f"Failed to load pretrain model from primary={primary!r}, raw_pretrain_ckpt={raw_pretrain_ckpt!r}"
                ) from last_error

        if getattr(self.finetune_config, "mtx_mask", None) is None:
            self.finetune_config.mtx_mask = getattr(pretrain_model.config, "mask_matrix", None)
        elif getattr(pretrain_model.config, "mask_matrix", None) is None:
            pretrain_model.config.mask_matrix = self.finetune_config.mtx_mask

        self.pretrain_model = pretrain_model
        return pretrain_model
    

    @abstractmethod
    def create_layers(self):
        raise NotImplementedError

    def load_ckpt(self, ckpt=None):
        if ckpt is not None:
            assert os.path.exists(ckpt), f"Checkpoint file does not exist: {ckpt}"
        else:
            ckpt = getattr(self.finetune_config, "finetune_ckpt", None)
            if ckpt is None:
                raise ValueError("finetune checkpoint is not specified")
            assert os.path.exists(ckpt), f"Checkpoint file does not exist: {ckpt}"

        print(f"Loading checkpoint from {ckpt}")
        old_state = self.state_dict()
        new_state = torch.load(ckpt, map_location="cpu")

        if "state_dict" in new_state:
            new_state = new_state["state_dict"]

        num = len([key for key in new_state.keys() if key.startswith("model.")])
        if new_state and num / len(new_state) > 0.9:
            new_state = {k[6:]: v for k, v in new_state.items() if k.startswith("model.")}
            print("Loading from pl module, remove prefix 'model.'")
            remapped = {}
            for k, v in new_state.items():
                # old Lightning： pretrain_model.embedding / transformer_blocks under chrombert
                if k.startswith("pretrain_model.embedding.") or k.startswith("pretrain_model.transformer_blocks."):
                    new_k = "pretrain_model.chrombert." + k[len("pretrain_model."):]
                    remapped[new_k] = v
                elif k.startswith("pool_flank_window.pretrain_model.embedding.") or k.startswith("pool_flank_window.pretrain_model.transformer_blocks."):
                    new_k = "pool_flank_window.pretrain_model.chrombert." + k[len("pool_flank_window.pretrain_model."):]
                    remapped[new_k] = v
                else:
                    remapped[k] = v
            new_state = remapped
            print("Loading from pl module, replace 'pretrain_model' with 'pretrain_model.chrombert'")

            
        total_keys = len(new_state)
        new_state = {k: v for k, v in new_state.items() if k in old_state}
        print(f"Loaded {len(new_state)}/{total_keys} parameters")
        old_state.update(new_state)
        self.load_state_dict(old_state)
        return None

    def display_trainable_parameters(self, verbose=True):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        output = {"total_params": total_params, "trainable_params": trainable_params}
        print(output)
        if verbose:
            for name, parameter in self.named_parameters():
                print(name, ": trainable" if parameter.requires_grad else ": frozen")
        return output

    def get_pretrain(self):
        if hasattr(self, "pretrain_model"):
            return self.pretrain_model
        if getattr(self.finetune_config, "task", None) == "gep":
            return self.pool_flank_window.pretrain_model
        raise ValueError("pretrain_model is not specified")

    def freeze_pretrain(self, trainable=2):
        pretrain_model = self.get_pretrain()
        if hasattr(pretrain_model, "chrombert") and hasattr(pretrain_model.chrombert, "freeze"):
            pretrain_model.chrombert.freeze(trainable)
        elif hasattr(pretrain_model, "freeze"):
            pretrain_model.freeze(trainable)
        else:
            raise AttributeError("pretrain_model does not support freeze(trainable)")
        return self

    def save_pretrain(self, save_path):
        pretrain_model = self.get_pretrain()
        if hasattr(pretrain_model, "save_pretrained"):
            pretrain_model.save_pretrained(save_path)
            return save_path
        state_dict = pretrain_model.state_dict()
        torch.save(state_dict, save_path)
        return state_dict

    def get_embedding_manager(self, **kwargs):
        pretrain_model = self.get_pretrain()
        finetune_config = _clone_config(self.finetune_config)
        _update_config(finetune_config, **kwargs)
        model_emb = ChromBERTEmbedding(
            pretrain_model,
            finetune_config.mtx_mask,
            getattr(finetune_config, "ignore", False),
            getattr(finetune_config, "ignore_index", None),
        )
        return model_emb

    def save_ckpt(self, save_path):
        torch.save(self.state_dict(), save_path)
        return None
