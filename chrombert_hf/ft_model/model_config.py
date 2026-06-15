from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from chrombert_hf.pretrain_model import ChromBERTConfig

@dataclass
class ChromBERTFTConfig:
    genome: str = field(default="hg38", metadata={"help": "hg38 for human, and mm10 for mouse"})
    task: str = field(default="general", metadata={"help": "task of the model"})
    dim_output: int = field(default=1, metadata={"help": "dimension of output"})

    # HF-compatible pretrain source: local model dir or Hub repo id
    pretrained_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging Face model directory or repo id"},
    )
    pretrain_model: Any = field(
        default=None,
        repr=False,
        compare=False,
        metadata={"help": "Runtime preloaded pretrain model", "serialize": False},
    )
    trust_remote_code: bool = field(default=True)
    pretrained_model_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Local/backup pretrain path: used when pretrained_model_name_or_path is unset,
    # or as fallback when primary load (e.g. Hub) fails (see BasicModel.init_pretrain_model).
    pretrain_ckpt: Optional[str] = field(default=None, metadata={"help": "pretrained model path"})

    lite: bool = field(default=False, metadata={"help": "whether to use lite model"})

    mtx_mask: Optional[str] = field(default=None, metadata={"help": "mask matrix path; optional if bundled in HF model"})
    dropout: float = field(default=0.1, metadata={"help": "dropout rate"})
    finetune_ckpt: Optional[str] = field(default=None, metadata={"help": "loading finetune checkpoint"})

    ignore: bool = False
    ignore_index: Tuple[Optional[List], Optional[List]] = field(
        default=(None, None),
        metadata={"help": "ignore index for regulators and gsmids"},
    )

    gep_flank_window: int = field(default=4, metadata={"help": "the number of flank regions"})
    gep_parallel_embedding: bool = field(default=False, metadata={"help": "whether to use parallel embedding"})
    gep_gradient_checkpoint: bool = field(default=False, metadata={"help": "whether to use gradient checkpoint"})
    gep_zero_inflation: bool = field(default=False, metadata={"help": "whether to use zero inflation header"})

    prompt_kind: str = field(default="cistrome", metadata={"help": "prompt data class"})
    prompt_dim_external: int = field(default=512, metadata={"help": "dimension of external data"})
    dnabert2_ckpt: Optional[str] = field(default=None, metadata={"help": "loading dnabert2 checkpoint"})

    def __post_init__(self):
        self.validation()

        tmp = self.ignore_index
        if tmp is None:
            tmp = (None, None)

        tmp0 = tmp[0].tolist() if isinstance(tmp[0], np.ndarray) else tmp[0]
        tmp1 = tmp[1].tolist() if isinstance(tmp[1], np.ndarray) else tmp[1]
        self.ignore_index = (tmp0, tmp1)

        # if self.pretrained_model_name_or_path is None and self.pretrain_ckpt is not None:
        #     self.pretrained_model_name_or_path = self.pretrain_ckpt

    @property
    def pretrain_source(self) -> Optional[str]:
        return self.pretrained_model_name_or_path or self.pretrain_ckpt

    def to_dict(self):
        state = {}
        for key, field_info in self.__dataclass_fields__.items():
            if field_info.metadata.get("serialize", True) is False:
                continue
            state[key] = deepcopy(getattr(self, key))
        return state

    def __iter__(self):
        for name, value in self.to_dict().items():
            yield name, value

    @classmethod
    def load(
        cls,
        config: Union[str, Dict[str, Any], "ChromBERTFTConfig", None] = None,
        **kwargs: Any,
    ):
        if config is None:
            config_dict = {}
        elif isinstance(config, str):
            with open(config, "r") as f:
                config_dict = json.load(f)
        elif isinstance(config, Dict):
            config_dict = deepcopy(config)
        elif isinstance(config, ChromBERTFTConfig):
            config_dict = config.to_dict()
        else:
            raise TypeError(f"config must be a str, Dict, or ChromBERTFTConfig, but got {type(config)}")

        config_dict.update(kwargs)
        loaded = cls(**config_dict)
        loaded.validation()
        return loaded

    def clone(self):
        cloned = ChromBERTFTConfig.load(self.to_dict())
        cloned.pretrain_model = self.pretrain_model
        return cloned

    def validation(self):
        assert self.genome in ["hg38", "mm10"], f"genome must be one of ['hg38', 'mm10'], but got {self.genome}"
        task_available = ["general", "gep", "prompt"]
        assert self.task in task_available, f"task must be one of {task_available}, but got {self.task}"
        prompt_kind_available = ["cistrome", "expression", "dna", "sequence", "cctp_sequence"]
        assert self.prompt_kind in prompt_kind_available, f"header must be one of {prompt_kind_available}, but got {self.prompt_kind}"
        return None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Warning: '{key}' is not a valid field name in ChromBERTFTConfig")
        if self.pretrained_model_name_or_path is None and self.pretrain_ckpt is not None:
            self.pretrained_model_name_or_path = self.pretrain_ckpt
        return None

    @property
    def n_prompt_parts(self):
        if self.prompt_kind in ["cistrome", "expression"]:
            return 2
        if self.prompt_kind == "dna":
            return 1
        raise ValueError(f"prompt_kind must be one of ['cistrome', 'expression', 'dna'], but got {self.prompt_kind}")

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4)

    def __repr__(self):
        return f"ChromBERTFTConfig:\n{str(self)}"

    def init_model(self, **kwargs):
        finetune_config = self.clone()
        finetune_config.update(**kwargs)

        pretrain_model = finetune_config.pretrain_model
        pretrain_source = None if pretrain_model is not None else finetune_config.pretrain_source
        if pretrain_source is None and pretrain_model is None:
            print("Warning: neither pretrained_model_name_or_path nor pretrain_model is specified in fine-tune model initiation")

        if finetune_config.task == "gep":
            from .gep_ft_model import ChromBERTGEP

            model = ChromBERTGEP(
                finetune_config=finetune_config,
                pretrained_model_name_or_path=pretrain_source,
                pretrain_model=pretrain_model,
                trust_remote_code=finetune_config.trust_remote_code,
                pretrained_model_kwargs=finetune_config.pretrained_model_kwargs,
            )
        elif finetune_config.task == "prompt":
            from .prompt_ft_model import ChromBERTPrompt

            model = ChromBERTPrompt(
                finetune_config=finetune_config,
                pretrained_model_name_or_path=pretrain_source,
                pretrain_model=pretrain_model,
                trust_remote_code=finetune_config.trust_remote_code,
                pretrained_model_kwargs=finetune_config.pretrained_model_kwargs,
            )
        else:
            from .general_ft_model import ChromBERTGeneral

            model = ChromBERTGeneral(
                finetune_config=finetune_config,
                pretrained_model_name_or_path=pretrain_source,
                pretrain_model=pretrain_model,
                trust_remote_code=finetune_config.trust_remote_code,
                pretrained_model_kwargs=finetune_config.pretrained_model_kwargs,
            )

        if finetune_config.finetune_ckpt is not None:
            model.load_ckpt(finetune_config.finetune_ckpt)
        return model

    @classmethod
    def get_ckpt_type(cls, ckpt):
        raise NotImplementedError("HF-compatible config no longer infers raw ckpt type from pretrain checkpoints")


def get_preset_model_config(
    preset: str = "default",
    basedir: str = os.path.expanduser("~/.cache/chrombert/data"),
    **kwargs,
):
    basedir = os.path.abspath(basedir)
    assert os.path.exists(basedir), f"{basedir=} does not exist"
    if not os.path.exists(preset):
        list_presets_available = os.listdir(os.path.join(os.path.dirname(__file__), "presets"))
        list_presets_available = [x.split(".")[0] for x in list_presets_available]

        if preset not in list_presets_available:
            raise ValueError(f"preset must be one of {list_presets_available}, but got {preset}")

        preset_file = os.path.join(os.path.dirname(__file__), "presets", f"{preset}.json")
    else:
        preset_file = preset

    with open(preset_file, "r") as f:
        config = json.load(f)
    config.update(kwargs)

    for key, value in config.items():
        if key in ["mtx_mask", "finetune_ckpt"]:
            print(f"update path: {key} = {value}")
            if value is not None and os.path.abspath(value) != value:
                config[key] = os.path.join(basedir, value)
                assert os.path.exists(config[key]), f"{key}={config[key]} does not exist"

    return ChromBERTFTConfig(**config)
