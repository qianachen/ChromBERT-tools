from __future__ import annotations


from .basic_model import BasicModel, _clone_config, _update_config
from chrombert_hf.pretrain_model import ChromBERTEmbedding
from .ft_utils import GepHeader, GeneralHeader, PoolFlankWindow



class ChromBERTGEP(BasicModel):
    """
    Fine-tune a Hugging Face ChromBERT checkpoint for GEP tasks.
    """

    def create_layers(self):
        pretrain_model = self.init_pretrain_model()
        self.flank_region_num = int(self.finetune_config.gep_flank_window) * 2 + 1
        self.pool_flank_window = PoolFlankWindow(
            flank_region_num=self.flank_region_num,
            pretrain_model=pretrain_model,
            parallel_embedding=getattr(self.finetune_config, "gep_parallel_embedding", False),
            gradient_checkpoint=getattr(self.finetune_config, "gep_gradient_checkpoint", False),
        )
        # init_pretrain_model() registers the same module as self.pretrain_model; PoolFlankWindow
        # already owns it. Drop the root registration so state_dict has a single backbone prefix.
        self._modules.pop("pretrain_model", None)

        header_cls = GepHeader if getattr(self.finetune_config, "gep_zero_inflation", False) else GeneralHeader
        self.ft_header = header_cls(
            pretrain_model.config.hidden_dim,
            self.finetune_config.dim_output,
            self.finetune_config.mtx_mask,
            getattr(self.finetune_config, "ignore", False),
            getattr(self.finetune_config, "ignore_index", None),
            getattr(self.finetune_config, "dropout", 0.1),
        )
        return None

    def forward(self, batch):
        input_ids = batch["input_ids"]
        position_ids = batch["position_ids"]
        pool_out = self.pool_flank_window(input_ids, position_ids)
        header_out = self.ft_header(pool_out)
        return header_out

    def get_embedding_manager(self, **kwargs):
        pretrain_model = self.get_pretrain()
        finetune_config = _clone_config(self.finetune_config)
        _update_config(finetune_config, **kwargs)
        pool_model = PoolFlankWindow(
            flank_region_num=int(finetune_config.gep_flank_window) * 2 + 1,
            pretrain_model=pretrain_model,
            parallel_embedding=getattr(finetune_config, "gep_parallel_embedding", False),
            gradient_checkpoint=getattr(finetune_config, "gep_gradient_checkpoint", False),
        )
        return ChromBERTEmbedding(
            pool_model,
            finetune_config.mtx_mask,
            getattr(finetune_config, "ignore", False),
            getattr(finetune_config, "ignore_index", None),
        )
