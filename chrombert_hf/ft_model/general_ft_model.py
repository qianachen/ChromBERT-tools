from __future__ import annotations


from .basic_model import BasicModel
from .ft_utils import GeneralHeader, extract_last_hidden_state



class ChromBERTGeneral(BasicModel):
    """
    Fine-tune a Hugging Face ChromBERT checkpoint for general tasks.
    """

    def create_layers(self):
        self.pretrain_model = self.init_pretrain_model()

        self.ft_header = GeneralHeader(
            self.pretrain_model.config.hidden_dim,
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
        chrombert_out = self.pretrain_model(
            input_ids=input_ids.long(),
            position_ids=position_ids.long(),
            return_dict=True,
        )
        chrombert_out = extract_last_hidden_state(chrombert_out)
        header_out = self.ft_header(chrombert_out)
        return header_out
