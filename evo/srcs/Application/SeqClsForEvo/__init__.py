import torch
from torch import nn
from ...BaseModel.StripedHyena import StripedHyena
from ...BaseModel import StripedHyenaPreTrainedModel
from ....util.utils import dotdict
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import logging
from typing import Optional, Tuple, Union
import torch.nn.init as init
import torch.nn.functional as F

logger = logging.get_logger(__name__)




class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.inv_sqrt_dim = 1.0 / (hidden_size ** 0.5)

    def forward(self, x):
        norm_x = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed
    


class SeqClsForEvo(StripedHyenaPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        model_config = dotdict(config.to_dict())
        self.backbone = StripedHyena(model_config)
        self.config = config
        self.backbone.gradient_checkpointing = False

        self.num_labels = config.num_labels

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            # RMSNorm(512),
            # nn.GELU(),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, config.num_labels)
        )

        self.classifier.apply(self._init_weights)

        self.post_init()
        self.force_dtype()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    def force_dtype(self):
        self.backbone.to_bfloat16_except_poles_residues()
        self.classifier.to(torch.float32)

    def get_input_embeddings(self):
        return self.backbone.embedding_layer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        past_key_values=None,
        return_dict: bool = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        logits, past_key_values = self.backbone(
            input_ids,
            padding_mask=attention_mask,
            inference_params_dict=past_key_values if use_cache else None,
        )

        masked_logits = logits * attention_mask.unsqueeze(-1)
        pooled = masked_logits.sum(dim=1) / (attention_mask.sum(dim=1, keepdim=True) + 1e-8)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            return (loss, logits) if loss is not None else logits

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    @classmethod
    def can_generate(cls) -> bool:
        return False
