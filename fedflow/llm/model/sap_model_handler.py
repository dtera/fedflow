# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
from dataclasses import dataclass

from fedflow.llm.arguments import FedArguments
from fedflow.register import args
from fedflow.util.model_utils import BaseModelTokenizerHandler


@dataclass
class SAPModelTokenizerHandler(BaseModelTokenizerHandler):
    from fedflow.llm.sap.modeling_llama_sap import LlamaForCausalLM
    model_cls = LlamaForCausalLM
    fed_args: FedArguments = lambda: args["fed_args"]

    def model_post_init(self, model):
        # only lora M is trainable
        for n, p in model.named_parameters():
            if "lora_mobile" not in n:
                p.requires_grad = False
        return super().model_post_init(self, model)

    @classmethod
    def get_base_model(cls, state_dict=None, **kwargs):
        return super().get_base_model(state_dict=state_dict, **kwargs)
