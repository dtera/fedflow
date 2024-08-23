# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
from dataclasses import dataclass

from fedflow.llm.arguments import ModelArguments
from fedflow.util.model_utils import BaseModelTokenizerHandler


@dataclass
class SAPModelTokenizerHandler(BaseModelTokenizerHandler):
    from fedflow.llm.sap.modeling_llama_sap import LlamaForCausalLM
    model_cls = LlamaForCausalLM

    def model_post_init(self, model, model_args: ModelArguments):
        # only lora M is trainable
        for n, p in model.named_parameters():
            if "lora_mobile" not in n:
                p.requires_grad = False
        return super().model_post_init(self, model, model_args)
