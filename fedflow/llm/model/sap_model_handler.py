# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
from dataclasses import dataclass

from peft import get_peft_model

from fedflow.llm.arguments import FedArguments
from fedflow.register import args, register_common
from fedflow.util import init_tcp_server, BaseModelTokenizerHandler


@dataclass
class SAPModelTokenizerHandler(BaseModelTokenizerHandler):
    fed_args: FedArguments = lambda: args["fed_args"]

    @classmethod
    def model_post_init(cls, model):
        if cls.fed_args().is_vender():
            model = get_peft_model(model, cls.config_args().peft_lora_config)
            print(model)
            model.print_trainable_parameters()
            socket_ = init_tcp_server(ip="0.0.0.0", port=cls.fed_args().comm_port)
            register_common("socket", socket_)
            return model
        else:
            # only lora M is trainable
            for n, p in model.named_parameters():
                if "lora_mobile" not in n:
                    p.requires_grad = False
            return super().model_post_init(model)

    @classmethod
    def get_base_model(cls, state_dict=None, **kwargs):
        if not cls.fed_args().is_vender():
            return super().get_base_model(state_dict=state_dict, **kwargs)
        return super().get_base_model(state_dict=state_dict, **kwargs)
