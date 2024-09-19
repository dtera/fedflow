# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
from dataclasses import dataclass

from peft import get_peft_model

from fedflow.llm.arguments import FedArguments
from fedflow.register import args, register_common
from fedflow.util import ClientChannel, ServerChannel
from fedflow.util.model_utils import BaseModelTokenizerHandler


@dataclass
class SAPModelTokenizerHandler(BaseModelTokenizerHandler):
    fed_args: FedArguments = lambda: args["fed_args"]

    @classmethod
    def model_post_init(cls):
        if cls.fed_args().is_vender():
            sock_channel = ServerChannel(ip="0.0.0.0", port=cls.fed_args().comm_port)
            register_common("sock_channel", sock_channel)

            cls.model = get_peft_model(cls.model, cls.config_args().peft_lora_config)
            print(cls.model)
            cls.model.print_trainable_parameters()
            return cls.model
        return super().model_post_init()

    @classmethod
    def get_base_model(cls, state_dict=None, **kwargs):
        if not cls.fed_args().is_vender():
            sock_channel = ClientChannel(cls.fed_args().part_id, cls.fed_args().comm_addr)
            register_common("sock_channel", sock_channel)

            from fedflow.llm.fed_model import FedPreTrainedModelForCustomer
            from fedflow.util.config_utils import FedPretrainedConfig
            config = FedPretrainedConfig(cls.config_args())
            return FedPreTrainedModelForCustomer(config, cls.tokenizer)
        return super().get_base_model(state_dict=state_dict, **kwargs)
