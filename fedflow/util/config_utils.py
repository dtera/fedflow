# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
from transformers import PretrainedConfig

from fedflow.llm import FedLoraConfig


class FedPretrainedConfig(PretrainedConfig):
    def __init__(self, lora_config_args: FedLoraConfig, **kwargs):
        self.lora_config_args = lora_config_args
        self.peft_lora_config = lora_config_args.peft_lora_config
        super().__init__(**kwargs)
