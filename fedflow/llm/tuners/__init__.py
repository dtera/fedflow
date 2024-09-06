# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
from peft import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING, LoraConfig
from peft.mapping import PEFT_TYPE_TO_TUNER_MAPPING
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING

from fedflow.llm.fed_model import FedPeftModelForCausalLM
from fedflow.llm.tuners.fed_peft_types import FedPeftType, FedTaskType
from fedflow.llm.tuners.lora import (
    FedLoraModel, FedLinear
)

PEFT_TYPE_TO_MODEL_MAPPING.__setitem__(FedPeftType.FED_LORA, FedLoraModel)
PEFT_TYPE_TO_TUNER_MAPPING.__setitem__(FedPeftType.FED_LORA, FedLoraModel)
PEFT_TYPE_TO_CONFIG_MAPPING.__setitem__(FedPeftType.FED_LORA, LoraConfig)
MODEL_TYPE_TO_PEFT_MODEL_MAPPING.__setitem__(FedTaskType.FED_CAUSAL_LM, FedPeftModelForCausalLM)
