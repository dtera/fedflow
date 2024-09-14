# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
import enum

from peft import PeftType, TaskType

from fedflow.util import extend_enum


@extend_enum(PeftType)
class FedPeftType(str, enum.Enum):
    FED_LORA = "FED_LORA"


@extend_enum(TaskType)
class FedTaskType(str, enum.Enum):
    FED_CAUSAL_LM = "FED_CAUSAL_LM"
