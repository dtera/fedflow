# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
import enum

from peft import PeftType, TaskType

from fedflow.util.utils import extend_enum


@extend_enum(PeftType, is_raw_enum=True)
class FedPeftType(str, enum.Enum):
    FED_LORA = "FED_LORA"


@extend_enum(TaskType, is_raw_enum=True)
class FedTaskType(str, enum.Enum):
    FED_CAUSAL_LM = "FED_CAUSAL_LM"
