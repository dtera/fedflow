# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import torch
from peft import LoraConfig
from peft.tuners.lora.layer import Linear
from peft.tuners.tuners_utils import BaseTunerLayer
from transformers.pytorch_utils import Conv1D

from fedflow.register import commons
from fedflow.util import ServerChannel, ClientChannel


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class FedLinear(Linear):
    # Lora implemented in a dense layer
    def __init__(self, base_layer, adapter_name: str, r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0.0,
                 fan_in_fan_out: bool = False, is_target_conv_1d_layer: bool = False,
                 init_lora_weights: Union[bool, str] = True, use_rslora: bool = False, use_dora: bool = False,
                 **kwargs) -> None:
        self.target_name = kwargs.pop("target_name")
        super().__init__(base_layer, adapter_name, r, lora_alpha, lora_dropout, fan_in_fan_out, is_target_conv_1d_layer,
                         init_lora_weights, use_rslora, use_dora, **kwargs)
        self.sock_channel: ServerChannel = commons["sock_channel"]

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)

            if not self.use_dora[active_adapter]:
                result = result + self._apply_fed_lora(dropout(x), lora_A, lora_B, scaling)
            else:
                x = dropout(x)
                result = result + self._apply_dora(x, lora_A, lora_B, scaling, active_adapter)
        return result.to(torch_result_dtype)

    def _apply_fed_lora(self, x, lora_A, lora_B, scaling):
        h = lora_A(x)
        # send compressed activations to customer
        self.sock_channel.send_tensor(h)
        # receive comparessed and transformed activations from customer
        m_h = self.sock_channel.recv_tensor().to(x.device).to(x.dtype)
        return lora_B(m_h) * scaling

    def _apply_dora(self, x, lora_A, lora_B, scaling, active_adapter):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        # TODO
        return _apply_fed_lora(x, lora_A, lora_B, scaling)


class FedLoraMLayer(torch.nn.Module):
    def __init__(self, r_v2c: int, r_c2v: int, target_modules: list[str], **kwargs) -> None:
        """ M matrix
        1. receive compressed activations from vender as input
        2. transform on activations
        3. send back activations

        Args:
            r_v2c (int): dimension of vender 2 customer
            r_c2v (int): dimension of customer 2 vender
            target_modules (list[str]): target modules
        """
        super().__init__(**kwargs)
        self.target_modules = target_modules
        # q,k,v lora stacked together
        self.lora_M = torch.nn.ParameterDict()
        for target_module in self.target_modules:
            self.lora_M[target_module] = torch.nn.Parameter(
                torch.zeros((r_v2c, r_c2v)), requires_grad=False
            )
        self.sock_channel: ClientChannel = commons["sock_channel"]

    def forward(self):
        for target_module in self.target_modules:
            # receive compressed activations
            x = self.sock_channel.recv_tensor().to(self.lora_M[target_module].device).to(
                self.lora_M[target_module].dtype)
            # transform them
            x = x @ self.lora_M[target_module]
            # send them back to cloud
            self.sock_channel.send_tensor(x)


class FedLoraMStack(torch.nn.Module):
    def __init__(self, r_v2c: int, r_c2v: int, target_modules: list[str], num_layers_to_transform: int,
                 **kwargs) -> None:
        """Stack of M

        Args:
            r_v2c (int): dimension of vender 2 customer
            r_c2v (int): dimension of customer 2 vender
            target_modules (list[str]): target modules
            num_layers_to_transform (int): number of m does not necessarily equal to number of decoder layers.
        """
        super().__init__(**kwargs)
        self.layers = torch.nn.ModuleList(
            [FedLoraMLayer(r_v2c, r_c2v, target_modules) for _ in range(num_layers_to_transform)]
        )

    def forward(self):
        for i, layer in enumerate(self.layers):
            # print(f"{i} th layer")
            layer()


def dispatch_default(
        target: torch.nn.Module,
        adapter_name: str,
        lora_config: LoraConfig,
        **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(lora_config.loftq_config)
        # TODO: create a new module for Embedding Layer
        new_module = None
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        kwargs.update(lora_config.loftq_config)
        # TODO: create a new module for Conv2d Layer
        new_module = None
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        # create a new module for Linear Layer
        new_module = FedLinear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        # create a new module for Conv1D Layer
        new_module = FedLinear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module


from inspect import isfunction

dispatchers = []
for k, v in [(k, v) for k, v in globals().items() if isfunction(v) and k.startswith("dispatch_")]:
    dispatchers.append(v)
