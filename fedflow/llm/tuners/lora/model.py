# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
import torch.nn
from peft import LoraModel, LoraConfig
from peft.utils import _get_submodules

from fedflow.llm.tuners.lora import dispatchers
from fedflow.register import args, send_queue


class FedLoraModel(LoraModel):
    """
        Creates Federated Low Rank Adapter (LoRA) model from a pretrained transformers model.

        The method is described in detail in https://arxiv.org/abs/2106.09685.

        Args:
            model ([`torch.nn.Module`]): The model to be adapted.
            config ([`FedLoraConfig`]): The configuration of the Federated Lora model.
            adapter_name (`str`): The name of the adapter, defaults to `"default"`.

        Returns:
            `torch.nn.Module`: The Federated Lora model.
    """

    def __init__(self, model, config: LoraConfig, adapter_name) -> None:
        super().__init__(model, config, adapter_name)
        self.output_layer_name = args["lora_config_args"].output_layer_name
        del_module_keys = [key for key, module in model.named_modules() if
                           isinstance(module, torch.nn.Embedding) or key.split(".")[-1] in ["lora_embedding_A",
                                                                                            "lora_embedding_B",
                                                                                            self.output_layer_name]]
        for del_key in del_module_keys:
            parent, target, target_name = _get_submodules(model, del_key)
            if target_name == self.output_layer_name:
                send_queue.put(target.weight)
            delattr(parent, target_name)

    def _create_and_replace(
            self,
            lora_config,
            adapter_name,
            target,
            target_name,
            parent,
            current_key,
    ):
        FedLoraModel.target_name = target_name
        super()._create_and_replace(lora_config, adapter_name, target, target_name, parent, current_key)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        kwargs.update({"target_name": FedLoraModel.target_name})
        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module
