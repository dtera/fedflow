# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.

from peft import LoraModel, LoraConfig

from fedflow.llm.tuners.lora import dispatchers


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

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
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
