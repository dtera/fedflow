# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
import logging
import socket
from dataclasses import dataclass, asdict

import torch
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments

from fedflow.llm.arguments import ModelArguments, FedLoraConfig
from fedflow.register import models, args
from fedflow.util import send_tensor, recv_tensor, CommProfiler

DEFAULT_MODEL_INIT_KWARGS = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}


@dataclass
class BaseModelTokenizerHandler:
    model_cls = AutoModelForCausalLM
    model_args: ModelArguments = lambda: args["model_args"]

    @classmethod
    def _model_args(cls, update_args=None):
        args = DEFAULT_MODEL_INIT_KWARGS.copy()
        args.update((k, v) for k, v in asdict(cls.model_args()).items() if k in args)
        if update_args is not None:
            args.update(update_args)
        return args

    def model_post_init(self, model):
        print(model)
        return model

    def tokenizer_post_init(self, tokenizer):
        if self.model_args().pad_with_unk_token:
            tokenizer.pad_token = tokenizer.unk_token
        print(tokenizer)
        return tokenizer

    @classmethod
    def get_base_model(cls, state_dict=None, **kwargs):
        model = cls.model_cls.from_pretrained(cls.model_args().pretrained_model_name_or_path, **cls._model_args(),
                                              state_dict=state_dict, **kwargs)
        return cls.model_post_init(cls, model)

    @classmethod
    def get_base_tokenizer(cls, **kwargs):
        tokenizer_name_or_path = cls.model_args().tokenizer_name_or_path if cls.model_args().tokenizer_name_or_path else (
            cls.model_args().pretrained_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **cls._model_args())
        return cls.tokenizer_post_init(cls, tokenizer)

    @classmethod
    def get_base_model_and_tokenizer(cls, state_dict=None, **kwargs):
        return cls.get_base_model(state_dict=state_dict, **kwargs), cls.get_base_tokenizer(**kwargs)

    @classmethod
    def get_config(cls, **kwargs):
        return AutoConfig.from_pretrained(cls.model_args().pretrained_model_name_or_path, trust_remote_code=True)


def _get_model_handler(model_type: str):
    return models[model_type] if model_type in models else BaseModelTokenizerHandler


def get_base_model_and_tokenizer(**kwargs):
    """common model and tokenizer getter, so we don't have tweak in training scripts
    Returns:
        model, tokenizer
    Examples:
        ```python
        model, tokenizer = get_base_model_and_tokenizer("sap")
        ```
    """
    model_args: ModelArguments = args["model_args"]
    state_dict = None
    if model_args.weights_path is not None:
        state_dict = torch.load(model_args.weights_path)
    return _get_model_handler(model_args.model_type).get_base_model_and_tokenizer(state_dict, **kwargs)


def get_model_config(**kwargs):
    """model config getter
    Returns:
        model_config
    """
    return _get_model_handler(model_args.model_type).get_config(**kwargs)


def adapt_with_lora(model):
    lora_config_args: FedLoraConfig = args["lora_config_args"]
    if lora_config_args.use_lora:
        model = get_peft_model(model, LoraConfig(**lora_config_args.peft_lora_config))
    # set a,b trainable
    if lora_config_args.trainable_a:
        logging.info("Set lora a trainable")
        for n, p in model.named_parameters():
            if "lora_A" in n:
                p.requires_grad = True
    if lora_config_args.trainable_b:
        logging.info("Set lora b trainable")
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.requires_grad = True
    return model


def save_on_zero_3(trainer, model):
    training_args: TrainingArguments = args["training_args"]
    # check if zero3 mode enabled
    if training_args.hf_deepspeed_config.is_zero3():
        # use deepspeed engine internal function to gather state dict
        # state_dict_zero3 contains whole parameters of base and lora adapters
        # we will not extract lora parameters since peft save_pretrained will do that
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/peft_model.py#L125
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/utils/save_and_load.py#L19
        logging.info("start save state_dict_zero3")
        state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        if training_args.local_rank == 0:
            state_dict = state_dict_zero3
    else:
        state_dict = None
    if training_args.local_rank == 0:
        from peft import PeftModel

        if isinstance(model, PeftModel):
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            logging.info("save done")

        to_save = get_sap_dict(state_dict)
        if to_save is not None:
            import torch

            torch.save(to_save, f"{training_args.output_dir}/sap.bin")
            logging.info(f"save sap success to {training_args.output_dir}")


def get_sap_dict(state_dict):
    """return additional parameters (A,B,M) of private lora

    Args:
        state_dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    to_save = {}
    v3_flag = False
    for k, v in state_dict.items():
        if "lora_mobile" in k:
            v3_flag = True
        if "lora_" in k:
            to_save[k] = v
    return to_save if v3_flag else None


############################################PrivateLora############################################
class PLAB(torch.nn.Module):
    def __init__(self, in_features: int, rcd: int, rdc: int, out_features: int):
        """A,B matrices on Cloud, Q,K,V are stacked to execute in parallel

        Args:
            in_features (int):
            rcd (int):
            rdc (int):
            out_features (int):
        """
        super().__init__()
        self.lora_A = torch.nn.Parameter(
            torch.zeros((3, rcd, in_features)), requires_grad=False
        )  # 3 for q,k,v, stacked together for parallel execution
        self.lora_B = torch.nn.Parameter(
            torch.zeros((3, out_features, rdc)), requires_grad=False
        )

    def forward(self, x, s: socket.socket, buffer_size=int(2048e3)):
        x = x.to(self.lora_A.dtype).to(self.lora_A.device)
        # compress activations
        x = x @ self.lora_A.transpose(2, 1)
        # send compressed activations to edge device
        send_tensor(s, x)
        # receive comparessed and transformed activations from edge device
        x = recv_tensor(s, buffer_size).to(self.lora_B.device).to(self.lora_B.dtype)
        # de compress to hidden dimension
        x = x @ self.lora_B.transpose(2, 1)
        # chunk for q,k,v
        return x.chunk(3, 0)


class PLM(torch.nn.Module):
    def __init__(self, rcd: int, rdc: int, **kwargs) -> None:
        """ M matrix
        1. receive compressed activations from cloud as input
        2. transform on activations
        3. send back activations

        Args:
            rcd (int): dimension of cloud 2 device
            rdc (int): dimension of device 2 cloud
        """
        super().__init__(**kwargs)
        # q,k,v lora stacked together
        self.lora_M = torch.nn.Parameter(
            torch.zeros((3, rdc, rcd)), requires_grad=False
        )

    def forward(
            self, s: socket.socket, buffer_size=int(2048e3), profiler: CommProfiler = None
    ):
        """
        Args:
            s (socket.socket):
            buffer_size (int, optional): useless but i'm lazy. Defaults to int(2048e3).
            profiler (CommProfiler, optional): if not None profiler will do performance profile. Defaults to None.
        """
        # receive compressed activations
        x = recv_tensor(s, buffer_size).to(self.lora_M.device).to(self.lora_M.dtype)
        # transform them
        x = x @ self.lora_M.transpose(2, 1)
        # send them back to cloud
        send_tensor(s, x, profiler=profiler)


class PLMStack(torch.nn.Module):
    def __init__(self, num_hidden_layers: int, rcd: int, rdc: int, **kwargs) -> None:
        """Stack of M

        Args:
            num_hidden_layers (int): number of m does not necessarily equal to number of decoder layers.
            rcd (int): dimension of M matrix, indicates transmission base for cloud 2 device connection
            rdc (int): dimension of M matrix, indicates transmission base for device 2 cloud connection
        """
        super().__init__(**kwargs)
        self.layers = torch.nn.ModuleList(
            [PLM(rcd, rdc) for _ in range(num_hidden_layers)]
        )

    def forward(
            self, s: socket.socket, buffer_size=int(2048e3), profiler: CommProfiler = None
    ):
        """
        Args:
            s (socket.socket):
            buffer_size (int, optional): useless but i'm lazy. Defaults to int(2048e3).
            profiler (CommProfiler, optional): if not None profiler will do performance profile. Defaults to None.
        """
        for i, layer in enumerate(self.layers):
            # print(f"{i} th layer")
            layer(s, buffer_size, profiler=profiler)
############################################PrivateLora############################################
