# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
from dataclasses import asdict

from transformers import TrainingArguments
from transformers.integrations import TensorBoardCallback


def tb_add_text(trainer, model):
    from fedflow.llm.arguments import FedLoraConfig
    from fedflow.register import args

    training_args: TrainingArguments = args["training_args"]
    lora_config_args: FedLoraConfig = args["lora_config_args"]
    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, TensorBoardCallback):
            if training_args.local_rank == 0:
                if cb.tb_writer is None:
                    cb._init_summary_writer(args=training_args)
                cb.tb_writer.add_text(
                    tag="exp_param", text_string=str(asdict(lora_config_args))
                )
                try:
                    cb.tb_writer.add_text(
                        tag="lora_config",
                        text_string=str(asdict(lora_config)),
                    )
                except Exception:
                    pass
                try:
                    private_lora_module = model.model.layers[0].self_attn.q_lora
                    cb.tb_writer.add_text(
                        tag="sap_lora",
                        text_string=str(private_lora_module),
                    )
                except Exception:
                    pass
