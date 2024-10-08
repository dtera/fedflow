# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
import logging

import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset
from transformers import Trainer

from fedflow.llm import parse_args, fetch_args_from_dataclass, get_last_checkpoint_, FedTrainer
from fedflow.util import (
    prepare_dataset, add_eval_callback, tb_add_text
)
from fedflow.util.model_utils import (
    get_base_model_and_tokenizer, save_on_zero_3
)


def train(model_args, data_args, training_args, train_dataset, model, tokenizer):
    from fedflow.register import data_collators
    # Initialize Trainer
    trainer = (Trainer if model_args.is_baseline() else FedTrainer)(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collators[data_args.data_collator](tokenizer, **fetch_args_from_dataclass(
            data_collators[data_args.data_collator], data_args)) if data_args.data_collator in data_collators else None,
    )
    # Add some text records to tensorboard
    tb_add_text(trainer, model)
    # Add eval callback
    add_eval_callback(train_dataset, trainer, tokenizer)
    # Detecting last checkpoint.
    last_checkpoint = get_last_checkpoint_()
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        if training_args.local_rank == 0:
            logging.info(f"checkpoint = {checkpoint}")
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
        logging.info("=" * 60)
        logging.info("Start Training ....")
        logging.info("=" * 60)
        trainer.train(resume_from_checkpoint=checkpoint)
        dist.init_process_group('gloo', init_method='file:///tmp/fedflow', rank=0, world_size=1)
        dist.barrier()
        save_on_zero_3(trainer, model)
    if training_args.deepspeed is not None:
        dist.barrier()


if __name__ == "__main__":
    # Parse arguments
    model_args, data_args, training_args, lora_config_args, fed_args = parse_args()

    # Load pretrained model and tokenizer
    tokenizer, model = get_base_model_and_tokenizer()

    # Prepare dataset
    train_dataset = prepare_dataset() if training_args.do_train and not fed_args.is_vender() else TensorDataset(
        torch.zeros((1, 1)))

    # Training
    train(model_args, data_args, training_args, train_dataset, model, tokenizer)
