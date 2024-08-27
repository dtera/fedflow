# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
import logging

import torch.distributed as dist
from transformers import Trainer

from fedflow.llm.arguments import parse_args, fetch_args_from_dataclass, get_last_checkpoint_
from fedflow.util.data_utils import CausalCollator, prepare_dataset
from fedflow.util.eval_utils import add_eval_callback
from fedflow.util.model_utils import get_base_model_and_tokenizer, adapt_with_lora, save_on_zero_3
from fedflow.util.train_utils import tb_add_text


def main():
    # Parse arguments
    model_args, data_args, training_args, lora_config_args = parse_args()

    # Load pretrained model and tokenizer
    model, tokenizer = get_base_model_and_tokenizer(model_args)

    # Prepare dataset
    train_dataset = prepare_dataset(data_args) if training_args.do_train else None

    # Initialize Trainer
    trainer = Trainer(
        model=adapt_with_lora(lora_config_args, model),  # Adapt with original lora
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=CausalCollator(tokenizer, **fetch_args_from_dataclass(CausalCollator, data_args)),
    )

    # Add some text records to tensorboard
    tb_add_text(trainer, training_args, lora_config_args, model)

    # Add eval callback
    add_eval_callback(training_args, lora_config_args, train_dataset, trainer, tokenizer)

    # Detecting last checkpoint.
    last_checkpoint = get_last_checkpoint_(training_args)

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
        dist.barrier()
        save_on_zero_3(training_args, trainer, model)
    if training_args.deepspeed is not None:
        dist.barrier()


if __name__ == "__main__":
    main()
