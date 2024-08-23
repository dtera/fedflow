# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
import logging
import os
from dataclasses import asdict

import torch.distributed as dist
from peft import get_peft_model, LoraConfig
from transformers import (
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from fedflow.llm.arguments import parse_args
from fedflow.util.data_utils import prepare_dataset
from fedflow.util.model_utils import get_base_model_and_tokenizer, save_on_zero_3


def main():
    model_args, data_args, training_args, sap_lora_config_args = parse_args()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                             "Use --overwrite_output_dir to overcome.")
        elif (last_checkpoint is not None and training_args.resume_from_checkpoint is None):
            logging.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer, only FederatedLLM m is trainable if it's private lora
    model, tokenizer = get_base_model_and_tokenizer(model_args)

    # adapt with original lora
    if not sap_lora_config_args.is_federated and sap_lora_config_args.baseline_use_lora:
        model = get_peft_model(model, LoraConfig(**sap_lora_config_args.peft_lora_config))

    # set a,b trainable
    if sap_lora_config_args.trainable_a:
        logging.info("Set lora a trainable")
        for n, p in model.named_parameters():
            if "lora_A" in n:
                p.requires_grad = True
    if sap_lora_config_args.trainable_b:
        logging.info("Set lora b trainable")
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.requires_grad = True

    if training_args.do_train:
        train_dataset = prepare_dataset(data_args)

    # Data collator
    from fedflow.util.data_utils import CausalCollator

    collator_kwargs = {
        "target_max_len": data_args.target_max_len,
        "predict_with_generate": data_args.predict_with_generate,
        "train_on_source": data_args.train_on_source,
        "input_field": "input",
        "target_field": "output",
    }
    data_collator = CausalCollator(
        tokenizer, source_max_len=data_args.source_max_len, **collator_kwargs
    )

    # add callback

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # add some text records to tensorboard
    from transformers.integrations import TensorBoardCallback

    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, TensorBoardCallback):
            if training_args.local_rank == 0:
                if cb.tb_writer is None:
                    cb._init_summary_writer(args=training_args)
                cb.tb_writer.add_text(
                    tag="exp_param", text_string=str(asdict(sap_lora_config_args))
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
                        tag="private_lora",
                        text_string=str(private_lora_module),
                    )
                except Exception:
                    pass

    if training_args.do_eval:
        logging.info("add eval callback")
        from fedflow.util.eval_utils import EvalHarnessCallBack

        if training_args.eval_steps < 1.0:
            eval_steps = min(
                int(
                    len(train_dataset)
                    // (training_args.per_device_train_batch_size * 4)
                    * training_args.eval_steps
                ),
                1,
            )
        else:
            eval_steps = training_args.eval_steps
        trainer.add_callback(
            EvalHarnessCallBack(
                trainer=trainer,
                tokenizer=tokenizer,
                tasks=sap_lora_config_args.eval_tasks,
                eval_steps=eval_steps,
                eval_start=sap_lora_config_args.eval_start,
                do_init_eval=sap_lora_config_args.do_init_eval,
                eval_batch_size=training_args.per_device_eval_batch_size,
            )
        )

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
        logging.info("=" * 40)
        logging.info("Start Training ....")
        logging.info("=" * 40)
        trainer.train(resume_from_checkpoint=checkpoint)
        dist.barrier()
        save_on_zero_3(training_args, trainer, model)
    if training_args.deepspeed is not None:
        dist.barrier()


if __name__ == "__main__":
    main()
