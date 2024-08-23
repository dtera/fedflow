# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

from peft import LoraConfig
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)


@dataclass
class SAPLoraConfig:
    """arguments used in FederatedLLM experiment"""
    eval_tasks: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "evaluation task name."
        },
    )
    eval_start: Optional[int] = field(
        default=0,
        metadata={"help": "eval from which step"},
    )
    do_init_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "whether eval before training"},
    )
    is_federated: Optional[bool] = field(
        default=False,
        metadata={"help": "if True then FederatedLLM, otherwise lora or ft"},
    )
    baseline_use_lora: Optional[bool] = field(
        default=True,
        metadata={"help": "use lora"},
    )
    peft_lora_config: Optional[LoraConfig] = field(
        default=None,
        metadata={
            "help": "lora configuration for target modules and target layers"
        },
    )
    trainable_a: Optional[bool] = field(
        default=False,
        metadata={"help": "whether matrix a is trainable in FederatedLLM"},
    )
    trainable_b: Optional[bool] = field(
        default=False,
        metadata={"help": "whether matrix b is trainable in FederatedLLM"},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_type: str = field(
        default="sap",
        metadata={
            "help": "model_type (str): model type, something like baseline, sap, see model_utils.py"
        },
    )
    weights_path: str = field(
        default=None,
        metadata={
            "help": "weights_path (str, optional): custom weights path. Defaults to None, see model_utils.py"
        },
    )
    pretrained_model_name_or_path: str = field(
        default="bigscience/bloomz-7b1-mt",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast: bool = field(
        default=False,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    pad_with_unk_token: bool = field(
        default=True,
        metadata={
            "help": "Whether to padding with unk_token or not."
        },
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Whether to trust remote code or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_dataset_name_or_paths: dict[(str, str)] = field(default_factory=dict)
    dataset_root_dir: str = field(default="./data/")
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    shuffle_data: Optional[bool] = field(default=False)
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    source_max_len: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    target_max_len: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    aistudio_reader_num_processes: int = field(default=4)
    source_prefix: Optional[str] = field(
        default=None,
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."
        },
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the :obj:`decoder_start_token_id`.Useful for"
                " multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token needs to"
                " be the target language token.(Usually it is the target language token)"
            )
        },
    )
    predict_with_generate: bool = field(default=False)
    train_on_source: bool = field(default=False)


def parse_args():
    # See all possible arguments in src/transformers/training_args.py，or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, SAPLoraConfig)
    )
    # If we pass only one argument to the script and it's the path to a json file, let's parse it to get our arguments.
    model_args, data_args, training_args, sap_lora_config_args = (
        (parser.parse_json_file(json_file=os.path.abspath(sys.argv[1])) if sys.argv[1].endswith(".json")
         else parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1])))
        if len(sys.argv) == 2 else parser.parse_args_into_dataclasses()
    )
    # Log on each process the small summary:
    logging.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logging.info(f"Training/evaluation parameters {training_args}")
    logging.info(f"model parameters {model_args}")
    logging.info(f"data parameters {data_args}")
    logging.info(f"sap_lora_config parameters {sap_lora_config_args}")

    return model_args, data_args, training_args, sap_lora_config_args
