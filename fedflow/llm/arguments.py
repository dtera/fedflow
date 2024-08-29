# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional, List

from peft import LoraConfig
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint

from fedflow.register import args, register_arg

logging.basicConfig(level=logging.INFO)


@dataclass
class FedArguments:
    """arguments used in FederatedLLM"""
    comm_port: Optional[int] = field(
        default=10000,
        metadata={
            "help": "socket port."
        },
    )
    comm_addr: Optional[str] = field(
        default="127.0.0.1:10000",
        metadata={
            "help": "socket address."
        },
    )
    role: Optional[str] = field(
        default="vender",
        metadata={
            "help": "role: vender or customer."
        },
    )
    part_id: Optional[str] = field(
        default="1000",
        metadata={
            "help": "party id."
        },
    )


@dataclass
class FedLoraConfig:
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
    use_lora: Optional[bool] = field(
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
    input_field: Optional[str] = field(default="input", metadata={"help": ("input field")})
    target_field: Optional[str] = field(default="output", metadata={"help": ("target field")})
    predict_with_generate: bool = field(default=False)
    train_on_source: bool = field(default=False)


def get_last_checkpoint_():
    training_args: TrainingArguments = args["training_args"]
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
    return last_checkpoint


def fetch_args_from_dataclass(data_class, dc_obj):
    return fetch_args_from_dict(data_class, asdict(dc_obj))


def fetch_args_from_dict(data_class, kwargs):
    return {k: v for k, v in kwargs.items() if k in data_class.__dict__['__match_args__']}


def parse_args():
    # See all possible arguments in src/transformers/training_args.py，or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, FedLoraConfig, FedArguments)
    )
    # If we pass only one argument to the script and it's the path to a json file, let's parse it to get our arguments.
    model_args, data_args, training_args, lora_config_args, fed_args = (
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
    logging.info(f"sap_lora_config parameters {lora_config_args}")
    logging.info(f"fed_args parameters {fed_args}")
    register_arg("model_args", model_args)
    register_arg("data_args", data_args)
    register_arg("training_args", training_args)
    register_arg("lora_config_args", lora_config_args)
    register_arg("fed_args", fed_args)
    return model_args, data_args, training_args, lora_config_args, fed_args
