# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
import copy
from dataclasses import dataclass, field
from typing import List

import torch
from datasets import load_from_disk, load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from fedflow.llm.arguments import DataTrainingArguments
from fedflow.register import datas

IGNORE_INDEX = -100


@dataclass
class CausalCollator:
    """collator adapted from artido/qlora"""

    tokenizer: PreTrainedTokenizerBase
    source_max_len: int
    train_on_source: bool = field(default=False)
    input_field: str = field(default="input")
    target_field: str = field(default="output")
    target_max_len: int = field(default=10)
    predict_with_generate: bool = field(default=False)

    def __call__(self, features):
        # print(len(features))
        if not self.train_on_source:
            sources = [
                self.tokenizer.bos_token + feature[self.input_field]
                for feature in features
            ]
        else:
            sources = [
                self.tokenizer.bos_token
                + feature[self.input_field]
                + feature[self.target_field]
                + self.tokenizer.eos_token
                for feature in features
            ]
            max_length = self.source_max_len + self.target_max_len
            tokens = self.tokenizer(
                sources,
                padding="max_length",
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            labels = tokens["input_ids"].clone().detach()
            labels[labels == self.tokenizer.pad_token_id] = -100
            tokens["labels"] = labels
            return tokens

        if self.predict_with_generate:
            tokens = self.tokenizer(
                sources,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=self.source_max_len,
            )
            tokens["idx"] = [feature["idx"] for feature in features]
            return tokens

        else:
            targets = []
            for feature in features:
                if isinstance(feature[self.target_field], list):
                    targets += [
                        ", ".join(feature[self.target_field]) + self.tokenizer.eos_token
                    ]
                else:
                    targets += [feature[self.target_field] + self.tokenizer.eos_token]

            # Tokenize
            tokenized_sources_with_prompt = self.tokenizer(
                sources,
                max_length=self.source_max_len,
                truncation=True,
                add_special_tokens=False,
            )
            tokenized_targets = self.tokenizer(
                targets,
                max_length=self.target_max_len,
                truncation=True,
                add_special_tokens=False,
            )
            # Build the input and labels for causal LM
            input_ids = []
            labels = []
            for tokenized_source, tokenized_target in zip(
                    tokenized_sources_with_prompt["input_ids"],
                    tokenized_targets["input_ids"],
            ):
                if not self.predict_with_generate:
                    input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                    if not self.train_on_source:
                        labels.append(
                            torch.tensor(
                                [IGNORE_INDEX for _ in range(len(tokenized_source))]
                                + copy.deepcopy(tokenized_target)
                            )
                        )
                    else:
                        labels.append(
                            torch.tensor(
                                copy.deepcopy(tokenized_source + tokenized_target)
                            )
                        )
                else:
                    input_ids.append(torch.tensor(tokenized_source))
            # Apply padding
            input_ids = pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = (
                pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
                if not self.predict_with_generate
                else None
            )
            data_dict = {
                "input_ids": input_ids,
                "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            }
            if labels is not None:
                data_dict["labels"] = labels
            # print(data_dict["input_ids"].shape)
            # for k in data_dict.keys():
            #     print(data_dict[k].shape)

            return data_dict


@dataclass
class BaseDatasetHandler:
    available_configs: List[str]
    available_splits: List[str]

    def format_dataset(self, config, dataset):
        print(dataset)
        return dataset

    def load_dataset(self, dataset_path: str, config=None):
        if dataset_path.startswith("@"):
            return load_dataset(dataset_path[1:])
        return load_from_disk(dataset_path)

    @classmethod
    def get_dataset(cls, dataset_path: str, config: str, split, check=True):
        if dataset_path is None:
            raise ValueError()
        if check:
            if cls.available_configs and len(cls.available_configs) > 0 and config not in cls.available_configs:
                raise ValueError()
            if cls.available_splits and len(cls.available_splits) > 0 and split not in cls.available_splits:
                raise ValueError()
        dataset = cls.load_dataset(cls, dataset_path.rstrip('/\\'), config)
        return cls.format_dataset(cls, config, dataset[split] if split else dataset)

    @classmethod
    def show_all_configs(cls):
        return cls.available_configs


def prepare_dataset(data_args: DataTrainingArguments):
    """prepare dataset

    Args:
        data_args: Arguments pertaining to what data we are going to input our model for training and eval.

    Returns:
        _type_: processed dataset
    """
    datasets = {}
    for dataset_name, dataset_path in data_args.train_dataset_name_or_paths.items():
        name, config = dataset_name.split("-")
        datasets[dataset_name] = (datas[name] if name in datas else BaseDatasetHandler).get_dataset(
            dataset_path, config, "train", name in datas
        )
    for k, dataset in datasets.items():
        if data_args.max_train_samples:
            datasets[k] = dataset.select(range(min(len(dataset), data_args.max_train_samples)))

    from datasets import concatenate_datasets
    dataset = concatenate_datasets(
        [dataset for dataset in datasets.values()]
    ) if len(datasets.values()) > 1 else list(datasets.values())[0]
    if data_args.shuffle_data:
        dataset = dataset.shuffle()
    return dataset
