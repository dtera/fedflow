# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
from dataclasses import dataclass
from typing import List

from datasets import load_from_disk, load_dataset

from fedflow.llm.arguments import DataTrainingArguments
from fedflow.register import datas, args


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


def prepare_dataset():
    """prepare dataset
    Returns:
        _type_: processed dataset
    """
    data_args: DataTrainingArguments = args["data_args"]
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
