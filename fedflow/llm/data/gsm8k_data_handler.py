# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
from dataclasses import dataclass

from fedflow.util.data_utils import BaseDatasetHandler


@dataclass
class GSM8KDatasetHandler(BaseDatasetHandler):
    available_configs = ["train", "test"]
    available_splits = ["train", "test"]

    def format_dataset(self, config, dataset):
        dataset = dataset.map(
            lambda x: {
                "input": "Question: {}\nAnswer:".format(x["question"]),
                "output": x["answer"],
            },
            remove_columns=["question", "answer"],
        )
        return dataset
