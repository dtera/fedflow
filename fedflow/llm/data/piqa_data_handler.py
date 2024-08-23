# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
from dataclasses import dataclass

from fedflow.util.data_utils import BaseDatasetHandler


@dataclass
class PIQADatasetHandler(BaseDatasetHandler):
    available_configs = ["zs", "train"]
    available_splits = ["train", "validation", "test"]
    option_dict = {
        "fs": {0: "A", 1: "B"},
        "zs": {0: "A", 1: "B"},
        "train": {0: "A", 1: "B"},
    }

    def format_dataset(self, config, dataset):
        def format_sample(sample):
            temp = {
                "input": "Question: {}\nA. {}\nB. {}\nAnswer:".format(
                    sample["goal"], sample["sol1"], sample["sol2"]
                ),
                "output": self.option_dict[config][sample["label"]]
            }
            return temp

        dataset = dataset.map(format_sample)
        return dataset
