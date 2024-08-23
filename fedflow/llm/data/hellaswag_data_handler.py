# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
import re
from dataclasses import dataclass

from fedflow.util.data_utils import BaseDatasetHandler


@dataclass
class HellaswagDatasetHandler(BaseDatasetHandler):
    available_configs = ["zs", "train"]
    available_splits = ["train", "validation", "test"]
    option_dict = {
        "zs": {0: "A", 1: "B", 2: "C", 3: "D"},
        "train": {0: "A", 1: "B", 2: "C", 3: "D"},
    }

    def format_dataset(self, config, dataset):
        def preprocess(text):
            text = text.strip()
            # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
            text = text.replace(" [title]", ". ")
            text = re.sub("\[.*?]", "", text)
            text = text.replace("  ", " ")
            return text

        def format_sample(sample):
            ctx = sample["ctx_a"] + " " + sample["ctx_b"].capitalize()
            temp = {
                "input": "{}: {} ___\n{}\nAnswer:".format(
                    sample["activity_label"],
                    ctx,
                    "\n".join(
                        [
                            "{}. {}".format(self.option_dict[config][i], preprocess(t))
                            for i, t in enumerate(sample["endings"])
                        ]
                    ),
                ),
                "output": "{}. {}".format(
                    self.option_dict[config][int(sample["label"])],
                    sample["endings"][int(sample["label"])],
                )
            }
            return temp

        dataset = dataset.map(format_sample)
        return dataset
