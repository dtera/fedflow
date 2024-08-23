# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
import os
from dataclasses import dataclass

from datasets import load_from_disk, load_dataset

from fedflow.util.data_utils import BaseDatasetHandler


@dataclass
class MMLUDatasetHandler(BaseDatasetHandler):
    available_configs = ["train", "fs", "zs"]
    available_splits = ["train", "validation", "test"]

    def format_dataset(self, config, dataset):
        if config != "train":
            return dataset

        def format_mmlu(sample):
            ins = sample["question"]
            if not ins.endswith("?"):
                ins += "____"
            abcd = sample["choices"]
            ins += f"\nA.{abcd[0]} B.{abcd[1]} C.{abcd[2]} D.{abcd[3]}\nAnswer:"
            op_dict = {0: "A", 1: "B", 2: "C", 3: "D"}
            return {"input": ins, "output": op_dict[sample["answer"]]}

        return dataset.map(
            format_mmlu, remove_columns=["question", "choices", "answer"]
        )

    def load_dataset(self, dataset_path: str, config: str):
        if config == "zs":
            mmlu_dataset = load_dataset(
                "json",
                data_files={
                    "validation": os.path.join(
                        f"{dataset_path}/zero_shot_mmlu_val.json"
                    ),
                    "test": os.path.join(
                        f"{dataset_path}/zero_shot_mmlu_test.json"
                    ),
                },
            )
        # MMLU Five-shot (Eval/Test only)
        elif config == "fs":
            mmlu_dataset = load_dataset(
                "json",
                data_files={
                    "validation": os.path.join(
                        f"{dataset_path}/five_shot_mmlu_val.json"
                    ),
                    "test": os.path.join(
                        f"{dataset_path}/five_shot_mmlu_test.json"
                    ),
                },
            )
        elif config == "train":
            mmlu_dataset = {
                "train": load_from_disk(
                    f"{dataset_path}/mmlu_auxiliary_train"
                )
            }
        return mmlu_dataset
