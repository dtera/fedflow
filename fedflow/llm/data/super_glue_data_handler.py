# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
from dataclasses import dataclass

from fedflow.util.data_utils import BaseDatasetHandler


@dataclass
class SuperGLUEDatasetHandler(BaseDatasetHandler):
    available_configs = ["boolq", "multirc", "cb", "rte", "wic", "wsc"]
    available_splits = ["train", "validation", "test"]

    def format_dataset(self, config, dataset):
        if config == "boolq":
            outputs = {0: "No", 1: "Yes"}
            dataset = dataset.map(
                lambda x: {
                    "idx": x["idx"],
                    "input": f'{x["passage"]} {x["question"][0].upper() + x["question"][1:]}{"" if x["question"].endswith("?") else "?"}\n',
                    # noqa
                    "output": outputs[x["label"]],
                    "labels": x["label"],
                },
                remove_columns=["question", "passage", "idx", "label"],
            )
        elif config == "multirc":
            outputs = {0: "No", 1: "Yes"}
            dataset = dataset.map(
                lambda x: {
                    "idx": x["idx"]["answer"],
                    "input": f'{x["paragraph"]}\nQuestion: {x["question"]}\nI found this answer "{x["answer"]}". Is that correct? Yes or No?\n',
                    # noqa
                    "output": outputs[x["label"]],
                    "labels": x["label"],
                },
                remove_columns=["paragraph", "question", "answer", "idx", "label"],
            )
        elif config == "copa":
            capitalization: str = "correct"
            effect_conj: str = " so "
            cause_conj: str = " because "
            outputs = None

            def get_conjucture(sample):
                if sample["question"] == "effect":
                    conjunction = effect_conj
                elif sample["question"] == "cause":
                    conjunction = cause_conj
                else:
                    raise NotImplementedError
                return conjunction

            def get_prompt(sample):
                premise = sample["premise"].rstrip()
                if premise.endswith(
                        "."
                ):  # TODO Add other scripts with different punctuation
                    premise = premise[:-1]
                conjunction = get_conjucture(sample)
                prompt = premise + conjunction
                if capitalization == "upper":
                    prompt = prompt.upper()
                elif capitalization == "lower":
                    prompt = prompt.lower()
                return prompt

            def encode(sample):
                prompt = get_prompt(sample)
                return prompt

            def capitalize(c):
                if capitalization == "correct":
                    words = c.split(" ")
                    if words[0] != "I":
                        words[0] = words[0].lower()
                    return " ".join(words)
                elif capitalization == "bug":
                    return c
                elif capitalization == "upper":
                    return c.upper()
                elif capitalization == "lower":
                    return c.lower()
                else:
                    raise NotImplementedError

            def verbalize(sample, candidate):
                prompt = get_prompt(sample)
                return prompt + capitalize(candidate)

            def encode_sfc(sample):
                conjunction = get_conjucture(sample)
                return conjunction.strip()

            def verbalize_sfc(sample, candidate):
                conjunction = get_conjucture(sample)
                sfc_prompt = conjunction.strip() + " " + capitalize(candidate)
                return sfc_prompt

            dataset = dataset.map(
                lambda x: {
                    "idx": x["idx"],
                    "input": get_prompt(x),
                    "output": capitalize(x[f"choice{x['label'] + 1}"]),
                    "labels": x["label"],
                }
            )
        elif config == "record":
            outputs = None
            dataset = dataset.map(
                lambda x: {
                    "idx": x["idx"],
                    "input": f'{x["passage"]}\n{x["query"]}\nQuestion: what is the "@placeholder"\nAnswer: ',
                    "output": x["answers"],
                    "labels": None,
                }
            )
        elif config == "rte":
            outputs = {0: "Yes", 1: "No"}
            dataset = dataset.map(
                lambda x: {
                    "idx": x["idx"],
                    "input": f'{x["premise"]}\nDoes this mean that "{x["hypothesis"]}" is true? Yes or No?\n',
                    "output": outputs[x["label"]],
                    "labels": x["label"],
                },
                remove_columns=["premise", "hypothesis", "idx", "label"],
            )
        elif config == "cb":
            outputs = {0: "Yes", 1: "No", 2: "Maybe"}
            dataset = dataset.map(
                lambda x: {
                    "idx": x["idx"],
                    "input": f'Suppose {x["premise"]} Can we infer that "{x["hypothesis"]}"? Yes, No, or Maybe?',
                    "output": outputs[x["label"]],
                    "labels": x["label"],
                }
            )
        elif config in ["wic"]:
            outputs = {0: "No", 1: "Yes"}
            dataset = dataset.map(
                lambda x: {
                    "idx": x["idx"],
                    "input": f'Does the word "{x["word"]}" have the same meaning in these two sentences? Yes, No?\n{x["sentence1"]}\n{x["sentence2"]}\n',
                    # noqa
                    "output": outputs[x["label"]],
                    "labels": x["label"],
                },
                remove_columns=[
                    "word",
                    "sentence1",
                    "sentence2",
                    "start1",
                    "start2",
                    "end1",
                    "end2",
                    "idx",
                    "label",
                ],
            )
        elif config in ["wsc"]:
            outputs = {0: "No", 1: "Yes"}
            dataset = dataset.map(
                lambda x: {
                    "idx": x["idx"],
                    "input": f'{x["text"]}\nIn the previous sentence, does the pronoun "{x["span2_text"].lower()}" refer to {x["span1_text"]}? Yes or No?\n',
                    # noqa
                    "output": outputs[x["label"]],
                    "labels": x["label"],
                }
            )
        else:
            raise NotImplementedError()

        return super().format_dataset(self, config, dataset)
