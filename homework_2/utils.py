import random
import logging
from typing import List
from argparse import Namespace
from itertools import chain

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForMultipleChoice, AutoTokenizer

class Preprocessor(object):
    def __init__(self, tokenizer: AutoTokenizer, args: Namespace):
        self._tokenizer = tokenizer
        self._args = args
    
    def __call__(self, examples):
        first_sentences = [[q] * 4 for q in examples["question"]]
        second_sentences = [
            [f"{examples[context][i]}" for context in [f"context{j}" for j in range(4)]] for i in range(len(examples["id"]))
        ]
        if "label" in examples.keys():
            labels = examples["label"]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = self._tokenizer(
            first_sentences,
            second_sentences,
            max_length=self._args.max_length,
            padding="max_length" if self._args.pad_to_max_length else False,
            truncation=True,
        )
        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        if "label" in examples.keys():
            tokenized_inputs["labels"] = labels
            
        return tokenized_inputs

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def render_exp_name(args: Namespace, fields: List[str]):
    exp_name_l = list()
    for field in fields:
        field_value = getattr(args, field)
        field_pair = f"{field}-{field_value}"
        exp_name_l.append(field_pair)

    exp_name = '_'.join(exp_name_l)
    return exp_name

def move_batch_to_device(batch, device):
    for k in batch:
        batch[k] = batch[k].to(device)
    
    return batch

def construct_raw_dataset(data_dict_l: List[dict], context_l: List[str], mode: str, to_dict: bool = True):
    if mode not in ["train", "valid", "test"]:
        raise ValueError("mode must be train/valid/test")

    raw_dataset = list()
    for data_dict in data_dict_l:
        # ID & Question
        raw_dict = {
            "id": data_dict["id"],
            "question": data_dict["question"]
        }
        # Contexts
        context_ids = data_dict["paragraphs"]
        for i, context_id in enumerate(context_ids):
            context = context_l[context_id]
            raw_dict[f"context{i}"] = context
        # Label
        if mode != "test":
            raw_dict["label"] = context_ids.index(data_dict["relevant"])
        
        raw_dataset.append(raw_dict)

    if to_dict:
        raw_dataset_d = {k: list() for k in raw_dataset[0].keys()}
        for d in raw_dataset:
            for k, v in d.items():
                raw_dataset_d[k].append(v)
        raw_dataset = raw_dataset_d

    return raw_dataset

encoder_mappings = {
    "BERT": "bert-base-chinese",
    "BERTWWMEXT": "hfl/chinese-bert-wwm-ext",
    "RoBERTaWWMEXT": "hfl/chinese-roberta-wwm-ext"
}