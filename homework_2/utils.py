import logging
from typing import List
from argparse import Namespace
from itertools import chain

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
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

def render_exp_name(args: Namespace, fields: List[str]):
    exp_name_l = list()
    for field in fields:
        field_value = getattr(args, field)
        field_pair = f"{field}-{field_value}"
        exp_name_l.append(field_pair)

    exp_name = '_'.join(exp_name_l)
    return exp_name

def prepare_logger(accelerator, datasets, transformers) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format=r"%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt=r"%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    return logger

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

def evaluate_model(valid_loader: DataLoader, model: AutoModelForMultipleChoice, args: Namespace, accelerator: Accelerator, metric) -> float:
    model.eval()
    for batch in valid_loader:
        batch = move_batch_to_device(batch, args.device)
        with torch.no_grad():
            outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        metric.add_batch(
            predictions=accelerator.gather(preds),
            references=accelerator.gather(batch["labels"])
        )
    
    eval_metric = metric.compute()["accuracy"]
    model.train()
    return eval_metric