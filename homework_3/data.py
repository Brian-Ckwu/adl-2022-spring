import jsonlines

from typing import List
from argparse import Namespace

import pandas as pd

from torch.utils.data import Dataset, DataLoader

from transformers import T5Tokenizer

from utils import load_config

class T5SummaryDataset(Dataset):
    def __init__(self, texts: List[str], summaries: List[str], tokenizer: T5Tokenizer, max_target_length: int = 64, prefix: str = ""):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.prefix = "" # NOTE: mT5 doesn't use prefix

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]
        # token_text = self.tokenizer(text, truncation=True, return_tensors="pt")
        # token_summary = self.tokenizer(summary, truncation=True, return_tensors="pt")
        return text, summary
    
    def collate_fn(self, samples):
        texts = list()
        summaries = list()

        for text, summary in samples:
            texts.append(self.prefix + text)
            summaries.append(summary)
        
        padded_texts = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            padded_summaries = self.tokenizer(summaries, max_length=self.max_target_length, truncation=True, padding=True, return_tensors="pt")
        return padded_texts, padded_summaries

def extract_maintexts_and_titles(jsonl_file: str):
    maintexts = list()
    titles = list()
    with jsonlines.open(jsonl_file) as dicts:
        for d in dicts:
            maintexts.append(d["maintext"])
            titles.append(d["title"])
    return maintexts, titles

def extract_ids(jsonl_file: str):
    ids = list()
    with jsonlines.open(jsonl_file) as dicts:
        for d in dicts:
            ids.append(d["id"])
    return ids

if __name__ == "__main__":
    config = load_config("./config.json")
    args = Namespace(**config)

    texts, titles = extract_maintexts_and_titles("./data/train.jsonl")
    
    df = pd.DataFrame(
        {
            "texts": texts,
            "titles": titles
        }
    )

    print(df.texts.str.len().describe())
    print(df.titles.str.len().describe())