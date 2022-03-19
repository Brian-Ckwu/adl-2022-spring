import torch

from typing import List, Dict
from torch.utils.data import Dataset
from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        batch_tokens = list()
        labels = list()
        test_ids = list()

        for sample in samples:
            batch_tokens.append(sample["text"])
            if "intent" in sample: # if train mode
                labels.append(sample["intent"])
            elif "id" in sample:
                test_ids.append(sample["id"])
            else:
                raise KeyError("Please check that the data is correct.")
        
        padded_ids = torch.LongTensor(self.vocab.encode_batch(batch_tokens)) # NOTE: consider max_len --> not useful
        label_indices = torch.LongTensor([self.label2idx(label) for label in labels])
        return (padded_ids, label_indices) if len(label_indices) > 0 else (padded_ids, test_ids)

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
