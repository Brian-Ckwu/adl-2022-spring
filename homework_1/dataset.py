import torch

from typing import List, Dict, Tuple
from torch.utils.data import Dataset
from utils import Vocab, pad_to_len


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
            tokens = [token.lower() for token in sample["text"].split()]
            batch_tokens.append(tokens)
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

class SlotTagDataset(Dataset):
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
        self._idx2label = {idx: tag for tag, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Tuple:
        # TODO: implement slot tagging version
        token_ids = list()
        token_lens = list()
        tag_ids = list()
        data_ids = list()

        for sample in samples:
            token_ids.append(sample["tokens"])
            token_lens.append(len(sample["tokens"]))
            if "tags" in sample:
                tag_ids.append([self.label2idx(tag) for tag in sample["tags"]])
            data_ids.append(sample["id"])
        
        padded_token_ids = torch.LongTensor(self.vocab.encode_batch(token_ids))
        token_lens = torch.LongTensor(token_lens)
        padded_tag_ids = torch.LongTensor(pad_to_len(tag_ids, padded_token_ids.shape[1], -100))

        return (padded_token_ids, token_lens, padded_tag_ids) if len(tag_ids) > 0 else (padded_token_ids, token_lens, data_ids)

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]