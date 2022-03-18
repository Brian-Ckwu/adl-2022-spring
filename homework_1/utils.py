import json
import torch
import torch.nn as nn
import numpy as np
import random
import pathlib
from argparse import Namespace
from typing import Callable, Iterable, List, Dict

class Vocab:
    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(self, vocab: Iterable[str]) -> None:
        self.token2idx = {
            Vocab.PAD: 0,
            Vocab.UNK: 1,
            **{token: i for i, token in enumerate(vocab, 2)},
        }

    @property
    def pad_id(self) -> int:
        return self.token2idx[Vocab.PAD]

    @property
    def unk_id(self) -> int:
        return self.token2idx[Vocab.UNK]

    @property
    def tokens(self) -> List[str]:
        return list(self.token2idx.keys())

    def token_to_id(self, token: str) -> int:
        return self.token2idx.get(token, self.unk_id)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(token) for token in tokens]

    def encode_batch(
        self, batch_tokens: List[List[str]], to_len: int = None
    ) -> List[List[int]]:
        batch_ids = [self.encode(tokens) for tokens in batch_tokens]
        to_len = max(len(ids) for ids in batch_ids) if to_len is None else to_len
        padded_ids = pad_to_len(batch_ids, to_len, self.pad_id)
        return padded_ids


def pad_to_len(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:
    paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
    return paddeds

def same_seeds(seed: int = 7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(args: Namespace, model: nn.Module, objective_metric: float):
    ckpt_name = f"hidden-{args.hidden_size}_nlayers-{args.num_layers}_dropout-{args.dropout}_lr-{args.lr}_batch-{args.batch_size}"
    ckpt_dir: pathlib.PosixPath = args.model_dir / ckpt_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # save config
    args_d = dict()
    for key, value in vars(args).items():
        if type(value) != pathlib.PosixPath:
            args_d[key] = value
    with open(ckpt_dir / "config.json", mode="wt") as f:
        json.dump(args_d, f)

    # save model
    torch.save(model.state_dict(), ckpt_dir / "model.pth")

    # save evaluation results
    with open(ckpt_dir / "metrics.txt", mode="wt") as f:
        f.write(str(objective_metric))


def grid_search(trainer: Callable, args: Namespace, target_hparams: Dict[str, List]):
    return