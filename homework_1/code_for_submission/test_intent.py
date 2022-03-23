import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab
from train_intent import evaluate_model


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    test_set = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False, pin_memory=True, collate_fn=test_set.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model = SeqClassifier(
        embeddings,
        args.rnn_type,
        args.hidden_size,
        args.num_layers,
        args.rnn_dropout,
        args.mlp_dropout,
        args.bidirectional,
        test_set.num_classes,
    ).to(args.device)
    # load weights into model
    ckpt_state_dict = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt_state_dict)

    # TODO: predict dataset
    model.eval()
    preds: List[Tuple[str, str]] = list()
    for x, test_ids in test_loader:
        x = x.to(args.device)
        with torch.no_grad():
            scores = model(x)
            pred_indices = scores.argmax(dim=-1).cpu().detach().tolist()
            for pred_idx, test_id, in zip(pred_indices, test_ids):
                pred_label = test_set.idx2label(pred_idx)
                preds.append((test_id, pred_label))
    
    # TODO: write prediction to file (args.pred_file)
    write_preds(preds, args)

def predict_whole(model: SeqClassifier, dataset: SeqClsDataset, dataloader: DataLoader):
    model.eval()
    preds: List[Tuple[str, str]] = list()

    for x, test_ids in dataloader:
        x = x.to(args.device)
        with torch.no_grad():
            scores = model(x)
            pred_indices = scores.argmax(dim=-1).cpu().detach().tolist()
            for pred_idx, test_id, in zip(pred_indices, test_ids):
                pred_label = dataset.idx2label(pred_idx)
                preds.append((test_id, pred_label))

    return preds

def write_preds(preds: List[Tuple[str, str]], args: Namespace) -> None:
    save_path = args.pred_file
    with open(save_path, mode="wt") as f:
        f.write(f"id,intent\n")
        for pair in preds:
            test_id = pair[0]
            y_pred = pair[1]
            f.write(f"{test_id},{y_pred}\n")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to models.",
        default="./models/intent_best.pth"
    )
    parser.add_argument(
        "--pred_file",
        type=Path,
        default="./preds/intent_best.csv"
    )
    # seed
    parser.add_argument("--seed", type=int, default=7)

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--rnn_type", type=str, default="LSTM")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--rnn_dropout", type=float, default=0.50)
    parser.add_argument("--mlp_dropout", type=float, default=0.50)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)