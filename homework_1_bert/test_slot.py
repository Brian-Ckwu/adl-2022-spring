import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from dataset import SlotTagDataset
from model import SlotTagger
from utils import Vocab


def main(args):
    # Read Data
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    test_set = SlotTagDataset(data, vocab, tag2idx, args.max_len)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False, pin_memory=True, collate_fn=test_set.collate_fn)

    # Load Model
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model = SlotTagger(embeddings, args).to(args.device)
    ckpt_state_dict = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt_state_dict)

    # Predict Test Set
    model.eval()
    test_preds = list() # (test_id, IOBs)

    for x_batch, seq_lens, test_ids in test_loader:
        x_batch, seq_lens = x_batch.to(args.device), seq_lens.to(args.device)
        with torch.no_grad():
            scores_padded = model(x_batch, seq_lens)
            scores_seqs = model.separate_seqs(scores_padded, seq_lens)
            preds_seqs = model.scores_seqs_to_preds(scores_seqs)
            assert len(preds_seqs) == len(test_ids)
            for preds_seq, test_id in zip(preds_seqs, test_ids):
                IOB = [test_set.idx2label(idx) for idx in preds_seq.cpu().tolist()]
                IOB_str = ' '.join(IOB)
                test_preds.append((test_id, IOB_str))
    
    # TODO: write prediction to file (args.pred_file)
    write_preds(test_preds, args)

def write_preds(preds: List[Tuple[str, str]], args: Namespace) -> None:
    save_path = args.pred_file
    with open(save_path, mode="wt") as f:
        f.write(f"id,tags\n")
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
        default="./data/slot/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to models.",
        default="./models/slot_best.pth"
    )
    parser.add_argument(
        "--pred_file",
        type=Path,
        default="./preds/slot_test.csv"
    )
    # seed
    parser.add_argument("--seed", type=int, default=7)

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--rnn_type", type=str, default="LSTM")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--rnn_dropout", type=float, default=0.50)
    parser.add_argument("--mlp_dropout", type=float, default=0.50)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--num_class", type=int, default=9)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
