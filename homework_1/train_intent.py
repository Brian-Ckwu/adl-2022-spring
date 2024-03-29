import time
import json
import pickle
from genericpath import samestat
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import trange
from torch.utils.data import DataLoader

from dataset import SeqClsDataset
from utils import *
from model import SeqClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def trainer(args) -> float: # return the objective metric
    start_time = time.time()
    # handle directories
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir = args.model_dir / render_model_name(args, hparams=[
        "rnn_type", 
        "hidden_size", 
        "num_layers", 
        "rnn_dropout",
        "mlp_dropout", 
        "lr", 
        "batch_size", 
        "optimizer",
        "scheduler"]
    )
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    same_seeds(args.seed)
    # Data
    print(f"Loading data...")
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }

    train_loader = DataLoader(datasets[TRAIN], args.batch_size, shuffle=True, pin_memory=True, collate_fn=datasets[TRAIN].collate_fn)
    val_loader = DataLoader(datasets[DEV], args.batch_size, shuffle=False, pin_memory=True, collate_fn=datasets[DEV].collate_fn)

    # Model
    print(f"Init model & optimizer...")
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model = SeqClassifier(
        embeddings=embeddings, 
        rnn_type=args.rnn_type, 
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers, 
        rnn_dropout=args.rnn_dropout, 
        mlp_dropout=args.mlp_dropout,
        bidirectional=args.bidirectional, 
        num_class=len(intent2idx)
    ).to(args.device)
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = getattr(torch.optim.lr_scheduler, args.scheduler)(optimizer, T_max=args.num_epoch, verbose=True)

    # Optimization
    print(f"\n ===== Start training ===== \n")
    train_log = {
        "train_acc": list(),
        "train_loss": list(),
        "val_acc": list(),
        "val_loss": list(),
    }
    no_improve_epochs = 0
    best_val_acc = 0
    for epoch in trange(args.num_epoch, desc="Epoch"):
        model.train()
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            # make prediction & calculate loss
            scores = model(x)
            loss = model.calc_loss(scores, y)

            # back-proppagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate Train Set & Dev Set
        train_acc, train_loss = evaluate_model(model, train_loader, args.device) if args.record_train else (0, 0)
        val_acc, val_loss = evaluate_model(model, val_loader, args.device)

        scheduler.step()

        for key, value in zip(["train_acc", "train_loss", "val_acc", "val_loss"], [train_acc, train_loss, val_acc, val_loss]):
            train_log[key].append(value)
        print(f"Train ACC: {train_acc}; Train loss: {train_loss}; Val ACC: {val_acc}; Val loss: {val_loss}")
        if val_acc > best_val_acc:
            # save config, model, and evaluation result
            best_val_acc = val_acc
            no_improve_epochs = 0
            args.best_epoch = epoch + 1
            save_checkpoint(args, model, best_val_acc)
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs.")
            if no_improve_epochs > args.es_epoch:
                print(f"Training halted at epoch {epoch + 1}.")
                break

    # save train_log and print the best ACC
    save_train_log(train_log, args)
    print("\n\n Best ACC: {} \n\n".format(best_val_acc))
    
    end_time = time.time()
    train_minutes = (end_time - start_time) / 60
    print(f"The training schedule took {train_minutes} minutes.")

    return best_val_acc, train_log

def evaluate_model(model: SeqClassifier, loader: DataLoader, device: str) -> tuple:
    model.eval()
    correct, samples = 0, 0 # acc
    total_loss = 0 # loss
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            scores = model(x)
            # acc
            correct += (scores.argmax(dim=-1) == y).cpu().detach().sum().item()
            samples += len(y)
            # loss
            loss = model.calc_loss(scores, y)
            total_loss += loss.cpu().detach().item() * len(y)
    
    acc = correct / samples
    mean_loss = total_loss / len(loader.dataset)
    return acc, mean_loss

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        help="Directory to save the models, configs, and evaluations.",
        default="./models/intent/",
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

    # optimizer
    parser.add_argument("--optimizer", type=str, default="NAdam")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0)

    # schedular
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR")

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    # training
    parser.add_argument(
        "--device", type=str, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--es_epoch", type=int, default=10)
    parser.add_argument("--record_train", type=bool, default=True)

    args = parser.parse_args()
    return args

def render_model_name(args: Namespace, hparams: List[str]) -> str:
    model_name_l = list()
    for hparam in hparams:
        value = getattr(args, hparam)
        model_name_l.append(f"{hparam}-{value}")
    model_name = '_'.join(model_name_l)
    return model_name


if __name__ == "__main__":
    args = parse_args()
    trainer(args)