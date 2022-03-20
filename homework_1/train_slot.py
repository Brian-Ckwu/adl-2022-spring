import time
import json
import pickle
from genericpath import samestat
from argparse import ArgumentParser, Namespace
from pathlib import Path, PosixPath
from typing import Dict, List, Callable

import pandas as pd
import torch
from tqdm import trange
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import SlotTagDataset
from utils import *
from model import SlotTagger

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
    train_loader, val_loader = get_train_val_loaders(args)

    # Model
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model = SlotTagger(embeddings, args).to(args.device)

    # Optimization
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)
    # TODO: add scheduler

    no_improve_epochs = 0
    best_joint_acc = 0
    train_log = {
        "train_joint_acc": list(), "train_token_acc": list(), "train_loss": list(),
        "val_joint_acc": list(), "val_token_acc": list(), "val_loss": list()
    }
    for epoch in trange(args.num_epoch, desc="Epoch"):
        model.train()
        for x_batch, seq_lens, tag_padded in train_loader:
            # move to device
            x_batch = x_batch.to(args.device)
            seq_lens = seq_lens.to(args.device)
            tag_padded = tag_padded.to(args.device)

            # inference & calculate loss
            scores_padded = model(x_batch, seq_lens)
            loss = model.calc_loss(scores_padded, tag_padded)

            # back_propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate Train Set & Dev Set
        train_joint_acc, train_token_acc, train_loss = evaluate_model(model, train_loader, args.device) if args.record_train else (0, 0)
        val_joint_acc, val_token_acc, val_loss = evaluate_model(model, val_loader, args.device)
        for key, value in zip(["train_joint_acc", "train_token_acc", "train_loss", "val_joint_acc", "val_token_acc", "val_loss"], [train_joint_acc, train_token_acc, train_loss, val_joint_acc, val_token_acc, val_loss]):
            train_log[key].append(value)
        print(f"Train Joint ACC: {train_joint_acc}; Train loss: {train_loss}; Val Joint ACC: {val_joint_acc}; Val loss: {val_loss}")
        # TODO: scheduler.step()

        if val_joint_acc > best_joint_acc:
            best_joint_acc = val_joint_acc
            no_improve_epochs = 0
            args.best_epoch = epoch + 1
            save_checkpoint(args, model, best_joint_acc)
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs.")
            if no_improve_epochs > args.es_epoch:
                print(f"Training halted at epoch {epoch + 1}")
                break

    save_train_log(train_log, args)

    return 

def evaluate_model(model: SlotTagger, loader: DataLoader, device: str) -> tuple:
    model.eval()
    joint_correct, nsamples = 0, 0 # joint acc
    token_correct, ntokens = 0, 0 # token acc
    total_loss = 0 # loss

    for x_batch, seq_lens, tag_padded in loader:
        x_batch, seq_lens, tag_padded = x_batch.to(device), seq_lens.to(device), tag_padded.to(device)
        with torch.no_grad():
            scores_padded = model(x_batch, seq_lens)
            scores_seqs = model.separate_seqs(scores_padded, seq_lens)
            preds_seqs = model.scores_seqs_to_preds(scores_seqs)
            tags_seqs = model.separate_seqs(tag_padded, seq_lens)
            # joint acc
            joint_correct += sum([torch.all(torch.eq(preds, tags)).cpu().detach().item() for preds, tags in zip(preds_seqs, tags_seqs)])
            nsamples += len(tags_seqs)

            # token acc
            token_correct += sum([(preds == tags).cpu().detach().sum().item() for  preds, tags in zip(preds_seqs, tags_seqs)])
            ntokens += seq_lens.cpu().detach().sum().item()

            # loss
            loss = model.calc_loss(scores_padded, tag_padded)
            total_loss += loss.cpu().detach().item() * seq_lens.cpu().detach().sum().item()
    
    joint_acc = joint_correct / nsamples
    token_acc = token_correct / ntokens
    return joint_acc, token_acc, total_loss

def get_train_val_loaders(args: Namespace):
    print(f"Loading data...")
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in ["train", "eval"]}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SlotTagDataset] = {
        split: SlotTagDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

    train_loader = DataLoader(datasets["train"], args.batch_size, shuffle=True, pin_memory=True, collate_fn=datasets["train"].collate_fn)
    val_loader = DataLoader(datasets["eval"], args.batch_size, shuffle=False, pin_memory=True, collate_fn=datasets["eval"].collate_fn)

    return train_loader, val_loader

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        help="Directory to save the models, configs, and evaluations.",
        default="./models/slot/",
    )
    # seed
    parser.add_argument("--seed", type=int, default=7)

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--rnn_type", type=str, default="LSTM")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--rnn_dropout", type=float, default=0.50)
    parser.add_argument("--mlp_dropout", type=float, default=0.50)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--num_class", type=int, default=9)

    # optimizer
    parser.add_argument("--optimizer", type=str, default="NAdam")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0)

    # scheduler # TODO: add scheduler
    parser.add_argument("--scheduler", type=str, default="NoScheduler")

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=str, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=25)
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
    # with open("./slot_hparams_config.json") as f:
    #     hparams_config = json.load(f)
    
    # assert torch.cuda.is_available()
    # best_hparams, values = optimize_hparams(trainer, args, hparams_config, n_trials=25)
    # print(best_hparams)
    # print(values)