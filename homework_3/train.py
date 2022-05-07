import json
from tqdm.auto import tqdm
from pathlib import Path
from argparse import Namespace

import torch
from torch.utils.data import DataLoader
from transformers import MT5ForConditionalGeneration, T5Tokenizer, Adafactor

from data import extract_maintexts_and_titles, T5SummaryDataset
from utils import set_seed, load_config, move_dict_to_device

def trainer(args: Namespace):
    # Configuration
    set_seed(args.seed)
    Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)

    # Data
    # load data
    train_texts, train_titles = extract_maintexts_and_titles(args.train_jsonl); assert len(train_texts) == len(train_titles)
    valid_texts, valid_titles = extract_maintexts_and_titles(args.valid_jsonl); assert len(valid_texts) == len(valid_titles)
    print(f"Data loaded. Showing examples:\n \
        Train text (size = {len(train_texts)}): {train_texts[0][:50]}..., title: {train_titles[0]};\n \
        Valid text (size = {len(valid_texts)}): {valid_texts[0][:50]}..., title: {valid_titles[0]} \
    ")

    # tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
    tokenizer.model_max_length = args.max_text_len

    # dataset & dataloader
    train_set = T5SummaryDataset(train_texts, train_titles, tokenizer, args.max_target_len)
    valid_set = T5SummaryDataset(valid_texts, valid_titles, tokenizer, args.max_target_len)

    train_loader = DataLoader(train_set, args.bs, shuffle=True, pin_memory=True, collate_fn=train_set.collate_fn)
    valid_loader = DataLoader(valid_set, args.bs, shuffle=False, pin_memory=True, collate_fn=valid_set.collate_fn)
    print(f"Finish Dataset & DataLoader construction.")

    # Model
    model = MT5ForConditionalGeneration.from_pretrained(args.t5_model).to(args.device)
    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    print("Model & optimizer loaded.")

    # Optimization
    print("Start training...")
    best_loss = float("inf")
    steps = 0
    for epoch in range(args.nepochs):
        print(f"\n===== Training at epoch {epoch + 1} =====\n")
        for loader_idx, batch in enumerate(train_loader):
            model.train()

            X, y = batch
            X = move_dict_to_device(X, args.device)
            y = move_dict_to_device(y, args.device)

            outputs = model(**X, labels=y["input_ids"])
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            steps += 1

            # evaluation
            if (steps % args.log_steps == 0):
                print(f"Evaluating model at step {steps}...")
                total_loss = 0
                model.eval()
                for batch in tqdm(valid_loader):
                    X, y = batch
                    X = move_dict_to_device(X, args.device)
                    y = move_dict_to_device(y, args.device)

                    with torch.no_grad():
                        outputs = model(**X, labels=y["input_ids"])
                        total_loss += outputs.loss.detach().cpu().item() * len(y["input_ids"])
                
                valid_loss = total_loss / len(valid_loader.dataset)
                print(f"Validation loss: {valid_loss:.4f}")

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    model.save_pretrained(args.model_save_dir)
                    print("Best model saved.")
                    (Path(args.model_save_dir) / "best_loss.txt").write_text(str(best_loss))

    # print some generated summaries for examination
    # X, y = next(iter(valid_loader))
    # X = move_dict_to_device(X, args.device)
    # y = move_dict_to_device(y, args.device)
    # print(tokenizer.decode(X["input_ids"][0]))    
    
    # outputs = model.generate(X["input_ids"])
    # print(outputs)
    # print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    return

if __name__ == "__main__":
    config = load_config("./config.json")
    args = Namespace(**config)

    trainer(args)