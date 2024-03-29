import json
from tqdm.auto import tqdm
from pathlib import Path
from argparse import Namespace

import torch
from torch.utils.data import DataLoader
from transformers import MT5ForConditionalGeneration, T5Tokenizer, Adafactor
from accelerate import Accelerator

from data import extract_maintexts_and_titles, T5SummaryDataset
from utils import set_seed, load_config, move_dict_to_device
from valid import generate_summaries, calc_rouge

import wandb

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
    train_set = T5SummaryDataset(train_texts, train_titles, tokenizer, max_target_length=args.max_target_len)
    valid_set = T5SummaryDataset(valid_texts, valid_titles, tokenizer, max_target_length=512) # NOTE: remove the constraint of valid length to effectively validate model

    train_loader = DataLoader(train_set, args.bs // args.grad_accum_steps, shuffle=True, collate_fn=train_set.collate_fn)
    valid_loader = DataLoader(valid_set, args.bs // args.grad_accum_steps, shuffle=False, collate_fn=valid_set.collate_fn)
    print(f"Finish Dataset & DataLoader construction.")

    # Model
    model = MT5ForConditionalGeneration.from_pretrained(args.t5_model).to(args.device)
    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    accelerator = Accelerator(device_placement=False, fp16=args.fp16)
    print("Model & optimizer loaded.")

    # Optimization
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    print(f"Start training: total training steps = {int(len(train_loader) / args.grad_accum_steps * args.nepochs)}")

    train_log = {
        "rouge_1": list(),
        "rouge_2": list(),
        "rouge_L": list(),
        "valid_loss": list(),
        "steps": list()       
    }

    best_metric = float("-inf")
    steps = -1
    for epoch in range(args.nepochs):
        print(f"\n===== Training at epoch {epoch + 1} =====\n")
        train_loss = 0
        for loader_idx, batch in enumerate(train_loader):
            model.train()

            X, y = batch
            X = move_dict_to_device(X, args.device)
            y = move_dict_to_device(y, args.device)

            outputs = model(**X, labels=y["input_ids"])
            loss = outputs.loss / args.grad_accum_steps
            accelerator.backward(loss)

            train_loss += loss.detach().cpu().item()

            if (loader_idx % args.grad_accum_steps == args.grad_accum_steps - 1) or (loader_idx == len(train_loader) - 1):
                optimizer.step()
                optimizer.zero_grad()
                steps += 1
                
                # log train loss
                wandb.log({"train_loss": train_loss})
                train_loss = 0

                # evaluation
                if (steps % args.log_steps == 0) or (loader_idx == len(train_loader) - 1):
                    print(f"Evaluating model at step {steps}...")
                    torch.cuda.empty_cache()
                    valid_loss = calc_valid_loss(valid_loader, model, args)
                    all_preds = generate_summaries(valid_loader, model, tokenizer, args)
                    rouge_1, rouge_2, rouge_L = calc_rouge(all_preds, valid_titles).values()

                    wandb.log({
                        "rouge_1": rouge_1,
                        "rouge_2": rouge_2,
                        "rouge_L": rouge_L,
                        "valid_loss": valid_loss
                    })
                    for k, v in zip(["rouge_1", "rouge_2", "rouge_L", "valid_loss", "steps"], [rouge_1, rouge_2, rouge_L, valid_loss, steps]):
                        train_log[k].append(v)
                    (Path(args.model_save_dir) / "train_log.json").write_text(json.dumps(train_log))

                    print(f"Validation | rouge-1 = {rouge_1:.2f}; rouge-2 = {rouge_2:.2f}; rouge-L = {rouge_L:.2f}; valid_loss = {valid_loss:.4f}")

                    if rouge_2 > best_metric:
                        best_metric = rouge_2
                        model.save_pretrained(args.model_save_dir)
                        print("Best model saved.")
                        (Path(args.model_save_dir) / "best_metric.txt").write_text(str(best_metric))

    wandb.run.summary["best_metric"] = best_metric
    wandb.finish(exit_code=0)
    return

def calc_valid_loss(data_loader: DataLoader, model: MT5ForConditionalGeneration, args: Namespace):
    valid_loss = 0
    model.eval()
    for batch in tqdm(data_loader):
        X, y = batch
        X = move_dict_to_device(X, args.device)
        y = move_dict_to_device(y, args.device)

        with torch.no_grad():
            outputs = model(**X, labels=y["input_ids"])
            valid_loss += outputs.loss.detach().cpu().item() * len(y["input_ids"])
        
    valid_loss /= len(data_loader.dataset)
    return valid_loss

if __name__ == "__main__":
    config = load_config("./config.json")
    args = Namespace(**config)

    wandb_config = {
        "nepochs": args.nepochs,
        "bs": args.bs,
        "max_text_len": args.max_text_len,
        "max_target_len": args.max_target_len,
        "log_steps": args.log_steps
        # "fp16": args.fp16
    }
    exp_name = '__'.join([f"{k}-{v}" for k, v in wandb_config.items()])
    args.model_save_dir = Path(args.save_dir) / exp_name

    wandb.init(
        project="adl-hw3",
        name=exp_name,
        config=wandb_config
    )

    trainer(args)