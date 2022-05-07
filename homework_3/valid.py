import warnings
warnings.filterwarnings("ignore")

from tqdm.auto import tqdm
from typing import List
from pathlib import Path
from argparse import Namespace

import torch
from torch.utils.data import DataLoader
from transformers import MT5ForConditionalGeneration, T5Tokenizer, Adafactor

from data import extract_maintexts_and_titles, T5SummaryDataset
from utils import set_seed, load_config, move_dict_to_device

def validate(args: Namespace):
    # Configuration
    set_seed(args.seed)

    # Data
    # load data
    valid_texts, valid_titles = extract_maintexts_and_titles(args.valid_jsonl); assert len(valid_texts) == len(valid_titles)
    print(f"Data loaded. Showing examples:\n \
        Valid text (size = {len(valid_texts)}): {valid_texts[0][:50]}..., title: {valid_titles[0]} \
    ")

    # tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
    tokenizer.model_max_length = args.max_text_len

    # dataset & dataloader
    valid_set = T5SummaryDataset(valid_texts, valid_titles, tokenizer, max_target_length=512) # NOTE: remove the constraint of valid length to effectively validate model
    valid_loader = DataLoader(valid_set, args.bs, shuffle=False, pin_memory=True, collate_fn=valid_set.collate_fn)
    print(f"Finish Dataset & DataLoader construction.")

    # Model
    model = MT5ForConditionalGeneration.from_pretrained(args.model_save_dir).to(args.device)
    print("Model loaded.")

    # Evaluation
    pred_titles = generate_summaries(valid_loader, model, tokenizer, args); assert len(pred_titles) == len(valid_titles)
    rouges = calc_rouge(pred_titles, valid_titles)

    return rouges
    
def generate_summaries(data_loader: DataLoader, model: MT5ForConditionalGeneration, tokenizer: T5Tokenizer, args: Namespace):
    all_outputs = list()

    model.eval()
    for batch in tqdm(data_loader):
        X, y = batch
        X = move_dict_to_device(X, args.device)
        y = move_dict_to_device(y, args.device)

        with torch.no_grad():
            raw_outputs = model.generate(
                X["input_ids"],
                **args.pred_args
            )
            outputs = tokenizer.batch_decode(raw_outputs, skip_special_tokens=True)
        all_outputs += outputs

    return all_outputs

def calc_rouge(preds: List[str], labels: List[str]):
    from twrouge import get_rouge
    rouges = get_rouge(preds, labels)
    arranged_rouges = {
        "rouge-1": rouges["rouge-1"]['f'] * 100,
        "rouge-2": rouges["rouge-2"]['f'] * 100,
        "rouge-L": rouges["rouge-l"]['f'] * 100
    }
    return arranged_rouges

if __name__ == "__main__":
    config = load_config("./config.json")
    args = Namespace(**config)

    rouge_scores = validate(args)
    print(rouge_scores)