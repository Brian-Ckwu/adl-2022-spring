import warnings
warnings.filterwarnings("ignore")

import json
from argparse import Namespace, ArgumentParser

from torch.utils.data import DataLoader
from transformers import MT5ForConditionalGeneration, T5Tokenizer

from data import extract_maintexts, extract_ids, T5SummaryDataset
from utils import set_seed, load_config
from valid import generate_summaries, set_decode_algo

def predict(args: Namespace):
    # Configuration
    set_seed(args.seed)

    # Data
    # load data
    test_ids = extract_ids(args.input_jsonl)
    test_texts = extract_maintexts(args.input_jsonl)
    test_titles = ['' for _ in range(len(test_ids))]
    assert len(test_ids) == len(test_texts) == len(test_titles)
    print(f"Data loaded. Showing examples:\n \
        Test text (size = {len(test_texts)}): {test_texts[0][:50]}... \
    ")

    # tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_save_dir)
    tokenizer.model_max_length = args.max_text_len

    # dataset & dataloader
    test_set = T5SummaryDataset(test_texts, test_titles, tokenizer, max_target_length=512) # NOTE: remove the constraint of valid length to effectively validate model
    test_loader = DataLoader(test_set, args.bs, shuffle=False, pin_memory=True, collate_fn=test_set.collate_fn)
    print(f"Finish Dataset & DataLoader construction.")

    # Model
    model = MT5ForConditionalGeneration.from_pretrained(args.model_save_dir).to(args.device)
    print("Model loaded.")

    # Evaluation
    pred_titles = generate_summaries(test_loader, model, tokenizer, args); assert len(pred_titles) == len(test_titles)

    # Make prediction dicts
    assert len(pred_titles) == len(test_ids)
    pred_dicts = list()
    for pred_title, test_id in zip(pred_titles, test_ids):
        d = {"title": pred_title, "id": test_id}
        pred_dicts.append(d)

    return pred_dicts

def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--input_jsonl",
        type=str,
        help="Path to the testing file.",
        required=True
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        help="Path to the prediction file.",
        required=True
    )

    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    cmd_args = parse_args()

    config = load_config("./config.json")
    args = Namespace(**config)
    args.input_jsonl = cmd_args.input_jsonl
    args.output_jsonl = cmd_args.output_jsonl

    decode_config = load_config("./decode_config.json")
    pred_args = decode_config[args.decode_algo]
    set_decode_algo(args, pred_args)    

    # prediction
    output_dicts = predict(args)

    # write file
    with open(args.output_jsonl, mode="wt", encoding="utf-8") as f:
        for output_dict in output_dicts:
            f.write(json.dumps(output_dict) + '\n')