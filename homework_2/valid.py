import json
from pathlib import Path
from argparse import Namespace, ArgumentParser

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    default_data_collator
)

from utils import Preprocessor, set_seed, render_exp_name, construct_raw_dataset, move_batch_to_device, encoder_mappings, convert_context_choices_to_context_indices, prepare_e2e_data_dict_l, calc_e2e_em
from utils_qa import QAPreprocessor, construct_raw_qa_dataset, post_processing_function, create_and_fill_np_array

def validate(args: Namespace):
    # Load data
    valid_dict_l = json.loads(Path(args.validation_file).read_bytes())
    context_l = json.loads(Path(args.context_file).read_bytes())

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_mappings[args.tokenizer_name])

    # Context Selection
    # data
    raw_valid_set_d = construct_raw_dataset(valid_dict_l, context_l, mode="valid")
    raw_valid_set = Dataset.from_dict(raw_valid_set_d)

    preprocessor = Preprocessor(tokenizer, args)
    valid_set = raw_valid_set.map(
        preprocessor, batched=True, remove_columns=raw_valid_set.column_names
    )

    data_collator = default_data_collator
    valid_loader = DataLoader(valid_set, batch_size=args.bs, shuffle=False, collate_fn=data_collator)
    
    # model
    cs_model = AutoModelForMultipleChoice.from_pretrained(encoder_mappings[args.encoder])
    cs_model.load_state_dict(torch.load(args.cs_save_path / "best_model.pth", map_location=args.device))

    # predict context
    pred_choices = list()
    cs_model = cs_model.to(args.device)
    cs_model.eval()
    for batch in tqdm(valid_loader):
        batch = move_batch_to_device(batch, args.device)
        with torch.no_grad():
            outputs = cs_model(**batch)
        preds = outputs.logits.argmax(dim=-1).detach().cpu().tolist()
        pred_choices += preds
    pred_context_indices = convert_context_choices_to_context_indices(pred_choices, valid_dict_l)

    # Question Answering
    # data
    e2e_valid_dict_l = prepare_e2e_data_dict_l(valid_dict_l, pred_context_indices)
    e2e_qa_raw_valid_set_d = construct_raw_qa_dataset(e2e_valid_dict_l, context_l, mode="test")
    e2e_qa_raw_valid_set = Dataset.from_dict(e2e_qa_raw_valid_set_d)

    qa_preprocessor = QAPreprocessor(tokenizer, args)
    e2e_qa_valid_set = e2e_qa_raw_valid_set.map(
        qa_preprocessor.prepare_validation_features,
        batched=True,
        remove_columns=e2e_qa_raw_valid_set.column_names,
        desc="Running tokenizer on valid dataset"
    )
    e2e_qa_valid_set_for_model = e2e_qa_valid_set.remove_columns(["example_id", "offset_mapping"])

    data_collator = default_data_collator
    e2e_qa_valid_loader = DataLoader(e2e_qa_valid_set_for_model, batch_size=args.bs, shuffle=False, collate_fn=data_collator)

    # model
    qa_model = AutoModelForQuestionAnswering.from_pretrained(encoder_mappings[args.encoder])
    qa_model.load_state_dict(torch.load(args.qa_save_path / "best_model.pth", map_location=args.device))

    # predict answer
    qa_model = qa_model.to(args.device)
    qa_model.eval()
    all_start_logits = []
    all_end_logits = []
    for batch in tqdm(e2e_qa_valid_loader):
        batch = move_batch_to_device(batch, args.device)
        with torch.no_grad():
            outputs = qa_model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            all_start_logits.append(start_logits.cpu().numpy())
            all_end_logits.append(end_logits.cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, e2e_qa_valid_set, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, e2e_qa_valid_set, max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    args.output_dir = Path(args.model_dir) / "qa" # NOTE: add output_dir for compatibility
    prediction = post_processing_function(e2e_qa_raw_valid_set, e2e_qa_valid_set, outputs_numpy, args)
    em = calc_e2e_em(prediction.predictions, valid_dict_l)
    print(f"Exact Match (EM): {em}")

    return em

def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--context_file",
        type=str,
        help="Path to the context file"
    )
    parser.add_argument(
        "--valid_file",
        type=str,
        help="Path to the valid file"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to the model directory"
    )

    args_from_cmd = parser.parse_args()
    return args_from_cmd

if __name__ == "__main__":
    args_from_cmd = parse_args()

    args = Namespace(**json.loads(Path("./eval_config.json").read_bytes()))
    args.context_file = args_from_cmd.context_file
    args.validation_file = args_from_cmd.valid_file
    args.model_dir = args_from_cmd.model_dir

    args.exp_name = render_exp_name(args, fields=["encoder", "bs", "optimizer", "lr"])
    args.cs_save_path = Path(args.model_dir) / "context_selection" / args.exp_name
    args.qa_save_path = Path(args.model_dir) / "qa" / args.exp_name

    set_seed(args.seed)
    validate(args)