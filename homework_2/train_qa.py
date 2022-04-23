"""
    Reference: Huggingface example code
    https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py
"""

"""
    Import Packages
"""

import json
import logging
import math
from pathlib import Path
from argparse import Namespace

import torch
from datasets import Dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator
)

from utils_qa import QAPreprocessor, construct_raw_qa_dataset, post_processing_function, create_and_fill_np_array, postprocess_qa_predictions
from utils import set_seed, render_exp_name, move_batch_to_device, encoder_mappings

"""
    Configuration
"""
args = Namespace(**json.loads(Path("./qa_config.json").read_bytes()))
args.exp_name = render_exp_name(args, fields=["encoder", "bs", "optimizer", "lr"])
args.save_path = Path(args.output_dir) / args.exp_name
args.save_path.mkdir(parents=True, exist_ok=True)

args.encoder = encoder_mappings[args.encoder]
args.tokenizer_name = encoder_mappings[args.tokenizer_name]

logger = logging.getLogger(__name__)
logging.basicConfig(
    format=r"%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt=r"%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.setLevel(logging.INFO)

set_seed(args.seed)

"""
    Data
"""
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)    
assert args.max_length <= tokenizer.model_max_length

train_dict_l = json.loads(Path(args.train_file).read_bytes())
valid_dict_l = json.loads(Path(args.validation_file).read_bytes())
context_l = json.loads(Path(args.context_file).read_bytes())

raw_train_set_d = construct_raw_qa_dataset(train_dict_l, context_l, mode="train")
raw_valid_set_d = construct_raw_qa_dataset(valid_dict_l, context_l, mode="valid")

raw_train_set = Dataset.from_dict(raw_train_set_d)
raw_valid_set = Dataset.from_dict(raw_valid_set_d)

qa_preprocessor = QAPreprocessor(tokenizer, args)

# Dataset
train_set = raw_train_set.map(
    qa_preprocessor.prepare_train_features,
    batched=True,
    remove_columns=raw_train_set.column_names,
    desc="Running tokenizer on train dataset"
)

valid_set = raw_valid_set.map(
    qa_preprocessor.prepare_validation_features,
    batched=True,
    remove_columns=raw_valid_set.column_names,
    desc="Running tokenizer on valid dataset"
)
valid_set_for_model = valid_set.remove_columns(["example_id", "offset_mapping"])

valid_set_for_loss = raw_valid_set.map(
    qa_preprocessor.prepare_train_features,
    batched=True,
    remove_columns=raw_valid_set.column_names,
    desc="Running tokenizer on valid dataset for calculating loss"
)

# DataLoader
data_collator = default_data_collator

train_loader = DataLoader(train_set, batch_size=args.bs // args.grad_accum_steps, shuffle=True, collate_fn=data_collator)
valid_loader = DataLoader(valid_set_for_model, batch_size=args.bs // args.grad_accum_steps, shuffle=False, collate_fn=data_collator)
valid_loader_for_loss = DataLoader(valid_set_for_loss, batch_size=args.bs // args.grad_accum_steps, shuffle=False, collate_fn=data_collator)

"""
    Model, Optimizer, and Metric
"""
# TODO: encoder_mappings / train from scratch
config = AutoConfig.from_pretrained(args.encoder)
model = AutoModelForQuestionAnswering.from_pretrained(
    args.encoder,
    from_tf=bool(".ckpt" in args.encoder),
    config=config,
)
assert not args.version_2_with_negative
optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)
metric = load_metric("squad")

"""
    Optimization
"""
model = model.to(args.device)

total_batch_size = args.bs
num_update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
args.max_train_steps = args.nepochs * num_update_steps_per_epoch

logger.info("***** Running training *****")
logger.info(f"  Num examples = {len(train_set)}")
logger.info(f"  Num Epochs = {args.nepochs}")
logger.info(f"  Total train batch size = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {args.grad_accum_steps}")
logger.info(f"  Total optimization steps = {args.max_train_steps}")

progress_bar = tqdm(range(args.max_train_steps))
best_eval_metric = 0
completed_steps = 0
train_log = {
    "valid_loss": list(),
    "valid_EM": list(),
    "steps": list()
}

for epoch in range(1, args.nepochs + 1):
    for step, batch in enumerate(train_loader):
        # Training
        model.train()
        batch = move_batch_to_device(batch, args.device)
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / args.grad_accum_steps
        loss.backward()

        if (step % args.grad_accum_steps == 0) or (step == len(train_loader) - 1):
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1
        
        # Evaluation
        if (step % args.ckpt_steps == 0) or (step == len(train_loader) - 1):
            model.eval()
            all_start_logits = []
            all_end_logits = []
            total_loss = 0

            if args.calc_valid_loss:
                for batch_for_loss in tqdm(valid_loader_for_loss):
                    batch_for_loss = move_batch_to_device(batch, args.device)
                    with torch.no_grad():
                        outputs_for_loss = model(**batch_for_loss)
                        total_loss += outputs_for_loss.loss.detach().cpu().item()
                mean_total_loss = total_loss / len(valid_loader_for_loss)
                logger.info(f"\nValidation Loss: {mean_total_loss}\n")

            for batch in tqdm(valid_loader):
                batch = move_batch_to_device(batch, args.device)

                with torch.no_grad():
                    outputs = model(**batch)

                    start_logits = outputs.start_logits
                    end_logits = outputs.end_logits

                    all_start_logits.append(start_logits.cpu().numpy())
                    all_end_logits.append(end_logits.cpu().numpy())

            max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

            # concatenate the numpy array
            start_logits_concat = create_and_fill_np_array(all_start_logits, valid_set, max_len)
            end_logits_concat = create_and_fill_np_array(all_end_logits, valid_set, max_len)

            # delete the list of numpy arrays
            del all_start_logits
            del all_end_logits

            outputs_numpy = (start_logits_concat, end_logits_concat)
            prediction = post_processing_function(raw_valid_set, valid_set, outputs_numpy, args)
            eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)["exact_match"]
            logger.info(f"\nExact Match (EM): {eval_metric}\n")

            if eval_metric > best_eval_metric:
                best_eval_metric = eval_metric
                torch.save(model.state_dict(), args.save_path / "best_model.pth")
                (args.save_path / "best_metric.txt").write_text(str(best_eval_metric))
                print("Best model saved.")
            
            # Record training log
            train_log["valid_loss"].append(mean_total_loss)
            train_log["valid_EM"].append(eval_metric)
            train_log["steps"].append(completed_steps)
            (args.save_path / "train_log.json").write_text(json.dumps(train_log))

progress_bar.close()