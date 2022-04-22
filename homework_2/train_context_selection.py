"""
    Import Packages
"""
import json
import math
import logging
from tqdm.auto import tqdm
from pathlib import Path
from argparse import Namespace

import torch
from torch.utils.data import DataLoader

from datasets import load_metric, Dataset

from transformers import AutoConfig, AutoTokenizer, AutoModelForMultipleChoice, default_data_collator

from utils import encoder_mappings, set_seed, render_exp_name, construct_raw_dataset, move_batch_to_device, Preprocessor

"""
    Configuration
"""
args = Namespace(**json.loads(Path("./context_selection_config.json").read_bytes()))
args.exp_name = render_exp_name(args, fields=["encoder", "nepochs", "bs", "optimizer", "lr"])
args.save_path = Path(args.output_dir) / args.exp_name
args.save_path.mkdir(parents=True, exist_ok=True)

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
# Loading
train_dict_l = json.loads(Path(args.train_file).read_bytes())
valid_dict_l = json.loads(Path(args.validation_file).read_bytes())
context_l = json.loads(Path(args.context_file).read_bytes())

raw_train_set_d = construct_raw_dataset(train_dict_l, context_l, mode="train")
raw_valid_set_d = construct_raw_dataset(valid_dict_l, context_l, mode="valid")

raw_train_set = Dataset.from_dict(raw_train_set_d)
raw_valid_set = Dataset.from_dict(raw_valid_set_d)

# Preprocessing
tokenizer = AutoTokenizer.from_pretrained(encoder_mappings[args.tokenizer_name])
preprocessor = Preprocessor(tokenizer, args)

train_set = raw_train_set.map(
    preprocessor, batched=True, remove_columns=raw_train_set.column_names
)
valid_set = raw_valid_set.map(
    preprocessor, batched=True, remove_columns=raw_valid_set.column_names
)

# Make DataLoader
data_collator = default_data_collator
train_loader = DataLoader(train_set, batch_size=args.bs // args.grad_accum_steps, shuffle=True, collate_fn=data_collator)
valid_loader = DataLoader(valid_set, batch_size=args.bs // args.grad_accum_steps, shuffle=False, collate_fn=data_collator)

"""
    Model, Optimizer, and Metric
"""

config = AutoConfig.from_pretrained(encoder_mappings[args.encoder])

if args.encoder:
    model = AutoModelForMultipleChoice.from_pretrained(
        encoder_mappings[args.encoder],
        from_tf=bool(".ckpt" in encoder_mappings[args.encoder]), # tf == tensorflow
        config=config,
    )
else:
    logger.info("Training new model from scratch (no pre-trained weights)")
    model = AutoModelForMultipleChoice.from_config(config)
    
model.resize_token_embeddings(len(tokenizer))
model = model.to(args.device)

optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)
metric = load_metric("accuracy")

"""
    Optimization 
"""
# Logging
total_batch_size = args.bs
num_update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
args.max_train_steps = args.nepochs * num_update_steps_per_epoch

logger.info("***** Running training *****")
logger.info(f"  Num examples = {len(train_set)}")
logger.info(f"  Num Epochs = {args.nepochs}")
logger.info(f"  Total train batch size = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {args.grad_accum_steps}")
logger.info(f"  Total optimization steps = {args.max_train_steps}")

# Optimization Loop
progress_bar = tqdm(range(args.max_train_steps))
best_eval_metric = 0

for epoch in range(1, args.nepochs + 1):
    total_loss = 0
    for step, batch in enumerate(train_loader):
        model.train()
        batch = move_batch_to_device(batch, args.device)
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().cpu().float()
        loss = loss / args.grad_accum_steps
        loss.backward()
        
        if (step % args.grad_accum_steps == 0) or (step == len(train_loader) - 1):
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        if (step % args.ckpt_steps == 0) or (step == len(train_loader) - 1):
            model.eval()
            for batch in tqdm(valid_loader):
                batch = move_batch_to_device(batch, args.device)
                with torch.no_grad():
                    outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                metric.add_batch(
                    predictions=preds,
                    references=batch["labels"]
                )
    
            eval_metric = metric.compute()["accuracy"]
            logger.info(f"\nStep {step // args.grad_accum_steps}: acc = {eval_metric}")

            # save the best model
            if eval_metric > best_eval_metric:
                best_eval_metric = eval_metric
                torch.save(model.state_dict(), args.save_path / "best_model.pth")
                print("Best model saved.")

progress_bar.close()