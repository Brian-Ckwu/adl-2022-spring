import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from argparse import ArgumentParser, Namespace

import pandas as pd
from transformers import logging
from simpletransformers.classification import ClassificationModel, ClassificationArgs

logging.set_verbosity_error()

def parse_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "--target_score",
        type=str
    )
    arg_parser.add_argument(
        "--model_type",
        type=str
    )
    arg_parser.add_argument(
        "--model_name",
        type=str
    )
    arg_parser.add_argument(
        "--lowercase",
        type=bool
    )
    arg_parser.add_argument(
        "--device",
        type=int
    )

    args = arg_parser.parse_args()
    return args

cmd_args = parse_args()
print(vars(cmd_args))

# Configuration
FIELD = cmd_args.target_score
DEVICE = cmd_args.device

args = ClassificationArgs()

args.best_model_dir = f"./models/{FIELD}_{cmd_args.model_name}/best_model"
args.do_lower_case = cmd_args.lowercase
args.eval_batch_size = 16
args.evaluate_during_training = True
args.evaluate_during_training_steps = 100
args.evaluate_during_training_verbose = True
args.learning_rate = 3e-5
args.logging_steps = 1
args.manual_seed = 42
args.max_seq_length = 512
args.num_train_epochs = 4
args.optimizer = "AdamW"
args.output_dir = f"./models/{FIELD}_{cmd_args.model_name}"
args.overwrite_output_dir = True
args.save_steps = -1
args.train_batch_size = 16
args.wandb_project = f"adl-final-regressor"
args.wandb_kwargs = {
    "name": f"{FIELD}_{cmd_args.model_name}"
}
args.regression = True
args.use_multiprocessing = False
args.use_multiprocessing_for_evaluation	= False
args.dataloader_num_workers = 1

# Model
model = ClassificationModel(
    model_type=cmd_args.model_type,
    model_name=cmd_args.model_name,
    num_labels=1,
    args=args,
    cuda_device=DEVICE
)

# Data
task1_df = pd.read_csv("./data/salesbot_data/preprocessed/task1_annots.tsv", sep='\t', index_col="id")

total_df = task1_df[["text", FIELD]]
total_df.columns = ["text", "labels"]
train_df = total_df.sample(frac=0.8, replace=False)
valid_df = total_df.drop(labels=train_df.index)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

# Optimization
model.train_model(
    train_df=train_df,
    eval_df=valid_df
)