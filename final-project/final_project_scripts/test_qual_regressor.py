import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import pandas as pd
from typing import List
from argparse import Namespace

from transformers import logging
from simpletransformers.classification import ClassificationModel, ClassificationArgs

logging.set_verbosity_error()

def calc_dial_qual(pred_jsonl: str, field: str):
    # Configuration
    args = Namespace(
        prediction=pred_jsonl,
        model_type="bert",
        model_path=f"./models/{field}/best_model",
        field="relevance",
        device=0
    )

    model_args = ClassificationArgs(
        best_model_dir="./models",
        do_lower_case=True,
        eval_batch_size=16,
        max_seq_length=512,
        output_dir="./models",
        overwrite_output_dir=False,
        regression=True,
        manual_seed=42,
        use_multiprocessing=False,
        use_multiprocessing_for_evaluation=False,
        dataloader_num_workers=1
    )

    # Data
    dials = list()
    with open(args.prediction) as f:
        for line in f:
            dial = '\n'.join(json.loads(line.rstrip())["dialog"])
            dials.append(dial)
    
    # Model
    model = ClassificationModel(
        model_type=args.model_type,
        model_name=args.model_path,
        num_labels=1,
        args=model_args,
        cuda_device=args.device
    )

    # Inference
    res = model.predict(dials)
    score = float(res[1].mean())

    return score

if __name__ == "__main__":
    pred_jsonls = [
        "./outputs/test_dialoGPT-small.jsonl",
        "./outputs/test_dialoGPT-medium.jsonl",
        "./outputs/test_dialoGPT-large.jsonl",
        "/nfs/nas-7.1/ckwu/adl-2022-spring/adl-final-project/kevin/reproduced.jsonl",
        "/nfs/nas-7.1/ckwu/adl-2022-spring/adl-final-project/wlchen/num_generation-3.jsonl",
        "/nfs/nas-7.1/ckwu/adl-2022-spring/adl-final-project/wlchen/num_generation-5.jsonl",
        "/nfs/nas-7.1/ckwu/adl-2022-spring/adl-final-project/wlchen/num_generation-10.jsonl",
        "/nfs/nas-7.1/ckwu/adl-2022-spring/adl-final-project/wlchen/num_generation-20.jsonl"
    ]

    fields = ["relevance", "aggressiveness", "overall"]

    for pred_jsonl in pred_jsonls:
        for field in fields:
            print(f"Evaluating {pred_jsonl}...")
            score = calc_dial_qual(pred_jsonl, field)
            print(f"{field} score = {score:.3f}")