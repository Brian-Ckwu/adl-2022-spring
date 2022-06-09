from argparse import Namespace

from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs
)

model_type = "gpt2"
model_size = "medium"
model_name = f"microsoft/DialoGPT-{model_size}"
data_class = "simple"
device = 0

model_args = LanguageModelingArgs(
    output_dir=f"./models/dialo-GPT/{model_size}-{data_class}/",
    best_model_dir=f"./models/dialo-GPT/{model_size}-{data_class}/best_model",
    dataloader_num_workers=1,
    process_count=1,
    evaluate_during_training=False,
    max_seq_length=512,
    num_train_epochs=3,
    reprocess_input_data=True,
    save_model_every_epoch=True,
    use_multiprocessing=False,
    train_batch_size=1,
    gradient_accumulation_steps=4,
    wandb_project="adl-final-dialoGPT",
    wandb_kwargs={
        "name": f"{model_size}-{data_class}"
    },
    overwrite_output_dir=True,
    clean_text=True,
    dataset_type=data_class,
    mlm=False,
    tokenizer_name=model_name
)
model_args.special_tokens.append("<|endoftext|>")

model = LanguageModelingModel(model_type, model_name, args=model_args, cuda_device=device)

model.train_model(train_file="./data/gpt_fine_tuning.txt", show_running_loss=True)

