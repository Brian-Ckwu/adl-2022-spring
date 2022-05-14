# ADL-2022-Spring Homework 3

## Download the Model
    
    bash ./download.sh

The data (just for conveniency of testing the run.sh script) and model would be downloaded and unzipped.

## Generate the Prediction File
    
    bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl

For example, one can run the following command in the current directory to reproduce the public results.
    
    bash ./run.sh ./data/public.jsonl ./prediction.jsonl

## Reproducibility

To reproduce the model, the following command should be executed with internet access in the current directory.

### Train the Model

    python3.8 train.py

The trained model will be saved to `./models/reproduced_model`