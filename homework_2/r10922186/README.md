# ADL-2022-Spring Homework 2

## Download Models
    
    bash ./download.sh

The data (just for conveniency of testing the run.sh script) and models would be downloaded and unzipped.

## Generate the Prediction File
    
    bash ./run.sh /path/to/context.json /path/to/test.json  /path/to/pred/prediction.csv

For example, one can run the following command in the current directory to reproduce the inference results.
    
    bash ./run.sh ./data/context.json ./data/test.json ./prediction.csv

## Reproducibility

The following steps to reproduce the context selection (CS) and question answering (QA) models should be executed with internet access in the current directory.

### Train the Context Selection Model

    python3.8 train_cs.py

The trained CS model will be saved to `./models/cs/encoder-RoBERTaWWMEXT_bs-16_optimizer-Adam_lr-2e-05/best_model.pth`

### Train the Question Answering Model

    python3.8 train_qa.py

The trained QA model will be saved to `./models/qa/encoder-RoBERTaWWMEXT_bs-16_optimizer-Adam_lr-2e-05/best_model.pth`