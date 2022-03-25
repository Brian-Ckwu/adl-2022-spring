# ADL 2022 Spring Homework

## Download models
    bash ./download.sh

Cache, data, and models will be saved in respective directories.

## Predict on Test Data
    bash ./intent_cls.sh ./data/intent/test.json ./intent.csv
    bash ./slot_tag.sh ./data/slot/test.json ./slot.csv

The prediction files will be stored at ./intent.csv & ./slot.csv

## Reproducibility
    python3.8 train_intent.py
    python3.8 train_slot.py

The above terminal commands should be able to reproduce the models, which will be stored in ./models