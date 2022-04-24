import json
import pandas as pd
from pathlib import Path

from transformers import AutoConfig
from utils import encoder_mappings

def print_config(name: str) -> None:
    config = AutoConfig.from_pretrained(name)
    print(config)

if __name__ == "__main__":
    encoder = "RoBERTaWWMEXT"
    print_config(encoder_mappings[encoder])