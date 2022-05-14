import json
import random
import numpy as np
from regex import D
import torch
from typing import Dict

from pathlib import Path

def load_config(file_path: str):
    return json.loads(Path(file_path).read_bytes())

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def move_dict_to_device(d: Dict, device: str) -> Dict:
    for k in d.keys():
        d[k] = d[k].to(device)
    return d