import random
import os
import numpy as np
import torch
import json

def seed_all(seed_value, cuda_deterministic=False):
    """Set random seeds for reproducibility across different libraries."""
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def read_json_file(file_path):
    """Read a JSON file and return its contents."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data