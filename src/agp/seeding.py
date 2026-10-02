from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)


def derive_seed(base_seed: int, *tokens: str) -> int:
    value = base_seed
    for token in tokens:
        for char in token:
            value = (value * 1_000_003 + ord(char)) % (2**31 - 1)
    return value
