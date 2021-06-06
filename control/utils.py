import random
import numpy as np
import logging
import torch
from transformers import (
is_torch_available,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        # torch.backends.cudnn.deterministic = True # might effect performance
        # torch.backends.cudnn.benchmark = False

def write_sent(sents, path):
    with open(path, 'w', encoding='utf-8') as f:
        for s in sents:
            f.write(s + '\n')

def clean_text(text):
    return text.strip().replace("<s>", "").replace("<|endoftext|>", "").replace("<unk>", "")