# src/utils/determinism.py

import torch
import sys
import logging
from transformers import set_seed
from src.config.core import config
import random
import numpy as np

def set_absolute_determinism():
    """
    Locks all possible sources of randomness for
    maximum reproducibility.
    """
    random.seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)

    # 1. Base seed (Python, Numpy, Transformers)
    set_seed(config.experiment.seed)
    
    # 2. PyTorch seeds
    torch.manual_seed(config.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.experiment.seed)
    
    # 3. Directives for CUDA algorithms (the most critical point)
    # Forces the use of deterministic algorithms. May slow down training.
    torch.backends.cudnn.deterministic = True
    # Disables benchmarking which can select non-deterministic algos.
    torch.backends.cudnn.benchmark = False
    
    # 4. OS-specific warning
    if sys.platform == "darwin":
        logging.warning(
            "Exécution sur macOS ('darwin'). Le backend MPS de PyTorch ne garantit pas "
            "le déterminisme pour toutes les opérations. La reproductibilité bit à bit "
            "pourrait ne pas être atteinte sur cette plateforme."
        )
    else:
        logging.info(f"Stabilité absolue activée. Graine aléatoire fixée à : {config.experiment.seed}")
