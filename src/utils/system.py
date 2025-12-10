# ===================================================================
# src/utils/system.py
# ===================================================================

import subprocess
import torch
import os
from math import floor
from typing import Optional
import logging
from src.config.core import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_system_info():
    """Captures critical system and software information."""

    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    except Exception:
        git_hash = "N/A (git not found or not a git repository)"

    pip_freeze = subprocess.check_output(['pip', 'freeze']).strip().decode('utf-8')

    return {
        "git_commit_hash": git_hash,
        "python_environment": pip_freeze,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu_info": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }

def is_on_jean_zay() -> bool:
    """
    Detects if the script is running on the Jean Zay cluster.
    Based on the presence of 'jean-zay' in the HOSTNAME environment variable.
    Returns True if so, False otherwise.
    """
    hostname = os.environ.get('HOSTNAME', '').lower()
    return 'jean-zay' in hostname

def get_optimal_cpu_count() -> int:
    """
    Calculates the optimal number of CPU cores to use for multiprocessing.

    The strategy is as follows:
    1. Uses the total number of available cores.
    2. If more than one core, takes approximately 61.8% (golden ratio) of the total.
    3. Ensures at least 1 core is always returned.
    4. Allows capping the number of cores via the `max_cpu` parameter.

    Args:
        max_cpu (Optional[int]): The maximum number of cores to consider. 
                                 If None, uses all available cores.

    Returns:
        int: The number of processes to launch.
    """
    # if is_on_jean_zay():
    #     logging.warning("Jean Zay environment detection. Using only fixed cores to allocate only one compute node.")
    #     return 24

    max_cpu = config.tokenization.max_parallel_cpu

    # Determine base number of cores
    available_cpus = os.cpu_count()
    if available_cpus is None:
        return 1 # Safety fallback

    # Apply reduction strategy
    if available_cpus > 1:
        num_proc = int(floor(available_cpus * 0.618))
        if num_proc == 0:
            num_proc = 1
    else:
        num_proc = 1

    if max_cpu is not None:
        num_proc = min(num_proc, max_cpu)
        
    return num_proc

