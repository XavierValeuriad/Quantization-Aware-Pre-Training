# ===================================================================
# src/utils/statistics.py
#
# Statistical calculation module for rigorous split generation.
# Handles Sample Size Determination and post-hoc power estimation.
# ===================================================================

import math
import logging
from typing import Dict, Tuple
from pathlib import Path
import json
from src.config.core import config
from src.utils.path_manager import get_results_root

logger = logging.getLogger(__name__)

def calculate_required_sample_size(sigma: float, relative_error: float) -> int:
    """
    Calculates the required sample size for a given confidence level (Sigma)
    and relative error margin.
    Simplified formula: n = (Z / k)^2
    """
    if relative_error <= 0:
        raise ValueError("Relative error factor k must be > 0")
    return math.ceil((sigma / relative_error) ** 2)

def calculate_statistical_power(n: int, relative_error: float) -> float:
    """
    Calculates the achieved Z-score (Sigma) for a given size n and margin k.
    Z = sqrt(n) * k
    """
    if n <= 0: return 0.0
    return math.sqrt(n) * relative_error

def determine_split_sizes(
    total_samples: int,
    sigma: float,
    relative_error: float,
    fallback_ratios: Dict[str, float]
) -> Tuple[int, int, int, bool]:
    """
    Determines absolute sizes for Train, Validation, Test.
    
    Returns:
        (n_train, n_validation, n_test, is_statistically_significant)
    """
    n_required_per_eval_split = calculate_required_sample_size(sigma, relative_error)
    
    # Total needed for Val + Test (we want rigor on both if possible)
    n_eval_total_needed = n_required_per_eval_split * 2
    
    # We want to keep a majority for train (arbitrarily > 50% of total)
    # or at least a reasonable minimum size.
    # Here, we simply check if the dataset is large enough to satisfy Eval+Test
    # while leaving crumbs for Train.
    
    is_statistically_significant = True
    
    if total_samples > (n_eval_total_needed * 1.2): # 20% margin for minimum train
        # Case 1: Sufficient Dataset
        n_test = n_required_per_eval_split
        n_val = n_required_per_eval_split
        n_train = total_samples - n_val - n_test
        logger.info(f"Statistical Rigor ({sigma}σ) achieved. Splits set by axiom.")
        
    else:
        # Case 2: Dataset Too Small -> Fallback Ratios
        is_statistically_significant = False
        logger.warning(
            f"Dataset insufficient for rigor {sigma}σ (Required: {n_eval_total_needed}, Available: {total_samples}). "
            f"Switching to fallback ratios."
        )
        
        n_train = int(total_samples * fallback_ratios['train_ratio'])
        n_val = int(total_samples * fallback_ratios['validation_ratio'])
        # The rest goes to test to avoid rounding losses
        n_test = total_samples - n_train - n_val
        
        # Safety for microscopic datasets (< 3 examples)
        if n_train == 0 and total_samples > 0: n_train = total_samples; n_val=0; n_test=0

    return n_train, n_val, n_test, is_statistically_significant

def report_dataset_statistics(
    dataset_name: str,
    config_name: str,
    total_samples: int,
    splits: Dict[str, int], # {'train': N, 'validation': N, ...}
    target_sigma: float,
    target_k: float
):
    """
    Generates a JSON report on the statistical power achieved for this dataset.
    Saves to data/reports/statistics/
    """
    report_dir = get_results_root() / "statistics"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    safe_name = dataset_name.replace("/", "@")
    filename = f"{safe_name}_{config_name or 'default'}_stats.json"
    
    # Post-Hoc Calculation
    stats = {
        "dataset": dataset_name,
        "config": config_name,
        "total_samples": total_samples,
        "target_metrics": {
            "target_sigma": target_sigma,
            "target_k": target_k
        },
        "splits": {}
    }
    
    for split_name, n_samples in splits.items():
        # We calculate the Sigma achieved for this split with the target k
        achieved_sigma = calculate_statistical_power(n_samples, target_k)
        
        stats["splits"][split_name] = {
            "samples": n_samples,
            "ratio": round(n_samples / total_samples, 4) if total_samples > 0 else 0,
            "achieved_sigma": round(achieved_sigma, 4),
            "is_target_reached": achieved_sigma >= target_sigma if split_name in ['validation', 'test'] else None
        }
        
    with open(report_dir / filename, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)
        
    logger.info(f"Statistical report saved: {report_dir / filename}")