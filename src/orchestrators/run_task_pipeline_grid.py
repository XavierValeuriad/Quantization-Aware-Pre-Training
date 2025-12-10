"""
===================================================================
src/orchestrators/run_local_finetuning_grid.py
v1.0 : Local Grid Orchestrator for Finetuning & Eval.
Iterates sequentially over all combinations defined in config.yml.
===================================================================
"""

import logging
from src.utils.determinism import set_absolute_determinism
from src.config.core import config
from src.workers.pipeliner import (
    run_finetune_eval_pair,
    find_config_objects
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_local_finetuning_grid():
    logger.info("--- Launching FINETUNING + EVAL grid (Local / Sequential) ---")
    
    # --- DETERMINISM LOCKING ---
    set_absolute_determinism()

    # --- 1. Load Scope from Config ---
    # We use the loaded configuration (prod or dev) as the source of truth
    models_to_test = config.models
    quantization_schemes = [q for q in config.quantization if q.enabled]
    datasets_to_test = config.finetuning_datasets  # List of objects {id, configs, tokenizers}
    alpha_sweep = config.experiment.alpha_sweep

    logger.info(
        f"Models: {len(models_to_test)} | "
        f"Quants: {len(quantization_schemes)} | "
        f"Datasets: {len(datasets_to_test)} | "
        f"Alphas: {len(alpha_sweep)}"
    )

    # --- 2. Nested Loops (Grid Traversal) ---
    # Order: Quant -> Dataset -> Config -> Model -> Alpha
    # (Arbitrary order, but meant to keep heavy model loading grouped if possible;
    #  here we switch models often, so to optimize GPU loading we could invert this,
    #  but for clarity we follow the "Campaign" logic).

    for quant_scheme_cfg in quantization_schemes:
        quant_name = quant_scheme_cfg.name
        
        for dataset_obj in datasets_to_test:
            dataset_id = dataset_obj.id
            
            # A dataset may have multiple configs (e.g., "emea", "medline")
            for dataset_config_name in dataset_obj.configs:
                
                for model_id in models_to_test:
                    
                    # --- Preparation of Context ---
                    # Retrieve complete configuration objects via the pipeliner helper.
                    # This allows validating that everything exists before launching.
                    try:
                        quant_scheme, dataset_info = find_config_objects(
                            quant_name, 
                            dataset_id, 
                            dataset_config_name
                        )
                    except Exception as e:
                        logger.error(f"SKIPPING task due to config error: {e}")
                        continue

                    logger.info(f"############# NEW CONTEXT: {model_id} | {quant_name} | {dataset_id}/{dataset_config_name} #############")

                    # --- 3. Alpha Loop (Sequential) ---
                    for alpha in alpha_sweep:
                        log_prefix = f"{model_id}|{quant_name}|{dataset_id}/{dataset_config_name}|alpha={alpha:.2f}"
                        
                        try:
                            # Direct call to the worker (without using env variables)
                            run_finetune_eval_pair(
                                model_id=model_id,
                                quant_scheme=quant_scheme,
                                dataset_info=dataset_info,
                                dataset_config_name=dataset_config_name,
                                alpha=alpha,
                                log_prefix=log_prefix
                            )
                        
                        except Exception as e:
                            # Catch the error to avoid stopping the entire grid if one run fails
                            logger.error(f"CRITICAL FAILURE for {log_prefix}: {e}")
                            # Optional: continue or break depending on desired severity

    logger.info("--- Finetuning & Evaluation grid completed successfully ---")


if __name__ == "__main__":
    run_local_finetuning_grid()