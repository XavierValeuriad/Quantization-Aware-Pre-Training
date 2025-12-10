# ===================================================================
# src/orchestrators/run_alpha_grouped_task_pipeline.py
#
# v1.0 : Single Task Orchestrator (M x Q x D) with Alpha Groups.
#        - Reads ALPHA_GROUP.
#        - INTERNAL Loop over alphas in the group.
#        - Chains FT -> Eval for each alpha in the group.
#        - NO INTEGRATED REPORTING (PLOTTING).
# ===================================================================

import logging
import os
from typing import List, Optional

from src.utils.determinism import set_absolute_determinism

from src.workers.pipeliner import (
    get_task_parameters,
    find_config_objects,
    run_finetune_eval_pair,
    PipelineConfigError
)

# Base logging configuration for this orchestrator
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use a specific logger

def run_alpha_grouped_pipeline_no_report():
    """
    Executes FT->Eval for a specific GROUP of alphas for a task (M x Q x D).
    No plot is generated at the end.
    """
    target_alphas: List[float] = []
    try:
        # --- 1. Retrieve Task Parameters (M x Q x D) and Alpha Group ---
        model_id, quant_name, dataset_id, dataset_config_name, log_prefix_base = get_task_parameters()

        alpha_group_str = os.environ.get('ALPHA_GROUP')
        if not alpha_group_str:
            raise PipelineConfigError("Environment variable ALPHA_GROUP missing.")

        # Parse alpha group
        try:
            target_alphas = [float(a.strip()) for a in alpha_group_str.split(';') if a.strip()]
            if not target_alphas:
                 raise ValueError("Alpha list empty after parsing.")
            logger.info(f"Alpha group to process: {target_alphas} for {log_prefix_base}")
        except ValueError as e:
            raise PipelineConfigError(f"Invalid format for ALPHA_GROUP='{alpha_group_str}'. Expected: '0.1;0.9' or '0.5'. Error: {e}")

        # --- 2. Locking & Config Object Retrieval ---
        set_absolute_determinism()
        quant_scheme, dataset_info = find_config_objects(quant_name, dataset_id, dataset_config_name)

        for alpha in target_alphas:
            current_log_prefix = f"{log_prefix_base}|alpha={alpha:.2f}"
            logger.info(f"\n===== [ALPHA={alpha:.2f} of group {alpha_group_str}] =====")

            current_dataset_info = dataset_info # Use the found object

            # --- 3.1 Execute FT -> Eval Pair ---
            # We call the function but do not need to store its results
            # for plotting. The function already handles its own error logging.
            run_finetune_eval_pair(
                model_id=model_id,
                quant_scheme=quant_scheme,
                dataset_info=current_dataset_info,
                dataset_config_name=dataset_config_name, 
                alpha=alpha,
                log_prefix=current_log_prefix
            )
            # The success verification logic (successful_finetunes) is removed
            # as it was only used for plotting.
        # --- End of Group Alpha Loop ---

        logger.info(f"--- Alpha Group Pipeline FINISHED (no reporting) for {log_prefix_base} | Group {alpha_group_str} ---")

    except PipelineConfigError as config_err:
        logger.critical(f"PIPELINE CONFIGURATION ERROR: {config_err}")
        exit(1) # Error code for SLURM
    except Exception as main_err:
        logger.critical(f"FATAL ERROR in alpha group pipeline: {main_err}", exc_info=True)
        exit(1) # Error code for SLURM


if __name__ == "__main__":
    run_alpha_grouped_pipeline_no_report()