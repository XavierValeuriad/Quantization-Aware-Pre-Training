# ===================================================================
# src/workers/pipeliner.py
#
# v1.0 : Utility Functions for Orchestration Pipelines
#        - Task parameter reading
#        - Configuration object lookup
#        - Model path determination
#        - Secure execution of FT/Eval pairs

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from src.config.core import config, ConfigObject
from src.utils.path_manager import (
    get_original_model_path,
    get_custom_pretrained_model_path, 
    get_custom_pretrained_model_checkpoint_path,
    get_custom_finetuned_model_path,
    create_pretraining_manifest,
    create_finetuning_manifest
)
from src.workers.finetuner import run_finetuning_task
from src.workers.evaluator import run_evaluation_task

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineConfigError(ValueError):
    """Specific error for pipeline configuration issues."""
    pass

def get_task_parameters() -> Tuple[str, str, str, Optional[str], str]:
    """
    Reads and validates task parameters (M, Q, D) from environment variables.

    Returns:
        Tuple[str, str, str, Optional[str], str]: model_id, quant_name, dataset_id, dataset_config_name, log_prefix_base
    Raises:
        PipelineConfigError: If a required environment variable is missing.
    """
    model_id = os.environ.get('KAIROS_MODEL_ID')
    quant_name = os.environ.get('KAIROS_QUANT_NAME')
    dataset_id = os.environ.get('KAIROS_DATASET_ID')
    dataset_config_env = os.environ.get('KAIROS_DATASET_CONFIG') # Can be None or empty

    if not all([model_id, quant_name, dataset_id]):
        raise PipelineConfigError("Environment variables KAIROS_MODEL_ID, KAIROS_QUANT_NAME, KAIROS_DATASET_ID required.")

    # Normalize dataset_config_name ('none', '', None -> None, else use value)
    dataset_config_name = None if dataset_config_env is None or dataset_config_env.lower() == 'none' or dataset_config_env == '' else dataset_config_env

    log_prefix_base = f"{model_id}|{quant_name}|{dataset_id}/{dataset_config_name or 'default'}"
    logger.info(f"Task parameters retrieved for: {log_prefix_base}")

    return model_id, quant_name, dataset_id, dataset_config_name, log_prefix_base


def find_dataset_info(dataset_list, target_id, target_config):
    """Utility to find dataset_info."""
    if dataset_list is None: return None
    for item in dataset_list:
        # Case-insensitive comparison for ID
        if item.id.upper() == target_id.upper():
             item_configs = getattr(item, 'configs', [])
             # Explicit handling of 'default' and None/empty for target_config
             is_default_target = target_config is None or target_config == 'default' or target_config == ''

             # If dataset has no listed config AND target is 'default'
             if not item_configs and is_default_target:
                 return item # Implicitly found 'default'
             # If dataset has configs and the target (normalized) is in them
             elif item_configs and target_config in item_configs:
                 # Returns the full object. The worker will choose the config.
                 return item
             # Handle case where target_config='default' but 'default' is explicitly listed
             elif item_configs and 'default' in item_configs and is_default_target:
                 return item

    logger.debug(f"Dataset {target_id} with config '{target_config}' not found.")
    return None

def find_config_object(config_list, name_key, target_name):
    """Utility to find an object in a config list by its name."""
    if config_list is None: return None
    return next((item for item in config_list if getattr(item, name_key, None) == target_name), None)

def find_config_objects(quant_name: str, dataset_id: str, dataset_config_name: Optional[str]) -> Tuple[ConfigObject, ConfigObject]:
    """
    Finds ConfigObject objects for the quantization scheme and the dataset.

    Args:
        quant_name (str): Name of the quantization scheme.
        dataset_id (str): Dataset ID.
        dataset_config_name (Optional[str]): Dataset configuration name (or None for 'default').

    Returns:
        Tuple[ConfigObject, ConfigObject]: quant_scheme, dataset_info
    Raises:
        PipelineConfigError: If a configuration object is not found.
    """
    # Use of existing utility functions (potentially to move here if needed)
    quant_scheme = find_config_object(config.quantization, 'name', quant_name)
    if not quant_scheme:
        raise PipelineConfigError(f"Quantization scheme '{quant_name}' not found in config.quantization.")

    # Improved dataset_info search logic
    dataset_info = find_dataset_info(config.finetuning_datasets, dataset_id, dataset_config_name)
    if not dataset_info:
        # If specific config not found, try 'default' explicitly or None
        if dataset_config_name is not None and dataset_config_name != 'default':
             logger.warning(f"Config '{dataset_config_name}' not found for dataset '{dataset_id}', trying 'default'...")
             dataset_info = find_dataset_info(config.finetuning_datasets, dataset_id, 'default')
        if not dataset_info: # Try without config (implicitly default)
             dataset_info = find_dataset_info(config.finetuning_datasets, dataset_id, None)

        if not dataset_info:
             raise PipelineConfigError(f"Dataset '{dataset_id}' (config: {dataset_config_name or 'default'}) not found in config.finetuning_datasets.")

    logger.info(f"Configuration objects found: quant='{quant_scheme.name}', dataset='{dataset_info.id}/{dataset_config_name or 'default'}'")
    return quant_scheme, dataset_info


def get_pretrained_path_for_alpha(model_id: str, quant_scheme: ConfigObject, alpha: float) -> Path:
    """
    Determines the pre-trained model path (original or custom) for a given alpha.

    Args:
        model_id (str): Base model ID.
        quant_scheme (ConfigObject): Quantization scheme configuration object.
        alpha (float): Alpha value (proportion of pre-training budget).

    Returns:
        Path: Path to the required pre-trained model.
    Raises:
        FileNotFoundError: If the required path does not exist.
    """
    pretrain_manifest = create_pretraining_manifest(model_id, quant_scheme)

    if alpha == 0.0 :
        model_path_obj = get_original_model_path(model_id)
        logger.info(f"Alpha=0.0, utilisation du modèle original: {model_path_obj}")
    else:
        # For alpha > 0, we look for the specific custom pre-training checkpoint.
        model_path_obj = get_custom_pretrained_model_checkpoint_path(pretrain_manifest, alpha)
        logger.info(f"Alpha={alpha:.2f}, searching for custom pre-trained checkpoint: {model_path_obj}")

    if not model_path_obj.exists():
        raise FileNotFoundError(f"Required pre-trained model for alpha={alpha:.2f} not found at: {model_path_obj}")

    return model_path_obj


def run_finetune_eval_pair(
    model_id: str,
    quant_scheme: ConfigObject,
    dataset_info: ConfigObject,
    dataset_config_name: Optional[str],
    alpha: float,
    log_prefix: str # For contextual logging
) -> Optional[Dict[str, Any]]:
    """
    Executes the Fine-tuning -> Evaluation pair for a given alpha, with error handling.

    Args:
        model_id (str): Base model ID.
        quant_scheme (ConfigObject): Quantization scheme.
        dataset_info (ConfigObject): Information about the fine-tuning dataset.
        alpha (float): Alpha value.
        log_prefix (str): Prefix for log messages.

    Returns:
        Optional[Dict[str, Any]]: Evaluation results dictionary, or a dictionary
                                   containing 'eval_error' if a step fails. Can return None
                                   in case of very early error.
    """
    results: Dict[str, Any] = {}
    finetuning_successful = False

    try:
        # --- 1. Determine pre-trained model path ---
        pre_trained_model_path_obj = get_pretrained_path_for_alpha(model_id, quant_scheme, alpha)
        pre_trained_model_path = str(pre_trained_model_path_obj)

        # --- 2. Execute Fine-tuning ---
        logger.info(f"--- [{log_prefix}] Starting Fine-tuning ---")
        try:
            run_finetuning_task(
                pre_trained_model_path=pre_trained_model_path,
                model_id=model_id,
                quant_scheme=quant_scheme,
                dataset_info=dataset_info, # Utiliser dataset_info directement
                dataset_config_name=dataset_config_name, # <- UTILISER L'ARGUMENT DIRECTEMENT
                alpha=alpha
            )
            logger.info(f"--- [{log_prefix}] Fine-tuning finished ---")
            finetuning_successful = True
        except Exception as ft_e:
            logger.error(f"[{log_prefix}] ERROR during Fine-tuning: {ft_e}", exc_info=True)
            results = {"eval_error": f"Finetuning failed: {str(ft_e)}"}
            # Do not proceed to evaluation if fine-tuning fails

        # --- 3. Execute Evaluation (if FT successful) ---
        if finetuning_successful:
            logger.info(f"--- [{log_prefix}] Starting Evaluation ---")
            try:
                evaluation_results = run_evaluation_task(
                    model_id=model_id,
                    quant_scheme=quant_scheme,
                    dataset_info=dataset_info,
                    dataset_config_name=dataset_config_name,
                    alpha=alpha
                )
                if evaluation_results is None or evaluation_results.get("eval_error"):
                    logger.error(f"[{log_prefix}] Evaluation failed or incomplete. Results: {evaluation_results}")
                    results = evaluation_results or {"eval_error": "Evaluation returned None"}
                else:
                    logger.info(f"--- [{log_prefix}] Evaluation finished ---")
                    results = evaluation_results # Success

            except Exception as eval_e:
                logger.error(f"[{log_prefix}] ERROR during Evaluation: {eval_e}", exc_info=True)
                results = {"eval_error": f"Evaluation exception: {str(eval_e)}"}

    except FileNotFoundError as fnf_e:
        logger.error(f"[{log_prefix}] EARLY ERROR: {fnf_e}")
        results = {"eval_error": str(fnf_e)} # Error even before FT
    except Exception as pair_e:
        logger.error(f"[{log_prefix}] UNEXPECTED ERROR in run_finetune_eval_pair: {pair_e}", exc_info=True)
        results = {"eval_error": f"Unexpected error in pair: {str(pair_e)}"}

    return results
