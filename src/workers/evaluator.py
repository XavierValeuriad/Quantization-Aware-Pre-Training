# ===================================================================
# src/workers/evaluator.py
#
# v1.0 :
#   - Passes model_path and quant_scheme to the factory.
#   - No longer loads the model directly.
# ===================================================================

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

import torch
from transformers import Trainer, EvalPrediction 

from src.config.core import config, ConfigObject
from src.utils.path_manager import get_custom_finetuned_model_path, create_finetuning_manifest
from src.utils.tokenizer_loader import load_tokenizer
from src.utils.trainer_factory import create_trainer
from src.tasks.factory import get_task_specification
from src.utils.misc import default_json_serializer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_evaluation_task(
    model_id: str, 
    quant_scheme: ConfigObject, 
    dataset_info: ConfigObject,
    dataset_config_name: Optional[str],
    alpha: float 
) -> Optional[Dict[str, Any]]:
    """
    Evaluates a model (pre-trained if alpha=1, otherwise fine-tuned).
    Delegates JIT, model loading, and quantization to the factory.
    """
    dataset_name = dataset_info.id
    config_name = dataset_config_name
    log_prefix = f"{dataset_name}/{config_name or 'default'}"

    # --- 1. Retrieve Model Path for Evaluation ---
    manifest = create_finetuning_manifest(model_id, quant_scheme, dataset_info, dataset_config_name, alpha)
    model_path_to_evaluate: Path = get_custom_finetuned_model_path(manifest)

    if not model_path_to_evaluate.exists() or not (model_path_to_evaluate / "config.json").exists():
        logging.error(f"[{log_prefix}] Model not found for evaluation: {model_path_to_evaluate}.")
        return None 

    logging.info(f"--- Evaluation Worker : {model_path_to_evaluate.name} ---")

    # --- 2. Retrieve TaskSpec and Tokenizer ---
    try:
        task_spec = get_task_specification(dataset_info)
    except ValueError as e:
        logging.error(f"ABORT [{log_prefix}]: {e}")
        return None 

    try:
        # Load tokenizer FROM the model folder to be evaluated
        tokenizer = load_tokenizer(model_id) 
    except Exception as e:
        logging.error(f"ABORT [{log_prefix}]: Failed to load tokenizer from '{model_path_to_evaluate}'. Error: {e}", exc_info=True)
        return None 

    # --- 4. Trainer Creation via Factory ---
    trainer = None # For final display robustness
    try:
        trainer = create_trainer(
            model_path_or_id=str(model_path_to_evaluate), 
            tokenizer=tokenizer,
            dataset_info=dataset_info,
            output_dir=model_path_to_evaluate / "eval_temp", # Temporary subdirectory?
            max_steps=0, # Evaluation mode
            task_spec=task_spec,
            dataset_config_name=dataset_config_name,
            quant_scheme=quant_scheme 
        )
    except FileNotFoundError as e: 
         logging.error(f"ABORT [{log_prefix}]: Raw dataset missing for JIT. {e}")
         return None
    except Exception as e:
         logging.error(f"ABORT [{log_prefix}]: Failed to create Trainer. Error: {e}", exc_info=True)
         return None

    # --- 5. Pre-evaluation Checks ---
    if trainer.eval_dataset is None:
         logging.error(f"[{log_prefix}] No evaluation dataset prepared. Evaluation impossible.")
         return {"eval_error": "No evaluation dataset prepared."}
    if trainer.compute_metrics is None:
        logging.error(f"[{log_prefix}] Metric function unavailable. Evaluation impossible.")
        return {"eval_error": "Metric function (compute_fn) not found or disabled."}
    
    # --- 6. Evaluation Preparation ---
    eval_dataset = trainer.eval_dataset 
    eval_split_name = getattr(eval_dataset, 'split', 'unknown') 
    label_cols = [c for c in ("label", "labels", "start_positions") if c in eval_dataset.column_names]
    has_labels = len(label_cols) > 0

    # --- 7. Prediction + Metrics ---
    logging.info(f"[{log_prefix}] Evaluating on '{eval_split_name}' (Labels: {has_labels})...")
    raw_predictions: Optional[EvalPrediction] = None # Pour robustesse
    eval_results: Dict[str, Any] = {}
    
    try:
        raw_predictions = trainer.predict(eval_dataset)
        
        # Calculate metrics if prediction successful
        if has_labels:
            # logging.info(f"[{log_prefix}] Calculating standard metrics...")
            eval_preds = EvalPrediction(predictions=raw_predictions.predictions, label_ids=raw_predictions.label_ids)
            eval_results = trainer.compute_metrics(eval_preds) 
            if eval_results is None: eval_results = {"eval_error": "compute_metrics returned None", "metrics_skipped": True}
        else:
            # logging.info(f"[{log_prefix}] Split '{eval_split_name}' has no labels.")
            eval_results = {"metrics_skipped": True}

    except Exception as e:
        logging.error(f"[{log_prefix}] Error during evaluation: {e}", exc_info=True)
        eval_results = {"eval_error": f"Evaluation failed: {e}", "metrics_skipped": True}

    # Add final metadata
    eval_results["eval_split_used"] = eval_split_name
    eval_results["num_eval_examples"] = len(eval_dataset) if eval_dataset else 0
    
    # === Save Results ===
    results_path = model_path_to_evaluate / "evaluation_results.json"
    try:
        with open(results_path, 'w', encoding='utf-8') as f:
            # Use default=default_json_serializer to handle non-serializable types like np.float32
            json.dump(eval_results, f, indent=4, ensure_ascii=False, default=default_json_serializer)
        logger.info(f"[{log_prefix}] Evaluation results successfully saved to: {results_path}")
    except TypeError as e:
        logger.error(f"[{log_prefix}] Failed to serialize evaluation results to JSON: {e}. Results: {eval_results}")
        # Optional: attempt to save a simplified version or just the error?
        # For now, log the error but return results in memory.
    except IOError as e:
        logger.error(f"[{log_prefix}] Failed to write evaluation results to {results_path}: {e}")
        # Error is logged, but return results in memory anyway.    
        
    # logging.info(f"[{log_prefix}] Evaluation results: {json.dumps(eval_results, indent=2)}")
    return eval_results