"""
===================================================================
src/workers/finetuner.py
v1.0 : 
- Passes model_path_or_id and quant_scheme to the factory.
- No longer loads the model, no longer quantizes locally.
===================================================================
"""

import logging
import json
from pathlib import Path
from typing import Optional

from src.config.core import config, ConfigObject
from src.utils.tokenizer_loader import load_tokenizer
from src.utils.path_manager import get_custom_finetuned_model_path, create_finetuning_manifest 
from src.utils.trainer_factory import create_trainer
from src.utils.post_training_helpers import start_carbon_tracker, finalize_training 
from src.tasks.factory import get_task_specification


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_finetuning_task(
    pre_trained_model_path: str, # PATH of the pre-trained model to load
    model_id: str, 
    quant_scheme: ConfigObject, # Scheme to apply AFTER loading
    dataset_info: ConfigObject,
    dataset_config_name: Optional[str],
    alpha: float
):
    dataset_name = dataset_info.id
    config_name = dataset_config_name
    log_prefix = f"{dataset_name}/{config_name or 'default'}"

    logger.info(f"Starting finetuning request for {log_prefix}...")

    # --- 1. Path Preparation and Safeguards ---
    manifest = create_finetuning_manifest(model_id, quant_scheme, dataset_info, dataset_config_name, alpha)
    output_dir = get_custom_finetuned_model_path(manifest)

    # More robust check: if trainer_state exists, training is finished.
    if output_dir.exists() and (output_dir / "trainer_state.json").exists(): 
        logger.info(f"Model already finetuned (found trainer_state.json) at {output_dir}. Skipping.")
        return

    logger.info(f"--- Worker Fine-tuning : {output_dir.name} ---")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. TaskSpec and Tokenizer Retrieval ---
    try:
        task_spec = get_task_specification(dataset_info)
    except ValueError as e:
        logger.error(f"ABORT [{log_prefix}]: {e}")
        return

    try:
        # Load the tokenizer associated with the BASE model (model_id)
        tokenizer = load_tokenizer(model_id) 
    except Exception as e:
        logger.error(f"ABORT [{log_prefix}]: Failed to load tokenizer '{model_id}'. Error: {e}", exc_info=True)
        return

    finetuning_steps = int((1 - alpha) * config.experiment.total_steps_budget)
    if finetuning_steps <= 0 and alpha < 1.0 :
        # logging.warning(f"[{log_prefix}] Fine-tuning steps <= 0. Setting to 0.")
        finetuning_steps = 0 
    elif alpha == 1.0:
        # logging.info(f"[{log_prefix}] Alpha=1.0 -> Zero-shot (0 training steps).")
        finetuning_steps = 0

    # --- 5. Trainer Creation via Factory ---
    trainer = None # For finalize_training
    try:
        trainer = create_trainer(
            model_path_or_id=pre_trained_model_path, 
            tokenizer=tokenizer,
            dataset_info=dataset_info,
            output_dir=output_dir, # The trainer saves here
            max_steps=finetuning_steps,
            dataset_config_name=dataset_config_name,
            task_spec=task_spec,
            quant_scheme=quant_scheme 
        )
    except Exception as e:
        logger.error(f"ABORT [{log_prefix}]: Failed to create Trainer. Error: {e}", exc_info=True)
        return

    # --- 6. Training and Saving ---
    tracker = None
    try:
        if finetuning_steps > 0:
            logger.info(f"[{log_prefix}] Starting training for {finetuning_steps} steps...")
            tracker = start_carbon_tracker(output_dir) 
            if hasattr(trainer.model, 'train'): trainer.model.train() 
            train_result = trainer.train() # Capture the result
            # log metrics ? train_result.training_loss
            logging.info(f"[{log_prefix}] Training completed.")
        else:
            logging.info(f"[{log_prefix}] Skipping training (max_steps={finetuning_steps}).")

        logging.info(f"[{log_prefix}] Saving final model and trainer state to {output_dir}...")
        trainer.save_model(output_dir) 
        trainer.save_state() 
        logging.info(f"[{log_prefix}] Model and state saved successfully.")

    except Exception as e:
        logger.error(f"ABORT [{log_prefix}]: Error during training or saving. Error: {e}", exc_info=True)
    finally:
        # Ensure finalize is called even if trainer=None (if creation fails)
        finalize_training(trainer, tracker)