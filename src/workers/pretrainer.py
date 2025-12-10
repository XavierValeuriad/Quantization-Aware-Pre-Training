# src/workers/pretrainer.py
# Pre-training Execution Core v1.0
# Contains the shared logic to execute A SINGLE experiment.

import logging
import json
from datasets import load_from_disk
from transformers import (
    AutoModelForMaskedLM, DataCollatorForLanguageModeling,
    TrainingArguments, Trainer, TrainerCallback, TrainerState, TrainerControl
)
from src.modeling.factory import apply_quantization_scheme
from src.utils.path_manager import (
    get_pretraining_ready_dataset_path, get_custom_pretrained_model_checkpoint_path, 
    get_custom_pretrained_model_path, create_pretraining_manifest, 
    get_original_model_path
)
from src.utils.training_args_factory import create_training_arguments
from src.config.core import config
from src.utils.tokenizer_loader import load_tokenizer
from src.utils.post_training_helpers import (
    start_carbon_tracker, finalize_training
)
# Note: Ensure set_idr_torch is imported if necessary
# from src.utils.set_idr_torch import set_idr_torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- SHARED MODULE 1: The Callback ---

class AlphaCheckpointCallback(TrainerCallback):
    """
    Callback qui sauvegarde les checkpoints aux étapes définies par alpha.
    """
    def __init__(self, run_manifest: dict):
        self.step_to_alpha = {
            int(config.experiment.total_steps_budget * alpha): alpha 
            for alpha in config.experiment.alpha_sweep if config.experiment.total_steps_budget * alpha > 0
        }
        self.run_manifest = run_manifest
        logging.info(f"Callback Alpha configuré pour sauvegarder aux steps : {sorted(self.step_to_alpha.keys())}")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if state.global_step in self.step_to_alpha:
            alpha = self.step_to_alpha[state.global_step]
            checkpoint_dir = get_custom_pretrained_model_checkpoint_path(self.run_manifest, alpha)

            logging.info(f"Step {state.global_step} atteint ! Sauvegarde du checkpoint alpha={alpha:.2f}...")
            
            if model is not None:
                model.save_pretrained(str(checkpoint_dir))
            else:
                logging.error("Impossible de sauvegarder le checkpoint : référence au modèle non trouvée.")

            logging.info(f"Checkpoint alpha sauvegardé dans : {checkpoint_dir}")


# --- SHARED MODULE 2: The Execution Core ---

def execute_single_pretraining_run(model_id: str, quant_scheme: dict):
    """
    Executes a full pre-training run for a given
    (model_id, quant_scheme) combination.
    
    Args:
        model_id (str): The model identifier (e.g., "camembert-base").
        quant_scheme (dict): The configuration object for the quantization 
                             scheme (from config.yml).
    """
    
    quant_name = quant_scheme.name
    logging.info(f"--- ✧ Core Execution: Model='{model_id}', Quant='{quant_name}' ✧ ---")

    # --- 1. Dataset Management ---
    unified_dataset_path = get_pretraining_ready_dataset_path(model_id)

    if not unified_dataset_path.exists():
        logging.error(f"ABORT: Unified pre-training dataset not found at: {unified_dataset_path}")
        raise FileNotFoundError(f"Unified dataset missing for {model_id} at {unified_dataset_path}")        
    
    logging.info(f"Loading unified pre-training corpus from: {unified_dataset_path}")
    unified_dataset = load_from_disk(str(unified_dataset_path))
    
    if 'train' not in unified_dataset:
        logging.error(f"ABORT: 'train' split is missing in the unified dataset for {model_id}.")
        raise KeyError(f"'train' split missing for {model_id}")
        
    unified_train_dataset = unified_dataset['train']
    unified_eval_dataset = unified_dataset.get('test') # Robust if 'test' is missing
    
    if not unified_eval_dataset:
        logging.warning("No 'test' split found. Evaluation will be disabled.")
    
    logging.info(f"Corpora loaded (Train: {len(unified_train_dataset)}, Eval: {len(unified_eval_dataset) if unified_eval_dataset else 'N/A'})")

    # --- 2. Experiment Preparation (Manifest, Model, Trainer) ---
    logging.info(f"--- Preparing Run: {quant_name} ---")
            
    run_manifest = create_pretraining_manifest(model_id, quant_scheme)
    output_dir = get_custom_pretrained_model_path(run_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "run_manifest.json", 'w') as f:
        json.dump(run_manifest, f, indent=4)
    
    logging.info(f"Reproducibility manifest saved.")
    logging.info(f"Experiment path: {output_dir}")

    base_model_path = get_original_model_path(model_id)
    
    model = AutoModelForMaskedLM.from_pretrained(str(base_model_path))
    logging.info(f"[Pretrainer] : Model loaded from: {base_model_path}")
    model = apply_quantization_scheme(model, quant_scheme.to_dict())
    logging.info(f"[Pretrainer] : Model quantized.")
    tokenizer = load_tokenizer(model_id)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

    max_alpha = max(config.experiment.alpha_sweep) if config.experiment.alpha_sweep else 1.0
    max_steps = int(config.experiment.total_steps_budget * max_alpha)
    
    training_args = create_training_arguments(
        output_dir=output_dir,
        max_steps=max_steps,
        has_eval_dataset=(unified_eval_dataset is not None)
    )

    alpha_callback = AlphaCheckpointCallback(run_manifest=run_manifest)
    
    trainer = Trainer(
        model=model,    
        args=training_args,    
        data_collator=data_collator,
        train_dataset=unified_train_dataset,
        eval_dataset=unified_eval_dataset,
        callbacks=[alpha_callback]
    )

    try:
        model.tie_weights()
    except Exception as e:
        logging.warning(f"Error during tie_weights: {e}")
        raise e

    # --- 3. Training Launch ---
    model.train()
    for m in model.modules():
        m.train()
    
    logging.info("Starting training loop...")
    tracker = start_carbon_tracker(output_dir)
    try:
        trainer.train()
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise e
    finally:
        finalize_training(trainer, tracker)
    
    logging.info(f"--- Run {quant_name} completed ---")