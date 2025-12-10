# ===================================================================
# src/utils/training_args_factory.py
#
# v1.0 : Added an argument safeguard mechanism.
#        Prevents accidental overrides from the config file
#        for critical parameters managed by the factory.
# ===================================================================

import logging
from typing import Optional
import torch
from pathlib import Path
from transformers import TrainingArguments
from src.config.core import config
from dotenv import load_dotenv

#import src.utils.set_idr_torch

load_dotenv()

TASK_TYPE_OPTIMIZATION_METRIC = {
    "sequence_classification": {"metric_name": "f1", "greater_is_better": True},
    "sequence_regression": {"metric_name": "pearson", "greater_is_better": True},
    "token_classification": {"metric_name": "overall_f1", "greater_is_better": True},
    "multiple_choice": {"metric_name": "exact_match", "greater_is_better": True},
    "default": {"metric_name": "f1", "greater_is_better": True},
}

def create_training_arguments(
        output_dir: Path, 
        max_steps: int,
        has_eval_dataset: bool,
        task_spec: Optional = None
    ) -> TrainingArguments:
    """
    Constructs a TrainingArguments object from the central configuration.
    """
    logging.info(f"✧ KAIROS [TrainingArgsFactory] : Task spec: {task_spec}")
    training_config = config.training.to_dict()
    
    # --- Monitoring and Reproducibility Configuration ---
    num_points = training_config.pop("num_monitoring_points", 100)
    monitoring_interval = max(1, config.experiment.total_steps_budget // num_points if num_points > 0 and config.experiment.total_steps_budget > 0 else 1)
    seed = config.experiment.seed
    logging.info(f"Intervalle de monitoring (logging/eval/save) : {monitoring_interval} steps.")

    if not has_eval_dataset:
        logging.warning("Aucun dataset d'évaluation fourni. L'évaluation en cours d'entraînement est désactivée.")

    # --- Contextual Logic Application ---
    load_best = has_eval_dataset
    metric_for_best_model = None
    greater_is_better = None

    if has_eval_dataset:

        eval_strategy = "steps"
        eval_steps_value = monitoring_interval
        load_best = True
        save_strategy_value = "steps"

        if task_spec:
            # CONTEXT: FINE-TUNING / EVALUATION (encapsulated logic)
            optim_config = task_spec.get_optimization_metric()
            metric_for_best_model = optim_config['name']
            greater_is_better = optim_config['greater_is_better']
            logging.info(f"Contexte Fine-tuning détecté. Métrique d'optimisation : {metric_for_best_model}")
        else:
            # CONTEXT: PRE-TRAINING (Fallback)
            metric_for_best_model = "eval_loss"
            greater_is_better = False
            logging.info(f"Contexte Pré-entraînement détecté. Métrique d'optimisation : {metric_for_best_model}")
    else:
        logging.warning("Aucun dataset d'évaluation valide fourni ou métrique non initialisée. Désactivation explicite de l'évaluation.")
        eval_strategy = "no" # <-- Explicitly disable
        eval_steps_value = None
        load_best = False
        save_strategy_value = "steps" # We always save at regular intervals
        metric_for_best_model = None
        greater_is_better = None

    # 1. Define arguments managed internally by this factory.
    managed_args = {
        "output_dir", "max_steps", "logging_strategy", "logging_steps",
        "evaluation_strategy", "eval_steps", "save_strategy", "save_steps",
        "seed", "data_seed", "full_determinism", "load_best_model_at_end", "report_to", 
        "push_to_hub", "metric_for_best_model", "run_name"#, "eval_on_start"
    }

    # 2. Check and clean user configuration.
    for arg in managed_args:
        if arg in training_config:
            logging.warning(
                f"Le paramètre '{arg}' est géré par l'usine pour garantir la cohérence A*. "
                f"La valeur fournie dans votre configuration sera ignorée."
            )
            training_config.pop(arg)

    # ==================== ✧ FP16 CORRECTION ✧ ====================
    # Check CUDA availability and modify config BEFORE passing it.
    if not torch.cuda.is_available():
        if training_config.get("fp16"):
            logging.warning("GPU CUDA non détecté. Le mode FP16 va être désactivé.")
            training_config["fp16"] = False
        if training_config.get("fp16_full_eval"):
            logging.warning("GPU CUDA non détecté. Le mode fp16_full_eval va être désactivé.")
            training_config["fp16_full_eval"] = False
    
    return TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        
        # Monitoring strategies (saving is always active)
        logging_strategy="steps",
        logging_steps=monitoring_interval,
        evaluation_strategy=eval_strategy,
        eval_steps=eval_steps_value,
        save_strategy=save_strategy_value,
        save_steps=monitoring_interval,
        #eval_on_start=True,
        
        # Reproducibility
        seed=seed,
        data_seed=seed,
        full_determinism=True,
        
        # Trainer Behavior
        load_best_model_at_end=load_best,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_total_limit=1,

        # Reporting
        report_to=["tensorboard"], # Activate reporting
        run_name=output_dir.name, # Consistent naming for W&B/TB runs
        push_to_hub=False,
        
        # SECURE unpacking of remaining parameters
        **training_config
    )