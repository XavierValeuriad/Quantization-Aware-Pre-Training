# ===================================================================
# src/orchestrators/run_pretraining_array.py
#
# v1.0 : Unit Task Orchestrator
# ===================================================================

import logging
import os
from src.utils.determinism import set_absolute_determinism
from src.config.core import config
from src.workers.pretrainer import execute_single_pretraining_run

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pretraining_task():
    """
    Executes a single pre-training task based on
    MODEL_ID and QUANT_NAME environment variables.
    """
    logging.info("--- Lancement TÂCHE DE PRÉ-ENTRAÎNEMENT (v1.1 Array) ---")

    # --- 1. Récupérer les paramètres de la tâche depuis l'environnement ---
    model_id = os.environ.get('MODEL_ID')
    quant_name = os.environ.get('QUANT_NAME')

    if not model_id or not quant_name:
        logging.error("ABANDON : MODEL_ID ou QUANT_NAME non défini dans l'environnement.")
        raise EnvironmentError("Variables d'environnement pour le job array manquantes.")

    logging.info(f"Task assigned: Model='{model_id}', Quantization='{quant_name}'")

    # --- 2. Locking and Validation ---
    set_absolute_determinism()

    if model_id not in config.models:
        logging.error(f"ABORT: Model '{model_id}' (env var) not found in config.yml.")
        raise ValueError(f"Model '{model_id}' not found in config.")

    quant_scheme = next((q for q in config.quantization if q.name == quant_name), None)
    
    if not quant_scheme:
        logging.error(f"ABORT: Scheme '{quant_name}' (env var) not found or disabled in config.yml.")
        raise ValueError(f"Quantization scheme '{quant_name}' not found or disabled.")
    
    # --- 3. Delegation of execution to core ---
    try:
        execute_single_pretraining_run(
            model_id=model_id,
            quant_scheme=quant_scheme
        )
    except Exception as e:
        logging.error(f"CRITICAL FAILURE of task {model_id} / {quant_name}: {e}")
        # Raises exception so Slurm marks the array job as FAILED
        raise e

    logging.info("--- Pre-training task completed successfully ---")


if __name__ == "__main__":
    run_pretraining_task()