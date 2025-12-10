# ===================================================================
# src/training/pretraining.py
#
# v1.0 : Refactored Grid Orchestrator.
# ===================================================================

import logging
from src.utils.determinism import set_absolute_determinism
from src.config.core import config
from src.workers.pretrainer import execute_single_pretraining_run 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pretraining_grid():
    logging.info("--- Lancement de la grille de PRÉ-ENTRAÎNEMENT (v1.0) ---")

    # --- DETERMINISM LOCKING ---
    set_absolute_determinism()

    # --- Experiment Parameters ---
    quantization_schemes = config.quantization
    models_to_train = config.models

    if not quantization_schemes:
        logging.warning("No quantization scheme is enabled ('enabled: true') in config.yml.")

    # --- Main Loop: Per Model ---
    for model_id in models_to_train:
        logging.info(f"############# NEW CAMPAIGN: Model '{model_id}' #############")

        # --- 2. Loop over Quantization Schemes ---
        for quant_scheme in quantization_schemes:
            
            try:
                execute_single_pretraining_run(
                    model_id=model_id,
                    quant_scheme=quant_scheme
                )
            except Exception as e:
                quant_name = quant_scheme.name
                logging.error(f"CRITICAL FAILURE of run {model_id} / {quant_name}: {e}")

    logging.info("--- Pre-training grid completed successfully ---")

if __name__ == "__main__":
    run_pretraining_grid()