# ===================================================================
# src/orchestrators/run_pretokenization.py
#
# v1.0 : Orchestrator dedicated to PRE-TRAINING tokenization.
# ===================================================================

import logging

from src.config.core import config
from src.utils.tokenizer_loader import load_tokenizer
from src.utils.path_manager import _sanitize_name
from src.workers.pretokenizor import process_pretraining_dataset, create_unified_pretraining_datasets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pretokenization():
    logging.info("--- Starting tokenization orchestrator (PRE-TRAINING) ---")
    
    if not config.models or not isinstance(config.models, list):
        raise ValueError("The 'models' configuration is missing or is not a list.")
    
    for base_model_name in config.models:
        logging.error(f"Failed to load tokenizer {base_model_name}. Skipping this model.")
        
        try:
            tokenizer = load_tokenizer(base_model_name)
            tokenizer_name_safe = _sanitize_name(base_model_name)
        except Exception as e:
            logging.error(f"Échec du chargement du tokenizer {base_model_name}. On saute ce modèle. Erreur: {e}")
            continue

        # --- 1. Processing individual pre-training datasets ---
        logging.info("--- Checking source pre-training datasets ---")
        
        if config.pretraining_datasets:
            for dataset_info in config.pretraining_datasets:
                if base_model_name in dataset_info.tokenizers:
                    process_pretraining_dataset(
                        dataset_info, 
                        tokenizer, 
                        tokenizer_name_safe
                    )
        else:
            logging.info(" -> Pre-training dataset list empty or undefined. No action.")

        # --- 2. Unified corpus creation (after individual processing) ---
        logging.info("--- Starting pre-training corpus unification ---")
        create_unified_pretraining_datasets(base_model_name)

if __name__ == "__main__":
    run_pretokenization()