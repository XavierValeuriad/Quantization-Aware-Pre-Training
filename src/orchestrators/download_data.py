# ===================================================================
# src/orchestrators/download_data.py
#
# v1.0 : Complete standardization with path_manager.
#        All path construction logic is delegated,
#        increasing robustness and consistency.
# ===================================================================

import logging

# --- SINGLE CONFIGURATION ENTRY POINT ---
from src.config.core import config
# ------------------------------------------------

# --- CENTRALIZED PATH MANAGEMENT ---
from src.utils.path_manager import get_raw_dataset_path
# ----------------------------------------

from src.workers.hf_dataset_downloader import download_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_dataset_list(dataset_list, is_finetuning: bool):
    """
    Iterates over a list of datasets, determines their destination path
    via path_manager, and initiates their download.
    """
    if not dataset_list:
        logging.info(f"The {'fine-tuning' if is_finetuning else 'pre-training'} dataset list is empty. No action.")
        return
        
    for dataset_info in dataset_list:
        dataset_id = dataset_info.id
        configs_to_download = dataset_info.configs if hasattr(dataset_info, 'configs') else None
        
        # DELEGATION: The path_manager decides the path
        dataset_path = get_raw_dataset_path(dataset_id, is_finetuning=is_finetuning)
        
        try:
            # Parent folder creation is managed by download_dataset
            download_dataset(dataset_id, dataset_path, configs_to_download)
        except Exception:
            logging.error(f"Processing failed for {dataset_id}. Moving to next.", exc_info=True)
            continue

def run_data_download():
    """Main function of the download orchestrator."""
    logging.info("--- Starting download orchestrator (v1.0) ---")

    logging.info("--- Processing pre-training datasets ---")
    process_dataset_list(config.pretraining_datasets, is_finetuning=False)
    
    logging.info("--- Processing fine-tuning datasets ---")
    process_dataset_list(config.finetuning_datasets, is_finetuning=True)
    
    logging.info("--- Download orchestrator finished ---")

if __name__ == "__main__":
    run_data_download()