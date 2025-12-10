# ===================================================================
# src/orchestrators/download_models.py
# ===================================================================
import logging
from src.config.core import config
from src.utils.path_manager import get_original_model_path
from src.workers.hf_model_downloader import download_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_model_download():
    """Orchestrates the download of all models listed in the config."""
    logging.info("--- Starting model acquisition orchestrator ---")
    
    if not config.models:
        logging.warning("No models listed in the configuration. Operation ending.")
        return

    for model_id in config.models:
        logging.info(f"--- Processing model: {model_id} ---")
        output_path = get_original_model_path(model_id)
        
        try:
            download_model(model_id=model_id, output_dir=output_path)
        except Exception as e:
            logging.error(f"Download failed for {model_id}. Error: {e}", exc_info=True)

    logging.info("--- Model acquisition completed ---")

if __name__ == "__main__":
    run_model_download()