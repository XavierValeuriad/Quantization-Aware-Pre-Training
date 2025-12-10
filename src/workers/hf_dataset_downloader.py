# ===================================================================
# src/workers/hf_dataset_downloader.py
#
# v1.0 : Explicit Authentication. The script now reads the
#        token from the HUGGING_FACE_HUB_TOKEN environment variable
#        to ensure authentication.
# ===================================================================

import argparse
import logging
import os 
import json
from pathlib import Path
from datasets import get_dataset_config_names, load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_dataset(dataset_id: str, base_output_dir: Path, configs_to_download: list[str] | None):
    """
    Downloads one or more configurations of a dataset from Hugging Face.
    Uses explicit authentication via environment variable.
    """
    try:
        logging.info(f"Processing dataset: {dataset_id}")
        
        # === EXPLICIT TOKEN READING ===
        auth_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not auth_token:
            logging.warning("HUGGING_FACE_HUB_TOKEN environment variable not found. Download might fail for private repositories.")
        # =================================

        available_configs = get_dataset_config_names(
            dataset_id, 
            token=auth_token, # Explicit token passing
            trust_remote_code=True
        )
        
        if len(available_configs) > 10:
            preview = ", ".join(available_configs[:3])
            logging.info(f"Available configurations: {len(available_configs)} (e.g., {preview}, ...)")
        else:
            logging.info(f"Available configurations: {available_configs}")
        if configs_to_download:
            target_configs = [c for c in configs_to_download if c in available_configs]
            if not target_configs:
                logging.warning(f"None of the requested configurations {configs_to_download} were found.")
                return
        else:
            logging.info("No specific configuration requested. Downloading all available configurations.")
            target_configs = available_configs
        logging.info(f"Configurations to download: {target_configs}")

        for config_name in target_configs:
            config_output_dir = base_output_dir / config_name
            if config_output_dir.exists():
                logging.info(f"Dataset '{config_output_dir}' is already downloaded. Skipping.")
                continue

            logging.info(f"--- Downloading configuration: {config_name} ---")
            
            dataset = load_dataset(
                dataset_id, 
                name=config_name, 
                token=auth_token, # Explicit token passing
                trust_remote_code=True
            )
            
            logging.info(f"Saving configuration '{config_name}' to disk...")
            dataset.save_to_disk(config_output_dir)
            logging.info(f"Configuration '{config_name}' saved to {config_output_dir}")

    except Exception as e:
        logging.error(f"An error occurred for {dataset_id}: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face multi-configuration dataset downloader.")
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--base_output_dir", type=Path, required=True)
    parser.add_argument("--configs", nargs='*', help="Optional list of specific configurations to download.")

    args = parser.parse_args()
    download_dataset(args.dataset_id, args.base_output_dir, args.configs)