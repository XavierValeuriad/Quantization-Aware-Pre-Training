# ===================================================================
# ✧ scripts/utils/make_dataset.py
#
# v1.0 : Architectural Standardization. The script now saves
#        the dataset in a configuration subfolder and creates the
#        'dataset_configs.json' manifest for perfect integration.
# ===================================================================

import logging
from pathlib import Path
import sys
import os
import shutil
import json

# --- Autonomous Path Resolution ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# ------------------------------------

from datasets import load_from_disk
from src.data_processing.nachos_dataset import NachosDataset
from src.utils.path_manager import get_data_root, get_raw_dataset_path, get_raw_dataset_config_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_nachos_dataset():
    """
    Generates the NACHOS dataset and structures it as a standard
    multi-configuration dataset, with a JSON manifest.
    """
    dataset_id = "Dr-BERT@NACHOS"
    # --- EXPLICIT CONFIGURATION DECLARATION ---
    config_name = "documents"
    # ------------------------------------------------
    
    source_file = get_data_root() / "sources" / "pretraining" / dataset_id / "DOCUMENT_BY_DOCUMENT.cleaned.txt"
    
    # DESTINATION: Uses the new function to point to the config subfolder
    final_dataset_path = get_raw_dataset_config_path(dataset_id, config_name, is_finetuning=False)
    
    # The parent folder that will contain the JSON manifest
    dataset_parent_path = get_raw_dataset_path(dataset_id, is_finetuning=False)

    temp_cache_dir = get_data_root() / ".cache" / f"{dataset_id}_temp"
    
    # ... (source file existence check)
    if not source_file.exists():
        logging.error(f"Source file not found: {source_file}")
        return
        
    if final_dataset_path.exists():
        logging.warning(f"Final dataset already exists. Cleaning {final_dataset_path.parent}")
        shutil.rmtree(final_dataset_path.parent)

    logging.info(f"--- Starting generation of dataset '{dataset_id}' (config: {config_name}) ---")

    try:
        builder = NachosDataset(
            cache_dir=str(temp_cache_dir),
            data_files={'train': [str(source_file)]}
        )
        
        builder.download_and_prepare(max_shard_size="500MB")
        final_dataset = builder.as_dataset()
        
        logging.info(f"Saving final DatasetDict to {final_dataset_path}...")
        final_dataset.save_to_disk(str(final_dataset_path))
        
        # --- MANIFEST CREATION ---
        manifest_path = dataset_parent_path / "dataset_configs.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump([config_name], f, indent=2)
        logging.info(f"Configuration manifest created: {manifest_path}")
        # -----------------------------
        
        logging.info("Cleaning up temporary files...")
        shutil.rmtree(temp_cache_dir)
        
        logging.info("--- MISSION ACCOMPLISHED ---")
        reloaded_dataset = load_from_disk(str(final_dataset_path))
        print("Dataset final généré :")
        print(reloaded_dataset)

    except Exception as e:
        logging.error("Une erreur est survenue lors de la génération du dataset.", exc_info=True)

if __name__ == "__main__":
    generate_nachos_dataset()