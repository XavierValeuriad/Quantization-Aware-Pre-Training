# ===================================================================
# src/workers/hf_model_downloader.py
# ===================================================================

import argparse
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# Logging configuration for clear output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_model(model_id: str, output_dir: Path):
    """ 
    Downloads a model and its tokenizer from Hugging Face Hub
    and saves them to a local directory.
    """
    if output_dir.exists():
        logging.info(f"Model {model_id} is already downloaded. Skipping.")
        return
    
    try:
        logging.info(f"Starting download for model: {model_id}")

        # Ensure destination directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Downloading and saving the tokenizer
        logging.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(output_dir)
        logging.info(f"Tokenizer saved to {output_dir}")

        # Downloading and saving the model
        logging.info("Downloading model (this may take some time)...")
        model = AutoModel.from_pretrained(model_id)
        model.save_pretrained(output_dir)
        logging.info(f"Model saved to {output_dir}")

        logging.info(f"Download of {model_id} completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred while downloading {model_id}: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face model downloader.")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID on Hugging Face Hub (e.g., 'camembert-base').")
    parser.add_argument("--output_dir", type=Path, required=True, help="Save directory path.")

    args = parser.parse_args()
    
    download_model(args.model_id, args.output_dir)
