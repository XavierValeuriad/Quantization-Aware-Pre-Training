# ===================================================================
# src/modeling/tokenizer_loader.py
#
# v1.0 : Module dedicated to tokenizer loading.
#        Ensures that the tokenizer is always loaded from its
#        source of truth: the original model repository.
# ===================================================================

import logging
from transformers import AutoTokenizer, PreTrainedTokenizer
from src.utils.path_manager import get_original_model_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_tokenizer(model_id: str) -> PreTrainedTokenizer:
    """
    Loads a tokenizer from the original model path.

    Args:
        model_id (str): The Hugging Face model identifier (e.g., 'google-bert/bert-base-uncased').

    Returns:
        PreTrainedTokenizer: The loaded tokenizer instance.
    """
    logging.info(f"Loading tokenizer for '{model_id}' from the original repository...")   
    
    original_model_path = get_original_model_path(model_id)
    
    if not original_model_path.exists():
        logging.error(f"Original model directory not found at: {original_model_path}")
        raise FileNotFoundError(f"Unable to load tokenizer: path {original_model_path} does not exist. Did you run the 02_download_models.bash script?")        
    
    tokenizer = AutoTokenizer.from_pretrained(str(original_model_path), use_fast=True)
    logging.info("Tokenizer loaded successfully.")
    return tokenizer