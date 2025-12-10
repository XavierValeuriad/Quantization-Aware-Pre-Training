# ===================================================================
# src/reporting/count_pretraining_tokens.py
# ===================================================================

import logging
import csv
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm

from src.config.core import config
from src.utils.path_manager import get_tokenized_dataset_path, get_results_root, get_project_root

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def count_tokens_in_dataset(dataset_path: Path):
    """
    Loads a tokenized dataset from disk and counts tokens in 'input_ids' column.
    Returns a dictionary: {split_name: total_tokens}
    """
    try:
        dataset_dict = load_from_disk(str(dataset_path))
    except Exception as e:
        logger.error(f"Failed to load dataset at {dataset_path}: {e}")
        return None

    counts = {}
    
    # Iterate over available splits (train, test, validation, etc.)
    for split in dataset_dict.keys():
        ds = dataset_dict[split]
        if "input_ids" not in ds.column_names:
            logger.warning(f"Split '{split}' does not contain 'input_ids'. Skipping.")
            continue
            
        # We can sum the lengths of the input_ids lists.
        # Since these are pre-training datasets grouped by block_size, 
        # usually every row has length 'block_size' (except maybe the last one).
        # But to be precise, we iterate or use a fast mapping.
        
        # Fast estimation if all rows are same length (common in pretraining):
        # total_tokens = len(ds) * len(ds[0]['input_ids']) 
        # However, let's be rigorous:
        
        logger.info(f"Counting tokens for split '{split}'...")
        
        # Method 1 (Exact & Memory Efficient): 
        # Flattening might be heavy. Let's use map/sum or list comp if RAM allows.
        # Given potential size, let's process in batches or just sum lengths.
        
        # Actually, since it's Arrow, we can get the column as a numpy array or list relatively fast
        # providing it fits in RAM. If it's huge, we might need a batched approach.
        # Let's assume KAIROS standard pretraining blocks (usually 512 or 2048 fixed).
        
        # Efficient Counting:
        # Sum of lengths of all input_ids
        total_tokens = sum(len(ids) for ids in tqdm(ds['input_ids'], desc=f"Counting {split}", leave=False))
        
        counts[split] = total_tokens
        
    return counts

def run_token_counting():
    logger.info("Starting Pre-training Token Count...")
    
    results = []

    # Iterate over datasets defined in config
    for dataset_info in config.pretraining_datasets:
        ds_id = dataset_info.id
        
        # Check if dataset has configs and tokenizers
        configs_to_process = dataset_info.configs if hasattr(dataset_info, 'configs') and dataset_info.configs else [None]
        tokenizers = dataset_info.tokenizers if hasattr(dataset_info, 'tokenizers') else []
        
        if not tokenizers:
            logger.warning(f"Dataset {ds_id} has no tokenizers defined. Skipping.")
            continue

        for config_name in configs_to_process:
            for tokenizer_name in tokenizers:
                
                # Construct path using existing logic
                # Note: pretokenizor.py uses is_finetuning=False
                dataset_path = get_tokenized_dataset_path(
                    dataset_name=ds_id,
                    config_name=config_name,
                    tokenizer_name=tokenizer_name,
                    is_finetuning=False
                )

                if not dataset_path.exists():
                    logger.warning(f"Tokenized dataset not found at: {dataset_path}")
                    # We add a row with zeros or N/A to indicate missing data
                    results.append({
                        "dataset_id": ds_id,
                        "config_name": config_name or "default",
                        "tokenizer": tokenizer_name,
                        "split": "N/A",
                        "token_count": 0,
                        "status": "Missing"
                    })
                    continue

                logger.info(f"Processing: {ds_id} | Config: {config_name} | Tokenizer: {tokenizer_name}")
                
                counts = count_tokens_in_dataset(dataset_path)
                
                if counts:
                    for split, count in counts.items():
                        results.append({
                            "dataset_id": ds_id,
                            "config_name": config_name or "default",
                            "tokenizer": tokenizer_name,
                            "split": split,
                            "token_count": count,
                            "status": "OK"
                        })
                        logger.info(f"  -> {split}: {count:,} tokens")
                else:
                    results.append({
                        "dataset_id": ds_id,
                        "config_name": config_name or "default",
                        "tokenizer": tokenizer_name,
                        "split": "Error",
                        "token_count": 0,
                        "status": "LoadError"
                    })

    # --- Export to CSV ---
    output_dir = get_results_root()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "pretraining_token_counts.csv"
    
    if results:
        fieldnames = ["dataset_id", "config_name", "tokenizer", "split", "token_count", "status"]
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
            
        logger.info(f"Token counts saved to: {output_csv}")
        
        # Print Summary to Console
        print("\n" + "="*80)
        print(f"{'DATASET':<25} | {'TOKENIZER':<20} | {'SPLIT':<10} | {'TOKENS':>15}")
        print("-" * 80)
        for r in results:
            if r['status'] == "OK":
                print(f"{r['dataset_id']:<25} | {r['tokenizer']:<20} | {r['split']:<10} | {r['token_count']:>15,}")
        print("="*80 + "\n")
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    run_token_counting()