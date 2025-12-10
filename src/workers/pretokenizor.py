# ===================================================================
# src/workers/pretokenizor.py
#
# v1.0 : Worker dedicated to PRE-TRAINING tokenization.
#        Purged of all fine-tuning logic (TaskSpecification).
# ===================================================================

import logging
from math import ceil
import shutil

from datasets import load_from_disk, DatasetDict, concatenate_datasets
from src.config.core import config
from src.utils.path_manager import get_pretraining_ready_dataset_path, get_tokenized_dataset_path, get_raw_dataset_config_path
from src.utils.tokenizer_loader import load_tokenizer
from src.utils.system import get_optimal_cpu_count

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _resolve_text_column(ds, preferred: str | None = None) -> str:
    """
    Detects the main text column in a raw dataset.
    (Pre-training logic preserved)
    """
    # 1) Respect preference if it really exists
    cols = list(ds.column_names)  # ds = Dataset (not DatasetDict here)
    if preferred and preferred in cols:
        return preferred

    # 2) Common candidates
    candidates = [
        "text", "content", "book_text", "body", "document",
        "article", "passage", "wiki", "raw", "data", "paragraph"
    ]
    for c in candidates:
        if c in cols:
            return c

    # 3) String type features if available
    try:
        feats = ds.features
        for c in cols:
            f = feats.get(c)
            if hasattr(f, "dtype") and str(getattr(f, "dtype")) == "string":
                return c
    except Exception:
        pass

    # 4) Sample inspection
    sample = ds.select(range(min(1, len(ds))))[:] if len(ds) else {}
    for c in cols:
        try:
            v = sample[c][0]
            if isinstance(v, str):
                return c
        except Exception:
            continue

    raise KeyError(
        f"No text column detected. Available columns: {cols}. "
        f"Please specify `text_column` in the dataset config."
    )

def process_pretraining_dataset(dataset_info, tokenizer, tokenizer_name_safe):
    """
    Processes ONE pre-training dataset (e.g., 'wikipedia/20220301.fr').
    Tokenizes, groups, and saves the intermediate version.
    """
    dataset_name = dataset_info.id

    if not hasattr(dataset_info, 'configs') or not dataset_info.configs:
        logging.warning(f"ABORT: No 'configs' key found or list is empty for dataset entry '{dataset_name}' in configuration file.")
        return
    
    config_names = dataset_info.configs

    for config_name in config_names:
        logging.info(f"--- Pre-training Processing: '{dataset_name}' (config: {config_name or 'default'}) ---")        
        
        tokenized_path = get_tokenized_dataset_path(
            dataset_name, 
            config_name, 
            tokenizer_name_safe,
            is_finetuning=False # <-- DEDICATED PRE-TRAINING
        )

        if tokenized_path.exists():
            logging.info(f"  -> Tokenized version by '{tokenizer_name_safe}' already exists. Skipping.")
            continue

        raw_config_path = get_raw_dataset_config_path(dataset_name, config_name, is_finetuning=False)
        if not raw_config_path.exists():
            logging.error(f"ABORT: Raw configuration folder '{config_name}' not found at {raw_config_path}. Ensure download.")
            continue
            
        raw_dataset = load_from_disk(str(raw_config_path))

        # --- Partition Protocol (for test split) ---
        rigor_config = config.evaluation_rigor
        Z = rigor_config.confidence_level_sigma
        k = rigor_config.relative_error_factor_k
        n_required = ceil((Z / k) ** 2)
        logging.info(f"  -> Required axiomatic sample size: {n_required} examples.")

        # =================================================================
        # --- PRE-TRAINING PIPELINE (Unique logic for this worker) ---
        # =================================================================
        
        logging.info("  -> Pre-training Mode: applying tokenization and grouping BEFORE splitting.")

        if config.tokenization.max_pretraining_samples:
            limit = config.tokenization.max_pretraining_samples
            logging.info(f"  -> [DEV MODE] Truncating dataset to the first {limit} examples.")
            for split in raw_dataset.keys():
                current_size = len(raw_dataset[split])
                if current_size > limit:
                    raw_dataset[split] = raw_dataset[split].select(range(limit))
                    logging.info(f"     -> Split '{split}' reduced from {current_size} to {len(raw_dataset[split])} examples.")

        # STEP 1: Concatenate all raw splits
        unified_raw_dataset = concatenate_datasets([raw_dataset[split] for split in raw_dataset.keys()])
        logging.info(f"  -> Unified raw corpus with {len(unified_raw_dataset)} documents.")

        # STEP 2: Text column detection
        try:
            text_column = _resolve_text_column(
                unified_raw_dataset, 
                getattr(dataset_info, 'text_column', None)
            )
            logging.info(f"  -> Text column detected: '{text_column}'.")
        except KeyError as e:
            logging.error(f"ABANDON : {e}")
            continue
        
        # STEP 3: Generic tokenization function
        def tokenize_function(examples):
            texts = examples[text_column]
            normed = []
            for t in texts:
                if isinstance(t, str):
                    normed.append(t)
                elif t is None:
                    normed.append("")
                elif isinstance(t, (bytes, bytearray)):
                    try:
                        normed.append(t.decode("utf-8", errors="ignore"))
                    except Exception:
                        normed.append(str(t))
                else:
                    normed.append(str(t))
            return tokenizer(normed, return_attention_mask=True)
        
        num_proc = get_optimal_cpu_count()
        logging.info(f"  -> Using {num_proc} processor cores.")

        # STEP 4: Tokenize the unified corpus
        tokenized_dataset = unified_raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc, 
            remove_columns=unified_raw_dataset.column_names,
            desc="Running tokenizer on dataset"
        )
 
        block_size = tokenizer.model_max_length
        if block_size > 2048:
            logging.warning(f"Tokenizer has a `model_max_length` ({block_size}) > 2048. Capping at 2048.")
            block_size = 2048

        # STEP 5: Grouping function
        def group_texts(examples):
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        
        # STEP 6: Group tokens
        grouped_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=num_proc, 
            desc=f"Grouping texts into blocks of {block_size}"
        )

        # STEP 7: Split final corpus
        logging.info("  -> Creating evaluation split from grouped final corpus.")
        num_total_samples = len(grouped_dataset)
        
        test_size = min(n_required, num_total_samples - 1)

        if test_size > 0:
            final_dataset = grouped_dataset.train_test_split(test_size=test_size, seed=config.experiment.seed)
            logging.info(f"  -> Final splits created ('train'/{len(final_dataset['train'])}, 'test'/{len(final_dataset['test'])})")
        else:
            logging.warning("  -> Dataset too small for test split. Creating single 'train' split.")
        
        logging.info(f"  -> Starting save (pre-training) to: {tokenized_path}")
        final_dataset.save_to_disk(str(tokenized_path))
        logging.info(f"  -> Pre-training dataset tokenized and saved successfully.")


def create_unified_pretraining_datasets(model_id: str):
    """
    Orchestrates the creation of unified pre-training datasets.
    (Pre-training logic preserved)
    """
    logging.info(f"--- UNIFICATION OF PRE-TRAINING DATASETS FOR '{model_id}' ---")
    
    # --- 1. Collection of relevant tokenized datasets ---
    datasets_to_concatenate = []
    for dataset_info in config.pretraining_datasets:
        if model_id in dataset_info.tokenizers:
            configs_to_process = dataset_info.configs if hasattr(dataset_info, 'configs') else [None]
            for config_name in configs_to_process:
                dataset_path = get_tokenized_dataset_path(
                    dataset_info.id, config_name, model_id, is_finetuning=False
                )
                if dataset_path.exists():
                    logging.info(f"     -> Found: {dataset_path}")
                    datasets_to_concatenate.append(load_from_disk(str(dataset_path)))
                else:
                    logging.warning(f"     -> WARNING: Individual dataset not found at {dataset_path}. It will be ignored.")
    
    if not datasets_to_concatenate:
        logging.warning(f"  -> No dataset found for '{model_id}'. Unable to create unified corpus.")
        return

    # --- 2. Merging splits ---
    train_splits = [ds['train'] for ds in datasets_to_concatenate if 'train' in ds]
    eval_splits = [ds['test'] for ds in datasets_to_concatenate if 'test' in ds]
    
    if not train_splits:
        logging.error(f"  -> ERROR: No 'train' split found for '{model_id}'.")
        return

    unified_train = concatenate_datasets(train_splits)
    unified_eval = concatenate_datasets(eval_splits) if eval_splits else None
        
    logging.info(f"  -> Unified corpora created (Train: {len(unified_train)} ex., Val: {len(unified_eval) if unified_eval else 0} ex.)")

    # ==================== ✧ FINAL FORMATTING PASS ✧ ====================
    tokenizer = load_tokenizer(model_id)
    max_seq_length = tokenizer.model_max_length
    if max_seq_length > 2048: max_seq_length = 2048
    num_proc = get_optimal_cpu_count()

    logging.info(f"  -> Applying final formatting pass (padding & truncation to {max_seq_length})...")

    def final_format_pass(batch):
        """Truncates & pads already tokenized IDs."""
        ids_list = batch["input_ids"]
        masks_list = batch.get("attention_mask")
        labels_list = batch.get("labels")

        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token or tokenizer.unk_token})
        pad_id = tokenizer.pad_token_id

        new_ids, new_masks = [], []
        new_labels = [] if labels_list is not None else None
        new_token_types = []

        for i, ids in enumerate(ids_list):
            ids = ids[:max_seq_length]
            if masks_list is not None:
                mask = masks_list[i][:max_seq_length]
            else:
                mask = [1] * len(ids)
            if labels_list is not None:
                labs = labels_list[i][:max_seq_length]

            if len(ids) < max_seq_length:
                pad_len = max_seq_length - len(ids)
                ids = ids + [pad_id] * pad_len
                mask = mask + [0] * pad_len
                if labels_list is not None:
                    labs = labs + [-100] * pad_len

            new_ids.append(ids)
            new_masks.append(mask)
            if labels_list is not None:
                new_labels.append(labs)
            new_token_types.append([0] * max_seq_length)

        out = {
            "input_ids": new_ids,
            "attention_mask": new_masks,
            "token_type_ids": new_token_types,
        }
        if labels_list is not None:
            out["labels"] = new_labels
        return out

    formatted_train = unified_train.map(final_format_pass, batched=True, num_proc=num_proc)
    formatted_train = formatted_train.select_columns(["input_ids", "attention_mask", "token_type_ids", "labels"])
    final_dataset = DatasetDict({'train': formatted_train})
    
    if unified_eval:
        formatted_eval = unified_eval.map(final_format_pass, batched=True, num_proc=num_proc)
        formatted_eval = formatted_eval.select_columns(["input_ids", "attention_mask", "token_type_ids", "labels"])
        final_dataset['test'] = formatted_eval
            
    # --- 4. Saving the final DatasetDict ---
    output_path = get_pretraining_ready_dataset_path(model_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logging.info(f"  -> Cleaning up previous ready-to-use pre-training dataset at `{output_path}`.")
        shutil.rmtree(output_path)
    
    logging.info(f"  -> Saving ready-to-use pre-training dataset to: {output_path}")
    final_dataset.save_to_disk(str(output_path))