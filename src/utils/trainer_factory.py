# ===================================================================
# src/utils/trainer_factory.py
#
# v1.0 : Compliant + Quantization Reintegration.
# ===================================================================

import logging
from typing import Any, Callable, Optional, Type
from pathlib import Path

from datasets import DatasetDict, load_from_disk, Dataset
from transformers import Trainer, TrainerCallback, AutoModelForSequenceClassification, AutoModelForTokenClassification, PreTrainedModel 

from src.config.core import ConfigObject, config
from src.utils.training_args_factory import create_training_arguments
from src.utils.path_manager import get_raw_dataset_config_path, get_tokenized_dataset_path
from src.tasks.base import BaseTaskSpecification
from src.modeling.factory import apply_quantization_scheme

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Callbacks and Exceptions ---
class GuardSaveUntilFirstEval(TrainerCallback): pass
class MissingEvalSplitError(Exception): pass

# --- Internal JIT Function ---
def _get_or_create_tokenized_dataset(
    model_path_or_id: str,
    dataset_info: ConfigObject,
    dataset_config_name: Optional[str],
    tokenizer: Any,
    task_spec: BaseTaskSpecification,
    max_steps: int,
) -> DatasetDict:
    """
    Loads or creates the tokenized dataset via JIT (Just-In-Time).
    Uses the normalized tokenizer name for caching.
    """
    dataset_name = dataset_info.id
    config_name = dataset_config_name

    # Uses the manifest to get a stable tokenizer name for the path
    # Uses placeholders for info not needed here (quant, alpha)

    tokenized_path = get_tokenized_dataset_path(
        dataset_name=dataset_name,
        config_name=config_name,
        tokenizer_name=model_path_or_id,
        is_finetuning=True
    )

    # --- 1. Check Cache ---
    if tokenized_path.exists():
        logging.debug(f"  -> [JIT] Cache found. Loading from {tokenized_path}")
        try:
            return load_from_disk(str(tokenized_path))
        except Exception as e:
            logging.error(f"  -> [JIT] Error loading cache {tokenized_path}: {e}. Retokenizing.", exc_info=True)
            # Delete corrupt cache? Optional.

    # --- 2. Cache missing or corrupt: JIT Tokenization ---
    logging.info(f"  -> [JIT] Cache not found or corrupt. Launching JIT tokenization for {dataset_name}/{config_name or 'default'}...")

    raw_config_path = get_raw_dataset_config_path(dataset_name, config_name, is_finetuning=True)
    if not raw_config_path.exists():
        raise FileNotFoundError(f"[JIT] Raw dataset missing: {raw_config_path}")

    raw_dataset = load_from_disk(str(raw_config_path))
    processor_fn = task_spec.get_preprocessing_function(tokenizer)

    # 3. Apply processor
    first_split_name = next(iter(raw_dataset))
    original_columns = raw_dataset[first_split_name].column_names

    # Use num_proc to speed up? Can be configured via `config`
    num_proc = None # Disabled by default for JIT to avoid complexity/overhead

    tokenized_dataset_mapped = raw_dataset.map(
         processor_fn,
         batched=True,
         num_proc=num_proc,
         desc=f"[JIT] Tokenizing {dataset_name}/{config_name or 'default'}"
    )

    # 4. Handle Columns (Critical logic to preserve correct info)
    final_columns = tokenized_dataset_mapped[first_split_name].column_names
    columns_to_keep = []

    # Keep columns added by the tokenizer (input_ids, attention_mask, etc.)
    # And potentially specific necessary columns (offset_mapping, example_id...)
    # + label columns ('label', 'labels', 'start_positions', 'end_positions')    
    
    # Simpler strategy: identify what was added + keep original labels
    added_columns = [c for c in final_columns if c not in original_columns]
    columns_to_keep.extend(added_columns)
    
    label_keys = [
        "label", "labels", 
        "start_positions", "end_positions", "label_ids",
        "ner_tags", "pos_tags",
        "input_features", "decoderm_input_ids"
    ] # Exhaustive list
    for lk in label_keys:
        if lk in original_columns and lk not in columns_to_keep:
             # Ensure the column still exists after the map (sometimes renamed?)
             if lk in final_columns:
                  columns_to_keep.append(lk)
             elif lk == "label" and "labels" in final_columns and "labels" not in columns_to_keep:
                  # Case where processor renames 'label' to 'labels'
                  columns_to_keep.append("labels")
                  
    # Specific cases required by certain metric calculators
    if task_spec.__class__.__name__ == "SquadV2Task": # Or use a flag/property
         # SquadComputer needs these columns IN eval_dataset_features
         squad_req = ["offset_mapping", "example_id", "unique_id"]
         for req in squad_req:
              if req in final_columns and req not in columns_to_keep:
                   columns_to_keep.append(req)

    columns_to_keep = list(dict.fromkeys(columns_to_keep)) # Deduplicate

    has_text_inputs = all(col in columns_to_keep for col in ["input_ids", "attention_mask"])
    has_audio_inputs = "input_features" in columns_to_keep

    has_text_inputs = all(col in columns_to_keep for col in ["input_ids", "attention_mask"])
    has_audio_inputs = "input_features" in columns_to_keep

    # Verify essential columns are present
    essential_hf_cols = ["input_ids", "attention_mask"]    
    if not (has_text_inputs or has_audio_inputs):
         logging.warning(f"[JIT] Colonnes HF essentielles manquantes (Texte ou Audio) après filtrage: {columns_to_keep}. Ne filtre pas.")
         final_tokenized_dataset = tokenized_dataset_mapped 
    elif not all(col in columns_to_keep for col in essential_hf_cols):
         logging.warning(f"[JIT] Colonnes HF essentielles manquantes après filtrage: {columns_to_keep}. Ne filtre pas.")
         final_tokenized_dataset = tokenized_dataset_mapped # Garde tout en cas de doute
    else:
         final_tokenized_dataset = tokenized_dataset_mapped.select_columns(columns_to_keep)

    is_eval_only = (max_steps == 0)

    target_split_name = "test" if is_eval_only else "validation"

    # Vérification immédiate de l'existence du split
    if target_split_name not in final_tokenized_dataset:
        available_splits = list(final_tokenized_dataset.keys())
        error_msg = (
            f"[{log_prefix}] ⛔ CRITICAL DATASET ERROR: Required split '{target_split_name}' "
            f"is missing for context {'EVALUATION' if is_eval_only else 'FINE-TUNING'}. "
            f"Available splits: {available_splits}. "
            f"Strict Mode Enabled: No silent fallback allowed."
        )
        logging.error(error_msg)
        raise ValueError(error_msg)

    # 5. Save to cache
    try:
        # logging.info(f"  -> [JIT] Saving to cache at: {tokenized_path}")
        final_tokenized_dataset.save_to_disk(str(tokenized_path))
    except Exception as e:
        logging.error(f"  -> [JIT] Error saving cache {tokenized_path}: {e}", exc_info=True)
        # Continue with in-memory dataset anyway

    return final_tokenized_dataset


# --- Main Factory ---
def create_trainer(
    model_path_or_id: str, 
    tokenizer: Any, 
    dataset_info: ConfigObject, 
    dataset_config_name: Optional[str], 
    output_dir: Path, # Kept for TrainingArguments
    max_steps: int, 
    task_spec: BaseTaskSpecification,
    quant_scheme: Optional[ConfigObject] = None 
) -> Trainer:
    """
    v1.0 Factory. Handles JIT, num_labels, model loading, 
    quantization, metrics.
    """
    log_prefix = f"{dataset_info.id}/{dataset_info.configs[0] if hasattr(dataset_info, 'configs') and dataset_info.configs else 'default'}"
    
    # --- 0. Provisioning JIT ---
    try:
        tokenized_dataset = _get_or_create_tokenized_dataset(
            model_path_or_id,
            dataset_info, 
            dataset_config_name,
            tokenizer, 
            task_spec,
            max_steps
        )
    except Exception as e:
        logging.error(f"[{log_prefix}] ÉCHEC CRITIQUE JIT: {e}", exc_info=True)
        raise e 

    # --- 1. Determination num_labels ---
    num_labels: Optional[int] = None
    train_split: Optional[Dataset] = tokenized_dataset.get("train")
    # Determine num_labels even if max_steps=0, because the model loaded
    # for evaluation must also have the correct head.
    if train_split: 
        try:
            num_labels = task_spec.get_num_labels(train_split)
        except Exception as e:
            logging.warning(f"[{log_prefix}] Impossible de déterminer num_labels via task_spec: {e}", exc_info=True)
            
    # --- 2. Model Loading ---
    model_class: Type[PreTrainedModel] = task_spec.get_model_class()
    model_args = dict()

    if num_labels is not None and model_class in {AutoModelForSequenceClassification, AutoModelForTokenClassification}:
         model_args["num_labels"] = num_labels
         logging.info(f"[{log_prefix}] Configuring {model_class.__name__} with num_labels={num_labels}")

    try:
        # Loads from provided path (can be pre-trained or already fine-tuned)
        model = model_class.from_pretrained(model_path_or_id, **model_args)
        logging.info(f"[{log_prefix}] Model loaded from: {model_path_or_id}")
    except Exception as e:
         logging.error(f"[{log_prefix}] CRITICAL model load FAILURE: {e}", exc_info=True)
         raise e

    # --- 3. Quantization Application ---
    if quant_scheme and quant_scheme.enabled:
        try:
            logging.info(f"[{log_prefix}] Applying quantization scheme: {quant_scheme.name}")
            
            modules_to_ignore = ["classifier", "score", "head", "lm_head"]

            model = apply_quantization_scheme(
                model, 
                quant_scheme.to_dict(),
                ignore_modules=modules_to_ignore
            )
            logging.info(f"[{log_prefix}] Quantization applied successfully.")
        except Exception as e:
             logging.error(f"[{log_prefix}] CRITICAL quantization FAILURE: {e}", exc_info=True)
             raise e
    else:
        logging.info(f"[{log_prefix}] Quantization disabled or scheme not provided.")

    # --- 4. Post-Load Adjustments ---
    try:
         model = task_spec.post_init_model(model, train_split)
    except Exception as e:
         logging.warning(f"[{log_prefix}] Error post_init_model: {e}", exc_info=True)

    # --- 5. Eval Split Prep and Metrics ---
    eval_dataset_features: Optional[Dataset] = None
    compute_fn: Optional[Callable] = None
    raw_eval_dataset: Optional[Dataset] = None 

    is_eval_only = (max_steps == 0)

    # STRICT SPLIT SELECTION LOGIC
    # If pure eval -> 'test', else (fine-tuning) -> 'validation'
    target_split_name = "test" if is_eval_only else "validation"

    # We present the TaskSpec only with the allowed split to avoid fuzzy selection
    # Create a temporary DatasetDict containing only the target
    restricted_dataset_dict = DatasetDict({target_split_name: tokenized_dataset[target_split_name]})

    try:
        # Pass restricted dict. TaskSpec MUST use what is given.
        eval_dataset_features = task_spec.prepare_eval_dataset(
            restricted_dataset_dict, model.config 
        )
    except Exception as e:
        logging.error(f"[{log_prefix}] Error prepare_eval_dataset on '{target_split_name}': {e}", exc_info=True)
        raise e

    if eval_dataset_features is None:
        raise RuntimeError(f"[{log_prefix}] TaskSpec returned None for required split '{target_split_name}'.")

    # Formal identification of split for raw data loading
    eval_split_name = target_split_name 

    if eval_dataset_features is not None:
        eval_split_name = getattr(eval_dataset_features, 'split', None) or next((s for s in ['validation', 'test'] if s in tokenized_dataset), None)
        if eval_split_name:
             try:
                 raw_ds_path_str = str(get_raw_dataset_config_path(
                     dataset_id=dataset_info.id, 
                     config_name=dataset_config_name,
                     is_finetuning=True
                 ))
                 raw_eval_dataset = load_from_disk(raw_ds_path_str)[eval_split_name]
             except Exception as e: # Broader capture for robustness
                 logging.error(f"[{log_prefix}] Erreur chargement dataset brut '{eval_split_name}': {e}", exc_info=False) # Pas besoin de trace complète souvent
                 raw_eval_dataset = None # Ensure None if load fails

        try:
            compute_fn = task_spec.get_metric_computer(
                tokenizer=tokenizer,
                eval_dataset_features=eval_dataset_features, 
                raw_eval_dataset=raw_eval_dataset, # Pass None if load failed 
                model=model,
            )
        except Exception as e:
            logging.error(f"[{log_prefix}] FAILURE metric instantiation: {e}", exc_info=True)
            compute_fn = None 
            if max_steps == 0: raise RuntimeError(f"[{log_prefix}] Failed to create metrics in eval mode.") from e

    # --- 6. Data Collator ---
    try:
        data_collator = task_spec.get_data_collator(tokenizer)
    except Exception as e:
        logging.error(f"[{log_prefix}] Error creating data collator: {e}", exc_info=True)
        raise e 

    # --- 7. Training Arguments ---
    can_evaluate = (eval_dataset_features is not None and compute_fn is not None)
    
    training_args = create_training_arguments(
        output_dir=output_dir, # Use provided output_dir
        max_steps=max_steps,
        has_eval_dataset=can_evaluate,
        task_spec=task_spec 
    )
    
    # --- 8. Trainer Instantiation ---
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset.get("train") if max_steps > 0 else None, 
        eval_dataset=eval_dataset_features if can_evaluate else None, 
        data_collator=data_collator,
        compute_metrics=compute_fn if can_evaluate else None, 
        callbacks=[GuardSaveUntilFirstEval()] 
    )
    
    logging.info(f"[{log_prefix}] Trainer instantiated.")
    return trainer