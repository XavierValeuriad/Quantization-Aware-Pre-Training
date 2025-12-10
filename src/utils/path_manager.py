# src/utils/path_manager.py

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
from src.config.core import config
from src.config.core import ConfigObject


# --- The 3 base functions read roots from the config ---

def get_data_root() -> Path:
    """Returns the persistent data root ($STORE/Quantization-Aware-Pre-Training/data)."""
    return config.paths.data_root

def get_cache_root() -> Path:
    """Returns the volatile/ephemeral data root ($SCRATCH/lrec2026/cache)."""
    return config.paths.cache_root
    

def _sanitize_name(name: str) -> str:
    """Replaces invalid characters for file/folder names."""
    return name.replace('/', '@')

def get_project_root() -> Path:
    """Finds and returns the absolute project root (containing pyproject.toml)."""
    current_path = Path(__file__).resolve()
    while not (current_path / 'pyproject.toml').exists():
        if current_path.parent == current_path:
            raise FileNotFoundError("Project root (with pyproject.toml) not found.")
        current_path = current_path.parent
    return current_path

def get_original_model_path(model_id: str) -> Path:
    """
    Constructs the path to the original downloaded model repository.
    This is the source of truth for the tokenizer.
    """
    safe_model_name = _sanitize_name(model_id)
    return get_data_root() / "models" / "pretrained" / safe_model_name / "original"

def _generate_run_id(manifest: Dict) -> str:
    """
    Generates a unique identifier (hash) from a complete experiment manifest.
    """
    # Convert to canonical JSON string (sorted keys, no spaces)
    manifest_string = json.dumps(manifest, sort_keys=True, separators=(',', ':'))
    
    # Calculate SHA256 hash and truncate for readability
    return hashlib.sha256(manifest_string.encode('utf-8')).hexdigest()[:16]


def create_pretraining_manifest(model_id: str, quant_scheme: ConfigObject) -> dict:
    """
    Constructs a "refined" manifest containing only the parameters
    that define a pre-training experiment.
    """
    # Filter compatible pre-training datasets
    compatible_pt_datasets = [
        ds.to_dict() for ds in config.pretraining_datasets 
        if model_id in ds.tokenizers
    ]
    compatible_pt_datasets_sorted = sorted(compatible_pt_datasets, key=lambda ds: ds['id'])

    # Build final manifest, identical to the one in `main.py`
    run_manifest = {
        "model_id": model_id,
        "total_steps_budget": config.experiment.total_steps_budget,
        "quantization": quant_scheme.to_dict(),
        "tokenization": config.tokenization.to_dict(),
        "training": config.training.to_dict(),
        "pretraining_datasets": compatible_pt_datasets_sorted,
        "alpha_sweep": config.experiment.alpha_sweep,
        "seed": config.experiment.seed,
    }
    return run_manifest

def _create_finetuning_manifest(
    model_id: str,
    quant_scheme: ConfigObject,
    dataset_info: ConfigObject,
    alpha: float
) -> dict:
    """
    Constructs a unique manifest for a fine-tuning experiment (Internal/Legacy).
    """

    # This manifest includes parameters defining a unique fine-tuning run:
    # base model, quantization, task (dataset), and alpha point.
    refined_config_content = {
        "model_id": model_id,
        "quantization": quant_scheme.to_dict(),
        "tokenization": config.tokenization.to_dict(),
        "training": config.training.to_dict(),
        "finetuning_task": {
            "dataset_id": dataset_info.id,
            "dataset_config": dataset_info.configs[0] if hasattr(dataset_info, 'configs') else 'default'
        },
        "alpha": alpha,
        "seed": config.experiment.seed
    }
    return refined_config_content

def create_finetuning_manifest(
    model_id: str,
    quant_scheme: ConfigObject,
    dataset_info: ConfigObject,
    dataset_config_name: Optional[str],
    alpha: float
) -> dict:
    """
    Constructs a unique manifest for a fine-tuning experiment.
    """

    # This manifest includes parameters defining a unique fine-tuning run:
    # base model, quantization, task (dataset), and alpha point.
    refined_config_content = {
        "model_id": model_id,
        "total_steps_budget": config.experiment.total_steps_budget,
        "quantization": quant_scheme.to_dict(),
        "tokenization": config.tokenization.to_dict(),
        "training": config.training.to_dict(),
        "finetuning_task": {
            "dataset_id": dataset_info.id,
            "dataset_config_name": dataset_config_name if dataset_config_name is not None else 'default'
        },
        "alpha": alpha,
        "seed": config.experiment.seed
    }
    return refined_config_content

def get_custom_pretrained_model_path(run_manifest: dict) -> Path:
    """
    Generates the path for a pre-training experiment.
    The final folder name is prefixed by the quantization scheme.
    Ex: .../custom/alpha_0.1_checkpoint/bf16_a8bde...
    """
    run_id = _generate_run_id(run_manifest)
    model_id = run_manifest["model_id"]
    safe_model_name = _sanitize_name(model_id)
    
    # Extract quantization scheme name for the prefix
    quant_scheme_name = run_manifest['quantization']['name']
    run_folder_name = f"{quant_scheme_name}_{run_id}"
    
    return get_data_root() / "models" / "pretrained" / safe_model_name / "custom" / run_folder_name

def get_custom_pretrained_model_checkpoint_path(run_manifest: dict, alpha: float):

    quant_scheme_name = run_manifest['quantization']['name']
    checkpoint_name = f"{quant_scheme_name}_alpha_{alpha:.2f}_checkpoint"
    custom_pretrained_model_path = get_custom_pretrained_model_path(run_manifest)

    return custom_pretrained_model_path / checkpoint_name

# --- Ensure you have a stable hash function ---
# (If not already existing)
def _generate_params_hash(params: Dict[str, Any], length: int = 16) -> str:
    """Generates a short and stable hash from a parameter dict."""
    # Convert to sorted JSON for stability
    params_string = json.dumps(params, sort_keys=True, ensure_ascii=True)
    hasher = hashlib.sha256(params_string.encode('utf-8'))
    return hasher.hexdigest()[:length]

def get_custom_finetuned_model_path(run_manifest: dict) -> Path:
    """
    Generates the path for a fine-tuning experiment.
    The directory structure includes the task, and the final folder name is prefixed by quantization.
    Ex: .../glue@mrpc/alpha_0.1_checkpoint/bf16_a8bde...
    """
    run_id = _generate_run_id(run_manifest)
    model_id = run_manifest["model_id"]
    safe_model_name = _sanitize_name(model_id)
    
    # 1. Create task subfolder
    task_info = run_manifest['finetuning_task']
    task_name = f"{task_info['dataset_id']}@{task_info['dataset_config_name']}"
    safe_task_name = _sanitize_name(task_name)
    
    # 2. Prefix the experiment folder name with quantization
    quant_scheme_name = run_manifest['quantization']['name']
    run_folder_name = f"{quant_scheme_name}_{run_id}"

    alpha = run_manifest['alpha']
    checkpoint_name = f"alpha_{alpha:.2f}_checkpoint"
    
    return get_data_root() / "models" / "finetuned" / safe_model_name / safe_task_name / checkpoint_name / run_folder_name


def get_tokenized_dataset_path(dataset_name: str, config_name: str | None, tokenizer_name: str, is_finetuning: bool) -> Path:
    dataset_type_folder = "finetuning" if is_finetuning else "pretraining"
    safe_tokenizer_name = _sanitize_name(tokenizer_name)
    safe_dataset_name = _sanitize_name(dataset_name)
    safe_config_name = config_name if config_name else "default"
    return get_data_root() / "datasets" / "tokenized" / dataset_type_folder / safe_tokenizer_name / safe_dataset_name / safe_config_name

def get_pretraining_ready_dataset_path(tokenizer_name: str) -> Path:
    return get_tokenized_dataset_path("_pretraining_ready_", None, tokenizer_name, False)

def get_results_root() -> Path:
    """
    Returns the root path of the results folder (data/results),
    based on the project root.
    """
    # get_data_root() already finds the project root and returns .../data
    # We can simply use it and append /results (or map to reports as per previous logic).
    return get_data_root() / ".." / "reports"

def get_raw_dataset_path(dataset_id: str, is_finetuning: bool) -> Path:
    """
    Constructs the destination path for a downloaded raw dataset.
    The folder name is a "sanitized" version of the dataset ID.
    """
    dataset_type_folder = "finetuning" if is_finetuning else "pretraining"
    safe_dataset_name = _sanitize_name(dataset_id)
    
    # The final path includes the dataset name to avoid conflicts
    # Ex: .../raw/pretraining/c4
    return get_data_root() / "datasets" / "raw" / dataset_type_folder / safe_dataset_name

def get_raw_dataset_config_path(dataset_id: str, config_name: str, is_finetuning: bool) -> Path:
    """
    Constructs the path to the data folder for a SPECIFIC configuration
    of a raw dataset.
    Ex: .../raw/pretraining/glue/cola
    """
    # Reuses parent function to ensure consistency  
    return get_raw_dataset_path(dataset_id, is_finetuning) / config_name
