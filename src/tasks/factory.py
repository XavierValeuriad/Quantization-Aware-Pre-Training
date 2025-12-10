# ===================================================================
# src/tasks/factory.py
#
# v1.0 : Factory and Central Registry for TaskSpecifications.
# ===================================================================

import logging
from typing import Optional

from src.config.core import ConfigObject
from src.tasks.base import BaseTaskSpecification

from src.tasks import drbenchmark

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- ✧ The Registry ✧ ---
# Maps a unique ID (dataset/config) to a task constructor.
TASK_REGISTRY = {

    # Uses specific classes imported via drbenchmark/__init__.py
    "dr-bert/cas/pos": lambda info: drbenchmark.CasTask(),
    "dr-bert/essai/pos": lambda info: drbenchmark.EssaiTask(),
    "dr-bert/quaero/emea": lambda info: drbenchmark.QuaeroTask(), 
    "dr-bert/quaero/medline": lambda info: drbenchmark.QuaeroTask(),

}

def get_task_specification(dataset_info: ConfigObject) -> BaseTaskSpecification:
    """
    Main Factory (v2.1). Strict lookup in the registry.
    Constructs the key "id/config" or "id/default".
    """
    dataset_name = dataset_info.id
    # Uses 'None' if 'configs' is missing or empty
    config_name: Optional[str] = dataset_info.configs[0] if hasattr(dataset_info, 'configs') and dataset_info.configs else None
    
    # Constructs the normalized key
    # If config_name is None, use '/default' to match the registry
    key_config_part = config_name if config_name else "default" 
    key = f"{dataset_name}/{key_config_part}".lower() # Lowercase normalization

    # Handles aliases or special cases (MNLI) BEFORE the main lookup
    if key == "glue/mnli_mismatched" or key == "glue/mnli_matched":
         key = "glue/mnli"
         logging.debug(f"Clé MNLI normalisée en: {key}")

    # Strict Lookup
    builder = TASK_REGISTRY.get(key)
    
    if builder:
        # logging.info(f"[Factory] : Specification found for '{key}'.")
        try:
            return builder(dataset_info)
        except Exception as e:
            logger.error(f"[TaskFactory] Error during task instantiation for '{key}': {e}", exc_info=True)
            raise RuntimeError(f"TaskSpec instantiation failed '{key}'") from e
    else:
        # Error if not found -> Missing configuration or typo
        logger.error(f"[TaskFactory] Task not handled! Key '{key}' (built from id='{dataset_name}', config='{config_name}') not found in TASK_REGISTRY.")
        # Show only a few keys to avoid polluting logs
        available_keys_sample = list(TASK_REGISTRY.keys())[:10] 
        logger.error(f"Examples of available keys: {available_keys_sample}...")
        raise ValueError(f"Unregistered task: '{key}'. Check TASK_REGISTRY and config.yaml.")