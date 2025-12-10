# ===================================================================
# src/config/core.py
#
# Central configuration management module.
# v1.0: Introduction of configuration override for test environments
#       (prod_config.yaml < dev_config.yaml).
# ===================================================================

import yaml
from pathlib import Path
from dotenv import load_dotenv
import collections.abc
import os
import logging

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error() 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_world_size() -> int:
    """
    Dynamically retrieves the total number of processes (GPUs) in the distributed job.

    This function is designed to be infrastructure-agnostic:
    - In a distributed environment (SLURM, Torchrun), it reads 'WORLD_SIZE'.
    - Locally or if the variable is undefined, it returns 1.
    """
    # The "Gold Standard": 'WORLD_SIZE' is the standard variable
    # used by PyTorch Distributed, Horovod, etc.
    world_size = os.environ.get('WORLD_SIZE')

    if world_size is not None:
        return int(world_size)

    # Fallback method: Direct calculation from SLURM variables.
    # Useful for debugging or specific use cases.
    elif 'SLURM_JOB_ID' in os.environ:
        try:
            num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
            gpus_per_node_str = os.environ.get('SLURM_GPUS_ON_NODE')
            
            # SLURM_GPUS_ON_NODE might not be defined, or be a string.
            if gpus_per_node_str:
                gpus_per_node = int(gpus_per_node_str)
                return num_nodes * gpus_per_node
            else:
                # If undefined, assume we are using tasks as an indicator
                return int(os.environ.get('SLURM_NTASKS', 1))

        except (ValueError, TypeError):
            # In case of parsing error, remain conservative.
            return 1
            
    # Default case: Local execution on a single machine.
    else:
        return 1

class ConfigObject:
    """
    Recursive class that transforms a dictionary into an object
    where keys are accessible as attributes.
    Now handles lists containing dictionaries.
    """
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            elif isinstance(value, list):
                setattr(self, key, [ConfigObject(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    def to_dict(self):
        """Recursively converts the ConfigObject into a standard dictionary."""
        output = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigObject):
                output[key] = value.to_dict()
            elif isinstance(value, list):
                output[key] = [
                    item.to_dict() if isinstance(item, ConfigObject) else item
                    for item in value
                ]
            else:
                output[key] = value
        return output

def find_project_root() -> Path:
    """Finds the project root by searching for the pyproject.toml file."""
    current_path = Path.cwd()
    while current_path != current_path.parent:
        if (current_path / "pyproject.toml").exists():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError("Racine du projet (contenant 'pyproject.toml') non trouvée.")

def _initialize_and_resolve_paths(is_dev_env: bool) -> dict:
    """
    Defines the 3 root paths based on the environment ($WORK, etc.)
    or a symmetrical local structure, inferring the project name.
    """
    # 1. Project name and location inference
    project_root_path = find_project_root_from_script()
    project_name = project_root_path.name  # ex: "lrec2026"
    project_parent = project_root_path.parent

    # 2. Base root resolution (HPC or Local)
    # On Jean-Zay, WORK is the project parent folder. Locally, it is the project folder itself.
    work_root = Path(os.getenv("WORK", project_root_path)).resolve()
    scratch_root = Path(os.getenv("SCRATCH", project_parent / ".." / "SCRATCH")).resolve()
    store_root = scratch_root

    env_folder = "dev" if is_dev_env else "prod"
    
    # 3. Final path definition for configuration
    paths = {
        "project_root": work_root,
        # Data is in STORE/ENV/<project_name>/data
        "data_root": store_root / env_folder / project_name / "data",
        # Cache is in SCRATCH/ENV/<project_name>/cache
        "cache_root": scratch_root / env_folder / project_name / "cache"
    }
    
    logging.info(f"PROJECT_ROOT (WORK) : {paths['project_root']}")
    logging.info(f"DATA_ROOT (STORE)   : {paths['data_root']}")
    logging.info(f"CACHE_ROOT (SCRATCH): {paths['cache_root']}")
    
    # Create directories to ensure structure exists
    paths['data_root'].mkdir(parents=True, exist_ok=True)
    paths['cache_root'].mkdir(parents=True, exist_ok=True)

    total_gpus = get_world_size()
    paths['total_gpus'] = total_gpus 

    logging.info(f"✧ {total_gpus} GPU(s) ont été détectées pour ce run.")
    
    return paths

def _recursive_update(base_dict: dict, new_dict: dict) -> dict:
    """
    Recursively updates a dictionary.
    Values in new_dict overwrite those in base_dict.
    If a key is present in both and both values are dictionaries,
    the function calls itself recursively.
    """
    for key, value in new_dict.items():
        if isinstance(value, collections.abc.Mapping) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = _recursive_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def find_project_root_from_script() -> Path:
    """Finds the project root based on this script's location."""
    current_path = Path(__file__).resolve()
    while not (current_path / 'pyproject.toml').exists():
        if current_path.parent == current_path:
            raise FileNotFoundError("Project root (with pyproject.toml) not found.")
        current_path = current_path.parent
    return current_path

def load_config() -> ConfigObject:
    """
    Loads configuration from prod and test files.
    The test configuration overrides production if and only if
    the test file exists and contains the key 'dev: true'.
    """
    try:
        project_root = find_project_root()
        config_dir = project_root / "configs"
        prod_config_path = config_dir / "prod_config.yaml"
        dev_config_path = config_dir / "dev_config.yaml"
        
        if not prod_config_path.exists():
            raise FileNotFoundError(f"Production configuration file not found: {prod_config_path}")
        
        with open(prod_config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}

        is_dev_active = False
        if dev_config_path.exists():
            with open(dev_config_path, 'r', encoding='utf-8') as f:
                dev_config_data = yaml.safe_load(f) or {}
            
            if dev_config_data and dev_config_data.get('dev') is True:
                print("INFO: Dev configuration active. Overriding production configuration.")
                config_data = _recursive_update(config_data, dev_config_data)
                is_dev_active = True

        # --- DYNAMIC PATH INJECTION ---
        path_config = _initialize_and_resolve_paths(is_dev_env=is_dev_active)
        config_data['paths'] = path_config
        # ----------------------------------------
        
        return ConfigObject(config_data)

    except Exception as e:
        print(f"Erreur critique lors du chargement de la configuration : {e}")
        raise


# === EXPLICIT .ENV LOADING ===
try:
    PROJECT_ROOT = find_project_root()
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        # Pass explicit path to load_dotenv
        load_dotenv(dotenv_path=dotenv_path)
    else:
        # This is not an error, .env is optional
        print("INFO: Fichier .env non trouvé à la racine du projet. Utilisation des variables d'environnement existantes.")
except FileNotFoundError as e:
    print(f"AVERTISSEMENT: {e}. Impossible de localiser la racine du projet pour charger .env.")
# ======================================

# --- Single Entry Point ---
def _set_cache_env_vars(cache_path: Path):
    """Configure les variables d'environnement pour tous les caches."""
    hf_cache = cache_path / "huggingface"
    uv_cache = cache_path / "uv"
    uv_home = cache_path / "uv_home"
    peotry_cache = cache_path / "poetry"
    poetry_config = cache_path / "poetry_config"
    
    os.environ['HF_HOME'] = str(hf_cache)
    os.environ['UV_CACHE_DIR'] = str(uv_cache)
    os.environ['POETRY_CACHE_DIR'] = str(peotry_cache)
    os.environ['POETRY_CONFIG_DIR'] = str(poetry_config)

    hf_cache.mkdir(parents=True, exist_ok=True)
    uv_cache.mkdir(parents=True, exist_ok=True)
    peotry_cache.mkdir(parents=True, exist_ok=True)
    poetry_config.mkdir(parents=True, exist_ok=True)

    logging.info(f"Cache Poetry (POETRY_CACHE_DIR) configuré pour : {peotry_cache}")
    logging.info(f"Cache Poetry (POETRY_CONFIG_DIR) configuré pour : {poetry_config}")
    logging.info(f"Cache Hugging Face (HF_HOME) configuré pour : {hf_cache}")
    logging.info(f"Cache UV (UV_CACHE_DIR) configuré pour : {uv_cache}")
 
config = load_config()
_set_cache_env_vars(config.paths.cache_root)