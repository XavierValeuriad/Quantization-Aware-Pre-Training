# ===================================================================
# ✧ SYSTEM :: src/reporting/data_collector.py
#
# v1.0 : AttributeError Fix (TaskSpec)
#        - Uses task_spec.get_optimization_metric()['name']
#          in compliance with the 'base.py' contract.
# ===================================================================

import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from src.config.core import config, ConfigObject
from src.utils.path_manager import (
    get_custom_finetuned_model_path,
    get_custom_pretrained_model_path,
    create_pretraining_manifest
)

logger = logging.getLogger(__name__) 

class CollectedData:
    def __init__(self, model_id: str, quant_scheme: ConfigObject,
                 dataset_info: ConfigObject, dataset_config_name: Optional[str],
                 alpha: float,
                 manifest: Dict[str, Any]):
        
        self.model_id = model_id
        self.quant_name = quant_scheme.name
        self.dataset_id = dataset_info.id
        self.dataset_config = dataset_config_name or 'default'
        self.alpha = alpha
        self.manifest = manifest 
        self.dataset_info: ConfigObject = dataset_info
        self.ft_path: Path = get_custom_finetuned_model_path(manifest)
        self.eval_json: Optional[Dict[str, Any]] = self._load_eval_json()
        self.log_csv: Optional[pd.DataFrame] = self._load_log_csv()

    def _load_eval_json(self) -> Optional[Dict[str, Any]]:
        json_path = self.ft_path / "evaluation_results.json"
        if not json_path.exists():
            logger.warning(f"MISSING Data (JSON): {json_path.relative_to(config.paths.data_root)}")
            return None
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get("kairos_save_error"):
                    logger.error(f"Serialization Error (JSON): {json_path.relative_to(config.paths.data_root)} - Error: {data['error_message']}")
                    return None
                return data
        except Exception as e:
            logger.error(f"Read Error (JSON): {json_path.relative_to(config.paths.data_root)}. Error: {e}")
            return None

    def _load_log_csv(self) -> Optional[pd.DataFrame]:
        csv_path = self.ft_path / "training_log_history.csv"
        if not csv_path.exists():
            logger.warning(f"MISSING Data (CSV): {csv_path.relative_to(config.paths.data_root)}")
            return None
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Read Error (CSV): {csv_path.relative_to(config.paths.data_root)}. Error: {e}")
            return None
            
    def is_valid(self) -> bool:
        return self.eval_json is not None or self.log_csv is not None

    def get_main_metric(self, task_spec: Any) -> Optional[float]: # task_spec est un BaseTaskSpecification
        """Extracts the main metric (e.g., 'eval_accuracy') from the JSON."""
        if self.eval_json is None:
            return None
            
        try:
            # task_spec is an object, we call its method
            metric_name = task_spec.get_optimization_metric()['name']
        except Exception as e:
            logger.warning(f"Impossible d'obtenir 'metric_name' depuis task_spec pour {self.dataset_id}. Erreur: {e}")
            return None
            
        # Gérer le bootstrap
        if config.evaluation.use_bootstrap:
            if metric_name not in self.eval_json:
                 logger.warning(f"Métrique bootstrap '{metric_name}' non trouvée dans {self.ft_path.name}")
                 return None
            metric_data = self.eval_json.get(metric_name, {})
            return metric_data.get('mean') 
        else:
            return self.eval_json.get(metric_name)


def load_pretraining_log(model_id: str, quant_scheme: ConfigObject) -> Optional[pd.DataFrame]:
    pt_manifest = create_pretraining_manifest(model_id, quant_scheme)
    pt_path = get_custom_pretrained_model_path(pt_manifest)
    csv_path = pt_path / "training_log_history.csv"
    
    if not csv_path.exists():
        logger.warning(f"Donnée MANQUANTE (PT CSV): {csv_path.relative_to(config.paths.data_root)}")
        return None
    try:
        df = pd.read_csv(csv_path)
        df['model_id'] = model_id
        df['quant_name'] = quant_scheme.name
        return df
    except Exception as e:
        logger.error(f"Erreur de lecture (PT CSV): {csv_path.relative_to(config.paths.data_root)}. Erreur: {e}")
        return None