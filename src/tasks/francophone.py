from typing import Type, Any, Dict, Callable
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, PreTrainedModel
from datasets import Dataset, DatasetDict
from src.tasks.base import BaseTaskSpecification
from src.metrics.computers import GenericAccuracyComputer

class FlueTask(BaseTaskSpecification):
    """
    Gestionnaire générique pour les tâches FLUE (XNLI, CLS, PAWS-X).
    Les colonnes d'entrée varient selon la config.
    """
    def __init__(self, dataset_info):
        self.config_name = dataset_info.configs[0] if dataset_info.configs else "xnli"

    def get_model_class(self) -> Type[PreTrainedModel]:
        return AutoModelForSequenceClassification

    def get_data_collator(self, tokenizer: Any) -> Any:
        return DataCollatorWithPadding(tokenizer=tokenizer)

    def get_metric_computer(self, tokenizer: Any, eval_dataset_features: Dataset, raw_eval_dataset: Dataset, model: PreTrainedModel):
        return GenericAccuracyComputer(tokenizer=tokenizer, eval_dataset_features=eval_dataset_features, raw_eval_dataset=raw_eval_dataset)

    def get_optimization_metric(self) -> Dict[str, Any]:
        return {"name": "accuracy", "greater_is_better": True}

    def get_num_labels(self, train_split: Dataset) -> int:
        return train_split.features["label"].num_classes

    def prepare_eval_dataset(self, tokenized_dataset: DatasetDict, model_config: Any) -> Dataset:
        # FLUE utilise souvent 'validation' ou 'test'
        if "test" in tokenized_dataset: return tokenized_dataset["test"]
        return tokenized_dataset["validation"]

    def get_preprocessing_function(self, tokenizer: Any) -> Callable:
        # Mapping des colonnes selon la sous-tâche FLUE
        task_cols = {
            "xnli": ("premise", "hypothesis"),
            "paws-x": ("sentence1", "sentence2"),
            "cls": ("review", None),
            # Ajout FSE/NSD si traité comme classification de phrase simple
            "fse": ("sentence", None), 
            "nsd": ("sentence", None)
        }
        
        col1, col2 = task_cols.get(self.config_name, ("text", None)) # Fallback

        def preprocess(examples):
            if col2:
                return tokenizer(examples[col1], examples[col2], truncation=True, max_length=512)
            return tokenizer(examples[col1], truncation=True, max_length=512)
        return preprocess


class FracasTask(BaseTaskSpecification):
    """
    Tâche d'inférence logique FraCaS.
    Format attendu: premise, hypothesis, label (Entailment/Neutral/Contradiction).
    """
    def get_model_class(self) -> Type[PreTrainedModel]:
        return AutoModelForSequenceClassification

    def get_data_collator(self, tokenizer: Any) -> Any:
        return DataCollatorWithPadding(tokenizer=tokenizer)

    def get_metric_computer(self, tokenizer: Any, eval_dataset_features: Dataset, raw_eval_dataset: Dataset, model: PreTrainedModel):
        return GenericAccuracyComputer(tokenizer=tokenizer, eval_dataset_features=eval_dataset_features, raw_eval_dataset=raw_eval_dataset)

    def get_optimization_metric(self) -> Dict[str, Any]:
        return {"name": "accuracy", "greater_is_better": True}

    def get_num_labels(self, train_split: Dataset) -> int:
        # FraCaS a généralement 3 labels (Yes, No, Unknown) -> mappés vers 0,1,2
        if hasattr(train_split.features["label"], "num_classes"):
            return train_split.features["label"].num_classes
        return 3

    def prepare_eval_dataset(self, tokenized_dataset: DatasetDict, model_config: Any) -> Dataset:
        # FraCaS est très petit, souvent pas de split validation officiel sur HF.
        # On utilise 'validation' si dispo, sinon on peut splitter train (mais ici on suppose dataset clean)
        return tokenized_dataset.get("validation", tokenized_dataset.get("train")) # Fallback train pour test technique (Attention overfitting)

    def get_preprocessing_function(self, tokenizer: Any) -> Callable:
        def preprocess(examples):
            # Vérifier les noms de colonnes exacts du dataset FraCaS chargé
            return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, max_length=256)
        return preprocess