# Proposal for: src/tasks/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Callable
from transformers import PreTrainedModel
from datasets import Dataset, DatasetDict

from src.metrics.base_computer import BaseMetricsComputer

# ===================================================================
# ✧ SYSTEM :: src/tasks/base.py
#
# v1.0 : Task Specification Interface.
#        Now includes the contract for the tokenization function.
# ===================================================================

class BaseTaskSpecification(ABC):
    """
    Unified interface (v1) defining the 'contract' for an evaluation
    or fine-tuning task. It provides all task-specific components.
    """

    def post_init_model(self, model, train_split):
        """Post-initialization hook (optional)."""
        return model

    @abstractmethod
    def get_model_class(self) -> Type[PreTrainedModel]:
        """Returns the Hugging Face model class (e.g., AutoModelForSequenceClassification)."""
        pass

    @abstractmethod
    def get_data_collator(self, tokenizer: Any) -> Any:
        """Returns the appropriate data collator instance."""
        pass

    @abstractmethod
    def get_metric_computer(
        self,
        tokenizer: Any,
        eval_dataset_features: Dataset,
        raw_eval_dataset: Dataset,
        model: PreTrainedModel
    ) -> BaseMetricsComputer:
        """Returns the callable object for compute_metrics."""
        pass

    @abstractmethod
    def get_optimization_metric(self) -> Dict[str, Any]:
        """Returns the steering metric (e.g., {'name': 'f1', 'greater_is_better': True})."""
        pass

    @abstractmethod
    def get_num_labels(self, train_split: Dataset) -> int:
        """Encapsulates logic for detecting num_labels (finetuner.py)."""
        pass

    @abstractmethod
    def prepare_eval_dataset(
        self,
        tokenized_dataset: DatasetDict,
        model_config: Any
    ) -> Dataset:
        """
        Encapsulates split selection logic ('validation'/'test'),
        label sanitizing, and handling of label-less splits (evaluator.py).
        """
        pass
    
    @abstractmethod
    def get_preprocessing_function(
        self,
        tokenizer: Any
    ) -> Callable:
        """
        Returns the preprocessing (tokenization) function
        specific to this task.
        """
        pass