# ===================================================================
# src/metrics/base_computer.py
#
# v1.0 : Abstract base class for the Factory Pattern.
# ===================================================================
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
from transformers import EvalPrediction
from datasets import Dataset

class BaseMetricsComputer(ABC):
    """
    Abstract base class for all metric computers.
    It is 'callable' and holds the necessary state (datasets).
    """
    def __init__(
        self,
        tokenizer: Any,
        eval_dataset_features: Dataset,
        raw_eval_dataset: Dataset,
        **kwargs
    ):
        """
        Initializes the computer with shared state.
        
        Args:
            tokenizer: The tokenizer (required by SQuAD).
            eval_dataset_features: The tokenized dataset (features, required by SQuAD).
            raw_eval_dataset: The raw dataset (examples, required by SQuAD).
            **kwargs: For specific configurations (e.g., n_best_size).
        """
        self.tokenizer = tokenizer
        self.eval_features = eval_dataset_features
        self.raw_examples = raw_eval_dataset
        self.config = kwargs
        logging.info(f"[BaseComputer] : {self.__class__.__name__} instantiated.")

    @abstractmethod
    def __call__(self, p: EvalPrediction) -> Dict[str, float]:
        """
        The 'compute_metrics' method that the Trainer will call.
        MUST be implemented by child classes.
        """
        raise NotImplementedError("The __call__ method must be implemented.")