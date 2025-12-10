# ===================================================================
# src/tasks/drbenchmark/ner.py
#
# v1.0 : Direct implementation based on QUAERO scripts.
#        - "Safe" Collator popping parasitic keys (e.g., ner_tags)
#        - Robust tokenization (fixed length, type checks)
#        - prepare_eval_dataset fallback without exotic args
# ===================================================================

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    PreTrainedTokenizer,
    PreTrainedModel,
)
import torch
from torch import nn

from src.tasks.base import BaseTaskSpecification
from src.utils.task_utils import get_eval_split_from_dict
from src.metrics.computers import SeqEvalComputer 

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# QUAERO Labels (Official Order)
# -------------------------------------------------------------------
QUAERO_NER_LABELS: List[str] = [
    "O",
    "B-ANAT", "I-ANAT",
    "B-CHEM", "I-CHEM",
    "B-DEVI", "I-DEVI",
    "B-DISO", "I-DISO",
    "B-GEOG", "I-GEOG",
    "B-LIVB", "I-LIVB",
    "B-OBJC", "I-OBJC",
    "B-PHEN", "I-PHEN",
    "B-PHYS", "I-PHYS",
    "B-PROC", "I-PROC",
]
LABEL_TO_ID: Dict[str, int] = {lab: i for i, lab in enumerate(QUAERO_NER_LABELS)}
ID_TO_LABEL: Dict[int, str] = {i: lab for i, lab in enumerate(QUAERO_NER_LABELS)}

IGNORE_INDEX = -100


# -------------------------------------------------------------------
# "Safe" Collator: removes non-handled keys from each feature before
# tokenizer.pad to avoid padding errors (e.g., ner_tags).
# -------------------------------------------------------------------
class SafeDataCollatorForTokenClassification(DataCollatorForTokenClassification):
    """Identical to DataCollatorForTokenClassification, but pops all non-standard
    keys before calling tokenizer.pad, to avoid ValueErrors like
    "Perhaps your features (`ner_tags`) have excessive nesting".
    """

    # Allowed keys (known by HF for padding)
    _allowed_keys = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "labels",
        # Potentially useful if already present and consistent:
        "special_tokens_mask",
        "offset_mapping",
        "position_ids",
    }

    def torch_call(self, features):
        # In-place feature cleanup
        # We remove everything that isn't explicitly supported
        for f in features:
            extra_keys = [k for k in list(f.keys()) if k not in self._allowed_keys]
            for k in extra_keys:
                f.pop(k, None)
        return super().torch_call(features)


# -------------------------------------------------------------------
# NER Task Specification (QUAERO)
# -------------------------------------------------------------------
class QuaeroTask(BaseTaskSpecification):
    """Spécification pour DrBenchmark/QUAERO (NER)."""

    LABEL_COLUMN = "ner_tags"
    TOKEN_MODE = "bio" # Used by SeqEvalComputer
    LABEL_LIST = QUAERO_NER_LABELS

    # -------------------- BaseTaskSpecification ---------------------

    def get_model_class(self):
        return AutoModelForTokenClassification

    def post_init_model(self, model, train_split):
        # FIX: Use self.LABEL_LIST directly for mappings
        # num_labels_expected = self.get_num_labels(train_split) # Optional: keep for verification        
        
        if self.LABEL_LIST: # Verify list exists
            # Use self.LABEL_LIST (list of strings) to create dictionaries
            id2label = {i: l for i, l in enumerate(self.LABEL_LIST)}
            label2id = {l: i for i, l in enumerate(self.LABEL_LIST)}
            
            # Apply to model config
            if hasattr(model, "config"):
                model.config.id2label = id2label
                model.config.label2id = label2id
                
                # Verify/Correct num_labels in config
                expected_num_labels = len(self.LABEL_LIST)
                if model.config.num_labels != expected_num_labels:
                    logger.warning(f"[{self.__class__.__name__}] Mismatch num_labels in model config ({model.config.num_labels}) vs. expected ({expected_num_labels}). Adjusting.")
                    model.config.num_labels = expected_num_labels
                    # Resize head if necessary (as in pos.py)
                    head = getattr(model, "classifier", None) or getattr(model, "scores", None)
                    if isinstance(head, nn.Linear) and head.out_features != expected_num_labels:
                        in_features = head.in_features
                        logger.warning(f"[{self.__class__.__name__}] Resizing classification head: {head.out_features} -> {expected_num_labels}")
                        new_head = nn.Linear(in_features, expected_num_labels)
                        std = getattr(model.config, "initializer_range", 0.02)
                        with torch.no_grad():
                            nn.init.normal_(new_head.weight, mean=0.0, std=std)
                            nn.init.zeros_(new_head.bias)
                        if hasattr(model, "classifier"): model.classifier = new_head
                        else: model.scores = new_head

        else:
            logger.error(f"[{self.__class__.__name__}] self.LABEL_LIST is empty or undefined!")

        return model

    def get_data_collator(self, tokenizer: Any):
        # "Safe" Collator to prevent raw columns (e.g., ner_tags)
        # from crashing tokenizer.pad
        return SafeDataCollatorForTokenClassification(tokenizer=tokenizer)

    def get_metric_computer(
        self,
        tokenizer: Any,
        eval_dataset_features: Dataset,
        raw_eval_dataset: Dataset,
        model: PreTrainedModel,
    ):
        # Passing fixed label list and "bio" mode
        return SeqEvalComputer(
            model=model,
            label_column=self.LABEL_COLUMN,
            token_mode=self.TOKEN_MODE,
            label_list=self.LABEL_LIST,
            tokenizer=tokenizer,
            eval_dataset_features=eval_dataset_features,
            raw_eval_dataset=raw_eval_dataset,
        )

    def get_optimization_metric(self) -> Dict[str, Any]:
        # Standard NER: optimizing overall F1
        return {"name": "overall_f1", "greater_is_better": True}

    def get_num_labels(self, train_split: Dataset) -> Optional[int]:
        # Fixed list
        return len(self.LABEL_LIST)

    def prepare_eval_dataset(self, tokenized_dataset: DatasetDict, model_config: Any) -> Optional[Dataset]:
        """Selects an evaluation split without calling unsupported kwargs."""
        # 1) Try SYSTEM utility function ("simple" signature)
        try:
            return get_eval_split_from_dict(tokenized_dataset)
        except TypeError:
            # 2) Ultra-robust fallback if signature changed on utility side
            for key in ("validation", "dev", "val", "test"):
                if key in tokenized_dataset:
                    return tokenized_dataset[key]
            logger.warning(f"[{self.__class__.__name__}] No evaluation split found. Evaluation disabled.")
            return None

    def get_preprocessing_function(self, tokenizer: PreTrainedTokenizer) -> Callable:
        """Tokenization + label alignment (robust, fixed length)."""

        max_length = getattr(tokenizer, "model_max_length", 512)
        label_all_tokens = False

        def tokenize_and_align_labels_ner(examples: Dict[str, Any]) -> Dict[str, Any]:
            # Tokenization with fixed length padding (avoids heterogeneous batches)
            tokenized = tokenizer(
                list(examples["tokens"]),
                truncation=True,
                max_length=max_length,
                padding="max_length",
                is_split_into_words=True,
            )

            aligned = []
            for i, labels_per_example in enumerate(examples[self.LABEL_COLUMN]):
                word_ids = tokenized.word_ids(batch_index=i)
                prev_word_idx = None
                label_ids: List[int] = []

                for widx in word_ids:
                    if widx is None:
                        label_ids.append(IGNORE_INDEX)
                    elif widx != prev_word_idx:
                        # First sub-token of the word
                        if 0 <= widx < len(labels_per_example):
                            label_ids.append(int(labels_per_example[widx]))
                        else:
                            logger.error(f"[NER Preproc] IndexError ex {i}: word_idx={widx} > len(labels)={len(labels_per_example)}.")
                            label_ids.append(IGNORE_INDEX)
                    else:
                        # Subsequent sub-token of the same word
                        if 0 <= widx < len(labels_per_example):
                            lab = int(labels_per_example[widx])
                            label_ids.append(lab if label_all_tokens else IGNORE_INDEX)
                        else:
                            label_ids.append(IGNORE_INDEX)

                    prev_word_idx = widx

                # Length security
                if len(label_ids) != max_length:
                    if len(label_ids) > max_length:
                        label_ids = label_ids[:max_length]
                    else:
                        label_ids = label_ids + [IGNORE_INDEX] * (max_length - len(label_ids))

                # Type check (avoid "excessive nesting")
                if not all(isinstance(x, int) for x in label_ids):
                    bad = [type(x) for x in label_ids if not isinstance(x, int)]
                    raise TypeError(f"[NER Preproc] label_ids contains non-integer types: {bad[:5]}")

                aligned.append(label_ids)

            tokenized["labels"] = aligned
            # NOTE: we do NOT REMOVE "ner_tags" here (no remove_columns on JIT side).
            # The "safe" collator handles this on the fly.
            return tokenized

        return tokenize_and_align_labels_ner
