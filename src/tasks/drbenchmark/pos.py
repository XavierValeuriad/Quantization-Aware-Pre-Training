# ===================================================================
# src/tasks/drbenchmark/pos.py
#
# v1.0 : POS Implementation (DrBenchmark)
#        - "Safe" Collator: removes parasitic keys (e.g., pos_tags)
#        - Robust tokenization + alignment (max_length padding)
#        - prepare_eval_dataset fallback via utility + backup
#        - Safe label_list deduction (ClassLabel or ID set)
#        - Fixed id↔label mapping in model.config
# ===================================================================

import logging
from typing import Any, Callable, Dict, List, Optional

from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from torch import nn
import torch

from src.tasks.base import BaseTaskSpecification
from src.utils.task_utils import get_eval_split_from_dict
from src.metrics.computers import SeqEvalComputer 

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


# -------------------------------------------------------------------
# "Safe" Collator: removes, before padding, any key not handled by
# tokenizer.pad to avoid ValueErrors (e.g., "pos_tags").
# -------------------------------------------------------------------
class SafeDataCollatorForTokenClassification(DataCollatorForTokenClassification):
    """Identical to HF collator but cleans features of non-standard
    keys (e.g., raw dataset columns) before padding.
    """

    _allowed_keys = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "labels",
        "special_tokens_mask",
        "offset_mapping",
        "position_ids",
    }

    def torch_call(self, features):
        for f in features:
            extra_keys = [k for k in list(f.keys()) if k not in self._allowed_keys]
            for k in extra_keys:
                f.pop(k, None)
        return super().torch_call(features)


# -------------------------------------------------------------------
# DrBenchmark POS Task Specification
# -------------------------------------------------------------------
class BaseDrbPosTask(BaseTaskSpecification):
    LABEL_COLUMN = "pos_tags"
    TOKEN_MODE = "token"  # Used by SeqEvalComputer for POS

    def __init__(self):
        super().__init__()
        self._label_list: Optional[List[str]] = None
        self._label_to_id: Optional[Dict[str, int]] = None
        self._id_to_label: Optional[Dict[int, str]] = None

    # -------------------- BaseTaskSpecification ---------------------

    def get_model_class(self):
        return AutoModelForTokenClassification

    def get_num_labels(self, train_split: Dataset) -> Optional[int]:
        """Infers the label list from the training split.

        - Nominal case: features[LABEL_COLUMN] = Sequence(ClassLabel) → .feature.names
        - Fallback: dataset with integers → we aggregate observed IDs (<= 1000 ex.)
        """
        if self._label_list is not None:
            return len(self._label_list)

        try:
            feat = train_split.features[self.LABEL_COLUMN]
            # Sequence(ClassLabel) case: child object has 'names'
            if hasattr(getattr(feat, "feature", None), "names"):
                self._label_list = list(feat.feature.names)
            else:
                raise AttributeError("feature.names missing")
        except Exception:
            # Fallback: collect seen IDs (integers >= 0) on a sample
            ids = set()
            n = min(1000, len(train_split))
            for ex in train_split.select(range(n)):
                for i in ex.get(self.LABEL_COLUMN, []):
                    if isinstance(i, int) and i >= 0:
                        ids.add(i)
            if not ids:
                raise ValueError(f"Unable to infer POS label list from '{self.LABEL_COLUMN}'.")
            max_id = max(ids)
            self._label_list = [f"LABEL_{i}" for i in range(max_id + 1)]

        self._label_to_id = {lab: i for i, lab in enumerate(self._label_list)}
        self._id_to_label = {i: lab for i, lab in enumerate(self._label_list)}
        return len(self._label_list)

    def post_init_model(self, model, train_split):
        # Ensure label_list is fixed
        if self._label_list is None:
            self.get_num_labels(train_split)

        expected = len(self._label_list or [])
        # id↔label mapping in config
        if hasattr(model, "config"):
            model.config.label2id = dict(self._label_to_id or {})
            model.config.id2label = {int(i): str(l) for i, l in (self._id_to_label or {}).items()}
            model.config.num_labels = expected

        # --- Verify and (if needed) resize head ---
        head = getattr(model, "classifier", None) or getattr(model, "scores", None)
        if isinstance(head, nn.Linear):
            out_features = head.out_features
            if out_features != expected and expected > 0:
                in_features = head.in_features
                logger.warning(f"[POS] Resizing classification head: {out_features} -> {expected}")
                new_head = nn.Linear(in_features, expected)
                # Clean init
                std = getattr(model.config, "initializer_range", 0.02)
                with torch.no_grad():
                    nn.init.normal_(new_head.weight, mean=0.0, std=std)
                    nn.init.zeros_(new_head.bias)
                # Replace standard attribute (BERT/RoBERTa/DeBERTa…)
                if hasattr(model, "classifier"):
                    model.classifier = new_head
                else:
                    model.scores = new_head
        else:
            logger.warning("[POS] Unable to identify classification head; assuming it is correct.")

        return model

    def get_data_collator(self, tokenizer: Any):
        # "Safe" Collator to tolerate raw columns (e.g., pos_tags)
        return SafeDataCollatorForTokenClassification(tokenizer=tokenizer)

    def get_metric_computer(
        self,
        tokenizer: Any,
        eval_dataset_features: Dataset,
        raw_eval_dataset: Dataset,
        model: PreTrainedModel,
    ):
        if not self._label_list:
            raise ValueError("label_list non initialisée pour POS.")
        return SeqEvalComputer(
            model=model,
            label_column=self.LABEL_COLUMN,
            token_mode=self.TOKEN_MODE,
            label_list=self._label_list,
            tokenizer=tokenizer,
            eval_dataset_features=eval_dataset_features,
            raw_eval_dataset=raw_eval_dataset,
        )

    def get_optimization_metric(self) -> Dict[str, Any]:
        # POS: Overall F1
        return {"name": "overall_f1", "greater_is_better": True}

    def prepare_eval_dataset(self, tokenized_dataset: DatasetDict, model_config: Any) -> Optional[Dataset]:
        """Selects an evaluation split robustly."""
        try:
            return get_eval_split_from_dict(tokenized_dataset)
        except TypeError:
            for key in ("validation", "dev", "val", "test"):
                if key in tokenized_dataset:
                    return tokenized_dataset[key]
            logger.warning("[POS] No evaluation split found. Evaluation disabled.")
            return None

    def get_preprocessing_function(self, tokenizer: PreTrainedTokenizer) -> Callable:
        """Tokenization (pre-split tokens) + POS label alignment.

        - Fixed length padding (= tokenizer.model_max_length) for homogeneous batches.
        - Standard alignment: only the first sub-token of a word gets the label,
          subsequent ones get IGNORE_INDEX (unless label_all_tokens=True).
        """
        max_length = getattr(tokenizer, "model_max_length", 512)
        label_all_tokens = False

        def tokenize_and_align_labels_pos(examples: Dict[str, Any]) -> Dict[str, Any]:
            enc = tokenizer(
                list(examples["tokens"]),
                truncation=True,
                max_length=max_length,
                padding="max_length",
                is_split_into_words=True,
            )

            labels_batch: List[List[int]] = []

            for i, lab_seq in enumerate(examples[self.LABEL_COLUMN]):
                word_ids = enc.word_ids(batch_index=i)
                prev_w = None
                aligned: List[int] = []

                for w in word_ids:
                    if w is None:
                        aligned.append(IGNORE_INDEX)
                    elif w != prev_w:
                        # First sub-token of the word
                        if 0 <= w < len(lab_seq):
                            aligned.append(int(lab_seq[w]))
                        else:
                            logger.error(f"[POS Preproc] IndexError ex {i}: word_idx={w} > len(labels)={len(lab_seq)}.")
                            aligned.append(IGNORE_INDEX)
                    else:
                        # Extra sub-token of the same word
                        if 0 <= w < len(lab_seq):
                            lab_id = int(lab_seq[w])
                            aligned.append(lab_id if label_all_tokens else IGNORE_INDEX)
                        else:
                            aligned.append(IGNORE_INDEX)
                    prev_w = w

                # Length security
                if len(aligned) != max_length:
                    if len(aligned) > max_length:
                        aligned = aligned[:max_length]
                    else:
                        aligned.extend([IGNORE_INDEX] * (max_length - len(aligned)))

                # Type check
                if not all(isinstance(x, int) for x in aligned):
                    bad = [type(x) for x in aligned if not isinstance(x, int)]
                    raise TypeError(f"[POS Preproc] labels alignés contiennent des types non-entiers: {bad[:5]}")

                labels_batch.append(aligned)

            enc["labels"] = labels_batch
            # IMPORTANT: we DO NOT remove "pos_tags" here (no JIT remove_columns).
            # The "safe" collator will remove parasitic columns before padding.
            return enc

        return tokenize_and_align_labels_pos


# Concrete implementations (aliases, if multiple DrBenchmark variants exist)
class CasTask(BaseDrbPosTask):
    def __init__(self):
        super().__init__()


class EssaiTask(BaseDrbPosTask):
    def __init__(self):
        super().__init__()
