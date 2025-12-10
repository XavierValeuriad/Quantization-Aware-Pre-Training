# ===================================================================
# src/tasks/task_utils.py
#
# v1.0 : Utility functions refactored from workers
#        for dataset and feature manipulation.
# ===================================================================

import logging
from datasets import Dataset, DatasetDict, ClassLabel, Sequence

def get_eval_split_from_dict(dataset_dict: DatasetDict, dataset_id: str):
    """
    Finds the available evaluation split ('validation' then 'test').
    Logic extracted from trainer_factory.py.
    """
    if "validation" in dataset_dict:
        return dataset_dict["validation"]
    if "test" in dataset_dict:
        return dataset_dict["test"]

    if dataset_id.startswith("glue") or "mnli" in dataset_id:
        if "validation_matched" in dataset_dict:
            llogging.info("GLUE Fallback -> Split 'validation_matched' found.")
            return dataset_dict["validation_matched"]
        if "test_matched" in dataset_dict:
            logging.info("GLUE Fallback -> Split 'test_matched' found.")
            return dataset_dict["test_matched"]
    
    return None # Will be raised as an error by trainer_factory

def sanitize_classification_labels(ds: Dataset, num_labels: int):
    """
    Cleans out-of-range labels for classification tasks.
    Logic extracted from evaluator.py.
    """
    def _fix(batch):
        if "label" not in batch:
            return {}
        fixed = []
        for y in batch["label"]:
            if y is None:
                fixed.append(-100)  # ignore
            else:
                try:
                    yi = int(y)
                    if yi < 0 or yi >= num_labels:
                        fixed.append(-100)
                    else:
                        fixed.append(yi)
                except Exception:
                    fixed.append(-100)
        return {"label": fixed}
    return ds.map(_fix, batched=True)

def get_num_labels_from_features(train_split: Dataset, task_type: str):
    """
    Detects num_labels from dataset features.
    Logic extracted from finetuner.py.
    """
    label_feature = train_split.features.get("labels") # The tokenizer standardizes to "labels"
    if label_feature is None:
         label_feature = train_split.features.get("label") # Fallback to "label"

    if label_feature:
        if isinstance(label_feature, ClassLabel):
            # Case sequence_classification
            return label_feature.num_classes
        elif isinstance(label_feature, Sequence) and isinstance(label_feature.feature, ClassLabel):
            # Case token_classification
            return label_feature.feature.num_classes
    
    if task_type == 'token_classification':
        # Manual deduction logic from finetuner.py
        try:
            sample_size = min(1000, len(train_split))
            sample = train_split.select(range(sample_size))
            unique_labels = set(label for sublist in sample['labels'] for label in sublist)
            max_label = max(l for l in unique_labels if l != -100)
            return max_label + 1
        except Exception as e:
            logging.error(f"Failed to manually deduce `num_labels` for NER: {e}")
            return None

    return None