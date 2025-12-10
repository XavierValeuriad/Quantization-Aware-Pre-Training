# ===================================================================
# src/metrics/computers.py
#
# v1.0 : Compliant - Purged Imports.
#        Contains CallableMetricsComputer, SeqEvalComputer, GenericAccuracyComputer.
# ===================================================================

import logging
import numpy as np
import os
from typing import Dict, Any, Callable, List, Literal
import re
import torch

from transformers import EvalPrediction, PreTrainedModel , PreTrainedTokenizer
from datasets import Dataset
import evaluate

from src.utils.path_manager import get_project_root
from src.metrics.base_computer import BaseMetricsComputer

# Imports for complex computers (SeqEvalComputer)
try:
    # Attempt to import external libs
    from seqeval.metrics import f1_score as seq_f1, precision_score as seq_prec, recall_score as seq_rec
    from sklearn.metrics import f1_score as sk_f1, precision_score as sk_prec, recall_score as sk_rec, accuracy_score
    _libs_available = True
except ImportError:
    # If missing, SeqEvalComputer will raise an error at init
    _libs_available = False
    # Define placeholders to avoid NameError if SeqEvalComputer is imported but not used
    seq_f1 = seq_prec = seq_rec = sk_f1 = sk_prec = sk_rec = accuracy_score = None

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Path Logic for Local Metrics ---
try:
    PROJECT_ROOT = get_project_root()
    METRICS_PATH = os.path.join(PROJECT_ROOT, "vendor/evaluate/metrics")
    logger.info(f"[Computers] : Metrics submodule path: {METRICS_PATH}")
    # Vérifier si le chemin et le script 'accuracy' existent
    _accuracy_path = os.path.join(METRICS_PATH, "accuracy")
    if not os.path.isdir(METRICS_PATH) or not os.path.exists(os.path.join(_accuracy_path, "accuracy.py")):
        logger.warning(f"Metrics submodule path '{METRICS_PATH}' or 'accuracy' metric invalid.")
        METRICS_PATH = None # Invalider si chemin ou métrique manque
    else:
        logger.info(f"[Computers] : OK Metrics submodule path: {METRICS_PATH}")
except Exception as e:
    logger.critical(f"Failed to define metrics submodule path: {e}")
    METRICS_PATH = None
# --------------------------------------------------------------------------


class GenericMcqaComputer(BaseMetricsComputer):
    """
    Computer for Multiple Choice Question tasks (MMLU, GPQA, ARC).
    Main metric: Accuracy.
    """
    def __call__(self, p: EvalPrediction) -> Dict[str, float]:
        # Les prédictions pour MCQA sont des logits [Batch, Num_Choices]
        # On veut l'index du choix ayant le score le plus élevé.
        preds = p.predictions
        
        # If it's a tuple (loss, logits), take logits
        if isinstance(preds, tuple):
            preds = preds[0]
            
        # Numpy conversion if necessary
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(p.label_ids, torch.Tensor):
            label_ids = p.label_ids.detach().cpu().numpy()
        else:
            label_ids = p.label_ids

        # Argmax to find the predicted class
        predicted_indices = np.argmax(preds, axis=1)
        
        # Simple Accuracy calculation
        # Element-wise comparison
        match = (predicted_indices == label_ids)
        accuracy = match.astype(np.float32).mean().item()
        
        return {"accuracy": accuracy}



class CallableMetricsComputer(BaseMetricsComputer):
    """Adapts a simple metric function to the BaseMetricsComputer interface."""
    def __init__(self, compute_fn: Callable, **kwargs):
        super().__init__(**kwargs)
        # Assurer que les arguments nécessaires sont passés à __init__
        assert 'tokenizer' in kwargs, "CallableMetricsComputer nécessite 'tokenizer'"
        assert 'eval_dataset_features' in kwargs, "CallableMetricsComputer nécessite 'eval_dataset_features'"
        assert 'raw_eval_dataset' in kwargs, "CallableMetricsComputer nécessite 'raw_eval_dataset'"
        self._compute_fn = compute_fn

    def __call__(self, p: EvalPrediction) -> Dict[str, float]:
        # Simply calls the provided function
        # Note: The function must have the signature (EvalPrediction) -> Dict
        return self._compute_fn(p)

class GenericAccuracyComputer(BaseMetricsComputer):
    """Generic computer to calculate accuracy (e.g., simple MCQ)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metric = None # Initialize to None
        metric_name = "accuracy"
        
        # Try loading from submodule first
        if KAIROS_METRICS_PATH:
            try:
                metric_path = os.path.join(KAIROS_METRICS_PATH, metric_name)
                self.metric = evaluate.load(metric_path)
                logger.info(f"Metric '{metric_name}' loaded from submodule: {metric_path}")
            except Exception as e:
                logger.error(f"Failed loading '{metric_name}' from submodule ({metric_path}): {e}. Attempting HF Hub fallback.")
                self.metric = None # Ensure None if failed
        
        # Fallback to HF Hub if submodule fails or is undefined
        if self.metric is None:
            try:
                self.metric = evaluate.load(metric_name)
                logger.warning(f"Metric '{metric_name}' loaded from HF Hub (fallback).")
            except Exception as e:
                logger.critical(f"Critical failure loading metric '{metric_name}' from submodule AND HF Hub: {e}")
                # __call__ will handle self.metric == None

    def __call__(self, eval_preds):
        if self.metric is None: 
            logger.error(f"Metric 'accuracy' not loaded. Calculation impossible.")
            return {"accuracy": 0.0, "metric_error": 1.0}
        
        predictions = eval_preds.predictions
        label_ids = eval_preds.label_ids

        preds = np.argmax(predictions, axis=1)
        if label_ids is None: 
            logger.warning("AccuracyComputer: label_ids is None.")
            return {"accuracy": 0.0}
        references = np.array(label_ids).flatten()
        if preds.shape != references.shape: 
            logger.error(f"AccuracyComputer: Mismatch shapes! preds: {preds.shape}, refs: {references.shape}")
            return {"accuracy": 0.0, "metric_error": 1.0}
            
        try:
            return self.metric.compute(predictions=preds, references=references)
        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}", exc_info=True)
            return {"accuracy": 0.0, "metric_error": 1.0}

TokenMode = Literal["bio", "token"]
BIO_TAG_RE = re.compile(r"^(?:O|[BI]-[A-Za-z0-9_.-]+)$")
IGNORE_INDEX = -100

class SeqEvalComputer(BaseMetricsComputer):
    """
    Computer (v1.0) for Token Classification (NER/POS) via seqeval or sklearn.
    Uses manual __init__ to manage inheritance.
    """

    def __init__(self, 
                 # Arguments specific to SeqEvalComputer
                 model: PreTrainedModel, 
                 label_column: str, 
                 token_mode: TokenMode, 
                 label_list: List[str], 
                 # Arguments required by BaseMetricsComputer
                 tokenizer: PreTrainedTokenizer, 
                 eval_dataset_features: Dataset, 
                 raw_eval_dataset: Dataset, 
                 **kwargs # To capture potential extra arguments
                 ):
        
        # 1. Explicit call to parent init
        super().__init__(tokenizer=tokenizer, 
                         eval_dataset_features=eval_dataset_features, 
                         raw_eval_dataset=raw_eval_dataset, 
                         **kwargs) # Also pass potential kwargs
        
        # 2. Initialization of SeqEvalComputer specific attributes
        self.model = model
        self.label_column = label_column
        self.token_mode = token_mode
        self.label_list = label_list
        # Attribute for id -> name mapping (will be defined below)
        self.id2name: Dict[int, str] = {} 

        # 3. Validation and initialization logic (former __post_init__)
        if not _libs_available:
            raise ImportError("'seqeval' and 'scikit-learn' libraries are required for SeqEvalComputer.")
             
        # Validation/Fallback for label_list
        if not self.label_list:
            logger.warning(f"SeqEvalComputer: label_list initially empty for {self.label_column}. Attempting fallback via model.config...")
            id2label = getattr(self.model.config, "id2label", None)
            if isinstance(id2label, dict) and id2label:
                # Ensure keys are integers and sorted
                try:
                    max_id = max(int(k) for k in id2label.keys())
                    # Create list in ID order
                    self.label_list = [id2label[str(i)] for i in range(max_id + 1)] 
                    logger.warning(f"SeqEvalComputer: label_list retrieved from model.config ({len(self.label_list)} labels).")
                except Exception as e:
                    logger.error(f"Error reconstructing label_list from id2label: {e}", exc_info=True)
                    raise ValueError(f"SeqEvalComputer requires a valid 'label_list'. Fallback model.config failed. id2label: {id2label}")
            else:
                raise ValueError("SeqEvalComputer requires a valid 'label_list' and model.config fallback failed.")
                 
        # BIO tags validation if necessary
        if self.token_mode == "bio":
            bad_tags = [tag for tag in self.label_list if not BIO_TAG_RE.match(str(tag))]
            if bad_tags:
                logger.error(f"Non-compliant BIO tags detected: {bad_tags}")
                raise ValueError(f"SeqEvalComputer(mode='bio'): non-BIO tags found: {bad_tags}")

        # Creation of id -> name mapping
        logger.info(f"[SeqEvalComputer] Initialized: mode={self.token_mode}, labels={len(self.label_list)} for column '{self.label_column}'.")
        try:
            self.id2name = {i: name for i, name in enumerate(self.label_list)}
            if len(self.id2name) != len(self.label_list):
                 logger.warning("Possible inconsistency during id2name creation (duplicates?).")
        except TypeError as e:
            logger.error(f"Error creating id2name. Invalid label_list? Type: {type(self.label_list)}, Content (partial): {self.label_list[:10]}...", exc_info=True)
            raise ValueError(f"id2name Error: {e}") from e
        except Exception as e: # Catch other potential errors
             logger.error(f"Unexpected error during id2name creation: {e}", exc_info=True)
             raise

    def __call__(self, p: EvalPrediction) -> Dict[str, float]:
        predictions, label_ids = p.predictions, p.label_ids
        
        # Shape validation and pred_ids calculation
        if not (isinstance(predictions, np.ndarray) and predictions.ndim == 3):
            logger.error(f"SeqEvalComputer: Invalid prediction format (expected logits). Shape: {getattr(predictions, 'shape', 'N/A')}")
            return {"overall_f1": 0.0, "metric_error": 1.0, "error_detail": "Invalid prediction format (expected logits)"}
        pred_ids = np.argmax(predictions, axis=2)
             
        if not (isinstance(label_ids, np.ndarray) and label_ids.ndim == 2):
            logger.error(f"SeqEvalComputer: Invalid label format. Shape: {getattr(label_ids, 'shape', 'N/A')}")
            return {"overall_f1": 0.0, "metric_error": 1.0, "error_detail": "Invalid label_ids format"}
            
        if pred_ids.shape != label_ids.shape:
            logger.error(f"SeqEvalComputer: Shape mismatch! Preds: {pred_ids.shape}, Labels: {label_ids.shape}")
            return {"overall_f1": 0.0, "metric_error": 1.0, "error_detail": "Shape mismatch preds/labels"}

        # Construction of label lists (strings) ignoring IGNORE_INDEX
        y_true_list, y_pred_list = [], []
        num_sequences = label_ids.shape[0]
        seq_len = label_ids.shape[1]

        for i in range(num_sequences):
            true_seq_tokens, pred_seq_tokens = [], []
            for j in range(seq_len):
                label_id = label_ids[i, j]
                pred_id = pred_ids[i, j]

                if label_id != IGNORE_INDEX:
                    true_label_name = self.id2name.get(label_id, "O")  # Fallback to "O" if ID unknown
                    pred_label_name = self.id2name.get(pred_id, "O")  # Fallback to "O"
                    true_seq_tokens.append(true_label_name)
                    pred_seq_tokens.append(pred_label_name)
            
            # Only add if the reference sequence is not empty after filtering
            if true_seq_tokens: 
                 y_true_list.append(true_seq_tokens)
                 y_pred_list.append(pred_seq_tokens)

        # Handling case where no valid tag is found
        if not y_true_list:
            logger.warning(f"SeqEvalComputer: No valid tags found after filtering {IGNORE_INDEX} for column '{self.label_column}'.")
            metrics_zero = {"overall_f1": 0.0, "overall_precision": 0.0, "overall_recall": 0.0, "metric_error": 0.0} # Not an error, just 0
            if self.token_mode != "bio": 
                metrics_zero["token_accuracy"] = 0.0
            return metrics_zero

        # Metric calculation with seqeval or sklearn
        try:
            if self.token_mode == "bio":
                results = {
                    "overall_f1": seq_f1(y_true_list, y_pred_list, zero_division=0),
                    "overall_precision": seq_prec(y_true_list, y_pred_list, zero_division=0),
                    "overall_recall": seq_rec(y_true_list, y_pred_list, zero_division=0),
                }
            else: # token_mode == "token"
                y_true_flat = [item for sublist in y_true_list for item in sublist]
                y_pred_flat = [item for sublist in y_pred_list for item in sublist]
                # Use 'weighted' to handle potential class imbalance (frequent in POS)
                results = {
                    "overall_f1":        sk_f1(y_true_flat, y_pred_flat, average="weighted", zero_division=0),
                    "overall_precision": sk_prec(y_true_flat, y_pred_flat, average="weighted", zero_division=0),
                    "overall_recall":    sk_rec(y_true_flat, y_pred_flat, average="weighted", zero_division=0),
                    "token_accuracy":    accuracy_score(y_true_flat, y_pred_flat),
                }
            # Add success flag to differentiate from error
            results["metric_error"] = 0.0 
            return results
            
        except Exception as e:
            logger.error(f"Error calculating metrics ({self.token_mode} mode): {e}", exc_info=True)
            error_metrics = {"overall_f1": 0.0, "metric_error": 1.0}
            if self.token_mode != "bio": error_metrics["token_accuracy"] = 0.0
            return error_metrics

class MultiLabelMetricsComputer(BaseMetricsComputer):
    """
    Computer for Multi-label classification (e.g., MORFITT).
    Handles Logits -> Sigmoid -> Threshold -> Metrics conversion.
    """
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def __call__(self, p: EvalPrediction) -> Dict[str, float]:
        predictions = p.predictions
        labels = p.label_ids

        # Sigmoid + Thresholding
        probs = 1.0 / (1.0 + np.exp(-predictions)) # Numpy Sigmoid
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= self.threshold)] = 1

        # Metric calculation (Weighted & Macro)
        # Note: Zero_division=0 to avoid warnings on empty classes
        try:
            f1_macro = sk_f1(labels, y_pred, average='macro', zero_division=0)
            f1_micro = sk_f1(labels, y_pred, average='micro', zero_division=0)
            f1_weighted = sk_f1(labels, y_pred, average='weighted', zero_division=0)
            acc = accuracy_score(labels, y_pred)
            
            # ROC AUC requires probabilities, not binary labels, 
            # and can fail if a class is never present.
            try:
                roc_auc = 0.0 # roc_auc_score(labels, probs, average='micro') # Disabled for safety for now
            except:
                roc_auc = 0.0

            return {
                "f1_macro": f1_macro,
                "f1_micro": f1_micro,
                "f1_weighted": f1_weighted,
                "accuracy": acc,
                # "roc_auc": roc_auc
            }
        except Exception as e:
            logger.error(f"Error calculating Multi-Label metrics: {e}", exc_info=True)
            return {"f1_weighted": 0.0, "metric_error": 1.0}

class GenerativeMetricsComputer(BaseMetricsComputer):
    """
    Computer for text generation tasks (Seq2Seq / Causal LM).
    Handles BLEU, ROUGE, and Exact Match (useful for Code/Math).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.bleu = evaluate.load("bleu")
            self.rouge = evaluate.load("rouge")
            self.exact_match = evaluate.load("exact_match")
        except Exception as e:
            logger.warning(f"GenerativeMetricsComputer: Impossible de charger certaines métriques HF: {e}")
            self.bleu = self.rouge = self.exact_match = None

    def __call__(self, p: EvalPrediction) -> Dict[str, float]:
        # In generation mode, 'predictions' contains generated token_ids
        preds = p.predictions
        labels = p.label_ids

        # If preds is a tuple (loss, logits), take logits or generated tokens
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 with pad_token_id for decoding
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)

        # Decode to text
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Simple cleanup
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        results = {}
        
        # Calculate Exact Match (Math / Short Code)
        if self.exact_match:
            em_score = self.exact_match.compute(predictions=decoded_preds, references=decoded_labels)
            results["exact_match"] = em_score["exact_match"]

        # Calculate BLEU (Translation / Paraphrase)
        if self.bleu:
            # BLEU expects lists of references
            bleu_score = self.bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
            results["bleu"] = bleu_score["bleu"]

        # Calculate ROUGE (Summary / Long Context)
        if self.rouge:
            rouge_score = self.rouge.compute(predictions=decoded_preds, references=decoded_labels)
            results.update(rouge_score)

        # Average generation length (diagnostic)
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        results["gen_len"] = np.mean(prediction_lens)

        return results

class AudioMetricsComputer(BaseMetricsComputer):
    """
    Computer for Automatic Speech Recognition (ASR).
    Mainly WER (Word Error Rate).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.wer = evaluate.load("wer")
        except Exception as e:
            logger.error(f"AudioMetricsComputer: Unable to load WER: {e}")
            self.wer = None

    def __call__(self, p: EvalPrediction) -> Dict[str, float]:
        if not self.wer: return {"wer": 0.0, "error": 1.0}

        preds = p.predictions
        labels = p.label_ids
        
        # Specific -100 handling for CTC/Seq2Seq ASR
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        wer_score = self.wer.compute(predictions=decoded_preds, references=decoded_labels)
        return {"wer": wer_score}