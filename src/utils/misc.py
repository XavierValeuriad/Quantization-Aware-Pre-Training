# ===================================================================
# src/utils/misc.py
#
# v1.0 : Added torch.Tensor handling
#        - JSON serializer for numpy/datetime/torch types
# ===================================================================

import numpy as np
import torch 
import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def default_json_serializer(obj: object) -> any:
    """
    Custom JSON serializer to handle non-standard data types
    commonly encountered in scientific pipelines (numpy, torch, datetime).

    To be used with `json.dump(..., default=default_json_serializer)`.

    Args:
        obj (object): The object to serialize.

    Returns:
        any: A serializable representation of the object (int, float, list, str).
    
    Raises:
        TypeError: If the type is not handled, lets json.dump raise the error.
    """
    
    # --- Numpy Types ---
    if isinstance(obj, np.integer):
        # Converts numpy integers (e.g., np.int64) to Python int
        return int(obj)
    elif isinstance(obj, np.floating):
        # Converts numpy floats (e.g., np.float32) to Python float
        return float(obj)
    elif isinstance(obj, np.ndarray):
        # Converts numpy arrays to Python lists
        return obj.tolist()
    
    # --- PyTorch Types ---
    elif isinstance(obj, torch.Tensor):
        # Detach the tensor from the computation graph (just in case)
        obj = obj.detach()
        
        if obj.numel() == 1:
            # If it is a scalar (e.g., loss, accuracy), extract the Python value
            return obj.item()
        else:
            # If it is a vector or more, convert to list
            # Ensure it is on CPU before .tolist()
            return obj.cpu().tolist()
    
    # --- Datetime Types ---
    elif isinstance(obj, (datetime.datetime, datetime.date)):
        # Converts dates/datetimes to ISO format string
        return obj.isoformat()
    
    # --- Bytes Types ---
    elif isinstance(obj, bytes):
        # Attempts to decode bytes to UTF-8, otherwise returns a representation
        try:
            return obj.decode('utf-8')
        except UnicodeDecodeError:
            return f"[bytes_data_len:{len(obj)}]"

    # --- Pathlib Handling ---
    elif isinstance(obj, Path):
        # Explicitly converts Path objects to string
        return obj.as_posix()

    # --- Unhandled Case ---
    logger.warning(f"default_json_serializer: Unhandled type encountered: {type(obj)}. Attempting str conversion.")    
    # Stratégie de dernier recours : tenter une conversion en str

    # Last resort strategy: attempt conversion to str
    try:
        return str(obj)
    except Exception:
        # If even str() fails, we raise the error for the caller (json.dump) to catch
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable (even with str())")