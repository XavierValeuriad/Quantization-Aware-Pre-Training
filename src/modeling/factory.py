# src/modeling/factory.py (v1.0)
""" Assembly Factory for creating quantized models (Hybrid Precision).

HYBRID QUANTIZATION STRATEGY (W1.58 / A8 / E8): This module implements a hardware-optimized 
mixed-precision strategy, based on state-of-the-art recommendations:

EMBEDDINGS (Int8): Maintaining high precision (8-bit) for embedding layers.
Justification: Aggressive quantization (<4 bits) of the embedding space causes irreversible 
"Semantic Collapse" even before model processing (Ma et al., 2024; "The Era of 1-bit LLMs").

LINEAR WEIGHTS (Ternary / 1.58-bit): Maximum compression ({-1, 0, 1}) for projection 
matrices (Wq, Wk, Wv, FFN), which represent >90% of the memory footprint.

ACTIVATIONS (Int8): 8-bit quantization of activations (LSQ). Justification:
- Hardware: Enables use of INT8 Tensor Cores on modern GPUs and prepares for efficient 
  inference on FPGA/ASIC (Integer Accumulators).
- Stability: Dettmers et al. ("LLM.int8()", 2022) demonstrate that outliers in activations 
  require at least 8 bits of precision to avoid destabilizing gradient flow.

To test: `python -m doctest -v src/modeling/factory.py`
"""
import logging
from typing import Optional, Dict, Any, List
import torch.nn as nn
import torch 

from src.modeling.quantization import (
    BaseQuantizer, LsqQuantizer, BinaryQuantizer, TernaryQuantizer,
    BitNetQuantizer, QuantizeLinear, QuantizeEmbedding
)

# Maps YAML strings to torch.dtype
DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16
}

class CastQuantizer(nn.Module):
    """
    Pseudo-quantizer that simply applies a precision cast.
    Activated when the quantization config is 'null' for a module.
    """
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamic runtime cast
        return x.to(self.dtype)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_quantizer(config: dict, target_dtype: torch.dtype) -> Optional[BaseQuantizer]:
    """
    Creates a quantizer instance from a configuration dictionary.
    """
    # Case 1: No quantization config -> Apply global precision
    if not config:
        return CastQuantizer(target_dtype)
    
    algo = config.get("algo")
    # Retrieve 'bits' without forcing its presence (as it is optional for BitNet/Ternary)
    bits = config.get("bits", None) 

    if algo is None:
        raise ValueError("Quantizer configuration must specify an 'algo'.")

    if algo == "BitNet_b158":
        # If 'bits' is present (e.g., 4, 8), activate INT-k mode (Activations)
        # If 'bits' is missing or null, activate Ternary mode (Weights)
        return BitNetQuantizer(num_bits=bits)
        
    elif algo == "LSQ":
        if bits is None:
             raise ValueError("LSQ algorithm requires a 'bits' parameter.")
        return LsqQuantizer(num_bits=bits)
        
    elif algo == "Ternary":
        return TernaryQuantizer()
        
    elif algo == "Binary":
        return BinaryQuantizer()
        
    else:
        raise NotImplementedError(f"Quantization algorithm '{algo}' is not recognized.")

def _replace_layers_recursive(
    model: nn.Module,
    embedding_quantizer: Optional[BaseQuantizer],
    weight_quantizer: Optional[BaseQuantizer],
    activation_quantizer: Optional[BaseQuantizer],
    ignore_modules: Optional[List[str]] = None
):
    """Internal function that traverses the model and replaces layers."""
    for name, module in model.named_children():

        # --- EXCLUSION LOGIC (PROBE/HEAD PROTECTION) ---
        if ignore_modules and any(ignored in name for ignored in ignore_modules):
            logging.debug(f"Skipping quantization for ignored module: {name}")
            continue
        # -----------------------------------------------

        if isinstance(module, nn.Embedding):
            embedding_args = {
                "num_embeddings": module.num_embeddings, "embedding_dim": module.embedding_dim,
                "padding_idx": module.padding_idx, "max_norm": module.max_norm,
                "norm_type": module.norm_type, "scale_grad_by_freq": module.scale_grad_by_freq,
                "sparse": module.sparse,
            }
            new_emb = QuantizeEmbedding(
                weight_quantizer=embedding_quantizer, **embedding_args
            ).to(module.weight.device)
            new_emb.load_state_dict(module.state_dict(), strict=False)
            setattr(model, name, new_emb)
            
        elif isinstance(module, nn.Linear):
            new_lin = QuantizeLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                weight_quantizer=weight_quantizer,
                activation_quantizer=activation_quantizer,
            ).to(module.weight.device)
            new_lin.load_state_dict(module.state_dict(), strict=False)
            setattr(model, name, new_lin)
        
        elif len(list(module.children())) > 0:
            _replace_layers_recursive(module, embedding_quantizer, weight_quantizer, activation_quantizer)


def _retie_lm_head_if_present(model):
    try:
        emb = model.roberta.embeddings.word_embeddings
        dec = model.lm_head.decoder
        if dec.weight.data_ptr() != emb.weight.data_ptr():
            dec.weight = emb.weight  # partager le poids
    except AttributeError:
        pass



def apply_quantization_scheme(
    model: nn.Module, 
    quant_scheme: Dict[str, Any],
    ignore_modules: Optional[List[str]] = None
) -> nn.Module:
    """
    Main entry point. Takes a model and a quantization plan,
    and returns the transformed model.

    >>> # --- Definition of test prerequisites ---
    >>> class MiniModel(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.emb = nn.Embedding(10, 20)
    ...         self.lin = nn.Linear(20, 30)
    >>> model = MiniModel()
    >>> # Define an assembly plan (quant_scheme)
    >>> scheme = { "name": "Test", "enabled": True, \\
    ...            "embedding_quant": {"algo": "LSQ", "bits": 8}, \\
    ...            "weight_quant": {"algo": "Ternary"}, \\
    ...            "activation_quant": {"algo": "LSQ", "bits": 8} }
    >>> # --- Execution of the function under test ---
    >>> q_model = apply_quantization_scheme(model, scheme)
    >>> # --- Validation of results ---
    >>> # 1. Has the Linear layer been replaced?
    >>> isinstance(q_model.lin, QuantizeLinear)
    True
    >>> # 2. Was the correct weight quantizer injected?
    >>> isinstance(q_model.lin.weight_quantizer, TernaryQuantizer)
    True
    >>> # 3. Was the correct activation quantizer injected?
    >>> isinstance(q_model.lin.activation_quantizer, LsqQuantizer)
    True
    >>> # 4. Are the quantizer parameters correct?
    >>> q_model.lin.activation_quantizer.num_bits
    8
    """
    if not quant_scheme.get("enabled", False):
        logging.info("Quantization scheme disabled. The model remains unchanged.")
        return model

    # 1. Determination of global target precision
    prec_str = quant_scheme.get("precision", "bf16") # Default BF16
    target_dtype = DTYPE_MAP.get(prec_str, torch.bfloat16)

    logging.info(f"Applying quantization scheme: {quant_scheme.get('name', 'Unnamed')} | Target Precision: {prec_str}")
    if ignore_modules:
        logging.info(f"Modules excluded from quantization: {ignore_modules}")
    
    embedding_quantizer = create_quantizer(quant_scheme.get("embedding_quant"), target_dtype)
    weight_quantizer = create_quantizer(quant_scheme.get("weight_quant"), target_dtype)
    activation_quantizer = create_quantizer(quant_scheme.get("activation_quant"), target_dtype)

    _replace_layers_recursive(
        model, 
        embedding_quantizer, 
        weight_quantizer, 
        activation_quantizer,
        ignore_modules=ignore_modules
    )

    if hasattr(model, "tie_weights") and callable(getattr(model, "tie_weights")):
        model.tie_weights()

    _retie_lm_head_if_present(model)
    
    logging.info("Model transformation completed.")
    return model
