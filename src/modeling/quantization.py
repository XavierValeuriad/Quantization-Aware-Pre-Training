# src/modeling/quantization.py (v2.0 - Consolidated and Fixed)
"""
Quantization Framework and Assembly Factory.

Implements standardized quantization operators for the LREC/DAZIP pipeline.

Supported Algorithms:
- LSQ (Learned Step Size Quantization): Esser et al. (2019)
- BitNet b1.58 (Ternary): Ma et al. (2024)
- Binary / Ternary (Legacy): Rastegari et al. (2016)

To test: `python -m doctest -v src/modeling/quantization.py`
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===================================================================
# GRADIENT UTILITIES
# ===================================================================

# Gradient utility for LSQ
def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Gradient Scale Operator.
    
    Forward: Identity (y = x)
    Backward: Scales the gradient (dL/dx = scale * dL/dy)
    
    Theoretical justification:
    Esser et al. (2020) demonstrate that the gradient magnitude for the step-size 's'
    must be normalized by 1/sqrt(N * Qp) to ensure convergence, because 's' affects
    all weights simultaneously.
    """
    y = x
    # backward: dL/dy = scale * dL/dx ; forward: y == x
    return (y - y.detach()) * scale + y.detach()

def round_pass(x: torch.Tensor) -> torch.Tensor:
    """
    Standard rounding with Straight-Through Estimator (STE).
    Forward: round(x)
    Backward: identity
    """
    return (torch.round(x) - x).detach() + x
    
# ===================================================================
# QUANTIZERS
# ===================================================================

class BaseQuantizer(nn.Module):
    """Abstract class defining the quantization interface."""
    def __init__(self, num_bits: int):
        super().__init__()
        if not isinstance(num_bits, int) or num_bits < 1:
            raise ValueError("num_bits must be a positive integer.")
        self.num_bits = num_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Each quantizer must implement its own forward method.")


class LsqQuantizer(BaseQuantizer):
    """
    Strict implementation of LSQ (Learned Step Size Quantization).
    Reference: https://openreview.net/forum?id=rkgO66VKDS
    
    Formulation:
        x_q = round(clip(x / s, n, p)) * s
    
    Where 's' is a trainable parameter that learns the optimal dynamic range
    of the weight/activation distribution.
    """
    def __init__(self, num_bits: int, layerwise: bool = True):
        super().__init__(num_bits)
        if num_bits <= 1:
            raise ValueError("LsqQuantizer is for num_bits > 1.")
        self.layerwise = layerwise
        self.s = nn.Parameter(torch.tensor(1.0))
        self.initialized = False

    @torch.no_grad()
    def _initialize_step_size(self, x: torch.Tensor):
        """
        Heuristic initialization of step-size 's' on the first pass.
        Based on the mean of absolute values: s = 2 * E(|x|) / sqrt(Q_max).
        This heuristic stabilizes the start of LSQ training.
        """
        q_max = 2**(self.num_bits - 1) - 1
        s_init = 2 * x.abs().mean() / math.sqrt(q_max)
        self.s.copy_(s_init.clamp(min=1e-8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Data-driven initialization on the fly to handle scale differences
        if not self.initialized:
            self._initialize_step_size(x)
            self.initialized = True

        q_min, q_max = -2**(self.num_bits - 1), 2**(self.num_bits - 1) - 1

        # Gradient Scaling according to Esser et al.
        # g = 1 / sqrt(N_elements * Q_max)
        g = 1.0 / math.sqrt(x.numel() * q_max) if x.numel() * q_max > 0 else 1.0
        s = grad_scale(self.s, g)
        
        # 1. Projection into quantized space
        x_scaled = x / (s + 1e-8)

        # 2. Rounding and Clamping (Implicit STE in torch.round/clamp)
        x_hat = torch.clamp(torch.round(x_scaled), q_min, q_max)

        # 3. Dequantization (Return to real scale)
        x_quant = x_hat * s

        # Straight-Through Estimator (STE) for backpropagation
        return (x_quant - x).detach() + x

class BinaryQuantizer(BaseQuantizer):
    """
    Binary Quantization (1-bit): Weights constrained to {-alpha, +alpha}.

    Source 1: "Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" by Hubara et al. (2016)
    Source 2: "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks" by Rastegari et al. (2016)
    
    >>> bq = BinaryQuantizer()
    >>> t = torch.tensor([-0.8, -0.1, 0.2, 0.9])
    >>> # Expected result is sign(t) * mean(abs(t)), i.e. [-1,-1,1,1] * 0.5
    >>> expected_output = torch.tensor([-0.5, -0.5, 0.5, 0.5])
    >>> torch.allclose(bq(t), expected_output)
    True
    """
    def __init__(self):
        super().__init__(num_bits=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = x.abs().mean()
        quant_x = alpha * torch.sign(x)
        return (quant_x - x).detach() + x

class TernaryQuantizer(BaseQuantizer):
    """
    Ternary Quantization (2-bits): Weights constrained to {-alpha, 0, +alpha}.
    Reference: Li et al., 'Ternary Weight Networks', 2016.
    Uses a threshold delta = 0.7 * E(|W|) for sparsity.
    
    >>> tq = TernaryQuantizer()
    >>> t = torch.tensor([-1.0, -0.2, 0.1, 0.8, 1.2])
    >>> # There should be at most 3 unique values in the output tensor.
    >>> torch.unique(tq(t)).numel() <= 3
    True
    """
    def __init__(self):
        super().__init__(num_bits=2) 
        self.threshold_ratio = 0.7 # Standard TWN hyperparameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Calculation of ternarization threshold
        delta = self.threshold_ratio * x.abs().mean()

        # Discretization
        quant_x = torch.where(x > delta, 1.0, 0.0)
        quant_x = torch.where(x < -delta, -1.0, quant_x)

        # Calculation of optimal scaling factors for positive and negative parts
        mask_positive = (quant_x > 0).float()
        mask_negative = (quant_x < 0).float()
        alpha_positive = torch.sum(x * mask_positive) / (torch.sum(mask_positive) + 1e-8)
        alpha_negative = torch.sum(x.abs() * mask_negative) / (torch.sum(mask_negative) + 1e-8)
        
        scaled_quant_x = alpha_positive * mask_positive - alpha_negative * mask_negative
        
        return (scaled_quant_x - x).detach() + x


class BitNetQuantizer(BaseQuantizer):
    """
    Versatile BitNet Quantizer (Weights & Activations).
    
    Supports two modes depending on configuration:
    1. 'Ternary' Mode (1.58 bits): If num_bits is None.
       - Scaling: Absolute Mean (AbsMean).
       - Values: {-1, 0, 1}.
       - Typical usage: Weights (W) in BitNet architecture.
       
    2. 'Integer' Mode (k-bits): If num_bits >= 2.
       - Scaling: Absolute Maximum (AbsMax) per-token/tensor.
       - Values: [-2^(k-1), 2^(k-1)-1] (without Zero-Point).
       - Typical usage: Activations (A) in BitNet (e.g., INT8, INT4).
    """
    def __init__(self, num_bits: Optional[int] = None, per_channel: bool = True, ch_axis: int | Tuple[int, ...] | None = None):
        # If num_bits is None, treat as "1.58 bits" (virtually 2 bits in storage)
        super().__init__(num_bits if num_bits is not None else 2)
        self.is_ternary = (num_bits is None)
        self.per_channel = per_channel
        self.ch_axis = ch_axis

    @staticmethod
    def _infer_reduce_dims(x: torch.Tensor) -> Tuple[int, ...]:
        # Inference logic for reduction dimensions
        # For Activations [Batch, Seq, Hidden] -> reduce on Hidden (-1) for per-token
        if x.ndim == 3:
            return (2,)
        # For Linear Weights [Out, In] -> reduce on In (1)
        if x.ndim == 2:
            return (1,)
        # Fallback
        return tuple(range(x.ndim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x

        # Determination of reduction dimensions for scale calculation
        if self.per_channel:
            reduce_dims = self.ch_axis if self.ch_axis is not None else self._infer_reduce_dims(x)
        else:
            reduce_dims = tuple(range(x.ndim))

        if self.is_ternary:
            # === MODE 1: TERNARY WEIGHTS (1.58 bits) ===
            # Scaling: Gamma = Mean(|W|)
            gamma = x.abs().mean(dim=reduce_dims, keepdim=True).clamp(min=1e-5).detach()
            
            # Normalization
            x_scaled = x / gamma
            
            # Rounding and Clamping {-1, 0, 1}
            x_quant = torch.clamp(torch.round(x_scaled), -1, 1)
            
            # Dequantization (STE implemented here via arithmetic operation)
            x_dequant = x_quant * gamma
            
        else:
            # === MODE 2: INT-K ACTIVATIONS (k bits) ===
            # Scaling: Gamma = Max(|X|) (AbsMax)
            # Q_p = 2^(b-1) - 1 (e.g., 127 for 8 bits, 7 for 4 bits)
            Q_p = 2**(self.num_bits - 1) - 1
            
            # Scale factor calculation (Absolute Maximum)
            gamma = x.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-5).detach()
            
            # Scaling factor to project onto integer interval
            scale = Q_p / gamma
            
            # Quantization: Round(x * scale) -> Clamping
            x_quant = torch.clamp(torch.round(x * scale), -Q_p, Q_p)
            
            # Dequantization
            x_dequant = x_quant / scale

        # Straight-Through Estimator (STE)
        return (x_dequant - x).detach() + x

# ===================================================================
# == SECTION 2: QUANTIZED LAYERS (PyTorch Wrappers)
# ===================================================================

class QuantizeLinear(nn.Linear):
    """
    Wrapper for nn.Linear injecting weight and activation quantization.
    Allows dynamically transforming a standard architecture into a quantized architecture.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_quantizer: Optional[BaseQuantizer] = None,
                 activation_quantizer: Optional[BaseQuantizer] = None):
        super().__init__(in_features, out_features, bias)
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Conditional application of quantization
        quant_weight = self.weight_quantizer(self.weight) if self.weight_quantizer else self.weight
        quant_input = self.activation_quantizer(input) if self.activation_quantizer else input
        return F.linear(quant_input, quant_weight, self.bias)

class QuantizeEmbedding(nn.Embedding):
    """Wrapper for nn.Embedding supporting lookup table quantization."""
    def __init__(self, weight_quantizer: Optional[BaseQuantizer] = None, **kwargs):
        super().__init__(**kwargs)
        self.weight_quantizer = weight_quantizer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        quant_weight = self.weight_quantizer(self.weight) if self.weight_quantizer else self.weight
        return F.embedding(input, quant_weight, self.padding_idx, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq, self.sparse)

