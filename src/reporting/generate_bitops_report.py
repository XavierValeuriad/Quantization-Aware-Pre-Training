
# ===================================================================
# src/reporting/generate_bitops_report.py
# ===================================================================

import math
import yaml
import logging
from pathlib import Path
from src.utils.path_manager import get_results_root, get_project_root

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_complexity_metrics(config_path: Path):
    """
    Generates a detailed report with 3 distinct gain metrics for the LREC/ACL paper.
    1. Size Gain: Memory compression (VRAM/Disk).
    2. BitOps Gain: Reduction in theoretical arithmetic complexity (W*A).
    3. LogicOps Gain: Reduction in silicon/FPGA surface area (with bonus for multiplier-free BitNet).
    """ 
    
    print(f"{'='*100}")
    print(f"COMPLEXITY CALCULATOR - BERT BASE")
    print(f"{'='*100}\n")

    # --- 1. CONFIG LOADING ---
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
        
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # Keep only enabled schemes
    active_schemes = [s for s in yaml_config.get("quantization", []) if s.get("enabled", False)]

    # --- 2. CONSTANTS (BERT Base / Seq=512) ---
    # These values correspond to the exact architecture of CamemBERT/DrBERT
    VOCAB_SIZE = 30522
    HIDDEN_SIZE = 768
    MAX_POSITION_EMBEDDINGS = 512
    TYPE_VOCAB_SIZE = 2
    NUM_HIDDEN_LAYERS = 12
    INTERMEDIATE_SIZE = 3072 
    SEQ_LEN = 512 # Worst Case Scenario for FPGA sizing

    # --- 3. PARAMETERS ---
    n_emb = (VOCAB_SIZE * HIDDEN_SIZE) + (MAX_POSITION_EMBEDDINGS * HIDDEN_SIZE) + (TYPE_VOCAB_SIZE * HIDDEN_SIZE)
    # Weights per encoder layer (Attention + FFN)
    params_per_layer = (4 * HIDDEN_SIZE * HIDDEN_SIZE) + (2 * HIDDEN_SIZE * INTERMEDIATE_SIZE)
    n_body = NUM_HIDDEN_LAYERS * params_per_layer
    n_pooler = HIDDEN_SIZE * HIDDEN_SIZE
    n_total = n_emb + n_body + n_pooler
    
    # --- 4. MACs (Multiply-Accumulate Operations) ---
    # Dominant linear approximation for Transformers (N * L)
    macs_per_inference = n_body * SEQ_LEN 

    print(f"--- BREAKDOWN ---")
    print(f"Total Params : {n_total/1e6:.1f} M")
    print(f"MACs (L={SEQ_LEN}) : {macs_per_inference/1e9:.2f} GMacs (Worst Case)\n")
    
    # --- 5. INITIALIZE RESULTS ---
    results = []

    # === FP32 BASELINE (Absolute Reference) ===
    baseline_size_mb = (n_total * 32) / 8 / (1024 * 1024)
    # Standard cost: N * 32 * 32
    baseline_ops = macs_per_inference * 32 * 32 
    
    results.append({
        "name": "FP32 Baseline (Ref.)",
        "b_emb": 32, "b_w": 32, "b_a": 32,
        "size_mb": baseline_size_mb,
        "bitops": baseline_ops / 1e12,
        "logicops": baseline_ops / 1e12,
        "gain_size": 1.0,
        "gain_bitops": 1.0,
        "gain_logicops": 1.0
    })

    # --- 6. SCHEME ANALYSIS ---
    for scheme in active_schemes:
        name = scheme['name']
        
        # Bit extraction (handling BitNet/Ternary cases)
        emb_conf = scheme.get('embedding_quant')
        w_conf = scheme.get('weight_quant')
        a_conf = scheme.get('activation_quant')
        
        b_emb = emb_conf.get('bits', 32) if emb_conf else 32
        if emb_conf and emb_conf.get('algo') in ["BitNet_b158", "Ternary"]: b_emb = 1.58

        b_w = w_conf.get('bits', 32) if w_conf else 32
        w_algo = w_conf.get('algo') if w_conf else "None"
        if w_algo in ["BitNet_b158", "Ternary"]: b_w = 1.58

        b_a = a_conf.get('bits', 32) if a_conf else 32
        # Default BitNet paper = 8, but respecting config (e.g., 6 for FPGA)
        if w_algo == "BitNet_b158" and a_conf and a_conf.get('bits') is None: b_a = 8

        # --- A. Memory Size (Physical Storage) ---
        # We count 2 bits for physical storage of 1.58b
        b_w_storage = 2 if b_w <= 2 else b_w
        b_emb_storage = 2 if b_emb <= 2 else b_emb
        
        total_bits_storage = (n_emb * b_emb_storage) + (n_body * b_w_storage) + (n_pooler * b_w_storage)
        size_mb = total_bits_storage / 8 / (1024 * 1024)
        gain_size = baseline_size_mb / size_mb if size_mb > 0 else 0
        
        # --- B. BitOps (Standard W*A) ---
        # Standard academic metric: Product of bit widths
        standard_w_cost = 2 if b_w <= 2 else b_w
        current_bitops = macs_per_inference * standard_w_cost * b_a
        gain_bitops = baseline_ops / current_bitops if current_bitops > 0 else 0
        
        # --- C. LogicOps (Real Hardware Cost / LUT Usage) ---
        # "Deeptech" metric: Accounts for multiplier-free architecture
        if w_algo in ["BitNet_b158", "Ternary"]:
            # BitNet: Weight is a selector (cost ~1 effective bit vs N bits)
            # The operation becomes a conditional accumulation
            effective_w_logic = 1.0 
        else:
            # LSQ: It is a multiplication, even if small. Cost is quadratic or linear depending on implementation.
            # We remain conservative by using the weight width.
            effective_w_logic = standard_w_cost

        current_logicops = macs_per_inference * effective_w_logic * b_a
        gain_logicops = baseline_ops / current_logicops if current_logicops > 0 else 0

        results.append({
            "name": name.replace('_', ' '),
            "b_emb": b_emb, "b_w": b_w, "b_a": b_a,
            "size_mb": size_mb,
            "bitops": current_bitops / 1e12,
            "logicops": current_logicops / 1e12,
            "gain_size": gain_size,
            "gain_bitops": gain_bitops,
            "gain_logicops": gain_logicops
        })

    # --- 7. CONSOLE DISPLAY ---
    header = f"{'CONF':<25} | {'E/W/A':<8} | {'SIZE':<8} {'(Gx)':<5} | {'BITOPS':<8} {'(Gx)':<5} | {'LOGIC':<8} {'(Gx)':<5}"
    print(header)
    print("-" * 100)
    for r in results:
        ewa = f"{r['b_emb']}/{r['b_w']}/{r['b_a']}".replace('.58', 'T').replace('.0', '')
        print(f"{r['name']:<25} | {ewa:<8} | {r['size_mb']:5.1f} {r['gain_size']:5.1f}x | {r['bitops']:5.2f}T {r['gain_bitops']:5.1f}x | {r['logicops']:5.2f}T {r['gain_logicops']:5.1f}x")
    print("-" * 100)

    # --- 8. LATEX GENERATION ---
    latex_content = generate_latex_table(results)
    
    output_dir = get_results_root()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_complexity_table.tex"
    
    with open(output_path, "w") as f:
        f.write(latex_content)
    
    print(f"\nLaTeX table generated: {output_path}")

def generate_latex_table(results):
    """Generates complete LaTeX code with 3 Gain columns."""
    rows = []
    for r in results:
        # Formatage : 1.58 -> 1.58 (T), 32.0 -> 32
        def fmt_bit(b): return "1.58" if b == 1.58 else f"{int(b)}"
        
        name = r['name'].replace("FP32 Baseline", "FP32 Baseline (Ref.)")
        
        # Bold for max gains in each category (LogicOps Gain for BitNet)
        gain_logic_str = f"{r['gain_logicops']:.1f}"
        if "BitNet" in name:
            gain_logic_str = f"\\textbf{{{gain_logic_str}}}"
            
        row = (
            f"{name} & "
            f"{fmt_bit(r['b_emb'])}/{fmt_bit(r['b_w'])}/{fmt_bit(r['b_a'])} & "
            f"{r['size_mb']:.1f} & {r['gain_size']:.1f}$\\times$ & "
            f"{r['bitops']:.2f} & {r['gain_bitops']:.1f}$\\times$ & "
            f"{r['logicops']:.2f} & {gain_logic_str}$\\times$ \\\\"
        )
        rows.append(row)

    content = r"""
\begin{table*}[h]
\centering
\caption{Theoretical estimation of memory footprint and computational complexity for the BERT Base architecture (SeqLen=512). \textbf{Size Gain} refers to the reduction in storage relative to FP32. \textbf{BitOps Gain} measures the reduction in standard arithmetic complexity ($W \times A$). \textbf{LogicOps Gain} reflects the actual hardware efficiency on FPGA, accounting for the elimination of multipliers in BitNet architectures (replaced by conditional additions).}
\label{tab:model_complexity}
\resizebox{\textwidth}{!}{
\begin{tabular}{lccccccc}
\toprule
\multirow{2}{*}{\textbf{Configuration}} & \textbf{Precision} & \multicolumn{2}{c}{\textbf{Memory}} & \multicolumn{2}{c}{\textbf{Arithmetic (BitOps)}} & \multicolumn{2}{c}{\textbf{Hardware (LogicOps)}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}
 & \textbf{E/W/A (bits)} & \textbf{Size (MB)} & \textbf{Gain} & \textbf{Ops (T)} & \textbf{Gain} & \textbf{Ops (T)} & \textbf{Gain} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\end{table*}
"""
    return content

if __name__ == "__main__":
    project_root = get_project_root()
    config_path = project_root / "configs" / "prod_config.yaml"
    calculate_complexity_metrics(config_path)