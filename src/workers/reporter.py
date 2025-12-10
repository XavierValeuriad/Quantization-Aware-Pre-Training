# ===================================================================
# src/workers/reporter.py
#
# v1.0 : Added plot_combined_alpha_curves.
# ===================================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

from src.utils.path_manager import get_data_root, get_results_root
from src.utils.system import is_on_jean_zay
from src.tasks.factory import get_task_specification 
from src.config.core import ConfigObject



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# --- Existing function (Corrected) ---
def generate_carbon_statement(emission_files: List[Path]):
    """
    Generates a carbon footprint statement by aggregating data from multiple emissions.csv files
    and produces a JSON file.
    """
    if not emission_files:
        logger.warning("No emission files provided for carbon statement.")
        return

    total_duration_seconds = 0.0
    total_cpu_energy_kwh = 0.0
    total_gpu_energy_kwh = 0.0
    total_ram_energy_kwh = 0.0
    total_emissions_kgco2eq = 0.0
    total_energy_kwh = 0.0
    run_details = [] 

    logger.info(f"Processing {len(emission_files)} emission file(s)...")

    for emission_file in emission_files:
        try:
            if not emission_file.exists():
                logger.warning(f"Emission file not found (skipped): {emission_file}")
                continue

            df = pd.read_csv(emission_file)
            if df.empty:
                logger.warning(f"Empty emission file (skipped): {emission_file}")
                continue

            # Assuming standard codecarbon format with a single relevant line
            row = df.iloc[0] 

            total_duration_seconds += row.get('duration', 0.0)
            total_cpu_energy_kwh += row.get('cpu_energy', 0.0)
            total_gpu_energy_kwh += row.get('gpu_energy', 0.0)
            total_ram_energy_kwh += row.get('ram_energy', 0.0)
            total_emissions_kgco2eq += row.get('emissions', 0.0)
            total_energy_kwh += row.get('energy_consumed', 0.0)

            # Collecting details per run
            run_details.append({
                'run_id': str(row.get('run_id', 'N/A')),
                'duration_seconds': float(row.get('duration', 0.0)),
                'emissions_kgco2eq': float(row.get('emissions', 0.0)),
                'energy_consumed_kwh': float(row.get('energy_consumed', 0.0))
            })

        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier {emission_file}: {e}", exc_info=True)

    # Constructing data structure for JSON
    carbon_data = {
        "summary": {
            "total_duration_seconds": total_duration_seconds,
            "total_duration_hours": total_duration_seconds / 3600.0,
            "total_energy_kwh": total_energy_kwh,
            "energy_breakdown_kwh": {
                "cpu": total_cpu_energy_kwh,
                "gpu": total_gpu_energy_kwh,
                "ram": total_ram_energy_kwh
            },
            "total_emissions_kgco2eq": total_emissions_kgco2eq,
            "total_runs_analyzed": len(run_details)
        },
        "details": run_details,
        "methodology": {
            "source": "codecarbon",
            "note": "Calculs basés sur l'intensité carbone du réseau électrique régional."
        }
    }

    # Output path for JSON file
    output_dir = get_results_root()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_file_path = output_dir / "carbon_statement.json"

    # Writing JSON file
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(carbon_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Carbon statement JSON generated successfully: {json_file_path}")
    except IOError as e:
        logger.critical(f"Unable to write carbon statement JSON file: {e}", exc_info=True)

def plot_combined_alpha_curves(
    pretrain_log_path: Path,
    finetune_log_paths: Dict[float, Path], # Dict: alpha -> Path
    output_plot_path: Path,
    eval_results_by_alpha: Dict[float, Optional[Dict[str, Any]]], # Dict: alpha -> results
    title: str, # Base title (M | Q | D)
    dataset_info: ConfigObject # To find optimization metric
):
    """
    Generates a combined PT + FT plot (all alphas) for an M x Q x D task.
    """
    logging.info(f"Generating combined alpha plot: {output_plot_path}")
    
    df_pretrain = None
    pretrain_max_step = 0
    
    # --- 1. Loading Pre-training Log (alpha=0) ---
    try:
        if pretrain_log_path.exists() and pretrain_log_path.stat().st_size > 0:
            df_pretrain = pd.read_csv(pretrain_log_path).dropna(subset=['step'])
            if not df_pretrain.empty:
                pretrain_max_step = df_pretrain['step'].max()
                # logging.info(f"PT Log (alpha=0) loaded ({len(df_pretrain)} points, max step {pretrain_max_step}).")
        else:
            logging.warning(f"PT Log not found/empty: {pretrain_log_path}")
    except Exception as e:
        logging.error(f"Error loading PT log {pretrain_log_path}: {e}")

    # --- 2. Loading & Combining Fine-tuning Logs (all alphas) ---
    all_ft_dfs = []
    valid_alphas = sorted(finetune_log_paths.keys()) # Iterate in order
    
    if not valid_alphas:
        logging.error("No fine-tuning logs found. Unable to generate plot.")
        return

    for alpha in valid_alphas:
        log_path = finetune_log_paths[alpha]
        try:
            if log_path.exists() and log_path.stat().st_size > 0:
                df_ft = pd.read_csv(log_path).dropna(subset=['step'])
                if not df_ft.empty:
                    df_ft['step'] += pretrain_max_step # Shift steps
                    df_ft['alpha'] = alpha # Adds alpha column
                    all_ft_dfs.append(df_ft)
            else:
                logging.warning(f"FT Log alpha={alpha:.2f} not found/empty: {log_path}")
        except Exception as e:
            logging.error(f"Error loading FT log {log_path}: {e}")

    if not all_ft_dfs:
         logging.error("No valid fine-tuning logs loaded. Unable to generate plot.")
         return
         
    df_finetune_combined = pd.concat(all_ft_dfs, ignore_index=True)

    # --- 3. Determine Optimization Metric to Plot ---
    metric_to_plot = None
    metric_col_name_in_df = None # Exact name in DataFrame (may have _mean prefix if bootstrap)
    greater_is_better = True # Défaut
    
    try:
        task_spec = get_task_specification(dataset_info)
        opt_metric_info = task_spec.get_optimization_metric()
        metric_name = opt_metric_info.get('name')
        greater_is_better = opt_metric_info.get('greater_is_better', True)
        
        # Search for corresponding column in FT logs (with or without eval_ prefix)
        possible_names = [f"eval_{metric_name}", metric_name]
        # If bootstrap, also look for _mean
        if config.evaluation.use_bootstrap:
             possible_names.insert(0, f"eval_{metric_name}_mean")
             possible_names.insert(1, f"{metric_name}_mean") # Less likely

        for name in possible_names:
             if name in df_finetune_combined.columns and not df_finetune_combined[name].isna().all():
                  metric_to_plot = metric_name # Keeps base name
                  metric_col_name_in_df = name # Uses the one found in df
                  logging.info(f"Metric to plot identified: '{metric_name}' via column '{metric_col_name_in_df}'")
                  break
        if not metric_to_plot:
             logging.warning(f"Unable to find column for metric '{metric_name}' in FT logs.")
             raise Exception(f"Metric not found for '{metric_name}' in FT logs.")

    except Exception as e:
         logging.warning(f"Unable to determine optimization metric via TaskSpec: {e}")

    # --- 4. Plot Creation ---
    fig, ax1 = plt.subplots(figsize=(18, 10))
    lines = []
    labels = []
    
    # Color palette for alphas
    alpha_colors = cm.get_cmap('cool', len(valid_alphas) + 1) # +1 to avoid colors too light/dark

    # --- Axis 1 (Left): Pre-training Loss ---
    color_pretrain = 'black'
    ax1.set_xlabel('Total Steps (Pre-training + Fine-tuning)')
    ax1.set_ylabel('Pre-training Loss', color=color_pretrain)
    ax1.tick_params(axis='y', labelcolor=color_pretrain)
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5, axis='y') 

    if df_pretrain is not None:
        pt_train = df_pretrain[df_pretrain['loss'].notna()]
        pt_eval = df_pretrain[df_pretrain['eval_loss'].notna()]
        if not pt_train.empty:
             line, = ax1.plot(pt_train['step'], pt_train['loss'], color=color_pretrain, alpha=0.6, linewidth=1.5, label='PT Loss (Train)')
             lines.append(line); labels.append('PT Loss (Train)')
        if not pt_eval.empty:
             line, = ax1.plot(pt_eval['step'], pt_eval['eval_loss'], color=color_pretrain, linestyle='--', alpha=0.7, linewidth=1.5, label='PT Loss (Eval)')
             lines.append(line); labels.append('PT Loss (Eval)')

    # --- Axis 2 (Right): Loss & Fine-tuning Metric (per alpha) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Fine-tuning Loss / Metric') 
    
    # Plot FT Loss & Metric for each alpha
    for i, alpha in enumerate(valid_alphas):
        df_alpha = df_finetune_combined[df_finetune_combined['alpha'] == alpha]
        color = alpha_colors(i / len(valid_alphas)) # Color based on alpha index
        
        # FT Loss (Train)
        ft_train = df_alpha[df_alpha['loss'].notna()]
        if not ft_train.empty:
             line, = ax2.plot(ft_train['step'], ft_train['loss'], color=color, alpha=0.5, linestyle='-', linewidth=1, label=f'FT Loss (α={alpha:.1f})')
             if alpha == valid_alphas[0] or alpha == valid_alphas[-1]: # Legend only for 1st/last alpha?
                 lines.append(line); labels.append(f'FT Loss (α={alpha:.1f})')

        # FT Loss (Eval)
        ft_eval = df_alpha[df_alpha['eval_loss'].notna()]
        if not ft_eval.empty:
             line, = ax2.plot(ft_eval['step'], ft_eval['eval_loss'], color=color, alpha=0.7, linestyle='--', linewidth=1.5, label=f'FT Eval Loss (α={alpha:.1f})')
             if alpha == valid_alphas[0] or alpha == valid_alphas[-1]:
                 lines.append(line); labels.append(f'FT Eval Loss (α={alpha:.1f})')

        # FT Metric (Eval)
        if metric_to_plot and metric_col_name_in_df in df_alpha.columns:
             ft_metric = df_alpha[df_alpha[metric_col_name_in_df].notna()]
             if not ft_metric.empty:
                  line, = ax2.plot(ft_metric['step'], ft_metric[metric_col_name_in_df], color=color, marker='.', markersize=4, linestyle=':', alpha=0.9, label=f'FT Metric ({metric_to_plot}, α={alpha:.1f})')
                  if alpha == valid_alphas[0] or alpha == valid_alphas[-1]:
                       lines.append(line); labels.append(f'FT Metric ({metric_to_plot}, α={alpha:.1f})')

    # --- Plot Finalization ---
    # Combined legend
    # Place legend outside to avoid hiding curves
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)

    # Vertical separation line PT/FT
    if pretrain_max_step > 0:
         ax1.axvline(x=pretrain_max_step, color='dimgrey', linestyle='-', linewidth=1.5)
        # Add text annotation?
         # ax1.text(pretrain_max_step * 1.01, ax1.get_ylim()[1] * 0.95, 'Start FT', rotation=90, verticalalignment='top', color='dimgrey')
    
    ax2.tick_params(axis='y') # Axis 2 labels color = black by default
    
    # Adjust Y limits if needed for better readability
    # ax1.set_ylim(bottom=0) # Loss >= 0
    # ax2.set_ylim(bottom=0) # Loss/Metric >= 0?

    fig.suptitle(f"Combined Training: {title}", fontsize=16)
    # Larger adjustment for bottom legend
    plt.tight_layout(rect=[0, 0.1, 1, 0.95]) 
    
    try:
        output_plot_path.parent.mkdir(parents=True, exist_ok=True) 
        plt.savefig(output_plot_path, bbox_inches='tight') # bbox_inches to include external legend
        plt.close(fig)
        logging.info(f"Combined alpha plot saved: {output_plot_path}")
    except Exception as e:
        logging.error(f"Error saving plot {output_plot_path}: {e}")
        plt.close(fig) 
