# ===================================================================
# src/reporting/table_generator.py
#
# v1.0 : Table A Overhaul (Combined & Multi-Model)
#        - Table A combines all models.
#        - Columns split by model (Task -> Model -> Metric).
#        - Bolding on best score (per Quant x Task, cross-alpha/model).
#        - Bolding on removed alpha.
# ===================================================================

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

from src.reporting.data_collector import CollectedData
from src.tasks.factory import get_task_specification
from src.utils.path_manager import _sanitize_name

logger = logging.getLogger(__name__)

# Canonical row order
QUANTIZATION_ORDER = [
    'FP32_Baseline',
    'LSQ_E8W8A8',
    'LSQ_E4W4A4',
    'LSQ_E2W2A2',
    'LSQ_E6W2A6',
    'BitNet_E6W1.58A6'
]

MODEL_SHORT_NAMES = {
    "camembert-base": "Camem.",
    "Dr-BERT/DrBERT-7GB": "DrBERT"
}

def create_all_tables(all_data: List[CollectedData], model_ids: List[str], output_dir: Path):
    """Main entry point to generate all tables."""
    logger.info(f"Starting LaTeX table generation in: {output_dir}")

    # 1. Prepare the main DataFrame
    records = []
    for data in all_data:
        if not data.eval_json:
            continue
            
        task_spec = get_task_specification(data.dataset_info)
        
        # --- Extended Metric Extraction Logic ---
        metric_name = "N/A"
        arrow = ""
        main_metric_value = None
        greater_is_better = True
        
        try:
            metric_dict = task_spec.get_optimization_metric()
            metric_name_raw = metric_dict['name']
            greater_is_better = metric_dict['greater_is_better']
             
            if metric_name_raw == 'overall_f1': metric_name = 'F1'
            elif metric_name_raw == 'exact_match': metric_name = 'EM'
            else: metric_name = metric_name_raw.replace('_', ' ').capitalize()
                 
            arrow = r' $\uparrow$' if greater_is_better else r' $\downarrow$'
            metric_header = f"{metric_name}{arrow}"
            main_metric_value = data.get_main_metric(task_spec)

        except Exception as e:
            logger.warning(f"Unable to extract main metric for {data.dataset_id}: {e}")
            metric_header = "Score"
             
        # Logic for task short name
        base_name = data.dataset_id.split('/')[-1]
        config_name = data.dataset_config
        task_short_name = base_name if config_name in ["default", "pos"] else f"{base_name}/{config_name}"
        
        if main_metric_value is not None:
            records.append({
                "model_id": data.model_id, "quant_name": data.quant_name,
                "alpha": data.alpha, "metric_score": main_metric_value,
                "task": task_short_name, "metric": metric_header,
                "greater_is_better": greater_is_better
            })
            
        if data.dataset_id == 'Dr-BERT/FrenchMedMCQA':
            hamming_loss = data.eval_json.get('eval_hamming_loss', data.eval_json.get('hamming_loss'))
            if hamming_loss is not None:
                hamming_header = r"Hamming $\downarrow$"
                records.append({
                    "model_id": data.model_id, "quant_name": data.quant_name,
                    "alpha": data.alpha, "metric_score": hamming_loss,
                    "task": task_short_name, "metric": hamming_header,
                    "greater_is_better": False
                })
            else: logger.warning(f"FrenchMedMCQA: 'eval_hamming_loss' not found in {data.ft_path.name}")

    if not records:
        logger.error("No evaluation data (JSON) found. Aborting table generation.")
        return

    df = pd.DataFrame(records).dropna(subset=['metric_score'])
    
    # 2. Generate combined Table A
    _generate_combined_detailed_table(df, model_ids, output_dir)

    # 3. Generate Tables B (one per model)
    for model_id in model_ids:
        model_df = df[df['model_id'] == model_id]
        if model_df.empty:
            logger.warning(f"No data for Table B of model: {model_id}")
            continue
        _generate_best_alpha_table_for_model(model_df, model_id, output_dir)


def _generate_combined_detailed_table(df: pd.DataFrame, model_ids: List[str], output_dir: Path):
    """Generates detailed Table A combining all models."""
    logger.info("Generating combined detailed table (Table A)...")
    
    try:
        # --- 1. Pivot Data (with model_id in columns) ---
        table_A = df.pivot_table(
            index=['quant_name', 'alpha'],
            columns=['task', 'model_id', 'metric'], 
            values='metric_score'
        )
        
        # --- 2. Store direction for bolding ---
        direction_map = df.set_index(['task', 'model_id', 'metric'])['greater_is_better'].to_dict()

        # --- 3. Sort (Rows and Columns) ---
        current_quant_index = table_A.index.get_level_values(0).unique()
        ordered_quant_index = [q for q in QUANTIZATION_ORDER if q in current_quant_index]
        table_A = table_A.reindex(index=ordered_quant_index, level=0)
        table_A = table_A.sort_index(axis=1) # Sorts by task, then model, then metric
        
        # --- 4. Format VALUES (to text) ---
        table_A_formatted = table_A.map(lambda x: f"{x * 100:.1f}" if pd.notna(x) else "-")
        
        # --- 5. Apply BOLD formatting (Best Score per Quant x Task) ---
        quant_names = table_A.index.get_level_values(0).unique()
        tasks = table_A.columns.get_level_values(0).unique()
        
        for quant in quant_names:
            for task in tasks:
                # Select columns for this quantizer and this task (all models, all metrics)
                sub_df_numeric = table_A.loc[quant, task]
                 
                # If empty, skip
                if sub_df_numeric.empty or sub_df_numeric.isnull().all().all():
                    continue

                # Find metric(s) associated with this task
                metrics_for_task = sub_df_numeric.columns.get_level_values(1).unique()

                for metric_header in metrics_for_task:
                    # Specific column (e.g., ('DrBERT', 'F1 $\uparrow$'))
                    metric_col_selector = (slice(None), metric_header) # slice(None) for all models
                     
                    # Retrieve direction
                    # Note: We take the direction of the first model found for this metric
                    first_model_for_metric = sub_df_numeric.columns.get_level_values(0)[sub_df_numeric.columns.get_level_values(1) == metric_header][0]
                    is_greater_better = direction_map.get((task, first_model_for_metric, metric_header), True)

                    # Scores for this metric and task (all alphas, all models)
                    scores_for_metric = sub_df_numeric.loc[:, metric_col_selector]

                    # Find best score
                    if is_greater_better:
                        best_score = scores_for_metric.max().max() # Max over alpha then over model
                    else:
                        best_score = scores_for_metric.min().min() # Min over alpha then over model

                    # Apply bold where score equals best score
                    # (there may be ties)
                    for alpha in scores_for_metric.index:
                        for model_id_col in scores_for_metric.columns.get_level_values(0):
                            current_score = scores_for_metric.loc[alpha, (model_id_col, metric_header)]
                            if pd.notna(current_score) and np.isclose(current_score, best_score):
                                coord_row = (quant, alpha)
                                coord_col = (task, model_id_col, metric_header)
                                val = table_A_formatted.loc[coord_row, coord_col]
                                if val != "-":
                                     table_A_formatted.loc[coord_row, coord_col] = f"\\textbf{{{val}}}"


        # --- 6. Format Index and Columns ---
        idx_l0_raw = table_A_formatted.index.get_level_values(0)
        idx_l1_raw = table_A_formatted.index.get_level_values(1)
        
        idx_l0_formatted = [qn.replace('_', ' ') for qn in idx_l0_raw]
        idx_l1_formatted = [f"{a * 100:.0f}" for a in idx_l1_raw] # Alpha without bold
        
        table_A_formatted.index = pd.MultiIndex.from_arrays(
            [idx_l0_formatted, idx_l1_formatted], 
            names=[r'quant name', r'$\alpha$ (\%)']
        )
        
        # Columns: Task (space) -> Model (short) -> Metric (latex)
        cols_l0 = table_A_formatted.columns.get_level_values(0).astype(str).str.replace('_', ' ') # Task
        cols_l1_raw = table_A_formatted.columns.get_level_values(1) # Model ID
        cols_l1 = [MODEL_SHORT_NAMES.get(mid, mid) for mid in cols_l1_raw] # Model Short Name
        cols_l2 = table_A_formatted.columns.get_level_values(2) # Metric (contient LaTeX)
        
        table_A_formatted.columns = pd.MultiIndex.from_arrays(
            [cols_l0, cols_l1, cols_l2], 
            names=['dataset', 'model', None] # Hides 'metric' name
        )

        # --- 7. Generate LaTeX ---
        n_cols = len(table_A_formatted.columns)
        n_idx_cols = table_A_formatted.index.nlevels
        column_format = 'c' * (n_idx_cols + n_cols)
        
        latex_A = table_A_formatted.to_latex(
            caption="Résultats détaillés combinés (par alpha).", # Generic caption
            label="tab:results_combined_detailed",
            na_rep="-",
            multicolumn_format='c',
            multirow=True,
            escape=False,
            column_format=column_format 
        )
        
        latex_A = latex_A.replace(r'\multirow[t]', r'\multirow')
        
        # Generic filename
        (output_dir / "table_combined_detailed_by_alpha.tex").write_text(latex_A)
        logger.info("Table 'combined_detailed_by_alpha' sauvegardée.")

    except Exception as e:
        logger.error(f"Échec de la génération de la table A combinée: {e}", exc_info=True)


def _generate_best_alpha_table_for_model(model_df: pd.DataFrame, model_id: str, output_dir: Path):
    """Generates Table B (best alpha) for a SINGLE model."""
    # This function remains quasi-identical, but uses _sanitize_name
    
    logger.info(f"Generating Best Alpha table for {model_id}...")
    
    safe_model_id_caption = model_id.replace('_', ' ')
    safe_model_id_filename = _sanitize_name(model_id)

    try:
        # Find best alpha (based on normalized mean)
        direction_map_b = model_df.set_index(['task', 'metric'])['greater_is_better'].to_dict()
        table_B_scores = model_df.pivot_table(index='alpha', columns=['task', 'metric'], values='metric_score')
        
        table_B_norm = table_B_scores.copy()
        for col in table_B_scores.columns:
            # Use get with a tuple key (task, metric)
            task_met_tuple = (col[0], col[1]) # Creates key tuple
            if not direction_map_b.get(task_met_tuple, True): 
                table_B_norm[col] = 1.0 - table_B_norm[col]
                 
        mean_scores_b = table_B_norm.mean(axis=1)
        best_alpha = mean_scores_b.idxmax()

        # Filter original DF for this best alpha
        best_df = model_df[model_df['alpha'] == best_alpha]

        # Pivot scores
        score_pivot_b = best_df.pivot_table(
            index='quant_name', columns=['task', 'metric'], values='metric_score'
        )
        
        # Sort and format
        current_quant_index_b = score_pivot_b.index.unique()
        ordered_quant_index_b = [q for q in QUANTIZATION_ORDER if q in current_quant_index_b]
        score_pivot_b = score_pivot_b.reindex(index=ordered_quant_index_b)
        score_pivot_b = score_pivot_b.sort_index(axis=1)

        table_B_formatted = score_pivot_b.map(lambda x: f"{x * 100:.1f}" if pd.notna(x) else "-")
        
        # Add alpha info
        table_B_formatted[r'Best $\alpha$ (\%)'] = f"{best_alpha * 100:.0f}"

        # Escape and rename
        table_B_formatted.index = table_B_formatted.index.astype(str).str.replace('_', ' ')
        table_B_formatted.index.name = 'quant name'
        
        cols_l0 = table_B_formatted.columns.get_level_values(0).astype(str).str.replace('_', ' ')
        cols_l1 = table_B_formatted.columns.get_level_values(1)
        table_B_formatted.columns = pd.MultiIndex.from_arrays([cols_l0, cols_l1], names=['dataset', None])

        # Generate LaTeX
        n_cols_b = len(table_B_formatted.columns)
        n_idx_cols_b = table_B_formatted.index.nlevels
        column_format_b = 'c' * (n_idx_cols_b + n_cols_b)
        
        latex_B_escaped = table_B_formatted.to_latex(
            caption=f"Meilleurs résultats ($\\alpha$ = {best_alpha*100:.0f}\\%) pour {safe_model_id_caption}.",
            label=f"tab:results_{safe_model_id_filename}_best_alpha",
            na_rep="-",
            multicolumn_format='c',
            escape=False,
            column_format=column_format_b 
        )
        
        (output_dir / f"table_{safe_model_id_filename}_best_alpha.tex").write_text(latex_B_escaped)
        logger.info(f"Table 'best_alpha' pour {model_id} sauvegardée.")

    except Exception as e:
        logger.error(f"Échec de la génération de la table B pour {model_id}: {e}", exc_info=True)