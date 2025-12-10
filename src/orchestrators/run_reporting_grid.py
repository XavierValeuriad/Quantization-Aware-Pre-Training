"""
===================================================================
✧ KAIROS_OS :: src/orchestrators/run_reporting_grid.py
v1.0 : path_manager import fix (Fix ImportError)
===================================================================
"""

import logging
from src.config.core import config

from src.utils.path_manager import (
    get_data_root,
    get_results_root,
    get_custom_pretrained_model_path,
    get_custom_finetuned_model_path,
    create_pretraining_manifest,
    create_finetuning_manifest
)
from src.workers.reporter import plot_training_curves, generate_carbon_statement

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_reporting_grid():
    """Main entry point to generate all report artifacts."""
    logging.info("--- Starting Final Report Generator (v1.3 Fix) ---")

    log_files_to_plot = []
    emission_files_to_process = []

    # --- 1. Structured Traversal of Pre-training Experiments ---
    logging.info("Searching for pre-training artifacts...")
    for model_id in config.models:
        for quant_scheme in config.quantization:
            manifest = create_pretraining_manifest(model_id, quant_scheme)
            
            # Call without model_id (already in manifest) ✧
            output_dir = get_custom_pretrained_model_path(manifest)

            logging.debug(f'Checking dir: {output_dir}')
            if output_dir.exists():
                if (output_dir / 'training_log_history.csv').exists():
                    log_files_to_plot.append(output_dir / 'training_log_history.csv')
                if (output_dir / 'emissions.csv').exists():
                    emission_files_to_process.append(output_dir / 'emissions.csv')

    # --- 2. Structured Traversal of Fine-tuning Experiments ---
    logging.info("Searching for fine-tuning artifacts...")
    for model_id in config.models:
        for quant_scheme in config.quantization:
            
            # We need the parent path (pre-training) to check for alpha checkpoint existence
            pretrain_manifest = create_pretraining_manifest(model_id, quant_scheme)
            parent_run_dir = get_custom_pretrained_model_path(pretrain_manifest)
            
            if not parent_run_dir.exists(): 
                continue

            for alpha in config.experiment.alpha_sweep:
                # Optional check for source alpha checkpoint existence
                # checkpoint_name = f"{quant_scheme.name}_alpha_{alpha:.2f}_checkpoint"
                # pre_trained_model_path = parent_run_dir / checkpoint_name
                # if not pre_trained_model_path.exists(): continue

                for dataset_info in config.finetuning_datasets:
                    # Check tokenizer compatibility
                    if model_id not in getattr(dataset_info, 'tokenizers', []): 
                        continue

                    # Dataset config handling (list or single)
                    configs = getattr(dataset_info, 'configs', ['default'])
                    
                    for ds_config_name in configs:
                        ft_manifest = create_finetuning_manifest(
                            model_id, 
                            quant_scheme, 
                            dataset_info, 
                            ds_config_name, 
                            alpha
                        )
                        
                        output_dir = get_custom_finetuned_model_path(ft_manifest)

                        logging.debug(f'Checking FT dir: {output_dir}')
                        if output_dir.exists():
                            if (output_dir / 'training_log_history.csv').exists():
                                log_files_to_plot.append(output_dir / 'training_log_history.csv')
                            if (output_dir / 'emissions.csv').exists():
                                emission_files_to_process.append(output_dir / 'emissions.csv')

    # --- 3. Artifact Generation ---
    logging.info(f"Found {len(log_files_to_plot)} history files to process.")

    results_dir = get_results_root() / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)

    for log_file in log_files_to_plot:
        # Creating a unique filename based on relative path
        relative_path_slug = str(log_file.relative_to(get_data_root())).replace('/', '_').replace('.csv', '.png')
        output_path = results_dir / relative_path_slug
        
        plot_training_curves(log_file, output_path)

    if emission_files_to_process:
        logging.info(f"Generating carbon statement for {len(emission_files_to_process)} files...")
        generate_carbon_statement(emission_files_to_process)
    else:
        logging.warning("No emission files found.")

    logging.info("--- Report generation completed ---")

if __name__ == "__main__":
    run_reporting_grid()