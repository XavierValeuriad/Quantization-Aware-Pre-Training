import logging
import pandas as pd
from pathlib import Path
from transformers import Trainer, TrainerCallback
from transformers.integrations import TensorBoardCallback
from codecarbon import EmissionsTracker
from src.modeling.quantization import LsqQuantizer, BitNetQuantizer, QuantizeLinear, QuantizeEmbedding
from src.config.core import config
from src.utils.system import is_on_jean_zay

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def start_carbon_tracker(output_dir: Path) -> EmissionsTracker:
    """
    Initializes and starts the carbon emissions tracker.
    """
    logging.info("✧!co2 :: Initialisation du tracker d'émissions carbone.")
    try:
        if is_on_jean_zay():
            logging.info("Jean-Zay environment detected for CodeCarbon configuration.")            
            EF_KG_PER_KWH = None
            tracker = EmissionsTracker(
                project_name="accml2026",
                measure_power_secs=15,
                cpu_power="off",               # or track_cpu_power=False depending on version
                gpu_ids=None,                  # None = all visible GPUs (your 4xH100)
                offline=True,                  # no network access on compute nodes
                # Datacenter
                pue=1.18,                      # Jean-Zay (warm water cooling) ~1.18 ; total ~1.21]
                country_iso_code=config.carbon_tracker.country_iso_code,
                emission_factor=EF_KG_PER_KWH  # if provided, overrides country_iso_code
            )
        else:
            logging.info("Local environment detected for CodeCarbon configuration.")
            tracker = EmissionsTracker(
                output_dir=str(output_dir),
                log_level="error", # To avoid polluting main logs
                save_to_file=True,
                save_to_api=False,
                tracking_mode="process",
            )
        tracker.start()
        return tracker
    except Exception as e:
        logging.warning(f"Unable to initialize carbon tracker. Error: {e}")
    return None

def finalize_training(trainer: Trainer, tracker: EmissionsTracker = None):
    """
    Finalizes training by saving log history and stopping the carbon tracker.
    """
    output_dir = Path(trainer.args.output_dir)

    # 1. Sauvegarde de l'historique des logs
    if trainer.state.log_history:
        try:
            log_history_path = output_dir / "training_log_history.csv"
            logging.info(f"✍️  Saving log history to: {log_history_path}")
            pd.DataFrame(trainer.state.log_history).to_csv(log_history_path, index=False)
        except Exception as e:
            logging.error(f"Failed to save log history. Error: {e}")

    # 2. Arrêt du tracker carbone
    if tracker:
        try:
            logging.info("✧!co2 :: Stopping carbon tracker and saving report.")
            emissions_data = tracker.stop()
            if emissions_data:
                logging.info(f"Estimated carbon emissions for this run: {emissions_data:.4f} kgCO2eq")
        except Exception as e:
            logging.error(f"Failed to stop carbon tracker. Error: {e}")