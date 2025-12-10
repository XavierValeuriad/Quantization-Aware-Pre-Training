# ===================================================================
# scripts/utils/clean_source_text.py
#
# v1.0 : Sanitization script for raw text files.
#        Detects and segments abnormally long lines that
#        cause bottlenecks in the pipeline.
# ===================================================================
import logging
from pathlib import Path
import sys

# --- Ensure 'src' is discoverable ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# ------------------------------------

from src.utils.path_manager import get_data_root

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sanitize_large_text_file(
    source_path: Path,
    dest_path: Path,
    max_chars_per_line: int = 2000, # Detection threshold for a "mega-line"
    split_char_chunk: int = 2000    # Size of new segments
):
    """
    Reads a source text file and writes a sanitized version,
    splitting lines that exceed `max_chars_per_line`.
    """
    if not source_path.exists():
        logging.error(f"Source file not found: {source_path}")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"--- Starting sanitization of {source_path} ---")
    logging.info(f"Destination: {dest_path}")
    logging.info(f"Mega-line detection threshold: {max_chars_per_line} characters")

    lines_processed = 0
    long_lines_found = 0

    with open(source_path, "r", encoding="utf-8") as f_in, \
         open(dest_path, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            lines_processed += 1
            if len(line) > max_chars_per_line:
                long_lines_found += 1
                logging.debug(f"Mega-line detected at line {lines_processed} (length: {len(line)}). Segmenting...")
                
                # Segment the mega-line into multiple sub-lines
                for i in range(0, len(line), split_char_chunk):
                    chunk = line[i:i + split_char_chunk]
                    f_out.write(chunk + '\n') # Write each segment as a new line
            else:
                f_out.write(line)
    
    logging.info("--- Sanitization completed ---")
    logging.info(f"Total lines read: {lines_processed}")
    logging.info(f"Mega-lines found and segmented: {long_lines_found}")

if __name__ == "__main__":
    dataset_id = "Dr-BERT@NACHOS"
    
    source_file = get_data_root() / "sources" / "pretraining" / dataset_id / "DOCUMENT_BY_DOCUMENT.txt"
    dest_file = get_data_root() / "sources" / "pretraining" / dataset_id / "DOCUMENT_BY_DOCUMENT.cleaned.txt"

    sanitize_large_text_file(source_file, dest_file)