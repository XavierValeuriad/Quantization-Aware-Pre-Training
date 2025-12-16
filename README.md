# Quantization-aware training: a tradeoff between training and fine-tuning for domain-specific language models (AccML 2026)

> **Official implementation for the AccML 2026 submission: "Quantization-aware training: a tradeoff between training and fine-tuning for domain-specific language models".**

This repository houses the complete experimental pipeline used to evaluate the impact of low-bit quantization (BitNet, LSQ) on language models pre-trained with adaptive budget allocation strategies. It is designed for high-performance computing (HPC) environments, specifically optimizing for NVIDIA H100 clusters using SLURM.

## 📑 Table of Contents

- [Scientific Context](#-scientific-context)
- [Project Architecture](#-project-architecture)
- [Installation & Environment](#-installation--environment)
- [The Pipeline](#-the-pipeline)
- [Configuration](#-configuration)
- [Reproducibility](#-reproducibility)
- [Citation](#-citation)
- [License](#-license)

## 🔬 Scientific Context

Current Large Language Models (LLMs) suffer from significant computational costs. This project explores the intersection of two optimization strategies:

1. Adaptive Pre-training: Dynamically adjusting the training budget ($\alpha$ ratio) based on loss convergence.
2. Aggressive Quantization: utilizing 1.58-bit (BitNet) and low-bit integer (LSQ) schemes during the fine-tuning and inference phases.

We provide a rigorous evaluation framework across 4 datasets (Biomedical) with a strict statistical validation protocol ($\sigma$ confidence intervals).

## 🏗 Project Architecture

The codebase follows a strictly modular "Orchestrator Pattern". Each stage of the pipeline is managed by a dedicated shell script that triggers a Python orchestrator within an isolated environment.

```
Quantization-Aware-Pre-Training/
├── configs/                 # Source of Truth (YAML)
│   ├── prod_config.yaml     # HPC Production parameters
│   └── dev_config.yaml      # Local Development overrides
├── scripts/                 # Entry points (SLURM & Bash)
│   ├── 01_create_environment.srun
│   ├── ...
│   └── 08_generate_reports.bash
├── src/                     # Source Code
│   ├── orchestrators/       # Python logic drivers
│   ├── models/              # Quantization layers (BitNet/LSQ)
│   └── training/            # Trainer loops & callbacks
├── poetry.lock              # Dependency Lockfile
└── pyproject.toml           # Project Definition
````

## ⚡ Installation & Environment

This project uses a hybrid Conda + Poetry approach to ensure bit-for-bit reproducibility on HPC clusters (Jean-Zay).

### 1. Prerequisites

- Python 3.10.4
- CUDA 12.x drivers (for H100 support)
- Access to a SLURM scheduler (optional for local dev)

### 2. BootstrapTo set up the environment, run the appropriate script for your context.

#### On HPC (SLURM/Jean-Zay):

```Bash
batch scripts/01_create_environement.srun
```

#### On Local Machine (Debian/Ubuntu):

```Bash
bash scripts/01_create_environment.bash
```

This will create a `conda_env` for system dependencies and a poetry `.venv` for Python libraries, isolated in your workspace.

## 🚀 The Pipeline

The experiment workflow is sequential. Execute the scripts in the numerical order provided in the `scripts/` directory.

| Order | Script | Type | Description | 
| :--- | :--- | :--- | :--- |
| **01** | `01_create_environment` | Setup | Builds the sovereign Conda+Poetry environment. |
**02** | `02_download_models.bash` | Assets | Fetches base weights (DrBERT, CamemBERT, etc.) from HF Hub.|
| **03** | `03_download_data.bash` | Assets | Downloads and shards raw datasets (Quaero, FLUE, etc.). |
| **04** | `04_run_pretokenization` | Process | SLURM/Bash. Converts raw text to tokenized arrow datasets. Offline mode. |
| **05** | `05_launch_pretraining` | Train | SLURM Array. Launches the Pre-training Grid (1 GPU/Task). |
| **06** | `06_launch_finetuning_and_evaluation_array.sbtach` | Eval | SLURM Array. Massive parallel evaluation (Alpha Grouped Pipeline). | 
| **07** | `07_download_results.bash` | Sync | Aggregates metrics from distributed nodes/remote storage. | 
| **08** | `08_generate_reports.bash` | Analysis | Generates LaTeX tables and visualization plots. | 

### 🧬 Special Case: NACHOS Dataset Ingestion

The **NACHOS** dataset serves as our primary pre-training corpus. Due to its distribution format (single monolithic text file >10GB) and specific artifacts (extremely long lines causing tokenizer OOM), we implement a dedicated two-stage ingestion pipeline **before** tokenization.


**Workflow:**

1.  **Sanitization (`scripts/utils/clean_source_text.py`):**
    Scans the raw `DOCUMENT_BY_DOCUMENT.txt`. It detects lines exceeding `2000` characters (often OCR artifacts or non-textual dumps) and safely segments them. This step is crucial to prevent buffer overflows in the tokenizer's Rust backend.

2.  **Structuring (`scripts/utils/make_dataset.py`):**
    Converts the sanitized text into a sharded Hugging Face `DatasetDict` (Arrow format) and generates the required `dataset_configs.json` manifest. This aligns the raw text with our standardized multi-config loader.

### Example: Launching the Evaluation Array

To launch the massive evaluation grid (Model × Quantization × Dataset):

```Bash
batch scripts/06_launch_finetuning_and_evaluation_array.sbatch
```

## ⚙ Configuration

All experimental parameters are centralized in `config.yaml`. Do not hardcode parameters in Python scripts.

```YAML
experiment:
  total_steps_budget: 30000
  alpha_sweep: [0.0, 0.1, 0.5, 0.9] # Adaptive ratios

quantization:
 - name: "BitNet_E2W1.58A2"
   enabled: true
   embedding_quant: { algo: "LSQ", bits: 2 }
   weight_quant: { algo: "BitNet_b158" }
   activation_quant: { algo: "BitNet_b158", bits: 2 }
```

To override for local testing, modify `configs/dev_config.yaml` and set `dev: true`.

## ♻️ Reproducibility

We prioritize "Scientific Discovery" standards:

1. Seeds: Fixed random seed (`42`) for data splitting and initialization.
2. Deterministic Algorithms: cuDNN deterministic mode enabled.
3. Offline Mode: All network calls are disabled during training/eval (`HF_HUB_OFFLINE=1`).

## 📚 Citation

If you use this code or our results, please cite the AccML 2026 paper:

```
In coming
```

## ⚖ License

This project is licensed under the **MIT License**. 