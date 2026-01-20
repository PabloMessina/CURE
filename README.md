# CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation

**Official PyTorch Implementation**

This repository contains the code and resources for **CURE**, a framework for training Biomedical Vision-Language Models (Biomed-VLMs) developed at the IVUL group, KAUST.

CURE enhances medical VLMs by reformulating training into a **fine-grained instructional format** and introducing an **Error-Aware Curriculum**. This curriculum dynamically adjusts sampling probabilities based on the model's performance across datasets (Inter-Dataset) and anatomical regions (Intra-Dataset), allowing the model to focus on challenging samples and underperforming regions.

## Supported Tasks & Prompts

CURE unifies diverse tasks into a single instructional framework. The model is trained to handle **Phrase Grounding (PG)**, **Grounded Report Generation (GRG)**, and **Anatomy-Grounded Report Generation (AGRG)**.

Standard Report Generation (RG) on MIMIC-CXR is achieved by leveraging these grounded capabilities in three distinct modes:

| Dataset | Task | Prompt | Example Output |
| :--- | :--- | :--- | :--- |
| **MS-CXR** | PG | `Ground the phrase: {phrase}` | `{phrase}: [x, y, w, h] ...` |
| **PadChest-GR** | PG | `Ground the phrase: {phrase}` | `{phrase}: [x, y, w, h] ...` |
| **PadChest-GR** | GRG | `Generate a grounded report.` | `Slight residual atelectasis [x,y,w,h]. ...` |
| **Chest ImaGenome** | AGRG | `Locate and describe the {location}.` | `Location of {loc}: [x,y,w,h]. Description: ...` |
| **Chest ImaGenome** | AGRG | `Locate the {location}.` | `Location of {loc}: [x,y,w,h].` |
| **Chest ImaGenome** | AGRG | `Describe the {location}.` | `Description of {loc}: ...` |
| **VinDr-CXR** (Zero-Shot) | PG | `Ground the phrase: {phrase}` | `Cardiomegaly: [x,y,w,h]` |
| **MIMIC-CXR** (Eval) | RG (via GRG) | `Generate a grounded report.` | *(Grounded report output)* |
| **MIMIC-CXR** (Eval) | RG (via AGRG) | `Locate and describe the {location}.` (Iterated $\times N$ locations) | *(Concat. of location-specific reports)* |
| **MIMIC-CXR** (Eval) | RG (Hybrid) | `Combine AGRG and GRG generations.` | *(Concat. of AGRG reports + GRG report)* |

## Methodology: Error-Aware Curriculum

Standard multi-task learning suffers from data imbalance. CURE mitigates this via a curriculum that operates at two levels:

1.  **Inter-Dataset Curriculum:** Re-weights the sampling probability of entire datasets (e.g., Chest ImaGenome vs. MS-CXR) based on aggregate error rates.
2.  **Intra-Dataset Curriculum:** Re-weights specific anatomical regions or semantic classes within a dataset based on fine-grained performance metrics (IoU and CXRFEScore).

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/PabloMessina/CURE.git](https://github.com/PabloMessina/CURE.git)
    cd CURE
    ```

2.  **Create and activate a virtual environment:**
    **Note:** This project is tested on **Python 3.10+**.

    ```bash
    # Using conda (Recommended for cluster usage)
    conda create -n vlm python=3.10.18
    conda activate vlm
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the project in editable mode:**

    ```bash
    pip install -e .
    ```

### Configuration (`.env`)

Create a `.env` file in the project root to handle dataset paths.

```bash
cp .env.example .env
# Edit .env with your specific paths (See .env.example for details)
```


## Cluster Usage (SLURM)

> [!NOTE]
> **Cluster Node Names:** The examples below use placeholders like `<node_name>`. Please replace these with the actual node names or partitions available on your specific SLURM cluster (e.g., `gpu-v100`, `node01`, etc.).

This project is optimized for SLURM clusters. All heavy lifting is handled via scripts in the `slurm/` directory.

### 1\. Training (MedGemma with CURE)

The primary training script is `scripts/train_medgemma.py`.

**Syntax:**

```bash
./slurm/submit_medgemma_training.sh <config_path> <num_nodes> <job_time_limit> <conda_env> <node_list>
```

**Example (CURE Curriculum Run):**

```bash
./slurm/submit_medgemma_training.sh \
    configs/training/multi_task_medgemma-4b-it_v20_custom_eval_500_steps_curriculum_3000_steps.yaml \
    1 \
    60 \
    3-00:00:00 \
    vlm \
    <your_node_name>
```

### 2\. Evaluation

Evaluation is task-specific. Below are examples for our CURE model (MedGemma) and baselines (MAIRA-2, CXRMate-RGG24).

#### A. Standard Report Generation (MIMIC-CXR)

This task evaluates the model's ability to generate full reports. For MedGemma (CURE), we often generate reports by iterating through specific anatomical regions (AGRG).

**CURE (MedGemma) Example:**
*Note the use of `image_transforms_kwargs` for deterministic CLAHE during evaluation.*

```bash
CONDA_ENV=vlm sbatch --nodelist=<your_node_name> --mem=35G --time=1-10:00:00 \
    slurm/launch_report_generation_evaluation.sh medgemma mimic-cxr \
    --split "test" \
    --generate_anatomy_grounded_report "right lung" "right lower lung zone" "right costophrenic angle" "left lung" "left lower lung zone" "right hilar structures" "left hilar structures" "left costophrenic angle" "mediastinum" "cardiac silhouette" "right chest wall" "neck" "upper mediastinum" \
    --medgemma_adapter_path /path/to/your/checkpoint-5500/ \
    --image_transforms_kwargs '{"use_model_specific_transforms": true, "model_name": "pil_with_augmentations", "image_size": [448, 448], "is_train": false, "augmenter_override_params": {"clahe_clip_limit": [3.0, 3.0]}}'
```

**Baseline (MAIRA-2) Example:**

```bash
CONDA_ENV=maira sbatch --nodelist=<your_node_name> --mem=50G \
    slurm/launch_report_generation_evaluation.sh maira-2 mimic-cxr \
    --split "test" \
    --max_new_tokens 500 \
    --generate_grounded_report
```

**Baseline (CXRMate-RRG24) Example:**

```bash
CONDA_ENV=maira sbatch --nodelist=<your_node_name> --mem=35G \
    slurm/launch_report_generation_evaluation.sh cxrmate-rrg24 mimic-cxr \
    --split "test"
```

#### B. Anatomy Grounded Report Generation (Chest ImaGenome)

We evaluate on a stratified subset of 1000 samples.

**CURE (MedGemma):**

```bash
CONDA_ENV=vlm sbatch slurm/launch_chest_imagenome_anatomy_grounded_report_gen_evaluation.sh \
    medgemma \
    --medgemma_adapter_path /path/to/checkpoint
```

**Baseline (MAIRA-2):**

```bash
CONDA_ENV=maira sbatch --nodelist=<your_node_name> --time=12:00:00 \
    slurm/launch_chest_imagenome_anatomy_grounded_report_gen_evaluation.sh maira-2 \
    --eval_locations "right shoulder" "left shoulder" "right arm" "left arm" "left breast" "right breast" \
    --location_eval_size 300 \
    --skip_sampling_beyond_core_sample
```

#### C. Grounded Report Generation (PadChest-GR / VinDr-CXR)

**MAIRA-2 on VinDr-CXR (Zero-Shot):**

```bash
CONDA_ENV=maira sbatch --mem=40G \
    slurm/launch_grounded_report_generation_evaluation.sh maira-2 vindrcxr \
    --split "test"
```

**CURE (MedGemma) on PadChest-GR:**

```bash
CONDA_ENV=vlm sbatch --mem=35G --nodelist=<your_node_name> slurm/launch_grounded_report_generation_evaluation.sh \
    medgemma padchest \
    --medgemma_adapter_path /path/to/fine_tuned_checkpoint \
    --image_transforms_kwargs '{"use_model_specific_transforms": true, "model_name": "pil_with_augmentations", "image_size": [448, 448], "is_train": false, "augmenter_override_params": {"clahe_clip_limit": [3.0, 3.0]}}' \
    --split "test"
```


#### D. Phrase Grounding (MS-CXR / VinDr-CXR / PadChest-GR)

**MAIRA-2 on VinDr-CXR (Zero-Shot):**

```bash
CONDA_ENV=maira sbatch --mem=90G \
    slurm/launch_phrase_grounding_evaluation.sh maira-2 vindrcxr \
    --split "test"
```

**CURE (MedGemma) on PadChest-GR:**

```bash
CONDA_ENV=vlm sbatch --mem=35G --nodelist=<your_node_name> slurm/launch_phrase_grounding_evaluation.sh \
    medgemma padchest-gr \
    --medgemma_adapter_path /path/to/fine_tuned_checkpoint \
    --image_transforms_kwargs '{"use_model_specific_transforms": true, "model_name": "pil_with_augmentations", "image_size": [448, 448], "is_train": false, "augmenter_override_params": {"clahe_clip_limit": [3.0, 3.0]}}' \
    --split "test"
```

### 3\. Interactive Mode

To run Jupyter Notebooks on a compute node (with or without GPU), use the `launch_interactive.sh` script.

**Usage:**

```bash
# Launch with 1 GPU and use the vlm conda environment
./slurm/launch_interactive.sh --gpu --env vlm
```

**Workflow:**

1.  Run the script. Wait for the job to be granted.
2.  The script will output an SSH tunnel command (e.g., `ssh -N -L 30019:node-hostname:30019 user@login-node`).
3.  Run that tunnel command in a **local** terminal.
4.  Open the localhost URL provided in the output.

-----

## Notebooks for Analysis

Once the evaluation scripts are run, the predictions will be saved in .jsonl files associated with each evaluated checkpoint. The SLURM logs will indicate the path to the predictions.

Once we have the prediction files, we can run the following notebooks to analyze the results, located in `notebooks/evaluation/`:

Report Generation:

  * `Evaluate_MIMICCXR_Report_Generation.ipynb`

Anatomy Grounded Report Generation:
  * `Evaluate_ChestImaGenome_Anatomy_Grounded_Report_Gen.ipynb`

Phrase Grounding:
  * `Evaluate_MS-CXR_Phrase_Grounding.ipynb`
  * `Evaluate_VinDr-CXR_Phrase_Grounding.ipynb`
  * `Evaluate_PadChest-GR_Phrase_Grounding.ipynb`

Grounded Report Generation:
  * `Evaluate_VinDr-CXR_Grounded_Report_Generation.ipynb`
  * `Evaluate_PadChest-GR_Grounded_Report_Generation.ipynb`
  
These notebooks are used to run evaluation metrics on the predictions. We employ a comprehensive suite of metrics including **IoU** (Localization), **CheXbert** (Clinical Accuracy), **RadGraph** (Entity/Relation Overlap), **CXRFEScore** (Factual Consistency), and **RaTEScore** (Entity-aware Text Similarity).

-----

## 🚀 Quick Start: Loading CURE from Hugging Face

We provide a pre-trained checkpoint on Hugging Face. You can load the model using 4-bit quantization for efficient inference even on consumer GPUs.

For a complete walkthrough including **visualization utilities** for phrase grounding and grounded reports, see the notebook at `notebooks/inference/Load and run CURE from Hugging Face.ipynb`.

### Python Example

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# Configuration
BASE_MODEL_ID = "google/medgemma-4b-it"
ADAPTER_ID = "pamessina/medgemma-4b-it-cure"

# 4-bit Quantization Config (Must match training settings)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

# Load Model & Adapter
base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

model.eval()
print("CURE model and processor are ready for inference.")

```

---

## 🛠 Support and Contributions

We welcome feedback and contributions to improve the reliability of Biomedical VLMs.

* **Issues:** If you encounter bugs, have questions about the methodology, or face issues with cluster setup, please **[open an issue](https://www.google.com/search?q=https://github.com/PabloMessina/ivul-biomed-vlm/issues)**. We aim to respond as quickly as possible.

* **Pull Requests:** Contributions to the `CURE` repository are welcome.

---


## Project Structure

```
CURE/
├── configs/
│   ├── prompts/           # LLM prompts for report labeling and annotation
│   └── training/          # Multi-task & Curriculum training YAML configs
├── notebooks/
│   ├── evaluation/        # Task-specific analysis (Cig, MIMIC, MS-CXR, etc.)
│   └── inference/         # Tutorial: Load and run CURE from Hugging Face
├── scripts/               # Core Python executables (Train, Eval, LLM labeling)
├── slurm/                 # SLURM cluster submission and interactive scripts
├── vlm_research_kit/      # Core library (Internal Logic)
│   ├── data/              # Dataset loaders (MIMIC, PadChest, VinDr, etc.)
│   ├── evaluation/        # Evaluation loop logic
│   ├── metrics/           # Bio-metrics (CheXbert, RadGraph, RaTEScore, etc.)
│   ├── training/          # Training factories and schedulers
│   └── utils/             # Image/BBox/Text processing utilities
├── .env.example           # Template for environment variables
├── pyproject.toml         # Project metadata and build system
└── requirements.txt       # Production dependencies
```

## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{messina2026cure,
  title={CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation},
  author={Messina, Pablo and ...},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## Acknowledgments

This work was supported by the **IVUL group at KAUST**.

Pablo Messina was supported in Chile by **ANID** through **iHEALTH** (ICN2021_004), **CENIA** (FB210017), and the **ANID Scholarship Program/Doctorado Becas Chile/2019-21191569**.