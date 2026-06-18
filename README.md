# CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://pablomessina.github.io/cure-project-page/)
[![CVPR Page](https://img.shields.io/badge/CVPR-Official%20Proceedings-blueviolet)](https://openaccess.thecvf.com/content/CVPR2026/html/Messina_CURE_Curriculum-guided_Multi-task_Training_for_Reliable_Anatomy_Grounded_Report_Generation_CVPR_2026_paper.html)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1E4OyQZ58tvqCrMB-zWQ6rb-yJtZ9wAQL?usp=sharing)
[![Hugging Face Model](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface)](https://huggingface.co/pamessina/medgemma-4b-it-cure)
[![Paper](https://img.shields.io/badge/arXiv-2601.15408-b31b1b.svg)](https://arxiv.org/abs/2601.15408)
[![Slides](https://img.shields.io/badge/Google-Slides-F4B400?logo=googledrive)](https://docs.google.com/presentation/d/1Oek2gXr8CdbEndeGAj4oZLC24tnITuNhLiL8d2yHsn8/edit?usp=sharing)
[![Live Oral](https://img.shields.io/badge/Live-Oral_Presentation-FF0000?logo=youtube)](https://youtu.be/w53pVA7ubvU?t=19417)
[![5-Min Video](https://img.shields.io/badge/5--Min-Video-FF0000?logo=youtube)](https://youtu.be/p7qbBITjtRc)

**Official PyTorch implementation of CURE. Accepted as an Oral Presentation at CVPR 2026.**

**CURE**, short for ***Cu**rriculum-guided Multi-task Training for **Re**liable Anatomy Grounded Report Generation*, is an error-aware curriculum learning framework for reliable anatomy-grounded radiology report generation with medical Vision-Language Models.

CURE improves grounding and report quality without requiring additional data. It addresses a key failure mode of grounded medical VLMs: because many grounding datasets are abnormality-biased, models often learn to associate visual grounding mainly with abnormalities and may hallucinate findings when asked to ground normal anatomy. CURE mitigates this by introducing a fine-grained anatomy-grounded task structure and an adaptive error-aware training curriculum.

This repository contains the code, configuration files, SLURM utilities, notebooks, and evaluation tools used to train and evaluate CURE.

## Highlights

- **CVPR 2026 Oral:** CURE was accepted as an oral presentation at CVPR 2026.
- **Hugging Face checkpoint:** A LoRA adapter finetuned from [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it) is available at [`pamessina/medgemma-4b-it-cure`](https://huggingface.co/pamessina/medgemma-4b-it-cure).
- **Project page:** See the project website for paper, videos, slides, and visual results: <https://pablomessina.github.io/cure-project-page/>.
- **Colab demo:** For the easiest inference experience, including image preprocessing and visualization utilities, use the official Colab notebook: <https://colab.research.google.com/drive/1E4OyQZ58tvqCrMB-zWQ6rb-yJtZ9wAQL?usp=sharing>.

## Contributions & Takeaways

CURE has two main components:

1. **Anatomy-Grounded Report Generation (AGRG):**  
   A task that teaches the model to localize and describe specific anatomical regions. By decomposing reports using Chest ImaGenome annotations, CURE provides grounding supervision for both normal and abnormal anatomy, reducing abnormality-biased grounding behavior.

2. **Error-Aware Curriculum:**  
   A dynamic adaptive sampling strategy that addresses dataset and class imbalance. The curriculum uses validation performance to prioritize datasets, anatomical regions, and semantic classes where the model currently performs worse.

**Main takeaway:** To build reliable medical VLMs, it is important to ground and describe normal anatomy, not only abnormalities, and to adaptively sample difficult datasets and classes during training.

## Key Results

CURE achieves the following:

- **67% relative reduction in abnormality hallucinations** on average across six anatomical locations. For example, clavicle hallucination rates dropped from approximately 60% to 1%.
- **Improved visual grounding** over strong baselines such as MAIRA-2 on Phrase Grounding and Grounded Report Generation on specific datasets including MS-CXR, PadChest-GR, and VinDr-CXR.
- **Competitive standard report generation quality** on MIMIC-CXR by combining AGRG outputs over 29 anatomical locations with standard Grounded Report Generation (GRG).

## Model Details

- **Model name:** CURE
- **Model type:** Medical Vision-Language Model adapter
- **Base model:** [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it)
- **Adapter:** [`pamessina/medgemma-4b-it-cure`](https://huggingface.co/pamessina/medgemma-4b-it-cure)
- **Adapter type:** LoRA / PEFT
- **Pipeline:** Image-text-to-text
- **Developed by:** Pablo Messina, Andrés Villa, Juan León Alcázar, Karen Sánchez, Carlos Hinojosa, Denis Parra, Álvaro Soto, Bernard Ghanem
- **Affiliations:** Pontificia Universidad Católica de Chile, King Abdullah University of Science and Technology (KAUST), CENIA, iHEALTH
- **License:** The base model and adapter are subject to the Health AI Developer Foundations terms. See: <https://developers.google.com/health-ai-developer-foundations/terms>


---

## Quick Start: Loading CURE from Hugging Face

We provide a pretrained CURE LoRA adapter on Hugging Face:

- **Base model:** [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it)
- **CURE adapter:** [`pamessina/medgemma-4b-it-cure`](https://huggingface.co/pamessina/medgemma-4b-it-cure)

The adapter can be loaded with 4-bit quantization for efficient inference on consumer GPUs.

For the easiest end-to-end inference experience, including the image transformation pipeline and bounding box visualization utilities, use the official Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1E4OyQZ58tvqCrMB-zWQ6rb-yJtZ9wAQL?usp=sharing)

A complete local walkthrough is also available in:

```text
notebooks/inference/Load and run CURE from Hugging Face.ipynb
```

### Python Example

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

BASE_MODEL_ID = "google/medgemma-4b-it"
ADAPTER_ID = "pamessina/medgemma-4b-it-cure"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, ADAPTER_ID)

processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
processor.tokenizer.padding_side = "left"

model.eval()
print("CURE model and processor are ready for inference.")
```

> [!IMPORTANT]
> For best performance, use the same image preprocessing pipeline used during evaluation. In particular, CURE uses the `pil_with_augmentations` transform pipeline with deterministic CLAHE settings during evaluation. See the inference notebook and evaluation examples for details.

## Supported Tasks and Prompt Interface

CURE is trained as an instruction-following medical vision-language model for grounded radiology report generation. It supports three main task families:

1. **Phrase Grounding (PG):** localize a requested finding or phrase.
2. **Grounded Report Generation (GRG):** generate a report with inline bounding boxes.
3. **Anatomy-Grounded Report Generation (AGRG):** locante and/or describe a specific anatomical region.

In the paper, **Anatomy-Grounded Report Generation (AGRG)** is used as an umbrella term for three anatomy-specific subtasks:

- **Locate:** localize an anatomical region.
- **Describe:** generate a description for an anatomical region.
- **Locate and describe:** jointly localize and describe an anatomical region.

Bounding boxes are represented as normalized coordinates in the format:

```text
[cx, cy, w, h]
```

where `cx` and `cy` are the box center coordinates, and `w` and `h` are the box width and height.

### Prompt Templates

| Task Family | Subtask | Prompt | Expected Output |
| :--- | :--- | :--- | :--- |
| **Phrase Grounding** | Phrase localization | `Ground the phrase: {phrase}` | `{phrase}: [cx, cy, w, h]` |
| **Grounded Report Generation** | Full grounded report | `Generate a grounded report.` | `Finding sentence [cx, cy, w, h]. ...` |
| **Anatomy-Grounded Report Generation** | Locate | `Locate the {location}.` | `Location of {location}: [cx, cy, w, h].` |
| **Anatomy-Grounded Report Generation** | Describe | `Describe the {location}.` | `Description of {location}: ...` |
| **Anatomy-Grounded Report Generation** | Locate and describe | `Locate and describe the {location}.` | `Location of {location}: [cx, cy, w, h]. Description: ...` |

### Examples

#### Phrase Grounding

```text
Ground the phrase: cardiomegaly
```

Example output:

```text
cardiomegaly: [0.52, 0.58, 0.31, 0.24]
```

#### Grounded Report Generation

```text
Generate a grounded report.
```

Example output:

```text
Mild left basilar atelectasis [0.41, 0.72, 0.18, 0.12]. No pleural effusion.
```

#### Anatomy-Grounded Report Generation: Locate

```text
Locate the cardiac silhouette.
```

Example output:

```text
Location of cardiac silhouette: [0.50, 0.61, 0.28, 0.24].
```

#### Anatomy-Grounded Report Generation: Describe

```text
Describe the left lower lung zone.
```

Example output:

```text
Description of left lower lung zone: Mild linear atelectatic opacity is present at the left lung base.
```

#### Anatomy-Grounded Report Generation: Locate and Describe

```text
Locate and describe the right lung.
```

Example output:

```text
Location of right lung: [0.35, 0.52, 0.31, 0.62]. Description: The right lung is clear without focal airspace opacity.
```

## Dataset and Evaluation Mapping

The same prompt interface is used across multiple datasets and evaluation settings.

| Dataset | Evaluation Setting | Task Family | Prompt Pattern |
| :--- | :--- | :--- | :--- |
| **MS-CXR** | Phrase grounding | PG | `Ground the phrase: {phrase}` |
| **PadChest-GR** | Phrase grounding | PG | `Ground the phrase: {phrase}` |
| **PadChest-GR** | Grounded report generation | GRG | `Generate a grounded report.` |
| **Chest ImaGenome** | Anatomy-grounded report generation | AGRG | `Locate the {location}.`, `Describe the {location}.`, or `Locate and describe the {location}.` |
| **VinDr-CXR** | Zero-shot phrase grounding | PG | `Ground the phrase: {phrase}` |
| **VinDr-CXR** | Zero-shot grounded report generation | GRG | `Generate a grounded report.` |
| **MIMIC-CXR** | Report generation via GRG | GRG | `Generate a grounded report.` |
| **MIMIC-CXR** | Report generation via AGRG | AGRG | Iterate over anatomical locations using AGRG prompts |
| **MIMIC-CXR** | Hybrid report generation | AGRG + GRG | Concatenate AGRG outputs with the GRG output |

For MIMIC-CXR report generation, CURE can be evaluated in three modes:

- **GRG mode:** generate one full grounded report using `Generate a grounded report.`
- **AGRG mode:** query multiple anatomical regions independently using AGRG prompts and concatenate the resulting anatomy-specific descriptions.
- **Hybrid mode:** combine anatomy-specific AGRG outputs with the GRG output.

## Methodology: Error-Aware Curriculum

Standard multi-task learning in medical VLMs suffers from strong data imbalance across datasets, tasks, abnormality classes, and anatomical regions. CURE mitigates this using an adaptive curriculum that updates sampling probabilities based on validation performance.

The curriculum operates at two levels:

1. **Inter-Dataset Curriculum:**  
   Re-weights the sampling probability of entire datasets, such as Chest ImaGenome, MS-CXR, and PadChest-GR, based on aggregate error rates.

2. **Intra-Dataset Curriculum:**  
   Re-weights anatomical regions or semantic classes within a dataset based on fine-grained performance metrics.

For grounded tasks, the curriculum uses localization and text-quality signals such as **IoU** and **CXRFEScore**. This allows the model to focus on difficult datasets, underperforming anatomical regions, and challenging semantic classes throughout training.

## Setup and Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/PabloMessina/CURE.git
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

### 1. Training MedGemma with CURE

The primary training script is:

```text
scripts/train_medgemma.py
```

Training jobs are submitted through the SLURM wrapper:

```text
slurm/submit_medgemma_training.sh
```

**Syntax:**

```bash
./slurm/submit_medgemma_training.sh <config_path> [num_gpus] [memory_gb] [time_limit] [conda_env] [target_node]
```

**Arguments:**

- `config_path`: Path to the training YAML configuration file.
- `num_gpus`: Number of GPUs to request. Defaults to `1`.
- `memory_gb`: Amount of RAM in GB to request. Defaults to `64`.
- `time_limit`: Maximum job runtime in SLURM format, e.g. `3-00:00:00`. Defaults to `1-00:00:00`.
- `conda_env`: Name of the conda environment to use. Defaults to `py313`.
- `target_node`: Optional node hostname. If omitted, SLURM will schedule the job on any available eligible node.

**Example: CURE curriculum run**

```bash
./slurm/submit_medgemma_training.sh \
  configs/training/multi_task_medgemma-4b-it_v20_custom_eval_500_steps_curriculum_3000_steps.yaml \
  1 \
  60 \
  3-00:00:00 \
  vlm \
  <your_node_name>
```

This requests one GPU, 60 GB of RAM, a three-day time limit, the `vlm` conda environment, and optionally targets `<your_node_name>`.

### 2\. Evaluation

Evaluation is task-specific. Below are examples for our CURE model (MedGemma) and baselines (MAIRA-2, CXRMate-RRG24).

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

## Notebooks

For inference and visualization, we recommend starting with the official Colab notebook or the local inference notebook:

- [Official Colab demo](https://colab.research.google.com/drive/1E4OyQZ58tvqCrMB-zWQ6rb-yJtZ9wAQL?usp=sharing)
- `notebooks/inference/Load and run CURE from Hugging Face.ipynb`

For qualitative examples, paper links, slides, videos, and visual results, see the project page: <https://pablomessina.github.io/cure-project-page/>.

### Evaluation Analysis

After running the evaluation scripts, predictions are saved as `.jsonl` files associated with each evaluated checkpoint. The SLURM logs indicate the paths to these prediction files.

Once the prediction files are available, use the notebooks in `notebooks/evaluation/` to compute metrics and analyze results.

**Report Generation**

- `Evaluate_MIMICCXR_Report_Generation.ipynb`

**Anatomy-Grounded Report Generation**

- `Evaluate_ChestImaGenome_Anatomy_Grounded_Report_Gen.ipynb`

**Phrase Grounding**

- `Evaluate_MS-CXR_Phrase_Grounding.ipynb`
- `Evaluate_VinDr-CXR_Phrase_Grounding.ipynb`
- `Evaluate_PadChest-GR_Phrase_Grounding.ipynb`

**Grounded Report Generation**

- `Evaluate_VinDr-CXR_Grounded_Report_Generation.ipynb`
- `Evaluate_PadChest-GR_Grounded_Report_Generation.ipynb`

These notebooks are used to run evaluation metrics on the generated predictions. We employ a comprehensive suite of metrics, including **IoU** for localization, **CheXbert** for clinical accuracy, **RadGraph** for entity/relation overlap, **CXRFEScore** for factual consistency, and **RaTEScore** for entity-aware text similarity.

---

## Project Structure

```
CURE/
├── configs/
│   ├── prompts/           # LLM prompts for report labeling and annotation
│   └── training/          # Multi-task & Curriculum training YAML configs
├── notebooks/
│   ├── evaluation/        # Task-specific analysis (Chest ImaGenome, MIMIC-CXR, MS-CXR, etc.)
│   └── inference/         # Tutorial: Load and run CURE from Hugging Face
├── scripts/               # Core Python executables (Train, Eval, LLM labeling)
├── slurm/                 # SLURM cluster submission and interactive scripts
├── vlm_research_kit/      # Core library (Internal Logic)
│   ├── data/              # Dataset loaders (MIMIC-CXR, PadChest-GR, VinDr-CXR, etc.)
│   ├── evaluation/        # Evaluation loop logic
│   ├── metrics/           # Bio-metrics (CheXbert, RadGraph, RaTEScore, etc.)
│   ├── training/          # Training factories and schedulers
│   └── utils/             # Image/BBox/Text processing utilities
├── .env.example           # Template for environment variables
├── pyproject.toml         # Project metadata and build system
└── requirements.txt       # Production dependencies
```

---

## Support and Contributions

We welcome feedback and contributions to improve the reliability of Biomedical VLMs.

- **Issues:** If you encounter bugs, have questions about the methodology, or face issues with cluster setup, please [open an issue](https://github.com/PabloMessina/CURE/issues). We aim to respond as quickly as possible.
- **Pull Requests:** Contributions to the `CURE` repository are welcome.

* **Pull Requests:** Contributions to the `CURE` repository are welcome.

## License and Terms

CURE is distributed as a PEFT/LoRA adapter for [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it). Use of the base model and adapter is subject to the applicable Health AI Developer Foundations terms:

<https://developers.google.com/health-ai-developer-foundations/terms>

Please ensure that your use complies with the relevant model license, dataset licenses, institutional policies, and regulations for medical AI research.

## Disclaimer

CURE is intended for research use. It is not a medical device and should not be used for clinical decision-making without appropriate validation, regulatory approval, and expert oversight.

## Citation

If you find this code or model useful, please cite our CVPR 2026 paper:

Paper: <https://arxiv.org/abs/2601.15408>

```bibtex
@InProceedings{Messina_2026_CVPR,
    author    = {Messina, Pablo and Villa, Andr\'es and Alcazar, Juan Leon and Sanchez, Karen and Hinojosa, Carlos and Parra, Denis and Soto, Alvaro and Ghanem, Bernard},
    title     = {CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {36279-36289}
}
```

## Acknowledgments

This work was conducted while P. Messina was a remote research intern at the Image and Video Understanding Lab (IVUL) at KAUST, under the supervision of B. Ghanem. P. Messina was supported by the ANID Scholarship Program (Doctorado Becas Chile 2019-21191569). We also acknowledge the support of Fondecyt grant 1231724. This work was also funded by ANID - Millennium Science Initiative Program - ICN2021_004 (iHEALTH) as well as ICN17_002 (IMFD), and by the National Center for Artificial Intelligence (CENIA) FB210017, Basal Funds for Centers of Excellence (ANID). The research reported in this publication was supported by funding from King Abdullah University of Science and Technology (KAUST) - Center of Excellence for Generative AI, under award number 5940.