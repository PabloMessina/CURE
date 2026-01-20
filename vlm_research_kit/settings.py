import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
from platformdirs import user_cache_dir

# Determine the project root directory. This assumes config.py is one level
# down from the project root (e.g., in vlm_research_kit/). Adjust if needed.
# If config.py is directly in the root, use Path('.')
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Construct the path to the .env file in the project root
DOTENV_PATH = PROJECT_ROOT / ".env"

# Directory for project metrics resources
PROJECT_METRICS_RESOURCES_DIR = PROJECT_ROOT / "vlm_research_kit" / "metrics" / "resources"

# Directory for LLM prompts
LLM_PROMPTS_DIR = PROJECT_ROOT / "configs" / "prompts"

# Directory for scripts
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Load the .env file. load_dotenv() will search for .env in the current
# directory or parent directories. Explicitly providing the path is robust.
# It's safe to call this even if the file doesn't exist or is empty.
load_dotenv(dotenv_path=DOTENV_PATH)

# ==== Load specific variables from the .env file ====

def raise_error_if_none(variable, variable_name):
    """Raises an error if the variable is None."""
    if variable is None:
        raise ValueError(f"Environment variable '{variable_name}' not set in the .env file.")

def raise_warning_if_none(variable, variable_name):
    """Raises a warning if the variable is None."""
    if variable is None:
        warnings.warn(
            f"Environment variable '{variable_name}' not set in the .env file.",
            UserWarning
        )

# --- Experiments Directory --
EXPERIMENTS_DIR = os.getenv("EXPERIMENTS_DIR")
raise_error_if_none(EXPERIMENTS_DIR, "EXPERIMENTS_DIR")

# -- Cache Directory --
CACHE_DIR = os.getenv("CACHE_DIR")
raise_warning_if_none(CACHE_DIR, "CACHE_DIR")

# -- CXRFESCORE Cache Directory --
CXRFESCORE_CACHE_DIR = user_cache_dir("cxrfescore_cache")

# -- CheXbert Cache Directory --
CHEXBERT_CACHE_DIR = user_cache_dir("chexbert_cache")

# -- RadGraph Cache Directory --
RADGRAPH_CACHE_DIR = user_cache_dir("radgraph_cache")

# -- RaTEScore Cache Directory --
RATESCORE_CACHE_DIR = user_cache_dir("ratescore_cache")

# -- HF Hub Token --
HF_TOKEN = os.getenv("HF_TOKEN")
raise_warning_if_none(HF_TOKEN, "HF_TOKEN")

# --- PadChest-GR ---
PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH = os.getenv("PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH")
raise_warning_if_none(PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH, "PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH")

PADCHEST_GR_MASTER_TABLE_CSV_PATH = os.getenv("PADCHEST_GR_MASTER_TABLE_CSV_PATH")
raise_warning_if_none(PADCHEST_GR_MASTER_TABLE_CSV_PATH, "PADCHEST_GR_MASTER_TABLE_CSV_PATH")

PADCHEST_GR_JPG_DIR = os.getenv("PADCHEST_GR_JPG_DIR")
raise_warning_if_none(PADCHEST_GR_JPG_DIR, "PADCHEST_GR_JPG_DIR")

PADCHEST_GR_PROGRESSION_PRIOR_STUDIES_JPG_DIR = os.getenv("PADCHEST_GR_PROGRESSION_PRIOR_STUDIES_JPG_DIR")
raise_warning_if_none(PADCHEST_GR_PROGRESSION_PRIOR_STUDIES_JPG_DIR, "PADCHEST_GR_PROGRESSION_PRIOR_STUDIES_JPG_DIR")

# --- MIMIC-CXR ---
MIMIC_CXR_METADATA_CSV_PATH = os.getenv("MIMIC_CXR_METADATA_CSV_PATH")
raise_warning_if_none(MIMIC_CXR_METADATA_CSV_PATH, "MIMIC_CXR_METADATA_CSV_PATH")

MIMIC_CXR_SPLIT_CSV_PATH = os.getenv("MIMIC_CXR_SPLIT_CSV_PATH")
raise_warning_if_none(MIMIC_CXR_SPLIT_CSV_PATH, "MIMIC_CXR_SPLIT_CSV_PATH")

MIMIC_CXR_POSTPROCESSED_REPORTS_JSON_PATH = os.getenv("MIMIC_CXR_POSTPROCESSED_REPORTS_JSON_PATH")
raise_warning_if_none(MIMIC_CXR_POSTPROCESSED_REPORTS_JSON_PATH, "MIMIC_CXR_POSTPROCESSED_REPORTS_JSON_PATH")

MIMIC_CXR_IMAGES_DIR = os.getenv("MIMIC_CXR_IMAGES_DIR")
raise_warning_if_none(MIMIC_CXR_IMAGES_DIR, "MIMIC_CXR_IMAGES_DIR")

# --- MS-CXR ---
MS_CXR_LOCAL_ALIGNMENT_CSV_PATH = os.getenv('MS_CXR_LOCAL_ALIGNMENT_V1.1.0_CSV_PATH')
raise_warning_if_none(MS_CXR_LOCAL_ALIGNMENT_CSV_PATH, "MS_CXR_LOCAL_ALIGNMENT_V1.1.0_CSV_PATH")

# --- Chest ImaGenome ---
CHEST_IMAGENOME_LOCATION_REPORT_SNIPPETS_PATH = os.getenv("CHEST_IMAGENOME_LOCATION_REPORT_SNIPPETS_PATH")
raise_warning_if_none(CHEST_IMAGENOME_LOCATION_REPORT_SNIPPETS_PATH, "CHEST_IMAGENOME_LOCATION_REPORT_SNIPPETS_PATH")

# --- VinDr-CXR ---
VINDRCXR_ANNOTATIONS_DIR = os.getenv("VINDRCXR_ANNOTATIONS_DIR")
raise_warning_if_none(VINDRCXR_ANNOTATIONS_DIR, "VINDRCXR_ANNOTATIONS_DIR")

VINDRCXR_TRAIN_JPG_DIR = os.getenv("VINDRCXR_TRAIN_JPG_DIR")
raise_warning_if_none(VINDRCXR_TRAIN_JPG_DIR, "VINDRCXR_TRAIN_JPG_DIR")

VINDRCXR_TEST_JPG_DIR = os.getenv("VINDRCXR_TEST_JPG_DIR")
raise_warning_if_none(VINDRCXR_TEST_JPG_DIR, "VINDRCXR_TEST_JPG_DIR")

# --- Gemini Annotations ---
GEMINI_2_5_FLASH_LITE_MIMIC_CXR_ANATOMY_SPECIFIC_REPORTS_JSONL_PATH = os.getenv("GEMINI_2_5_FLASH_LITE_MIMIC_CXR_ANATOMY_SPECIFIC_REPORTS_JSONL_PATH")
raise_warning_if_none(GEMINI_2_5_FLASH_LITE_MIMIC_CXR_ANATOMY_SPECIFIC_REPORTS_JSONL_PATH, "GEMINI_2_5_FLASH_LITE_MIMIC_CXR_ANATOMY_SPECIFIC_REPORTS_JSONL_PATH")

GEMINI_2_5_FLASH_LITE_ANNOTATED_MINI_REPORTS_JSONL_PATH = os.getenv("GEMINI_2_5_FLASH_LITE_ANNOTATED_MINI_REPORTS_JSONL_PATH")
raise_warning_if_none(GEMINI_2_5_FLASH_LITE_ANNOTATED_MINI_REPORTS_JSONL_PATH, "GEMINI_2_5_FLASH_LITE_ANNOTATED_MINI_REPORTS_JSONL_PATH")

# --- Final messages ---
print(f"Config loaded. Project Root: {PROJECT_ROOT}")
print(f"Experiments Dir: {EXPERIMENTS_DIR}")