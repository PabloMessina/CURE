import os
import logging
from typing import Any, Optional, Tuple
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    AutoModel,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)


def load_medgemma_model(
    base_model_id: str, adapter_path: Optional[str]
) -> Tuple[torch.nn.Module, Any]:
    """
    Loads the MedGemma model. If an adapter_path is provided, it loads the
    fine-tuned version with 4-bit quantization. Otherwise, it loads the
    original base model.
    """
    if adapter_path:
        logger.info(
            f"Loading base model ({base_model_id}) with 4-bit quantization for LoRA adapter..."
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        logger.info(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        logger.info("Fine-tuned MedGemma model loaded successfully.")
    else:
        logger.info(
            f"Loading original base model ({base_model_id}) without adapter..."
        )
        model = AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        logger.info("Original MedGemma model loaded successfully.")

    logger.info(f"Loading processor for {base_model_id}...")
    processor = AutoProcessor.from_pretrained(base_model_id)
    processor.tokenizer.padding_side = "left"

    model.eval()
    logger.info("MedGemma model and processor set to evaluation mode.")
    return model, processor


def load_maira2_model(
    model_id: str, revision: str
) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    """Loads the MAIRA-2 model and its processor."""
    log_revision = revision if revision else "main (latest)"
    logger.info(f"Loading MAIRA-2 model ({model_id}, revision {log_revision})...")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN env var not set. Model download may fail.")

    # For better performance and memory usage on modern GPUs (Ampere series+),
    # load the model in bfloat16.
    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "auto"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token,
        revision=revision,
        torch_dtype=torch_dtype,
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token,
        revision=revision,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(
        f"MAIRA-2 model and processor loaded and moved to {device} in evaluation mode."
    )
    return model, processor


def load_cxrmate_rrg24_model(
    model_id: str = "aehrc/cxrmate-rrg24",
    revision: Optional[str] = "ef90c2725315efaabf1eb1762d9b903c5e57acdd",
) -> Tuple[AutoModel, AutoTokenizer]:
    """
    Loads the CXRMate-RRG24 model and tokenizer.

    Args:
        model_id: The Hugging Face model ID.
        revision: The specific git revision of the model to load.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    logger.info(f"Loading CXRMate-RRG24 model from '{model_id}' at revision '{revision}'...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision
    )
    model = (
        AutoModel.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        )
        .to(device=device)
        .eval()
    )
    logger.info("CXR-Mate-RRG model loaded successfully.")
    # For consistency with other functions, we return model, tokenizer
    # The tokenizer will be used as the 'processor'
    return model, tokenizer