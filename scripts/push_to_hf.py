#!/usr/bin/env python3
"""
Push LoRA fine-tuned models to Hugging Face Hub.

This script supports:
1. Uploading just the LoRA adapter (recommended, smaller file size)
2. Merging LoRA weights with the base model and uploading the full model

Example usage:

# Upload just the LoRA adapter
python scripts/push_to_hf.py \
    --checkpoint_path /path/to/checkpoint-1000 \
    --repo_id your-username/model-name \
    --base_model_id google/medgemma-4b-it

# Upload merged model (LoRA weights merged into base model)
python scripts/push_to_hf.py \
    --checkpoint_path /path/to/checkpoint-1000 \
    --repo_id your-username/model-name \
    --base_model_id google/medgemma-4b-it \
    --merge_weights

# Upload from topk-checkpoints directory
python scripts/push_to_hf.py \
    --checkpoint_path /path/to/experiment/topk-checkpoints/checkpoint-500-eval_overall_iou_cxrfescore-0.750000 \
    --repo_id your-username/model-name \
    --base_model_id google/medgemma-4b-it \
    --private
"""

import argparse
import fnmatch
import logging
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import create_repo, upload_folder
from peft import PeftModel
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

from vlm_research_kit.settings import HF_TOKEN
from vlm_research_kit.utils.logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Push LoRA fine-tuned models to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help=(
            "Path to the checkpoint directory containing the LoRA adapter. "
            "This can be a standard checkpoint (e.g., checkpoint-1000) or "
            "a topk checkpoint directory."
        ),
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., 'username/model-name').",
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        required=True,
        help="Base model ID from Hugging Face (e.g., 'google/medgemma-4b-it').",
    )
    
    # Optional arguments
    parser.add_argument(
        "--merge_weights",
        action="store_true",
        help=(
            "Merge LoRA weights into the base model before uploading. "
            "This creates a full model that doesn't require PEFT for inference, "
            "but results in a much larger upload. Default is to upload only the adapter."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private. Default is public.",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default=None,
        help="Custom commit message for the upload.",
    )
    parser.add_argument(
        "--model_card_template",
        type=str,
        default=None,
        help="Path to a custom model card template (README.md).",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token. If not provided, uses HF_TOKEN from .env file.",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Use Flash Attention 2 when loading the model (requires compatible GPU).",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load base model in 4-bit quantization (for merging on limited VRAM).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Local directory to save the prepared model before uploading. "
            "If not specified, uses a temporary directory."
        ),
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="Additional tags for the model card (space-separated).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Prepare the model locally but don't upload to Hugging Face.",
    )
    
    return parser.parse_args()


def load_base_model(
    base_model_id: str,
    use_flash_attention: bool = False,
    load_in_4bit: bool = False,
    device_map: str = "auto",
):
    """
    Load the base model from Hugging Face.
    
    Args:
        base_model_id: The base model ID.
        use_flash_attention: Whether to use Flash Attention 2.
        load_in_4bit: Whether to load in 4-bit quantization.
        device_map: Device map for model loading.
    
    Returns:
        The loaded model.
    """
    logger.info(f"Loading base model: {base_model_id}")
    
    model_kwargs = {
        "device_map": device_map,
        "torch_dtype": torch.bfloat16,
    }
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2")
    
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = quantization_config
        logger.info("Loading model in 4-bit quantization")
    
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        **model_kwargs,
    )
    
    logger.info("Base model loaded successfully")
    return model


def load_adapter_and_merge(
    base_model,
    checkpoint_path: Path,
    merge_weights: bool = False,
):
    """
    Load the LoRA adapter and optionally merge with base model.
    
    Args:
        base_model: The loaded base model.
        checkpoint_path: Path to the checkpoint directory.
        merge_weights: Whether to merge weights into the base model.
    
    Returns:
        The model (merged or with adapter).
    """
    logger.info(f"Loading LoRA adapter from: {checkpoint_path}")
    
    # Load the PEFT model
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        is_trainable=False,
    )
    logger.info("LoRA adapter loaded successfully")
    
    if merge_weights:
        logger.info("Merging LoRA weights into base model...")
        model = model.merge_and_unload()
        logger.info("Weights merged successfully")
    
    return model


def upload_adapter_only(
    checkpoint_path: Path,
    repo_id: str,
    hf_token: str,
    private: bool = False,
    commit_message: Optional[str] = None,
    dry_run: bool = False,
):
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
    
    # Create or get the repository
    if not dry_run:
        logger.info(f"Creating/getting repository: {repo_id}")
        create_repo(repo_id=repo_id, token=hf_token, private=private, exist_ok=True)
    
    commit_msg = commit_message or f"Upload LoRA adapter from {checkpoint_path.name}"
    
    # We use ONLY allow_patterns to be strictly explicit. 
    # This prevents accidental uploads of large optimizer files.
    relevant_patterns = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "chat_template.jinja",
        "tokenizer_config.json",
        "tokenizer.json",
        "processor_config.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "README.md",
    ]

    if dry_run:
        logger.info("Dry run mode - Files that WOULD be uploaded:")
        for file in checkpoint_path.iterdir():
            if any(fnmatch.fnmatch(file.name, p) for p in relevant_patterns):
                logger.info(f"  [WILL UPLOAD] {file.name}")
            else:
                logger.info(f"  [SKIPPING]    {file.name}")
        return
    
    logger.info(f"Uploading selected adapter files to {repo_id}...")
    upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=repo_id,
        token=hf_token,
        commit_message=commit_msg,
        allow_patterns=relevant_patterns, # Strict inclusion
    )
    
    logger.info(f"Successfully uploaded adapter to: https://huggingface.co/{repo_id}")


def upload_merged_model(
    checkpoint_path: Path,
    repo_id: str,
    base_model_id: str,
    hf_token: str,
    private: bool = False,
    commit_message: Optional[str] = None,
    output_dir: Optional[str] = None,
    use_flash_attention: bool = False,
    load_in_4bit: bool = False,
    dry_run: bool = False,
):
    """
    Merge LoRA weights into base model and upload the full model.
    
    Args:
        checkpoint_path: Path to the checkpoint directory.
        repo_id: Hugging Face repository ID.
        base_model_id: Base model ID.
        hf_token: Hugging Face token.
        private: Whether to make the repo private.
        commit_message: Custom commit message.
        output_dir: Local directory to save merged model.
        use_flash_attention: Whether to use Flash Attention.
        load_in_4bit: Whether to load in 4-bit for merging.
        dry_run: If True, don't actually upload.
    """
    import tempfile
    
    checkpoint_path = Path(checkpoint_path)
    
    # Validate checkpoint path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
    
    # Set up output directory
    if output_dir:
        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="merged_model_")
        save_dir = Path(temp_dir)
    
    logger.info(f"Merged model will be saved to: {save_dir}")
    
    # Load base model
    base_model = load_base_model(
        base_model_id=base_model_id,
        use_flash_attention=use_flash_attention,
        load_in_4bit=load_in_4bit,
    )
    
    # Load adapter and merge
    merged_model = load_adapter_and_merge(
        base_model=base_model,
        checkpoint_path=checkpoint_path,
        merge_weights=True,
    )
    
    # Save merged model
    logger.info(f"Saving merged model to: {save_dir}")
    merged_model.save_pretrained(save_dir)
    
    # Also save the processor
    logger.info("Saving processor...")
    processor = AutoProcessor.from_pretrained(base_model_id)
    processor.save_pretrained(save_dir)
    
    if dry_run:
        logger.info("Dry run mode - skipping upload to Hugging Face")
        logger.info(f"Merged model saved to: {save_dir}")
        return
    
    # Create or get the repository
    logger.info(f"Creating/getting repository: {repo_id}")
    create_repo(
        repo_id=repo_id,
        token=hf_token,
        private=private,
        exist_ok=True,
    )
    
    # Upload the merged model folder
    commit_msg = commit_message or f"Upload merged model from {checkpoint_path.name}"
    
    logger.info(f"Uploading merged model to {repo_id}...")
    logger.info("This may take a while for large models...")
    
    upload_folder(
        folder_path=str(save_dir),
        repo_id=repo_id,
        token=hf_token,
        commit_message=commit_msg,
    )
    
    logger.info(f"Successfully uploaded merged model to: https://huggingface.co/{repo_id}")
    
    # Clean up temporary directory if we created one
    if not output_dir:
        import shutil
        logger.info(f"Cleaning up temporary directory: {save_dir}")
        shutil.rmtree(save_dir)


def main():
    args = parse_args()
    
    # Get HF token
    hf_token = args.hf_token or HF_TOKEN
    if not hf_token:
        raise ValueError(
            "Hugging Face token not found. "
            "Either set HF_TOKEN in your .env file or pass --hf_token argument."
        )
    
    checkpoint_path = Path(args.checkpoint_path)
    
    logger.info("=" * 60)
    logger.info("Push to Hugging Face Hub")
    logger.info("=" * 60)
    logger.info(f"Checkpoint path: {checkpoint_path}")
    logger.info(f"Repository ID: {args.repo_id}")
    logger.info(f"Base model: {args.base_model_id}")
    logger.info(f"Merge weights: {args.merge_weights}")
    logger.info(f"Private repo: {args.private}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 60)
    
    if args.merge_weights:
        upload_merged_model(
            checkpoint_path=checkpoint_path,
            repo_id=args.repo_id,
            base_model_id=args.base_model_id,
            hf_token=hf_token,
            private=args.private,
            commit_message=args.commit_message,
            tags=args.tags,
            model_card_template=args.model_card_template,
            output_dir=args.output_dir,
            use_flash_attention=args.use_flash_attention,
            load_in_4bit=args.load_in_4bit,
            dry_run=args.dry_run,
        )
    else:
        upload_adapter_only(
            checkpoint_path=checkpoint_path,
            repo_id=args.repo_id,
            hf_token=hf_token,
            private=args.private,
            commit_message=args.commit_message,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
