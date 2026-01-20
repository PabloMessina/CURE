import logging
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# --- Type Hints ---
BatchTokenizerFn = Callable[
    [
        List[str] # List of report strings
    ],
    Dict[str, torch.Tensor] # Tokenized batch dictionary
]
LossComputationFn = Callable[
    [
        nn.Module,                          # model
        Any,                                # batch (use Any, as it might not always be dict)
        torch.device,                       # device
        Optional[PreTrainedTokenizerBase]   # tokenizer
    ],
    torch.Tensor # return loss
]

# =============================================================================
# 1. Batch Tokenization Strategies (for Training DataLoaders)
# =============================================================================

def _cxrmate_rrg24_tokenize_batch__full_report(
    report_list: List[str],
    tokenizer: PreTrainedTokenizerBase,
    max_len: int,
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes a batch of full reports using cxrmate-style teacher forcing logic.

    Treats the entire report string as the 'findings' section for tokenization,
    adding BOS, SEP (after findings), and EOS tokens. Prepares inputs and labels
    suitable for autoregressive language model training.

    Args:
        report_list: List of full report strings for the batch.
        tokenizer: The tokenizer instance.
        max_len: Maximum sequence length for tokenization (excluding special tokens
                 added before truncation).

    Returns:
        A dictionary containing tokenized tensors:
        - 'decoder_input_ids': Input IDs for the decoder (shifted).
        - 'decoder_attention_mask': Attention mask for the decoder inputs.
        - 'label_ids': Target label IDs for loss calculation.

    Reference:
        Inspired by https://huggingface.co/aehrc/cxrmate-rrg24/blob/main/modelling_cxrrg.py#L346
    """
    # Format reports: BOS + report + SEP + EOS
    # Note: The original code used findings + SEP + impression + EOS.
    # This version uses report + SEP + EOS, effectively treating the whole report as 'findings'.
    reports_formatted = [f'{tokenizer.bos_token}{r}{tokenizer.sep_token}{tokenizer.eos_token}'
                         for r in report_list]

    # Tokenize reports
    tokenized = tokenizer(
        reports_formatted,
        padding='longest',
        truncation=True,
        max_length=max_len + 1,  # +1 allows space for the label shift
        return_tensors='pt',
        return_token_type_ids=False,
        add_special_tokens=False, # We added them manually
    )

    # Create labels (target) by shifting input_ids right (remove BOS)
    # Shape: [batch_size, seq_len]
    label_ids = tokenized['input_ids'][:, 1:].detach().clone()

    # Create decoder_input_ids (input) by removing the last token (often EOS)
    # Shape: [batch_size, seq_len]
    decoder_input_ids = tokenized['input_ids'][:, :-1]

    # Create decoder_attention_mask by shifting the original mask right
    # Shape: [batch_size, seq_len]
    decoder_attention_mask = tokenized['attention_mask'][:, 1:]

    batch_dict = {
        'decoder_input_ids': decoder_input_ids,
        'decoder_attention_mask': decoder_attention_mask,
        'label_ids': label_ids,
    }

    return batch_dict


def _cxrmate_rrg24_tokenize_batch__prompt_guided_report(
    prompt_list: List[str],
    report_list: List[str],
    tokenizer: PreTrainedTokenizerBase,
    max_len: int,
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes a batch of reports with prompt-guided teacher forcing logic.
    Combines prompts and reports into a single sequence for tokenization,
    adding BOS, SEP (after prompt), and EOS tokens. Prepares inputs and labels
    suitable for autoregressive language model training.
    Args:
        prompt_list: List of prompt strings for the batch.
        report_list: List of full report strings for the batch.
        tokenizer: The tokenizer instance.
        max_len: Maximum sequence length for tokenization (excluding special tokens
                 added before truncation).
    Returns:
        A dictionary containing tokenized tensors:
        - 'decoder_input_ids': Input IDs for the decoder (shifted).
        - 'decoder_attention_mask': Attention mask for the decoder inputs.
        - 'label_ids': Target label IDs for loss calculation.
    Reference:
        Inspired by https://huggingface.co/aehrc/cxrmate-rrg24/blob/main/modelling_cxrrg.py#L346
    """
    
    # Format text inputs: BOS + prompt + SEP + report + EOS
    text_inputs = [
        f'{tokenizer.bos_token}{p}{tokenizer.sep_token}{r}{tokenizer.eos_token}'
        for p, r in zip(prompt_list, report_list)
    ]    

    # Tokenize reports
    tokenized = tokenizer(
        text_inputs,
        padding='longest',
        truncation=True,
        max_length=max_len + 1,  # +1 allows space for the label shift
        return_tensors='pt',
        return_token_type_ids=False,
        add_special_tokens=False, # We added them manually
    )

    # Obtain tensors for labels and decoder inputs
    label_ids = tokenized['input_ids'].detach().clone()
    decoder_input_ids = tokenized['input_ids']
    decoder_attention_mask = tokenized['attention_mask']

    # Find the position of the separator token in each sequence
    sep_token_positions = (
        (decoder_input_ids == tokenizer.sep_token_id).int().argmax(dim=1)
    )

    # Mask label_ids up to and including the separator token by setting it to pad_token_id
    for i, pos in enumerate(sep_token_positions):
        label_ids[i, : pos + 1] = tokenizer.pad_token_id

    # Apply shifting logic:

    # Create labels (target) by shifting input_ids right (remove BOS)
    # Shape: [batch_size, seq_len]
    label_ids = label_ids[:, 1:]

    # Create decoder_input_ids (input) by removing the last token (often EOS)
    # Shape: [batch_size, seq_len]
    decoder_input_ids = decoder_input_ids[:, :-1]

    # Create decoder_attention_mask by shifting the original mask right
    # Shape: [batch_size, seq_len]
    decoder_attention_mask = decoder_attention_mask[:, 1:]

    batch_dict = {
        'decoder_input_ids': decoder_input_ids,
        'decoder_attention_mask': decoder_attention_mask,
        'label_ids': label_ids,
    }

    return batch_dict



# --- Add other tokenization functions for different models/tasks ---


# --- Factory for Batch Tokenizer ---

def get_batch_tokenizer_strategy(
    tokenization_config: Dict[str, Any], # Renamed for clarity
    tokenizer: PreTrainedTokenizerBase
) -> Optional[BatchTokenizerFn]: # Return optional, None can mean default/no tokenization needed here
    """
    Selects and prepares a batch tokenization function based on model identifier.

    Args:
        tokenization_config: Configuration dictionary containing tokenization-specific
                            settings like 'strategy_name' and 'max_length'.
        tokenizer: The instantiated PreTrainedTokenizerBase.

    Returns:
        A callable batch tokenizer function (BatchTokenizerFn) prepared with
        necessary arguments, or None if no specific strategy is found/needed.

    Raises:
        KeyError: If 'max_length' is missing in tokenizer_config for a strategy
                  that requires it.
        ValueError: If the model_identifier requires a strategy but is not supported.
    """
    try:
        strategy_name = tokenization_config["strategy_name"]
    except KeyError:
        raise KeyError(
            "'strategy_name' missing in tokenization_config. "
            "Please provide a valid strategy name."
        )
    logger.info(f"Using tokenization strategy: '{strategy_name}'")

    # --- Strategy Selection ---
    if strategy_name == "cxrmate_rrg24_full_report":
        try:
            max_len = tokenization_config["max_length"]
            logger.info(f"Selected tokenizer strategy: '{strategy_name}' with max_length={max_len}")
            return partial(
                _cxrmate_rrg24_tokenize_batch__full_report,
                tokenizer=tokenizer,
                max_len=max_len,
            )
        except KeyError:
            logger.error(f"'max_length' not found in tokenizer_config for strategy '{strategy_name}'."
                         f" Config keys: {list(tokenization_config.keys())}")
            raise KeyError("'max_length' missing in tokenizer_config")
        
    elif strategy_name == "cxrmate_rrg24_prompt_guided_report":
        try:
            max_len = tokenization_config["max_length"]
            logger.info(f"Selected tokenizer strategy: '{strategy_name}' with max_length={max_len}")
            return partial(
                _cxrmate_rrg24_tokenize_batch__prompt_guided_report,
                tokenizer=tokenizer,
                max_len=max_len,
            )
        except KeyError:
            logger.error(f"'max_length' not found in tokenizer_config for strategy '{strategy_name}'."
                         f" Config keys: {list(tokenization_config.keys())}")
            raise KeyError("'max_length' missing in tokenizer_config")

    # --- Add other model identifiers and their strategies here ---
    # elif "some-other-model" in model_identifier.lower():
    #     # ... implement and return partial function for that model ...
    #     pass

    else:
        raise ValueError(
            f"Tokenization strategy '{strategy_name}' is not supported. "
            "Please provide a valid strategy name."
        )


# =============================================================================
# 2. Forward Pass & Loss Computation Strategies (for Training Loop)
# =============================================================================

def _compute_cxrmate_rrg24_loss(
    model: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    tokenizer: PreTrainedTokenizerBase, # Keep tokenizer arg for consistency
) -> torch.Tensor:
    """Computes loss for cxrmate-rrg24-style models."""
    # Assumes batch already contains tokenized data from the collate_fn
    # and is on the correct device.
    images = batch["pixel_values"]
    decoder_input_ids = batch["decoder_input_ids"]
    decoder_attention_mask = batch["decoder_attention_mask"]
    label_ids = batch["label_ids"]

    # Get token type ids needed by this specific model's forward
    decoder_token_type_ids = model.token_ids_to_token_type_ids(decoder_input_ids).to(device)

    # Run the forward pass without labels.
    outputs = model(
        pixel_values=images,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        decoder_token_type_ids=decoder_token_type_ids,
        labels=None, # Compute loss externally
        return_dict=True # Ensure Seq2SeqLMOutput is returned
    )

    # Slice out the logits corresponding only to the text (decoder) tokens
    # Shape: (batch_size, decoder_seq_len, vocab_size)
    total_seq_len = outputs.logits.shape[1]
    decoder_seq_len = decoder_input_ids.shape[1]
    encoder_seq_len = total_seq_len - decoder_seq_len
    logits_text = outputs.logits[:, encoder_seq_len:, :]

    # Compute the loss
    loss = F.cross_entropy(
        logits_text.reshape(-1, logits_text.shape[-1]), # (batch*seq_len, vocab_size)
        label_ids.reshape(-1), # (batch*seq_len)
        ignore_index=tokenizer.pad_token_id
    )
    return loss


# --- Add other loss computation functions ---

def _compute_default_loss(
    model: nn.Module,
    batch: Dict[str, Any],
    device: torch.device, # Keep device arg for consistency
    tokenizer: PreTrainedTokenizerBase, # Keep tokenizer arg for consistency
) -> torch.Tensor:
    """Default strategy: assumes model forward returns dict with 'loss' key."""
    if not isinstance(batch, dict):
        raise TypeError("Default loss computation requires batch to be a dictionary.")
    outputs = model(**batch)
    if not isinstance(outputs, dict) or 'loss' not in outputs:
        raise TypeError(
            "Default loss computation requires model output dict with 'loss' key."
        )
    return outputs['loss']


# --- Factory for Loss Computation ---

_LOSS_STRATEGIES: Dict[str, LossComputationFn] = {
    "aehrc/cxrmate-rrg24": _compute_cxrmate_rrg24_loss,
    "default": _compute_default_loss, # Explicitly add the default
    # Add other strategies here
}

def get_loss_computation_strategy(strategy_name: str) -> LossComputationFn:
    """Selects the appropriate loss computation function."""
    strategy_fn = _LOSS_STRATEGIES[strategy_name.lower()]
    return strategy_fn