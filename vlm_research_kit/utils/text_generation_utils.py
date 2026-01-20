import torch
import transformers
import re
import json
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

def build_generation_kwargs(
    model_identifier: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    decoding_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Builds generation keyword arguments with defaults and model-specific adjustments.

    Args:
        model_identifier (str): The name or path of the model (e.g., 'aehrc/cxrmate-rrg24').
                                Used to apply model-specific defaults if known.
        tokenizer: The tokenizer associated with the model.
        decoding_config (Optional[Dict[str, Any]]): A dictionary of user-provided
                                                    generation parameters to override
                                                    defaults (e.g., {'max_length': 1024, 'num_beams': 5}).

    Returns:
        Dict[str, Any]: A dictionary of keyword arguments suitable for
                        model.generate().
    """
    if decoding_config is None:
        decoding_config = {}

    model_identifier = model_identifier.lower() # Normalize to lowercase for consistency

    # --- Default Generation Arguments ---
    # These can be common defaults suitable for many models
    generation_kwargs = {
        "max_length": 512,
        "num_beams": 4,
        "do_sample": False,
        # Add other common defaults like temperature, top_k, top_p if needed
        # "temperature": 1.0,
        # "top_k": 50,
    }
    logger.info(f"Base generation kwargs: {generation_kwargs}")

    # --- Model-Specific Adjustments ---
    # Add conditions based on model_identifier or potentially model.config attributes
    if 'cxrmate-rrg24' in model_identifier:
        logger.info(f"Applying specific kwargs for model: {model_identifier}")
        # Add bad_words_ids for this specific model
        nf_token_id = tokenizer.convert_tokens_to_ids('[NF]')
        ni_token_id = tokenizer.convert_tokens_to_ids('[NI]')
        generation_kwargs["bad_words_ids"] = [[nf_token_id], [ni_token_id]]
        # You might add other specific settings for this model here if needed

    # Add more elif conditions here for other models requiring specific defaults
    # elif 'some-other-model' in model_identifier:
    #    generation_kwargs['repetition_penalty'] = 1.2
    #    logger.info(f"Applied specific kwargs for {model_identifier}")
        
    elif model_identifier == 'default':
        pass # Default case, no specific adjustments
    else:
        raise ValueError(f"Unknown model identifier: {model_identifier}")

    # --- Apply User Overrides ---
    # deconding_config takes precedence over defaults and model-specific adjustments
    if decoding_config:
        logger.info(f"Applying user overrides: {decoding_config}")
        generation_kwargs.update(decoding_config)
        if "max_new_tokens" in generation_kwargs and "max_length" in generation_kwargs:
            # Remove max_length if max_new_tokens is specified
            logger.warning(
                f"Both 'max_new_tokens' ({generation_kwargs['max_new_tokens']}) "
                f"and 'max_length' ({generation_kwargs['max_length']}) are set. "
                "Using 'max_new_tokens' and removing 'max_length'."
            )
            generation_kwargs.pop("max_length")
    else:
        logger.info("No user overrides provided. Using default generation kwargs.")

    logger.info(f"Final generation kwargs: {generation_kwargs}")
    return generation_kwargs

def concatenate_report_sections(findings: Optional[str], impression: Optional[str]) -> str:
    """
    Concatenate findings and impression sections into a single report string.
    Ensures proper punctuation and spacing.
    Args:
        findings (Optional[str]): The findings section of the report.
        impression (Optional[str]): The impression section of the report.
    Returns:
        str: The concatenated report string.
    """
    findings = findings.strip() if findings else ''
    impression = impression.strip() if impression else ''
    if findings and impression:
        if findings.endswith('.'):
            report = f'{findings} {impression}'
        else:
            report = f'{findings}. {impression}'
    elif findings:
        report = findings
    elif impression:
        report = impression
    else:
        report = ''
    return report

def _apply_cxrmate_rrg24_report_cleaning(texts: List[str]) -> List[str]:
    """
    Apply specific cleaning operations to the generated texts for cxrmate-rrg24 model.
    This includes removing unwanted tokens and ensuring proper formatting.
    Concretely, we remove [BOS] and [EOS], and replace [SEP] with a space.

    Args:
        texts (List[str]): The list of generated report strings.

    Returns:
        List[str]: The cleaned list of report strings.
    """
    cleaned_texts = []
    for text in texts:
        # Remove unwanted tokens and clean the text
        text = text.replace('[BOS]', '').replace('[EOS]', '').replace('[SEP]', ' ')
        # Strip leading/trailing whitespace
        cleaned_text = text.strip()
        cleaned_texts.append(cleaned_text)
    return cleaned_texts

@torch.no_grad()
def generate_and_decode_reports(
    model_identifier: str,
    model: transformers.PreTrainedModel,
    pixel_values: torch.Tensor,
    generation_kwargs: Dict[str, Any],
    tokenizer: transformers.PreTrainedTokenizerBase,
    prompts: Optional[List[str]] = None,  # <-- New optional argument
    skip_special_tokens: bool = True,
    clean_up_tokenization_spaces: bool = True,
    run_model_eval: bool = True,
) -> List[str]:
    """
    Generates text for a batch using model.generate and decodes the output.

    Args:
        model_identifier (str): The name or path of the model (e.g., 'aehrc/cxrmate-rrg24').
        model: The model to use for generation.
        pixel_values: Batch of preprocessed image tensors (on the correct device).
        generation_kwargs: Keyword arguments passed directly to model.generate().
                             These should be pre-configured (e.g., using
                             build_generation_kwargs).
        tokenizer: The tokenizer corresponding to the model.
        prompts (Optional[List[str]]): A batch of text prompts. If provided,
                                       generation is conditioned on these prompts.
        skip_special_tokens (bool): Whether to skip special tokens during decoding.
                                    Set to False if you need to see EOS, BOS, etc.
                                    or if special tokens are part of the expected output.
        clean_up_tokenization_spaces (bool): Whether to clean up tokenization spaces
                                             during decoding.
        run_model_eval (bool): Whether to set the model to evaluation mode.

    Returns:
        List[str]: A list of generated and decoded report strings.
    """
    model_identifier = model_identifier.lower() # Normalize to lowercase for consistency
    if run_model_eval:
        model.eval() # Ensure model is in eval mode

    if prompts:
        # --- Prompted Generation ---
        # 1. Format prompts with BOS and SEP tokens as required by the model
        formatted_prompts = [
            f"{tokenizer.bos_token}{p}{tokenizer.sep_token}" for p in prompts
        ]

        # 2. Tokenize prompts with padding
        prompt_inputs = tokenizer(
            formatted_prompts, return_tensors="pt", padding=True, padding_side="left",
        )
        prompt_input_ids = prompt_inputs.input_ids.to(pixel_values.device)
        # prompt_attention_mask = prompt_inputs.attention_mask.to(pixel_values.device)        

        # 3. Adjust generation kwargs for prompted generation
        current_gen_kwargs = generation_kwargs.copy()
        # Adjust max_length to account for the prompt's length
        prompt_len = prompt_input_ids.shape[1]
        if "max_length" in current_gen_kwargs:
            current_gen_kwargs["max_length"] += prompt_len
        if "max_new_tokens" not in current_gen_kwargs:
            logger.warning(
                "'max_new_tokens' not in generation_kwargs. "
                "Using 'max_length' is less reliable for prompted generation."
            )

         # 4. Generate text conditioned on both image and prompt
        output_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=prompt_input_ids,
            # attention_mask=prompt_attention_mask,
            **current_gen_kwargs,
        )

        # 5. Decode only the newly generated tokens (after the prompt)
        newly_generated_ids = output_ids[:, prompt_len:]
        generated_texts = tokenizer.batch_decode(
            newly_generated_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    else:
        # --- Unprompted (Image-only) Generation ---
        output_ids = model.generate(
            pixel_values=pixel_values,
            **generation_kwargs,
        )

        # Decode the full sequence of generated tokens
        if 'cxrmate-rrg24' in model_identifier: # Specific handling for cxrmate-rrg24
            # generated_findings_list, generated_impression_list = model.split_and_decode_sections(
            #     output_ids, tokenizer,
            # )
            # # Concatenate findings and impression sections
            # generated_texts = [
            #     concatenate_report_sections(findings, impression)
            #     for findings, impression in zip(generated_findings_list, generated_impression_list)
            # ]
            generated_texts = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            generated_texts = _apply_cxrmate_rrg24_report_cleaning(generated_texts)
        elif model_identifier == 'default': # Default case
            # General decoding for other models
            generated_texts = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
        else:
            raise ValueError(f"Unknown model identifier: {model_identifier}")

    return generated_texts


_COMMA_SEPARATED_LIST_REGEX = re.compile(r'\[\s*(\".+?\"(\s*,\s*\".+?\")*)?\s*\]?')

def parse_facts(txt):
    facts_str = _COMMA_SEPARATED_LIST_REGEX.search(txt).group()
    if facts_str[-1] != ']': facts_str += ']'
    facts = json.loads(facts_str)
    seen = set()
    clean_facts = []
    for fact in facts:
        if fact not in seen:
            clean_facts.append(fact)
            seen.add(fact)
    return clean_facts


def _substrings_are_equal(text, i, j, k):
    for x in range(k):
        if text[i+x] != text[j+x]:
            return False
    return True


def remove_consecutive_repeated_words_from_text(text, ks=[1, 2, 3, 4, 5, 6, 7, 8]):
    # Sanity checks
    assert type(ks) is int or type(ks) is list
    if type(ks) is int:
        ks = [ks]
    else:
        assert len(ks) > 0
        assert all(type(x) is int for x in ks)

    tokens = text.split()
    lower_tokens = text.lower().split()
    dedup_tokens = []
    dedup_lower_tokens = []

    for k in ks:
        for i in range(len(lower_tokens)):
            # if current word is part of a k-word phrase that is repeated -> skip
            skip = False
            for j in range(k):
                s = i - j # start index
                e = s + k-1 # end index
                if s - k >= 0 and e < len(lower_tokens) and _substrings_are_equal(lower_tokens, s, s-k, k):
                    skip = True
                    break
            if skip:
                continue
            dedup_tokens.append(tokens[i])
            dedup_lower_tokens.append(lower_tokens[i])
        tokens = dedup_tokens
        lower_tokens = dedup_lower_tokens
        dedup_tokens = []
        dedup_lower_tokens = []
    return ' '.join(tokens)