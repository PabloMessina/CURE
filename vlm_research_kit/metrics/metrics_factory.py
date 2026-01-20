import logging
import re
from typing import Any, Dict, List, Union, Callable, Tuple
from functools import partial

import torch
import torch.nn as nn
from torchmetrics import Metric

# --- Import specific metric classes ---
from torchmetrics.text import BLEUScore, ROUGEScore
from tqdm import tqdm
from vlm_research_kit.metrics.biomedical.chexbert import CHEXBERT_SHORT_CLASS_NAMES, F1CheXbertBatch
from vlm_research_kit.metrics.biomedical.cxrfescore import CXRFEScore
from vlm_research_kit.metrics.biomedical.radgraph import RadGraphScorer
from vlm_research_kit.metrics.biomedical.ratescore import RaTEScoreScorer
# from .vqa.accuracy import VQAAccuracy # Example

logger = logging.getLogger(__name__)

# --- Metric Registry ---
_METRIC_REGISTRY: Dict[str, Union[type[Metric], type[nn.Module], type[CXRFEScore], type[RadGraphScorer], partial]] = {
    "BLEU-1": partial(BLEUScore, n_gram=1),
    "BLEU-2": partial(BLEUScore, n_gram=2),
    "BLEU-3": partial(BLEUScore, n_gram=3),
    "BLEU-4": partial(BLEUScore, n_gram=4),
    "ROUGE-1": partial(ROUGEScore, rouge_keys='rouge1'),
    "ROUGE-2": partial(ROUGEScore, rouge_keys='rouge2'),
    "ROUGE-L": partial(ROUGEScore, rouge_keys='rougeL'),
    "ROUGE-Lsum": partial(ROUGEScore, rouge_keys='rougeLsum'),
    "CheXbert": F1CheXbertBatch,   # This is nn.Module
    "CXRFEScore": CXRFEScore,
    "RadGraph": RadGraphScorer,
    "RaTEScore": RaTEScoreScorer,
    # "VQA_Accuracy": VQAAccuracy, # Example
}

_METRICS_TAKING_DEVICE_IN_INIT = ["CheXbert", "BERTScore", "CXRFEScore", "RadGraph", "RaTEScore"] # Add others if needed

# Define a type alias for the return type for clarity
MetricObject = Union[Metric, nn.Module, Callable]

SUPPORTED_REPORT_GENERATION_METRICS = [
    "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
    "ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum",
    "BERTScore", "CheXbert", "CXRFEScore", "RadGraph", "RaTEScore"
]

def initialize_metrics(
    metrics_config: Union[List[str], Dict[str, Dict[str, Any]]],
    device: Union[str, torch.device]
) -> Dict[str, MetricObject]:
    """
    Initializes metric objects (torchmetrics.Metric or compatible nn.Module)
    based on configuration.

    Parses the configuration, looks up metric classes/partials in the registry,
    instantiates them with specified arguments, and moves them to the target device.

    Args:
        metrics_config: Configuration for the metrics. Can be:
            - A list of metric names (e.g., ["BLEU-4", "CheXbert"])
            - A dictionary mapping metric names to specific keyword arguments
              (e.g., {"CheXbert": {"batch_size": 64}, "BLEU-4": {}})
        device: The target device (e.g., 'cuda', 'cpu', torch.device object)
                for the metric computations. Individual metrics might override
                this if 'device' is specified in their specific config dict.

    Returns:
        A dictionary mapping metric names to their instantiated and
        device-placed metric objects (Metric, nn.Module, or other Callable).
        Example: {'BLEU-4': <BLEUScore obj>, 'CheXbert': <F1CheXbertBatch obj>}

    Raises:
        ValueError: If an unsupported metric name is encountered.
        TypeError: If metric configuration format is invalid.
        Exception: Propagates errors during metric instantiation.

    Note:
        Metrics inheriting from nn.Module (like CheXbert) might not conform to
        the standard torchmetrics interface (update/compute/reset) and may
        require special handling in the evaluation loop.
    """
    logger.info(f"Initializing metrics for device: {device}")
    initialized_metrics: Dict[str, MetricObject] = {} # Use the type alias

    # Determine the format of the input config
    if isinstance(metrics_config, list):
        metric_items = {name: {} for name in metrics_config}
        logger.info(f"Initializing metrics from list: {metrics_config}")
    elif isinstance(metrics_config, dict):
        metric_items = metrics_config
        logger.info(f"Initializing metrics from dict: {list(metrics_config.keys())}")
    else:
        raise TypeError(
            "Metric configuration must be a list of metric names or a "
            f"dictionary mapping names to config dicts. Got: {type(metrics_config)}"
        )

    # Iterate through the metrics to be initialized
    for metric_name, specific_kwargs in metric_items.items():
        logger.debug(f"Processing metric: '{metric_name}' with config: {specific_kwargs}")

        if not isinstance(specific_kwargs, dict):
            raise TypeError(
                f"Specific config for metric '{metric_name}' must be a dictionary. "
                f"Got: {type(specific_kwargs)}"
            )

        # --- Lookup Metric Class/Partial ---
        metric_builder = _METRIC_REGISTRY.get(metric_name)
        if metric_builder is None:
            logger.error(f"Unsupported metric name: '{metric_name}'")
            raise ValueError(
                f"Metric '{metric_name}' is not supported. "
                f"Supported metrics are: {list(_METRIC_REGISTRY.keys())}"
            )

        # --- Determine Effective Device ---
        effective_device = specific_kwargs.get("device", device)
        if isinstance(effective_device, str):
            effective_device = torch.device(effective_device)

        # --- Instantiate Metric ---
        try:
            final_kwargs = specific_kwargs.copy()
            
            # Ensure device kwarg is passed to custom metrics that expect it
            metric_takes_device_in_init = metric_name in _METRICS_TAKING_DEVICE_IN_INIT
            if metric_takes_device_in_init:
                if 'device' not in final_kwargs:
                    final_kwargs['device'] = effective_device
                # We assume __init__ handles device placement internally

            logger.debug(f"Instantiating '{metric_name}' with kwargs: {final_kwargs}")
            metric_instance = metric_builder(**final_kwargs)

            # --- Move to Device (if not handled by init and has .to method) ---
            # This applies to standard torchmetrics.Metric and also nn.Module instances
            # that didn't explicitly take 'device' in their init args we passed.
            if not metric_takes_device_in_init:
                if hasattr(metric_instance, 'to') and callable(metric_instance.to):
                    metric_instance = metric_instance.to(effective_device)
                    logger.debug(f"Moved '{metric_name}' to device: {effective_device} using .to()")
                else:
                    # This case should be rare for Metric or nn.Module
                    logger.warning(
                        f"Metric '{metric_name}' instance does not have a 'to' method "
                        f"and didn't seem to take 'device' in init. "
                        f"Assuming it handles device internally or doesn't need it."
                    )
            # If it took device in init, we trust it handled placement.
            elif isinstance(metric_instance, nn.Module):
                logger.debug(f"Metric '{metric_name}' is nn.Module and took 'device' in init. "
                             "Assuming device handled internally.")

            # Store in the flat dictionary
            initialized_metrics[metric_name] = metric_instance
            logger.info(f"Successfully initialized metric: '{metric_name}' (type: {type(metric_instance).__name__}) on device '{effective_device}'")

        except Exception as e:
            logger.error(
                f"Failed to instantiate or move metric '{metric_name}' with "
                f"kwargs {final_kwargs}. Error: {e}",
                exc_info=True
            )
            raise

    logger.info("Finished initializing all metrics.")
    return initialized_metrics


# This is a simple regex to remove bounding box coordinates
# from the text. It assumes the coordinates are in the format
# [(x1,y1,x2,y2),(x3,y3,x4,y4),...]
_grounding_pattern = r"\s*\[\(\d+\.\d+\s*(,\s*\d+\.\d+\s*)*\)\s*(,\s*\(\d+\.\d+\s*(,\s*\d+\.\d+\s*)*\)\s*)*\]"

def remove_grounding_from_text(text):
    """
    Remove grounding information from the text.
    E.g. 'Bilateral apical pleural thickening [(0.72,0.08,0.26,0.12),(0.41,0.19,0.17,0.10)]'
    -> 'Bilateral apical pleural thickening'

    Args:
        text: The input text string from which to remove grounding information.

    Returns:
        The modified text string with grounding information removed.
    """
    return re.sub(_grounding_pattern, '', text).strip()


def compute_report_generation_metrics(
    metrics: Dict[str, MetricObject],
    predictions: List[str],
    references: List[str],
    return_per_sample_metrics: bool = False,
    handle_grounded_reports: bool = False,
    verbose: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """
    Computes report generation metrics.

    Args:
        metrics: Dictionary of initialized metric objects.
        predictions: List of predicted strings (e.g., generated captions).
        references: List of reference strings (e.g., ground truth captions).
        return_per_sample_metrics: If True, also return metrics calculated
            for each individual sample (if supported by the metric).
            Defaults to False.
        handle_grounded_reports: If True, remove grounding information from
            the text before computing metrics. This is useful for
            report generation tasks where the grounding information
            is not relevant for the evaluation.
        verbose: If True, print detailed information during computation.

    Returns:
        If return_per_sample_metrics is False (default):
            A dictionary containing the aggregated computed metric values.
        If return_per_sample_metrics is True:
            A tuple containing:
            - Element 0: A dictionary with the aggregated computed metric values.
            - Element 1: A list of dictionaries, where each dictionary
              corresponds to a sample and contains the per-sample metric
              values for that sample. The list order matches the input
              predictions/references.
    """
    if verbose:
        logger.info("Computing report generation metrics.")
    overall_metrics = {}
    if return_per_sample_metrics:
        num_samples = len(predictions)
        per_sample_metrics = [{} for _ in range(num_samples)]

    if handle_grounded_reports:
        if verbose:
            logger.info("Removing grounding information from text.")
        predictions_with_grounding = predictions # Keep original for reference
        references_with_grounding = references # Keep original for reference 
        predictions = [remove_grounding_from_text(pred) for pred in predictions]
        references = [remove_grounding_from_text(ref) for ref in references]

    for metric_name, metric in metrics.items():

        if metric_name.startswith("BLEU"):
            if verbose:
                logger.info(f"Computing {metric_name} metric.")
            references_ = [[ref] for ref in references]  # BLEU expects a list of references per prediction
            overall_metrics[metric_name] = metric(predictions, references_).item()
            if return_per_sample_metrics:
                for i in tqdm(range(num_samples), desc=f"Computing {metric_name} per-sample metrics", mininterval=1):
                    per_sample_metrics[i][metric_name] = metric([predictions[i]], [references_[i]]).item()

        elif metric_name.startswith("ROUGE"):
            if verbose:
                logger.info(f"Computing {metric_name} metric.")
            rouge_scores = metric(predictions, references)
            # We will use "_fmeasure" to get the F1 score for ROUGE
            # E.g. results["ROUGE-L"] = rouge_scores['rougeL_fmeasure'].item()
            key = f'rouge{metric_name.split("-")[1]}_fmeasure'
            overall_metrics[metric_name] = rouge_scores[key].item()
            if return_per_sample_metrics:
                logger.warning(
                    "Per-sample metrics for ROUGE are not supported yet due to performance issues. "
                    "Returning overall metrics only."
                )
                pass # Avoid per-sample metrics for ROUGE for now because it's too slow
                # for i in tqdm(range(num_samples), desc=f"Computing {metric_name} per-sample metrics", mininterval=1):
                #     per_sample_metrics[i][metric_name] = metric(predictions[i], references[i])[key].item()

        elif metric_name == "BERTScore":
            if verbose:
                logger.info(f"Computing {metric_name} metric.")
            bert_scores = metric(predictions, references)
            f1s = bert_scores['f1']
            ps = bert_scores['precision']
            rs = bert_scores['recall']
            overall_metrics["BERTScore_F1"] = torch.mean(f1s).item()
            overall_metrics["BERTScore_P"] = torch.mean(ps).item()
            overall_metrics["BERTScore_R"] = torch.mean(rs).item()
            if return_per_sample_metrics:
                if verbose:
                    logger.info('Computing per-sample BERTScore metrics.')
                for i in range(num_samples):
                    per_sample_metrics[i]["BERTScore_F1"] = f1s[i].item()
                    per_sample_metrics[i]["BERTScore_P"] = ps[i].item()
                    per_sample_metrics[i]["BERTScore_R"] = rs[i].item()

        elif metric_name == "CheXbert":
            if verbose:
                logger.info(f"Computing {metric_name} metric.")
            chexbert_results = metric(hyps=predictions, refs=references)
            overall_metrics["CheXbert_accuracy"] = chexbert_results['accuracy']
            overall_metrics["CheXbert_classification_report"] = chexbert_results['classification_report']
            sim_results = metric.compute_cosine_similarity(hyps=predictions, refs=references)
            overall_metrics["CheXbert_cosine_similarity"] = sim_results['mean_similarity']
            if return_per_sample_metrics:
                if verbose:
                    logger.info('Computing per-sample CheXbert metrics.')
                accuracies = chexbert_results['per_element_accuracy']
                similarities = sim_results['per_pair_similarity']
                ref_labels = chexbert_results['ref_labels']
                hyp_labels = chexbert_results['hyp_labels']
                for i in range(num_samples):
                    per_sample_metrics[i]["CheXbert_accuracy"] = accuracies[i]
                    per_sample_metrics[i]["CheXbert_cosine_similarity"] = similarities[i]
                    for j, name in enumerate(CHEXBERT_SHORT_CLASS_NAMES):
                        per_sample_metrics[i][f"CheXbert_{name}_gt"] = ref_labels[i, j]
                        per_sample_metrics[i][f"CheXbert_{name}_pred"] = hyp_labels[i, j]

        elif metric_name == "CXRFEScore":
            if verbose:
                logger.info(f"Computing {metric_name} metric.")
            cxrfescore_results = metric(hyps=predictions, refs=references)
            overall_metrics["CXRFEScore"] = cxrfescore_results['mean_similarity']
            if return_per_sample_metrics:
                similarities = cxrfescore_results['per_pair_similarity']
                for i in range(num_samples):
                    per_sample_metrics[i]["CXRFEScore"] = similarities[i]

        elif metric_name == "RadGraph":
            if verbose:
                logger.info(f"Computing {metric_name} metric.")
            radgraph_results = metric(hyps=predictions, refs=references)
            overall_metrics["RadGraph_simple_reward"] = radgraph_results['mean_simple_reward']
            overall_metrics["RadGraph_partial_reward"] = radgraph_results['mean_partial_reward']
            overall_metrics["RadGraph_complete_reward"] = radgraph_results['mean_complete_reward']
            if return_per_sample_metrics:
                simple_rewards = radgraph_results['per_pair_simple_reward']
                partial_rewards = radgraph_results['per_pair_partial_reward']
                complete_rewards = radgraph_results['per_pair_complete_reward']
                for i in range(num_samples):
                    per_sample_metrics[i]["RadGraph_simple_reward"] = simple_rewards[i]
                    per_sample_metrics[i]["RadGraph_partial_reward"] = partial_rewards[i]
                    per_sample_metrics[i]["RadGraph_complete_reward"] = complete_rewards[i]

        elif metric_name == "RaTEScore":
            if verbose:
                logger.info(f"Computing {metric_name} metric.")
            ratescore_results = metric(hyps=predictions, refs=references)
            overall_metrics["RaTEScore"] = ratescore_results['mean_ratescore']
            if return_per_sample_metrics:
                ratescores = ratescore_results['per_pair_ratescore']
                for i in range(num_samples):
                    per_sample_metrics[i]["RaTEScore"] = ratescores[i].item()
        
        else:
            if verbose:
                logger.info(f"Computing {metric_name} metric.")
            # For other metrics, we assume they are callable and return a single value-
            overall_metrics[metric_name] = metric(predictions, references).item()
            # If return_per_sample_metrics is True, we assume they support per-sample metrics
            if return_per_sample_metrics:
                for i in tqdm(range(num_samples), desc=f"Computing {metric_name} per-sample metrics", mininterval=1):
                    per_sample_metrics[i][metric_name] = metric([predictions[i]], [references[i]]).item()
    
    if return_per_sample_metrics:
        if verbose:
            logger.info("Finished computing report generation metrics with per-sample results.")
        return overall_metrics, per_sample_metrics
    else:
        if verbose:
            logger.info("Finished computing report generation metrics.")
        return overall_metrics


def flatten_report_generation_metrics(
    results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Flattens the results of report generation metrics into a single dictionary.

    Args:
        results: Dictionary with computed metric values, possibly with nested dictionaries.

    Returns:
        A flattened dictionary with metric names and their corresponding values.
    """
    flattened_results = {}

    for metric_name, value in results.items():
        if isinstance(value, dict):
            # Handle special cases first:
            if metric_name == "CheXbert_classification_report":
                flattened_results['CheXbert_macro_f1'] = value['macro avg']['f1-score']
                flattened_results['CheXbert_micro_f1'] = value['micro avg']['f1-score']
                # Ignore other values for now
            else: # Default case for nested dictionaries
                for sub_metric_name, sub_value in value.items():
                    flattened_results[f"{metric_name}_{sub_metric_name}"] = sub_value
        else:
            flattened_results[metric_name] = value
    return flattened_results


def calculate_weighted_score(metric_values: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Calculate a weighted score based on metric values and their corresponding weights.

    Args:
        metric_values: Dictionary of metric names and their computed values.
        weights: Dictionary of metric names and their corresponding weights.

    Returns:
        A weighted score calculated as the sum of (metric value * weight) for each metric.
    """
    total_score = sum(metric_values[name] * weights[name] for name in weights)
    total_weight = sum(weights.values())
    if total_weight == 0:
        raise ValueError("Total weight cannot be zero.")
    weighted_score = total_score / total_weight
    return weighted_score