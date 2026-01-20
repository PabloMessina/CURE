import hashlib
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from radgraph import RadGraph
from radgraph.rewards import (
    exact_entity_token_if_all_match_reward,
    exact_entity_token_if_rel_exists_reward,
    exact_entity_token_match_reward,
)
from tqdm import tqdm

from vlm_research_kit.settings import RADGRAPH_CACHE_DIR

# Logger setup
logger = logging.getLogger(__name__)


def hash_string(s: str) -> str:
    """Creates a unique and file-safe hash for a given string."""
    return f'{len(s)}_{hashlib.sha256(s.encode("utf-8")).hexdigest()}'


class RadGraphScorer:
    """
    Computes RadGraph-based rewards between generated and reference reports.

    This class encapsulates the RadGraph model to extract structured information
    (entities and relations) from reports. It uses a caching mechanism to avoid
    re-processing the same reports, improving efficiency.

    The primary method, `compute`, calculates rewards at three levels:
    - simple: Based on exact entity matches.
    - partial: Rewards entity matches if a relation also exists.
    - complete: Rewards entity matches only if the full relation (subject,
                relation, object) matches.
    """

    def __init__(
        self,
        radgraph_model_name: str = "modern-radgraph-xl",
        device: Optional[Union[str, torch.device]] = None,
        use_cache: bool = True,
        cache_dir: str = RADGRAPH_CACHE_DIR,
        verbose: bool = False,
    ):
        """
        Initializes the RadGraphScorer.

        Args:
            radgraph_model_name: The name of the RadGraph model to use.
            use_cache: If True, enables in-memory and disk caching for
                       RadGraph labels.
            cache_dir: Directory to store the cache file if use_cache is True.
            verbose: If True, enables verbose logging and progress bars.
        """
        self.verbose = verbose
        if self.verbose:
            # Set logger to INFO to display progress messages
            logger.setLevel(logging.INFO)
        else:
            # Otherwise, only show warnings and errors
            logger.setLevel(logging.WARNING)

        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(self.cache_dir, "radgraph_labels.pkl")
        self.cache: Dict[str, dict] = {}

        if self.use_cache:
            self._load_cache()

        logger.info(
            f"Initializing RadGraphScorer with model: {radgraph_model_name}"
        )
        logger.info(
            f"Using cache: {self.use_cache}, cache file: {self.cache_path}, device: {device}"
        )

        # Determine the device to use
        if device is not None:
            device_str = str(device)
            if device_str == "cuda":
                cuda = 0         
            elif device_str.startswith("cuda"):
                cuda = int(device_str.split(":")[1])
            elif device_str == "cpu":
                cuda = -1
            else:
                raise ValueError(f"Invalid device: {device}")
        else:
            cuda = None
        logger.info(f"Using cuda: {cuda}")
        
        # Initialize the RadGraph model once
        self.radgraph_model = RadGraph(model=radgraph_model_name, cuda=cuda)
        logger.info(f"Using device: {self.radgraph_model.device}")

    def _load_cache(self):
        """Loads the RadGraph label cache from disk if it exists."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
                if self.verbose:
                    logger.info(
                        f"Loaded {len(self.cache)} RadGraph labels from {self.cache_path}."
                    )
            except (pickle.UnpicklingError, EOFError):
                logger.warning(
                    f"Could not load RadGraph labels from {self.cache_path}. "
                    "Starting with an empty cache."
                )

    def _save_cache_file(self):
        """
        Atomically saves the RadGraph label cache to a pickle file.

        Writes to a temporary file first, then renames it to the final
        destination to prevent corruption if the process is interrupted.
        """
        if not self.use_cache:
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        temp_path = self.cache_path + ".tmp"

        try:
            with open(temp_path, "wb") as f:
                pickle.dump(self.cache, f)
            # Atomically move the temporary file to the final location
            os.replace(temp_path, self.cache_path)
            if self.verbose:
                logger.info(
                    f"Saved {len(self.cache)} entries to {self.cache_path}."
                )
        except Exception as e:
            logger.error(f"Failed to save cache to {self.cache_path}: {e}")
            # Clean up the temporary file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def save_cache(self):
        """Saves the current in-memory cache to disk."""
        self._save_cache_file()

    def _get_labels_for_reports(
        self, reports: List[str]
    ) -> Dict[str, dict]:
        """
        Generates RadGraph labels for a list of reports, using a cache.

        Args:
            reports: A list of unique report strings.

        Returns:
            A dictionary mapping each report string to its RadGraph labels.
        """
        report_to_labels = {}
        uncached_reports_map = {}  # Maps report text to its hash
        uncached_reports_list = []

        # Identify which reports are not in the cache
        for report in reports:
            if not report:  # Handle empty strings
                report_to_labels[report] = {}
                continue

            report_hash = hash_string(report)
            if self.use_cache and report_hash in self.cache:
                report_to_labels[report] = self.cache[report_hash]
            else:
                if report not in uncached_reports_map:
                    uncached_reports_map[report] = report_hash
                    uncached_reports_list.append(report)

        # Process uncached reports in a single batch
        if uncached_reports_list:
            if self.verbose:
                logger.info(
                    f"Cache miss for {len(uncached_reports_list)} unique reports. "
                    "Invoking RadGraph model..."
                )

            # RadGraph returns dict where keys are string indices '0', '1',...
            annotations = self.radgraph_model(uncached_reports_list)

            for i, report in enumerate(uncached_reports_list):
                labels = annotations[str(i)]
                report_to_labels[report] = labels
                if self.use_cache:
                    report_hash = uncached_reports_map[report]
                    self.cache[report_hash] = labels

        elif self.verbose:
            logger.info("All reports found in cache.")

        return report_to_labels

    @staticmethod
    def _calculate_reward_for_pair(
        hyp_labels: dict, ref_labels: dict
    ) -> Tuple[float, float, float]:
        """
        Calculates simple, partial, and complete rewards for a single pair.
        """
        # Handle cases where a report is empty or fails to produce entities
        if (
            not hyp_labels
            or not hyp_labels.get("entities")
            or not ref_labels
            or not ref_labels.get("entities")
        ):
            return 0.0, 0.0, 0.0

        simple = exact_entity_token_match_reward(hyp_labels, ref_labels)
        partial = exact_entity_token_if_rel_exists_reward(
            hyp_labels, ref_labels
        )
        complete = exact_entity_token_if_all_match_reward(
            hyp_labels, ref_labels
        )

        return simple, partial, complete

    def compute(
        self, hyps: List[str], refs: List[str]
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Computes RadGraph rewards for lists of hypothesis and reference reports.

        Args:
            hyps: A list of hypothesis (generated) report strings.
            refs: A list of reference (ground-truth) report strings.

        Returns:
            A dictionary containing mean and per-pair scores for simple,
            partial, and complete rewards.
        """
        if len(hyps) != len(refs):
            raise ValueError(
                "The number of hypothesis and reference reports must be the same."
            )

        # 1. Gather all unique reports to process them efficiently
        unique_reports = sorted(list(set(hyps + refs)))

        # 2. Get RadGraph labels for all unique reports
        report_to_labels_map = self._get_labels_for_reports(unique_reports)

        # 3. Compute rewards for each pair
        per_pair_simple, per_pair_partial, per_pair_complete = [], [], []

        pair_iterator = (
            tqdm(
                zip(hyps, refs),
                total=len(hyps),
                desc="Computing RadGraph rewards",
            )
            if self.verbose
            else zip(hyps, refs)
        )

        for hyp_report, ref_report in pair_iterator:
            hyp_labels = report_to_labels_map[hyp_report]
            ref_labels = report_to_labels_map[ref_report]

            simple, partial, complete = self._calculate_reward_for_pair(
                hyp_labels, ref_labels
            )
            per_pair_simple.append(simple)
            per_pair_partial.append(partial)
            per_pair_complete.append(complete)

        per_pair_simple_np = np.array(per_pair_simple, dtype=np.float32)
        per_pair_partial_np = np.array(per_pair_partial, dtype=np.float32)
        per_pair_complete_np = np.array(per_pair_complete, dtype=np.float32)

        results = {
            "mean_simple_reward": np.mean(per_pair_simple_np).item(),
            "mean_partial_reward": np.mean(per_pair_partial_np).item(),
            "mean_complete_reward": np.mean(per_pair_complete_np).item(),
            "per_pair_simple_reward": per_pair_simple_np,
            "per_pair_partial_reward": per_pair_partial_np,
            "per_pair_complete_reward": per_pair_complete_np,
        }

        if self.verbose:
            logger.info("RadGraph reward calculation complete.")

        return results

    def __call__(
        self, hyps: List[str], refs: List[str]
    ) -> Dict[str, Union[float, np.ndarray]]:
        return self.compute(hyps=hyps, refs=refs)