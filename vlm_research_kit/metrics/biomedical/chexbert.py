import logging
import os
import pickle
import re  # For cleaning text in batch method
import textwrap
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

# Import the original class from the installed library
from f1chexbert import F1CheXbert
from matplotlib.patches import Rectangle
from nltk.tokenize import sent_tokenize  # For sentence splitting

# Import necessary functions for metric calculation
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity  # For cosine similarity
from tqdm import tqdm  # For progress bar

from vlm_research_kit.nltk_setup import ensure_nltk_resources
from vlm_research_kit.settings import CHEXBERT_CACHE_DIR

# Ensure the NLTK punkt tokenizer is downloaded
ensure_nltk_resources()

logger = logging.getLogger(__name__)

def merge_labels(labels_list):        
    merged = np.zeros((14,), np.int8)
    merged[-1] = 1 # default to no findings
    for labels in labels_list:
        if labels[-1] == 0: # there is a finding
            merged[-1] = 0
        for i in range(0, len(labels)-1): # iterate over all labels except the last one
            if labels[i] == 1:
                merged[i] = 1
    return merged

CHEXBERT_CLASS_NAMES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices", "No Finding"
]

CHEXBERT_CLASS_NAME_TO_SHORT = {
    "Enlarged Cardiomediastinum": "EC",
    "Cardiomegaly": "CM",
    "Lung Opacity": "LO",
    "Lung Lesion": "LL",
    "Edema": "E",
    "Consolidation": "C",
    "Pneumonia": "P",
    "Atelectasis": "A",
    "Pneumothorax": "PT",
    "Pleural Effusion": "PE",
    "Pleural Other": "PO",
    "Fracture": "F",
    "Support Devices": "SD",
    "No Finding": "NF",
}

CHEXBERT_SHORT_TO_CLASS_NAME = {v: k for k, v in CHEXBERT_CLASS_NAME_TO_SHORT.items()}

CHEXBERT_SHORT_CLASS_NAMES = [CHEXBERT_CLASS_NAME_TO_SHORT[name] for name in CHEXBERT_CLASS_NAMES]

assert len(CHEXBERT_CLASS_NAMES) == len(CHEXBERT_CLASS_NAME_TO_SHORT)
assert len(CHEXBERT_CLASS_NAMES) == len(CHEXBERT_SHORT_TO_CLASS_NAME)

class F1CheXbertBatch(F1CheXbert):
    """
    An enhanced version of F1CheXbert that processes reports in batches
    for significantly faster inference, especially on GPU.

    Inherits from the original F1CheXbert, reusing its model loading,
    tokenizer, and basic structure. Overrides the forward method to
    implement batching.
    """
    def __init__(
        self,
        default_batch_size: int = 32,
        verbose: bool = False,
        use_cache: bool = True,
        cache_dir: str = CHEXBERT_CACHE_DIR,
        **kwargs
    ):
        """
        Initializes the F1CheXbertBatch instance.

        Args:
            default_batch_size: The default batch size to use if not specified
                                in the forward call. Defaults to 32.
            verbose: If True, enables verbose logging. Defaults to False.
            use_cache: If True, enables in-memory and disk caching for sentences,
                       labels, and embeddings. Defaults to True.
            cache_dir: Directory to store cache files if use_cache is True.
                       Defaults to CHEXBERT_CACHE_DIR.
            **kwargs: Arguments to pass to the parent F1CheXbert constructor
                      (e.g., refs_filename, hyps_filename, device).
        """
        if use_cache and cache_dir is None:
            raise ValueError("cache_dir must be provided if use_cache is True.")

        # Initialize the parent class (loads model, tokenizer, etc.)
        super().__init__(**kwargs)
        self.default_batch_size = default_batch_size
        self.verbose = verbose
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.embedding_dimension = self.model.bert.config.hidden_size # 768

        # --- Initialize Caches ---
        self.sentence_to_labels_cache = {}
        self.sentence_to_embedding_cache = {}
        if self.use_cache:
            self._load_cache()

        self.model.eval() # Ensure model is in eval mode

        if self.verbose:
            logger.info(f"F1CheXbertBatch initialized. Using device: {self.device}")
            logger.info(f"Default batch size: {self.default_batch_size}")
            logger.info(f"Using cache: {self.use_cache}, Cache directory: {self.cache_dir}")

    def _load_cache(self):
        """Loads sentence-to-label and sentence-to-embedding caches from disk."""
        labels_cache_path = os.path.join(self.cache_dir, "sentence_to_labels.pkl")
        embed_cache_path = os.path.join(self.cache_dir, "sentence_to_embedding.pkl")

        if os.path.exists(labels_cache_path):
            try:
                with open(labels_cache_path, "rb") as f:
                    self.sentence_to_labels_cache = pickle.load(f)
                if self.verbose:
                    logger.info(f"Loaded {len(self.sentence_to_labels_cache)} sentence-to-label mappings from {labels_cache_path}.")
            except (pickle.UnpicklingError, EOFError):
                logger.warning(f"Could not load label cache from {labels_cache_path}. Starting with an empty cache.")

        if os.path.exists(embed_cache_path):
            try:
                with open(embed_cache_path, "rb") as f:
                    self.sentence_to_embedding_cache = pickle.load(f)
                if self.verbose:
                    logger.info(f"Loaded {len(self.sentence_to_embedding_cache)} sentence embeddings from {embed_cache_path}.")
            except (pickle.UnpicklingError, EOFError):
                logger.warning(f"Could not load embedding cache from {embed_cache_path}. Starting with an empty cache.")

    def _save_cache_file(self, data: dict, filename: str):
        """
        Atomically saves a cache dictionary to a pickle file.

        Writes to a temporary file first, then renames it to the final
        destination to prevent corruption if the process is interrupted.

        Args:
            data: The dictionary object to save.
            filename: The target filename (e.g., "sent_to_facts.pkl").
        """
        if not self.use_cache:
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        
        final_path = os.path.join(self.cache_dir, filename)
        temp_path = final_path + ".tmp"

        try:
            with open(temp_path, "wb") as f:
                pickle.dump(data, f)
            # Atomically move the temporary file to the final location
            os.replace(temp_path, final_path)
            if self.verbose:
                logger.info(
                    f"Saved {len(data)} entries to {final_path}."
                )
        except Exception as e:
            logger.error(f"Failed to save cache to {final_path}: {e}")
            # Clean up the temporary file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _save_sentence_to_labels_cache(self):
        """Saves the sentence-to-label cache to disk."""
        self._save_cache_file(
            self.sentence_to_labels_cache, "sentence_to_labels.pkl"
        )

    def _save_sentence_to_embedding_cache(self):
        """Saves the sentence-to-embedding cache to disk."""
        self._save_cache_file(
            self.sentence_to_embedding_cache, "sentence_to_embedding.pkl"
        )

    def save_cache(self):
        """Saves the current in-memory caches to disk."""
        self._save_sentence_to_labels_cache()
        self._save_sentence_to_embedding_cache()

    def get_labels_batch(self,
                         reports: List[str],
                         batch_size: Optional[int] = None,
                         mode="rrg") -> List[List[Union[int, str]]]:
        """
        Processes a list of reports in batches to generate CheXbert labels.

        Args:
            reports: A list of report strings.
            batch_size: The number of reports to process in each batch.
                        Uses instance default if None.
            mode: The labeling mode ('rrg' or 'classification'). Defaults to 'rrg'.

        Returns:
            A list of lists, where each inner list contains the 14 labels
            for the corresponding input report.
        """
        if batch_size is None:
            batch_size = self.default_batch_size
        if not reports:
            return []

        if self.use_cache:
            # Use list of Nones as placeholder for easy indexing
            cached_labels = [self.sentence_to_labels_cache.get(r, None) for r in reports]
            missing_indices = [i for i, label in enumerate(cached_labels) if label is None]

            if not missing_indices:
                if self.verbose:
                    logger.info(f"All {len(reports)} reports/sentences found in label cache.")
                return cached_labels

            if self.verbose:
                logger.info(f"Label cache miss for {len(missing_indices)}/{len(reports)} reports/sentences.")
            
            # Process only the missing reports
            reports_to_process = [reports[i] for i in missing_indices]
        else:
            reports_to_process = reports

        new_labels = []

        self.model.eval()  # Ensure model is in eval mode

        if self.verbose:
            iterator = tqdm(range(0, len(reports_to_process), batch_size), desc="Processing batches")
        else:
            iterator = range(0, len(reports_to_process), batch_size)

        with torch.no_grad(): # Disable gradient calculations for inference
            for i in iterator:
                batch_reports = reports_to_process[i : i + batch_size]

                # Clean reports (simple cleaning)
                cleaned_batch = []
                for report in batch_reports:
                    if not isinstance(report, str):
                        report = str(report) # Handle non-strings
                    report = report.strip().replace('\n', ' ')
                    report = re.sub(r'\s+', ' ', report).strip() # Replace multiple spaces
                    cleaned_batch.append(report if report else " ") # Handle empty strings

                # Tokenize batch using the tokenizer from the parent class
                inputs = self.tokenizer(
                    cleaned_batch,
                    padding="longest",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                    return_attention_mask=True,
                )

                # Move tensors to the device specified in the parent class
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)

                # Get model outputs using the model from the parent class
                batch_outputs = self.model(input_ids, attention_mask=attention_mask)

                # Process outputs for the batch
                num_reports_in_batch = input_ids.shape[0]
                batch_results = [[] for _ in range(num_reports_in_batch)]

                for task_output in batch_outputs: # Iterate over each task output (14 classes)
                    predictions = task_output.argmax(dim=1) # shape: (num_reports_in_batch, num_classes)
                    for report_idx in range(num_reports_in_batch):
                        pred_class = predictions[report_idx].item()
                        if mode == "rrg":
                            label = 1 if pred_class == 1 or pred_class == 3 else 0
                            batch_results[report_idx].append(label)
                        elif mode == "classification":
                            # Map to -1, 0, 1, ''
                            if pred_class == 0:
                                label = ''
                            elif pred_class == 1:
                                label = 1
                            elif pred_class == 2:
                                label = 0
                            elif pred_class == 3:
                                label = -1
                            else:
                                label = ''
                            batch_results[report_idx].append(label)
                        else:
                            raise NotImplementedError(f"Mode '{mode}' not implemented.")
                new_labels.extend(batch_results)

        if self.use_cache:
            # Fill in the missing labels in our original list
            for i, orig_idx in enumerate(missing_indices):
                cached_labels[orig_idx] = new_labels[i]
            assert None not in cached_labels, "None found in cached labels after filling missing ones."
            # Update the cache dictionary and save
            self.sentence_to_labels_cache.update(dict(zip(reports_to_process, new_labels)))
            return cached_labels
        else:
            return new_labels
    
    def get_embeddings_batch(
        self, texts: List[str], batch_size: int = None
    ) -> np.ndarray:
        """
        Processes a list of text strings in batches to generate their embeddings
        using the BERT encoder from the CheXbert model. The [CLS] token
        embedding is used as the sentence embedding.

        Args:
            texts: A list of text strings.
            batch_size: The number of texts to process in each batch.
                        Uses instance default if None.

        Returns:
            A numpy array of embeddings, where each row corresponds to the
            embedding of the input text string.
        """
        if batch_size is None:
            batch_size = self.default_batch_size
        if not texts:
            return np.array([])

        if self.use_cache:
            text_embeddings = np.empty((len(texts), self.embedding_dimension), dtype=np.float32)
            missing_indices = []
            for i, text in enumerate(texts):
                embedding = self.sentence_to_embedding_cache.get(text, None)
                if embedding is None:
                    missing_indices.append(i)
                else:
                    text_embeddings[i] = embedding
            if not missing_indices:
                if self.verbose:
                    logger.info(f"All {len(texts)} texts found in embedding cache.")
                return text_embeddings
            if self.verbose:
                logger.info(f"Embedding cache miss for {len(missing_indices)}/{len(texts)} texts.")
            texts_to_process = [texts[i] for i in missing_indices]
        else:
            texts_to_process = texts


        new_embeddings = []
        self.model.eval()  # Ensure model is in eval mode

        if self.verbose:
            iterator = tqdm(range(0, len(texts_to_process), batch_size), desc="Generating embeddings")
        else:
            iterator = range(0, len(texts_to_process), batch_size)

        with torch.no_grad():  # Disable gradient calculations for inference
            for i in iterator:
                batch_texts = texts_to_process[i : i + batch_size]

                # Clean texts (simple cleaning)
                cleaned_batch = []
                for text in batch_texts:
                    if not isinstance(text, str):
                        text = str(text)  # Handle non-strings
                    text = text.strip().replace("\n", " ")
                    text = re.sub(r"\s+", " ", text).strip()  # Replace multiple spaces
                    cleaned_batch.append(text if text else " ")  # Handle empty strings

                # Tokenize batch using the tokenizer from the parent class
                # Return tensors for PyTorch
                inputs = self.tokenizer(
                    cleaned_batch,
                    padding="longest",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                    return_attention_mask=True,
                )

                # Move tensors to the device specified in the parent class
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)

                # Get the outputs from the BERT encoder.
                # The model attribute in CheXbert is the full model which
                # includes the BERT encoder and the task-specific heads.
                # We need to access the BERT encoder part.
                bert_output = self.model.bert(
                    input_ids, attention_mask=attention_mask
                )[0]  # [0] is the last hidden state

                # Extract the [CLS] token embedding (first token)
                cls_embeddings = bert_output[:, 0, :].squeeze(dim=1)

                # Move embeddings to CPU and convert to numpy
                new_embeddings.append(cls_embeddings.cpu().numpy())

        new_embeddings = np.concatenate(new_embeddings, axis=0)
        if self.use_cache:
            text_embeddings[missing_indices] = new_embeddings
            self.sentence_to_embedding_cache.update(dict(zip(texts_to_process, new_embeddings)))
            return text_embeddings
        else:
            return new_embeddings

    def compute_cosine_similarity(
        self, hyps: List[str], refs: List[str], batch_size: int = None
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Computes the average sentence-level cosine similarity between hypothesis
        and reference reports.

        Steps:
        1. Split both hypothesis and reference reports into sentences.
        2. Identify all unique sentences across all reports.
        3. Generate embeddings for all unique sentences in batches.
        4. For each hypothesis/reference pair:
            a. Get embeddings for their respective sentences.
            b. Compute a pairwise cosine similarity matrix.
            c. Calculate the mean of row-wise maximums and the mean of
               column-wise maximums.
            d. The average of these two means is the score for the pair.
        5. Calculate the overall average score across all pairs.

        Args:
            hyps: A list of hypothesis report strings.
            refs: A list of reference report strings.
            batch_size: Batch size for embedding generation. Uses instance
                        default if None.

        Returns:
            A dictionary containing:
                - overall_mean_similarity: The average similarity across all pairs.
                - per_pair_similarity: A numpy array of similarity scores for each pair.
        """
        if len(hyps) != len(refs):
            raise ValueError(
                "The number of hypothesis reports and reference reports must be the same."
            )
        if batch_size is None:
            batch_size = self.default_batch_size

        if self.verbose:
            logger.info("Splitting reports into sentences for cosine similarity...")
        hyps_sentences = [sent_tokenize(report) for report in hyps]
        refs_sentences = [sent_tokenize(report) for report in refs]

        unique_sentences = set()
        for sentences in hyps_sentences:
            unique_sentences.update(sentences)
        for sentences in refs_sentences:
            unique_sentences.update(sentences)
        unique_sentences = list(unique_sentences)

        if self.verbose:
            logger.info(f"Unique sentences found: {len(unique_sentences)}")

        if not unique_sentences:
            if self.verbose:
                logger.warning(
                    "No unique sentences found. Returning zero similarity."
                )
            return {
                "overall_mean_similarity": 0.0,
                "per_pair_similarity": np.zeros(len(hyps), dtype=np.float32),
            }

        # Compute embeddings for unique sentences
        if self.verbose:
            logger.info("Generating embeddings for unique sentences (batch)...")
        unique_sentence_embeddings = self.get_embeddings_batch(
            unique_sentences, batch_size=batch_size
        )
        sentence2embedding_index = {
            sentence: i for i, sentence in enumerate(unique_sentences)
        }

        # Compute similarity for each hyp/ref pair
        per_pair_similarity = []
        if self.verbose:
            pair_iterator = tqdm(
                zip(hyps_sentences, refs_sentences),
                total=len(hyps),
                desc="Computing pair-wise cosine similarity",
            )
        else:
            pair_iterator = zip(hyps_sentences, refs_sentences)

        for hyp_sents, ref_sents in pair_iterator:
            if not hyp_sents or not ref_sents:
                # Handle cases with empty sentences in either hyp or ref
                per_pair_similarity.append(0.0)
                continue

            # Get embeddings for the current hyp and ref sentences
            hyp_embeddings = np.array(
                [
                    unique_sentence_embeddings[sentence2embedding_index[s]]
                    for s in hyp_sents
                ]
            )
            ref_embeddings = np.array(
                [
                    unique_sentence_embeddings[sentence2embedding_index[s]]
                    for s in ref_sents
                ]
            )

            # Compute pairwise cosine similarity matrix
            # Shape: (num_hyp_sentences, num_ref_sentences)
            similarity_matrix = cosine_similarity(hyp_embeddings, ref_embeddings)

            # Compute mean of row-wise maximums (each hyp sentence's max similarity to any ref sentence)
            mean_row_max = np.mean(np.max(similarity_matrix, axis=1)) if similarity_matrix.shape[1] > 0 else 0.0

            # Compute mean of column-wise maximums (each ref sentence's max similarity to any hyp sentence)
            mean_col_max = np.mean(np.max(similarity_matrix, axis=0)) if similarity_matrix.shape[0] > 0 else 0.0

            # The average of these two metrics is the score for this pair
            pair_score = (mean_row_max + mean_col_max) / 2.0
            per_pair_similarity.append(pair_score)

        per_pair_similarity_np = np.array(per_pair_similarity, dtype=np.float32)
        overall_mean_similarity = np.mean(per_pair_similarity_np).item()

        if self.verbose:
            logger.info("Cosine similarity calculation complete.")

        return {
            "mean_similarity": overall_mean_similarity,
            "per_pair_similarity": per_pair_similarity_np,
        }

    def visualize_cosine_similarity(
        self,
        ref_report: str,
        cand_report: str,
        figsize: Tuple[int, int] = (10, 8),
        fontsize: int = 10,
        wrap_width: int = 40,
    ):
        """
        Visualizes the sentence-level cosine similarity between a reference and
        candidate report using CheXbert embeddings.

        Args:
            ref_report (str): The ground truth report.
            cand_report (str): The generated (hypothesis) report.
            figsize (Tuple[int, int]): The size of the matplotlib figure.
            fontsize (int): The font size for the annotations in the matrix.
            wrap_width (int): The character width at which to wrap sentence labels.
        """
        # 1. Tokenize reports into sentences
        ref_sents = [s.strip() for s in sent_tokenize(ref_report) if s.strip()]
        cand_sents = [s.strip() for s in sent_tokenize(cand_report) if s.strip()]

        print("--- Candidate Report (Y-axis) ---")
        print(cand_report)

        print("\n--- Reference Report (X-axis) ---")
        print(ref_report)

        if not ref_sents or not cand_sents:
            print("\nCould not generate visualization: One or both reports have no sentences.")
            return

        # 2. Generate embeddings for sentences
        ref_embeddings = self.get_embeddings_batch(ref_sents)
        cand_embeddings = self.get_embeddings_batch(cand_sents)

        # 3. Compute cosine similarity matrix
        # sim_matrix[i, j] is the similarity between cand_sents[i] and ref_sents[j]
        sim_matrix = cosine_similarity(cand_embeddings, ref_embeddings)

        # 4. Create wrapped labels
        wrapped_ref_labels = [textwrap.fill(s, width=wrap_width) for s in ref_sents]
        wrapped_cand_labels = [textwrap.fill(s, width=wrap_width) for s in cand_sents]

        # 5. Plot the heatmap
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(sim_matrix, cmap="viridis", vmin=0, vmax=1)

        # Create colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Cosine Similarity", rotation=-90, va="bottom")

        # Set ticks and labels
        ax.set_xticks(np.arange(len(wrapped_ref_labels)))
        ax.set_yticks(np.arange(len(wrapped_cand_labels)))
        ax.set_xticklabels(wrapped_ref_labels, rotation=90)
        ax.set_yticklabels(wrapped_cand_labels)

        ax.set_xlabel("Reference Sentences", fontweight="bold")
        ax.set_ylabel("Candidate Sentences", fontweight="bold")
        ax.set_title("Sentence Similarity Matrix (CheXbert)", fontweight="bold", fontsize=14)

        # 6. Annotate the heatmap
        for i in range(len(cand_sents)):
            for j in range(len(ref_sents)):
                color = "white" if sim_matrix[i, j] < 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{sim_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=fontsize,
                )

        # Highlight cells that are max in their row and/or column
        highlight_cells = set()
        # Max in rows
        for i in range(sim_matrix.shape[0]):
            max_val = np.max(sim_matrix[i, :])
            max_cols = np.where(sim_matrix[i, :] == max_val)[0]
            for j in max_cols:
                highlight_cells.add((i, j))
        # Max in columns
        for j in range(sim_matrix.shape[1]):
            max_val = np.max(sim_matrix[:, j])
            max_rows = np.where(sim_matrix[:, j] == max_val)[0]
            for i in max_rows:
                highlight_cells.add((i, j))
        # Draw squares
        for (i, j) in highlight_cells:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='red', lw=3))

        fig.tight_layout()
        plt.show()

        # 7. Calculate and print the final score for this pair
        mean_row_max = np.mean(np.max(sim_matrix, axis=1))
        mean_col_max = np.mean(np.max(sim_matrix, axis=0))
        pair_score = (mean_row_max + mean_col_max) / 2.0
        print(f"\nCheXbert Cosine Similarity Score for this pair: {pair_score:.4f}")

    # Override the forward method
    def forward(self, hyps: List[str], refs: List[str], batch_size: Optional[int] = None,
                split_into_sentences_first: bool = True) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculates F1CheXbert metrics using batch processing for label generation.

        Args:
            hyps: A list of hypothesis report strings.
            refs: A list of reference report strings.
            batch_size: Batch size for CheXbert label generation. Uses instance
                        default if None.
            split_into_sentences_first: If True, splits reports into sentences
                                        before processing, predicting labels for
                                        each sentence, and then merging them.
                                        This is useful for long reports with
                                        multiple findings. Defaults to True.

        Returns:
            A dictionary containing:
                - accuracy: Overall accuracy of the predictions.
                - per_element_accuracy: Accuracy for each pair of reference and hypothesis.
                - classification_report: Detailed classification report.
                - ref_labels: Reference labels as a numpy array.
                - hyp_labels: Hypothesis labels as a numpy array.
        """
        if batch_size is None:
            batch_size = self.default_batch_size

        # --- Preprocessing (if split_into_sentences_first is True) ---
        if split_into_sentences_first:
            if self.verbose:
                logger.info("Splitting reports into sentences...")
            hyps_sentences = [sent_tokenize(report) for report in hyps]
            refs_sentences = [sent_tokenize(report) for report in refs]
            unique_sentences = set()
            for sentences in hyps_sentences:
                unique_sentences.update(sentences)
            for sentences in refs_sentences:
                unique_sentences.update(sentences)
            unique_sentences = list(unique_sentences)
            if self.verbose:
                logger.info(f"Unique sentences found: {len(unique_sentences)}")
            sentence2index = {sentence: i for i, sentence in enumerate(unique_sentences)}

            # Compute CheXbert labels for unique sentences
            if self.verbose:
                logger.info("Generating CheXbert labels for unique sentences...")
            unique_labels = self.get_labels_batch(unique_sentences, batch_size=batch_size, mode="rrg")

        # --- Reference Labels ---
        if split_into_sentences_first:
            refs_chexbert = []
            for sentences in refs_sentences:
                s_idxs = [sentence2index[s] for s in sentences if s in sentence2index]
                labels = merge_labels([unique_labels[i] for i in s_idxs])
                refs_chexbert.append(labels)
        else:
            if self.verbose:
                logger.info("Generating CheXbert labels for references (batch)...")
            refs_chexbert = self.get_labels_batch(refs, batch_size=batch_size, mode="rrg")

        # --- Hypothesis Labels ---
        if split_into_sentences_first:
            hyps_chexbert = []
            for sentences in hyps_sentences:
                s_idxs = [sentence2index[s] for s in sentences if s in sentence2index]
                labels = merge_labels([unique_labels[i] for i in s_idxs])
                hyps_chexbert.append(labels)
        else:
            if self.verbose:
                logger.info("Generating CheXbert labels for hypotheses (batch)...")
            hyps_chexbert = self.get_labels_batch(hyps, batch_size=batch_size, mode="rrg")

        # --- Calculations ---
        if self.verbose:
            logger.info("Calculating metrics...")
        refs_chexbert_np = np.array(refs_chexbert)
        hyps_chexbert_np = np.array(hyps_chexbert)

        # Basic shape validation
        if refs_chexbert_np.shape != hyps_chexbert_np.shape:
            raise ValueError(f"Shape mismatch between reference labels ({refs_chexbert_np.shape}) and hypothesis labels ({hyps_chexbert_np.shape}).")
        if refs_chexbert_np.ndim != 2 or refs_chexbert_np.shape[1] != len(self.target_names):
            raise ValueError(f"Unexpected shape for reference labels: {refs_chexbert_np.shape}. Expected (num_reports, {len(self.target_names)}).")

        # Accuracy and per-element accuracy
        accuracy = np.mean(refs_chexbert_np == hyps_chexbert_np)
        pe_accuracy = np.mean(refs_chexbert_np == hyps_chexbert_np, axis=1).astype(np.float32)

        # Classification report
        cr = classification_report(refs_chexbert_np, hyps_chexbert_np, target_names=self.target_names, output_dict=True, zero_division=0)
        if self.verbose:
            logger.info("Metrics calculation complete.")

        return {
            'accuracy': accuracy.item(),
            'per_element_accuracy': pe_accuracy,
            'classification_report': cr,
            'ref_labels': refs_chexbert_np,
            'hyp_labels': hyps_chexbert_np,
        }