import logging
import os
import pickle
import textwrap
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

# Ensure the NLTK punkt tokenizer is downloaded
from vlm_research_kit.nltk_setup import ensure_nltk_resources
from vlm_research_kit.settings import CXRFESCORE_CACHE_DIR
from vlm_research_kit.utils.text_generation_utils import (
    parse_facts,
    remove_consecutive_repeated_words_from_text,
)

ensure_nltk_resources()

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]

def get_text_collate_batch_fn(tokenizer_func):
    def collate_batch_fn(batch):
        encoding = tokenizer_func(batch)
        batch_dict = {}
        batch_dict['encoding'] = encoding
        return batch_dict
    return collate_batch_fn

def create_text_dataset_and_dataloader(texts, batch_size, num_workers, tokenizer_func):
    # Create collate batch function
    collate_batch_fn = get_text_collate_batch_fn(tokenizer_func)
    # Create dataset
    dataset = TextDataset(texts)
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch_fn,
        pin_memory=True,
    )
    return dataset, dataloader


class CXRFEScore:
    """
    A metric for computing fact-level cosine similarity between radiology
    reports using a T5 fact extractor and CXR-BERT embeddings.

    Steps:
    1. Splits hypothesis and reference reports into sentences.
    2. Uses a T5 fact extractor to extract facts from the sentences.
    3. Gathers all unique facts to avoid redundant computations.
    4. Generates fact embeddings for all unique facts in batches
       using a specified CXR-BERT model.
    5. For each hypothesis/reference pair, it computes a pairwise cosine
       similarity matrix between their fact embeddings.
    6. The final similarity score for a pair is the average of the mean of
       row-wise maximums and the mean of column-wise maximums from the matrix.
    """

    SUPPORTED_MODELS = [
        "pamessina/CXRFE",
        "microsoft/BiomedVLP-CXR-BERT-specialized",
        "microsoft/BiomedVLP-BioViL-T",
    ]

    MODEL_EMBEDDING_DIMENSIONS = {
        "pamessina/CXRFE": 128,
        "microsoft/BiomedVLP-CXR-BERT-specialized": 128,
        "microsoft/BiomedVLP-BioViL-T": 128,
    }

    def __init__(
        self,
        encoder_model_name: str = "pamessina/CXRFE",
        extractor_model_name: str = "pamessina/T5FactExtractor",
        device: Optional[str] = None,
        batch_size: int = 128,
        num_workers: int = 3,
        verbose: bool = False,
        use_cache: bool = True,
        cache_dir: str = CXRFESCORE_CACHE_DIR,
    ):
        """
        Initializes the CXRFEScore metric.

        Args:
            model_name: The Hugging Face model name. Must be one of the
                        supported models.
            device: The device to run the model on ('cuda', 'cpu').
                    Automatically detects if None.
            batch_size: Default batch size for embedding generation.
            verbose: If True, enables verbose logging and progress bars.
            use_cache: If True, enables in-memory and disk caching for facts
                       and embeddings.
            cache_dir: Directory to store cache files if use_cache is True.
        """
        if encoder_model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model name: {encoder_model_name}. "
                f"Please choose from: {self.SUPPORTED_MODELS}"
            )

        if use_cache and cache_dir is None:
            raise ValueError(
                "cache_dir must be provided if use_cache is True."
            )

        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.encoder_model_name = encoder_model_name
        self.extractor_model_name = extractor_model_name
        self.default_batch_size = batch_size
        self.default_num_workers = num_workers
        self.verbose = verbose        
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.embedding_dimension = self.MODEL_EMBEDDING_DIMENSIONS[self.encoder_model_name]
        # --- Initialize Caches ---
        self.sent_to_facts_cache = {}
        self.fact_to_embedding_cache = {}
        if self.use_cache:
            self._load_cache()

        if self.verbose:
            logger.info(f"Initializing CXRFEScore with encoder model: {self.encoder_model_name} and"
                        f" extractor model: {self.extractor_model_name}, use_cache: {self.use_cache},"
                        f" cache_dir: {self.cache_dir}, default batch size: {self.default_batch_size},"
                        f" default num workers: {self.default_num_workers}.")
            logger.info(f"Using device: {self.device}")

        # --- Initialize Encoder Model (CXR-BERT) ---
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(
            encoder_model_name, trust_remote_code=True
        )
        self.encoder_model = AutoModel.from_pretrained(
            encoder_model_name, trust_remote_code=True
        )
        self.encoder_model.to(self.device)
        self.encoder_model.eval()
        
        # --- Initialize Extractor Model (T5) ---
        self.extractor_tokenizer = T5TokenizerFast.from_pretrained(
            extractor_model_name
        )
        self.extractor_model = T5ForConditionalGeneration.from_pretrained(
            extractor_model_name
        )
        self.extractor_model.to(self.device)
        self.extractor_model.eval()

    def _load_cache(self):
        """Loads fact and embedding caches from disk if they exist."""
        facts_cache_path = os.path.join(self.cache_dir, "sent_to_facts.pkl")
        embed_cache_path = os.path.join(self.cache_dir, "fact_to_embedding.pkl")

        if os.path.exists(facts_cache_path):
            try:
                with open(facts_cache_path, "rb") as f:
                    self.sent_to_facts_cache = pickle.load(f)
                if self.verbose:
                    logger.info(f"Loaded {len(self.sent_to_facts_cache)} sentence-to-fact mappings from {facts_cache_path}.")
            except (pickle.UnpicklingError, EOFError):
                logger.warning(f"Could not load sentence-to-fact mappings from {facts_cache_path}. Starting with an empty cache.")

        if os.path.exists(embed_cache_path):
            try:
                with open(embed_cache_path, "rb") as f:
                    self.fact_to_embedding_cache = pickle.load(f)
                if self.verbose:
                    logger.info(f"Loaded {len(self.fact_to_embedding_cache)} fact embeddings from {embed_cache_path}.")
            except (pickle.UnpicklingError, EOFError):
                logger.warning(f"Could not load fact embeddings from {embed_cache_path}. Starting with an empty cache.")

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

    def _save_sent_to_facts_cache(self):
        """Saves the sentence-to-facts cache to disk."""
        self._save_cache_file(
            self.sent_to_facts_cache, "sent_to_facts.pkl"
        )

    def _save_fact_to_embedding_cache(self):
        """Saves the fact-to-embedding cache to disk."""
        self._save_cache_file(
            self.fact_to_embedding_cache, "fact_to_embedding.pkl"
        )

    def save_cache(self):
        """Saves the current in-memory caches to disk."""
        self._save_sent_to_facts_cache()
        self._save_fact_to_embedding_cache()

    @staticmethod
    def _aggregate_facts(report_sents: List[str], sent_to_facts_map: Dict[str, List[str]]) -> List[str]:
        """
        Aggregates facts from a list of sentences into a list of unique facts.
        """
        report_facts = []
        seen_facts = set()
        for sent in report_sents:
            sent = sent.strip()
            facts = sent_to_facts_map[sent]
            # If no facts are found, use the sentence itself if the sentence contains words (at least one letter)
            if len(facts) == 0 and any(char.isalpha() for char in sent):
                facts = [sent]
            for fact in facts:
                if fact not in seen_facts:
                    report_facts.append(fact)
                    seen_facts.add(fact)
        return report_facts        

    def _extract_facts_batch(
        self, sentences: List[str], batch_size: Optional[int] = None, num_workers: Optional[int] = None
    ) -> List[List[str]]:
        """
        Extracts facts from a batch of sentences using the T5 fact extractor.
        """
        if batch_size is None:
            batch_size = self.default_batch_size
        if num_workers is None:
            num_workers = self.default_num_workers
        
        if self.verbose:
            logger.info(f"Extracting facts from {len(sentences)} sentences in batches of {batch_size} with {num_workers} workers.")

        if self.use_cache:
            output = [self.sent_to_facts_cache.get(sent, None) for sent in sentences]
            missing_indices = [i for i in range(len(sentences)) if output[i] is None]
            if not missing_indices:
                if self.verbose:
                    logger.info(f"All sentences found in cache. Returning {len(output)} sentences-to-facts mappings.")
                return output
            else:
                if self.verbose:
                    logger.info(f"Cache miss for {len(missing_indices)}/{len(sentences)} sentences.")
                sentences_to_process = [sentences[i] for i in missing_indices]
        else:
            sentences_to_process = sentences

        # Create dataset and dataloader
        _, dataloader = create_text_dataset_and_dataloader(
            texts=sentences_to_process,
            batch_size=batch_size,
            num_workers=num_workers,
            tokenizer_func=lambda x: self.extractor_tokenizer(x, padding='longest', return_tensors='pt'),
        )

        # Run inference
        new_facts = [None] * len(sentences_to_process)
        offset = 0
        with torch.no_grad():
            if self.verbose:
                iterator = tqdm(dataloader, total=len(dataloader), mininterval=2)
            else:
                iterator = dataloader
            for batch in iterator:
                encoding = batch['encoding']
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                max_len = input_ids.shape[1] * 4 # 4x input length
                output_ids = self.extractor_model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                  max_new_tokens=max_len, num_beams=1)
                output_texts_batch = self.extractor_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                for i, output_text in enumerate(output_texts_batch):
                    facts = parse_facts(output_text)
                    facts = [remove_consecutive_repeated_words_from_text(f) for f in facts]
                    new_facts[offset+i] = facts
                offset += len(output_texts_batch)
        assert offset == len(sentences_to_process)
        assert None not in new_facts

        if self.use_cache:
            for i, orig_idx in enumerate(missing_indices):
                output[orig_idx] = new_facts[i]
            self.sent_to_facts_cache.update({sent: facts for sent, facts in zip(sentences_to_process, new_facts)})
        else:
            output = new_facts
        assert None not in output
        return output

    def _get_embeddings_batch(
        self, texts: List[str], batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Processes a list of text strings in batches to generate their embeddings
        using the specified CXR-BERT model.

        Args:
            texts: A list of text strings (sentences).
            batch_size: The number of texts to process in each batch.
                        Uses instance default if None.

        Returns:
            A numpy array of embeddings, where each row corresponds to the
            embedding of an input text string.
        """
        if batch_size is None:
            batch_size = self.default_batch_size

        if self.verbose:
            logger.info(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}.")

        if not texts:
            return np.array([])        

        if self.use_cache:
            text_embeddings = np.empty((len(texts), self.embedding_dimension), dtype=np.float32)
            missing_indices = []
            for i, text in enumerate(texts):
                embedding = self.fact_to_embedding_cache.get(text, None)
                if embedding is None:
                    missing_indices.append(i)
                else:
                    text_embeddings[i] = embedding
            if not missing_indices:
                if self.verbose:
                    logger.info(f"All texts found in cache. Returning {len(text_embeddings)} embeddings.")
                return text_embeddings
            else:
                if self.verbose:
                    logger.info(f"Cache miss for {len(missing_indices)} texts.")
                texts_to_process = [texts[i] for i in missing_indices]
        else:
            texts_to_process = texts

        self.encoder_model.eval() # Ensure model is in eval mode

        new_embeddings = []

        iterator = (
            tqdm(range(0, len(texts_to_process), batch_size), desc="Generating embeddings")
            if self.verbose
            else range(0, len(texts_to_process), batch_size)
        )

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts_to_process[i : i + batch_size]

                # CXR-BERT models from Microsoft use `batch_encode_plus`
                inputs = self.encoder_tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=batch_texts,
                    add_special_tokens=True,
                    padding="longest",
                    return_tensors="pt",
                )

                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)

                # These models have a specific method for projected embeddings
                batch_embeddings = self.encoder_model.get_projected_text_embeddings(
                    input_ids=input_ids, attention_mask=attention_mask
                )

                new_embeddings.append(batch_embeddings.cpu().numpy())

        new_embeddings = np.concatenate(new_embeddings, axis=0) # (len(texts_to_process), 128)
        
        if self.use_cache:
            text_embeddings[missing_indices] = new_embeddings
            self.fact_to_embedding_cache.update({text: embedding for text, embedding in zip(texts_to_process, new_embeddings)})
            return text_embeddings
        else:
            return new_embeddings

    def compute(
        self, hyps: List[str], refs: List[str], batch_size: Optional[int] = None, num_workers: Optional[int] = None
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Computes the average sentence-level cosine similarity between
        hypothesis and reference reports.

        Args:
            hyps: A list of hypothesis report strings.
            refs: A list of reference report strings.
            batch_size: Batch size for embedding generation. Uses instance
                        default if None.
            num_workers: The number of workers to use for the dataloader.
                        Uses instance default if None.

        Returns:
            A dictionary containing:
                - mean_similarity: The average similarity across all pairs.
                - per_pair_similarity: A numpy array of similarity scores for
                                       each pair.
        """
        if len(hyps) != len(refs):
            raise ValueError(
                "The number of hypothesis and reference reports must be the same."
            )

        if self.verbose:
            logger.info("Splitting reports into sentences...")
        hyps_sents_per_report = [sent_tokenize(report) for report in hyps]
        refs_sents_per_report = [sent_tokenize(report) for report in refs]

        # Gather all unique sentences from all reports to extract facts efficiently
        all_unique_sents = set()
        for report_sents in hyps_sents_per_report:
            all_unique_sents.update(s.strip() for s in report_sents if s.strip())
        for report_sents in refs_sents_per_report:
            all_unique_sents.update(s.strip() for s in report_sents if s.strip())
        all_unique_sents_list = list(all_unique_sents)

        if self.verbose:
            logger.info(f"Found {len(all_unique_sents_list)} unique sentences. Extracting facts...")

        # Extract facts from all unique sentences
        facts_per_unique_sent = self._extract_facts_batch(all_unique_sents_list, batch_size=batch_size, num_workers=num_workers)
        sent_to_facts = {sent: facts for sent, facts in zip(all_unique_sents_list, facts_per_unique_sent)}

        hyps_facts = [self._aggregate_facts(sents, sent_to_facts) for sents in hyps_sents_per_report]
        refs_facts = [self._aggregate_facts(sents, sent_to_facts) for sents in refs_sents_per_report]

        # Gather all unique facts to avoid redundant computations
        unique_facts = set()
        for facts in hyps_facts:
            unique_facts.update(facts)
        for facts in refs_facts:
            unique_facts.update(facts)
        unique_facts = list(unique_facts)

        if self.verbose:
            logger.info(f"Found {len(unique_facts)} unique facts.")

        if not unique_facts:
            return {
                "mean_similarity": 0.0,
                "per_pair_similarity": np.zeros(len(hyps), dtype=np.float32),
            }

        unique_embeddings = self._get_embeddings_batch(
            unique_facts, batch_size=batch_size
        )
        fact_to_embedding_idx = {
            fact: i for i, fact in enumerate(unique_facts)
        }

        per_pair_similarity = []
        pair_iterator = (
            tqdm(
                zip(hyps_facts, refs_facts),
                total=len(hyps),
                desc="Computing pair-wise similarity",
            )
            if self.verbose
            else zip(hyps_facts, refs_facts)
        )

        for hyp_facts, ref_facts in pair_iterator:
            # Filter out empty sentences after tokenization
            hyp_facts = [f.strip() for f in hyp_facts if f.strip()]
            ref_facts = [f.strip() for f in ref_facts if f.strip()]

            if not hyp_facts or not ref_facts:
                per_pair_similarity.append(0.0)
                continue

            hyp_indices = [fact_to_embedding_idx[f] for f in hyp_facts]
            ref_indices = [fact_to_embedding_idx[f] for f in ref_facts]

            hyp_embeddings = unique_embeddings[hyp_indices]
            ref_embeddings = unique_embeddings[ref_indices]

            similarity_matrix = cosine_similarity(hyp_embeddings, ref_embeddings)

            mean_row_max = np.mean(np.max(similarity_matrix, axis=1))
            mean_col_max = np.mean(np.max(similarity_matrix, axis=0))

            pair_score = (mean_row_max + mean_col_max) / 2.0
            per_pair_similarity.append(pair_score)

        per_pair_similarity_np = np.array(per_pair_similarity, dtype=np.float32)
        mean_similarity = np.mean(per_pair_similarity_np).item()

        if self.verbose:
            logger.info("Similarity calculation complete.")

        return {
            "mean_similarity": mean_similarity,
            "per_pair_similarity": per_pair_similarity_np,
        }

    def __call__(
        self, hyps: List[str], refs: List[str], batch_size: Optional[int] = None, num_workers: Optional[int] = None
    ) -> Dict[str, Union[float, np.ndarray]]:
        return self.compute(hyps=hyps, refs=refs, batch_size=batch_size, num_workers=num_workers)

    def visualize_fact_similarity(
        self,
        ref_report: str,
        cand_report: str,
        figsize: Tuple[int, int] = (10, 8),
        fontsize: int = 10,
        wrap_width: int = 40,
    ):
        """
        Visualizes the sentence-level cosine similarity between a reference and
        candidate report.

        Args:
            metric (CXRFEScore): An initialized instance of the metric class
                                        used for embedding generation.
            ref_report (str): The ground truth report.
            cand_report (str): The generated (hypothesis) report.
            figsize (Tuple[int, int]): The size of the matplotlib figure.
            fontsize (int): The font size for the annotations in the matrix.
            wrap_width (int): The character width at which to wrap sentence labels.
        """
        # 1. Tokenize reports into sentences first
        ref_sents = [s.strip() for s in sent_tokenize(ref_report) if s.strip()]
        cand_sents = [s.strip() for s in sent_tokenize(cand_report) if s.strip()]

        # 2. Extract facts from the sentences
        ref_facts = self._extract_facts_batch(ref_sents)
        cand_facts = self._extract_facts_batch(cand_sents)

        ref_facts = self._aggregate_facts(ref_sents, {sent: facts for sent, facts in zip(ref_sents, ref_facts)})
        cand_facts = self._aggregate_facts(cand_sents, {sent: facts for sent, facts in zip(cand_sents, cand_facts)})

        print("--- Candidate Report (Y-axis) ---")
        print(cand_report)
        print("\n--- Extracted Candidate Facts ---")
        for fact in cand_facts:
            print(f"- {fact}")

        print("\n--- Reference Report (X-axis) ---")
        print(ref_report)
        print("\n--- Extracted Reference Facts ---")
        for fact in ref_facts:
            print(f"- {fact}")

        if not ref_facts or not cand_facts:
            print("\nCould not generate visualization: One or both reports have no facts.")
            return

        # 3. Generate embeddings for facts
        ref_embeddings = self._get_embeddings_batch(ref_facts)
        cand_embeddings = self._get_embeddings_batch(cand_facts)

        # 4. Compute cosine similarity matrix
        # sim_matrix[i, j] is the similarity between cand_facts[i] and ref_facts[j]
        sim_matrix = cosine_similarity(cand_embeddings, ref_embeddings)

        # 5. Create wrapped labels using the facts
        wrapped_ref_labels = [
            textwrap.fill(s, width=wrap_width) for s in ref_facts
        ]
        wrapped_cand_labels = [
            textwrap.fill(s, width=wrap_width) for s in cand_facts
        ]

        # 6. Plot the heatmap
        fig, ax = plt.subplots(figsize=figsize)
        from matplotlib.colors import LinearSegmentedColormap

        # Create a custom colormap that interpolates from red (-1) to white (0) to blue (1)
        colors = [
            (1.0, 0.0, 0.0),  # Red for -1
            (1.0, 0.5, 0.0),  # Orange for -0.5
            (1.0, 1.0, 1.0),  # White for 0
            (0.0, 0.5, 1.0),  # Light blue for 0.5
            (0.0, 0.0, 1.0),  # Blue for 1
        ]
        positions = [0.0, 0.25, 0.5, 0.75, 1.0]  # -1, -0.5, 0, 0.5, 1 mapped to [0, 0.25, 0.5, 0.75, 1]
        custom_cmap = LinearSegmentedColormap.from_list("red_white_blue", list(zip(positions, colors)))

        im = ax.imshow(sim_matrix, cmap=custom_cmap, vmin=-1, vmax=1)

        # Create colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Cosine Similarity", rotation=-90, va="bottom")

        # Set ticks and labels
        ax.set_xticks(np.arange(len(wrapped_ref_labels)))
        ax.set_yticks(np.arange(len(wrapped_cand_labels)))
        ax.set_xticklabels(wrapped_ref_labels, rotation=90)
        ax.set_yticklabels(wrapped_cand_labels)

        ax.set_xlabel("Reference Facts", fontweight="bold")
        ax.set_ylabel("Candidate Facts", fontweight="bold")
        ax.set_title("Fact Similarity Matrix", fontweight="bold", fontsize=14)

        # 6. Annotate the heatmap with similarity values
        for i in range(len(cand_facts)):
            for j in range(len(ref_facts)):
                color = "black" if -0.5 <= sim_matrix[i, j] <= 0.5 else "white"
                ax.text(
                    j,
                    i,
                    f"{sim_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=fontsize,
                )

        # Highlight cells that are max in their row and/or column with a purple square
        from matplotlib.patches import Rectangle
        highlight_cells = set()
        # Max in rows (handling ties)
        for i in range(sim_matrix.shape[0]):
            max_val = np.max(sim_matrix[i, :])
            max_cols = np.where(sim_matrix[i, :] == max_val)[0]
            for j in max_cols:
                highlight_cells.add((i, j))
        # Max in columns (handling ties)
        for j in range(sim_matrix.shape[1]):
            max_val = np.max(sim_matrix[:, j])
            max_rows = np.where(sim_matrix[:, j] == max_val)[0]
            for i in max_rows:
                highlight_cells.add((i, j))
        # Draw purple squares
        for (i, j) in highlight_cells:
            # Use light green
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='lightgreen', lw=3))

        fig.tight_layout()
        plt.show()

        # 7. Calculate and print the final score for this pair
        if sim_matrix.size > 0:
            mean_row_max = np.mean(np.max(sim_matrix, axis=1))
            mean_col_max = np.mean(np.max(sim_matrix, axis=0))
            pair_score = (mean_row_max + mean_col_max) / 2.0
            print(f"\nCXRFEScore for this pair: {pair_score:.4f}")
        else:
            print("\nCXRFEScore for this pair: 0.0")