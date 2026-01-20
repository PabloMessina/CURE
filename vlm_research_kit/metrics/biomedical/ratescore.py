import hashlib
import logging
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from RaTEScore import RaTEScore
from RaTEScore.utils import compute, post_process, sentence_split
from tqdm import tqdm

from vlm_research_kit.settings import RATESCORE_CACHE_DIR

# Logger setup
logger = logging.getLogger(__name__)


def hash_string(s: str) -> str:
    """Creates a unique and file-safe hash for a given string."""
    return f'{len(s)}_{hashlib.sha256(s.encode("utf-8")).hexdigest()}'


class SyntaxWarningFilter(logging.Filter):
    def filter(self, record):
        return "eligible syntax" not in record.getMessage()


class RaTEScoreScorer:
    """
    Computes RaTEScore between generated and reference reports with efficient,
    multi-level caching (Sentence-Level).
    """

    def __init__(
        self,
        bert_model: str = "Angelakeke/RaTE-NER-Deberta",
        eval_model: str = "FremyCompany/BioLORD-2023-C",
        batch_size: int = 32,
        device: Optional[Union[str, torch.device]] = None,
        use_cache: bool = True,
        cache_dir: str = RATESCORE_CACHE_DIR,
        verbose: bool = False,
        print_debug_info: bool = False,
    ):
        self.verbose = verbose
        self.print_debug_info = print_debug_info
        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
        
        # --- LOGGING FIXES ---
        logging.getLogger().addFilter(SyntaxWarningFilter())
        logging.getLogger("PyRuSH").setLevel(logging.ERROR)
        logging.getLogger("PyRuSH.PyRuSHSentencizer").setLevel(logging.ERROR)
        try:
            from loguru import logger as loguru_logger
            loguru_logger.remove()
        except ImportError:
            pass
            
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        
        self.ner_cache_path = os.path.join(self.cache_dir, "ner_sentence_cache.pkl")
        self.embed_cache_path = os.path.join(self.cache_dir, "embedding_cache.pkl")
        self.split_cache_path = os.path.join(self.cache_dir, "split_cache.pkl")

        # Cache maps sentence_hash -> List[Tuple[str, str]]
        self.sentence_to_ner_cache: Dict[str, List[Tuple[str, str]]] = {}
        # Cache maps entity_tuple -> np.ndarray
        self.entity_to_embedding_cache: Dict[Tuple[str, str], np.ndarray] = {}
        # Cache maps report_hash -> List[str] (sentences)
        self.report_to_sentences_cache: Dict[str, List[str]] = {}

        if self.use_cache:
            self._load_cache()

        use_gpu = False
        if device is not None:
            device_str = str(device)
            use_gpu = device_str != "cpu"
        elif torch.cuda.is_available():
            use_gpu = True

        logger.info(f"Initializing RaTEScore with use_gpu={use_gpu}")

        self.scorer = RaTEScore(
            bert_model=bert_model,
            eval_model=eval_model,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )

    def _load_cache_file(self, path: str) -> dict:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if self.verbose:
                    logger.info(f"Loaded {len(data)} entries from {path}.")
                return data
            except (pickle.UnpicklingError, EOFError):
                logger.warning(f"Could not load cache from {path}. Starting fresh.")
        return {}

    def _load_cache(self):
        """Loads caches from disk."""
        self.sentence_to_ner_cache = self._load_cache_file(self.ner_cache_path)
        self.entity_to_embedding_cache = self._load_cache_file(self.embed_cache_path)
        self.report_to_sentences_cache = self._load_cache_file(self.split_cache_path)

    def _save_cache_file(self, data: dict, path: str):
        if not self.use_cache:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_path = path + ".tmp"
        try:
            with open(temp_path, "wb") as f:
                pickle.dump(data, f)
            os.replace(temp_path, path)
            if self.verbose:
                logger.info(f"Saved {len(data)} entries to {path}.")
        except Exception as e:
            logger.error(f"Failed to save cache to {path}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def save_cache(self):
        self._save_cache_file(self.sentence_to_ner_cache, self.ner_cache_path)
        self._save_cache_file(self.entity_to_embedding_cache, self.embed_cache_path)
        self._save_cache_file(self.report_to_sentences_cache, self.split_cache_path)

    def _get_ner_for_texts(
        self, texts: List[str]
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Retrieves NER entities for a list of reports with strict alignment.
        """
        
        # --- 1. Map Reports to Sentences (CPU) ---
        split_start_time = time.time()

        report_to_sentences = {}
        unique_missing_sentences = set()

        for report in texts:
            if not report.strip():
                report_to_sentences[report] = []
                continue
            
            report_hash = hash_string(report)
            
            # 1. Check Cache
            if self.use_cache and report_hash in self.report_to_sentences_cache:
                valid_sents = self.report_to_sentences_cache[report_hash]
            else:
                # 2. Cache Miss: Run expensive split
                # sentence_split returns (list_of_sentences, boolean_mask)
                sents, _ = sentence_split([report])
                
                # Filter extremely short junk strings immediately
                valid_sents = [s for s in sents if len(s.strip()) >= 3]
                
                # Update Cache
                if self.use_cache:
                    self.report_to_sentences_cache[report_hash] = valid_sents

            report_to_sentences[report] = valid_sents
            
            # Check which sentences need inference (Existing logic)
            for s in valid_sents:
                s_hash = hash_string(s)
                if not (self.use_cache and s_hash in self.sentence_to_ner_cache):
                    unique_missing_sentences.add(s)

        split_end_time = time.time()
        if self.print_debug_info or self.verbose:
            logger.info(f"Sentence splitting time: {split_end_time - split_start_time:.2f} seconds")

        # --- 2. Run Inference on Missing Sentences (GPU) ---
        inference_start_time = time.time()
        missing_sentences_list = sorted(list(unique_missing_sentences))

        if missing_sentences_list:
            if self.verbose:
                logger.info(f"Computing NER for {len(missing_sentences_list)} unique new sentences...")

            iterator = range(0, len(missing_sentences_list), self.batch_size)
            if self.verbose:
                iterator = tqdm(iterator, desc="NER Inference")

            for i in iterator:
                batch_texts = missing_sentences_list[i : i + self.batch_size]

                # Tokenize
                inputs = self.scorer.tokenizer(
                    batch_texts,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.scorer.device)

                with torch.no_grad():
                    outputs = self.scorer.model(**inputs)
                
                # Get indices
                batch_predicted_indices = torch.argmax(outputs.logits, dim=2).tolist()
                input_ids = inputs["input_ids"].cpu()
                pad_token_id = self.scorer.tokenizer.pad_token_id

                # Post-process
                for j, text in enumerate(batch_texts):
                    predicted_labels = [
                        self.scorer.idx2label[label] 
                        for label in batch_predicted_indices[j]
                    ]
                    
                    non_pad_mask = input_ids[j] != pad_token_id
                    non_pad_length = non_pad_mask.sum().item()
                    non_pad_input_ids = input_ids[j][:non_pad_length]
                    tokenized_text = self.scorer.tokenizer.convert_ids_to_tokens(non_pad_input_ids)

                    extracted_pairs = post_process(
                        tokenized_text, predicted_labels, self.scorer.tokenizer
                    )

                    # Update Cache
                    s_hash = hash_string(text)
                    self.sentence_to_ner_cache[s_hash] = extracted_pairs
        
        inference_end_time = time.time()
        if self.print_debug_info or self.verbose:
            logger.info(f"NER inference time: {inference_end_time - inference_start_time:.2f} seconds")

        # --- 3. Reconstruct Report Entities and Debug Log ---
        reconstruction_start_time = time.time()
        report_to_ner_map = {}
        
        for report in texts:
            sentences = report_to_sentences.get(report, [])
            report_entities = []
            
            # DEBUGGING LOGIC
            if self.print_debug_info:
                # Use a separator line to make it readable
                logger.info(f"\n--- Report Analysis: '{report}' ---")
                if not sentences:
                    logger.info("   (No valid sentences found or empty input)")

            for sent in sentences:
                s_hash = hash_string(sent)
                ents = self.sentence_to_ner_cache.get(s_hash, [])
                report_entities.extend(ents)
                
                if self.print_debug_info:
                    logger.info(f"   Sentence: '{sent}'")
                    if ents:
                        logger.info(f"   -> Entities: {ents}")
                    else:
                        logger.info("   -> Entities: None")

            report_to_ner_map[report] = report_entities
        
        reconstruction_end_time = time.time()
        if self.print_debug_info or self.verbose:
            logger.info(f"Report reconstruction time: {reconstruction_end_time - reconstruction_start_time:.2f} seconds")

        return report_to_ner_map

    def _get_embeddings_for_entities(
        self, entities: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], np.ndarray]:
        """Gets embeddings for a list of entity tuples, using a cache."""
        entity_to_embedding = {}
        uncached_entities = []

        for entity in entities:
            if self.use_cache and entity in self.entity_to_embedding_cache:
                entity_to_embedding[entity] = self.entity_to_embedding_cache[entity]
            else:
                if entity not in uncached_entities:
                    uncached_entities.append(entity)

        if uncached_entities:
            if self.verbose:
                logger.info(
                    f"Cache miss for {len(uncached_entities)} unique entities. Generating embeddings..."
                )
            entity_texts = [e[0] for e in uncached_entities]
            new_embeddings = []
            iterator = (
                tqdm(range(0, len(entity_texts), self.batch_size), desc="Generating embeddings")
                if self.verbose else range(0, len(entity_texts), self.batch_size)
            )
            with torch.no_grad():
                for i in iterator:
                    batch_texts = entity_texts[i : i + self.batch_size]
                    encoded = self.scorer.eval_tokenizer(
                        batch_texts,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                        max_length=30,
                    ).to(self.scorer.device)
                    embeds = self.scorer.eval_model(**encoded).last_hidden_state[:, 0, :]
                    new_embeddings.append(embeds.cpu().numpy())

            if new_embeddings:
                new_embeddings = np.concatenate(new_embeddings, axis=0)
            
            for entity, embedding in zip(uncached_entities, new_embeddings):
                entity_to_embedding[entity] = embedding
                if self.use_cache:
                    self.entity_to_embedding_cache[entity] = embedding
        return entity_to_embedding

    def compute(
        self, hyps: List[str], refs: List[str]
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Computes RaTEScore."""
        if len(hyps) != len(refs):
            raise ValueError("Hyps and Refs must have the same length.")

        # 1. Get NER results for all unique reports
        ner_start_time = time.time()
        unique_reports = sorted(list(set(hyps + refs)))
        report_to_ner_map = self._get_ner_for_texts(unique_reports)
        ner_end_time = time.time()
        if self.print_debug_info or self.verbose:
            logger.info(f"NER computation time: {ner_end_time - ner_start_time:.2f} seconds")

        # 2. Gather all unique entities and get their embeddings
        embed_start_time = time.time()
        all_entities = set()
        for ner_list in report_to_ner_map.values():
            all_entities.update(ner_list)
        entity_to_embedding_map = self._get_embeddings_for_entities(list(all_entities))
        embed_end_time = time.time()
        if self.print_debug_info or self.verbose:
            logger.info(f"Embedding computation time: {embed_end_time - embed_start_time:.2f} seconds")

        # 3. Compute scores
        compute_start_time = time.time()
        scores = []
        pair_iterator = (
            tqdm(zip(hyps, refs), total=len(hyps), desc="Computing RaTEScore")
            if self.verbose else zip(hyps, refs)
        )

        for hyp, ref in pair_iterator:
            hyp_entities = report_to_ner_map.get(hyp, [])
            ref_entities = report_to_ner_map.get(ref, [])

            if not hyp_entities or not ref_entities:
                scores.append(0.5 if not hyp_entities and not ref_entities else 0.0)
                continue

            hyp_embeds = torch.from_numpy(
                np.array([entity_to_embedding_map[e] for e in hyp_entities])
            )
            hyp_types = [e[1] for e in hyp_entities]
            ref_embeds = torch.from_numpy(
                np.array([entity_to_embedding_map[e] for e in ref_entities])
            )
            ref_types = [e[1] for e in ref_entities]

            precision = compute(
                ref_embeds, hyp_embeds, ref_types, hyp_types, self.scorer.affinity_matrix,
            )
            recall = compute(
                hyp_embeds, ref_embeds, hyp_types, ref_types, self.scorer.affinity_matrix,
            )

            if precision + recall == 0:
                f1_score = 0.0
            else:
                f1_score = 2 * precision * recall / (precision + recall)
            scores.append(f1_score)

        scores_np = np.array(scores, dtype=np.float32)

        compute_end_time = time.time()
        if self.print_debug_info or self.verbose:
            logger.info(f"RaTEScore computation time: {compute_end_time - compute_start_time:.2f} seconds")

        results = {
            "mean_ratescore": np.mean(scores_np).item(),
            "per_pair_ratescore": scores_np,
        }

        if self.verbose:
            logger.info("RaTEScore calculation complete.")

        return results

    def __call__(
        self, hyps: List[str], refs: List[str]
    ) -> Dict[str, Union[float, np.ndarray]]:
        return self.compute(hyps=hyps, refs=refs)