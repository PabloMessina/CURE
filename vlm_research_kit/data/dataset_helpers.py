import logging
import math
import numpy as np
import random
from typing import List
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _get_balancedly_distributed_class_indices(class_weights):
    assert len(class_weights) > 0
    assert all(w >= 0 for w in class_weights)
    if all(w == class_weights[0] for w in class_weights):
        return np.arange(len(class_weights), dtype=int) # all classes have the same weight
    if any(w == 0 for w in class_weights):
        # remove classes with zero weight
        i2i = {}
        class_weights_ = []
        for i, w in enumerate(class_weights):
            if w > 0:
                i2i[len(i2i)] = i
                class_weights_.append(w)
        class_weights = class_weights_
        assert len(class_weights) > 0
        assert all(w > 0 for w in class_weights)
    else:
        i2i = None
    w_sum = sum(class_weights)
    ws = [w / w_sum for w in class_weights]
    w_min = min(ws)
    assert w_min > 0
    freqs = [max(int(20*w/w_min),1) for w in ws]
    count = sum(freqs)
    indices = [None] * count
    class_ids = list(range(len(class_weights)))
    class_ids.sort(key = lambda i : freqs[i], reverse=True)
    available_slots = list(range(count))
    for i in class_ids:
        assert len(available_slots) >= freqs[i]
        step = len(available_slots) / freqs[i]
        for j in range(freqs[i]):
            jj = int(j * step)
            indices[available_slots[jj]] = i
        available_slots = [s for s in available_slots if indices[s] is None]
    indices = [i for i in indices if i is not None]
    if i2i is not None:
        indices = [i2i[i] for i in indices]
    return np.array(indices, dtype=int)


class WeightedCompositeDataset(Dataset):
    def __init__(self, datasets, weights, dataset_names=None):
        assert len(datasets) == len(weights)
        assert all(w >= 0 for w in weights)
        if dataset_names is not None:
            assert len(dataset_names) == len(datasets)
            self.name_to_dataset = {name: dataset for name, dataset in zip(dataset_names, datasets)}
            assert len(self.name_to_dataset) == len(datasets)
        n_bef = len(datasets)
        pos_indices = [i for i in range(n_bef) if weights[i] > 0]
        datasets = [datasets[i] for i in pos_indices]
        weights = [weights[i] for i in pos_indices]
        n_aft = len(datasets)
        if n_aft < n_bef:
            logger.warning(f'Removed {n_bef - n_aft} datasets with zero weight', bold=True)
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.weights = weights        
        self._init_indices(weights)
        self._len = self._compute_len()
    
    def _init_indices(self, weights):
        indices = _get_balancedly_distributed_class_indices(weights)
        dataset_counts = np.zeros((len(weights), len(indices)), dtype=int)
        for i in range(len(weights)):
            for j in range(len(indices)):
                dataset_counts[i][j] = (indices[j] == i) + (dataset_counts[i][j-1] if j > 0 else 0)
            assert dataset_counts[i][-1] > 0, (i, dataset_counts[i], indices)

        self.indices = indices
        self.counts = dataset_counts

    def _compute_len(self):
        num_iter = 0
        for dataset, counts in zip(self.datasets, self.counts):
            num_iter = max(num_iter, math.ceil(len(dataset) / counts[-1]))
        return num_iter * len(self.indices)

    def update_weights(self, weights: List[float]):
        self.weights = weights
        self._init_indices(weights)
        self._len = self._compute_len()
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, i):
        indices = self.indices        
        ii = i % len(indices)
        idx = indices[ii]
        dataset = self.datasets[idx]
        dataset_len = len(dataset)
        # assert idx < len(self.datasets)
        counts = self.counts[idx]
        j = (i // len(indices)) * counts[-1] + (counts[ii - 1] if ii > 0 else 0)
        j = j % dataset_len
        return self.datasets[idx][j]


class CompositeDataset(Dataset):
    """
    A wrapper dataset that composes multiple datasets together.
    """
    def __init__(self, datasets, dataset_names=None):
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.index_pairs = []
        for i,dataset in enumerate(datasets):
            self.index_pairs.extend([(i, j) for j in range(len(dataset))])
        self.shuffle_indices()

    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, idx):
        dataset_idx, index_in_dataset = self.index_pairs[idx]
        return self.datasets[dataset_idx][index_in_dataset]

    def shuffle_indices(self):
        random.shuffle(self.index_pairs)
        for dataset in self.datasets:
            if hasattr(dataset, "shuffle_indices"):
                dataset.shuffle_indices()
            else:
                logger.warning(f'Dataset {type(dataset).__name__} has no shuffle_indices method. Skipping.')

    
class RandomSubsetDataset(Dataset):
    """
    A wrapper that takes a random subset of a given size from a base dataset.
    Crucially, it has a `resample()` method to generate a new random subset,
    allowing for different subsets during each evaluation epoch.
    """
    def __init__(self, base_dataset, subset_size, ensure_multiple_of: int = 1):
        if subset_size > len(base_dataset):
            logger.warning(
                f"Subset size ({subset_size}) is larger than the base dataset size "
                f"({len(base_dataset)}). Using the full dataset."
            )
            subset_size = len(base_dataset)
        subset_size = math.ceil(subset_size / ensure_multiple_of) * ensure_multiple_of # Ensure subset size is a multiple of ensure_multiple_of
        subset_size = min(subset_size, len(base_dataset)) # Ensure subset size is at most the size of the base dataset
        self.base_dataset = base_dataset
        self.subset_size = subset_size
        self.indices = []
        self.resample() # Generate the initial random subset

    def resample(self):
        """Generates a new random list of indices."""
        logger.info(f"Resampling validation set to {self.subset_size} random samples.")
        n = len(self.base_dataset)
        self.indices = random.sample(range(n), self.subset_size)
        assert len(self.indices) == self.subset_size, f"Expected {self.subset_size} samples, but got {len(self.indices)}."

    def __len__(self):
        return self.subset_size

    def __getitem__(self, idx):
        # Map the requested index to our random list of indices
        original_idx = self.indices[idx]
        return self.base_dataset[original_idx]
    

class FormattedDataset(Dataset):
    """A wrapper dataset that applies a formatting function on the fly."""

    def __init__(self, base_dataset, format_fn):
        self.base_dataset = base_dataset
        self.format_fn = format_fn

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        example = self.base_dataset[idx]
        return self.format_fn(example)

    def __getattr__(self, name):
        """
        Forwards attribute calls to the base_dataset.
        """
        return getattr(self.base_dataset, name)


def create_uniform_subset_from_indices_list(target_size: int, indices_list: List[List[int]], ensure_multiple_of: int = 1):
    """
    Alters the dataset to be a uniformly sampled subset of the original data.

    A new random subset is generated each time this method is called.

    Args:
        target_size: The desired approximate size of the validation subset.
        indices_list: A list of lists of indices to sample from.
        ensure_multiple_of: The desired multiple of the target size.
    """
    num_groups = len(indices_list)
    target_size = max(target_size, num_groups) # Ensure target size is at least the number of groups
    target_size = math.ceil(target_size / ensure_multiple_of) * ensure_multiple_of # Ensure target size is a multiple of ensure_multiple_of
    new_indices = []
    num_samples_per_group = [0] * num_groups # Number of samples to sample from each group
    total_num_samples = sum(len(indices) for indices in indices_list) # Total number of samples in the dataset
    assert total_num_samples > 0, "Total number of samples in the dataset is 0."
    indices_ranking = [(i, len(indices)) for i, indices in enumerate(indices_list)]
    indices_ranking.sort(key=lambda x: x[1]) # Sort by increasing length
    
    if target_size > total_num_samples: # If the target size is larger than the total number of samples, we will sample with replacement
        logger.warning(f"Target size ({target_size}) is larger than the number of samples in the dataset ({total_num_samples})." " We will sample with replacement.")
        excess_size = target_size - total_num_samples
        excess_size_per_group = excess_size // num_groups
        excess_size_remainder = excess_size % num_groups
        for i in range(num_groups):
            num_samples_per_group[i] = len(indices_list[i]) # All groups will have all of their samples
            num_samples_per_group[i] += excess_size_per_group # Add the excess size to the group
            if i < excess_size_remainder: # If there is a remainder, add one more sample to the group
                num_samples_per_group[i] += 1 # Add one more sample to the group
    else: # If the target size is smaller than the total number of samples, we will sample without replacement
        start_idx = 0
        count = 0
        for h in range(1, indices_ranking[-1][1] + 1): # For each height, if we see the groups as a sorted histogram
            while indices_ranking[start_idx][1] < h:
                start_idx += 1
            assert start_idx < num_groups, f"start_idx ({start_idx}) is out of bounds for num_groups ({num_groups})."
            for i in range(start_idx, num_groups):
                num_samples_per_group[indices_ranking[i][0]] += 1
                count += 1
                if count == target_size:
                    break
            if count == target_size:
                break
        assert count == target_size, f"Expected {target_size} samples, but got {count}."

    groups_with_replacement_count = 0

    for i, indices in enumerate(indices_list):
        k = num_samples_per_group[i]
        if k <= len(indices): # Sample without replacement
            new_indices.extend(random.sample(indices, k=k))
        else: # Sample with replacement
            new_indices.extend(random.choices(indices, k=k))
            groups_with_replacement_count += 1
    assert len(new_indices) == target_size, f"Expected {target_size} samples, but got {len(new_indices)}."

    if groups_with_replacement_count > 0:
        logger.warning(f"Used replacement sampling for {groups_with_replacement_count} groups to create the subset.")

    return new_indices


if __name__ == "__main__":
    class DummyDataset(Dataset):
        def __init__(self, name, length):
            self.name = name
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return f"{self.name}{idx}"

    # Create dummy datasets
    A = DummyDataset("A", 3000)
    B = DummyDataset("B", 5000)
    C = DummyDataset("C", 500)

    datasets = [A, B, C]
    weights = [0.5, 0.3, 0.2]  # Example weights

    # Create the composite dataset
    composite = WeightedCompositeDataset(datasets, weights)

    print(f"Composite dataset length: {len(composite)}")
    print("First pass through the composite dataset:")

    # Track frequency
    from collections import Counter

    freq = Counter()
    for i in range(80):
        item = composite[i]
        print(f"{i}: {item}")
        # The dataset name is the first character
        freq[item[0]] += 1

    # Print empirical frequencies
    total = sum(freq.values())
    print("\nEmpirical frequencies:")
    for ds, count in freq.items():
        print(f"  Dataset {ds}: {count} ({count/total:.2%})")

    print("\nIntended weights:")
    for ds, w in zip(['A', 'B', 'C'], weights):
        print(f"  Dataset {ds}: {w:.2%}")