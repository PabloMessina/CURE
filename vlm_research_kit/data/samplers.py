import logging
from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)

class BatchesPerEpochSampler(Sampler[int]):
    """
    A PyTorch Sampler that yields a specific number of samples per epoch,
    allowing for iteration over a fixed number of batches regardless of
    the dataset's full size.

    Handles shuffling and samples with replacement if the requested number
    of samples exceeds the dataset size.

    Args:
        data_source: The dataset to sample from. Must implement `__len__`.
        batches_per_epoch: The exact number of batches desired per epoch.
        batch_size: The size of each batch.
        generator: (Optional) A PyTorch random number generator.
    """

    def __init__(
        self,
        data_source: Sized,
        batches_per_epoch: int,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if not isinstance(data_source, Sized):
            raise TypeError("data_source should be an instance of Sized.")
        if not isinstance(batches_per_epoch, int) or batches_per_epoch <= 0:
            raise ValueError(
                f"batches_per_epoch should be a positive integer, "
                f"got {batches_per_epoch}"
            )
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"batch_size should be a positive integer, got {batch_size}"
            )

        self.data_source = data_source
        self.dataset_size = len(data_source)
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.num_samples_per_epoch = self.batches_per_epoch * self.batch_size
        self.generator = generator

        if self.dataset_size == 0:
            raise ValueError(
                "The dataset is empty. Cannot initialize BatchesPerEpochSampler."
            )

        logger.info(
            f"Initialized BatchesPerEpochSampler: "
            f"Dataset size={self.dataset_size}, "
            f"Batches/Epoch={self.batches_per_epoch}, "
            f"Batch Size={self.batch_size}, "
            f"Samples/Epoch={self.num_samples_per_epoch}"
        )
        if self.num_samples_per_epoch > self.dataset_size:
            logger.warning(
                f"Requested samples per epoch ({self.num_samples_per_epoch}) "
                f"exceeds dataset size ({self.dataset_size}). "
                f"Sampling with replacement will occur."
            )


    def __iter__(self) -> Iterator[int]:
        """
        Yields a sequence of indices for one epoch.

        If num_samples_per_epoch <= dataset_size, it yields a shuffled subset
        without replacement.
        If num_samples_per_epoch > dataset_size, it yields indices sampled
        randomly with replacement.
        """
        if self.num_samples_per_epoch <= self.dataset_size:
            # Sample without replacement: shuffle all indices and take the first N
            indices = torch.randperm(
                self.dataset_size, generator=self.generator
            ).tolist()
            yield from indices[: self.num_samples_per_epoch]
        else:
            # Sample with replacement: randomly choose N indices from 0 to dataset_size-1
            indices = torch.randint(
                high=self.dataset_size,
                size=(self.num_samples_per_epoch,),
                dtype=torch.int64,
                generator=self.generator,
            ).tolist()
            yield from indices

    def __len__(self) -> int:
        """Returns the total number of samples yielded per epoch."""
        return self.num_samples_per_epoch
    

class SubsetRandomSampler(Sampler[int]):
    """
    Samples a random subset of elements each epoch without replacement.

    This is useful for large validation sets where you want to evaluate on a
    different random subset during each validation phase of a training loop.

    Args:
        data_source (Sized): The entire dataset from which to sample.
        sample_size (int): The number of elements to sample each epoch.
        generator (torch.Generator, optional): Generator used for reproducibility.
    """
    def __init__(self, data_source: Sized, sample_size: int, generator: Optional[torch.Generator] = None) -> None:
        self.data_source = data_source
        self.sample_size = sample_size
        self.generator = generator

        if not isinstance(self.sample_size, int) or self.sample_size <= 0:
            raise ValueError(
                f"sample_size must be a positive integer, but got {self.sample_size}"
            )
        if self.sample_size >= len(self.data_source):
            raise ValueError(
                f"sample_size ({self.sample_size}) cannot be greater than or equal to "
                f"dataset size ({len(self.data_source)})"
            )

    def __iter__(self) -> Iterator[int]:
        """
        Generates a new list of randomly shuffled indices for each epoch.
        """
        # Generate a random permutation of all indices
        n = len(self.data_source)
        if self.generator is None:
            indices = torch.randperm(n).tolist()
        else:
            indices = torch.randperm(n, generator=self.generator).tolist()

        # Return the first `sample_size` indices from the shuffled list
        return iter(indices[:self.sample_size])

    def __len__(self) -> int:
        """
        The number of samples drawn in one epoch.
        """
        return self.sample_size
