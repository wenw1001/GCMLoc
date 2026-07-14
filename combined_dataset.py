"""
combined_dataset.py — mix multiple datasets (e.g. KITTI + Argoverse) for
canonical multi-dataset training.

Concatenates the datasets and provides a WeightedRandomSampler so each dataset
is sampled at a chosen probability regardless of its size.

Sampling convention (matches canonical_normalization_plan.md §5.4 / T5):
    per-sample weight = W_dataset / len(dataset)
    → P(dataset) = W_dataset / sum(W)
e.g. weights (kitti=6, argo=1) → KITTI 85.7%;  (kitti=1, argo=2) → KITTI 33%.
"""
import numpy as np
import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler


class CombinedDataset(ConcatDataset):
    """ConcatDataset of named sub-datasets with per-dataset sampling weights."""

    def __init__(self, datasets, weights=None):
        super().__init__(datasets)
        self.sub_lengths = [len(d) for d in datasets]
        if weights is None:
            weights = [1.0] * len(datasets)
        assert len(weights) == len(datasets)
        self.weights = [float(w) for w in weights]
        # expose the sub-datasets' CameraModel (same class for all) so training
        # code that references `dataset.model` (e.g. the val loss) works.
        self.model = getattr(datasets[0], 'model', None)

    def make_sampler(self, num_samples=None):
        """WeightedRandomSampler so each sub-dataset is seen at P ∝ its weight."""
        per_sample = []
        for w, n in zip(self.weights, self.sub_lengths):
            per_sample.append(np.full(n, w / max(n, 1), dtype=np.float64))
        sample_weights = torch.as_tensor(np.concatenate(per_sample),
                                         dtype=torch.double)
        if num_samples is None:
            num_samples = len(self)
        return WeightedRandomSampler(sample_weights, num_samples=num_samples,
                                     replacement=True)
