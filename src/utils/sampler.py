from torch.utils.data import WeightedRandomSampler
import torch
from collections import Counter


def class_distribution(labels):
    counter = Counter(labels)
    distribution = [counter[i] for i in range(max(labels) + 1)]
    return distribution


def weighted_ramdom_sampler(labels):

    # Calculate class weights to be used in the WeightedRandomSampler
    class_weights = 1.0 / torch.tensor(class_distribution(labels), dtype=torch.float)

    # Create a sampler that samples with replacement based on class weights
    sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(labels), replacement=True)

    return sampler
