import numpy as np


def normalize_features(features, mean, std):
    return (features - mean) / std


def compute_stats(numeric_features):
    mean = np.mean(numeric_features, axis=0)
    std = np.std(numeric_features, axis=0)
    return mean, std
