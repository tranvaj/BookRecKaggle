import numpy as np
import torch
import torch.nn.functional as F


def binary_cross_entropy(labels: np.ndarray, logits: np.ndarray) -> float:
    labels_tensor = torch.as_tensor(labels, dtype=torch.float32)
    logits_tensor = torch.as_tensor(logits, dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(logits_tensor, labels_tensor).item()


def recall_at_k(labels: np.ndarray, logits: np.ndarray, k: int) -> float:
    ranked_indices = np.argsort(-logits, axis=1)
    top_k_labels = np.take_along_axis(labels, ranked_indices[:, :k], axis=1)
    return float(np.mean(top_k_labels.sum(axis=1) / labels.sum(axis=1)))


def ndcg_at_k(labels: np.ndarray, logits: np.ndarray, k: int) -> float:
    ranked_indices = np.argsort(-logits, axis=1)
    ranked_labels = np.take_along_axis(
        labels,
        ranked_indices[:, :k],
        axis=1,
    )
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (ranked_labels * discounts).sum(axis=1)

    ideal_labels = np.sort(labels, axis=1)[:, ::-1][:, :k]
    ideal_dcg = (ideal_labels * discounts).sum(axis=1)
    return float(np.mean(dcg / ideal_dcg))


def mean_reciprocal_rank(labels: np.ndarray, logits: np.ndarray) -> float:
    ranked_indices = np.argsort(-logits, axis=1)
    ranked_labels = np.take_along_axis(labels, ranked_indices, axis=1)
    first_relevant_rank = np.argmax(ranked_labels > 0, axis=1) + 1
    return float(np.mean(1.0 / first_relevant_rank))


def recall_at_50(labels: np.ndarray, logits: np.ndarray) -> float:
    return recall_at_k(labels, logits, k=50)

def recall_at_100(labels: np.ndarray, logits: np.ndarray) -> float:
    return recall_at_k(labels, logits, k=100)

def ndcg_at_50(labels: np.ndarray, logits: np.ndarray) -> float:
    return ndcg_at_k(labels, logits, k=50)
