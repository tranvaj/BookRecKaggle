import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from bookrec_dataset import ImplicitDataset
from data_preprocessing import (
    create_id_mappings,
    encode_ids,
    get_splits,
    load_ds,
)
from model import RecommenderMLP
from train import train, validation_loop


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validation_bce(labels, logits):
    labels = torch.as_tensor(labels, dtype=torch.float32)
    logits = torch.as_tensor(logits, dtype=torch.float32)

    return F.binary_cross_entropy_with_logits(
        logits,
        labels,
    ).item()

def recall_at_k(labels, logits, k):
    ranked_indices = np.argsort(-logits, axis=1)
    top_k_indices = ranked_indices[:, :k]
    top_k_labels = np.take_along_axis(labels, top_k_indices, axis=1)

    relevant_per_user = labels.sum(axis=1)
    recalled_per_user = top_k_labels.sum(axis=1)
    return float(np.mean(recalled_per_user / relevant_per_user))


def ndcg_at_k(labels, logits, k):
    ranked_indices = np.argsort(-logits, axis=1)
    top_k_indices = ranked_indices[:, :k]
    ranked_labels = np.take_along_axis(labels, top_k_indices, axis=1)

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (ranked_labels * discounts).sum(axis=1)

    ideal_labels = np.sort(labels, axis=1)[:, ::-1][:, :k]
    ideal_dcg = (ideal_labels * discounts).sum(axis=1)
    return float(np.mean(dcg / ideal_dcg))


def mean_reciprocal_rank(labels, logits):
    ranked_indices = np.argsort(-logits, axis=1)
    ranked_labels = np.take_along_axis(labels, ranked_indices, axis=1)

    first_relevant_rank = np.argmax(ranked_labels > 0, axis=1) + 1
    return float(np.mean(1.0 / first_relevant_rank))


def recall_at_10(labels, logits):
    return recall_at_k(labels, logits, k=10)


def ndcg_at_10(labels, logits):
    return ndcg_at_k(labels, logits, k=10)


def main():
    set_seed(42)

    # Every row is a positive interaction, regardless of rating.
    interactions = load_ds()

    train_df, val_df, test_df = get_splits(
        interactions,
        val_frac=0.1,
        test_frac=0.1,
        seed=42,
    )

    user_to_index, item_to_index = create_id_mappings(
        train_df
    )

    train_data = encode_ids(
        train_df,
        user_to_index,
        item_to_index,
    )

    val_data = encode_ids(
        val_df,
        user_to_index,
        item_to_index,
    )

    test_data = encode_ids(
        test_df,
        user_to_index,
        item_to_index,
    )

    # Prevent any known interaction from being sampled as a negative.
    all_interactions = pd.concat(
        [train_data, val_data, test_data],
        ignore_index=True,
    )

    train_dataset = ImplicitDataset(
        train_interactions=train_data,
        all_interactions=all_interactions,
        num_items=len(item_to_index),
        negatives_per_positive=4,
        fixed_negatives=False,
    )

    # Fixed validation negatives make metrics reproducible.
    val_dataset = ImplicitDataset(
        train_interactions=val_data,
        all_interactions=all_interactions,
        num_items=len(item_to_index),
        negatives_per_positive=99,
        fixed_negatives=True,
        seed=42,
    )

    test_dataset = ImplicitDataset(
        train_interactions=test_data,
        all_interactions=all_interactions,
        num_items=len(item_to_index),
        negatives_per_positive=99,
        fixed_negatives=True,
        seed=43,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = RecommenderMLP(
        num_users=len(user_to_index),
        num_items=len(item_to_index),
        embedding_dim=32,
        hidden_dims=(64, 32),
        dropout=0.2,
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )
    epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ranking_metrics = [
        validation_bce,
        recall_at_10,
        ndcg_at_10,
        mean_reciprocal_rank,
    ]

    model, scores, losses, learning_rates = train(
        model=model,
        train_dl=train_loader,
        val_dl=val_loader,
        loss=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        val_metrics=ranking_metrics,
        save_val_metric="ndcg_at_10",
        device=device,
        output_path="best_implicit_model.pt",
        load_best_model=True,
        early_stopping_patience=3,
        metric_mode="max",
    )

    test_scores = validation_loop(
        test_loader,
        model,
        ranking_metrics,
        device,
    )
    print(f"Test metrics: {test_scores}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "user_to_index": user_to_index,
            "item_to_index": item_to_index,
            "embedding_dim": 32,
        },
        "implicit_recommender.pt",
    )


if __name__ == "__main__":
    main()
