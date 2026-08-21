# Book recommendation

This project explores Book-Crossing recommendations as two separate collaborative
filtering tasks:

- **Implicit feedback:** every recorded user-book row is an interaction. The model
  learns from observed positives and dynamically sampled unobserved pairs.
- **Explicit feedback:** ratings from 1 through 10 are modeled as numeric targets;
  rating-zero rows are excluded from this task.

Keeping the tasks separate avoids mixing binary ranking objectives with rating
regression while allowing data loading and training mechanics to be shared.

## Structure

```text
bookrec/
├── data.py                   # Loading, splitting, and ID encoding
├── training.py               # Shared training and evaluation loops
├── implicit/
│   ├── datasets.py          # Negative sampling and user-level evaluation
│   ├── model.py             # Binary interaction MLP
│   ├── metrics.py           # Recall@K, NDCG@K, MRR, BCE
│   └── evaluation.py
└── explicit/
    ├── datasets.py          # Numeric rating examples
    ├── model.py             # Bias-aware rating MLP
    ├── metrics.py           # RMSE and MAE
    ├── baselines.py         # Global/user/item means
    └── evaluation.py

scripts/
├── train_implicit.py
└── train_explicit.py
```

## Implicit model

All rows, including `Book-Rating == 0`, are positive interactions. Four unobserved
items are sampled for every positive during training. Validation and test use one
fixed ranking per user containing all held-out positives and enough sampled
unobserved items to reach 1,000 candidates.

The model returns raw logits and is trained with `BCEWithLogitsLoss`. Model
selection uses NDCG@10; test reporting includes Recall@10, NDCG@10, MRR, and BCE.

```bash
.venv/bin/python -m scripts.train_implicit
```

## Explicit model

Only ratings from 1 through 10 are used. The prediction is:

```text
global mean + user bias + item bias + MLP(user embedding, item embedding)
```

The model is trained with MSE and evaluated against global-, user-, and item-mean
baselines using RMSE and MAE.

```bash
.venv/bin/python -m scripts.train_explicit
```

Both scripts build user and item mappings from training data. Validation/test
interactions with cold-start IDs are excluded because ID-only collaborative
filtering cannot learn embeddings for unseen users or books.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
