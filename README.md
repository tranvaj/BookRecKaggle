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
│   ├── baselines.py         # Random, popularity, and implicit ALS
│   ├── model.py             # MLP and GMF + MLP (NeuMF)
│   ├── metrics.py           # Recall@K, NDCG@K, MRR, BCE
│   └── evaluation.py
└── explicit/
    ├── datasets.py          # Numeric rating examples
    ├── model.py             # Bias-aware rating MLP
    ├── metrics.py           # RMSE and MAE
    ├── baselines.py         # Global/user/item means
    └── evaluation.py

scripts/
├── implicit/
│   ├── tune.py              # Tune MLP or NeuMF with Optuna
│   ├── train.py             # Train MLP or NeuMF
│   ├── train_als.py         # Train the ALS baseline
│   └── evaluate.py          # Compare models on the test set
└── explicit/
    ├── train.py
    └── evaluate.py
```

## Implicit model

All rows, including `Book-Rating == 0`, are positive interactions. Four unobserved
items are sampled for every positive during training. Validation and test use one
fixed ranking per user containing all held-out positives and enough sampled
unobserved items to reach 1,000 candidates.

Two neural models are available: a concatenation-only MLP and NeuMF, which
combines a GMF branch with an MLP branch. Both return raw logits and are trained
with `BCEWithLogitsLoss`. Model selection uses NDCG@50; test reporting includes
Recall@50, NDCG@50, Recall@100, and BCE.
It is compared with random ranking, most-popular items, and implicit ALS using
the same sampled test candidates.

Optuna runs a separate resumable study for each neural model and maximizes
validation NDCG@50. Each model keeps its hyperparameters and checkpoints in its
own artifact directory:

```text
artifacts/implicit/
├── mlp/
│   ├── hpo.db
│   ├── best_hparams.json
│   ├── best_model.pt
│   └── model_with_mappings.pt
├── neumf/
│   ├── hpo.db
│   ├── best_hparams.json
│   ├── best_model.pt
│   └── model_with_mappings.pt
└── als/
    └── model.pt
```

Tune and train each neural model independently, train ALS once, and then run one
evaluation that compares all models on the same test candidates:

```bash
.venv/bin/python -m scripts.implicit.tune --model mlp
.venv/bin/python -m scripts.implicit.tune --model neumf
.venv/bin/python -m scripts.implicit.train --model mlp
.venv/bin/python -m scripts.implicit.train --model neumf
.venv/bin/python -m scripts.implicit.train_als
.venv/bin/python -m scripts.implicit.evaluate
```

## Explicit model

Only ratings from 1 through 10 are used. The prediction is:

```text
global mean + user bias + item bias + MLP(user embedding, item embedding)
```

The model is trained with MSE and evaluated against global-, user-, and item-mean
baselines using RMSE and MAE.

```bash
.venv/bin/python -m scripts.explicit.train
.venv/bin/python -m scripts.explicit.evaluate
```

Both scripts build user and item mappings from training data. Validation/test
interactions with cold-start IDs are excluded because ID-only collaborative
filtering cannot learn embeddings for unseen users or books.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
