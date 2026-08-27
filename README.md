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
│   ├── model.py             # MLP, NeuMF, and probability ensemble
│   ├── metrics.py           # Recall@K, NDCG@K, MRR, BCE
│   └── evaluation.py
└── explicit/
    ├── datasets.py          # Numeric rating examples
    ├── model.py             # Explicit-rating MLP
    ├── hyperparameters.py   # HPO search space and defaults
    ├── metrics.py           # RMSE and MAE
    ├── baselines.py         # Global/user/item means
    └── evaluation.py

scripts/
├── implicit/
│   ├── tune.py              # Tune MLP or NeuMF with Optuna
│   ├── train.py             # Train a model or deep-ensemble member
│   ├── train_ensemble.py    # Train multiple independent members
│   ├── train_als.py         # Train the ALS baseline
│   └── evaluate.py          # Compare models on the test set
└── explicit/
    ├── tune.py
    ├── train.py
    ├── train_ensemble.py
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
Recall@50, NDCG@50, and Recall@100.
It is compared with random ranking, most-popular items, and implicit ALS using
the same sampled test candidates. The improved implicit model is a deep ensemble
of independently trained copies of the exact same MLP architecture. The number
of members and their seeds are configurable, and their probabilities are
averaged directly. It contains no calibration layer, ALS component, graph
network, secondary architecture, popularity blend, or other hybrid component.

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
├── als/
│   └── model.pt
└── mlp_ensemble/
│   ├── seed_100/model_with_mappings.pt
│   ├── seed_110/model_with_mappings.pt
│   └── seed_120/model_with_mappings.pt
```

Tune and train each neural model independently. The original MLP checkpoint is
trained normally. Ensemble training reuses that same training function and puts
each independent member in a `seed_<n>` directory:

```bash
.venv/bin/python -m scripts.implicit.tune --model mlp
.venv/bin/python -m scripts.implicit.tune --model neumf
.venv/bin/python -m scripts.implicit.train --model mlp
.venv/bin/python -m scripts.implicit.train --model neumf
.venv/bin/python -m scripts.implicit.train_ensemble --num-members 3
.venv/bin/python -m scripts.implicit.evaluate \
  --ensemble-dir artifacts/implicit/mlp_ensemble
```

Without `--seeds`, member seeds are `100, 110, 120, ...`. To choose them:

```bash
.venv/bin/python -m scripts.implicit.train_ensemble \
  --num-members 3 --seeds 7 17 42 \
  --output-dir artifacts/implicit/mlp_ensemble
```

## Explicit model

Only ratings from 1 through 10 are used. The prediction is:

```text
MLP(user embedding, item embedding)
```

The model is trained with MSE and evaluated against global-, user-, and item-mean
baselines using RMSE and MAE. The explicit ensemble averages rating predictions
from independently trained copies of this same MLP architecture.

Optuna runs 25 resumable trials and minimizes validation RMSE. The selected
configuration is saved to `artifacts/explicit/mlp/best_hparams.json`; training
uses the defaults when that file does not exist. The explicit MLP writes all of
its artifacts beneath `artifacts/explicit/mlp/`; ensemble members are stored in
`artifacts/explicit/mlp_ensemble/seed_<n>/`.

```bash
.venv/bin/python -m scripts.explicit.tune
.venv/bin/python -m scripts.explicit.train
.venv/bin/python -m scripts.explicit.train_ensemble --num-members 3
.venv/bin/python -m scripts.explicit.evaluate \
  --ensemble-dir artifacts/explicit/mlp_ensemble
```

As with the implicit ensemble, default seeds are `100, 110, 120, ...`; custom
seeds can be supplied with `--seeds`.

## Verified results

Model selection used validation data only. With seed 42, the held-out test
evaluation uses 1,000 candidates per user for implicit ranking and known
user/item pairs for explicit rating prediction.

| Implicit model | Recall@50 | NDCG@50 | Recall@100 |
|---|---:|---:|---:|
| MLP | 0.5487 | 0.2918 | 0.6442 |
| NeuMF | 0.5388 | 0.2956 | 0.6276 |
| MLP probability ensemble | **0.5769** | **0.3043** | **0.6773** |

| Explicit model | RMSE | MAE |
|---|---:|---:|
| MLP | 1.6070 | 1.2327 |
| MLP ensemble (3) | **1.5882** | **1.2228** |

Both scripts build user and item mappings from training data. Validation/test
interactions with cold-start IDs are excluded because ID-only collaborative
filtering cannot learn embeddings for unseen users or books.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
