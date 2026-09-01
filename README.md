# Book recommendation

This project explores Book-Crossing recommendations as two collaborative-
filtering tasks:

- **Implicit feedback:** every recorded user-book row is an interaction. Models
  can rank books from either a known user ID or a sparse binary history such as a
  single favorite book.
- **Explicit feedback:** ratings from 1 through 10 are modeled as numeric targets;
  rating-zero rows are excluded from this task.

Keeping the tasks separate avoids mixing binary ranking objectives with rating
regression while allowing data loading and training mechanics to be shared.

## Structure

```text
app/
├── main.py                   # FastAPI lifespan and endpoints
└── schemas.py                # HTTP request and response contracts

bookrec/
├── data.py                   # Loading, splitting, and ID encoding
├── catalog.py                # ISBN metadata and title-to-ISBN resolution
├── training.py               # Shared training and evaluation loops
├── implicit/
│   ├── datasets.py          # Negative sampling and sparse-history batching
│   ├── baselines.py         # Random, popularity, and implicit ALS
│   ├── model.py             # ID models, history MLP, and ensembles
│   ├── inference.py         # Checkpoint loading and low-level scoring
│   ├── recommender.py       # Serving interface for ISBN histories
│   ├── metrics.py           # Ranking metrics
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
│   ├── tune.py              # Tune neural models with Optuna
│   ├── train.py             # Train a model or deep-ensemble member
│   ├── train_ensemble.py    # Train multiple independent members
│   ├── tune_als.py          # Tune ALS with Optuna
│   ├── train_als.py         # Train the ALS baseline
│   └── evaluate.py          # Compare models on the test set
└── explicit/
    ├── tune.py
    ├── train.py
    ├── train_ensemble.py
    └── evaluate.py
```

## History-vector model

This is the direct solution to “I like *The Lord of the Rings*, what else
should I read?” The `history_mlp` model does not have a user-ID embedding.
Instead, it projects the books in the supplied interaction history and scores
their learned interaction with each candidate-book embedding:

```text
history = normalize(sum(item_embedding[known_book] for known_book in known_books))
candidate = normalize(item_embedding[candidate_book])
score = MLP(history * candidate)
```

Histories are conceptually binary vectors over all books. They are passed as
only their nonzero item IDs and summed with `EmbeddingBag`, which is exactly a
bias-free linear projection of the dense binary vector without allocating it.
History books and candidates use the same learned item-embedding matrix, making
singleton training a symmetric item-to-item learning problem. Only their
elementwise interaction enters the MLP, preventing a candidate-only popularity
shortcut.

For every training positive, the target is removed from its history. Half of
training examples use that complete remaining history and half randomly select
one of its books, with the singleton resampled dynamically. Unobserved candidates
are sampled uniformly. Evaluation reports full history as the primary
personalization result and singleton context as the secondary one-book-query
diagnostic. Validation contexts contain training interactions; test contexts
contain training plus validation interactions. Held-out targets never enter the
context, and the input book is not a candidate negative.

This interaction-based scorer is architecture version 3. Version-1 checkpoints
used concatenated history and candidate features and could collapse to an almost
candidate-only popularity ranking; version 2 still used separate history and
candidate embeddings. Older checkpoints are rejected and must be retrained.
History-model tuning uses a new Optuna study inside the existing
`hpo.db`, so old trials are retained but not reused for the new architecture.
Portable checkpoints also store training-only item support counts. The demo
excludes candidates with fewer than five training interactions and warns when
the input edition has fewer than twenty; an interaction-only model cannot make
a reliable claim for a nearly unseen ISBN.

```bash
.venv/bin/python -m scripts.implicit.tune --model history_mlp
.venv/bin/python -m scripts.implicit.train --model history_mlp
.venv/bin/python -m scripts.implicit.train_ensemble \
  --model history_mlp --num-members 5
.venv/bin/python -m scripts.implicit.evaluate \
  --ensemble-dir artifacts/implicit/history_mlp_ensemble
```

Single checkpoints and history ensembles use the same low-level inference
interface:

```python
from bookrec.implicit.inference import (
    load_history_mlp,
    predict_history_probabilities,
)

loaded = load_history_mlp(
    "artifacts/implicit/history_mlp_ensemble"
)
history = [loaded.item_to_index["0618260250"]]
probabilities = predict_history_probabilities(
    loaded.model,
    history_items=history,
)  # one probability per encoded catalog item
```

Pass `artifacts/implicit/history_mlp` instead to load the single model. Remove
the history items from the candidate ranking before returning recommendations.

For application code, `HistoryMLPRecommender` loads the models, mappings,
metadata, and eligible candidates once. It accepts one ISBN or a list of ISBNs;
the same class works for a single checkpoint and an ensemble:

```python
from bookrec.implicit import HistoryMLPRecommender
from bookrec.data import load_dataset

recommender = HistoryMLPRecommender(
    "artifacts/implicit/history_mlp_ensemble",
    books=load_dataset("Books.csv"),
    min_candidate_interactions=5,
)
recommendations = recommender.recommend_by_isbn(
    ["0618260250"],
    top_k=10,
)
```

The serving responsibilities are deliberately separate. The catalog resolves
titles and retrieves display metadata, while the recommender only ranks ISBNs:

```python
query_isbns = recommender.catalog.resolve_titles(
    ["The Lord of the Rings"],
    min_training_interactions=20,
)
recommendations = recommender.recommend_by_isbn(query_isbns, top_k=10)
books = recommender.catalog.get_books(
    [result["isbn"] for result in recommendations]
)
```

Recommendations contain `isbn`, `title`, `rank`, and `score`, making the result
immediately readable. `get_books()` returns the remaining metadata, including
author, publication data, cover URL, and training-interaction count, in the
same order as the requested ISBNs. Scores are ensemble ranking scores learned
with sampled BCE and should not be presented as calibrated probabilities.

## Serving API

FastAPI loads the ensemble and catalog once during application startup. Its
three application operations remain independent:

```text
POST /books/resolve   titles -> ISBNs
POST /books/metadata  ISBNs -> metadata and training counts
POST /recommendations history ISBNs + top_k -> ISBNs, titles, ranks, and scores
```

Start the local development server from the repository root:

```bash
.venv/bin/fastapi dev app/main.py
```

The generated interactive documentation is available at
`http://127.0.0.1:8000/docs`.

## Implicit model

All rows, including `Book-Rating == 0`, are positive interactions. Unobserved
items are dynamically sampled for every positive during training. Validation
and test use one fixed ranking per user containing all held-out positives and
enough sampled unobserved items to reach 1,000 candidates.

Three neural models are available: a user-ID MLP, NeuMF, and `history_mlp`.
They return raw logits and are trained with `BCEWithLogitsLoss`. Model selection
uses NDCG@50; `history_mlp` uses full history for model selection. Test reporting
includes Recall@50, NDCG@50, and Recall@100.
It is compared with random ranking, most-popular items, and implicit ALS using
the same sampled test candidates. Each neural architecture can also be used as a
deep ensemble of independently trained copies of that exact architecture. The
number of members and their seeds are configurable, and their probabilities are
averaged directly. There is no calibration layer, ALS component, graph network,
secondary architecture, popularity blend, or other hybrid component.

Optuna runs a separate resumable study for each neural model and maximizes
validation NDCG@50. Each model keeps its hyperparameters and checkpoints in its
own artifact directory:

ALS has its own resumable Optuna study using the identical validation rankings.
It tunes the factor count, confidence scaling, regularization, and iteration
count, then `train_als` automatically loads the selected parameters.

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
├── history_mlp/
│   ├── hpo.db
│   ├── best_hparams.json
│   ├── best_model.pt
│   └── model_with_mappings.pt
├── als/
│   ├── hpo.db
│   ├── best_hparams.json
│   └── model.pt
├── mlp_ensemble/
│   ├── seed_100/model_with_mappings.pt
│   ├── seed_110/model_with_mappings.pt
│   └── seed_120/model_with_mappings.pt
└── history_mlp_ensemble/
    ├── seed_100/model_with_mappings.pt
    ├── seed_110/model_with_mappings.pt
    └── seed_120/model_with_mappings.pt
```

Tune and train each neural model independently. The original MLP checkpoint is
trained normally. Ensemble training reuses that same training function and puts
each independent member in a `seed_<n>` directory:

```bash
.venv/bin/python -m scripts.implicit.tune --model mlp
.venv/bin/python -m scripts.implicit.tune --model neumf
.venv/bin/python -m scripts.implicit.tune --model history_mlp
.venv/bin/python -m scripts.implicit.tune_als --num-trials 25
.venv/bin/python -m scripts.implicit.train --model mlp
.venv/bin/python -m scripts.implicit.train --model neumf
.venv/bin/python -m scripts.implicit.train --model history_mlp
.venv/bin/python -m scripts.implicit.train_als
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
| MLP probability ensemble (5) | **0.5879** | **0.3150** | **0.6855** |

| Explicit model | RMSE | MAE |
|---|---:|---:|
| MLP | 1.6070 | 1.2327 |
| MLP ensemble (5) | **1.5849** | **1.2213** |

All tasks build user and item mappings from training data. Validation/test
interactions with cold-start items are excluded because interaction-only
collaborative filtering cannot represent unseen books. The history MLP supports
an anonymous query made from one or more mapped books, while the user-ID MLP and
NeuMF require a user represented in the training mappings. History MLP results
are not included above until the new model and ensemble have been trained and
evaluated.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
