# Book recommendation skeleton

This project treats Book-Crossing ratings from 1 through 10 as explicit feedback.
Rows with rating 0 are excluded because they represent implicit feedback and are
not on the same rating scale.

The neural recommender predicts

`global mean + user bias + item bias + MLP(user embedding, item embedding)`.

## Pipeline

```python
import torch
from torch.utils.data import DataLoader

from baselines import evaluate_mean_baselines
from bookrec_dataset import BookRecDataset
from data_preprocessing import (
    build_id_mappings,
    encode_interactions,
    filter_known_interactions,
    get_splits,
    load_ds,
    prepare_explicit_ratings,
)
from model import RecommenderMLP
from train import evaluate, train_one_epoch

ratings = prepare_explicit_ratings(load_ds())
train_df, val_df, test_df = get_splits(ratings)
mappings = build_id_mappings(train_df)
val_known = filter_known_interactions(val_df, mappings)

print(f"Validation coverage: {len(val_known) / len(val_df):.1%}")
print(evaluate_mean_baselines(train_df, val_known))

train_data = encode_interactions(train_df, mappings)
val_data = encode_interactions(val_known, mappings, drop_unknown=False)

train_loader = DataLoader(BookRecDataset(train_data), batch_size=1024, shuffle=True)
val_loader = DataLoader(BookRecDataset(val_data), batch_size=2048)

model = RecommenderMLP(
    num_users=len(mappings.users),
    num_items=len(mappings.items),
    global_mean=train_df["Book-Rating"].mean(),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

for epoch in range(10):
    train_rmse = train_one_epoch(model, train_loader, optimizer, device)
    val_metrics = evaluate(model, val_loader, device)
    print(epoch + 1, train_rmse, val_metrics)
```

Fit all statistics and ID mappings on `train_df` only. Validation/test books not
seen during training are excluded from neural-model evaluation; the mean baselines
fall back to the global training mean for unknown users or books. Compare every
model on the same known-interaction subset and report its coverage.
