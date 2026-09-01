from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

import torch
from tqdm import tqdm
import numpy as np

def move_to_device(data, device):
    """Recursively move tensors in dicts/lists to device."""
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data

def train_loop(train_dl, model, loss_fn, optimizer, device, target_key="label", log_every=1):
    losses = []
    size = len(train_dl)
    
    model.train()

    pbar = tqdm(train_dl)
    for batch_idx, batch_dict in enumerate(pbar):
        batch = move_to_device(batch_dict, device)
        if target_key not in batch:
            raise KeyError(f"Batch is missing target key '{target_key}'. Available keys: {list(batch.keys())}")
        
        # Pop from batch (GPU) so its on the right device and removed from inputs
        y = batch.pop(target_key)
        
        # Compute prediction and loss
        pred = model(**batch)   
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch_idx % log_every == 0:
            losses.append(loss.detach().cpu().item())
        
        loss, current = loss.item(), (batch_idx + 1)
        pbar.set_description(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    return losses

def inference_loop(model, dl, device, target_key="label"):
    model.to(device)
    predictions = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_dict in tqdm(dl):
            batch = move_to_device(batch_dict, device)
            if target_key not in batch:
                raise KeyError(f"Batch is missing target key '{target_key}'. Available keys: {list(batch.keys())}")
        
            # Pop from batch (GPU) so its on the right device and removed from inputs
            y = batch.pop(target_key)
        
            # Compute prediction and loss
            pred = model(**batch)   

            predictions.extend(pred.cpu().tolist())
            if y is not None:
                labels.extend(y.cpu().tolist())
                
    return predictions, labels

def validation_loop(val_dl, model, val_metrics, device):
    predictions, labels = inference_loop(model, val_dl, device)
    predictions_np = np.array(predictions)
    labels_np = np.array(labels)
    scores = {metric.__name__: metric(labels_np, predictions_np) for metric in val_metrics}
    return scores

def predict(model, test_dl, device):
    model.to(device)
    predictions, _ = inference_loop(model, test_dl, device)
    return predictions

def train(model, train_dl, val_dl, loss, optimizer, scheduler, epochs, 
          val_metrics, save_val_metric, device, output_path, load_best_model=True, 
          early_stopping_patience=5, metric_mode='max'):
    
    scores = []
    losses = []
    lr_rates = []
    
    epochs_no_improve = 0
    
    if metric_mode == 'max':
        best_validation = -float('inf')
    elif metric_mode == 'min':
        best_validation = float('inf')
    else:
        raise ValueError("metric_mode must be 'min' or 'max'")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}\n-------------------------------")
        epoch_losses = train_loop(train_dl, model, loss, optimizer, device)
        epoch_scores = validation_loop(val_dl, model, val_metrics, device)
        print(f"Epoch {epoch+1}: Loss={np.mean(epoch_losses):.4f}")
        scheduler.step()
        
        current_metric_val = epoch_scores[save_val_metric]
        print(epoch_scores)
        
        scores.append(epoch_scores)
        losses.extend(epoch_losses)
        lr_rates.append(scheduler.get_last_lr()[0])

        
        # Saving best model, early stopping
        save_model = False
        if metric_mode == 'max':
            if current_metric_val > best_validation:
                best_validation = current_metric_val
                save_model = True
        elif metric_mode == 'min':
            if current_metric_val < best_validation:
                best_validation = current_metric_val
                save_model = True

        if save_model:
            torch.save(model.state_dict(), output_path)
            print(f"New best model saved with {save_val_metric}: {best_validation:.4f}")
            epochs_no_improve = 0 # Reset patience
        else:
            epochs_no_improve += 1

        # Early stopping logic
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            print(f"No improvement in {save_val_metric} for {early_stopping_patience} epochs.")
            break
            
    print("Done!")
    if load_best_model:
        print(f"Loading best model weights with {save_val_metric}: {best_validation:.4f}")
        model.load_state_dict(torch.load(output_path))
    
    return model, scores, losses, lr_rates

def predict(model, test_dl, device):
    model.to(device)
    predictions, _ = inference_loop(model, test_dl, device)
    return predictions